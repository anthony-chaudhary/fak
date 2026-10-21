package model

import (
	"errors"
	"fmt"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/computetrace"
	"reflect"
	"testing"
)

// Frozen role scalar bodies from public 6f2733b1; only method names changed.
func (m *Model) referenceV41RoleLayerStep(l int, plan V41AttentionPlan, x []float32, streams [][]float32, pos int, layerState, registry *V41AttentionState, scratch *v41ProjScratch) (work *v41LayerStepWork, err error) {
	cfg := m.Cfg
	attentionContinuation := newV41SharedAttentionContinuation(m, l, pos, plan, layerState, registry, scratch)
	if len(x) != cfg.HiddenSize {
		return nil, v41StageErr(v41StageAttention, l,
			fmt.Errorf("%w: layer role step hidden row has %d values, want %d", ErrV41ForwardStage, len(x), cfg.HiddenSize))
	}
	if len(streams) != 4 {
		return nil, v41StageErr(v41StageMHC, l,
			fmt.Errorf("%w: layer role step carries %d mHC streams, want 4", ErrV41ForwardStage, len(streams)))
	}
	for s := range streams {
		if len(streams[s]) != cfg.HiddenSize {
			return nil, v41StageErr(v41StageMHC, l,
				fmt.Errorf("%w: layer role step stream %d has %d values, want %d", ErrV41ForwardStage, s, len(streams[s]), cfg.HiddenSize))
		}
	}
	if registry == nil {
		return nil, v41StageErr(v41StageAttention, l,
			fmt.Errorf("%w: layer %d role step requires the shared source registry", ErrV41ForwardStage, l))
	}
	if scratch == nil {
		scratch = &v41ProjScratch{}
	}
	if pos != layerState.nextWindowPos {
		return nil, v41StageErr(v41StageAttention, l,
			fmt.Errorf("%w: layer %d role step position %d is not the next retained position %d",
				ErrV41ForwardStage, l, pos, layerState.nextWindowPos))
	}
	if pos < 0 || pos == int(^uint(0)>>1) || layerState.windowSize <= 0 ||
		len(layerState.window) != layerState.windowSize || layerState.headDim != cfg.HeadDim ||
		layerState.retainedWindowRows < 0 || layerState.retainedWindowRows > min(pos, layerState.windowSize) ||
		len(layerState.window[pos%layerState.windowSize]) != cfg.HeadDim {
		return nil, v41StageErr(v41StageAttention, l, fmt.Errorf("%w: role window state has invalid geometry", ErrV41ForwardStage))
	}
	windowStart := 0
	if window := cfg.windowForLayer(l); window > 0 {
		windowStart = max(0, pos-window+1)
	}
	if windowStart < pos-layerState.retainedWindowRows {
		return nil, v41StageErr(v41StageAttention, l, fmt.Errorf("%w: role window is not fully retained", ErrV41ForwardStage))
	}

	H, hd, nH := cfg.HiddenSize, cfg.HeadDim, cfg.NumHeads
	eps := float32(cfg.RMSNormEps)
	full, err := v41ForwardGeometry(cfg)
	if err != nil {
		return nil, err
	}
	if !full {
		return nil, v41StageErr(v41StageAttention, l,
			fmt.Errorf("%w: layer %d role step requires the full V4.1 geometry", ErrV41ForwardStage, l))
	}

	var mhcDevice *v41MHCDevice
	if full {
		if err := scratch.mhcCarry.validate(l, 1); err != nil {
			return nil, err
		}
		if err := m.v41AdmitMHC(l); err != nil {
			return nil, err
		}
		mhcDevice = scratch.mhcCarry.device
	}
	var resident *v41MHCResident
	if full {
		resident = scratch.mhcCarry.resident
	}
	if resident != nil {
		defer func() {
			if work == nil {
				resident.abort()
			}
		}()
	}
	mixBase := m.tensor(layerName(l, "mhc.base"))
	mixScale := m.tensor(layerName(l, "mhc.scale"))
	projectOutput := m.v41GroupedOutputProjector(l, nH, hd, cfg.OGroups, cfg.OLoraRank, H, scratch)

	// ---- mHC coefficient split (one position), byte-identical to the plain branch ----
	mhcFlat, mhcTransposed, mhcOK := m.v41MHCWeightLayout(l)
	if !mhcOK {
		return nil, v41StageErr(v41StageMHC, l,
			fmt.Errorf("%w: mHC mix weight holds no admitted geometry", ErrV41ForwardStage))
	}
	var mix v41MHCMix
	var collapsed []float32
	if resident != nil {
		rows, rerr := resident.beginLayer(l, hcItersOrDefault(cfg), hcEpsOrDefault(cfg), eps)
		if rerr != nil {
			return nil, rerr
		}
		collapsed = rows[0]
	} else {
		// Role steps require the full flattened mHC geometry, so the
		// historical pre-collapse host normalization has no consumer.
		var input []float32
		if mhcFlat {
			width, ok := checkedMulInt(4, H)
			if !ok || len(streams) != 4 {
				return nil, v41StageErr(v41StageMHC, l, errV41ProjectionResult)
			}
			input = make([]float32, 0, width)
			for _, stream := range streams {
				if len(stream) != H {
					return nil, v41StageErr(v41StageMHC, l, errV41ProjectionResult)
				}
				input = append(input, stream...)
			}
		}
		projectMHC := m.v41MHCProjector(l, H, eps, mhcFlat, mhcTransposed, scratch)
		mixes, err := projectMHC(input)
		if err != nil {
			return nil, err
		}
		mix, err = v41MHCSplitOn(mhcDevice, l, mixes, mixScale, mixBase, 4, hcItersOrDefault(cfg), hcEpsOrDefault(cfg))
		if err != nil {
			return nil, err
		}
		if mhcDevice != nil {
			collapsed, err = mhcDevice.preBF16(l, streams, scratch.mhcCarry.pre[0])
			if err != nil {
				return nil, err
			}
		} else {
			collapsed, err = v41MHCPre(streams, scratch.mhcCarry.pre[0])
			if err != nil {
				return nil, v41StageErr(v41StageMHC, l, err)
			}
		}
	}
	var qLat []float32
	prefixHandled := false
	if scratch.queryPrefix != nil {
		var nextCollapsed []float32
		nextCollapsed, qLat, prefixHandled, err = scratch.queryPrefix(l, collapsed)
		if err != nil {
			return nil, err
		}
		if prefixHandled {
			collapsed = nextCollapsed
		}
	}
	if !prefixHandled {
		collapsed, err = m.v41AttentionInputNormWithPublications(l, collapsed, eps, scratch.attentionNormFunc(), scratch.attentionBF16Func())
		if err != nil {
			return nil, err
		}
	}

	// ---- attention query for the one position (mirrors v41Layer's full branch) ----
	ropeDim := cfg.QKRopeHeadDim
	if ropeDim <= 0 || ropeDim%2 != 0 || ropeDim > hd {
		return nil, v41StageErr(v41StageAttention, l,
			fmt.Errorf("%w: attention qk_rope_head_dim must be a positive even value <= head_dim %d, got %d", ErrV41ForwardStage, hd, ropeDim))
	}
	if !prefixHandled {
		qLat, err = m.v41ProjMatRowsWithProjection(l, "attn.wq_a.weight", collapsed, cfg.QLoraRank, H, scratch.denseProjection())
		if err != nil {
			return nil, err
		}
		if err := m.v41QueryNormInPlaceWithPublish(l, qLat, eps, scratch.queryNorm.normFunc(), scratch.gatherOps.bf16PublishFunc()); err != nil {
			return nil, err
		}
	}
	scratch.activationTrace.record(l, pos, v41TraceStageQLatent, qLat)
	q, err := m.v41ProjMatRowsWithProjection(l, "attn.wq_b.weight", qLat, nH*hd, cfg.QLoraRank, scratch.denseProjection())
	if err != nil {
		return nil, err
	}
	if hd > v41KVLoraRank {
		return nil, v41StageErr(v41StageAttention, l,
			fmt.Errorf("%w: attention head_dim %d exceeds full KV latent rank %d", ErrV41ForwardStage, hd, v41KVLoraRank))
	}
	kvFull, err := m.v41ProjMatRowsWithProjection(l, "attn.wkv.weight", collapsed, v41KVLoraRank, H, scratch.denseProjection())
	if err != nil {
		return nil, err
	}
	kv := kvFull[:hd]
	if err := m.v41KVNormWithBinding(l, kv, eps, scratch.kvNorm, scratch.gatherOps); err != nil {
		return nil, err
	}
	cos, sin := v41RopeTableForLayer(cfg, l, pos)
	scratch.activationTrace.record(l, pos, v41TraceStageKVLatent, kv)
	q, kv, err = v41AttentionQKRoPEWithPublish(cfg, l, q, kv, cos, sin, nH, hd, ropeDim, scratch.tailRoPE, scratch.gatherOps.bf16PublishFunc())
	if err != nil {
		return nil, err
	}
	window := make([][]float32, 0, pos-windowStart+1)
	for key := windowStart; key < pos; key++ {
		window = append(window, layerState.window[key%layerState.windowSize])
	}
	window = append(window, kv)

	// ---- shared compressed stream: source publishes, reader resolves ----
	//
	// The shared registry (st.attn) is the model-scoped publication store the full
	// forward uses, while this layer's own state carries the compressor group
	// cursor. A source appends through ITS state and then mirrors the completed
	// rows into the registry so a later reader resolves them; a reader reads the
	// registry only. This mirrors v41Layer's source-then-consumer ordering (#12896)
	// at seq == 1.
	var sharedKV, indexKeys [][]float32
	var sourceIdx, sourceRows []int32
	var indexView *v41IndexerKeyView
	if scratch.indexerScore() != nil && scratch.gatherOps != nil && scratch.gatherOps.indexerKeyBudgetBytes > 0 {
		indexView = &v41IndexerKeyView{owner: registry, source: plan.KVSourceLayer}
	}
	if v41OwnsCompressedRows(plan) {
		if indexView != nil {
			indexView.owner, indexView.source = layerState, l
		}
		width := v41CompressorWidth(cfg)
		normWeight := m.tensor(layerName(l, "attn.compressor.norm.weight"))
		pool, perr := NewV41CompressorPool(plan.Ratio, width)
		if perr != nil {
			return nil, v41StageErr(v41StageCompress, l, perr)
		}
		if normalize := scratch.compressorNorm; normalize != nil {
			pool.normalize = func(pooled, gain []float32, eps float32) ([]float32, error) {
				return normalize(l, pooled, gain, eps)
			}
		}
		pool.bindGatedPool(l, scratch.gatherOps.compressorPoolFunc())
		projectKV := func(in []float32) ([]float32, error) {
			return m.v41ProjMatRowsWithProjection(l, "attn.compressor.wkv.weight", in, width, H, scratch.denseProjection())
		}
		projectScore := func(in []float32) ([]float32, error) {
			return m.v41ProjMatRowsWithProjection(l, "attn.compressor.wgate.weight", in, width, H, scratch.denseProjection())
		}
		var indexPub bool
		var projectIndex func([]float32) ([]float32, error)
		if indexSourceAt(cfg.DeepSeekV41, l) {
			indexPub = true
			projectIndex = func(row []float32) ([]float32, error) {
				keys, err := m.v41IndexKeysPublished(l, [][]float32{row}, scratch.denseProjection(), scratch.indexKeyNorm, scratch.gatherOps.bf16PublishFunc())
				if err != nil {
					return nil, err
				}
				return keys[0], nil
			}
		}
		_, _, _, _, _, err = layerState.appendCompressorSourcePublication(
			l, plan.Ratio, pos, collapsed, pool, projectKV, projectScore, normWeight, eps,
			v41CompressorPublication{
				projectIndex: projectIndex,
				finalize: func(start int, latent, key []float32) error {
					return m.v41CompressedPublicationRoPEWithDevice(l, start, latent, key, scratch.gatherOps.compressedPublicationRoPEFunc())
				},
				registry: registry,
				ref:      V41AttentionStateRef{LayerID: l, Ratio: plan.Ratio, IsKVSource: plan.KVSourceLayer == l, IsIndexSource: indexPub},
			})
		if err != nil {
			return nil, v41StageErr(v41StageCompress, l, err)
		}
		// Step owns the state synchronously. Attention consumes these rows before
		// returning; index scoring still packs its own flat key buffer.
		sharedKV, _ = layerState.borrowKVSourceRows(l)
		if indexPub {
			var ok bool
			indexKeys, ok = layerState.borrowIndexKeys(l)
			if len(sharedKV) != len(indexKeys) || (len(sharedKV) > 0 && !ok) {
				return nil, v41StageErr(v41StageIndexer, l, fmt.Errorf("%w: own index history is incomplete", ErrV41ForwardStage))
			}
		}
	} else {
		sharedKV, indexKeys, err = m.v41CompressedReaderStepRows(plan, registry, pos+1)
		if err != nil {
			return nil, err
		}
	}
	if indexSourceAt(cfg.DeepSeekV41, l) {
		sourceIdx, err = m.v41IndexRowsWithPreparedProjection(l, pos, qLat, collapsed, indexKeys, scratch.denseProjection(), scratch.indexerScoreBinding, scratch.indexScoreHealth, scratch.gatherOps, nil, nil, scratch.projectionRoute, indexView)
		if err != nil {
			return nil, err
		}
		sourceRows, err = m.v41RoleStepNormalizeIndex(l, plan, sourceIdx, len(indexKeys))
		if err != nil {
			return nil, err
		}
		if err := registry.PublishTopK(plan.Ratio, [][]int32{sourceRows}); err != nil {
			return nil, err
		}
	}
	// Canonical compressed IDs remain unoffset until the common composer adds
	// this layer's window width. Padding and repeated selections are preserved.
	// PublishTopK copied the source selection. Its fresh normalized row is
	// still private here; no callback ran between publication and attention.
	var ids []int32
	if plan.TopKWidth > 0 {
		var ierr error
		if indexSourceAt(cfg.DeepSeekV41, l) {
			ids = sourceRows
		} else {
			ids, ierr = m.v41RoleStepIndex(registry, l, pos, plan, qLat, collapsed, len(sharedKV))
		}
		if ierr != nil {
			return nil, ierr
		}
		if ids == nil {
			ids, ierr = m.v41RoleStepNormalizeIndex(l, plan, nil, len(sharedKV))
			if ierr != nil {
				return nil, ierr
			}
		}
	}
	if ids == nil {
		ids = make([]int32, len(sharedKV))
		for group := range ids {
			ids[group] = int32(group)
		}
	}
	attentionOpened := m.v41NowNanos()
	o, err := v41CombinedAttentionWithContinuation(l, pos, plan.kvGroupSize(cfg), nH, hd, q, m.tensor(layerName(l, "attn.sink")), window, sharedKV, ids, cfg.attnScale(), &v41ResidentKVView{
		owner: layerState, pos: pos, ringRows: layerState.windowSize, firstKey: windowStart,
		windowRows: len(window), compressedRows: len(sharedKV), sinkKey: layerName(l, "attn.sink"),
	}, scratch.sharedAttentionFunc(), attentionContinuation)
	m.v41NoteAttentionContraction(attentionOpened)
	if err != nil {
		return nil, v41StageErr(v41StageAttention, l, err)
	}
	o, err = v41RotatedOutputForProjection(cfg, l, pos, o, nH, hd, scratch.gatherOps)
	if err != nil {
		return nil, err
	}
	attnProjected, err := projectOutput(o)
	if err != nil {
		return nil, v41StageErr(v41StageAttention, l, err)
	}
	scratch.activationTrace.record(l, pos, v41TraceStageAttnOut, attnProjected)

	work, err = m.referenceV41RoleLayerStepFinish(l, pos, x, streams, mix, attnProjected, scratch)
	if err != nil {
		return nil, err
	}
	windowCommitted := false
	commitWindow := func() error {
		if windowCommitted {
			return nil
		}
		// State.Step would advance an owner's already-committed compressor cursor
		// twice. Retain this layer's own row, including on reader layers and
		// before the first completed source group.
		copy(layerState.window[pos%layerState.windowSize], kv)
		layerState.nextWindowPos, layerState.nextCompressRow = pos+1, pos+1
		retained := min(layerState.retainedWindowRows+1, layerState.windowSize)
		if configured := cfg.windowForLayer(l); configured > 0 {
			retained = min(retained, configured)
		}
		layerState.retainedCopies += retained - layerState.retainedWindowRows
		layerState.retainedWindowRows = retained
		windowCommitted = true
		return nil
	}
	finish := work.finish
	work.finish = func(routed []float32) error {
		if err := finish(routed); err != nil {
			return err
		}
		return commitWindow()
	}
	work.publishKV = commitWindow
	if resident != nil {
		work.cleanup = resident.abort
	}
	return work, nil
}

func (m *Model) referenceV41RoleLayerStepFinish(l, pos int, x []float32, streams [][]float32, mix v41MHCMix, attnOut []float32, scratch *v41ProjScratch) (work *v41LayerStepWork, err error) {
	cfg := m.Cfg
	H := cfg.HiddenSize
	projectFFN, err := m.v41FFNMHCProjector(l, scratch)
	if err != nil {
		return nil, err
	}
	resident := scratch.mhcCarry.resident
	var ffnResidual [][]float32
	var ffnMix v41MHCMix
	var ffnX []float32
	if resident != nil {
		rows, rerr := resident.ffnInput(l, [][]float32{attnOut}, hcItersOrDefault(cfg), hcEpsOrDefault(cfg), float32(cfg.RMSNormEps))
		if rerr != nil {
			return nil, rerr
		}
		ffnX, err = m.v41FFNNormWithPublish(l, rows[0], float32(cfg.RMSNormEps), scratch.ffnNorm.normFunc(), scratch.gatherOps.bf16PublishFunc())
	} else {
		ffnResidual, ffnMix, ffnX, err = m.v41FullFFNInput(l, streams, mix, attnOut, projectFFN, scratch)
	}
	if err != nil {
		return nil, err
	}
	routeCfg, err := v41RouterConfigFullGeometry(cfg)
	if err != nil {
		return nil, err
	}
	picks, err := m.v41RouterPicksForStep(l, ffnX, H, routeCfg, true, scratch, func() []float32 {
		return m.tensor(layerName(l, "ffn.gate.e_score_correction_bias"))
	})
	if err != nil {
		return nil, err
	}
	m.emitExpertRouteAt(l, pos, picks)
	work = &v41LayerStepWork{layer: l, input: ffnX, picks: picks, full: true, scratch: scratch}
	work.scalar = func() ([]float32, error) {
		routed, batched, err := m.v41RoutedLayerStep(l, picks, ffnX, scratch)
		if err != nil {
			return nil, err
		}
		if !batched {
			fullOutputs := make([][]float32, len(picks))
			for slot, pick := range picks {
				stem := "ffn.experts." + itoa(pick.expert)
				y, err := m.v41IncrementalExpert(l, stem, ffnX, pick.weight, scratch)
				if err != nil {
					return nil, err
				}
				fullOutputs[slot] = y
			}
			routed, err = v41FullRoutedSumWith(l, picks, fullOutputs, H, scratch.routedSumFunc())
			if err != nil {
				return nil, err
			}
		}

		return routed, nil
	}
	work.finish = func(routed []float32) error {
		posted := false
		var owner *Session
		var post func(compute.Tensor) error
		if resident != nil {
			owner = resident.d.s
			post = func(sum compute.Tensor) error {
				if err := resident.finishLayerChecked(l, sum); err != nil {
					return err
				}
				posted = true
				return nil
			}
		}
		moe, err := m.v41SharedExpertAddThenForStep(l, ffnX, routed, cfg, routeCfg, scratch, owner, post)
		if err != nil {
			return err
		}
		scratch.activationTrace.record(l, pos, v41TraceStageMoESum, moe)
		if resident != nil {
			if !posted {
				if err := resident.finishLayer(l, [][]float32{moe}); err != nil {
					return err
				}
			}
			if err := resident.publish(l); err != nil {
				return err
			}
			scratch.mhcCarry.nextLayer++
			return nil
		}
		next, err := v41MHCPostBF16On(scratch.mhcCarry.device, l, moe, ffnResidual, ffnMix)
		if err != nil {
			return v41StageErr(v41StageMHC, l, err)
		}
		for h := 0; h < 4; h++ {
			copy(streams[h], next[h])
		}
		copy(x, next[0])
		scratch.mhcCarry.pre[0] = ffnMix.pre
		scratch.mhcCarry.nextLayer++
		return nil
	}
	return work, nil
}

type v41RolePrepare func(int, V41AttentionPlan, []float32, [][]float32, int, *V41AttentionState, *V41AttentionState, *v41ProjScratch) (*v41LayerStepWork, error)

func v41RoleContinuationSeed(t *testing.T, m *Model, prefix []int) *v41ForwardState {
	t.Helper()
	st := &v41ForwardState{}
	if _, err := m.forwardV41(prefix, st); err != nil {
		t.Fatal(err)
	}
	return st
}

// fak-test:runtime fast est=2s lane=default
func TestV41RoleAttentionContinuationParity(t *testing.T) {
	for _, indexed := range []bool{false, true} {
		m := v41ReaderWidthFixture(t)
		if indexed {
			m = v41IncrementalExpertFixture(t, true, false)
		}
		for _, prefix := range [][]int{{1}, {1, 2}, {1, 2, 3}, {1, 2, 3, 4}} {
			var wantX []float32
			var wantS [][]float32
			var wantState *v41ForwardState
			var wantEvents []computetrace.Event
			for mode := 0; mode < 3; mode++ {
				st := v41RoleContinuationSeed(t, m, prefix)
				x, streams := v41LayerStepInputs(t, m, 4, true)
				recorder := computetrace.New(256)
				scratch := &v41ProjScratch{mhcCarry: newV41MHCCarry(1), activationTrace: &v41ActivationTraceState{recorder: recorder, inputDigest: "same-input", weightDigest: "same-weights"}}
				for l := 0; l < m.Cfg.NumLayers; l++ {
					plan, err := m.v41AttentionPlan(l)
					if err != nil {
						t.Fatal(err)
					}
					var work *v41LayerStepWork
					switch mode {
					case 0:
						work, err = m.referenceV41RoleLayerStep(l, plan, x, streams, len(prefix), st.layerState(l), st.attn, scratch)
					case 1:
						work, err = m.prepareV41LayerStepRole(l, plan, x, streams, len(prefix), st.layerState(l), st.attn, scratch)
					case 2:
						c, cause := m.prepareV41RoleAttention(l, plan, x, streams, len(prefix), st.layerState(l), st.attn, scratch)
						if cause != nil {
							t.Fatal(cause)
						}
						copied := c
						projected, cause := c.project(c.input)
						if cause != nil {
							c.close()
							t.Fatal(cause)
						}
						work, err = c.resume(projected)
						c.close()
						copied.close()
						if err == nil && (work.closed || c.work.roleAttentionState != v41RoleAttentionTransferred) {
							t.Fatal("lost transferred ownership")
						}
						if _, cause = copied.resume(projected); cause == nil {
							t.Fatal("copied continuation resumed twice")
						}
					}
					if err != nil {
						t.Fatal(err)
					}
					if err = work.complete(nil); err != nil {
						t.Fatal(err)
					}
				}
				if mode == 0 {
					wantX, wantS, wantState = x, streams, st
					wantEvents = recorder.Artifact().Events
					if len(wantEvents) == 0 {
						t.Fatal("frozen trace absent")
					}
					continue
				}
				events := recorder.Artifact().Events
				if len(events) != len(wantEvents) {
					t.Fatal("trace checkpoint count changed")
				}
				for i, event := range events {
					want := wantEvents[i]
					if event.Phase != want.Phase || event.Layer != want.Layer || event.Token != want.Token || !reflect.DeepEqual(event.Shapes, want.Shapes) || !reflect.DeepEqual(event.SampleIndices, want.SampleIndices) || !reflect.DeepEqual(event.SampleValues, want.SampleValues) {
						t.Fatalf("trace checkpoint %d changed", i)
					}
				}
				if !reflect.DeepEqual(x, wantX) || !reflect.DeepEqual(streams, wantS) || !reflect.DeepEqual(st.layers, wantState.layers) || !reflect.DeepEqual(st.attn, wantState.attn) {
					t.Fatalf("indexed=%t prefix=%d mode=%d differs from frozen scalar", indexed, len(prefix), mode)
				}
			}
		}
	}
}

// fak-test:runtime fast est=1s lane=default
func TestV41RoleAttentionContinuationBoundaries(t *testing.T) {
	m := v41ReaderWidthFixture(t)
	prefix := []int{1, 2, 3}
	sentinel := errors.New("projection stopped")
	plan, err := m.v41AttentionPlan(0)
	if err != nil {
		t.Fatal(err)
	}
	makeInput := func() (*v41ForwardState, []float32, [][]float32, *v41ProjScratch) {
		st := v41RoleContinuationSeed(t, m, prefix)
		x, s := v41LayerStepInputs(t, m, 4, true)
		return st, x, s, &v41ProjScratch{mhcCarry: newV41MHCCarry(1)}
	}
	old, xOld, sOld, scratchOld := makeInput()
	scratchOld.groupedOutput = func(int, []float32, int, int, int, int, int) ([]float32, v41DenseProjectionOutcome, error) {
		return nil, v41ProjectionError, sentinel
	}
	if _, err = m.referenceV41RoleLayerStep(0, plan, xOld, sOld, len(prefix), old.layerState(0), old.attn, scratchOld); !errors.Is(err, sentinel) {
		t.Fatalf("old projection stop=%v", err)
	}
	for _, action := range []string{"close", "bad-width"} {
		st, x, s, scratch := makeInput()
		c, err := m.prepareV41RoleAttention(0, plan, x, s, len(prefix), st.layerState(0), st.attn, scratch)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(st.layers, old.layers) || !reflect.DeepEqual(st.attn, old.attn) || !reflect.DeepEqual(x, xOld) || !reflect.DeepEqual(s, sOld) {
			t.Fatal("suspension changed existing publication boundary")
		}
		copied := c
		if action == "bad-width" {
			if work, err := c.resume(nil); err == nil || work != nil {
				t.Fatal("invalid width accepted")
			}
		} else {
			c.close()
		}
		copied.close()
		c.close()
		if _, err = copied.resume(make([]float32, m.Cfg.HiddenSize)); err == nil {
			t.Fatal("closed copy resumed")
		}
	}
}

// fak-test:runtime fast est=20s lane=default
func TestV41RoleAttentionContinuationAllocations(t *testing.T) {
	for _, indexed := range []bool{false, true} {
		m := v41ReaderWidthFixture(t)
		if indexed {
			m = v41IncrementalExpertFixture(t, true, false)
		}
		for _, prefix := range [][]int{{1}, {1, 2}} {
			measure := func(prepare v41RolePrepare) (float64, testing.BenchmarkResult) {
				st := v41RoleContinuationSeed(t, m, prefix)
				run := func() {
					x, streams := v41LayerStepInputs(t, m, 4, true)
					scratch := &v41ProjScratch{mhcCarry: newV41MHCCarry(1)}
					stats := &v41StepStats{}
					journal := newV41StepJournal(st, stats, st.attn, st.attn, m.Cfg.NumLayers)
					defer journal.rollback()
					for l := 0; l < m.Cfg.NumLayers; l++ {
						journal.stage(l, st.layerState(l))
						plan, err := m.v41AttentionPlan(l)
						if err != nil {
							panic(err)
						}
						work, err := prepare(l, plan, x, streams, len(prefix), st.layerState(l), st.attn, scratch)
						if err != nil {
							panic(err)
						}
						if err = work.complete(nil); err != nil {
							panic(err)
						}
					}
				}
				// AllocsPerRun counts process-wide mallocs, and a concurrent goroutine
				// can only add to them, so the minimum of several samples is the arm's count.
				allocs := testing.AllocsPerRun(10, run)
				for range 4 {
					allocs = min(allocs, testing.AllocsPerRun(10, run))
				}
				result := testing.Benchmark(func(b *testing.B) {
					b.ReportAllocs()
					for i := 0; i < b.N; i++ {
						run()
					}
				})
				return allocs, result
			}
			old, oldResult := measure(m.referenceV41RoleLayerStep)
			aa, aaResult := measure(m.referenceV41RoleLayerStep)
			now, nowResult := measure(m.prepareV41LayerStepRole)
			for _, sample := range []struct {
				name   string
				count  float64
				result testing.BenchmarkResult
			}{
				{"reference-A", old, oldResult}, {"reference-A-control", aa, aaResult}, {"extracted-B", now, nowResult},
			} {
				t.Logf("indexed=%t prefix=%d arm=%s count=%g N=%d MemBytes=%d MemAllocs=%d bytes/op=%d", indexed, len(prefix), sample.name, sample.count, sample.result.N, sample.result.MemBytes, sample.result.MemAllocs, sample.result.AllocedBytesPerOp())
			}
			// Keep the stable allocation-count gate. Global runtime byte counters
			// are diagnostic: prior unchanged-source paired runs showed small byte deltas
			// of 1 to 23 B/op. No same-method resolution is established yet.
			// A/A controls establish resolution; this log does not certify bytes
			// parity or waive the unresolved byte qualification for publication.
			if old != aa || now > old {
				t.Errorf("scalar allocation count regression/control instability: reference=%g repeat=%g extracted=%g", old, aa, now)
			}
		}
	}
}

// fak-test:runtime fast est=1s lane=default
func TestV41RoleAttentionContinuationFailures(t *testing.T) {
	m := v41ReaderWidthFixture(t)
	prefix := []int{1, 2, 3}
	sentinel := errors.New("selected operation failed")
	for _, site := range []string{"projection", "ffn-input", "router"} {
		for _, panics := range []bool{false, true} {
			var oldState *v41ForwardState
			var oldX []float32
			var oldS [][]float32
			var oldErr error
			var oldPanic any
			for mode := 0; mode < 2; mode++ {
				st := v41RoleContinuationSeed(t, m, prefix)
				x, s := v41LayerStepInputs(t, m, 4, true)
				scratch := &v41ProjScratch{mhcCarry: newV41MHCCarry(1)}
				fail := func() error {
					if panics {
						panic(sentinel)
					}
					return sentinel
				}
				switch site {
				case "projection":
					scratch.groupedOutput = func(int, []float32, int, int, int, int, int) ([]float32, v41DenseProjectionOutcome, error) {
						return nil, v41ProjectionError, fail()
					}
				case "ffn-input":
					scratch.mhcFFNProjection = func(int, []float32, int, float32, bool, bool) ([]float32, v41DenseProjectionOutcome, error) {
						return nil, v41ProjectionError, fail()
					}
				case "router":
					scratch.projectionRoute = newV41ProjectionRouteBinding(nil, func(int, []float32, []float32, v41RouterConfig, bool) ([]routePick, error) { return nil, fail() })
				}
				plan, err := m.v41AttentionPlan(0)
				if err != nil {
					t.Fatal(err)
				}
				var work *v41LayerStepWork
				var caught any
				func() {
					defer func() { caught = recover() }()
					if mode == 0 {
						work, err = m.referenceV41RoleLayerStep(0, plan, x, s, len(prefix), st.layerState(0), st.attn, scratch)
					} else {
						work, err = m.prepareV41LayerStepRole(0, plan, x, s, len(prefix), st.layerState(0), st.attn, scratch)
					}
				}()
				if work != nil {
					work.close()
					t.Fatal("failed preparation exposed successful work")
				}
				if panics && caught != sentinel {
					t.Fatalf("%s lost panic: %v", site, caught)
				}
				if !panics && !errors.Is(err, sentinel) {
					t.Fatalf("%s lost error: %v", site, err)
				}
				if mode == 0 {
					oldState, oldX, oldS, oldErr, oldPanic = st, x, s, err, caught
					continue
				}
				if fmt.Sprint(err) != fmt.Sprint(oldErr) || caught != oldPanic || !reflect.DeepEqual(st.layers, oldState.layers) || !reflect.DeepEqual(st.attn, oldState.attn) || !reflect.DeepEqual(x, oldX) || !reflect.DeepEqual(s, oldS) {
					t.Fatalf("%s panic=%t changed failure boundary", site, panics)
				}
			}
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestV41RoleAttentionContinuationCleanupOwnership(t *testing.T) {
	makePending := func(cause any) (v41RoleAttentionContinuation, *v41PlainContinuationFreeBackend) {
		b := &v41PlainContinuationFreeBackend{Backend: compute.Default(), cause: cause}
		r := &v41MHCResident{d: &v41MHCDevice{s: &Session{Backend: b}}}
		r.layer.attn = compute.NewF32(b, []int{1}, []float32{1})
		return v41RoleAttentionContinuation{resident: r, work: &v41LayerStepWork{}}, b
	}
	c, b := makePending(nil)
	copied := c
	c.close()
	copied.close()
	if b.frees != 1 {
		t.Fatalf("copied close frees=%d", b.frees)
	}
	c, b = makePending(nil)
	c.work.roleAttentionState = v41RoleAttentionTransferred
	c.close()
	if b.frees != 0 {
		t.Fatal("close stole transferred resource")
	}
	c.work.cleanup = c.resident.abort
	c.work.close()
	c.work.close()
	if b.frees != 1 {
		t.Fatal("work did not own exactly one cleanup")
	}
	primary, secondary := errors.New("primary"), errors.New("secondary cleanup")
	c, b = makePending(secondary)
	copied = c
	var got any
	func() { defer func() { got = recover() }(); defer c.close(); panic(primary) }()
	if got != secondary || b.frees != 1 {
		t.Fatalf("cleanup panic precedence=%v frees=%d", got, b.frees)
	}
	copied.close()
	if b.frees != 1 {
		t.Fatal("copy retried failed cleanup")
	}
}

// fak-test:runtime fast est=200ms lane=default
func TestV41RoleAttentionContinuationFailureAllocationCost(t *testing.T) {
	m := v41ReaderWidthFixture(t)
	prefix := []int{1, 2, 3}
	sentinel := errors.New("projection refusal")
	measure := func(prepare v41RolePrepare) float64 {
		st := v41RoleContinuationSeed(t, m, prefix)
		x, streams := v41LayerStepInputs(t, m, 4, true)
		scratch := &v41ProjScratch{mhcCarry: newV41MHCCarry(1)}
		scratch.groupedOutput = func(int, []float32, int, int, int, int, int) ([]float32, v41DenseProjectionOutcome, error) {
			return nil, v41ProjectionError, sentinel
		}
		plan, err := m.v41AttentionPlan(0)
		if err != nil {
			t.Fatal(err)
		}
		return testing.AllocsPerRun(10, func() {
			journal := newV41StepJournal(st, &v41StepStats{}, st.attn, st.attn, m.Cfg.NumLayers)
			journal.stage(0, st.layerState(0))
			defer journal.rollback()
			work, err := prepare(0, plan, x, streams, len(prefix), st.layerState(0), st.attn, scratch)
			if work != nil || !errors.Is(err, sentinel) {
				panic("projection failure oracle did not fire")
			}
		})
	}
	old, now := measure(m.referenceV41RoleLayerStep), measure(m.prepareV41LayerStepRole)
	t.Logf("projection failure allocation cost: frozen=%g extracted=%g delta=%g; early private holder/window closure", old, now, now-old)
}

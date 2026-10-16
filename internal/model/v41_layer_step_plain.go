package model

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// v41PlainLayerStep carries one plain per-layer position from the mHC collapse
// through its deferred MoE finish, so each phase can stay a small method while
// the work's scalar and finish callbacks see the same values the prepare built.
type v41PlainLayerStep struct {
	m          *Model
	l, pos     int
	x          []float32
	streams    [][]float32
	layerState *V41AttentionState
	scratch    *v41ProjScratch
	full       bool
	eps        float32
	mhcDevice  *v41MHCDevice
	resident   *v41MHCResident

	mix         v41MHCMix
	collapsed   []float32
	kv          []float32
	attnOut     []float32
	routeCfg    v41RouterConfig
	ffnX        []float32
	ffnResidual [][]float32
	ffnMix      v41MHCMix
	picks       []routePick
}

func (m *Model) prepareV41PlainLayerStep(l int, x []float32, streams [][]float32, pos int, layerState *V41AttentionState, scratch *v41ProjScratch) (work *v41LayerStepWork, err error) {
	cfg := m.Cfg
	full, err := v41ForwardGeometry(cfg)
	if err != nil {
		return nil, err
	}
	st := &v41PlainLayerStep{m: m, l: l, pos: pos, x: x, streams: streams, layerState: layerState, scratch: scratch, full: full, eps: float32(cfg.RMSNormEps)}
	if full {
		if err := scratch.mhcCarry.validate(l, 1); err != nil {
			return nil, err
		}
		if err := m.v41AdmitMHC(l); err != nil {
			return nil, err
		}
		st.mhcDevice = scratch.mhcCarry.device
		st.resident = scratch.mhcCarry.resident
	}
	if st.resident != nil {
		defer func() {
			if work == nil {
				st.resident.abort()
			}
		}()
	}
	H, hd, nH := cfg.HiddenSize, cfg.HeadDim, cfg.NumHeads
	attnNorm := m.tensor(layerName(l, "attn_norm.weight"))
	mixBase := m.tensor(layerName(l, "mhc.base"))
	mixScale := m.tensor(layerName(l, "mhc.scale"))
	projectOutput := m.v41GroupedOutputProjector(l, nH, hd, cfg.OGroups, cfg.OLoraRank, H, scratch)
	if err := m.v41RequirePlainLayerStepWeights(l); err != nil {
		return nil, err
	}
	sink := m.tensor(layerName(l, "attn.sink"))
	gateBias := m.tensor(layerName(l, "ffn.gate.e_score_correction_bias"))

	if err := st.collapseMHC(attnNorm, mixScale, mixBase); err != nil {
		return nil, err
	}
	qLat, prefixHandled, err := st.queryInput()
	if err != nil {
		return nil, err
	}
	q, err := st.projectQKV(qLat, prefixHandled)
	if err != nil {
		return nil, err
	}
	visibleKeys, err := st.visibleKeys()
	if err != nil {
		return nil, err
	}
	if err := st.attend(q, sink, visibleKeys, projectOutput); err != nil {
		return nil, err
	}
	if err := st.ffnInput(); err != nil {
		return nil, err
	}
	st.picks, err = m.v41RouterPicksForStep(l, st.ffnX, H, st.routeCfg, full, scratch, func() []float32 { return gateBias })
	if err != nil {
		return nil, err
	}
	m.emitExpertRouteAt(l, pos, st.picks)
	work = &v41LayerStepWork{layer: l, input: st.ffnX, picks: st.picks, full: full, scratch: scratch}
	work.scalar = st.routed
	work.finish = st.finish
	if st.resident != nil {
		work.cleanup = st.resident.abort
	}
	return work, nil
}

func (m *Model) v41RequirePlainLayerStepWeights(l int) error {
	for _, leaf := range []string{
		"attn.wq_a.weight", "attn.wq_b.weight", "attn.wkv.weight",
		"ffn.gate.weight",
		"ffn.shared_experts.w1.weight", "ffn.shared_experts.w3.weight",
		"ffn.shared_experts.w2.weight",
	} {
		if !m.hasResidentWeight(layerName(l, leaf)) {
			return v41StageErr(v41StageAttention, l,
				fmt.Errorf("%w: missing tensor %s", ErrV41ForwardStage, layerName(l, leaf)))
		}
	}
	return nil
}

func (st *v41PlainLayerStep) collapseMHC(attnNorm, mixScale, mixBase []float32) error {
	m, l, cfg, scratch := st.m, st.l, st.m.Cfg, st.scratch
	H := cfg.HiddenSize
	mhcFlat, mhcTransposed, mhcOK := m.v41MHCWeightLayout(l)
	if !mhcOK {
		return v41StageErr(v41StageMHC, l,
			fmt.Errorf("%w: mHC mix weight holds no admitted geometry", ErrV41ForwardStage))
	}
	if st.resident != nil {
		rows, err := st.resident.beginLayer(l, hcItersOrDefault(cfg), hcEpsOrDefault(cfg), st.eps)
		if err != nil {
			return err
		}
		st.collapsed = rows[0]
		return nil
	}
	var xn []float32
	if !st.full {
		xn = rmsnormCfg(st.x, attnNorm, st.eps, cfg)
	}
	input := xn
	if mhcFlat {
		width, ok := checkedMulInt(4, H)
		if !ok || len(st.streams) != 4 {
			return v41StageErr(v41StageMHC, l, errV41ProjectionResult)
		}
		input = make([]float32, 0, width)
		for _, stream := range st.streams {
			if len(stream) != H {
				return v41StageErr(v41StageMHC, l, errV41ProjectionResult)
			}
			input = append(input, stream...)
		}
	}
	projectMHC := m.v41MHCProjector(l, H, st.eps, mhcFlat, mhcTransposed, scratch)
	mixes, err := projectMHC(input)
	if err != nil {
		return err
	}
	st.mix, err = v41MHCSplitOn(st.mhcDevice, l, mixes, mixScale, mixBase, 4, hcItersOrDefault(cfg), hcEpsOrDefault(cfg))
	if err != nil {
		return err
	}
	// The reduced path pre-collapses four identical copies of the normalized
	// input, byte-identical to v41Layer's seq==1 branch.
	streams4 := st.streams
	if !st.full {
		streams4 = [][]float32{xn, xn, xn, xn}
	}
	pre := st.mix.pre
	if st.full {
		pre = scratch.mhcCarry.pre[0]
	}
	if st.mhcDevice != nil {
		st.collapsed, err = st.mhcDevice.preBF16(l, streams4, pre)
		return err
	}
	st.collapsed, err = v41MHCPre(streams4, pre)
	if err != nil {
		return v41StageErr(v41StageMHC, l, err)
	}
	return nil
}

func (st *v41PlainLayerStep) queryInput() (qLat []float32, prefixHandled bool, err error) {
	if !st.full {
		return nil, false, nil
	}
	l, scratch := st.l, st.scratch
	if scratch.queryPrefix != nil {
		var nextCollapsed []float32
		nextCollapsed, qLat, prefixHandled, err = scratch.queryPrefix(l, st.collapsed)
		if err != nil {
			return nil, false, err
		}
		if prefixHandled {
			st.collapsed = nextCollapsed
			return qLat, true, nil
		}
	}
	st.collapsed, err = st.m.v41AttentionInputNormWithPublications(l, st.collapsed, st.eps, scratch.attentionNormFunc(), scratch.attentionBF16Func())
	if err != nil {
		return nil, false, err
	}
	return qLat, false, nil
}

func (st *v41PlainLayerStep) projectQKV(qLat []float32, prefixHandled bool) ([]float32, error) {
	m, l, pos, cfg, scratch := st.m, st.l, st.pos, st.m.Cfg, st.scratch
	H, hd, nH := cfg.HiddenSize, cfg.HeadDim, cfg.NumHeads
	ropeDim := cfg.QKRopeHeadDim
	if ropeDim <= 0 || ropeDim%2 != 0 || ropeDim > hd {
		return nil, v41StageErr(v41StageAttention, l,
			fmt.Errorf("%w: attention qk_rope_head_dim must be a positive even value <= head_dim %d, got %d", ErrV41ForwardStage, hd, ropeDim))
	}
	var err error
	if !prefixHandled {
		qLat, err = m.v41ProjMatRowsWithProjection(l, "attn.wq_a.weight", st.collapsed, cfg.QLoraRank, H, scratch.denseProjection())
		if err != nil {
			return nil, err
		}
		if st.full {
			if err := m.v41QueryNormInPlaceWithPublish(l, qLat, st.eps, scratch.queryNorm.normFunc(), scratch.gatherOps.bf16PublishFunc()); err != nil {
				return nil, err
			}
		}
	}
	scratch.activationTrace.record(l, pos, v41TraceStageQLatent, qLat)
	q, err := m.v41ProjMatRowsWithProjection(l, "attn.wq_b.weight", qLat, nH*hd, cfg.QLoraRank, scratch.denseProjection())
	if err != nil {
		return nil, err
	}
	var kv []float32
	if st.full {
		if hd > v41KVLoraRank {
			return nil, v41StageErr(v41StageAttention, l,
				fmt.Errorf("%w: attention head_dim %d exceeds full KV latent rank %d", ErrV41ForwardStage, hd, v41KVLoraRank))
		}
		kvFull, err := m.v41ProjMatRowsWithProjection(l, "attn.wkv.weight", st.collapsed, v41KVLoraRank, H, scratch.denseProjection())
		if err != nil {
			return nil, err
		}
		kv = kvFull[:hd]
		if err := m.v41KVNormWithBinding(l, kv, st.eps, scratch.kvNorm, scratch.gatherOps); err != nil {
			return nil, err
		}
	} else {
		kv, err = m.v41ProjMatRowsWithProjection(l, "attn.wkv.weight", st.collapsed, hd, H, scratch.denseProjection())
		if err != nil {
			return nil, err
		}
	}
	cos, sin := v41RopeTableForLayer(cfg, l, pos)
	scratch.activationTrace.record(l, pos, v41TraceStageKVLatent, kv)
	q, st.kv, err = v41AttentionQKRoPEWithPublish(cfg, l, q, kv, cos, sin, nH, hd, ropeDim, scratch.tailRoPE, scratch.gatherOps.bf16PublishFunc())
	if err != nil {
		return nil, err
	}
	return q, nil
}

// visibleKeys returns the ordered absolute key rows this position attends. The
// retained tail is derived from ring occupancy, min(nextWindowPos, windowSize)
// rows ending at nextWindowPos, rather than retainedWindowRows, which
// seedTemporal sets but the incremental Step does not grow.
func (st *v41PlainLayerStep) visibleKeys() ([]int32, error) {
	l, pos := st.l, st.pos
	windowKeys := v41PlainWindowKeys(pos, st.m.Cfg.windowForLayer(l))
	retainedRows := st.layerState.retainedTailRowCount()
	retainedLo := pos - retainedRows
	// A GLOBAL layer whose prefix outgrew the ring, or a LOCAL window wider than
	// the ring, has lost its oldest visible keys; contracting the surviving tail
	// would emit logits from a truncated context, so refuse before the commit.
	if len(windowKeys) > 0 {
		if firstVisible := int(windowKeys[0]); firstVisible < retainedLo {
			return nil, v41StageErr(v41StageAttention, l,
				fmt.Errorf("%w: layer %d cannot step position %d from a %d-row retained ring: visible keys [%d,%d] are not fully retained (oldest %d < %d)",
					ErrV41ForwardStage, l, pos, retainedRows, firstVisible, int(windowKeys[len(windowKeys)-1]), firstVisible, retainedLo))
		}
	}
	visibleKeys := make([]int32, 0, len(windowKeys))
	for _, k := range windowKeys {
		if int(k) < pos {
			if int(k) < retainedLo {
				continue
			}
			visibleKeys = append(visibleKeys, k)
		} else {
			visibleKeys = append(visibleKeys, int32(pos))
		}
	}
	if len(visibleKeys) == 0 {
		visibleKeys = append(visibleKeys, int32(pos))
	}
	return visibleKeys, nil
}

func (st *v41PlainLayerStep) attend(q, sink []float32, visibleKeys []int32, projectOutput func([]float32) ([]float32, error)) error {
	m, l, pos, cfg, scratch, layerState := st.m, st.l, st.pos, st.m.Cfg, st.scratch, st.layerState
	hd, nH := cfg.HeadDim, cfg.NumHeads
	// The ring is not advanced until this layer succeeds; no borrowed row
	// reaches a callback.
	flatKV, err := layerState.stepWindowKV(l, pos, visibleKeys, st.kv)
	if err != nil {
		return err
	}
	idx := v41PlainWindowIndexList(visibleKeys)
	opened := m.v41NowNanos()
	o, err := v41SparseAttentionSinkResident(l, q, flatKV, sink, idx, V41SparseAttentionSinkOptions{
		B: 1, M: 1, Heads: nH, HeadDim: hd, TopK: len(visibleKeys) + 1, N: len(visibleKeys), Softmax: cfg.attnScale(),
	}, &v41ResidentKVView{
		owner: layerState, pos: pos, ringRows: layerState.windowSize, firstKey: int(visibleKeys[0]),
		windowRows: len(visibleKeys), sinkKey: layerName(l, "attn.sink"),
	}, scratch.sharedAttentionFunc())
	m.v41NoteAttentionContraction(opened)
	if err != nil {
		return v41StageErr(v41StageAttention, l, err)
	}
	o, err = v41RotatedOutputForProjection(cfg, l, pos, o, nH, hd, scratch.gatherOps)
	if err != nil {
		return err
	}
	st.attnOut, err = projectOutput(o)
	if err != nil {
		return v41StageErr(v41StageAttention, l, err)
	}
	scratch.activationTrace.record(l, pos, v41TraceStageAttnOut, st.attnOut)
	return nil
}

func (st *v41PlainLayerStep) ffnInput() error {
	m, l, scratch := st.m, st.l, st.scratch
	var err error
	st.routeCfg, err = v41RouterConfigFullGeometry(m.Cfg)
	if err != nil {
		return err
	}
	switch {
	case st.resident != nil:
		rows, rerr := st.resident.ffnInput(l, [][]float32{st.attnOut}, hcItersOrDefault(m.Cfg), hcEpsOrDefault(m.Cfg), st.eps)
		if rerr != nil {
			return rerr
		}
		st.ffnX, err = m.v41FFNNormWithPublish(l, rows[0], st.eps, scratch.ffnNorm.normFunc(), scratch.gatherOps.bf16PublishFunc())
	case st.full:
		projectFFN, ferr := m.v41FFNMHCProjector(l, scratch)
		if ferr != nil {
			return ferr
		}
		st.ffnResidual, st.ffnMix, st.ffnX, err = m.v41FullFFNInput(l, st.streams, st.mix, st.attnOut, projectFFN, scratch)
	default:
		st.ffnX, err = m.v41FFNNormWithPublish(l, st.x, st.eps, scratch.ffnNorm.normFunc(), scratch.gatherOps.bf16PublishFunc())
	}
	return err
}

func (st *v41PlainLayerStep) routed() ([]float32, error) {
	m, l, scratch := st.m, st.l, st.scratch
	H := m.Cfg.HiddenSize
	if st.full {
		routed, batched, err := m.v41RoutedLayerStep(l, st.picks, st.ffnX, scratch)
		if err != nil {
			return nil, err
		}
		if batched {
			return routed, nil
		}
	}
	routed := make([]float32, H)
	var fullOutputs [][]float32
	if st.full {
		fullOutputs = make([][]float32, len(st.picks))
	}
	for slot, pick := range st.picks {
		stem := "ffn.experts." + itoa(pick.expert)
		y, err := m.v41IncrementalExpert(l, stem, st.ffnX, pick.weight, scratch)
		if err != nil {
			return nil, err
		}
		if st.full {
			fullOutputs[slot] = y
			continue
		}
		for i := range routed {
			routed[i] += pick.weight * y[i]
		}
	}
	if st.full {
		return v41FullRoutedSumWith(l, st.picks, fullOutputs, H, scratch.routedSumFunc())
	}
	return routed, nil
}

func (st *v41PlainLayerStep) finish(routed []float32) error {
	m, l, cfg, scratch := st.m, st.l, st.m.Cfg, st.scratch
	posted := false
	var owner *Session
	var post func(compute.Tensor) error
	if st.resident != nil {
		owner = st.resident.d.s
		post = func(sum compute.Tensor) error {
			if err := st.resident.finishLayerChecked(l, sum); err != nil {
				return v41StageErr(v41StageMHC, l, err)
			}
			posted = true
			return nil
		}
	}
	moe, err := m.v41SharedExpertAddThenForStep(l, st.ffnX, routed, cfg, st.routeCfg, scratch, owner, post)
	if err != nil {
		return err
	}
	scratch.activationTrace.record(l, st.pos, v41TraceStageMoESum, moe)
	next, err := st.postMix(moe, posted)
	if err != nil {
		return v41StageErr(v41StageMHC, l, err)
	}
	return st.commit(next)
}

func (st *v41PlainLayerStep) postMix(moe []float32, posted bool) ([][]float32, error) {
	switch {
	case st.resident != nil:
		if posted {
			return nil, nil
		}
		return nil, st.resident.finishLayer(st.l, [][]float32{moe})
	case st.full:
		return v41MHCPostBF16On(st.mhcDevice, st.l, moe, st.ffnResidual, st.ffnMix)
	}
	delta := make([]float32, st.m.Cfg.HiddenSize)
	for i := range delta {
		delta[i] = st.attnOut[i] + moe[i]
	}
	residual := [][]float32{st.collapsed, st.collapsed, st.collapsed, st.collapsed}
	return v41MHCPost(delta, residual, st.mix.post, st.mix.comb)
}

// commit appends the projected KV row: position 0 seeds the empty state through
// Prefill, later positions append through Step. Both are atomic, so a failure
// leaves the state unchanged and the caller's x/streams are not written.
func (st *v41PlainLayerStep) commit(next [][]float32) error {
	var err error
	if st.pos == 0 {
		err = st.layerState.Prefill([][]float32{st.kv}, nil)
	} else {
		err = st.layerState.Step(st.kv, nil)
	}
	if err != nil {
		return v41StageErr(v41StageAttention, st.l, err)
	}
	switch {
	case st.resident != nil:
		if err := st.resident.publish(st.l); err != nil {
			return err
		}
		st.scratch.mhcCarry.nextLayer++
	case st.full:
		for h := 0; h < 4; h++ {
			copy(st.streams[h], next[h])
		}
		copy(st.x, next[0])
		st.scratch.mhcCarry.pre[0] = st.ffnMix.pre
		st.scratch.mhcCarry.nextLayer++
	default:
		copy(st.x, next[0])
	}
	return nil
}

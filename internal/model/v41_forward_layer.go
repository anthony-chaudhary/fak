package model

import (
	"errors"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/model/ffn"
)

// v41RoleSchedule reports whether any configured layer resolves to a non-plain
// attention role. It is the release gate for the shared source registry: a role
// schedule must retain the source publications across a prefill so a later
// incremental Step can resolve them (#13480).
func (m *Model) v41RoleSchedule() bool {
	cfg := m.Cfg
	roles := m.v41AttentionRolesCached()
	for l := 0; l < cfg.NumLayers; l++ {
		if v41CompressRatioAt(cfg, l) > 0 {
			return true
		}
		if role, ok := roles[l]; ok && role != V41AttentionRolePerLayer {
			return true
		}
	}
	return false
}

// v41Layer applies one V4.1 decoder layer to x in place, updating the persistent
// mHC streams[t] for each position. tokens carries the ids for the positions in
// x so a declared Engram layer can hash them. The reduced path reproduces the
// pre-#13009 arithmetic exactly (four identical stand-in streams derived from the
// normalized input, and stream 0 of the post-mix written back); the full path
// reads and writes all four DISTINCT persistent streams.
func (m *Model) v41Layer(l int, tokens []int, x [][]float32, streams [][][]float32, full bool, hd, nH, H int, eps float32, hcIters int, hcEps float32, routeCfg v41RouterConfig, st *v41ForwardState, scratch *v41ProjScratch, trace *v41ActivationTraceState) error {
	cfg := m.Cfg
	if scratch == nil {
		scratch = &v41ProjScratch{}
	}
	var mhcDevice *v41MHCDevice
	if full {
		if err := scratch.mhcCarry.validate(l, len(x)); err != nil {
			return err
		}
		if err := m.v41AdmitMHC(l); err != nil {
			return err
		}
		mhcDevice = scratch.mhcCarry.device
	}
	var resident *v41MHCResident
	if full {
		resident = scratch.mhcCarry.resident
	}
	if resident != nil {
		// A refused or failed layer never publishes its pending panel.
		defer resident.abort()
	}
	scratch.projectionRoute = nil
	scratch.groupedOutput = nil
	scratch.mhcProjection = nil
	scratch.mhcFFNProjection = nil
	scratch.attentionNormBinding = nil
	scratch.queryNorm = nil
	scratch.queryPrefix = nil
	scratch.kvNorm = nil
	scratch.ffnNorm = nil
	scratch.compressorNorm = nil
	scratch.indexKeyNorm = nil
	scratch.indexerScoreBinding = nil
	scratch.indexScoreHealth = nil
	scratch.sharedActivation = nil
	scratch.tailRoPE = nil
	scratch.clearSharedAttention()
	scratch.setRoutedSum(nil)
	scratch.sharedAdd = nil
	scratch.gatherOps = nil
	if st != nil {
		scratch.projectionRoute = st.projectionRoute
		scratch.groupedOutput = st.groupedOutput
		scratch.mhcProjection = st.mhcProjection
		scratch.mhcFFNProjection = st.mhcFFNProjection
		scratch.attentionNormBinding = st.attentionNormBinding
		scratch.queryNorm = st.queryNorm
		scratch.queryPrefix = st.queryPrefix
		scratch.kvNorm = st.kvNorm
		scratch.ffnNorm = st.ffnNorm
		scratch.compressorNorm = st.compressorNorm
		scratch.indexKeyNorm = st.indexKeyNorm
		scratch.indexerScoreBinding = st.indexerScoreBinding
		scratch.indexScoreHealth = st.indexScoreHealth
		scratch.sharedActivation = st.sharedActivation
		scratch.tailRoPE = st.tailRoPE
		scratch.copySharedAttentionFrom(st)
		scratch.copyRoutedSumFrom(st)
		scratch.sharedAdd = st.sharedAdd
		scratch.gatherOps = st.gatherOps
	}
	defer func() {
		scratch.projectionRoute = nil
		scratch.groupedOutput = nil
		scratch.mhcProjection = nil
		scratch.mhcFFNProjection = nil
		scratch.attentionNormBinding = nil
		scratch.queryNorm = nil
		scratch.queryPrefix = nil
		scratch.kvNorm = nil
		scratch.ffnNorm = nil
		scratch.compressorNorm = nil
		scratch.indexKeyNorm = nil
		scratch.indexerScoreBinding = nil
		scratch.indexScoreHealth = nil
		scratch.sharedActivation = nil
		scratch.tailRoPE = nil
		scratch.clearSharedAttention()
		scratch.setRoutedSum(nil)
		scratch.sharedAdd = nil
		scratch.gatherOps = nil
	}()

	// Engram injection happens at the START of the layer, into the residual,
	// before attention and before attn_norm (ds41_graph_before_attention).
	if cfg.DeepSeekV41 != nil {
		for _, eng := range cfg.DeepSeekV41.EngramLayerIDs {
			if eng == l {
				var project v41EngramProjectionFunc
				if st != nil {
					project = st.engramProjection
				}
				if resident != nil {
					// The panel never leaves the device when the session binds the
					// Engram gate; device-only refuses rather than reach the host
					// round trip below.
					if gate := scratch.gatherOps.engramGateFunc(); gate != nil {
						geometry, layerRows, err := m.v41EngramLayerRows(l, tokens)
						if err != nil {
							return err
						}
						_, handled, err := m.v41EngramInjectResident(l, resident.streams, len(tokens), layerRows, eps, geometry, gate)
						if err != nil {
							return err
						}
						if handled {
							break
						}
					}
					host, err := resident.hostStreams(l)
					if err != nil {
						return err
					}
					for t := range x {
						copy(x[t], host[t][0])
					}
					if err := m.v41EngramInjectWithProjection(l, x, host, full, tokens, eps, project); err != nil {
						return err
					}
					if err := resident.setHostStreams(l, host); err != nil {
						return err
					}
					break
				}
				if err := m.v41EngramInjectWithProjection(l, x, streams, full, tokens, eps, project); err != nil {
					return err
				}
				break
			}
		}
	}

	attnNorm := m.tensor(layerName(l, "attn_norm.weight"))
	mixBase := m.tensor(layerName(l, "mhc.base"))
	mixScale := m.tensor(layerName(l, "mhc.scale"))
	projectOutput := m.v41GroupedOutputProjector(l, nH, hd, cfg.OGroups, cfg.OLoraRank, H, scratch)
	// The remaining per-layer projections are applied only through matRows, so
	// they read through the streaming-safe v41ProjMatRows at their use site
	// instead of materializing a whole f32 block here (#2150). Fail closed NOW,
	// with the same typed #13276 refusal, if any is absent from every store, so
	// the refuse happens before any layer arithmetic rather than mid-forward.
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
	// attn.sink is a 1-D per-head vector (not an isQuantWeight matmul leaf), so it
	// stays on the f32 manifest.
	sink := m.tensor(layerName(l, "attn.sink"))
	gateBias := m.tensor(layerName(l, "ffn.gate.e_score_correction_bias"))

	seq := len(x)
	// ---- mHC coefficient split (one split per layer) ----
	//
	// The mHC mix projection geometry is per-layer (it is a per-layer weight), so
	// resolve it once. The published artifact's hc_attn_fn is the flattened
	// four-stream projection (logical 24 x 4H, stored [4H, 24]); the reference
	// (inference/model.py mHC; v4_flash_oracle_test.go:168) computes it over the
	// four width-H streams laid end to end with a single shared flatten-RMS. The
	// reduced fixture uses the legacy 24 x H single-stream matmul. Both are
	// executed here; only the reduced path's arithmetic is held byte-identical to
	// pre-#13009.
	mhcFlat, mhcTransposed, mhcOK := m.v41MHCWeightLayout(l)
	if !mhcOK {
		return v41StageErr(v41StageMHC, l,
			fmt.Errorf("%w: mHC mix weight holds no admitted geometry", ErrV41ForwardStage))
	}
	projectMHC := m.v41MHCProjector(l, H, eps, mhcFlat, mhcTransposed, scratch)
	var hcByPos []v41MHCMix
	if resident == nil {
		hcByPos = make([]v41MHCMix, seq)
	}
	preByPos := make([][]float32, seq)
	if resident != nil {
		collapsedRows, err := resident.beginLayer(l, hcIters, hcEps, eps)
		if err != nil {
			return err
		}
		normalized, handled, err := scratch.attentionNormBinding.coldPanel(m, st, scratch, resident, l, collapsedRows, eps)
		if err != nil {
			return err
		}
		if handled {
			preByPos = normalized
		} else {
			for t := range collapsedRows {
				if preByPos[t], err = m.v41AttentionInputNormWithPublications(l, collapsedRows[t], eps, scratch.attentionNormFunc(), scratch.attentionBF16Func()); err != nil {
					return err
				}
			}
		}
	}
	for t := 0; t < seq && resident == nil; t++ {
		// The reduced path projects the normalized input through the legacy
		// single-stream [24, H] block; the full path projects the four DISTINCT
		// persistent streams through the flattened 4H residual with one shared RMS.
		// Only the reduced stand-in consumes a pre-collapse host norm.
		var xn []float32
		if !full {
			xn = rmsnormCfg(x[t], attnNorm, eps, cfg)
		}
		input := xn
		if mhcFlat {
			width, ok := checkedMulInt(4, H)
			if !ok || len(streams[t]) != 4 {
				return v41StageErr(v41StageMHC, l, errV41ProjectionResult)
			}
			input = make([]float32, 0, width)
			for _, stream := range streams[t] {
				if len(stream) != H {
					return v41StageErr(v41StageMHC, l, errV41ProjectionResult)
				}
				input = append(input, stream...)
			}
		}
		mixes, err := projectMHC(input)
		if err != nil {
			return err
		}
		mix, err := v41MHCSplitOn(mhcDevice, l, mixes, mixScale, mixBase, 4, hcIters, hcEps)
		if err != nil {
			return err
		}
		hcByPos[t] = mix
		// The mHC pre-collapse always reads a four-stream set collapsed by the
		// learned pre coefficients. The reduced path reconstructs the reference's
		// stand-in (four identical copies of the normalized input, byte-identical
		// to pre-#13009) rather than reading the persistent set, so its numerics
		// are unchanged. The full path reads the four DISTINCT persistent streams
		// carried into this layer (stream 0 = live hidden, 1..3 = residual).
		streams4 := streams[t]
		if !full {
			streams4 = [][]float32{xn, xn, xn, xn}
		}
		pre := mix.pre
		if full {
			pre = scratch.mhcCarry.pre[t]
		}
		var collapsed []float32
		if mhcDevice != nil {
			// The device publishes hc_pre's BF16 copy; the input norm's own BF16
			// widening of it is the identity.
			collapsed, err = mhcDevice.preBF16(l, streams4, pre)
			if err != nil {
				return err
			}
		} else {
			collapsed, err = v41MHCPre(streams4, pre)
			if err != nil {
				return v41StageErr(v41StageMHC, l, err)
			}
		}
		if full {
			collapsed, err = m.v41AttentionInputNormWithPublications(l, collapsed, eps, scratch.attentionNormFunc(), scratch.attentionBF16Func())
			if err != nil {
				return err
			}
		}
		preByPos[t] = collapsed
	}

	// ---- attention: projected q/kv per position, then sparse sink per position ----
	qHeads := make([][]float32, seq) // [t][nH*hd], rotated
	kvRows := make([][]float32, seq) // [t][hd], rotated (single KV head)
	qLatRows := make([][]float32, seq)
	// RoPE geometry, fail-closed. The reference rotates only the LAST
	// qk_rope_head_dim components of each head (the leading qk_nope_head_dim pass
	// through byte-identical) using the interleaved adjacent-pair convention. A
	// head width that cannot carry that slice (a non-positive, odd, or
	// head-wider rope dim) is refused with a typed error rather than silently
	// rotating a wrong sub-vector.
	ropeDim := cfg.QKRopeHeadDim
	if ropeDim <= 0 || ropeDim%2 != 0 || ropeDim > hd {
		return v41StageErr(v41StageAttention, l,
			fmt.Errorf("%w: attention qk_rope_head_dim must be a positive even value <= head_dim %d, got %d", ErrV41ForwardStage, hd, ropeDim))
	}
	// Batched query projections across the prefill panel: residentMatMulBatch
	// reads each attn.wq_a / attn.wq_b weight row ONCE and reuses it across all
	// seq prompt tokens, instead of the per-token GEMV loop re-streaming every
	// weight seq times (#13301). The per-token q_norm and RoPE glue below stays
	// scalar, so this is byte-for-byte the per-token v41ProjMatRows path (the
	// adapter's contract); seq==1 (decode) stays entirely on that scalar path.
	preFlat := make([]float32, seq*H)
	for t := 0; t < seq; t++ {
		copy(preFlat[t*H:(t+1)*H], preByPos[t])
	}
	qLatPanel, err := m.v41ProjPanelWithProjection(l, "attn.wq_a.weight", preFlat, cfg.QLoraRank, H, seq, scratch.denseProjection())
	if err != nil {
		return err
	}
	for t := 0; t < seq; t++ {
		qLat := qLatPanel[t*cfg.QLoraRank : (t+1)*cfg.QLoraRank]
		if full {
			// Reference: qr = self.q_norm(self.wq_a(x)). The RMSNorm over the
			// q-lora latent keeps its magnitude O(1) before the wq_b projection;
			// omitting it lets ~1e18 activations reach the 512-term attention
			// accumulation and overflow it to +Inf at HeadDim=512 (#13290).
			// Artifact-only: the reduced fixture carries no q_norm leaf and its
			// pre-#13009 arithmetic is unchanged. Written back into the panel row
			// so the batched wq_b panel reads the normed latents.
			if err := m.v41QueryNormInPlaceWithPublish(l, qLat, eps, scratch.queryNorm, scratch.gatherOps.bf16PublishFunc()); err != nil {
				return err
			}
		}
		// #13325 q_latent: the q-lora latent as fed to wq_b, after the optional q
		// RMSNorm. Captured BEFORE the wq_b panel consumes it.
		trace.record(l, t, v41TraceStageQLatent, qLat)
	}
	qPanel, err := m.v41ProjPanelWithProjection(l, "attn.wq_b.weight", qLatPanel, nH*hd, cfg.QLoraRank, seq, scratch.denseProjection())
	if err != nil {
		return err
	}
	kvRank := hd
	if full {
		if hd > v41KVLoraRank {
			return v41StageErr(v41StageAttention, l,
				fmt.Errorf("%w: attention head_dim %d exceeds full KV latent rank %d", ErrV41ForwardStage, hd, v41KVLoraRank))
		}
		kvRank = v41KVLoraRank
	}
	kvPanel, err := m.v41ProjPanelWithProjection(l, "attn.wkv.weight", preFlat, kvRank, H, seq, scratch.denseProjection())
	if err != nil {
		return err
	}
	for t := 0; t < seq; t++ {
		q := qPanel[t*nH*hd : (t+1)*nH*hd]
		qLat := qLatPanel[t*cfg.QLoraRank : (t+1)*cfg.QLoraRank]
		// KV latent seam. The full path admits and projects attn.wkv at the
		// published latent rank (v41KVLoraRank = 512); the attention contraction
		// below consumes a per-position row of width hd (head_dim), so the full
		// projection is taken at the published latent rank and sliced to the first
		// hd columns for the sink contraction. On a valid full config
		// hd == v41KVLoraRank == 512, so the slice is a no-op; an hd wider than the
		// latent rank is refused rather than silently mis-read. The reduced fixture
		// keeps its HeadDim-wide projection byte-for-byte.
		kv := kvPanel[t*kvRank : t*kvRank+hd]
		// Reference: kv = self.kv_norm(self.wkv(x)) then RoPE on the rope tail
		// (_window_kv). The RMSNorm over the KV vector keeps its magnitude O(1);
		// omitting it lets an unbounded projection reach the 512-term attention
		// accumulation and overflow it to +Inf at HeadDim=512 (#13290). Applied to
		// the full head_dim before the rope tail; artifact-only, so the reduced
		// fixture's pre-#13009 arithmetic is unchanged.
		if full {
			if err := m.v41KVNormInPlaceWithPublish(l, kv, eps, scratch.kvNorm, scratch.gatherOps.bf16PublishFunc()); err != nil {
				return err
			}
		}
		// #13325 kv_latent: the KV row after the optional kv RMSNorm and BEFORE
		// RoPE, so a rotary or geometry fault is distinguishable from a projection
		// fault. Captured here because the RoPE just below rotates `kv` in place.
		trace.record(l, t, v41TraceStageKVLatent, kv)
		cos, sin := v41RopeTableForLayer(cfg, l, t)
		q, kv, err = v41AttentionQKRoPEWithPublish(cfg, l, q, kv, cos, sin, nH, hd, ropeDim, scratch.tailRoPE, scratch.gatherOps.bf16PublishFunc())
		if err != nil {
			return err
		}
		qHeads[t] = q
		kvRows[t] = kv
		qLatRows[t] = qLat
	}

	// ---- CED/CSA2 compressor + lightning indexer stages (#13006, #12896) ----
	//
	// A compressed producer pools its pre-attention inputs through the compressor;
	// shared readers consume that publication without private compression. A layer
	// declaring an in-range index source scores its projected index query against
	// those compressed keys and selects rows (v41IndexRows). Both stages execute
	// here and fail closed with a typed *V41ForwardError on malformed geometry; a
	// config that declares an in-range stage without its weights is refused at
	// admission, not here.
	//
	// #12896 maps the sink contraction onto the reference's compressed KV cache:
	// a compressed layer contracts the COMPRESSED stream directly (block-causal
	// visibility, one pooled key/value row per group) instead of expanding the
	// pooled latent back onto causal positions. A declared shared-KV source layer
	// publishes its compressed rows into the session state, and a later reader
	// layer resolves its KV stream from that published source.
	plan, err := v41AttentionPlanFor(cfg, l, m.v41AttentionRolesCached())
	if err != nil {
		return err
	}
	ownsCompressed := v41OwnsCompressedRows(plan)
	var compressedKV [][]float32
	if ownsCompressed {
		compressed, err := m.v41ColdCompressedRows(l, plan.Ratio, kvRows, preByPos, full, st, scratch)
		if err != nil {
			return err
		}
		compressedKV = compressed
	}
	sharedKV := compressedKV
	var indexKeys [][]float32
	var reusedKeys []bool
	var keyMemo []v41IndexKeyMemoRow
	if plan.Role == V41AttentionRoleReader {
		if st == nil {
			return v41StageErr(v41StageAttention, l, fmt.Errorf("%w: shared reader requires source state", ErrV41ForwardStage))
		}
		registry, err := st.attentionState(hd, 8, cfg.IndexHeadDim)
		if err != nil {
			return v41StageErr(v41StageAttention, l, err)
		}
		sharedKV, indexKeys, err = m.v41CompressedReaderRows(plan, registry, seq)
		if err != nil {
			return err
		}
	} else if ownsCompressed && indexSourceAt(cfg.DeepSeekV41, l) {
		indexKeys, reusedKeys, keyMemo, err = m.v41IndexKeysReusing(l, compressedKV, st, scratch)
		if err != nil {
			return err
		}
	}
	// Both publications derive from the same RoPE-free normalized latent.
	// Rotate only these fresh producer rows, never borrowed source rows; a key
	// reused from the previous fold is already rotated.
	for group, latent := range compressedKV {
		var key []float32
		if indexKeys != nil && (reusedKeys == nil || !reusedKeys[group]) {
			key = indexKeys[group]
		}
		if err := m.v41CompressedPublicationRoPEWithDevice(l, group*plan.Ratio, latent, key, scratch.gatherOps.compressedPublicationRoPEFunc()); err != nil {
			return err
		}
	}
	if keyMemo != nil && st != nil && st.indexKeyMemo != nil {
		for group := range keyMemo {
			if !reusedKeys[group] {
				keyMemo[group].key = append([]float32(nil), indexKeys[group]...)
			}
		}
		st.indexKeyMemo[l] = keyMemo
	}
	var indexList []int32
	if plan.Ratio > 0 && indexSourceAt(cfg.DeepSeekV41, l) {
		// Projection, normalization, rotation and memo copies have completed.
		// Scalar and canonically bound native scoring read causal prefixes of
		// one stable snapshot. Ordinary callbacks keep fresh mutable inputs.
		// Revocation copies only the current prefix, then ends panel reuse.
		var keyPanel []float32
		binding := scratch.indexerScoreBinding
		borrow := binding.prepareBorrow(m, st, scratch)
		if (binding.scoreFunc() == nil || borrow.valid()) && seq > 1 {
			keyPanel = v41ColdIndexerKeyPanel(indexKeys, cfg.IndexHeadDim)
		}
		for t := 0; t < seq; t++ {
			if binding.scoreFunc() != nil && !borrow.valid() {
				keyPanel = nil
			}
			groups := min((t+1)/plan.kvGroupSize(cfg), len(indexKeys))
			var prefix []float32
			if keyPanel != nil {
				end := groups * cfg.IndexHeadDim
				prefix = keyPanel[:end:end]
			}
			localIdx, err := m.v41IndexRowsWithPreparedBinding(l, t, qLatRows[t], preByPos[t], indexKeys[:groups], scratch.denseProjection(), binding, scratch.indexScoreHealth, scratch.gatherOps, prefix, &borrow)
			if borrow.revoked {
				keyPanel = nil
			}
			if err != nil {
				return err
			}
			row, err := m.v41AttentionIndexList(plan, st, localIdx, hd, groups, 1)
			if err != nil {
				return err
			}
			indexList = append(indexList, row...)
		}
	} else {
		localIdx, err := m.v41IndexRowsWithDeviceOps(l, seq-1, qLatRows[seq-1], preByPos[seq-1], indexKeys, scratch.denseProjection(), scratch.indexerScore(), scratch.indexScoreHealth, scratch.gatherOps)
		if err != nil {
			return err
		}
		indexList, err = m.v41AttentionIndexList(plan, st, localIdx, hd, len(sharedKV), seq)
		if err != nil {
			return err
		}
	}

	// V41AttentionState is the session-owned validation anchor for the projected
	// KV window. It runs no projection and never changes the arithmetic; it fails
	// closed on malformed geometry before the sink contraction reads the rows.
	//
	// Each layer gets independent temporal ownership. Completed source rows stay
	// in the per-forward registry; retained session state contains only the
	// configured window tail and incomplete compressor group.
	if st != nil {
		layerState, err := NewV41AttentionState(hd, 8)
		if err != nil {
			return v41StageErr(v41StageAttention, l, err)
		}
		if cfg.IndexHeadDim > 0 {
			layerState.indexHeadDim = cfg.IndexHeadDim
		}
		// Seed the retained temporal state. For a compressed source layer the
		// incomplete trailing group's PRE-ATTENTION carriers are passed through so
		// the incremental source append (#13480) can continue a mid-group prefix
		// without replaying the projected latents; a non-compressed layer passes
		// none and keeps the historical seed byte-for-byte.
		var seedInputs [][]float32
		seedRatio := 0
		if ownsCompressed {
			seedRatio = plan.Ratio
			if partial := len(preByPos) % plan.Ratio; partial > 0 {
				seedInputs = preByPos[len(preByPos)-partial:]
			}
		}
		if err := layerState.seedTemporalWithInputs(kvRows, seedInputs, seedRatio, cfg.windowForLayer(l)); err != nil {
			return v41StageErr(v41StageAttention, l, err)
		}
		if ownsCompressed {
			ownPlan := plan
			ownPlan.KVSourceLayer = l
			updates := m.v41AttentionSourceUpdates(ownPlan, compressedKV, qLatRows, indexKeys)
			if err := layerState.publishUpdates(updates); err != nil {
				return v41StageErr(v41StageAttention, l, err)
			}
		}
		registry, err := st.attentionState(hd, 8, cfg.IndexHeadDim)
		if err != nil {
			return v41StageErr(v41StageAttention, l, err)
		}
		// A declared source layer publishes its compressed rows and index keys so
		// later readers can resolve them within this same forward pass (the
		// V41AttentionState source-then-consumer ordering).
		updates := m.v41AttentionSourceUpdates(plan, compressedKV, qLatRows, indexKeys)
		if len(updates) > 0 {
			if err := registry.publishUpdates(updates); err != nil {
				return v41StageErr(v41StageAttention, l, err)
			}
		}
		// No owner becomes visible until both its rows and the registry rows
		// passed validation. Registry publication above has no partial-error path.
		st.setLayerState(l, cfg.NumLayers, layerState)
		// An index source publishes its own per-position top-k selection so a
		// later reader layer can reuse it without recomputing the scoring path.
		// The selection is the source's local index list, one row per published
		// query position.
		if indexSourceAt(cfg.DeepSeekV41, l) && indexList != nil && plan.TopKWidth > 0 {
			rows := make([][]int32, seq)
			for t := range rows {
				rows[t] = append([]int32(nil), indexList[t*plan.TopKWidth:(t+1)*plan.TopKWidth]...)
			}
			if err := registry.PublishTopK(plan.Ratio, rows); err != nil {
				return v41StageErr(v41StageIndexer, l, err)
			}
		}
	}

	// Keep the exact seeded owner that a successful forward commits for Step.
	// Decline the whole cold layer before device effects if its window or bank
	// would exceed the existing mirror's capacity or increase KV upload volume.
	coldAttentionOwner := v41ColdAttentionResidentOwner(st.layerState(l), seq, cfg.windowForLayer(l), len(sharedKV), scratch.sharedAttentionFunc())
	scale := cfg.attnScale()
	attnOut := make([][]float32, seq)
	attnProjectionInput := make([][]float32, seq)
	for t := 0; t < seq; t++ {
		// The reference concatenates this layer's window with selected source
		// groups before ONE sink contraction, even before a group completes.
		if plan.Ratio > 0 || plan.Role == V41AttentionRoleReader {
			keys := v41PlainWindowKeys(t, cfg.windowForLayer(l))
			window := make([][]float32, len(keys))
			for i, key := range keys {
				window[i] = kvRows[key]
			}
			var ids []int32
			if indexList != nil {
				width := plan.topKWidth()
				if width <= 0 || len(indexList) != seq*width {
					return v41StageErr(v41StageIndexer, l, fmt.Errorf("%w: compressed selection does not cover every query", ErrV41ForwardStage))
				}
				ids = indexList[t*width : (t+1)*width]
			} else {
				ids = make([]int32, len(sharedKV))
				for group := range ids {
					ids[group] = int32(group)
				}
			}
			attentionOpened := m.v41NowNanos()
			o, err := v41CombinedAttentionResident(l, t, plan.kvGroupSize(cfg), nH, hd, qHeads[t], sink, window, sharedKV, ids, scale,
				v41ColdAttentionResidentView(coldAttentionOwner, l, t, len(window), len(sharedKV)), scratch.sharedAttentionFunc())
			m.v41NoteAttentionContraction(attentionOpened)
			if err != nil {
				return v41StageErr(v41StageAttention, l, err)
			}
			o, err = v41RotatedOutputForProjection(cfg, l, t, o, nH, hd, scratch.gatherOps)
			if err != nil {
				return err
			}
			attnProjectionInput[t] = o
			continue
		}
		// #13303: the plain layer's visible keys are its CONFIGURED causal
		// window, not the unconditioned full prefix. A positive
		// Config.Window[l] restricts the row set to the trailing w causal keys
		// max(0,t-w+1)..t; the -1 sentinel (and a nil/short Window, which
		// windowForLayer defaults to -1) keeps the historical full-causal
		// prefix 0..t byte-for-byte. The window changes only WHICH ordered rows
		// are contracted; the score/softmax/value math is untouched.
		window := cfg.windowForLayer(l)
		keys := v41PlainWindowKeys(t, window)
		idx := v41PlainWindowIndexList(keys)
		rows := len(keys)
		opt := V41SparseAttentionSinkOptions{
			B: 1, M: 1, Heads: nH, HeadDim: hd, TopK: rows + 1, N: rows, Softmax: scale,
		}
		flatKV, borrowedWindow := v41ColdPlainAttentionInput(qHeads[t], sink, kvRows, keys, idx, opt, coldAttentionOwner, scratch.sharedAttentionFunc())
		attentionOpened := m.v41NowNanos()
		o, err := v41ColdPlainAttentionSink(l, qHeads[t], flatKV, sink, borrowedWindow, idx, opt,
			v41ColdAttentionResidentView(coldAttentionOwner, l, t, rows, 0), scratch.sharedAttentionFunc())
		m.v41NoteAttentionContraction(attentionOpened)
		if err != nil {
			return v41StageErr(v41StageAttention, l, err)
		}
		o, err = v41RotatedOutputForProjection(cfg, l, t, o, nH, hd, scratch.gatherOps)
		if err != nil {
			return err
		}
		attnProjectionInput[t] = o
	}

	panelHandled := false
	if seq > 1 && !(full && st != nil && st.callbackOwner != nil && st.callbackOwner.M == m) {
		v41GroupedPanelDecline(l, seq, v41GroupedPanelCallerReason(full, st != nil, st != nil && st.callbackOwner != nil, st != nil && st.callbackOwner != nil && st.callbackOwner.M == m))
	}
	if seq > 1 && full && st != nil && st.callbackOwner != nil && st.callbackOwner.M == m {
		totalIn, valid := checkedMulInt(nH, hd)
		panelN, panelValid := checkedMulInt(seq, totalIn)
		if valid && panelValid {
			flat := make([]float32, panelN)
			for t, row := range attnProjectionInput {
				if len(row) != totalIn {
					return v41StageErr(v41StageAttention, l, errV41ProjectionResult)
				}
				copy(flat[t*totalIn:(t+1)*totalIn], row)
			}
			projected, outcome, cause := st.callbackOwner.v41GroupedOutputPanel(l, flat, seq, nH, hd, cfg.OGroups, cfg.OLoraRank, H)
			switch outcome {
			case v41ProjectionHandled:
				if cause != nil || len(projected) != seq {
					if cause == nil {
						cause = errV41ProjectionResult
					}
					return v41StageErr(v41StageAttention, l, cause)
				}
				for t, row := range projected {
					if len(row) != H {
						return v41StageErr(v41StageAttention, l, errV41ProjectionResult)
					}
					attnOut[t] = row
				}
				panelHandled = true
			case v41ProjectionError:
				return v41StageErr(v41StageAttention, l, cause)
			case v41ProjectionDeclined:
			default:
				return v41StageErr(v41StageAttention, l, errV41ProjectionResult)
			}
		}
	}
	if !panelHandled {
		for t, row := range attnProjectionInput {
			projected, cause := projectOutput(row)
			if cause != nil {
				return v41StageErr(v41StageAttention, l, cause)
			}
			attnOut[t] = projected
		}
	}
	for t, row := range attnOut {
		trace.record(l, t, v41TraceStageAttnOut, row)
	}

	// ---- MoE: router + shared expert + routed experts ----
	//
	// The router projects the normalized panel and routes each position in order;
	// its picks are retained for the contraction loop below. The layer-scoped
	// expert cache is reset here so its working set is
	// exactly this layer's routed experts (#13296).
	scratch.v41LayerCacheReset()
	if scratch.expertLayerCacheBytes == 0 {
		if v41TestLayerCacheBudgetOverride > 0 {
			scratch.expertLayerCacheBytes = v41TestLayerCacheBudgetOverride
		} else {
			scratch.expertLayerCacheBytes = m.v41LayerExpertCacheBudget()
		}
	}
	perTokenPicks := make([][]routePick, seq)
	// Reuse one normalized host row for the router, every routed expert, and
	// the shared expert. Device normalization uploads and reads back only once
	// per layer/token; grouped contraction consumes this same panel.
	ffnInputs := make([][]float32, seq)
	var ffnResidual [][][]float32
	var ffnMixes []v41MHCMix
	var projectFFN func([]float32) ([]float32, error)
	if full {
		// Every attention projection has completed. Reusing scratch.mhc is now safe.
		projectFFN, err = m.v41FFNMHCProjector(l, scratch)
		if err != nil {
			return err
		}
		if resident == nil {
			ffnResidual = make([][][]float32, seq)
			ffnMixes = make([]v41MHCMix, seq)
		}
	}
	var ffnCollapsed [][]float32
	if resident != nil {
		if ffnCollapsed, err = resident.ffnInput(l, attnOut, hcIters, hcEps, eps); err != nil {
			return err
		}
	}
	for t := 0; t < seq; t++ {
		var xn []float32
		if resident != nil {
			xn, err = m.v41FFNNormWithPublish(l, ffnCollapsed[t], eps, scratch.ffnNorm, scratch.gatherOps.bf16PublishFunc())
		} else if full {
			ffnResidual[t], ffnMixes[t], xn, err = m.v41FullFFNInput(l, streams[t], hcByPos[t], attnOut[t], projectFFN, scratch)
		} else {
			xn, err = m.v41FFNNormWithPublish(l, x[t], eps, scratch.ffnNorm, scratch.gatherOps.bf16PublishFunc())
		}
		if err != nil {
			return err
		}
		if len(xn) != H {
			return v41StageErr(v41StageMoE, l, fmt.Errorf("%w: invalid normalized FFN row width", ErrV41ForwardStage))
		}
		ffnInputs[t] = xn
	}
	ffnElements, ok := checkedMulInt(seq, H)
	if !ok {
		return v41StageErr(v41StageMoE, l, fmt.Errorf("%w: FFN panel dimensions overflow", ErrV41ForwardStage))
	}
	var ffnFlat []float32
	if seq == 1 {
		ffnFlat = ffnInputs[0]
	} else {
		ffnFlat = make([]float32, ffnElements)
		for t, xn := range ffnInputs {
			copy(ffnFlat[t*H:(t+1)*H], xn)
		}
	}
	routerPanel, err := m.v41ProjPanelWithProjection(l, "ffn.gate.weight", ffnFlat, cfg.NumExperts, H, seq, scratch.denseProjection())
	if err != nil {
		return err
	}
	var panelPicks [][]routePick
	panelRouted := false
	for t := 0; t < seq; t++ {
		// Keep the first successful route/emission at its original boundary,
		// including lazy process trace initialization and observer side effects.
		// Only the remaining rows may share selected router completions.
		if t == 1 {
			panelPicks, panelRouted, err = m.v41ColdRouterPanel(l, routerPanel[cfg.NumExperts:], gateBias, seq-1, routeCfg, full, scratch)
			if err != nil {
				return v41StageErr(v41StageMoE, l, err)
			}
		}
		var picks []routePick
		if panelRouted {
			picks = panelPicks[t-1]
		} else {
			routerLogits := routerPanel[t*cfg.NumExperts : (t+1)*cfg.NumExperts]
			picks, err = v41RouteForGeometryWith(l, routerLogits, gateBias, routeCfg, full, scratch.routerTopK())
			if err != nil {
				return v41StageErr(v41StageMoE, l, err)
			}
		}
		m.emitExpertRouteAt(l, t, picks)
		perTokenPicks[t] = picks
	}

	// ---- routed-expert contraction ----
	//
	// #13304: a multi-token prefill panel is contracted EXPERT-MAJOR when its
	// routed union exceeds the layer cache. The grouped path materializes one
	// expert triple, contracts every (token, slot) row assigned to it, releases
	// it, then replays each token's weighted sum in original slot order -- so a
	// panel whose alternating routes re-read the same experts faults each distinct
	// projection once, and peak retained expert materialization stays one triple.
	// When the union fits the layer cache the two paths are equivalent (the cache
	// served the repeats either way); the single-token path keeps the historical
	// token-major stream byte-for-byte.
	var routedByToken [][]float32
	if m.v41EPRequested() {
		if !full || st == nil || st.routedPrefill == nil {
			return fmt.Errorf("%w: missing Session routed-prefill binding", ErrV41ExpertParallel)
		}
		routedByToken, err = st.routedPrefill(l, perTokenPicks, ffnInputs, scratch)
		if err != nil {
			return err
		}
		if len(routedByToken) != seq {
			return fmt.Errorf("%w: invalid distributed prefill panel", ErrV41ExpertParallel)
		}
		for _, values := range routedByToken {
			if len(values) != H {
				return fmt.Errorf("%w: invalid distributed prefill row", ErrV41ExpertParallel)
			}
		}
	} else if seq > 1 && !v41ForceTokenMajor {
		// #13511: thread the session's optional device gate/up callback into the
		// grouped (expert-major) contraction too, so a multi-token prefill offers
		// each routed row to the same device seam the token-major arm uses
		// (#13358) instead of unconditionally running the host SwiGLU. A nil
		// callback (Model.Forward, or no device backend) keeps the grouped path
		// byte-for-byte.
		routedByToken, err = m.v41ContractRoutedGroupedFresh(l, ffnInputs, perTokenPicks, scratch, cfg, st)
		if err != nil {
			return err
		}
	} else {
		routedByToken = make([][]float32, seq)
		// The multi-token token-major arm (test-forced, or a panel that the
		// grouped path did not take) contracts each pick through the row-parallel
		// twin; the seq == 1 decode fallback through this same arm keeps the
		// package-level serial contraction, because three tiny expert GEMVs per
		// pick would pay the parFor dispatch barrier with no bandwidth to reclaim.
		contract := v41SwiGLU
		if seq > 1 {
			contract = v41SwiGLUParallel
		}
		for t := 0; t < seq; t++ {
			xn := ffnInputs[t]
			routed := make([]float32, H)
			if full {
				outputs := make([][]float32, len(perTokenPicks[t]))
				var routedExpert v41RoutedExpertFunc
				var gateUp v41ExpertGateUpFunc
				var down v41ExpertDownFunc
				if st != nil {
					routedExpert, gateUp, down = st.routedExpert, st.expertGateUp, st.expertDown
				}
				for slot, pick := range perTokenPicks[t] {
					stem := "ffn.experts." + itoa(pick.expert)
					outputs[slot], err = m.v41FullRoutedExpertPublished(l, stem, xn, pick.weight, cfg, routedExpert, gateUp, down,
						func() ([]float32, []float32, []float32, error) { return m.v41ExpertTripleInto(l, stem, scratch, true) },
						func() ([]float32, error) { return m.hostExpertDown(l, stem, scratch) }, seq > 1, scratch.gatherOps.bf16PublishFunc())
					if err != nil {
						return err
					}
				}
				routedByToken[t], err = v41FullRoutedSumWith(l, perTokenPicks[t], outputs, H, scratch.routedSumFunc())
				if err != nil {
					return err
				}
				continue
			}
			for _, pick := range perTokenPicks[t] {
				stem := "ffn.experts." + itoa(pick.expert)
				// #13358: offer the pick to the session's optional device gate/up
				// seam first. A handled result returns the I-wide fused intermediate
				// from the backend, so the existing host down contraction below runs
				// over it WITHOUT ever materializing the gate/up f32 weights. A
				// decline (or no callback at all) keeps the historical host triple
				// byte-for-byte. A selected device failure must surface.
				if st != nil && st.expertGateUp != nil {
					h, outcome, gerr := st.expertGateUp(l, stem, xn)
					switch outcome {
					case v41GateUpError:
						return v41StageErr(v41StageMoE, l, gerr)
					case v41GateUpHandled:
						// #13704: offer the I-wide intermediate to the device down seam.
						// When it handles, all three projections ran on the backend and no
						// host expert GEMM remains; a decline keeps the historical host f32
						// down contraction byte-for-byte.
						if st.expertDown != nil {
							yd, dOutcome, derr := st.expertDown(l, stem, h)
							switch dOutcome {
							case v41DownError:
								return v41StageErr(v41StageMoE, l, derr)
							case v41DownHandled:
								contractOpen := m.v41NowNanos()
								if contractOpen != 0 {
									m.v41NoteExpertContractionNanos(m.v41NowNanos() - contractOpen)
								} else {
									m.v41NoteExpertContraction()
								}
								_ = ffn.AddScaled(routed, yd, pick.weight)
								continue
							}
						}
						w2, err := m.hostExpertDown(l, stem, scratch)
						if err != nil {
							return err
						}
						contractOpen := m.v41NowNanos()
						y := matRows(w2, h, H, cfg.MoEIntermediateSize)
						if contractOpen != 0 {
							m.v41NoteExpertContractionNanos(m.v41NowNanos() - contractOpen)
						} else {
							m.v41NoteExpertContraction()
						}
						_ = ffn.AddScaled(routed, y, pick.weight)
						continue
					}
				}
				w1, w3, w2, err := m.v41ExpertTripleInto(l, stem, scratch, true)
				if err != nil {
					return err
				}
				// #13299: time the routed-expert contraction (the SwiGLU the pick
				// applies) separately from the fault/dequant that produced its
				// weights, so the ledger can attribute the 492 s first token
				// (fak#13294) to scalar contraction vs tier IO. Inert with no clock.
				contractOpen := m.v41NowNanos()
				y := contract(w1, w3, w2, xn, cfg.MoEIntermediateSize, H, cfg)
				if contractOpen != 0 {
					m.v41NoteExpertContractionNanos(m.v41NowNanos() - contractOpen)
				} else {
					m.v41NoteExpertContraction()
				}
				_ = ffn.AddScaled(routed, y, pick.weight)
			}
			routedByToken[t] = routed
		}
	}

	var nextPanel [][][]float32
	if full && resident == nil {
		nextPanel = make([][][]float32, seq)
	}
	var moeRows [][]float32
	if resident != nil {
		moeRows = make([][]float32, seq)
	}
	var sharedChain v41ColdSharedResidentFunc
	var sharedPanel []float32
	var sharedPanelSelected bool
	var sharedPanelFault *v41SharedPanelDeferredFault
	if seq > 1 && full && st != nil && st.callbackOwner != nil && st.callbackOwner.M == m &&
		scratch.denseProjection() != nil && scratch.sharedActivation != nil && scratch.gatherOps.bf16PublishFunc() != nil {
		sharedChain = st.callbackOwner.v41ColdSharedExpertChain(l)
		sharedPanel, sharedPanelSelected, err = st.callbackOwner.v41ColdSharedExpertPanel(l)(ffnFlat, seq)
		if err != nil {
			if !errors.As(err, &sharedPanelFault) {
				return err
			}
			err = nil
		}
	}
	for t := 0; t < seq; t++ {
		if sharedPanelFault != nil && t == sharedPanelFault.Row {
			return sharedPanelFault.finalize()
		}
		xn := ffnInputs[t]
		routed := routedByToken[t]
		var shared []float32
		var sharedResident bool
		if sharedPanelSelected {
			shared = sharedPanel[t*H : (t+1)*H]
			sharedResident = true
		} else if sharedChain != nil {
			shared, sharedResident, err = sharedChain(xn)
		}
		if err == nil && !sharedResident {
			shared, err = m.v41SharedExpertSwiGLUWithPublish(l, xn, cfg, scratch.denseProjection(), scratch.sharedActivation, scratch.gatherOps.bf16PublishFunc())
		}
		if err != nil {
			return err
		}
		moe, err := v41SharedExpertAddWith(l, routed, shared, routeCfg, scratch.sharedAdd)
		if err != nil {
			return v41StageErr(v41StageMoE, l, err)
		}
		// #13325 moe_sum: the summed routed + shared expert output, captured BEFORE
		// the delta/residual mixing so an expert-contraction fault is distinguishable
		// from an mHC post-mix fault.
		trace.record(l, t, v41TraceStageMoESum, moe)

		if resident != nil {
			moeRows[t] = moe
		} else if full {
			nextPanel[t], err = v41MHCPostBF16On(mhcDevice, l, moe, ffnResidual[t], ffnMixes[t])
			if err != nil {
				return err
			}
		} else {
			delta := make([]float32, H)
			for i := range delta {
				delta[i] = attnOut[t][i] + moe[i]
			}
			residual := [][]float32{preByPos[t], preByPos[t], preByPos[t], preByPos[t]}
			next, err := v41MHCPost(delta, residual, hcByPos[t].post, hcByPos[t].comb)
			if err != nil {
				return v41StageErr(v41StageMHC, l, err)
			}
			copy(x[t], next[0])
		}
	}
	if resident != nil {
		// Every token row succeeded: publish the pending panel and its carry.
		if err := resident.finishLayer(l, moeRows); err != nil {
			return err
		}
		if err := resident.publish(l); err != nil {
			return err
		}
		scratch.mhcCarry.nextLayer++
		return nil
	}
	if full {
		// All token rows succeeded: stream and carry publication has no failing work.
		for t := range nextPanel {
			for h := 0; h < 4; h++ {
				copy(streams[t][h], nextPanel[t][h])
			}
			copy(x[t], nextPanel[t][0])
			scratch.mhcCarry.pre[t] = ffnMixes[t].pre
		}
		scratch.mhcCarry.nextLayer++

	}
	return nil
}

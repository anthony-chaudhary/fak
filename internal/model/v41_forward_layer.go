package model

import "fmt"

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
	scratch.clearLayerSessionBindings()
	if st != nil {
		scratch.bindLayerSessionBindings(st)
	}
	defer scratch.clearLayerSessionBindings()

	if err := m.v41LayerEngramInject(l, tokens, x, streams, full, eps, st, scratch, resident); err != nil {
		return err
	}

	attnNorm := m.tensor(layerName(l, "attn_norm.weight"))
	mixBase := m.tensor(layerName(l, "mhc.base"))
	mixScale := m.tensor(layerName(l, "mhc.scale"))
	projectOutput := m.v41GroupedOutputProjector(l, nH, hd, cfg.OGroups, cfg.OLoraRank, H, scratch)
	if err := m.v41LayerRequireProjectionWeights(l); err != nil {
		return err
	}
	// attn.sink is a 1-D per-head vector (not an isQuantWeight matmul leaf), so it
	// stays on the f32 manifest.
	sink := m.tensor(layerName(l, "attn.sink"))
	gateBias := m.tensor(layerName(l, "ffn.gate.e_score_correction_bias"))

	seq := len(x)
	hcByPos, preByPos, err := m.v41LayerMHCPre(l, x, streams, full, H, eps, hcIters, hcEps, attnNorm, mixBase, mixScale, mhcDevice, resident, st, scratch)
	if err != nil {
		return err
	}
	qkv, err := m.v41LayerProjectQKV(l, seq, preByPos, full, hd, nH, H, eps, scratch, trace)
	if err != nil {
		return err
	}
	src, err := m.v41LayerAttentionSources(l, seq, hd, full, qkv.kvRows, preByPos, st, scratch)
	if err != nil {
		return err
	}
	indexList, err := m.v41LayerIndexList(l, seq, hd, src, qkv.qLatRows, preByPos, st, scratch)
	if err != nil {
		return err
	}
	if st != nil {
		if err := m.v41LayerPublishAttentionState(l, seq, hd, src, qkv, preByPos, indexList, st); err != nil {
			return err
		}
	}
	attnProjectionInput, err := m.v41LayerAttentionContract(l, seq, nH, hd, src, qkv, indexList, sink, st, scratch)
	if err != nil {
		return err
	}
	attnOut, err := m.v41LayerProjectAttentionOutput(l, seq, full, nH, hd, H, attnProjectionInput, projectOutput, st)
	if err != nil {
		return err
	}
	for t, row := range attnOut {
		trace.record(l, t, v41TraceStageAttnOut, row)
	}

	return m.v41LayerMoE(v41LayerMoEInput{
		l: l, x: x, streams: streams, full: full, H: H, eps: eps, hcIters: hcIters, hcEps: hcEps,
		routeCfg: routeCfg, st: st, scratch: scratch, trace: trace, mhcDevice: mhcDevice, resident: resident,
		gateBias: gateBias, hcByPos: hcByPos, preByPos: preByPos, attnOut: attnOut,
	})
}

// clearLayerSessionBindings drops every per-layer session callback so a
// layer never observes a binding from a different forward.
func (scratch *v41ProjScratch) clearLayerSessionBindings() {
	scratch.projectionRoute = nil
	scratch.groupedOutput = nil
	scratch.mhcProjection = nil
	scratch.mhcFFNProjection = nil
	scratch.attentionInputPanel = nil
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
}

func (scratch *v41ProjScratch) bindLayerSessionBindings(st *v41ForwardState) {
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

// v41LayerEngramInject runs a declared Engram layer's injection at the START of
// the layer, into the residual, before attention and before attn_norm
// (ds41_graph_before_attention).
func (m *Model) v41LayerEngramInject(l int, tokens []int, x [][]float32, streams [][][]float32, full bool, eps float32, st *v41ForwardState, scratch *v41ProjScratch, resident *v41MHCResident) error {
	cfg := m.Cfg
	if cfg.DeepSeekV41 == nil {
		return nil
	}
	for _, eng := range cfg.DeepSeekV41.EngramLayerIDs {
		if eng != l {
			continue
		}
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
					return nil
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
			return resident.setHostStreams(l, host)
		}
		return m.v41EngramInjectWithProjection(l, x, streams, full, tokens, eps, project)
	}
	return nil
}

// v41LayerRequireProjectionWeights fails closed with the typed #13276 refusal
// before any layer arithmetic when a projection leaf is absent from every store.
// These projections are applied only through matRows, so they read through the
// streaming-safe v41ProjMatRows at their use site instead of materializing a
// whole f32 block (#2150).
func (m *Model) v41LayerRequireProjectionWeights(l int) error {
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

// v41LayerMHCPre computes the per-position mHC mix and the collapsed,
// normalized attention input (one coefficient split per layer).
//
// The mHC mix projection geometry is per-layer, so it is resolved once. The
// published artifact's hc_attn_fn is the flattened four-stream projection
// (logical 24 x 4H, stored [4H, 24]); the reference (inference/model.py mHC;
// v4_flash_oracle_test.go:168) computes it over the four width-H streams laid
// end to end with a single shared flatten-RMS. The reduced fixture uses the
// legacy 24 x H single-stream matmul; only its arithmetic is held
// byte-identical to pre-#13009.
func (m *Model) v41LayerMHCPre(l int, x [][]float32, streams [][][]float32, full bool, H int, eps float32, hcIters int, hcEps float32, attnNorm, mixBase, mixScale []float32, mhcDevice *v41MHCDevice, resident *v41MHCResident, st *v41ForwardState, scratch *v41ProjScratch) ([]v41MHCMix, [][]float32, error) {
	cfg := m.Cfg
	scratch.attentionInputPanel = nil
	seq := len(x)
	mhcFlat, mhcTransposed, mhcOK := m.v41MHCWeightLayout(l)
	if !mhcOK {
		return nil, nil, v41StageErr(v41StageMHC, l,
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
			return nil, nil, err
		}
		normalized, handled, err := scratch.attentionNormBinding.coldPanel(m, st, scratch, resident, l, collapsedRows, eps)
		if err != nil {
			return nil, nil, err
		}
		if handled {
			preByPos = normalized
		} else {
			for t := range collapsedRows {
				if preByPos[t], err = m.v41AttentionInputNormWithPublications(l, collapsedRows[t], eps, scratch.attentionNormFunc(), scratch.attentionBF16Func()); err != nil {
					return nil, nil, err
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
				return nil, nil, v41StageErr(v41StageMHC, l, errV41ProjectionResult)
			}
			input = make([]float32, 0, width)
			for _, stream := range streams[t] {
				if len(stream) != H {
					return nil, nil, v41StageErr(v41StageMHC, l, errV41ProjectionResult)
				}
				input = append(input, stream...)
			}
		}
		mixes, err := projectMHC(input)
		if err != nil {
			return nil, nil, err
		}
		mix, err := v41MHCSplitOn(mhcDevice, l, mixes, mixScale, mixBase, 4, hcIters, hcEps)
		if err != nil {
			return nil, nil, err
		}
		hcByPos[t] = mix
		// The reduced path reconstructs the reference's stand-in (four identical
		// copies of the normalized input, byte-identical to pre-#13009) rather than
		// reading the persistent set; the full path reads the four DISTINCT
		// persistent streams (stream 0 = live hidden, 1..3 = residual).
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
				return nil, nil, err
			}
		} else {
			collapsed, err = v41MHCPre(streams4, pre)
			if err != nil {
				return nil, nil, v41StageErr(v41StageMHC, l, err)
			}
		}
		if full {
			collapsed, err = m.v41AttentionInputNormWithPublications(l, collapsed, eps, scratch.attentionNormFunc(), scratch.attentionBF16Func())
			if err != nil {
				return nil, nil, err
			}
		}
		preByPos[t] = collapsed
	}
	return hcByPos, preByPos, nil
}

// v41LayerQKV is the per-position projected attention input: rotated query
// heads [t][nH*hd], rotated single-head KV rows [t][hd], and q-lora latents.
type v41LayerQKV struct {
	qHeads   [][]float32
	kvRows   [][]float32
	qLatRows [][]float32
}

// v41LayerProjectQKV projects q/kv per position across the prefill panel.
//
// residentMatMulBatch reads each attn.wq_a / attn.wq_b weight row ONCE and
// reuses it across all seq prompt tokens instead of re-streaming every weight
// seq times (#13301). The per-token q_norm and RoPE glue stays scalar, so this
// is byte-for-byte the per-token v41ProjMatRows path; seq==1 (decode) stays
// entirely on that scalar path.
func (m *Model) v41LayerProjectQKV(l, seq int, preByPos [][]float32, full bool, hd, nH, H int, eps float32, scratch *v41ProjScratch, trace *v41ActivationTraceState) (v41LayerQKV, error) {
	defer func() { scratch.attentionInputPanel = nil }()
	cfg := m.Cfg
	out := v41LayerQKV{
		qHeads:   make([][]float32, seq),
		kvRows:   make([][]float32, seq),
		qLatRows: make([][]float32, seq),
	}
	// The reference rotates only the LAST qk_rope_head_dim components of each
	// head using the interleaved adjacent-pair convention. A head width that
	// cannot carry that slice is refused rather than silently rotating a wrong
	// sub-vector.
	ropeDim := cfg.QKRopeHeadDim
	if ropeDim <= 0 || ropeDim%2 != 0 || ropeDim > hd {
		return out, v41StageErr(v41StageAttention, l,
			fmt.Errorf("%w: attention qk_rope_head_dim must be a positive even value <= head_dim %d, got %d", ErrV41ForwardStage, hd, ropeDim))
	}
	// The old joined panel is a snapshot reused after query-normalization and
	// wq_b callbacks. Borrow only the private canonical producer's chunks and
	// only when that entire interval cannot expose or mutate their host rows.
	loan := scratch.attentionInputPanel
	if !full || trace != nil || seq != len(preByPos) || !loan.valid(scratch) || !loan.matches(l, H, preByPos) {
		loan = nil
	}
	var preFlat []float32
	if loan == nil {
		preFlat = make([]float32, seq*H)
		for t := 0; t < seq; t++ {
			copy(preFlat[t*H:(t+1)*H], preByPos[t])
		}
	}
	qLatPanel, err := m.v41ProjPanelWithInput(l, "attn.wq_a.weight", preFlat, cfg.QLoraRank, H, seq, scratch.denseProjection(), loan, scratch)
	if err != nil {
		return out, err
	}
	queryHandled := false
	if full {
		queryHandled, err = scratch.queryNorm.panel(m, scratch, l, qLatPanel, seq, eps, func(row int, values []float32) {
			trace.record(l, row, v41TraceStageQLatent, values)
		})
		if err != nil {
			return out, err
		}
	}
	if !queryHandled {
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
				if err := m.v41QueryNormInPlaceWithPublish(l, qLat, eps, scratch.queryNorm.normFunc(), scratch.gatherOps.bf16PublishFunc()); err != nil {
					return out, err
				}
			}
			// #13325 q_latent: captured BEFORE the wq_b panel consumes it.
			trace.record(l, t, v41TraceStageQLatent, qLat)
		}
	}
	qPanel, err := m.v41ProjPanelWithProjection(l, "attn.wq_b.weight", qLatPanel, nH*hd, cfg.QLoraRank, seq, scratch.denseProjection())
	if err != nil {
		return out, err
	}
	kvRank := hd
	if full {
		if hd > v41KVLoraRank {
			return out, v41StageErr(v41StageAttention, l,
				fmt.Errorf("%w: attention head_dim %d exceeds full KV latent rank %d", ErrV41ForwardStage, hd, v41KVLoraRank))
		}
		kvRank = v41KVLoraRank
	}
	kvPanel, err := m.v41ProjPanelWithInput(l, "attn.wkv.weight", preFlat, kvRank, H, seq, scratch.denseProjection(), loan, scratch)
	if err != nil {
		return out, err
	}
	for t := 0; t < seq; t++ {
		q := qPanel[t*nH*hd : (t+1)*nH*hd]
		qLat := qLatPanel[t*cfg.QLoraRank : (t+1)*cfg.QLoraRank]
		// KV latent seam. The full path projects attn.wkv at the published latent
		// rank (v41KVLoraRank = 512) and slices the first hd columns for the sink
		// contraction. On a valid full config hd == v41KVLoraRank == 512, so the
		// slice is a no-op; an hd wider than the latent rank is refused above. The
		// reduced fixture keeps its HeadDim-wide projection byte-for-byte.
		kv := kvPanel[t*kvRank : t*kvRank+hd]
		// Reference: kv = self.kv_norm(self.wkv(x)) then RoPE on the rope tail
		// (_window_kv). Omitting the RMSNorm lets an unbounded projection overflow
		// the 512-term attention accumulation to +Inf at HeadDim=512 (#13290).
		// Artifact-only, so the reduced fixture's pre-#13009 arithmetic is unchanged.
		if full {
			if err := m.v41KVNormWithBinding(l, kv, eps, scratch.kvNorm, scratch.gatherOps); err != nil {
				return out, err
			}
		}
		// #13325 kv_latent: captured BEFORE RoPE rotates `kv` in place, so a rotary
		// or geometry fault is distinguishable from a projection fault.
		trace.record(l, t, v41TraceStageKVLatent, kv)
		cos, sin := v41RopeTableForLayer(cfg, l, t)
		q, kv, err = v41AttentionQKRoPEWithPublish(cfg, l, q, kv, cos, sin, nH, hd, ropeDim, scratch.tailRoPE, scratch.gatherOps.bf16PublishFunc())
		if err != nil {
			return out, err
		}
		out.qHeads[t] = q
		out.kvRows[t] = kv
		out.qLatRows[t] = qLat
	}
	return out, nil
}

// v41LayerSources is a layer's resolved attention plan and its compressed
// KV / index-key publications.
type v41LayerSources struct {
	plan           V41AttentionPlan
	ownsCompressed bool
	compressedKV   [][]float32
	sharedKV       [][]float32
	indexKeys      [][]float32
}

// v41LayerAttentionSources runs the CED/CSA2 compressor and resolves the
// lightning-indexer keys (#13006, #12896).
//
// A compressed producer pools its pre-attention inputs through the compressor;
// shared readers consume that publication without private compression. A
// compressed layer contracts the COMPRESSED stream directly (block-causal
// visibility, one pooled key/value row per group) instead of expanding the
// pooled latent back onto causal positions. A declared shared-KV source layer
// publishes its compressed rows into the session state, and a later reader
// layer resolves its KV stream from that published source. Malformed geometry
// fails closed with a typed *V41ForwardError; a config that declares an
// in-range stage without its weights is refused at admission, not here.
func (m *Model) v41LayerAttentionSources(l, seq, hd int, full bool, kvRows, preByPos [][]float32, st *v41ForwardState, scratch *v41ProjScratch) (v41LayerSources, error) {
	cfg := m.Cfg
	var src v41LayerSources
	plan, err := v41AttentionPlanFor(cfg, l, m.v41AttentionRolesCached())
	if err != nil {
		return src, err
	}
	src.plan = plan
	src.ownsCompressed = v41OwnsCompressedRows(plan)
	if src.ownsCompressed {
		compressed, err := m.v41ColdCompressedRows(l, plan.Ratio, kvRows, preByPos, full, st, scratch)
		if err != nil {
			return src, err
		}
		src.compressedKV = compressed
	}
	src.sharedKV = src.compressedKV
	var reusedKeys []bool
	var keyMemo []v41IndexKeyMemoRow
	if plan.Role == V41AttentionRoleReader {
		if st == nil {
			return src, v41StageErr(v41StageAttention, l, fmt.Errorf("%w: shared reader requires source state", ErrV41ForwardStage))
		}
		registry, err := st.attentionState(hd, 8, cfg.IndexHeadDim)
		if err != nil {
			return src, v41StageErr(v41StageAttention, l, err)
		}
		src.sharedKV, src.indexKeys, err = m.v41CompressedReaderRows(plan, registry, seq)
		if err != nil {
			return src, err
		}
	} else if src.ownsCompressed && indexSourceAt(cfg.DeepSeekV41, l) {
		src.indexKeys, reusedKeys, keyMemo, err = m.v41IndexKeysReusing(l, src.compressedKV, st, scratch)
		if err != nil {
			return src, err
		}
	}
	// Both publications derive from the same RoPE-free normalized latent.
	// Rotate only these fresh producer rows, never borrowed source rows; a key
	// reused from the previous fold is already rotated.
	for group, latent := range src.compressedKV {
		var key []float32
		if src.indexKeys != nil && (reusedKeys == nil || !reusedKeys[group]) {
			key = src.indexKeys[group]
		}
		if err := m.v41CompressedPublicationRoPEWithDevice(l, group*plan.Ratio, latent, key, scratch.gatherOps.compressedPublicationRoPEFunc()); err != nil {
			return src, err
		}
	}
	if keyMemo != nil && st != nil && st.indexKeyMemo != nil {
		for group := range keyMemo {
			if !reusedKeys[group] {
				keyMemo[group].key = append([]float32(nil), src.indexKeys[group]...)
			}
		}
		st.indexKeyMemo[l] = keyMemo
	}
	return src, nil
}

// v41LayerIndexList scores each query against the compressed index keys and
// returns the flattened per-query selection.
func (m *Model) v41LayerIndexList(l, seq, hd int, src v41LayerSources, qLatRows, preByPos [][]float32, st *v41ForwardState, scratch *v41ProjScratch) ([]int32, error) {
	cfg := m.Cfg
	plan := src.plan
	indexKeys := src.indexKeys
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
			localIdx, err := m.v41IndexRowsWithPreparedProjection(l, t, qLatRows[t], preByPos[t], indexKeys[:groups], scratch.denseProjection(), binding, scratch.indexScoreHealth, scratch.gatherOps, prefix, &borrow, scratch.projectionRoute)
			if borrow.revoked {
				keyPanel = nil
			}
			if err != nil {
				return nil, err
			}
			row, err := m.v41AttentionIndexList(plan, st, localIdx, hd, groups, 1)
			if err != nil {
				return nil, err
			}
			indexList = append(indexList, row...)
		}
		return indexList, nil
	}
	localIdx, err := m.v41IndexRowsWithDeviceOps(l, seq-1, qLatRows[seq-1], preByPos[seq-1], indexKeys, scratch.denseProjection(), scratch.indexerScore(), scratch.indexScoreHealth, scratch.gatherOps)
	if err != nil {
		return nil, err
	}
	return m.v41AttentionIndexList(plan, st, localIdx, hd, len(src.sharedKV), seq)
}

// v41LayerPublishAttentionState seeds the layer's session-owned
// V41AttentionState and publishes source rows into the per-forward registry.
//
// V41AttentionState is the validation anchor for the projected KV window. It
// runs no projection and never changes the arithmetic; it fails closed on
// malformed geometry before the sink contraction reads the rows. Each layer
// gets independent temporal ownership: completed source rows stay in the
// per-forward registry; retained session state contains only the configured
// window tail and incomplete compressor group.
func (m *Model) v41LayerPublishAttentionState(l, seq, hd int, src v41LayerSources, qkv v41LayerQKV, preByPos [][]float32, indexList []int32, st *v41ForwardState) error {
	cfg := m.Cfg
	plan := src.plan
	layerState, err := NewV41AttentionState(hd, 8)
	if err != nil {
		return v41StageErr(v41StageAttention, l, err)
	}
	if cfg.IndexHeadDim > 0 {
		layerState.indexHeadDim = cfg.IndexHeadDim
	}
	// For a compressed source layer the incomplete trailing group's PRE-ATTENTION
	// carriers are passed through so the incremental source append (#13480) can
	// continue a mid-group prefix without replaying the projected latents; a
	// non-compressed layer passes none and keeps the historical seed byte-for-byte.
	var seedInputs [][]float32
	seedRatio := 0
	if src.ownsCompressed {
		seedRatio = plan.Ratio
		if partial := len(preByPos) % plan.Ratio; partial > 0 {
			seedInputs = preByPos[len(preByPos)-partial:]
		}
	}
	if err := layerState.seedTemporalWithInputs(qkv.kvRows, seedInputs, seedRatio, cfg.windowForLayer(l)); err != nil {
		return v41StageErr(v41StageAttention, l, err)
	}
	if src.ownsCompressed {
		ownPlan := plan
		ownPlan.KVSourceLayer = l
		updates := m.v41AttentionSourceUpdates(ownPlan, src.compressedKV, qkv.qLatRows, src.indexKeys)
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
	updates := m.v41AttentionSourceUpdates(plan, src.compressedKV, qkv.qLatRows, src.indexKeys)
	if len(updates) > 0 {
		if err := registry.publishUpdates(updates); err != nil {
			return v41StageErr(v41StageAttention, l, err)
		}
	}
	// No owner becomes visible until both its rows and the registry rows
	// passed validation. Registry publication above has no partial-error path.
	st.setLayerState(l, cfg.NumLayers, layerState)
	// An index source publishes its own per-position top-k selection (one row per
	// published query position) so a later reader layer can reuse it without
	// recomputing the scoring path.
	if indexSourceAt(cfg.DeepSeekV41, l) && indexList != nil && plan.TopKWidth > 0 {
		rows := make([][]int32, seq)
		for t := range rows {
			// PublishTopK synchronously copies each row into registry-owned storage.
			rows[t] = indexList[t*plan.TopKWidth : (t+1)*plan.TopKWidth : (t+1)*plan.TopKWidth]
		}
		if err := registry.PublishTopK(plan.Ratio, rows); err != nil {
			return v41StageErr(v41StageIndexer, l, err)
		}
	}
	return nil
}

// v41LayerAttentionContract runs the sparse sink contraction per position and
// returns the rotated attention output rows awaiting the output projection.
func (m *Model) v41LayerAttentionContract(l, seq, nH, hd int, src v41LayerSources, qkv v41LayerQKV, indexList []int32, sink []float32, st *v41ForwardState, scratch *v41ProjScratch) ([][]float32, error) {
	cfg := m.Cfg
	plan := src.plan
	sharedKV := src.sharedKV
	// Keep the exact seeded owner that a successful forward commits for Step.
	// Decline the whole cold layer before device effects if its window or bank
	// would exceed the existing mirror's capacity or increase KV upload volume.
	coldAttentionOwner := v41ColdAttentionResidentOwner(st.layerState(l), seq, cfg.windowForLayer(l), len(sharedKV), scratch.sharedAttentionFunc())
	scale := cfg.attnScale()
	attnProjectionInput := make([][]float32, seq)
	for t := 0; t < seq; t++ {
		// The reference concatenates this layer's window with selected source
		// groups before ONE sink contraction, even before a group completes.
		if plan.Ratio > 0 || plan.Role == V41AttentionRoleReader {
			keys := v41PlainWindowKeys(t, cfg.windowForLayer(l))
			window := make([][]float32, len(keys))
			for i, key := range keys {
				window[i] = qkv.kvRows[key]
			}
			var ids []int32
			if indexList != nil {
				width := plan.topKWidth()
				if width <= 0 || len(indexList) != seq*width {
					return nil, v41StageErr(v41StageIndexer, l, fmt.Errorf("%w: compressed selection does not cover every query", ErrV41ForwardStage))
				}
				ids = indexList[t*width : (t+1)*width]
			} else {
				ids = make([]int32, len(sharedKV))
				for group := range ids {
					ids[group] = int32(group)
				}
			}
			attentionOpened := m.v41NowNanos()
			o, err := v41CombinedAttentionResident(l, t, plan.kvGroupSize(cfg), nH, hd, qkv.qHeads[t], sink, window, sharedKV, ids, scale,
				v41ColdAttentionResidentView(coldAttentionOwner, l, t, len(window), len(sharedKV)), scratch.sharedAttentionFunc())
			m.v41NoteAttentionContraction(attentionOpened)
			if err != nil {
				return nil, v41StageErr(v41StageAttention, l, err)
			}
			o, err = v41RotatedOutputForProjection(cfg, l, t, o, nH, hd, scratch.gatherOps)
			if err != nil {
				return nil, err
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
		flatKV, borrowedWindow := v41ColdPlainAttentionInput(qkv.qHeads[t], sink, qkv.kvRows, keys, idx, opt, coldAttentionOwner, scratch.sharedAttentionFunc())
		attentionOpened := m.v41NowNanos()
		o, err := v41ColdPlainAttentionSink(l, qkv.qHeads[t], flatKV, sink, borrowedWindow, idx, opt,
			v41ColdAttentionResidentView(coldAttentionOwner, l, t, rows, 0), scratch.sharedAttentionFunc())
		m.v41NoteAttentionContraction(attentionOpened)
		if err != nil {
			return nil, v41StageErr(v41StageAttention, l, err)
		}
		o, err = v41RotatedOutputForProjection(cfg, l, t, o, nH, hd, scratch.gatherOps)
		if err != nil {
			return nil, err
		}
		attnProjectionInput[t] = o
	}
	return attnProjectionInput, nil
}

// v41LayerProjectAttentionOutput applies the grouped output projection,
// offering a multi-token full panel to the session's callback owner first.
func (m *Model) v41LayerProjectAttentionOutput(l, seq int, full bool, nH, hd, H int, attnProjectionInput [][]float32, projectOutput func([]float32) ([]float32, error), st *v41ForwardState) ([][]float32, error) {
	cfg := m.Cfg
	attnOut := make([][]float32, seq)
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
					return nil, v41StageErr(v41StageAttention, l, errV41ProjectionResult)
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
					return nil, v41StageErr(v41StageAttention, l, cause)
				}
				for t, row := range projected {
					if len(row) != H {
						return nil, v41StageErr(v41StageAttention, l, errV41ProjectionResult)
					}
					attnOut[t] = row
				}
				panelHandled = true
			case v41ProjectionError:
				return nil, v41StageErr(v41StageAttention, l, cause)
			case v41ProjectionDeclined:
			default:
				return nil, v41StageErr(v41StageAttention, l, errV41ProjectionResult)
			}
		}
	}
	if !panelHandled {
		for t, row := range attnProjectionInput {
			projected, cause := projectOutput(row)
			if cause != nil {
				return nil, v41StageErr(v41StageAttention, l, cause)
			}
			attnOut[t] = projected
		}
	}
	return attnOut, nil
}

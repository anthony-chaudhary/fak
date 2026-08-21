package model

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// BackendForwardOperationError is a fail-closed compute-HAL operation failure. The
// affected Session is closed before this value is raised; Unwrap preserves the backend's
// typed CUDA error for callers that recover at the request boundary.
type BackendForwardOperationError struct {
	Backend string
	Forward ForwardPathKind
	Path    string
	Layer   int
	Stage   string
	Cause   error
}

func (e *BackendForwardOperationError) Error() string {
	return fmt.Sprintf(
		"model: backend %q forward %q via %q failed closed at layer %d (%s): %v; session closed, no CPU retry",
		e.Backend, e.Forward, e.Path, e.Layer, e.Stage, e.Cause,
	)
}

func (e *BackendForwardOperationError) Unwrap() error { return e.Cause }

type qwen35HALState struct {
	backend Qwen35GDNBackend
	layers  []qwen35HALLayerState
}

type qwen35HALLayerState struct {
	conv      compute.Tensor
	recurrent compute.Tensor
}

type qwen35PartialRoPEBackend interface {
	PartialRoPEQK(
		q, k compute.Tensor,
		pos, nQHeads, nKHeads, headDim, rotaryDim int,
		theta float64,
	) (compute.Tensor, compute.Tensor)
}

type qwen35SigmoidGateBackend interface {
	SigmoidMulInPlace(x, gate compute.Tensor)
}

type qwen35QueryGateSplitBackend interface {
	SplitQwen35QueryGate(qg compute.Tensor, nHeads, headDim int) (compute.Tensor, compute.Tensor)
}

func (s *Session) initQwen35HALState(gdn Qwen35GDNBackend) {
	if s == nil || s.M == nil || !s.M.Cfg.IsQwen35Hybrid() {
		return
	}
	cfg := s.M.Cfg
	_, nV, kHd, vHd, _, _, convDim := cfg.linearAttnDims()
	state := &qwen35HALState{backend: gdn, layers: make([]qwen35HALLayerState, cfg.NumLayers)}
	for l := 0; l < cfg.NumLayers; l++ {
		if !cfg.isLinearAttnLayer(l) {
			continue
		}
		state.layers[l] = qwen35HALLayerState{
			conv: s.uploadHostF32(
				[]int{cfg.LinearConvKernelDim - 1, convDim},
				make([]float32, (cfg.LinearConvKernelDim-1)*convDim),
				compute.MemoryKVCache,
				"qwen35-gdn-conv-state layer "+itoa(l),
			),
			recurrent: s.uploadHostF32(
				[]int{nV, kHd, vHd},
				make([]float32, nV*kHd*vHd),
				compute.MemoryKVCache,
				"qwen35-gdn-recurrent-state layer "+itoa(l),
			),
		}
	}
	s.qwen35HAL = state
}

// cloneQwen35HALState deep-copies the recurrent Qwen3.5/3.6 device state. The
// state is semantically part of the prefix just as much as attention KV is: restoring
// only halKV yields plausible but incorrect continuations after a GDN layer.
func cloneQwen35HALState(src *qwen35HALState, backend compute.Backend) (*qwen35HALState, error) {
	if src == nil {
		return nil, nil
	}
	cloner, ok := backend.(compute.TensorCloner)
	if !ok {
		return nil, fmt.Errorf("model: backend %T cannot clone Qwen3.5 recurrent state", backend)
	}
	out := &qwen35HALState{backend: src.backend, layers: make([]qwen35HALLayerState, len(src.layers))}
	for i := range src.layers {
		if src.layers[i].conv.Buf() == nil && src.layers[i].recurrent.Buf() == nil {
			continue
		}
		var err error
		out.layers[i].conv, err = cloner.CloneTensor(src.layers[i].conv)
		if err != nil {
			out.free(backend)
			return nil, fmt.Errorf("model: clone Qwen3.5 layer %d convolution state: %w", i, err)
		}
		out.layers[i].recurrent, err = cloner.CloneTensor(src.layers[i].recurrent)
		if err != nil {
			out.free(backend)
			return nil, fmt.Errorf("model: clone Qwen3.5 layer %d recurrent state: %w", i, err)
		}
	}
	return out, nil
}

func (q *qwen35HALState) free(backend compute.Backend) {
	if q == nil || backend == nil {
		return
	}
	for i := range q.layers {
		if q.layers[i].conv.Buf() != nil {
			backend.Free(q.layers[i].conv)
		}
		if q.layers[i].recurrent.Buf() != nil {
			backend.Free(q.layers[i].recurrent)
		}
		q.layers[i] = qwen35HALLayerState{}
	}
}

func (s *Session) closeQwen35HALState() {
	if s == nil || s.qwen35HAL == nil || s.Backend == nil {
		return
	}
	s.qwen35HAL.free(s.Backend)
	s.qwen35HAL = nil
}

func (s *Session) failBackendForward(layer int, stage string, cause error) {
	if cause == nil {
		cause = fmt.Errorf("unknown backend operation failure")
	}
	err := &BackendForwardOperationError{
		Backend: s.Backend.Name(), Forward: ForwardQwen35GDN, Path: Qwen35GDNCUDAPath,
		Layer: layer, Stage: stage, Cause: cause,
	}
	s.halFailure = err
	s.Close()
	panic(err)
}

func (s *Session) ensureOpenBackendSession() {
	if s == nil || !s.halClosed {
		return
	}
	if s.halFailure != nil {
		panic(s.halFailure)
	}
	panic(fmt.Errorf("model: backend session is closed"))
}

// qwen35LinearHAL is the only linear-attention branch in the compute-HAL loop. It
// performs the ordinary block input RMSNorm, then hands every remaining token-mixer
// operand and both persistent state tensors to the exact whole-operation backend seam.
// It never resolves self_attn.q_proj, calls Backend.Attention, or reads state to host.
func (s *Session) qwen35LinearHAL(layer int, residual compute.Tensor, eps float32) {
	if s.qwen35HAL == nil || s.qwen35HAL.backend == nil {
		s.failBackendForward(layer, "admission", fmt.Errorf("validated Qwen35 GDN backend state is missing"))
	}
	cfg := s.M.Cfg
	nK, nV, kHd, vHd, _, _, _ := cfg.linearAttnDims()
	p := func(suffix string) string { return layerName(layer, suffix) }
	xn := s.Backend.RMSNorm(residual, s.normWeightHAL(p("input_layernorm.weight")), eps)
	state := &s.qwen35HAL.layers[layer]
	oldConv, oldRecurrent := state.conv, state.recurrent
	output, nextConv, nextRecurrent, err := s.qwen35HAL.backend.Qwen35GDNDecode(
		xn,
		s.matWeightHAL(p("linear_attn.in_proj_qkv.weight")),
		s.matWeightHAL(p("linear_attn.in_proj_z.weight")),
		s.matWeightHAL(p("linear_attn.in_proj_b.weight")),
		s.matWeightHAL(p("linear_attn.in_proj_a.weight")),
		s.weightHAL(p("linear_attn.conv1d.weight")),
		s.weightHAL(p("linear_attn.A_log")),
		s.weightHAL(p("linear_attn.dt_bias")),
		s.weightHAL(p("linear_attn.norm.weight")),
		s.matWeightHAL(p("linear_attn.out_proj.weight")),
		oldConv, oldRecurrent,
		nK, nV, kHd, vHd, cfg.LinearConvKernelDim, eps,
	)
	if err != nil {
		s.failBackendForward(layer, "Qwen35GDNDecode", err)
	}
	if output.Buf() == nil || !output.Ready() || nextConv.Buf() == nil || !nextConv.Ready() || nextRecurrent.Buf() == nil || !nextRecurrent.Ready() {
		s.failBackendForward(layer, "Qwen35GDNDecode result", fmt.Errorf("backend returned an absent or unready output/state tensor"))
	}
	// The production operation's mutable state contract is in-place. Enforce it here so
	// a backend cannot silently substitute transient state that Recycle will reclaim.
	if nextConv.Buf() != oldConv.Buf() || nextRecurrent.Buf() != oldRecurrent.Buf() {
		if nextConv.Buf() != nil && nextConv.Buf() != oldConv.Buf() {
			s.Backend.Free(nextConv)
		}
		if nextRecurrent.Buf() != nil && nextRecurrent.Buf() != oldRecurrent.Buf() {
			s.Backend.Free(nextRecurrent)
		}
		s.failBackendForward(layer, "Qwen35GDNDecode state identity", fmt.Errorf("backend replaced persistent in-place state"))
	}
	state.conv, state.recurrent = nextConv, nextRecurrent
	s.Backend.AddInPlace(residual, output)
}

// normWeightHAL uploads the configured (1+w) gain when this architecture requests
// NormGain1p. GDN's own gated norm is deliberately not routed here: its norm.weight is
// passed verbatim to Qwen35GDNDecode, matching the CPU/reference semantics.
func (s *Session) normWeightHAL(name string) compute.Tensor {
	if !s.M.Cfg.NormGain1p {
		return s.weightHAL(name)
	}
	key := name + "#norm-gain-1p"
	if t, ok := s.halW[key]; ok {
		return t
	}
	meta, ok := s.M.manifest[name]
	if !ok {
		panic("model: missing tensor " + name)
	}
	data := append([]float32(nil), s.M.tensor(name)...)
	for i := range data {
		data[i]++
	}
	t := s.uploadHostF32(meta.Shape, data, compute.MemoryWeights, "hal-weight "+key)
	s.halW[key] = t
	return t
}

func (s *Session) derivedWeightHAL(key string, shape []int, data []float32) compute.Tensor {
	if t, ok := s.halW[key]; ok {
		return t
	}
	t := s.uploadHostF32(shape, data, compute.MemoryWeights, "hal-weight "+key)
	s.halW[key] = t
	return t
}

func splitQwen35HeadInterleavedRows(src []float32, nHeads, headDim, rowWidth int) (query, gate []float32) {
	rows := nHeads * headDim
	query = make([]float32, rows*rowWidth)
	gate = make([]float32, rows*rowWidth)
	for h := 0; h < nHeads; h++ {
		qSrc := (h * 2 * headDim) * rowWidth
		gSrc := (h*2*headDim + headDim) * rowWidth
		dst := h * headDim * rowWidth
		copy(query[dst:dst+headDim*rowWidth], src[qSrc:qSrc+headDim*rowWidth])
		copy(gate[dst:dst+headDim*rowWidth], src[gSrc:gSrc+headDim*rowWidth])
	}
	return query, gate
}

func dequantQ8Tensor(qt *q8Tensor) []float32 {
	out := make([]float32, qt.out*qt.in)
	for row := 0; row < qt.out; row++ {
		for col := 0; col < qt.in; col++ {
			out[row*qt.in+col] = float32(qt.q[row*qt.in+col]) * qt.d[row*qt.nblk+col/qBlk]
		}
	}
	return out
}

func (s *Session) qwen35QueryWeightsHAL(layer int) (query, gate compute.Tensor) {
	cfg := s.M.Cfg
	name := layerName(layer, "self_attn.q_proj.weight")
	var src []float32
	switch {
	case s.M.has(name):
		src = s.M.tensor(name)
	case s.M.q8w[name] != nil:
		// Qwen3.6's gated q_proj is normalize-sensitive, so the GGUF loader holds the
		// already-normalized matrix in Q8 only. Split that resident copy once at session
		// setup rather than incorrectly requiring a discarded f32 manifest tensor.
		src = dequantQ8Tensor(s.M.q8w[name])
	default:
		panic("model: missing tensor " + name)
	}
	q, g := splitQwen35HeadInterleavedRows(src, cfg.NumHeads, cfg.HeadDim, cfg.HiddenSize)
	rows := cfg.NumHeads * cfg.HeadDim
	return s.derivedWeightHAL(name+"#query", []int{rows, cfg.HiddenSize}, q),
		s.derivedWeightHAL(name+"#gate", []int{rows, cfg.HiddenSize}, g)
}

func (s *Session) qwen35QueryBiasHAL(layer int) (query, gate compute.Tensor) {
	cfg := s.M.Cfg
	name := layerName(layer, "self_attn.q_proj.bias")
	q, g := splitQwen35HeadInterleavedRows(s.M.tensor(name), cfg.NumHeads, cfg.HeadDim, 1)
	rows := cfg.NumHeads * cfg.HeadDim
	return s.derivedWeightHAL(name+"#query", []int{rows}, q),
		s.derivedWeightHAL(name+"#gate", []int{rows}, g)
}

func (s *Session) readQwen35FullAttention(layer int, label string, tensor compute.Tensor) []float32 {
	data := s.Backend.Read(tensor)
	if data == nil {
		s.failBackendForward(layer, label, fmt.Errorf("backend returned an unreadable tensor"))
	}
	return append([]float32(nil), data...)
}

func qwen35HALKVLayer(cfg Config, layer int) int {
	if !cfg.IsQwen35Hybrid() {
		return layer
	}
	n := 0
	for l := 0; l < layer; l++ {
		if !cfg.isLinearAttnLayer(l) {
			n++
		}
	}
	return n
}

// qwen35FullAttentionHAL retains the hybrid's ordinary full-attention layers. The
// Optional device capabilities keep split-query partial RoPE and output gating resident.
// Backends without those capabilities retain the explicit host correctness fallback. The
// linear-attention branch above never reaches this function and remains wholly resident.
func (s *Session) qwen35FullAttentionHAL(layer, pos int, residual compute.Tensor, eps, scale float32, grp int) {
	be, cfg := s.Backend, s.M.Cfg
	hd, nH, nKV := cfg.HeadDim, cfg.NumHeads, cfg.NumKVHeads
	p := func(suffix string) string { return layerName(layer, suffix) }
	xn := be.RMSNorm(residual, s.normWeightHAL(p("input_layernorm.weight")), eps)

	var q, gate compute.Tensor
	if cfg.AttnOutputGate {
		if splitter, ok := be.(qwen35QueryGateSplitBackend); ok {
			qg := be.MatMul(s.matWeightHAL(p("self_attn.q_proj.weight")), xn)
			if cfg.AttentionBias {
				be.AddBias(qg, s.weightHAL(p("self_attn.q_proj.bias")))
			}
			q, gate = splitter.SplitQwen35QueryGate(qg, nH, hd)
		} else {
			qWeight, gateWeight := s.qwen35QueryWeightsHAL(layer)
			q, gate = be.MatMul(qWeight, xn), be.MatMul(gateWeight, xn)
			if cfg.AttentionBias {
				qBias, gateBias := s.qwen35QueryBiasHAL(layer)
				be.AddBias(q, qBias)
				be.AddBias(gate, gateBias)
			}
		}
	} else {
		q = be.MatMul(s.matWeightHAL(p("self_attn.q_proj.weight")), xn)
		if cfg.AttentionBias {
			be.AddBias(q, s.weightHAL(p("self_attn.q_proj.bias")))
		}
	}
	kRaw := be.MatMul(s.matWeightHAL(p("self_attn.k_proj.weight")), xn)
	v := be.MatMul(s.matWeightHAL(p("self_attn.v_proj.weight")), xn)
	if cfg.AttentionBias {
		be.AddBias(kRaw, s.weightHAL(p("self_attn.k_proj.bias")))
		be.AddBias(v, s.weightHAL(p("self_attn.v_proj.bias")))
	}

	kvLayer := qwen35HALKVLayer(cfg, layer)
	theta := cfg.ropeThetaForLayer(layer)
	if cfg.rotaryDim() != hd {
		// The device capability takes theta directly, so use it only for the unscaled
		// Qwen path. Scaled/YaRN configurations retain the exact cached-inv-freq fallback.
		if partial, ok := be.(qwen35PartialRoPEBackend); ok && cfg.RopeScaling == "" && cfg.LongRope == nil {
			var kRope compute.Tensor
			q, kRope = partial.PartialRoPEQK(q, kRaw, pos, nH, nKV, hd, cfg.rotaryDim(), theta)
			s.halKV.AppendKV(kvLayer, kRaw, kRope, v, pos)
		} else {
			qHost := s.readQwen35FullAttention(layer, "partial-RoPE query read", q)
			kHost := s.readQwen35FullAttention(layer, "partial-RoPE key read", kRaw)
			cos, sin := ropeRowForLayer(cfg, layer, pos)
			ropeRowQKInto(qHost, kHost, cos, sin, hd, nH, nKV)
			q = s.uploadHostF32([]int{nH * hd}, qHost, compute.MemoryActivation, "qwen35-full-attn-rope-q")
			kRope := s.uploadHostF32([]int{nKV * hd}, kHost, compute.MemoryActivation, "qwen35-full-attn-rope-k")
			s.halKV.AppendKV(kvLayer, kRaw, kRope, v, pos)
		}
	} else {
		if rope, ok := be.(ropeInPlaceBackend); ok {
			q = rope.RoPEInPlace(q, pos, nH, hd, theta)
		} else {
			q = be.RoPE(q, pos, nH, hd, theta)
		}
		if appender, ok := s.halKV.(kvRoPEAppender); ok {
			appender.AppendKVRoPE(kvLayer, kRaw, v, pos, nKV, hd, theta)
		} else {
			kRope := be.RoPE(kRaw, pos, nKV, hd, theta)
			s.halKV.AppendKV(kvLayer, kRaw, kRope, v, pos)
		}
	}

	attnOut := be.Attention(q, s.halKV, kvLayer, true, grp, scale)
	if cfg.AttnOutputGate {
		if gated, ok := be.(qwen35SigmoidGateBackend); ok {
			gated.SigmoidMulInPlace(attnOut, gate)
		} else {
			gateHost := s.readQwen35FullAttention(layer, "full-attention gate read", gate)
			outHost := s.readQwen35FullAttention(layer, "full-attention output read", attnOut)
			if len(gateHost) != len(outHost) {
				s.failBackendForward(layer, "full-attention output gate", fmt.Errorf("gate width %d does not match attention width %d", len(gateHost), len(outHost)))
			}
			for i := range outHost {
				outHost[i] *= sigmoidf(gateHost[i])
			}
			attnOut = s.uploadHostF32([]int{nH * hd}, outHost, compute.MemoryActivation, "qwen35-full-attn-gated-output")
		}
	}
	out := be.MatMul(s.matWeightHAL(p("self_attn.o_proj.weight")), attnOut)
	if cfg.AttentionBias && s.M.has(p("self_attn.o_proj.bias")) {
		be.AddBias(out, s.weightHAL(p("self_attn.o_proj.bias")))
	}
	be.AddInPlace(residual, out)
}

func (s *Session) qwen35SequencePrefillRequest(ids []int, needLogits bool) compute.Qwen35SequencePrefillRequest {
	cfg := s.M.Cfg
	nK, nV, kHd, vHd, _, _, _ := cfg.linearAttnDims()
	req := compute.Qwen35SequencePrefillRequest{
		Path: compute.Qwen35SequencePrefillPath, TokenIDs: append([]int(nil), ids...), StartPos: s.halKV.Len(),
		TokenEmbedding: s.weightHAL("model.embed_tokens.weight"), OutputNorm: s.normWeightHAL("model.norm.weight"), Output: s.lmHeadMatHAL(),
		Layers: make([]compute.Qwen35SequenceLayer, cfg.NumLayers), States: make([]compute.Qwen35SequenceState, cfg.NumLayers), KV: s.halKV,
		Hidden: cfg.HiddenSize, Intermediate: cfg.IntermediateSize, NumHeads: cfg.NumHeads, NumKVHeads: cfg.NumKVHeads,
		HeadDim: cfg.HeadDim, RotaryDim: cfg.rotaryDim(), NumKeyHeads: nK, NumValueHeads: nV, KeyHeadDim: kHd, ValueHeadDim: vHd,
		ConvKernel: cfg.LinearConvKernelDim, RMSNormEpsilon: float32(cfg.RMSNormEps), RoPEThetaForLayer: make([]float64, cfg.NumLayers), NeedLogits: needLogits,
	}
	for l := 0; l < cfg.NumLayers; l++ {
		req.RoPEThetaForLayer[l] = cfg.ropeThetaForLayer(l)
		p := func(suffix string) string { return layerName(l, suffix) }
		layer := compute.Qwen35SequenceLayer{InputNorm: s.normWeightHAL(p("input_layernorm.weight")), PostNorm: s.normWeightHAL(p("post_attention_layernorm.weight")), Gate: s.matWeightHAL(p("mlp.gate_proj.weight")), Up: s.matWeightHAL(p("mlp.up_proj.weight")), Down: s.matWeightHAL(p("mlp.down_proj.weight"))}
		if cfg.isLinearAttnLayer(l) {
			layer.Linear = true
			layer.GDNInQKV = s.matWeightHAL(p("linear_attn.in_proj_qkv.weight"))
			layer.GDNInZ = s.matWeightHAL(p("linear_attn.in_proj_z.weight"))
			layer.GDNInB = s.matWeightHAL(p("linear_attn.in_proj_b.weight"))
			layer.GDNInA = s.matWeightHAL(p("linear_attn.in_proj_a.weight"))
			layer.GDNConv = s.weightHAL(p("linear_attn.conv1d.weight"))
			layer.GDNALog = s.weightHAL(p("linear_attn.A_log"))
			layer.GDNDTBias = s.weightHAL(p("linear_attn.dt_bias"))
			layer.GDNNorm = s.weightHAL(p("linear_attn.norm.weight"))
			layer.GDNOut = s.matWeightHAL(p("linear_attn.out_proj.weight"))
			state := s.qwen35HAL.layers[l]
			req.States[l] = compute.Qwen35SequenceState{Conv: state.conv, Recurrent: state.recurrent}
		} else {
			layer.Q = s.matWeightHAL(p("self_attn.q_proj.weight"))
			layer.K = s.matWeightHAL(p("self_attn.k_proj.weight"))
			layer.V = s.matWeightHAL(p("self_attn.v_proj.weight"))
			layer.O = s.matWeightHAL(p("self_attn.o_proj.weight"))
			if s.M.hasWeight(p("self_attn.q_norm.weight")) {
				layer.QNorm = s.normWeightHAL(p("self_attn.q_norm.weight"))
			}
			if s.M.hasWeight(p("self_attn.k_norm.weight")) {
				layer.KNorm = s.normWeightHAL(p("self_attn.k_norm.weight"))
			}
		}
		req.Layers[l] = layer
	}
	return req
}

func (s *Session) tryQwen35SequencePrefill(ids []int, needLogits bool) (compute.Qwen35SequencePrefillResult, bool, error) {
	if s == nil || s.M == nil || s.Backend == nil || !s.M.Cfg.IsQwen35Hybrid() || len(ids) < 2 {
		return compute.Qwen35SequencePrefillResult{}, false, nil
	}
	seq, advertised, err := qwen35SequencePrefillBackend(s.Backend)
	if err != nil || !advertised {
		return compute.Qwen35SequencePrefillResult{}, advertised, err
	}
	startPos := s.halKV.Len()
	result, err := seq.Qwen35SequencePrefill(s.qwen35SequencePrefillRequest(ids, needLogits))
	if err != nil {
		return result, true, &BackendForwardOperationError{Backend: s.Backend.Name(), Forward: ForwardQwen35GDN, Path: compute.Qwen35SequencePrefillPath, Layer: -1, Stage: "sequence prefill", Cause: err}
	}
	if result.Tokens != len(ids) || s.halKV.Len() != startPos+len(ids) || result.LastHidden.Buf() == nil || !result.LastHidden.Ready() || (needLogits && (result.Logits.Buf() == nil || !result.Logits.Ready())) {
		return result, true, &BackendForwardOperationError{Backend: s.Backend.Name(), Forward: ForwardQwen35GDN, Path: compute.Qwen35SequencePrefillPath, Layer: -1, Stage: "sequence result", Cause: fmt.Errorf("malformed result: tokens=%d want=%d kv_len=%d want=%d", result.Tokens, len(ids), s.halKV.Len(), startPos+len(ids))}
	}
	return result, true, nil
}

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

func (s *Session) closeQwen35HALState() {
	if s == nil || s.qwen35HAL == nil || s.Backend == nil {
		return
	}
	for l := range s.qwen35HAL.layers {
		state := &s.qwen35HAL.layers[l]
		if state.conv.Buf() != nil {
			s.Backend.Free(state.conv)
			state.conv = compute.Tensor{}
		}
		if state.recurrent.Buf() != nil {
			s.Backend.Free(state.recurrent)
			state.recurrent = compute.Tensor{}
		}
	}
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
		s.weightHAL(p("linear_attn.in_proj_qkv.weight")),
		s.weightHAL(p("linear_attn.in_proj_z.weight")),
		s.weightHAL(p("linear_attn.in_proj_b.weight")),
		s.weightHAL(p("linear_attn.in_proj_a.weight")),
		s.weightHAL(p("linear_attn.conv1d.weight")),
		s.weightHAL(p("linear_attn.A_log")),
		s.weightHAL(p("linear_attn.dt_bias")),
		s.weightHAL(p("linear_attn.norm.weight")),
		s.weightHAL(p("linear_attn.out_proj.weight")),
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

func (s *Session) qwen35QueryWeightsHAL(layer int) (query, gate compute.Tensor) {
	cfg := s.M.Cfg
	name := layerName(layer, "self_attn.q_proj.weight")
	q, g := splitQwen35HeadInterleavedRows(s.M.tensor(name), cfg.NumHeads, cfg.HeadDim, cfg.HiddenSize)
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
// current generic HAL has no split-query/output-gate or partial-RoPE primitives, so this
// correctness spine performs only those full-attention transforms through explicit host
// readback. The linear-attention branch above never reaches this function and remains
// wholly backend-resident; a future full-attention fused operation can replace this
// bounded bridge without changing GDN dispatch or state ownership.
func (s *Session) qwen35FullAttentionHAL(layer, pos int, residual compute.Tensor, eps, scale float32, grp int) {
	be, cfg := s.Backend, s.M.Cfg
	hd, nH, nKV := cfg.HeadDim, cfg.NumHeads, cfg.NumKVHeads
	p := func(suffix string) string { return layerName(layer, suffix) }
	xn := be.RMSNorm(residual, s.normWeightHAL(p("input_layernorm.weight")), eps)

	var q, gate compute.Tensor
	if cfg.AttnOutputGate {
		qWeight, gateWeight := s.qwen35QueryWeightsHAL(layer)
		q, gate = be.MatMul(qWeight, xn), be.MatMul(gateWeight, xn)
	} else {
		q = be.MatMul(s.matWeightHAL(p("self_attn.q_proj.weight")), xn)
	}
	kRaw := be.MatMul(s.matWeightHAL(p("self_attn.k_proj.weight")), xn)
	v := be.MatMul(s.matWeightHAL(p("self_attn.v_proj.weight")), xn)
	if cfg.AttentionBias {
		if cfg.AttnOutputGate {
			qBias, gateBias := s.qwen35QueryBiasHAL(layer)
			be.AddBias(q, qBias)
			be.AddBias(gate, gateBias)
		} else {
			be.AddBias(q, s.weightHAL(p("self_attn.q_proj.bias")))
		}
		be.AddBias(kRaw, s.weightHAL(p("self_attn.k_proj.bias")))
		be.AddBias(v, s.weightHAL(p("self_attn.v_proj.bias")))
	}

	kvLayer := qwen35HALKVLayer(cfg, layer)
	theta := cfg.ropeThetaForLayer(layer)
	if cfg.rotaryDim() != hd {
		qHost := s.readQwen35FullAttention(layer, "partial-RoPE query read", q)
		kHost := s.readQwen35FullAttention(layer, "partial-RoPE key read", kRaw)
		cos, sin := ropeRowForLayer(cfg, layer, pos)
		ropeRowQKInto(qHost, kHost, cos, sin, hd, nH, nKV)
		q = s.uploadHostF32([]int{nH * hd}, qHost, compute.MemoryActivation, "qwen35-full-attn-rope-q")
		kRope := s.uploadHostF32([]int{nKV * hd}, kHost, compute.MemoryActivation, "qwen35-full-attn-rope-k")
		s.halKV.AppendKV(kvLayer, kRaw, kRope, v, pos)
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
	out := be.MatMul(s.matWeightHAL(p("self_attn.o_proj.weight")), attnOut)
	if cfg.AttentionBias && s.M.has(p("self_attn.o_proj.bias")) {
		be.AddBias(out, s.weightHAL(p("self_attn.o_proj.bias")))
	}
	be.AddInPlace(residual, out)
}

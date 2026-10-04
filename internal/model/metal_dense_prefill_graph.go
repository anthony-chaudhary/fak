//go:build darwin && arm64 && cgo

package model

// metal_dense_prefill_graph.go — the dense resident-Q4_K prefill layer graph (#13599 prefill
// scope). prefillBatchedQ4K's host route pays one synchronous command buffer per projection
// (q, k, v, o, gate, up, down: ~7 per layer) with CPU RMSNorm, Q8 panel quantize, SwiGLU and
// residual adds between them, plus a host copy-in/copy-out per call. This route folds
// everything between two host attentions into ONE command buffer:
//
//	[tail of layer l-1]  o_proj(attn) -> X += o -> RMSNorm -> gate,up -> SwiGLU -> down -> X += down
//	[head of layer l]    RMSNorm(X) -> q,k,v
//
// so a prefill pays L+1 command buffers instead of ~7L. The causal attention, q/k/v bias,
// qk-norm, RoPE and KV append stay on the host between graphs (prefill attention on device is
// a separate lane, #13695), which is why q/k/v come back pre-bias exactly as the host route
// produces them. Projections reuse the hybrid graph's per-format encoder (Q4_K / Q6_K / Q8 with
// a lazily device-quantized activation), so the GEMM kernels are the same ones the host route
// dispatches (#13692 owns those kernels).
//
// Fail-open: every segment either succeeds atomically (X overwritten with the device residual,
// q/k/v returned) or reports ok=false having changed nothing, and the caller runs the identical
// work on the host route. Nothing host-visible (X, the KV cache) is mutated before FinishRead
// returns a completed receipt.

import (
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

// denseQ4KPrefillGraphOff is the process-wide kill switch for the dense prefill layer graph.
// The route is on by default whenever its admission holds; SetDenseQ4KPrefillGraph(false)
// pins every prefill to the per-projection host route (A/B receipts, bisection).
var denseQ4KPrefillGraphOff atomic.Bool

// SetDenseQ4KPrefillGraph enables (true, the default) or disables the dense resident-Q4_K
// prefill layer graph for this process.
func SetDenseQ4KPrefillGraph(on bool) { denseQ4KPrefillGraphOff.Store(!on) }

// denseQ4KGraphFailSegmentForTest, when >= 0, makes the segment whose head layer is that
// index decline before encoding (the last segment uses NumLayers), exercising the mid-prefill
// host fallback.
var denseQ4KGraphFailSegmentForTest = -1

// denseQ4KPrefillGraphMaxRows bounds one graph's panel so a single layer graph stays well
// inside the bounded command-buffer wait (MG_Q4K_WAIT_LIMIT_MS) on a loaded host; longer
// prompts are walked in panels of this size by prefillBatchedQ4K.
const denseQ4KPrefillGraphMaxRows = 1024

// denseQ4KPrefillGraphEligible reports whether every layer of this session can run on the
// dense layer graph: Metal resident Q4_K on, a plain RMSNorm PreNorm SiLU-gated model whose
// o/gate/up/down projections carry no bias (q/k/v bias is applied on the host), no prism
// rotation, and every projection resident in a format the graph encodes.
func (s *Session) denseQ4KPrefillGraphEligible() bool {
	if denseQ4KPrefillGraphOff.Load() || s == nil || !s.MetalQ4K || !metalgemm.Available() {
		return false
	}
	m := s.M
	cfg := m.Cfg
	if m.prism != nil || cfg.LayerNorm || cfg.NormGain1p || !siluGatedMLP(cfg) || cfg.HiddenSize%32 != 0 {
		return false
	}
	for l := 0; l < cfg.NumLayers; l++ {
		lp := func(str string) string { return layerName(l, str) }
		if m.has(lp("self_attn.o_proj.bias")) || !m.biasFreeGatedMLP(lp("mlp.gate_proj.bias"), lp("mlp.up_proj.bias"), lp("mlp.down_proj.bias")) {
			return false
		}
		for _, name := range denseProjectionNames(lp) {
			if !qwen35MetalMTPProjectionReady(m, name) {
				return false
			}
		}
	}
	return true
}

// denseQ4KGraphSegment encodes the tail of layer prev (skipped when prev < 0) and the head of
// layer next (skipped when next < 0) in one command buffer over the host residual X [P,H],
// with attn [P, nH*hd] the host attention output of layer prev. On success X holds the device
// residual and q/k/v are layer next's pre-bias projections; on ok=false nothing changed.
func (s *Session) denseQ4KGraphSegment(prev int, attn []float32, next int, X []float32, P int) (q, k, v []float32, receipt metalgemm.GraphReceipt, ok bool) {
	m, cfg := s.M, s.M.Cfg
	H := cfg.HiddenSize
	eps := float32(cfg.RMSNormEps)
	head := next
	if head < 0 {
		head = cfg.NumLayers
	}
	if head == denseQ4KGraphFailSegmentForTest {
		return nil, nil, nil, receipt, false
	}
	g, err := metalgemm.BeginProjectionGraph(X, nil, nil, P, H)
	if err != nil {
		return nil, nil, nil, receipt, false
	}
	defer g.Free()
	// Fail-closed: a mode the graph cannot honor (e.g. MM32 at P=32) leaves it on the scalar
	// kernel, which is the host route's kernel for every P but 32.
	g.SetQ4KGEMMMode(metalgemm.Q4KGEMMModeForPrompt(P))
	x, err := g.Input(H)
	if err != nil {
		return nil, nil, nil, receipt, false
	}
	quantized := map[*metalgemm.GraphResult]*metalgemm.QuantizedGraphResult{}
	if prev >= 0 {
		lp := func(str string) string { return layerName(prev, str) }
		a, err := g.Upload(attn, cfg.NumHeads*cfg.HeadDim)
		if err != nil {
			return nil, nil, nil, receipt, false
		}
		o, err := qwen35GraphProjection(g, s, lp("self_attn.o_proj.weight"), a, quantized)
		if err != nil || g.AddInPlace(x, o) != nil {
			return nil, nil, nil, receipt, false
		}
		xn2, err := g.RMSNorm(x, m.tensor(lp("post_attention_layernorm.weight")), eps, false)
		if err != nil {
			return nil, nil, nil, receipt, false
		}
		gu, err := qwen35GraphProjections(g, s, []string{lp("mlp.gate_proj.weight"), lp("mlp.up_proj.weight")}, xn2, quantized)
		if err != nil || g.SwiGLUInPlace(gu[0], gu[1]) != nil {
			return nil, nil, nil, receipt, false
		}
		down, err := qwen35GraphProjection(g, s, lp("mlp.down_proj.weight"), gu[0], quantized)
		if err != nil || g.AddInPlace(x, down) != nil {
			return nil, nil, nil, receipt, false
		}
	}
	terminal := []*metalgemm.GraphResult{x}
	if next >= 0 {
		lp := func(str string) string { return layerName(next, str) }
		xn, err := g.RMSNorm(x, m.tensor(lp("input_layernorm.weight")), eps, false)
		if err != nil {
			return nil, nil, nil, receipt, false
		}
		qkv, err := qwen35GraphProjections(g, s, []string{lp("self_attn.q_proj.weight"), lp("self_attn.k_proj.weight"), lp("self_attn.v_proj.weight")}, xn, quantized)
		if err != nil {
			return nil, nil, nil, receipt, false
		}
		terminal = append(terminal, qkv...)
	}
	outs, receipt, err := g.FinishRead(terminal...)
	if err != nil || !receipt.Committed || !receipt.CompletedWait {
		return nil, nil, nil, receipt, false
	}
	s.countMetalGraphCommandBuffer(1)
	copy(X, outs[0])
	if next >= 0 {
		q, k, v = outs[1], outs[2], outs[3]
	}
	return q, k, v, receipt, true
}

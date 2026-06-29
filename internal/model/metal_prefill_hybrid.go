//go:build darwin && arm64 && cgo

package model

// metal_prefill_hybrid.go — the Metal GPU twin of the Qwen3.6 hybrid (Gated-DeltaNet) prefill.
// Built by default on Apple Silicon with cgo. It is deliberately thin: the entire prefill body — both
// RMSNorms, the conv1d+SiLU mixer, the q/k L2-norm, the per-head delta-rule recurrent scan, the
// gated RMSNorm readout, the full-attention RoPE/GQA/output-gate, and every residual — lives in
// the backend-agnostic core prefillQwen35HybridViaMM (metal_prefill_hybrid_core.go), proven
// host-independently against the CPU template by TestQwen35HybridViaMMMatchesCPUTemplate. This
// file supplies only the one substitution that core abstracts: a GPU f16 GEMM for the projection
// /MLP matmuls. Keeping the GDN recurrence on the CPU and moving just the projections to the
// device is the measured lever (the projections are the prefill wall; the GDN scan is ~0.5%;
// #65, #977), and lifting requirePreNorm("Metal prefill") for the hybrid is what lets it use the
// Metal prefill at all (#71).
//
// Weights: like metalWeights(), the GPU holds an f16 copy of each projection, dequantized once
// from the Q8_0 store and cached per *Model. The hybrid's projection set is per-layer: every
// layer carries the three MLP matmuls, while the per-layer mixer is EITHER the five linear_attn
// projections (linear_attention layers) OR the four self_attn projections (full_attention
// layers), dispatched by isLinearAttnLayer — the same split the core's mm calls walk.

import (
	"sync"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

var (
	metalHybridMu sync.Mutex
	metalHybridWt = map[*Model]map[string]*metalgemm.Weight{} // per-Model name -> GPU f16 weight
)

// metalWeightsQwen35Hybrid returns this model's GPU projection table for the hybrid prefill,
// uploading it once. It mirrors metalWeights() (same dequantQ8 -> f16 Upload, big f32 buffer
// freed after each upload) but uploads the hybrid's per-layer projection set instead of the seven
// uniform standard-attention names.
func (m *Model) metalWeightsQwen35Hybrid() map[string]*metalgemm.Weight {
	metalHybridMu.Lock()
	defer metalHybridMu.Unlock()
	if w, ok := metalHybridWt[m]; ok {
		return w
	}
	cfg := m.Cfg
	w := make(map[string]*metalgemm.Weight, 8*cfg.NumLayers)
	upload := func(name string) {
		qt := m.q8(name)
		h := metalgemm.Upload(dequantQ8(qt), qt.out, qt.in)
		if h == nil {
			panic("model: metal hybrid weight upload failed for " + name)
		}
		w[name] = h
	}
	for l := 0; l < cfg.NumLayers; l++ {
		upload(layerName(l, "mlp.gate_proj.weight"))
		upload(layerName(l, "mlp.up_proj.weight"))
		upload(layerName(l, "mlp.down_proj.weight"))
		if cfg.isLinearAttnLayer(l) {
			upload(layerName(l, "linear_attn.in_proj_qkv.weight"))
			upload(layerName(l, "linear_attn.in_proj_z.weight"))
			upload(layerName(l, "linear_attn.in_proj_b.weight"))
			upload(layerName(l, "linear_attn.in_proj_a.weight"))
			upload(layerName(l, "linear_attn.out_proj.weight"))
		} else {
			upload(layerName(l, "self_attn.q_proj.weight"))
			upload(layerName(l, "self_attn.k_proj.weight"))
			upload(layerName(l, "self_attn.v_proj.weight"))
			upload(layerName(l, "self_attn.o_proj.weight"))
		}
	}
	metalHybridWt[m] = w
	return w
}

// prefillBatchedMetalQwen35Hybrid is the Metal hybrid prefill: it feeds the backend-agnostic core
// a GPU f16 GEMM (Y[P,out] = X[P,in] * W[name]^T) for each projection and lets the core run the
// recurrence/attention/norm body on the CPU. It fills the same f32 KV + linear-attn caches the
// CPU hybrid paths build (so decode stays valid) and returns the last token's post-final-norm
// hidden (caller applies the head). Reached only for a fresh prefill via metalQwen35HybridPrefillOK.
func (s *Session) prefillBatchedMetalQwen35Hybrid(ids []int) []float32 {
	m := s.M
	P := len(ids)
	gw := m.metalWeightsQwen35Hybrid()
	// mm runs Y[P,out] = X[P,in] * W[name]^T on the GPU into a fresh buffer; `in` is implicit in
	// the uploaded weight, so the core's hybridGemmFn signature drops it.
	mm := func(name string, X []float32, out int) []float32 {
		Y := make([]float32, P*out)
		gw[name].MatMul(X, P, Y)
		return Y
	}
	return s.prefillQwen35HybridViaMM(ids, mm)
}

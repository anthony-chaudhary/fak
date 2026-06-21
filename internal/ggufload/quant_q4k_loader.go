package ggufload

// quant_q4k_loader.go — the direct-q4 GGUF loader for the resident Q4_K path
// (QWEN36-NATIVE-PERF-PLAN-2026-06-19.md P1). Mirrors WeightSource.QuantModelProfile but
// routes every eligible Q4_K matmul tensor straight into a resident q4kTensor (raw GGUF
// bytes, no dequantF32 Q4→f32, no f32→Q8 re-quant): the ~10× load win + the drop in
// resident footprint, streaming the q4_k_m bytes llama.cpp streams.
//
// Eligibility (model.ResidentQ4KEligible) is the correctness gate: only IDENTITY-
// normalized matmul weights (MLP gate/up/down, self_attn.v_proj/o_proj, lm_head, expert
// FFN) are held raw, because the GGUF's ggml-layout bytes are already the HF layout the
// forward expects for those. The normalize-sensitive weights (qwen35 linear_attn family +
// rotary/gated self_attn q/k/qkv) MUST stay on the proven dequant→normalize→Q8 path —
// storing their raw bytes would feed wrongly-laid-out weights to the forward and produce
// garbage. The Q6_K matmul minority (often attn_qkv / ffn_down / lm_head in a q4_k_m mix)
// also falls through to Q8, since the resident q4kTensor holds Q4_K blocks only.

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// LoadModelQ4K loads a GGUF checkpoint through the direct-resident-Q4_K path: eligible
// Q4_K matmul tensors are held raw (no round-trip), and everything else follows the
// standard quant-on-load path (Q8_0 for the remaining matmul weights, f32 for small
// tensors). Run the returned model through a Session with Q4K=true.
func LoadModelQ4K(path string) (*model.Model, error) {
	ws, err := OpenWeights(path)
	if err != nil {
		return nil, err
	}
	defer ws.Close()
	return ws.QuantModelQ4K()
}

// QuantModelQ4K is the WeightSource form of LoadModelQ4K: QuantModelProfile with the
// eligible-Q4_K branch pulled before the dequant, so those tensors never pay the f32
// round-trip.
func (s *WeightSource) QuantModelQ4K() (*model.Model, error) {
	cfg, err := s.File.Config()
	if err != nil {
		return nil, err
	}
	builder := model.NewQuantBuilder(cfg, cfg.TieWordEmbeddings)
	for _, info := range s.File.Tensors {
		canon, ok := CanonicalTensorNameArch(info.Name, cfg.ModelType)
		if !ok {
			return nil, fmt.Errorf("gguf: no canonical mapping for tensor %s", info.Name)
		}
		shape, err := modelShapeFromGGUFDims(info.Name, info.Dims)
		if err != nil {
			return nil, err
		}
		raw, _, err := s.TensorBytes(info.Name)
		if err != nil {
			return nil, err
		}

		// Direct-resident-Q4_K fast path: an eligible Q4_K matmul weight is wrapped raw,
		// skipping dequantF32 (Q4→f32) and the f32→Q8 re-quant entirely. ResidentQ4KEligible
		// is the authoritative gate (it already excludes the normalize-sensitive + non-matmul
		// names), so AddResidentQ4K stores and we continue.
		if info.Type == TensorQ4_K && model.ResidentQ4KEligible(cfg, canon) {
			if err := builder.AddResidentQ4K(canon, shape, raw); err != nil {
				return nil, err
			}
			continue
		}

		// Everything else (the normalize-sensitive projections, Q6_K matmul weights, the
		// embedding, norms, biases, fused qkv) follows the standard path: dequant →
		// normalize → builder, which quantizes the remaining matmul weights to Q8_0 and
		// keeps the small tensors as f32.
		data, err := dequantF32(info, raw)
		if err != nil {
			return nil, err
		}
		data, err = normalizeCanonicalTensorData(canon, data, cfg)
		if err != nil {
			return nil, err
		}
		if err := builder.AddF32Tensor(canon, shape, data); err != nil {
			return nil, err
		}
	}
	return builder.Build()
}

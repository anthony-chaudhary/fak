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
	return LoadModelQ4KProfile(path, nil)
}

// LoadModelQ4KProfile is LoadModelQ4K with an optional load profiler so the direct-resident-Q4_K
// path streams the same load-progress lines the lean-Q8 path does (a 466 GB GLM-5.2 resident load
// must not be silent). Nil profiler = no progress, byte-identical to the old LoadModelQ4K.
func LoadModelQ4KProfile(path string, p *LoadProfiler) (*model.Model, error) {
	ws, err := OpenWeights(path)
	if err != nil {
		return nil, err
	}
	defer ws.Close()
	return ws.QuantModelQ4KProfile(p)
}

// QuantModelQ4K is the WeightSource form of LoadModelQ4K: QuantModelProfile with the
// eligible-Q4_K branch pulled before the dequant, so those tensors never pay the f32
// round-trip.
func (s *WeightSource) QuantModelQ4K() (*model.Model, error) {
	return s.QuantModelQ4KProfile(nil)
}

// QuantModelQ4KProfile is QuantModelQ4K with optional progress reporting (p.SetTotal/Tick).
//
// The per-tensor work (read + dequant + normalize + expert split) runs on a bounded worker
// pool (gguf_parload.go); the builder mutations + the MLA KV-b merge + the profiler are
// applied SERIALLY in original tensor order by a single collector, so the built model is
// byte-identical to a serial load — only the CPU-bound dequant is parallelized. This is the
// S1 lever against the ~100-min single-core GLM-5.2 load
// (docs/notes/GLM52-FAK-NATIVE-SERVE-LOAD-SPEED-2026-06-25.md): zero arithmetic change,
// every core busy. The collector also records the per-quant-type resident-vs-dequant
// breakdown (the S4 visibility) so the mixed-quant cost is legible without an external dump.
func (s *WeightSource) QuantModelQ4KProfile(p *LoadProfiler) (*model.Model, error) {
	cfg, err := s.File.Config()
	if err != nil {
		return nil, err
	}
	builder := model.NewQuantBuilder(cfg, cfg.TieWordEmbeddings)
	kvbHalf := map[int]glmKVBHalf{} // MLA KV-b 2->1 merge buffer (see QuantModelProfile)
	p.SetTotal(len(s.File.Tensors))

	// computeFn is the pure, concurrency-safe per-tensor work: it reads + dequantizes +
	// normalizes + splits, returning the builder mutations to apply. It touches no shared
	// state (TensorBytes copies; dequantF32 allocates fresh; the helpers are pure over the
	// read-only Config), so it is safe to run from many workers at once.
	computeFn := func(info TensorInfo) tensorWork {
		tw := tensorWork{tickBytes: tensorOnDiskBytes(info)}
		if cfg.ModelType == "glm_moe_dsa" {
			// Drop the MTP ("nextn") head + any vision tower the text forward never reads.
			if glmMoeDsaSkipGGUFTensor(info.Name) {
				return tw
			}
			// MLA KV-b half: dequant it; the collector buffers + merges the pair in order.
			if layer, half, ok := glmMoeDsaSplitKVB(info.Name); ok {
				shape, err := modelShapeFromGGUFDims(info.Name, info.Dims)
				if err != nil {
					tw.err = err
					return tw
				}
				raw, _, err := s.TensorBytes(info.Name)
				if err != nil {
					tw.err = err
					return tw
				}
				data, err := dequantF32(info, raw)
				if err != nil {
					tw.err = err
					return tw
				}
				tw.pending = []pendingTensor{{isKVBHalf: true, layer: layer, half: half, shape: shape, f32: append([]float32(nil), data...)}}
				return tw
			}
			// Batched routed experts: split the [E,out,in] blob 1->E. A block-aligned Q4_K /
			// Q5_K / Q6_K blob splits as RAW bytes -> resident (no dequant): the experts are
			// GLM-5.2's 417 GB bulk and unsloth UD-Q4_K_M is a MIXED quant, so handling all
			// three k-quants resident (not just Q4_K) is the load-time lever that turns the
			// whole expert load I/O-bound. Any other type falls to the f32 dequant-split.
			if layer, proj, ok := glmMoeDsaBatchedExpert(info.Name); ok {
				shape, err := modelShapeFromGGUFDims(info.Name, info.Dims)
				if err != nil {
					tw.err = err
					return tw
				}
				raw, _, err := s.TensorBytes(info.Name)
				if err != nil {
					tw.err = err
					return tw
				}
				tw.acctType, tw.acctExpert, tw.acctBytes = info.Type.String(), true, tensorOnDiskBytes(info)
				if blockBytes, residentable := residentExpertBlockBytes(info.Type); residentable {
					kqExperts, aligned, err := splitGLMMoeDsaExpertsKQuantRaw(layer, proj, shape, raw, blockBytes)
					if err != nil {
						tw.err = err
						return tw
					}
					if aligned && model.ResidentKQuantEligible(cfg, kqExperts[0].Name) {
						tw.pending = make([]pendingTensor, len(kqExperts))
						for i, ex := range kqExperts {
							tw.pending[i] = pendingTensor{resident: true, residentType: info.Type, name: ex.Name, shape: ex.Shape, raw: ex.Raw}
						}
						tw.acctResident, tw.acctTensors = true, len(kqExperts)
						return tw
					}
				}
				data, err := dequantF32(info, raw)
				if err != nil {
					tw.err = err
					return tw
				}
				experts, err := splitGLMMoeDsaExperts(layer, proj, shape, data)
				if err != nil {
					tw.err = err
					return tw
				}
				tw.pending = make([]pendingTensor, len(experts))
				for i, ex := range experts {
					tw.pending[i] = pendingTensor{resident: false, name: ex.Name, shape: ex.Shape, f32: ex.Data}
				}
				tw.acctResident, tw.acctTensors = false, len(experts)
				return tw
			}
		}
		canon, ok := CanonicalTensorNameArch(info.Name, cfg.ModelType)
		if !ok {
			tw.err = fmt.Errorf("gguf: no canonical mapping for tensor %s", info.Name)
			return tw
		}
		shape, err := modelShapeFromGGUFDims(info.Name, info.Dims)
		if err != nil {
			tw.err = err
			return tw
		}
		raw, _, err := s.TensorBytes(info.Name)
		if err != nil {
			tw.err = err
			return tw
		}
		tw.acctType, tw.acctExpert, tw.acctBytes, tw.acctTensors = info.Type.String(), false, tensorOnDiskBytes(info), 1
		// Direct-resident-Q4_K fast path: an eligible Q4_K matmul weight is wrapped raw,
		// skipping dequantF32 (Q4→f32) and the f32→Q8 re-quant entirely. raw is a fresh
		// TensorBytes copy, so handing it straight to the builder is safe.
		if info.Type == TensorQ4_K && model.ResidentQ4KEligible(cfg, canon) {
			tw.pending = []pendingTensor{{resident: true, residentType: info.Type, name: canon, shape: shape, raw: raw}}
			tw.acctResident = true
			return tw
		}
		// Everything else (normalize-sensitive projections, Q6_K matmul weights, embedding,
		// norms, biases, fused qkv) follows the standard dequant → normalize → builder path.
		data, err := dequantF32(info, raw)
		if err != nil {
			tw.err = err
			return tw
		}
		data, err = normalizeCanonicalTensorData(canon, data, cfg)
		if err != nil {
			tw.err = err
			return tw
		}
		tw.pending = []pendingTensor{{resident: false, name: canon, shape: shape, f32: data}}
		return tw
	}

	// applyFn owns all shared mutable state (builder, KV-b merge buffer, profiler) and runs
	// on the single collector goroutine in original tensor order.
	applyFn := func(tw tensorWork) error {
		p.Tick(tw.tickBytes)
		p.recordLoadPath(tw.acctType, tw.acctExpert, tw.acctResident, tw.acctBytes, tw.acctTensors)
		for _, pt := range tw.pending {
			switch {
			case pt.isKVBHalf:
				merged, ready, err := bufferGLMKVBHalf(kvbHalf, pt.layer, pt.half, pt.shape, pt.f32)
				if err != nil {
					return err
				}
				if ready {
					md, err := normalizeCanonicalTensorData(merged.Name, merged.Data, cfg)
					if err != nil {
						return err
					}
					if err := builder.AddF32Tensor(merged.Name, merged.Shape, md); err != nil {
						return err
					}
				}
			case pt.resident:
				switch pt.residentType {
				case TensorQ6_K:
					if err := builder.AddResidentQ6K(pt.name, pt.shape, pt.raw); err != nil {
						return err
					}
				case TensorQ5_K:
					if err := builder.AddResidentQ5K(pt.name, pt.shape, pt.raw); err != nil {
						return err
					}
				default: // TensorQ4_K
					if err := builder.AddResidentQ4K(pt.name, pt.shape, pt.raw); err != nil {
						return err
					}
				}
			default:
				if err := builder.AddF32Tensor(pt.name, pt.shape, pt.f32); err != nil {
					return err
				}
			}
		}
		return nil
	}

	if err := s.parallelQuantLoad(computeFn, applyFn); err != nil {
		return nil, err
	}
	if err := glmKVBUnpaired(kvbHalf); err != nil {
		return nil, err
	}
	if p != nil {
		p.EmitLoadPathSummary(p.Progress)
	}
	return builder.Build()
}

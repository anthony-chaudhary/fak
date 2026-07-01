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

// ExpertShard names the routed expert band [Lo,Hi) a rank owns during an expert-parallel
// load. Dense, attention, router, embeddings, and shared-expert tensors remain replicated; this
// band filters only batched routed-expert tensors before they enter the resident store.
type ExpertShard struct {
	Lo int
	Hi int
}

// ExpertShardForRank derives the contiguous expert band owned by rank under the same tiling as
// model.ExpertParallelPlan. It is a loader-facing helper so serve code can use the planner and the
// resident admission filter from one source of truth.
func ExpertShardForRank(numExperts, ranks, rank int) (ExpertShard, error) {
	plan, err := model.ExpertParallelPlan(numExperts, ranks)
	if err != nil {
		return ExpertShard{}, err
	}
	if rank < 0 || rank >= len(plan.Shards) {
		return ExpertShard{}, fmt.Errorf("gguf: expert-parallel rank %d outside [0,%d)", rank, len(plan.Shards))
	}
	shard := plan.Shards[rank]
	return ExpertShard{Lo: shard.Lo, Hi: shard.Hi}, nil
}

type q4kLoadOptions struct {
	expertShardSet bool
	expertShard    ExpertShard
}

// Q4KLoadOption configures the direct-resident-Q4_K GGUF load path.
type Q4KLoadOption func(*q4kLoadOptions)

// WithExpertShard keeps only routed experts in [lo,hi) when splitting batched MoE expert GGUF
// tensors. Use this for expert-parallel per-rank loads; omit it for the historical full load.
func WithExpertShard(lo, hi int) Q4KLoadOption {
	return func(o *q4kLoadOptions) {
		o.expertShardSet = true
		o.expertShard = ExpertShard{Lo: lo, Hi: hi}
	}
}

func resolveQ4KLoadOptions(cfg model.Config, opts []Q4KLoadOption) (q4kLoadOptions, error) {
	var out q4kLoadOptions
	for _, opt := range opts {
		if opt != nil {
			opt(&out)
		}
	}
	if !out.expertShardSet {
		return out, nil
	}
	if cfg.NumExperts <= 0 {
		return out, fmt.Errorf("gguf: expert shard requested for non-MoE config (NumExperts=%d)", cfg.NumExperts)
	}
	if out.expertShard.Lo < 0 || out.expertShard.Hi <= out.expertShard.Lo || out.expertShard.Hi > cfg.NumExperts {
		return out, fmt.Errorf("gguf: expert shard [%d,%d) outside [0,%d)", out.expertShard.Lo, out.expertShard.Hi, cfg.NumExperts)
	}
	return out, nil
}

func (o q4kLoadOptions) keepExpert(expert int) bool {
	if !o.expertShardSet {
		return true
	}
	return expert >= o.expertShard.Lo && expert < o.expertShard.Hi
}

func (o q4kLoadOptions) keptExperts(total int) int {
	if !o.expertShardSet {
		return total
	}
	lo, hi := o.expertShard.Lo, o.expertShard.Hi
	if lo < 0 {
		lo = 0
	}
	if hi > total {
		hi = total
	}
	if hi <= lo {
		return 0
	}
	return hi - lo
}

// LoadModelQ4KProfile is LoadModelQ4K with an optional load profiler so the direct-resident-Q4_K
// path streams the same load-progress lines the lean-Q8 path does (a 466 GB GLM-5.2 resident load
// must not be silent). Nil profiler = no progress, byte-identical to the old LoadModelQ4K.
func LoadModelQ4KProfile(path string, p *LoadProfiler) (*model.Model, error) {
	return LoadModelQ4KProfileOptions(path, p)
}

// LoadModelQ4KProfileOptions is LoadModelQ4KProfile with explicit load options.
func LoadModelQ4KProfileOptions(path string, p *LoadProfiler, opts ...Q4KLoadOption) (*model.Model, error) {
	return loadVia(path, func(ws *WeightSource) (*model.Model, error) {
		return ws.QuantModelQ4KProfileOptions(p, opts...)
	})
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
	return s.QuantModelQ4KProfileOptions(p)
}

// QuantModelQ4KProfileOptions is QuantModelQ4KProfile with explicit load options. The default
// option set is byte-compatible with QuantModelQ4KProfile; an expert shard only filters routed
// expert tensors after the GGUF batched expert split.
func (s *WeightSource) QuantModelQ4KProfileOptions(p *LoadProfiler, opts ...Q4KLoadOption) (*model.Model, error) {
	cfg, err := s.File.Config()
	if err != nil {
		return nil, err
	}
	loadOpts, err := resolveQ4KLoadOptions(cfg, opts)
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
				shape, data, err := s.dequantGGUFShapeF32(info)
				if err != nil {
					tw.err = err
					return tw
				}
				tw.pending = []pendingTensor{{isKVBHalf: true, layer: layer, half: half, shape: shape, f32: append([]float32(nil), data...)}}
				return tw
			}
		}
		if archUsesGGUFBatchedMoEExperts(cfg.ModelType) {
			// Batched routed experts: split the [E,out,in] blob 1->E. A block-aligned raw-quant
			// blob splits as RAW bytes -> resident (no dequant): the experts are the MoE bulk,
			// and unsloth UD quants can use IQ3_XXS/IQ4_XS/Q8_0 in addition to the K-quants.
			// Any other type falls to the f32 dequant-split.
			if layer, proj, ok := glmMoeDsaBatchedExpert(info.Name); ok {
				shape, raw, err := s.shapeAndBytes(info)
				if err != nil {
					tw.err = err
					return tw
				}
				tw.acctType, tw.acctExpert, tw.acctBytes = info.Type.String(), true, tensorOnDiskBytes(info)
				if blockWeights, blockBytes, residentable := residentExpertBlockGeometry(info.Type); residentable {
					kqExperts, aligned, err := splitGLMMoeDsaExpertsRawQuant(layer, proj, shape, raw, blockWeights, blockBytes)
					if err != nil {
						tw.err = err
						return tw
					}
					if aligned && model.ResidentKQuantEligible(cfg, kqExperts[0].Name) {
						kept := loadOpts.keptExperts(len(kqExperts))
						tw.pending = make([]pendingTensor, 0, kept)
						for i, ex := range kqExperts {
							if !loadOpts.keepExpert(i) {
								continue
							}
							tw.pending = append(tw.pending, pendingTensor{resident: true, residentType: info.Type, name: ex.Name, shape: ex.Shape, raw: ex.Raw})
						}
						tw.acctResident, tw.acctTensors = true, kept
						if loadOpts.expertShardSet {
							if b, err := scaleExpertBandBytes(uint64(tw.acctBytes), kept, len(kqExperts)); err == nil {
								tw.acctBytes = int64(b)
							}
						}
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
				kept := loadOpts.keptExperts(len(experts))
				tw.pending = make([]pendingTensor, 0, kept)
				for i, ex := range experts {
					if !loadOpts.keepExpert(i) {
						continue
					}
					tw.pending = append(tw.pending, pendingTensor{resident: false, name: ex.Name, shape: ex.Shape, f32: ex.Data})
				}
				tw.acctResident, tw.acctTensors = false, kept
				if loadOpts.expertShardSet {
					if b, err := scaleExpertBandBytes(uint64(tw.acctBytes), kept, len(experts)); err == nil {
						tw.acctBytes = int64(b)
					}
				}
				return tw
			}
		}
		canon, ok := CanonicalTensorNameArch(info.Name, cfg.ModelType)
		if !ok {
			tw.err = fmt.Errorf("gguf: no canonical mapping for tensor %s", info.Name)
			return tw
		}
		shape, raw, err := s.shapeAndBytes(info)
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
		// Direct-resident-k-quant fast path for a DENSE (non-expert) matmul weight whose GGUF type
		// is a residentable k-quant — the q4_k_m dense down_proj and lm_head load Q6_K, which the
		// Q4_K-only fast path above skips. Without this they take the dequant→Q8 path and miss BOTH
		// the resident int8 kQuantMatRows (Stage A) and the fused Metal Q6_K MLP (Stage B,
		// FusedMLPQ6Down) — the per-token weight traffic for ~4 GB of Q6_K weights stays on the
		// slowest route. ResidentKQuantEligible is the SAME identity-normalization gate as the Q4_K
		// path (it refuses the normalize-sensitive q/k/qkv/linear_attn projections), so skipping
		// normalizeCanonicalTensorData here is safe for exactly the identity weights (ffn_down,
		// o_proj, lm_head) it admits. The expert k-quants take the batched resident path above.
		if _, _, residentable := residentExpertBlockGeometry(info.Type); residentable &&
			info.Type != TensorQ4_K && model.ResidentKQuantEligible(cfg, canon) {
			tw.pending = []pendingTensor{{resident: true, residentType: info.Type, name: canon, shape: shape, raw: raw}}
			tw.acctResident = true
			return tw
		}
		// Everything else (normalize-sensitive projections, embedding, norms, biases, fused qkv)
		// follows the standard dequant → normalize → builder path.
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
				case TensorIQ3_XXS:
					if err := builder.AddResidentIQ3XXS(pt.name, pt.shape, pt.raw); err != nil {
						return err
					}
				case TensorIQ4_XS:
					if err := builder.AddResidentIQ4XS(pt.name, pt.shape, pt.raw); err != nil {
						return err
					}
				case TensorQ8_0:
					if err := builder.AddResidentQ8_0(pt.name, pt.shape, pt.raw); err != nil {
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

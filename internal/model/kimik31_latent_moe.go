package model

// kimik31_latent_moe.go — the Kimi-K3.1 latent-MoE host oracle (ticket
// 1611-kimi-k31, port leaf sglang-kimi-k31-4 host side; goal Wave 1.2/1.3). In
// K3.1 the routed experts do NOT compute in the model's hidden width: a
// per-layer down-projection maps the hidden vector into a narrower LATENT space,
// the 896 experts run there, and an up-projection maps the combined expert
// output back to hidden width. Between the two the latent is RMSNorm'd —
// and the load-bearing ordering detail is that a tensor-parallel all-reduce of
// the routed sums must complete BEFORE that norm (sum(norm(x_i)) !=
// norm(sum(x_i))), so the norm is applied to the reduced latent, never per rank.
//
// SOURCE (direct port, pinned): SGLang
// python/sglang/srt/models/kimi_k3.py at
//
//   - `KimiK3MoE.__init__` builds `routed_expert_down_proj` (hidden -> latent),
//     `routed_expert_norm` (RMSNorm over the latent width, when
//     `latent_moe_use_norm`), and `routed_expert_up_proj` (latent -> hidden).
//   - `_latent_norm` applies that norm; `_reduce_latent` is the ordering witness
//     — "TP-partial routed sums must be reduced in latent space BEFORE the
//     RMSNorm (sum(norm(x_i)) != norm(sum(x_i)))".
//   - the experts' `hidden_size` is the latent width and their intermediate is
//     `moe_intermediate_size`, so the expert GEMMs live entirely in latent space.
//
// WHAT THIS OWNS: the host front/norm/tail composition for one layer and the
// reduce-before-norm ordering contract. WHAT THIS DOES NOT OWN: the routed
// expert GEMM itself (the router and expert execution leaves) and the CUDA
// kernels (the reference dispatches by capability; this host form is the eager
// reference a device path must match).

import "fmt"

// KimiK31LatentMoEConfig holds the K3.1 latent-MoE geometry. K3.1 ships Hidden
// 7168, Latent 3584, ExpertIntermediate 3072, LatentUseNorm true.
type KimiK31LatentMoEConfig struct {
	Hidden             int
	Latent             int
	ExpertIntermediate int
	LatentUseNorm      bool
	RMSNormEps         float32
}

// KimiK31LatentMoEWeights holds one layer's latent-MoE projections, in HF
// row-major orientation [out x in]: y[o] = sum_i W[o*in+i]*x[i].
//
// DownProj maps hidden -> latent ([Latent x Hidden]); UpProj maps latent ->
// hidden ([Hidden x Latent]); LatentNorm is the RMSNorm weight over the latent
// width ([Latent], required when LatentUseNorm).
type KimiK31LatentMoEWeights struct {
	DownProj   []float32
	UpProj     []float32
	LatentNorm []float32
}

// ErrKimiK31LatentMoEShape is the closed sentinel for fail-closed latent-MoE
// shape validation.
var ErrKimiK31LatentMoEShape = fmt.Errorf("model: kimi-k31 latent-moe shape mismatch")

// ValidateKimiK31LatentMoE fails closed on any malformed geometry/weights,
// naming the offending field.
func ValidateKimiK31LatentMoE(cfg KimiK31LatentMoEConfig, w KimiK31LatentMoEWeights) error {
	if cfg.Hidden <= 0 || cfg.Latent <= 0 || cfg.ExpertIntermediate <= 0 {
		return fmt.Errorf("%w: bad geometry hidden=%d latent=%d inter=%d",
			ErrKimiK31LatentMoEShape, cfg.Hidden, cfg.Latent, cfg.ExpertIntermediate)
	}
	if len(w.DownProj) != cfg.Latent*cfg.Hidden {
		return fmt.Errorf("%w: DownProj len %d != latent*hidden %d", ErrKimiK31LatentMoEShape, len(w.DownProj), cfg.Latent*cfg.Hidden)
	}
	if len(w.UpProj) != cfg.Hidden*cfg.Latent {
		return fmt.Errorf("%w: UpProj len %d != hidden*latent %d", ErrKimiK31LatentMoEShape, len(w.UpProj), cfg.Hidden*cfg.Latent)
	}
	if cfg.LatentUseNorm {
		if cfg.RMSNormEps <= 0 {
			return fmt.Errorf("%w: RMSNormEps %v must be > 0", ErrKimiK31LatentMoEShape, cfg.RMSNormEps)
		}
		if len(w.LatentNorm) != cfg.Latent {
			return fmt.Errorf("%w: LatentNorm len %d != latent %d", ErrKimiK31LatentMoEShape, len(w.LatentNorm), cfg.Latent)
		}
	}
	return nil
}

// KimiK31LatentRoutedInput applies the front: the hidden vector [Hidden] is
// down-projected into the latent [Latent] the routed experts consume.
func KimiK31LatentRoutedInput(cfg KimiK31LatentMoEConfig, w KimiK31LatentMoEWeights, x []float32) []float32 {
	if len(x) != cfg.Hidden {
		panic(fmt.Sprintf("model: kimi-k31 latent-moe: input length %d != Hidden %d", len(x), cfg.Hidden))
	}
	return kdaMatVecMul(w.DownProj, x, cfg.Latent, cfg.Hidden)
}

// KimiK31LatentNorm applies the latent RMSNorm (when configured) to the combined
// routed-expert output [Latent] and returns it. It is the ordering witness: the
// caller MUST reduce any tensor-parallel partial sums into this latent BEFORE
// calling — the norm of a sum is not the sum of the norms, so normalizing
// per-rank and reducing after would change the result. Pass the already-reduced
// latent here (a single-rank caller passes the expert output directly).
func KimiK31LatentNorm(cfg KimiK31LatentMoEConfig, w KimiK31LatentMoEWeights, latent []float32) []float32 {
	if len(latent) != cfg.Latent {
		panic(fmt.Sprintf("model: kimi-k31 latent-moe: latent length %d != Latent %d", len(latent), cfg.Latent))
	}
	if !cfg.LatentUseNorm {
		return latent
	}
	out := make([]float32, cfg.Latent)
	rmsnormInto(out, latent, w.LatentNorm, cfg.RMSNormEps)
	return out
}

// KimiK31LatentTail maps the normalized latent [Latent] back to hidden width
// [Hidden] through the up-projection.
func KimiK31LatentTail(cfg KimiK31LatentMoEConfig, w KimiK31LatentMoEWeights, latent []float32) []float32 {
	if len(latent) != cfg.Latent {
		panic(fmt.Sprintf("model: kimi-k31 latent-moe: latent length %d != Latent %d", len(latent), cfg.Latent))
	}
	return kdaMatVecMul(w.UpProj, latent, cfg.Hidden, cfg.Latent)
}

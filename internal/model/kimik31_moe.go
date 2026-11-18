package model

// kimik31_moe.go — the Kimi-K3.1 decoder layer's FFN/MoE block (ticket
// 1611-kimi-k31). This is the missing half of the L8 composite: the landed
// ForwardKimiK31Token ran only the layer's ATTENTION half (norm -> KDA/DSA ->
// residual add), so a K3.1 checkpoint's routed/shared expert compute — the bulk
// of the model — never executed. This file composes the block the way the L8
// keystone composes attention: over the landed oracles, in the reference's
// order, on synthetic weights so it is testable without a checkpoint.
//
// REFERENCE (direct port, pinned): SGLang a7d9f81ee2
// python/sglang/srt/models/kimi_k3.py, KimiK3DecoderLayer.forward +
// KimiK3MoE._forward_unfused. A K3.1 decoder layer is
//   attn_boundary.prepare  -> input_layernorm (pre-attention RMSNorm)
//   self_attn              -> KDA (69 layers) or MLA+gated-DSA (24 layers)
//   attn_boundary.finish   -> attention output + residual prefix
//   ffn_boundary.prepare   -> post_attention_layernorm
//   mlp                    -> KimiK3MoE (latent) or KimiK3MLP (layer 0 dense)
//   ffn_boundary.finish    -> mllp output + residual prefix
// The attention half is the landed composite; THIS file is the ffn_boundary half.
//
// The K3.1 MoE block itself (`_forward_unfused`, single-rank / no EP / no
// TP-shard reduce), in the reference's order:
//   1. router_logits = gate(x);  picks = TopK(noaux_tc: rank sigmoid(logit)+bias,
//      weight the unbiased sigmoid, renormalize, x routed_scaling_factor) — the
//      landed RouteKimiK31MoE oracle.
//   2. shared experts run on the ORIGINAL x (a KimiK3MLP whose intermediate is
//      moe_intermediate_size * num_shared_experts), in parallel with the routed.
//   3. the routed experts run in LATENT space: routed_input = down_proj(x);
//      per selected expert out_e = W2(SiTU(W1*routed_input, W3*routed_input))
//      (W1 gate, W3 up, W2 down — the K3 per-expert w1/w3/w2 layout);
//      the weighted sum (by the router weight) is reduced in latent space.
//   4. THE ORDERING CONTRACT: the latent RMSNorm is applied to the ALREADY
//      REDUCED latent (sum(norm(x_i)) != norm(sum(x_i))) — the landed
//      KimiK31LatentNorm takes the reduced latent by contract.
//   5. up_proj(latent_norm(reduced)) -> hidden width.
//   6. block output = routed_up_out + shared_out (the caller adds the residual
//      prefix; the latent-MoE layer owns that add in the reference).
//
// The DENSE layer (K3.1 first_k_dense_replace layer 0) takes the Experts == 0
// branch: no router, no routed experts, no latent front/tail — a single SiTU MLP
// at hidden width, the same KimiK3MLP the shared experts use.
//
// WHAT THIS OWNS: the FFN/MoE block composition for one token and the fail-closed
// param validation. WHAT THIS DOES NOT OWN: the expert GEMM kernels, the CUDA
// KDA/DSA kernels, the loader that fills these params from checkpoint tensors,
// and the attention-residual bank (its own leaf).

import "fmt"

// KimiK31ExpertWeights holds ONE routed expert's FFN weights in the K3.1
// per-expert w1/w3/w2 layout, all row-major [out x in] with
// y[o] = sum_i W[o*in+i]*x[i]. The experts compute in the LATENT width.
//
// W1 is the gate projection [In x Latent]; W3 the up projection
// [In x Latent]; W2 the down projection [Latent x In]. SiTU fuses W1/W3's
// outputs (gate*up) before W2, per the K3.1 activation.
type KimiK31ExpertWeights struct {
	W1 []float32 // [ExpertIntermediate x Latent]
	W3 []float32 // [ExpertIntermediate x Latent]
	W2 []float32 // [Latent x ExpertIntermediate]
}

// KimiK31MoEParams holds one K3.1 MoE layer's block weights. Every routed/shared
// expert reads it, so a loader builds one per MoE layer from the checkpoint's
// tensors; the block itself never reads a tensor store, so it is exercisable on
// synthetic weights (the same discipline the landed oracles use).
type KimiK31MoEParams struct {
	Hidden             int // model hidden width (down_proj input / up_proj output)
	Latent             int // routed_expert_hidden_size (the experts' working width)
	ExpertIntermediate int // moe_intermediate_size (one expert's FFN width, in latent)
	// SharedIntermediate is the always-on MLP's FFN width. On a MoE layer it is
	// the shared experts' combined width (moe_intermediate_size *
	// num_shared_experts); on a DENSE layer (Experts == 0, K3.1's
	// first_k_dense_replace layer 0) it is that layer's intermediate_size. The
	// reference builds the SAME KimiK3MLP for both, so they share this path.
	SharedIntermediate int
	LatentUseNorm      bool
	RMSNormEps         float32

	// Latent front/norm/tail (the landed KimiK31LatentMoE oracle's weights).
	DownProj   []float32 // [Latent x Hidden]
	UpProj     []float32 // [Hidden x Latent]
	LatentNorm []float32 // [Latent] (required when LatentUseNorm)

	// Router geometry + weights.
	Experts     int
	TopK        int
	RouteScale  float32
	Renormalize bool
	GroupCount  int
	WGate       []float32 // [Experts x Hidden]
	ScoreBias   []float32 // [Experts], optional (e_score_correction_bias)

	// Expert1[E] is routed expert E's w1/w3/w2; len == Experts.
	Expert1 []KimiK31ExpertWeights

	// Shared expert (a SiTU MLP at Hidden -> SharedIntermediate -> Hidden);
	// required when SharedIntermediate > 0.
	SharedWGate []float32 // [SharedIntermediate x Hidden]
	SharedWUp   []float32 // [SharedIntermediate x Hidden]
	SharedWDown []float32 // [Hidden x SharedIntermediate]

	// SiTU constants (K3.1: beta 4.0, linear_beta 25.0).
	SituBeta       float32
	SituLinearBeta float32
}

// ErrKimiK31MoEShape is the closed sentinel for fail-closed MoE block
// shape/domain validation failures.
var ErrKimiK31MoEShape = fmt.Errorf("model: kimi-k31 moe block shape mismatch")

// ValidateKimiK31MoEParams fails closed on any malformed geometry/weights,
// naming the offending field. Call it once when building params from tensors;
// the forward assumes validated params.
func ValidateKimiK31MoEParams(p *KimiK31MoEParams) error {
	if p == nil {
		return fmt.Errorf("%w: nil params", ErrKimiK31MoEShape)
	}
	if p.Hidden <= 0 || p.Latent <= 0 || p.ExpertIntermediate <= 0 {
		return fmt.Errorf("%w: bad geometry hidden=%d latent=%d inter=%d",
			ErrKimiK31MoEShape, p.Hidden, p.Latent, p.ExpertIntermediate)
	}
	if p.Experts == 0 {
		// The DENSE layer (K3.1 first_k_dense_replace layer 0): no router, no
		// routed experts, no latent front/tail — a single SiTU MLP at hidden
		// width, the same KimiK3MLP the shared experts use.
		if err := validateKimiK31DenseMLP(p); err != nil {
			return err
		}
		if p.SituBeta <= 0 {
			return fmt.Errorf("%w: situ beta %v must be > 0", ErrKimiK31MoEShape, p.SituBeta)
		}
		return nil
	}
	if p.TopK <= 0 || p.TopK > p.Experts {
		return fmt.Errorf("%w: router geometry experts=%d topk=%d", ErrKimiK31MoEShape, p.Experts, p.TopK)
	}
	if p.GroupCount < 1 {
		return fmt.Errorf("%w: group count %d must be >= 1", ErrKimiK31MoEShape, p.GroupCount)
	}
	if len(p.WGate) != p.Experts*p.Hidden {
		return fmt.Errorf("%w: WGate len %d != experts*hidden %d", ErrKimiK31MoEShape, len(p.WGate), p.Experts*p.Hidden)
	}
	if len(p.ScoreBias) != 0 && len(p.ScoreBias) != p.Experts {
		return fmt.Errorf("%w: ScoreBias len %d != experts %d", ErrKimiK31MoEShape, len(p.ScoreBias), p.Experts)
	}
	// The latent front/norm/tail are the landed oracle's; its validator owns
	// their shape contract.
	lf := KimiK31LatentMoEConfig{
		Hidden:             p.Hidden,
		Latent:             p.Latent,
		ExpertIntermediate: p.ExpertIntermediate,
		LatentUseNorm:      p.LatentUseNorm,
		RMSNormEps:         p.RMSNormEps,
	}
	lw := KimiK31LatentMoEWeights{DownProj: p.DownProj, UpProj: p.UpProj, LatentNorm: p.LatentNorm}
	if err := ValidateKimiK31LatentMoE(lf, lw); err != nil {
		return fmt.Errorf("%w: latent front/norm/tail: %v", ErrKimiK31MoEShape, err)
	}
	if len(p.Expert1) != p.Experts {
		return fmt.Errorf("%w: %d expert weight sets for %d experts", ErrKimiK31MoEShape, len(p.Expert1), p.Experts)
	}
	for e := range p.Expert1 {
		if len(p.Expert1[e].W1) != p.ExpertIntermediate*p.Latent {
			return fmt.Errorf("%w: expert %d W1 len %d != inter*latent %d", ErrKimiK31MoEShape, e, len(p.Expert1[e].W1), p.ExpertIntermediate*p.Latent)
		}
		if len(p.Expert1[e].W3) != p.ExpertIntermediate*p.Latent {
			return fmt.Errorf("%w: expert %d W3 len %d != inter*latent %d", ErrKimiK31MoEShape, e, len(p.Expert1[e].W3), p.ExpertIntermediate*p.Latent)
		}
		if len(p.Expert1[e].W2) != p.Latent*p.ExpertIntermediate {
			return fmt.Errorf("%w: expert %d W2 len %d != latent*inter %d", ErrKimiK31MoEShape, e, len(p.Expert1[e].W2), p.Latent*p.ExpertIntermediate)
		}
	}
	if p.SharedIntermediate > 0 {
		if err := validateKimiK31DenseMLP(p); err != nil {
			return err
		}
	}
	if p.SituBeta <= 0 {
		return fmt.Errorf("%w: situ beta %v must be > 0", ErrKimiK31MoEShape, p.SituBeta)
	}
	return nil
}

// validateKimiK31DenseMLP shape-checks the shared/dense SiTU MLP at hidden width
// (SharedWGate/Up are [SharedIntermediate x Hidden], SharedWDown is
// [Hidden x SharedIntermediate]). It is shared by the MoE branch's always-on
// shared experts and the Experts == 0 dense-layer branch, which the reference
// builds as the same KimiK3MLP.
func validateKimiK31DenseMLP(p *KimiK31MoEParams) error {
	if p.SharedIntermediate <= 0 {
		return fmt.Errorf("%w: SharedIntermediate %d must be > 0", ErrKimiK31MoEShape, p.SharedIntermediate)
	}
	if len(p.SharedWGate) != p.SharedIntermediate*p.Hidden {
		return fmt.Errorf("%w: SharedWGate len %d != shared_inter*hidden %d", ErrKimiK31MoEShape, len(p.SharedWGate), p.SharedIntermediate*p.Hidden)
	}
	if len(p.SharedWUp) != p.SharedIntermediate*p.Hidden {
		return fmt.Errorf("%w: SharedWUp len %d != shared_inter*hidden %d", ErrKimiK31MoEShape, len(p.SharedWUp), p.SharedIntermediate*p.Hidden)
	}
	if len(p.SharedWDown) != p.Hidden*p.SharedIntermediate {
		return fmt.Errorf("%w: SharedWDown len %d != hidden*shared_inter %d", ErrKimiK31MoEShape, len(p.SharedWDown), p.Hidden*p.SharedIntermediate)
	}
	return nil
}

// ForwardKimiK31MoEToken runs one K3.1 MoE layer's FFN block over one token's
// post-attention RMSNorm output xn ([Hidden]) and returns the hidden-width block
// output ([Hidden]) — the routed + shared expert contribution the caller adds to
// the residual prefix (the layer's own ffn_boundary.finish). It calls the landed
// oracles (RouteKimiK31MoE, the latent front/norm/tail, SituAndMul) rather than
// re-deriving them, so its answer cannot drift from theirs.
func ForwardKimiK31MoEToken(p *KimiK31MoEParams, xn []float32) []float32 {
	if p == nil {
		panic("model: kimi-k31 moe: nil params")
	}
	if len(xn) != p.Hidden {
		panic(fmt.Sprintf("model: kimi-k31 moe: input length %d != Hidden %d", len(xn), p.Hidden))
	}

	// Dense layer (K3.1 first_k_dense_replace layer 0): no router and no routed
	// experts — a single SiTU MLP at hidden width, the reference's KimiK3MLP.
	if p.Experts == 0 {
		return kimiK31SharedExpert(p, xn)
	}

	// 1. Router: fp32 logits -> noaux_tc selection (bias-ranked) with unbiased,
	// renormalized, scaled weights.
	logits := kdaMatVecMul(p.WGate, xn, p.Experts, p.Hidden)
	picks, err := RouteKimiK31MoE(logits, p.ScoreBias, KimiK31RouterConfig{
		Experts:     p.Experts,
		TopK:        p.TopK,
		RouteScale:  p.RouteScale,
		Renormalize: p.Renormalize,
		GroupCount:  p.GroupCount,
	})
	if err != nil {
		// Validated params cannot produce a router refusal (finite weights,
		// in-range geometry); a refusal here is an unrepresentable state.
		panic(fmt.Sprintf("model: kimi-k31 moe: router refused validated params: %v", err))
	}

	// 2. Latent front: the routed experts read the down-projected latent.
	lf := KimiK31LatentMoEConfig{
		Hidden:             p.Hidden,
		Latent:             p.Latent,
		ExpertIntermediate: p.ExpertIntermediate,
		LatentUseNorm:      p.LatentUseNorm,
		RMSNormEps:         p.RMSNormEps,
	}
	lw := KimiK31LatentMoEWeights{DownProj: p.DownProj, UpProj: p.UpProj, LatentNorm: p.LatentNorm}
	routedInput := KimiK31LatentRoutedInput(lf, lw, xn)

	// 3. Routed experts in latent space, weighted by the router weight.
	//    Single rank: the summed latent IS the reduced latent the norm contract
	//    requires (no TP partial sums to reduce).
	//    The mean of an empty sum is the zero latent (an empty selection cannot
	//    happen with TopK >= 1 and a validated router, but the buffer is sized
	//    here regardless).
	acc := make([]float32, p.Latent)
	gate := make([]float32, p.ExpertIntermediate)
	up := make([]float32, p.ExpertIntermediate)
	for _, pk := range picks {
		e := &p.Expert1[pk.Expert]
		projectExpertGateUp(e, routedInput, gate, up, p.ExpertIntermediate, p.Latent)
		SituAndMul(gate, up, p.SituBeta, p.SituLinearBeta)
		down := kdaMatVecMul(e.W2, gate, p.Latent, p.ExpertIntermediate)
		w := pk.Weight
		for i := range acc {
			acc[i] += w * down[i]
		}
	}

	// 4. Reduce-before-norm: normalise the already-reduced latent.
	normed := KimiK31LatentNorm(lf, lw, acc)

	// 5. Up-projection tail -> hidden width.
	out := KimiK31LatentTail(lf, lw, normed)

	// 6. Shared experts run on the ORIGINAL x, added after the routed tail.
	if p.SharedIntermediate > 0 {
		shared := kimiK31SharedExpert(p, xn)
		for i := range out {
			out[i] += shared[i]
		}
	}
	return out
}

// projectExpertGateUp fills gate = W1 x and up = W3 x for one expert, in latent
// space. It exists so the per-expert projection allocates nothing on the hot
// loop (the caller owns the buffers) — the routed block runs TopK projections
// per token.
func projectExpertGateUp(e *KimiK31ExpertWeights, x, gate, up []float32, inter, latent int) {
	for i := 0; i < inter; i++ {
		var g, u float32
		row := i * latent
		for j := 0; j < latent; j++ {
			xj := x[j]
			g += e.W1[row+j] * xj
			u += e.W3[row+j] * xj
		}
		gate[i] = g
		up[i] = u
	}
}

// kimiK31SharedExpert is the always-on shared-expert MLP at hidden width: a
// SiTU gate/up pair at SharedIntermediate followed by the down projection. It
// mirrors KimiK3MLP with hidden_act == "situ" (the fused shared-expert width the
// reference builds as moe_intermediate_size * num_shared_experts).
func kimiK31SharedExpert(p *KimiK31MoEParams, xn []float32) []float32 {
	gate := kdaMatVecMul(p.SharedWGate, xn, p.SharedIntermediate, p.Hidden)
	up := kdaMatVecMul(p.SharedWUp, xn, p.SharedIntermediate, p.Hidden)
	SituAndMul(gate, up, p.SituBeta, p.SituLinearBeta)
	return kdaMatVecMul(p.SharedWDown, gate, p.Hidden, p.SharedIntermediate)
}

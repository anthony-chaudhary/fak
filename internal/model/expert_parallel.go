package model

import "fmt"

// expert_parallel.go — EXPERT-PARALLEL (EP) sharding for the MoE FFN: the GLM-5.2
// multi-GPU sub-lever forwardTPSupported() names but does not yet implement
// ("ForwardTP does not yet shard MoE FFN (expert-parallel is a separate sub-lever)",
// tensor_parallel_forward.go). It is the MoE counterpart of tensor_parallel.go's
// TensorParallelFFN, and lands with the SAME numeric discipline.
//
// WHY EP IS THE RIGHT MULTI-GPU DECOMPOSITION FOR GLM-5.2. A 753B glm_moe_dsa model's
// parameters are dominated by its routed experts (the per-layer mlp.experts.<e>.* GEMMs);
// the dense projections, router, and MLA+DSA attention are tiny by comparison. Head-
// parallel tensor parallelism is wrong for GLM's MLA (its latent KV is shared across
// heads), and column/row-parallel sharding of a single expert pays a collective on a
// matmul that already fits one device. The natural decomposition is to PARTITION THE
// EXPERTS across ranks: each rank holds a contiguous band of experts resident (≈ the
// model's parameter bulk / R per GPU), the router runs replicated on every rank (it is
// cheap and picks the SAME top-k), and each rank contributes only the picks it OWNS. The
// per-rank [H] residual partials are summed by exactly ONE AllReduceSum — far less traffic
// than a per-expert all-to-all, because experts are independent (no shared intermediate to
// gather, unlike the dense FFN's column/row split).
//
// THE NUMERIC CONTRACT (identical to tensor_parallel.go's row-parallel rung):
//
//   - BIT-EXACT at ranks=1. A 1-rank plan owns all experts in one band, so its single
//     partial accumulates the gate-weighted expert sum in the EXACT expert-ascending order
//     glmMoeFFN uses (glmRoute returns picks sorted by expert index), and AllReduceSum of
//     one part is the identity. So ExpertParallelDelta(ranks=1) == the monolith routed delta
//     at max|Δ|=0 — the "sharding-invariant vs the single-device path" rung, witnessable
//     with no multi-GPU hardware (the EP twin of ForwardTP(ranks=1)==Forward).
//
//   - REASSOCIATED across ranks. With R>1 the reduction becomes (band-0 ascending sum) +
//     (band-1 ascending sum) + … — a regrouping of the monolith's strictly-ascending sum.
//     Float addition is not associative, so this drifts ~1e-6 vs the monolith (the same
//     fdot non-associativity parallel.go documents), and is NOT claimed bit-exact across
//     ranks. It IS bit-exact vs expertParallelReference (the rank-order sum of the identical
//     per-rank partials), which is the invariant the gate pins — proving the collective
//     reduces in rank order, not that it adds something beyond the unavoidable round-off.
//
// The placement is a PURE routing decision over the SAME arithmetic: a pick runs through
// expertSwiGLU exactly as glmMoeFFN runs it, on whichever rank owns its expert. So the only
// thing EP can get wrong is which rank owns which expert (pinned by ExpertParallelPlan's
// tiling) and the reduction order (pinned by the reference) — never the expert math.
//
// SCOPE — honest, in-pattern. Like tensor_parallel.go before it, this lands the PROVEN
// primitive; it is NOT yet wired into the live glmMoeFFN forward, and the Collective it
// reduces through is LocalCollective (the single-box, bit-exact default) / BackendCollective
// on cpu-ref. A real cross-DEVICE reduction needs the NCCL/RCCL compute.CollectiveBackend
// rung (hardware-gated, native-753b-track-staged-plan.md P3) — only with that may a
// multi-GPU EP serve be CLAIMED. This file de-risks everything above that device line.
//
// It shards the STANDARD SwiGLU expert (GLM-5.2 / Mixtral / Qwen3-MoE). It FAILS CLOSED on
// gptoss (whose experts use the distinct expertGPTOSS form), and the GLM full-FFN wrapper
// fails closed on the qwen3.5 singular gated shared expert — refusing rather than silently
// serving the wrong arithmetic, mirroring forwardTPSupported's fail-closed-on-unsupported-arch.

// ExpertParallelPlan tiles the experts [0,numExperts) across `ranks` near-even contiguous
// bands — the expert-parallel placement plan. It is NewTPPlan specialized and named for the
// expert axis: rank r owns the experts in p.Shards[r].[Lo,Hi), so the resident weight a rank
// holds is that band's experts' gate/up/down projections (≈ the model's expert bulk / R).
// Contiguous ascending bands are load-bearing for the ranks=1 bit-exactness: glmRoute returns
// picks in expert-ascending order, so a rank's owned picks are a contiguous ascending run and
// the reduction regrouping is purely associative-reorder, never a reorder of unrelated terms.
//
// It fails closed exactly as NewTPPlan: ranks must be in [1,numExperts] (a rank with no
// experts has no work the collective can place).
func ExpertParallelPlan(numExperts, ranks int) (TPPlan, error) {
	if numExperts <= 0 {
		return TPPlan{}, fmt.Errorf("model: ExpertParallelPlan numExperts = %d, want > 0 (not an MoE config?)", numExperts)
	}
	return NewTPPlan(numExperts, ranks)
}

// expertParallelPartials computes each rank's [H] gate-weighted residual partial for the
// expert-parallel MoE FFN of layer l: rank r sums w·expertSwiGLU(e) over the routed picks
// whose expert e ∈ its band [Lo,Hi), iterating `picks` in their given order (glmRoute's
// expert-ascending order) so each band's partial matches the monolith's ascending sub-sum.
// A rank that owns none of the picked experts contributes an all-zero [H] partial (correct —
// it computes nothing and adds nothing under AllReduceSum). plan.Dim must == NumExperts.
//
// Factored out so ExpertParallelDelta (which reduces via the Collective) and
// expertParallelReference (which sums in rank order directly) share the IDENTICAL partial
// computation and differ ONLY in the reduction — the contract the bit-exact reference pins.
func (m *Model) expertParallelPartials(layer int, xn any, mat matKernel, picks []routePick, plan TPPlan) ([][]float32, error) {
	if err := plan.Validate(); err != nil {
		return nil, err
	}
	if plan.Dim != m.Cfg.NumExperts {
		return nil, fmt.Errorf("model: expertParallelPartials plan.Dim = %d, want NumExperts = %d", plan.Dim, m.Cfg.NumExperts)
	}
	// Fail closed on gptoss: its experts use the expertGPTOSS form (clamped gate/up, (up+1)*glu)
	// which moeFFN dispatches but this path does not — using expertSwiGLU here would silently
	// mis-compute. EP targets the standard SwiGLU expert (GLM-5.2 / Mixtral / Qwen3-MoE); a
	// gptoss EP path is a separate sub-lever, so refuse rather than serve the wrong arithmetic.
	if m.Cfg.isGPTOSS() {
		return nil, fmt.Errorf("model: expert-parallel does not shard gptoss experts (they use the expertGPTOSS form, not expertSwiGLU — a separate sub-lever)")
	}
	H := m.Cfg.HiddenSize
	parts := make([][]float32, len(plan.Shards))
	for r, s := range plan.Shards {
		p := make([]float32, H)
		for _, pk := range picks {
			if pk.expert < s.Lo || pk.expert >= s.Hi {
				continue
			}
			out := expertSwiGLU(m, layer, pk.expert, xn, mat)
			for i := 0; i < H; i++ {
				p[i] += pk.weight * out[i]
			}
		}
		parts[r] = p
	}
	return parts, nil
}

// ExpertParallelDelta computes the ROUTED-expert residual delta for layer l with the experts
// sharded across `plan`, reduced through the Collective (one AllReduceSum). It is bit-exact
// vs the monolith routed loop at ranks=1 and matches it within the AllReduce reassociation
// round-off across ranks (see the file header). A nil collective defaults to LocalCollective.
//
// It returns ONLY the routed contribution; the always-on shared-expert term (if any) is added
// by expertParallelGLMMoEDelta after the reduce, matching glmMoeFFN's routed-then-shared order.
func (m *Model) ExpertParallelDelta(layer int, xn any, mat matKernel, picks []routePick, plan TPPlan, coll Collective) ([]float32, error) {
	parts, err := m.expertParallelPartials(layer, xn, mat, picks, plan)
	if err != nil {
		return nil, err
	}
	if coll == nil {
		coll = LocalCollective{}
	}
	return coll.AllReduceSum(parts)
}

// expertParallelReference is the rank-order bit-exact oracle for ExpertParallelDelta: the
// identical per-rank partials, summed in rank order directly (sumPartialsRankOrder) rather
// than through the Collective. Pinning ExpertParallelDelta == this at max|Δ|=0 proves the
// collective reduces the expert partials in rank order — the contract the cross-rank result
// depends on — independently of the loose vs-monolith round-off bound.
func (m *Model) expertParallelReference(layer int, xn any, mat matKernel, picks []routePick, plan TPPlan) ([]float32, error) {
	parts, err := m.expertParallelPartials(layer, xn, mat, picks, plan)
	if err != nil {
		return nil, err
	}
	return sumPartialsRankOrder(parts), nil
}

// expertParallelGLMMoEDelta is the full glmMoeFFN twin under expert parallelism: the routed
// delta reduced across expert shards, plus the always-on GLM shared-expert term added ONCE
// after the reduce (the shared expert is replicated, not sharded — it fires every token). It
// is bit-exact vs glmMoeFFN's delta at ranks=1: the reduced routed sum equals the monolith
// routed loop, and the shared add is the identical glmSharedExperts call in the identical
// (routed-then-shared) order. A nil collective defaults to LocalCollective.
func (m *Model) expertParallelGLMMoEDelta(layer int, xn any, mat matKernel, picks []routePick, plan TPPlan, coll Collective) ([]float32, error) {
	// This wrapper handles only the GLM plural shared expert (mlp.shared_experts.*). A qwen3.5
	// model carries a DIFFERENT singular gated shared expert (mlp.shared_expert.* +
	// mlp.shared_expert_gate.weight) that moeFFN adds and this wrapper does not — fail closed so
	// it never silently drops that term (the routed ExpertParallelDelta alone stays correct).
	if m.has(qwen35SharedExpertName(layer, "gate.weight")) {
		return nil, fmt.Errorf("model: expertParallelGLMMoEDelta is the glmMoeFFN twin and does not add the qwen3.5 singular gated shared expert; use ExpertParallelDelta for the routed part")
	}
	delta, err := m.ExpertParallelDelta(layer, xn, mat, picks, plan, coll)
	if err != nil {
		return nil, err
	}
	if m.Cfg.NSharedExperts > 0 && m.hasWeight(layerName(layer, "mlp.shared_experts.gate_proj.weight")) {
		shared := glmSharedExperts(m, layer, xn, mat)
		for i := range delta {
			delta[i] += shared[i]
		}
	}
	return delta, nil
}

// SetExpertParallelRanks records the expert-parallel rank count the live MoE forward shards the
// routed delta across (Model.epRanks). 0 or 1 leaves the forward on the monolith glmMoeFFN (the
// no-op default); >1 makes ffnForLayer dispatch routed glm_moe_dsa layers through glmMoeEPFFN.
// It is the setter the serve flag (--expert-parallel N) drives; the ranks=1 path is bit-exact vs
// the monolith and needs no device — ranks>1 carry a real multi-GPU claim only once the device
// NCCL CollectiveBackend reduces the per-rank partials across GPUs.
func (m *Model) SetExpertParallelRanks(ranks int) { m.epRanks = ranks }

// ExpertParallelRanks reports the configured expert-parallel rank count (0/1 == monolith).
func (m *Model) ExpertParallelRanks() int { return m.epRanks }

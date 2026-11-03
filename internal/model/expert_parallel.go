package model

import "fmt"

// expert_parallel.go — EXPERT-PARALLEL (EP) sharding for the MoE FFN: the MoE counterpart of
// tensor_parallel.go's TensorParallelFFN, with the same numeric discipline.
//
// EP partitions the experts across ranks instead of sharding a single expert: a 753B
// glm_moe_dsa model's bytes are dominated by routed experts, head-parallel TP is wrong for
// GLM's MLA (latent KV is shared across heads), and column/row-sharding one expert pays a
// collective on a matmul that already fits one device. Each rank holds a contiguous band of
// experts, the router runs replicated on every rank, and the per-rank [H] partials are summed
// by exactly ONE AllReduceSum — far less traffic than a per-expert all-to-all, since experts
// are independent.
//
// Numeric contract:
//   - BIT-EXACT at ranks=1: the single band accumulates in the exact expert-ascending order
//     glmMoeFFN uses, and AllReduceSum of one part is the identity.
//   - REASSOCIATED across ranks: R>1 regroups the monolith's ascending sum, drifting ~1e-6
//     from it (float non-associativity); it is bit-exact vs expertParallelReference (the
//     rank-order sum of the identical partials), which is the invariant the gate pins.
//
// The placement is a pure routing decision over the same arithmetic: a pick runs through
// expertSwiGLU exactly as glmMoeFFN runs it, on whichever rank owns its expert.
//
// Wired into the live forward: ffnForLayer returns glmMoeEPFFN whenever epRanks > 1
// (`fak serve --expert-parallel N`). The default Collective is LocalCollective; a real
// cross-device reduction needs the NCCL/RCCL compute.CollectiveBackend rung (hardware-gated,
// native-753b-track-staged-plan.md P3) before a multi-GPU EP serve may be claimed. Shards the
// standard SwiGLU expert; fails closed on gptoss and on the qwen3.5 singular gated shared
// expert.

// ExpertParallelPlan tiles the experts [0,numExperts) across `ranks` near-even contiguous
// bands: rank r owns the experts in p.Shards[r].[Lo,Hi). Contiguous ascending bands are
// load-bearing for the ranks=1 bit-exactness: glmRoute returns picks in expert-ascending
// order, so the reduction regrouping is purely an associative reorder. Fails closed exactly
// as NewTPPlan.
func ExpertParallelPlan(numExperts, ranks int) (TPPlan, error) {
	if numExperts <= 0 {
		return TPPlan{}, fmt.Errorf("model: ExpertParallelPlan numExperts = %d, want > 0 (not an MoE config?)", numExperts)
	}
	return NewTPPlan(numExperts, ranks)
}

// expertParallelPartials computes each rank's [H] gate-weighted residual partial: rank r sums
// w·expertSwiGLU(e) over the picks whose expert falls in its band, in the given (expert-
// ascending) order, so each band's partial matches the monolith's ascending sub-sum. Shared by
// ExpertParallelDelta and expertParallelReference so they differ only in the reduction.
func (m *Model) expertParallelPartials(layer int, xn any, mat matKernel, picks []routePick, plan TPPlan) ([][]float32, error) {
	if err := plan.Validate(); err != nil {
		return nil, err
	}
	if plan.Dim != m.Cfg.NumExperts {
		return nil, fmt.Errorf("model: expertParallelPartials plan.Dim = %d, want NumExperts = %d", plan.Dim, m.Cfg.NumExperts)
	}
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

// expertParallelRankPartial computes only rank r's [H] partial — the single-band primitive a
// sharded (multi-process) EP serve needs, where this process's Model holds only
// plan.Shards[rank]'s experts. It is the one-band slice of expertParallelPartials, so the
// per-rank partials are byte-identical to the all-band path's parts[rank]. A sharded rank must
// fail closed on an absent band expert: mat.mul panics on a missing resident weight, and a
// monolith fallback would touch peer experts this rank does not hold.
func (m *Model) expertParallelRankPartial(layer int, xn any, mat matKernel, picks []routePick, plan TPPlan, rank int) ([]float32, error) {
	if err := plan.Validate(); err != nil {
		return nil, err
	}
	if plan.Dim != m.Cfg.NumExperts {
		return nil, fmt.Errorf("model: expertParallelRankPartial plan.Dim = %d, want NumExperts = %d", plan.Dim, m.Cfg.NumExperts)
	}
	if m.Cfg.isGPTOSS() {
		return nil, fmt.Errorf("model: expert-parallel does not shard gptoss experts (they use the expertGPTOSS form, not expertSwiGLU — a separate sub-lever)")
	}
	if rank < 0 || rank >= len(plan.Shards) {
		return nil, fmt.Errorf("model: expertParallelRankPartial rank %d outside [0,%d)", rank, len(plan.Shards))
	}
	H := m.Cfg.HiddenSize
	s := plan.Shards[rank]
	p := make([]float32, H)
	owned := make([]routePick, 0, len(picks))
	for _, pk := range picks {
		if pk.expert < s.Lo || pk.expert >= s.Hi {
			continue
		}
		if !m.hasWeight(expertName(layer, pk.expert, "gate_proj.weight")) {
			return nil, fmt.Errorf("model: expert-parallel rank %d missing resident band expert %d at layer %d (sharded load did not admit [%d,%d))", rank, pk.expert, layer, s.Lo, s.Hi)
		}
		owned = append(owned, pk)
	}
	// Without this capability gate the host batch path silently bypasses a resident
	// routed-kquant backend.
	if xf, ok := xn.([]float32); ok && len(owned) > 0 &&
		!routedExpertKQuantActive(mat) && m.hostBatchedGLMExperts(layer, xf, p, owned) {
		return p, nil
	}
	for _, pk := range owned {
		out := expertSwiGLU(m, layer, pk.expert, xn, mat)
		for i := 0; i < H; i++ {
			p[i] += pk.weight * out[i]
		}
	}
	return p, nil
}

// ExpertParallelDelta computes the routed-expert residual delta with the experts sharded across
// plan, reduced through the Collective (nil defaults to LocalCollective). It returns only the
// routed contribution; the shared-expert term is added by expertParallelGLMMoEDelta after the
// reduce, matching glmMoeFFN's routed-then-shared order.
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
// identical partials summed in rank order directly. Pinning ExpertParallelDelta == this at
// max|Δ|=0 proves the collective reduces in rank order.
func (m *Model) expertParallelReference(layer int, xn any, mat matKernel, picks []routePick, plan TPPlan) ([]float32, error) {
	parts, err := m.expertParallelPartials(layer, xn, mat, picks, plan)
	if err != nil {
		return nil, err
	}
	return sumPartialsRankOrder(parts), nil
}

// expertParallelGLMMoEDelta is the full glmMoeFFN twin under expert parallelism: the routed
// delta reduced across expert shards, plus the GLM shared-expert term added once after the
// reduce (the shared expert is replicated, not sharded). Bit-exact vs glmMoeFFN at ranks=1.
func (m *Model) expertParallelGLMMoEDelta(layer int, xn any, mat matKernel, picks []routePick, plan TPPlan, coll Collective) ([]float32, error) {
	if m.has(qwen35SharedExpertName(layer, "gate.weight")) {
		return nil, fmt.Errorf("model: expertParallelGLMMoEDelta is the glmMoeFFN twin and does not add the qwen3.5 singular gated shared expert; use ExpertParallelDelta for the routed part")
	}
	delta, err := m.ExpertParallelDelta(layer, xn, mat, picks, plan, coll)
	if err != nil {
		return nil, err
	}
	return m.addGLMSharedExpertsOnce(layer, xn, mat, delta), nil
}

// addGLMSharedExpertsOnce adds the replicated GLM shared-expert term to the reduced routed
// delta, in place, ONCE after the reduce: the shared expert is replicated on every rank, so
// adding it before the reduce would sum it N times. No-op without mlp.shared_experts.* weights.
func (m *Model) addGLMSharedExpertsOnce(layer int, xn any, mat matKernel, delta []float32) []float32 {
	if m.Cfg.NSharedExperts > 0 && m.hasWeight(layerName(layer, "mlp.shared_experts.gate_proj.weight")) {
		shared := glmSharedExperts(m, layer, xn, mat)
		for i := range delta {
			delta[i] += shared[i]
		}
	}
	return delta
}

// expertParallelRankLocalGLMMoEDelta is the sharded (multi-process) twin of
// expertParallelGLMMoEDelta: this process computes only its own band partial and reduces it
// across the process group through coll, so no process holds the full expert set. Bit-exact vs
// expertParallelGLMMoEDelta at the same rank count when each rank holds exactly
// plan.Shards[rank]. The GLM shared expert is replicated on every rank, so it is added once
// after the reduce; the caller hard-fails on the returned error rather than falling back to the
// monolith, which would touch peer experts this rank does not hold.
func (m *Model) expertParallelRankLocalGLMMoEDelta(layer int, xn any, mat matKernel, picks []routePick, plan TPPlan, rank int, coll Collective) ([]float32, error) {
	if m.has(qwen35SharedExpertName(layer, "gate.weight")) {
		return nil, fmt.Errorf("model: expertParallelRankLocalGLMMoEDelta is the glmMoeFFN twin and does not add the qwen3.5 singular gated shared expert")
	}
	part, err := m.expertParallelRankPartial(layer, xn, mat, picks, plan, rank)
	if err != nil {
		return nil, err
	}
	if coll == nil {
		coll = LocalCollective{}
	}
	// Each rank contributes ONLY its own part; the collective (a distCommCollective over the
	// process group, or the LocalCollective identity at ranks=1) sums them in rank order.
	delta, err := coll.AllReduceSum([][]float32{part})
	if err != nil {
		return nil, err
	}
	return m.addGLMSharedExpertsOnce(layer, xn, mat, delta), nil
}

// SetExpertParallelRanks records the expert-parallel rank count (Model.epRanks): 0/1 leaves the
// forward on the monolith glmMoeFFN; >1 makes ffnForLayer dispatch through glmMoeEPFFN. Driven
// by `--expert-parallel N`.
func (m *Model) SetExpertParallelRanks(ranks int) { m.epRanks = ranks }

// ExpertParallelRanks reports the configured expert-parallel rank count (0/1 == monolith).
func (m *Model) ExpertParallelRanks() int { return m.epRanks }

// SetExpertParallelRank marks this Model as a sharded EP rank holding only plan.Shards[rank]'s
// experts; the live forward then computes only this rank's band and reduces across the process
// group. Flips epRankSet, so the rank-local path is entered only on an explicit sharded serve.
// Pair with SetExpertParallelRanks (world size) and SetExpertParallelCollective (process group).
func (m *Model) SetExpertParallelRank(rank int) { m.epRank, m.epRankSet = rank, true }

// expertParallelRankLocal reports this Model's sharded EP rank and whether it was set. Unset
// (the default) keeps the live forward on the single-process all-band path.
func (m *Model) expertParallelRankLocal() (int, bool) { return m.epRank, m.epRankSet }

// IsExpertParallelRankLocal reports whether this Model is a sharded EP rank. The gateway checks
// it to avoid clobbering the sharded serve's process-group collective with a single-process
// device/Local collective, which would break the cross-process reduce.
func (m *Model) IsExpertParallelRankLocal() bool { return m.epRankSet }

// SetExpertParallelCollective records the Collective the EP decode path reduces the per-rank
// expert partials through (Model.epColl); nil restores the bit-exact LocalCollective default.
// The serve wires a BackendCollective over the device NCCL backend for multi-GPU serves.
func (m *Model) SetExpertParallelCollective(coll Collective) { m.epColl = coll }

// expertParallelCollective returns the Collective the EP decode path reduces through: the one
// the serve wired (m.epColl) or the single-box, bit-exact LocalCollective when none is set.
func (m *Model) expertParallelCollective() Collective {
	if m.epColl != nil {
		return m.epColl
	}
	return LocalCollective{}
}

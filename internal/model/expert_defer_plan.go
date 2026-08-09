package model

import (
	"fmt"
	"sort"
)

// expert_defer_plan.go — the router-weight partition half of the cross-layer deferred-expert
// CPU/GPU pipeline (#5239, child of the ktransformers study epic #3900).
//
// The mechanism being borrowed (inspire-only, Apache-2.0 -> Apache-2.0, clean-room, no bytes):
// ktransformers hides CPU cold-expert latency behind GPU work by running each token's
// LOWEST-router-weight experts one layer LATE and summing their partial onto the already-emitted
// token vector at layer L+1. The knob is `max_deferred`: `protected_k = num_experts_per_tok -
// max_deferred` experts are protected and run at their own layer, the rest are deferred. Upstream
// bands 1-4 deferred as a good latency/quality balance and 5-7 as noticeable accuracy loss — an
// explicit quality-for-latency dial for the VRAM-starved consumer-GPU + big-MoE world, where the
// GPU would otherwise stall on the host expert tail.
//
// fak's expert seam today overlaps DISK, not COMPUTE: splitKernel.mul (moe_offload.go) routes each
// weight to the host or device sub-kernel and runs it SYNCHRONOUSLY, and expert_readahead.go's
// MADV_WILLNEED prefetch overlaps mmap page-ins with compute. Nothing defers expert COMPUTE across
// a layer boundary.
//
// This file is the PURE half of that borrow, and only that half: given one token's router picks and
// a deferral budget it computes the immediate/deferred partition. No clock, no stream, no device, no
// cross-layer state — the partition is a deterministic function of the router weights alone, so it
// is witnessable against a fixture off the serve path. It mirrors how the residency gauges
// (#3902/#3901) landed as pure functions before any serve wiring.
//
// Deliberately NOT here (the fenced follow-on, which needs cross-layer session state and a device
// backend to mean anything): the splitKernel double-buffer where immediate work writes one output
// slot while deferred work writes the next, the layer-late incremental reconcile that SUMS the
// deferred partial onto the emitted vector rather than recomputing it, and the pipeline-depth valve
// that tolerates one in-flight deferred task across a sync boundary. Each of those is DRIVEN by the
// partition, which is why the partition lands first and alone. Nothing here is on the serve path
// yet, so today's forward is byte-for-byte unchanged.

// expertDeferProtectedK resolves the deferral grade: how many of a token's k routed experts stay
// PROTECTED (run at their own layer) when maxDeferred of them may run one layer late. It is
// ktransformers' `protected_k = num_experts_per_tok - max_deferred`, plus fak's range refusal.
//
// maxDeferred outside [0, k-1] is REFUSED rather than clamped, matching ResolveExpertSpill
// (expert_spill_fit.go): an operator's typo must not silently degrade into a routing nobody asked
// for. The upper bound is k-1 and not k because at least one expert must run immediately — with
// protected_k == 0 the token gets no expert contribution at its own layer and there is no immediate
// work left for the deferred host tail to hide behind, which is the entire point of the pipeline.
//
// maxDeferred == 0 is the identity grade (protectedK == k, nothing deferred) and is the default: it
// keeps today's synchronous forward exactly as it is.
func expertDeferProtectedK(k, maxDeferred int) (int, error) {
	if k < 1 {
		return 0, fmt.Errorf("model: expert defer k = %d, want >= 1", k)
	}
	if maxDeferred < 0 || maxDeferred > k-1 {
		return 0, fmt.Errorf("model: expert defer maxDeferred = %d out of range [0, %d]", maxDeferred, k-1)
	}
	return k - maxDeferred, nil
}

// expertDeferPartition is one token's routed top-k split into the work that runs at its own layer and the
// work that runs one layer late. Immediate and Deferred are disjoint and their concatenation is a
// permutation of the input picks: a deferred expert is POSTPONED, never dropped, which is what makes
// the layer-late reconcile a SUM onto the emitted vector rather than a recompute.
type expertDeferPartition struct {
	// Immediate is the protected set — the highest-router-weight picks, ordered by descending
	// weight (ties by ascending expert index, torch.topk's order).
	Immediate []routePick
	// Deferred is the complement — the lowest-router-weight picks, in the same order. It is nil
	// when nothing is deferred, so the identity grade allocates nothing.
	Deferred []routePick
}

// partitionDeferredExperts partitions ONE token's router picks into the immediate and deferred sets for a
// maxDeferred budget. picks is the router's top-k output for that token (route, moe.go), so len(picks)
// is num_experts_per_tok; maxDeferred is validated against it by expertDeferProtectedK and an
// out-of-range budget is returned as an error, never clamped.
//
// The ranking is computed here rather than trusting the caller's order: picks by (weight DESC,
// expert index ASC) — torch.topk's tie-break, the same one route() selects with — so the partition
// is a deterministic function of the pick SET and cannot silently follow whatever order a caller
// happened to build. A canonically ordered input therefore splits as a plain prefix/suffix, which is
// exactly upstream's `torch.topk` on the routing scores with the remainder masked out.
//
// The identity grade (maxDeferred == 0) returns the caller's slice itself as Immediate with a nil
// Deferred: no copy, no sort, no allocation, and the pick order preserved exactly, so wiring this in
// front of a forward at the default grade cannot perturb accumulation order. Otherwise both halves
// are views on ONE fresh backing array, and Immediate is capped at its length so appending to it can
// never overwrite the first deferred pick.
func partitionDeferredExperts(picks []routePick, maxDeferred int) (expertDeferPartition, error) {
	protectedK, err := expertDeferProtectedK(len(picks), maxDeferred)
	if err != nil {
		return expertDeferPartition{}, err
	}
	if protectedK >= len(picks) {
		return expertDeferPartition{Immediate: picks}, nil
	}
	ranked := make([]routePick, len(picks))
	copy(ranked, picks)
	sort.Slice(ranked, func(a, b int) bool {
		if ranked[a].weight != ranked[b].weight {
			return ranked[a].weight > ranked[b].weight
		}
		return ranked[a].expert < ranked[b].expert
	})
	return expertDeferPartition{
		Immediate: ranked[:protectedK:protectedK],
		Deferred:  ranked[protectedK:],
	}, nil
}

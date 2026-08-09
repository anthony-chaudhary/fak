package model

import (
	"reflect"
	"testing"
)

// hfPicks is the fixture: one token's router top-k in canonical HF order — descending gate weight,
// ties by ascending expert index — which is exactly what route() (moe.go) emits. k = 8, matching the
// num_experts_per_tok of the GLM/Qwen-class MoE checkpoints this seam serves.
func hfPicks() []routePick {
	return []routePick{
		{expert: 3, weight: 0.30},
		{expert: 11, weight: 0.22},
		{expert: 0, weight: 0.17},
		{expert: 47, weight: 0.11},
		{expert: 5, weight: 0.08},
		{expert: 62, weight: 0.06},
		{expert: 19, weight: 0.04},
		{expert: 7, weight: 0.02},
	}
}

// TestExpertDeferProtectedKGrade pins ktransformers' protected_k = num_experts_per_tok - max_deferred
// across the whole admissible band, including both endpoints.
func TestExpertDeferProtectedKGrade(t *testing.T) {
	for maxDeferred, want := range map[int]int{0: 8, 1: 7, 4: 4, 7: 1} {
		got, err := expertDeferProtectedK(8, maxDeferred)
		if err != nil {
			t.Fatalf("k=8 maxDeferred=%d: unexpected err: %v", maxDeferred, err)
		}
		if got != want {
			t.Fatalf("k=8 maxDeferred=%d: protectedK = %d, want %d", maxDeferred, got, want)
		}
	}
}

// TestExpertDeferProtectedKRefusesOutOfRange pins the fail-closed range: maxDeferred is refused
// outside [0, k-1] rather than clamped, and k == maxDeferred (defer EVERYTHING, so the token gets no
// expert contribution at its own layer and nothing is left to hide the host tail behind) is refused
// too. A degenerate k is refused before any arithmetic.
func TestExpertDeferProtectedKRefusesOutOfRange(t *testing.T) {
	for _, tc := range []struct{ k, maxDeferred int }{
		{8, -1}, // negative budget
		{8, 8},  // protected_k would be 0 — nothing runs immediately
		{8, 9},  // past k entirely
		{0, 0},  // no routed experts at all
		{-1, 0}, // malformed k
		{1, 1},  // k=1: the single expert can never be the deferred one
	} {
		if got, err := expertDeferProtectedK(tc.k, tc.maxDeferred); err == nil {
			t.Fatalf("k=%d maxDeferred=%d: got protectedK=%d, want refusal", tc.k, tc.maxDeferred, got)
		}
	}
}

// TestPartitionDeferredExpertsIdentityGrade pins the default: maxDeferred = 0 defers nothing, returns the
// caller's picks unchanged and in their exact original order, and allocates no deferred set — so
// wiring the plan in front of a forward at the default grade cannot perturb accumulation order.
func TestPartitionDeferredExpertsIdentityGrade(t *testing.T) {
	picks := hfPicks()
	plan, err := partitionDeferredExperts(picks, 0)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if !reflect.DeepEqual(plan.Immediate, hfPicks()) {
		t.Fatalf("Immediate = %+v, want the input unchanged", plan.Immediate)
	}
	if plan.Deferred != nil {
		t.Fatalf("Deferred = %+v, want nil at the identity grade", plan.Deferred)
	}
}

// TestPartitionDeferredExpertsPartition is the fixture witness: with k=8 and maxDeferred=3 the top
// protected_k=5 router weights run immediately and the 3 LOWEST run one layer late — upstream's
// `torch.topk` on the routing scores with the remainder masked out, expressed as two disjoint sets.
func TestPartitionDeferredExpertsPartition(t *testing.T) {
	plan, err := partitionDeferredExperts(hfPicks(), 3)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	wantImmediate := []routePick{
		{expert: 3, weight: 0.30},
		{expert: 11, weight: 0.22},
		{expert: 0, weight: 0.17},
		{expert: 47, weight: 0.11},
		{expert: 5, weight: 0.08},
	}
	wantDeferred := []routePick{
		{expert: 62, weight: 0.06},
		{expert: 19, weight: 0.04},
		{expert: 7, weight: 0.02},
	}
	if !reflect.DeepEqual(plan.Immediate, wantImmediate) {
		t.Fatalf("Immediate = %+v, want %+v", plan.Immediate, wantImmediate)
	}
	if !reflect.DeepEqual(plan.Deferred, wantDeferred) {
		t.Fatalf("Deferred = %+v, want %+v", plan.Deferred, wantDeferred)
	}
}

// TestPartitionDeferredExpertsRanksNotInputOrder pins that the partition is a function of the pick SET,
// not of the caller's slice order: a scrambled permutation of the same picks yields the identical
// partition. Without the explicit rank a caller handing picks in expert-index order would defer the
// HIGHEST-weight experts — the exact inversion of the borrow.
func TestPartitionDeferredExpertsRanksNotInputOrder(t *testing.T) {
	canonical, err := partitionDeferredExperts(hfPicks(), 3)
	if err != nil {
		t.Fatalf("canonical: unexpected err: %v", err)
	}
	scrambled := []routePick{
		{expert: 7, weight: 0.02},
		{expert: 0, weight: 0.17},
		{expert: 62, weight: 0.06},
		{expert: 3, weight: 0.30},
		{expert: 19, weight: 0.04},
		{expert: 47, weight: 0.11},
		{expert: 11, weight: 0.22},
		{expert: 5, weight: 0.08},
	}
	got, err := partitionDeferredExperts(scrambled, 3)
	if err != nil {
		t.Fatalf("scrambled: unexpected err: %v", err)
	}
	if !reflect.DeepEqual(got, canonical) {
		t.Fatalf("scrambled partition = %+v, want the canonical %+v", got, canonical)
	}
}

// TestPartitionDeferredExpertsTieBreak pins torch.topk's tie-break at the protection boundary: on equal
// router weights the LOWER expert index is protected, so the split is deterministic instead of
// depending on the sort's internal ordering.
func TestPartitionDeferredExpertsTieBreak(t *testing.T) {
	// Experts 9 and 4 both sit exactly on the protected_k=2 boundary at weight 0.25.
	picks := []routePick{
		{expert: 6, weight: 0.50},
		{expert: 9, weight: 0.25},
		{expert: 4, weight: 0.25},
	}
	plan, err := partitionDeferredExperts(picks, 1)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	want := []routePick{{expert: 6, weight: 0.50}, {expert: 4, weight: 0.25}}
	if !reflect.DeepEqual(plan.Immediate, want) {
		t.Fatalf("Immediate = %+v, want %+v (lower expert index wins the tie)", plan.Immediate, want)
	}
	if !reflect.DeepEqual(plan.Deferred, []routePick{{expert: 9, weight: 0.25}}) {
		t.Fatalf("Deferred = %+v, want expert 9", plan.Deferred)
	}
}

// TestPartitionDeferredExpertsInvariants sweeps the whole admissible budget band and pins the three
// properties the layer-late reconcile depends on: the partition is a permutation of the input (a
// deferred expert is POSTPONED, never dropped, so the reconcile is a SUM not a recompute), the sizes
// are exactly protected_k / max_deferred, and every deferred weight is <= every immediate weight
// (the deferred slice really is the LOWEST-router-weight tail).
func TestPartitionDeferredExpertsInvariants(t *testing.T) {
	for maxDeferred := 0; maxDeferred <= 7; maxDeferred++ {
		plan, err := partitionDeferredExperts(hfPicks(), maxDeferred)
		if err != nil {
			t.Fatalf("maxDeferred=%d: unexpected err: %v", maxDeferred, err)
		}
		if len(plan.Immediate) != 8-maxDeferred || len(plan.Deferred) != maxDeferred {
			t.Fatalf("maxDeferred=%d: sizes = (%d, %d), want (%d, %d)",
				maxDeferred, len(plan.Immediate), len(plan.Deferred), 8-maxDeferred, maxDeferred)
		}
		seen := map[int]float32{}
		for _, p := range append(append([]routePick{}, plan.Immediate...), plan.Deferred...) {
			if _, dup := seen[p.expert]; dup {
				t.Fatalf("maxDeferred=%d: expert %d appears in both halves", maxDeferred, p.expert)
			}
			seen[p.expert] = p.weight
		}
		for _, p := range hfPicks() {
			w, ok := seen[p.expert]
			if !ok || w != p.weight {
				t.Fatalf("maxDeferred=%d: pick %+v lost by the partition", maxDeferred, p)
			}
		}
		for _, d := range plan.Deferred {
			for _, i := range plan.Immediate {
				if d.weight > i.weight {
					t.Fatalf("maxDeferred=%d: deferred %+v outweighs immediate %+v", maxDeferred, d, i)
				}
			}
		}
	}
}

// TestPartitionDeferredExpertsIsPure pins that planning does not mutate the caller's picks (the router's
// own slice is reused downstream by moeFFN) and that the two halves cannot alias each other: growing
// Immediate must not overwrite the first deferred pick, which the capped slice expression prevents.
func TestPartitionDeferredExpertsIsPure(t *testing.T) {
	picks := []routePick{
		{expert: 7, weight: 0.02},
		{expert: 3, weight: 0.30},
		{expert: 11, weight: 0.22},
	}
	plan, err := partitionDeferredExperts(picks, 1)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if !reflect.DeepEqual(picks, []routePick{
		{expert: 7, weight: 0.02},
		{expert: 3, weight: 0.30},
		{expert: 11, weight: 0.22},
	}) {
		t.Fatalf("input mutated: %+v", picks)
	}
	firstDeferred := plan.Deferred[0]
	_ = append(plan.Immediate, routePick{expert: 99, weight: 9})
	if plan.Deferred[0] != firstDeferred {
		t.Fatalf("appending to Immediate clobbered Deferred[0]: %+v, want %+v", plan.Deferred[0], firstDeferred)
	}
}

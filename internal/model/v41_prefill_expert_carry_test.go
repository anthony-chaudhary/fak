package model

// v41_prefill_expert_carry_test.go -- the fak#13697 witness for the expert-major
// (grouped) prefill's routed-expert triple retention.
//
// The multi-token V4.1 prefill contracts its routed experts EXPERT-MAJOR (#13304,
// v41ContractRoutedGrouped): it plans the panel's picks into per-expert groups,
// materializes ONE expert triple per group through v41ExpertTripleInto, contracts
// every (token, slot) row of that group against the live triple, then releases it
// and moves to the next expert. Within a group the triple is reused directly, so
// the group is never re-read and the layer cache serves ZERO hits on this path --
// the already-landed #13294 witness (v41_prefill_expert_union_test.go) states
// exactly that.
//
// Before this leaf v41ExpertTripleInto unconditionally COPIED the materialized
// triple into the layer-scoped f32 cache, regardless of which caller asked. That
// retention is load-bearing for the token-major stream (a repeated expert across
// the token dimension is re-read and the cache serves it), but on the expert-major
// grouped path it is a dead copy: the panel never re-reads the triple, so the copy
// is allocated, memcpy'd, and then evicted unread -- on the very first-token path
// fak#13294 steers by. At the published geometry (I=2304, H=5120) one triple is
// 3 * 2304 * 5120 * 4 = ~141 MiB, and a panel retaining one per distinct expert
// across 40 layers is a large cumulative allocation on the critical path.
//
// This witness pins the fix: the grouped prefill must retain ZERO triples, while
// the token-major stream that exists to be served by the cache still retains
// (so the assertion is not vacuously satisfied by disabling the cache).

import "testing"

// TestV41GroupedPrefillCarriesNoUnreadTriples is the #13697 witness.
//
// A multi-token, maximally-overlapping prefill routes the same experts on every
// token. The grouped contraction materializes each distinct expert ONCE and never
// re-reads it, so it must retain zero triples. The token-major stream on the SAME
// fixture must retain at least one triple, or the grouping has disabled the cache
// for the arm it exists to serve.
func TestV41GroupedPrefillCarriesNoUnreadTriples(t *testing.T) {
	// The token-major override is a package global; restore it even on failure so
	// it cannot leak into a later test in the package.
	t.Cleanup(func() { v41ForceTokenMajor = false })

	ids := []int{1, 1, 1, 1, 1, 1, 1, 1}

	// Grouped (expert-major) arm: the panel materializes one triple per distinct
	// grouped expert and never re-reads it, so it must retain none.
	grouped, _ := v41TierOnlyBudgetModel(t, 0)
	enableV41RetentionWitness()
	v41ForceTokenMajor = false
	if _, err := grouped.forwardV41(ids, nil); err != nil {
		t.Fatalf("grouped %d-token prefill: %v", len(ids), err)
	}
	groupedRetentions := v41Retentions()
	if groupedRetentions != 0 {
		t.Fatalf("expert-major grouped prefill retained %d expert triples; "+
			"the grouped path materializes each expert once and never re-reads it, "+
			"so every retained copy is dead work on the first-token path (fak#13697)",
			groupedRetentions)
	}

	// Token-major arm: the stream this cache exists to serve must still retain, so
	// the zero above is attributable to the grouped caller's decision, not to a
	// disabled cache. Non-vacuous ceiling.
	tokenMajor, _ := v41TierOnlyBudgetModel(t, 0)
	enableV41RetentionWitness()
	v41ForceTokenMajor = true
	if _, err := tokenMajor.forwardV41(ids, nil); err != nil {
		t.Fatalf("token-major %d-token prefill: %v", len(ids), err)
	}
	tokenMajorRetentions := v41Retentions()
	v41ForceTokenMajor = false
	if tokenMajorRetentions == 0 {
		t.Fatal("token-major prefill retained no expert triples; the retention witness " +
			"is vacuous (the cache is disabled, so the grouped zero proves nothing)")
	}

	t.Logf("expert triple retentions over an %d-token prefill: grouped=%d token-major=%d",
		len(ids), groupedRetentions, tokenMajorRetentions)
}

// TestV41GroupedPrefillOutputUnchangedByCarryFix pins the correctness complement:
// disabling the grouped retention must not change the grouped prefill's output. It
// re-runs the existing #13304 equivalence on the SAME reduced fixture -- grouped
// and token-major must still agree bit-for-bit -- so the carry fix is proven to be
// a pure bookkeeping change, never an arithmetic one.
func TestV41GroupedPrefillOutputUnchangedByCarryFix(t *testing.T) {
	t.Cleanup(func() { v41ForceTokenMajor = false })
	ids := []int{1, 1, 1, 1, 1, 1, 1, 1}

	grouped, _ := v41TierOnlyBudgetModel(t, 0)
	v41ForceTokenMajor = false
	gAct, gErr := grouped.forwardV41(ids, nil)
	if gErr != nil {
		t.Fatalf("grouped prefill: %v", gErr)
	}

	tokenMajor, _ := v41TierOnlyBudgetModel(t, 0)
	v41ForceTokenMajor = true
	tmAct, tmErr := tokenMajor.forwardV41(ids, nil)
	v41ForceTokenMajor = false
	if tmErr != nil {
		t.Fatalf("token-major prefill: %v", tmErr)
	}

	if len(gAct.Logits) != len(tmAct.Logits) {
		t.Fatalf("grouped emitted %d logit rows, token-major %d", len(gAct.Logits), len(tmAct.Logits))
	}
	for pos := range tmAct.Logits {
		for i := range tmAct.Logits[pos] {
			if gAct.Logits[pos][i] != tmAct.Logits[pos][i] {
				t.Fatalf("position %d component %d: grouped %g, token-major %g "+
					"(dropping the unread grouped retention must not change arithmetic)",
					pos, i, gAct.Logits[pos][i], tmAct.Logits[pos][i])
			}
		}
	}
}

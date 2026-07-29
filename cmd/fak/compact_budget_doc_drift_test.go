package main

import (
	"fmt"
	"os"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// compactBudgetDoctrineDoc is the canonical policy page for --compact-history-budget
// (docs/long-context-defaults.md says so itself). It is the ONE published surface this lane owns
// that states the number a guard launch actually runs at.
const compactBudgetDoctrineDoc = "../../docs/long-context-defaults.md"

// TestCompactBudgetDoctrineDocMatchesTheResolver binds the published guard-effective compaction
// budget to the code that produces it. Issue #5430: `fak guard -h` prints the flag's
// pre-resolution default (gateway.DefaultCompactHistoryBudget, 48000) while guard.go immediately
// replaces it with resolveGuardCompactBudget's value on every launch that did not pass the flag
// explicitly — so the published number was off by 2x from the running one, and an operator sizing
// their traffic against the documented figure was reasoning about a budget that never applied.
//
// The doc is the surface this lane can correct; the flag help lives in cmd/fak/guard.go and is
// owned elsewhere, so it is still wrong (see the issue). This test makes sure the half that IS
// corrected cannot silently drift back: if resolveGuardCompactBudget ever returns a different
// number, the doc no longer names it and this reds — the same discipline
// TestClaudeGuardCompactHistoryBudget keeps for the dispatch worker's mirrored constant.
//
// It asserts the DOC against the RESOLVER, never the resolver against a frozen literal, so the
// witness is the relationship and not a second copy of the number.
func TestCompactBudgetDoctrineDocMatchesTheResolver(t *testing.T) {
	raw, err := os.ReadFile(compactBudgetDoctrineDoc)
	if err != nil {
		t.Fatalf("read %s: %v", compactBudgetDoctrineDoc, err)
	}
	doc := string(raw)

	// The value a plain `fak guard -- claude` really runs at: no explicit flag, so the resolver
	// substitutes the floor-aware budget for the lean flag default.
	effective := resolveGuardCompactBudget(gateway.DefaultCompactHistoryBudget, false)
	if effective == gateway.DefaultCompactHistoryBudget {
		t.Skip("resolver no longer diverges from the flag default — the #5430 gap this doc explains is closed")
	}
	// Anchored on the full CLAIM sentence, not a bare number: this page's fallback-priors table
	// already contains "128K" and "32K", so a bare-substring check would pass on a coincidence
	// somewhere else in the doc and witness nothing.
	wantClaim := fmt.Sprintf("is resolved to **%dK**", effective/1000)
	if effective%1000 != 0 {
		wantClaim = fmt.Sprintf("is resolved to **%d**", effective)
	}
	if !strings.Contains(doc, wantClaim) {
		t.Errorf("%s must state %q (resolveGuardCompactBudget(%d, false) = %d); the doc has drifted "+
			"from the resolver", compactBudgetDoctrineDoc, wantClaim,
			gateway.DefaultCompactHistoryBudget, effective)
	}
	// Naming the resolver is what lets a reader check the claim instead of trusting the number.
	if !strings.Contains(doc, "resolveGuardCompactBudget") {
		t.Errorf("%s must name resolveGuardCompactBudget so the resolution is checkable, not folklore", compactBudgetDoctrineDoc)
	}
	// The second correction the issue asks for: the budget's UNIT. It is applied to messages[]
	// alone, so a budget sized from total context is set too high and the cut never engages.
	for _, want := range []string{"`messages[]` alone", "system+tools block"} {
		if !strings.Contains(doc, want) {
			t.Errorf("%s must state what the budget is measured against (missing %q)", compactBudgetDoctrineDoc, want)
		}
	}
	// The doc must still carry the lean flag default too — a reader comparing `--help` against this
	// page needs both numbers or the correction reads as a contradiction rather than a resolution.
	if leanClaim := fmt.Sprintf("lean %dK line", gateway.DefaultCompactHistoryBudget/1000); !strings.Contains(doc, leanClaim) {
		t.Errorf("%s must still name the lean flag default as %q alongside the resolved value",
			compactBudgetDoctrineDoc, leanClaim)
	}
}

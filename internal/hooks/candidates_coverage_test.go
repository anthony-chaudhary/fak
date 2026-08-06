package hooks

// candidates_coverage_test.go — the SET-MEMBERSHIP half of the candidate denominator (#5602).
//
// candidates_test.go pins the ledger's own semantics (unreported ≠ zero, per-gate units,
// last-call-wins, concurrency). This file pins the property that makes the denominator a trust
// floor rather than a courtesy: EVERY gate the pre-commit runner actually invokes records one.
//
// Without this test the feature decays on the next commit. A gate added to PreCommitGates() with
// no NoteCandidates call reports `"candidates": null` forever, and null is indistinguishable in a
// dashboard from a gate that has not run yet — the exact ambiguity the issue exists to remove,
// rebuilt one gate at a time. Binding the registry to the ledger in the same direction the
// fail-closed audit binds it (failclosed_ledger_test.go) makes the omission a CI failure at the
// moment it is introduced, which is the only moment it is cheap to fix.
//
// SCOPE IS THE STAGED SET, DELIBERATELY. #5602's prose estimates "54 one-line populations", one
// per internal/hooks/gate_*.go. That count is wrong, and the difference is load-bearing: only the
// gates in PreCommitGates() take a *StagedDiff, and NoteCandidates is a *StagedDiff method. The
// rest are HygieneGates over a *TrackedTree (`fak hygiene`) — BRAND_CONSISTENCY, DEMO_COMMAND,
// BROWSER_CONTRACT, BARE_DEV_SPELLING and friends — which walk the whole tracked tree, not a
// staged set. "How many staged items did this gate judge" is not a question they can answer, so
// this test quantifies over the registry the runner reads rather than over a file glob.
//
// Stdlib-only, like the rest of this package (architest tier 1: hooks imports nothing internal).

import "testing"

// emptyStagedDiff is a rooted-but-empty diff: the "commit that staged nothing this gate cares
// about" case. It is the strongest probe for this property, because a gate that only reaches its
// NoteCandidates call AFTER its filter admits something would pass a populated fixture and still
// leave the zero case unreported — and the zero case is the one the issue is about.
func emptyStagedDiff(root string) *StagedDiff {
	return &StagedDiff{
		Root:        root,
		AddedByFile: map[string][]AddedLine{},
		Treeish:     ":",
	}
}

// TestEveryPreCommitGateRecordsACandidateDenominator is the CI gate. A new entry in
// PreCommitGates() whose Check never calls NoteCandidates fails here by name.
//
// A gate is allowed to FAIL (return an error) — the runner is fail-open by design (#5299) — and a
// gate that could not reach its evidence has nothing to count, so an error excuses the gate from
// the denominator requirement. What is NOT excused is running to completion and staying silent.
func TestEveryPreCommitGateRecordsACandidateDenominator(t *testing.T) {
	gates := PreCommitGates()
	if len(gates) == 0 {
		// Fail closed on an empty registry: a vacuous pass over zero gates would report this
		// property as held while checking nothing, which is the same class of lie as the empty
		// findings payload that motivated the issue.
		t.Fatalf("PreCommitGates() returned 0 gates; refusing to report a vacuous pass")
	}

	var silent []string
	for _, g := range gates {
		d := emptyStagedDiff(t.TempDir())
		_, err := g.Check(d)
		if err != nil {
			t.Logf("gate %s could not run (%v); excused from the denominator requirement", g.Name, err)
			continue
		}
		if _, _, ok := d.Candidates(g.Name); !ok {
			silent = append(silent, g.Name)
		}
	}

	if len(silent) > 0 {
		t.Errorf("gate(s) ran to completion but recorded NO candidate denominator: %v\n"+
			"each must call d.NoteCandidates(%q, n, unit) inside its Check, at the site its filter is\n"+
			"already computed — including the early-return path where the filter admits nothing,\n"+
			"since 0 is a real answer and silence is not (#5602, internal/hooks/candidates.go)", silent, silent[0])
	}
}

// TestGateDenominatorIsKeyedByRegisteredName pins the join the CLI depends on. buildGateReport
// (cmd/fak/hooks.go) looks the denominator up by Gate.Name, so a gate that records under a
// different string — a rename that missed the literal inside Check, or a typo — reports null
// while believing it reported a number. That failure is silent in every other test.
func TestGateDenominatorIsKeyedByRegisteredName(t *testing.T) {
	for _, g := range PreCommitGates() {
		d := emptyStagedDiff(t.TempDir())
		if _, err := g.Check(d); err != nil {
			continue
		}
		reported := d.ReportedGates()
		for _, name := range reported {
			if name != g.Name {
				t.Errorf("gate %q recorded its denominator under the name %q; buildGateReport looks it\n"+
					"up by the REGISTERED name and would render null for %q", g.Name, name, g.Name)
			}
		}
	}
}

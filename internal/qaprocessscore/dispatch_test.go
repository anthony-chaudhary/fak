package qaprocessscore

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/brittleness"
	"github.com/anthony-chaudhary/fak/internal/dogfoodissues"
)

// fixtureGaps builds a deterministic two-gap set: one regression gap (a revert of
// internal/foo with no accompanying _test.go) and one coverage gap (internal/bar below the
// absolute floor). barCov lets a caller vary only the run-varying coverage percentage to prove
// the dedup key is content-stable across it.
func fixtureGaps(barCov float64) []Gap {
	commits := []brittleness.Commit{
		c("deadbeef01", `Revert "feat(foo): bad landing"`, "internal/foo/foo.go"),
	}
	return Gaps(commits, map[string]float64{"internal/bar": barCov}, CoverageBaseline{Floor: 80.0})
}

func TestGapsFansBothKPIs(t *testing.T) {
	gaps := fixtureGaps(40.0)
	if len(gaps) != 2 {
		t.Fatalf("Gaps: got %d want 2 (regression + coverage) (%+v)", len(gaps), gaps)
	}
	// Regression gap first (worst-first: a bug reached the trunk), coverage gap second.
	if gaps[0].KPI != RegressionCatchKey || gaps[1].KPI != CoverageDisciplineKey {
		t.Fatalf("gap order/KPI wrong: %s, %s", gaps[0].KPI, gaps[1].KPI)
	}
	if gaps[0].Ref != "deadbeef01" {
		t.Errorf("regression gap must anchor on the revert SHA, got %q", gaps[0].Ref)
	}
	if gaps[1].Class != coverageBelowFloorClass || gaps[1].Ref != "internal/bar" {
		t.Errorf("coverage gap must anchor {class,pkg}, got %s %q", gaps[1].Class, gaps[1].Ref)
	}
}

func TestActionItemsAreDispatchable(t *testing.T) {
	items := ActionItems(fixtureGaps(40.0), "fak score qa-process --json")
	if len(items) != 2 {
		t.Fatalf("ActionItems: got %d want 2", len(items))
	}
	seen := map[string]bool{}
	for _, it := range items {
		if it.Key == "" || seen[it.Key] {
			t.Errorf("ActionItem key not unique/stable: %q", it.Key)
		}
		seen[it.Key] = true
		if it.DebtName != DebtKey || it.DebtCount != 1 {
			t.Errorf("debt fields wrong: %+v", it)
		}
		if it.ParentRef != qaProcessEpicRef {
			t.Errorf("each child must hang under the epic %s, got %q", qaProcessEpicRef, it.ParentRef)
		}
		if len(it.Labels) == 0 || it.Labels[0] != "track/E-testing-quality" {
			t.Errorf("must carry the E-testing-quality track label, got %v", it.Labels)
		}
		if len(it.Paths) == 0 || !strings.HasSuffix(it.Paths[0], "/**") {
			t.Errorf("must route by the owning package tree: %v", it.Paths)
		}
		if !strings.Contains(it.DoneCondition, it.Key) {
			t.Errorf("done condition must cite the stable key %q: %q", it.Key, it.DoneCondition)
		}
	}
	// The stable keys are content-addressed on {kpi, class, ref}, not the drifting Detail.
	for _, want := range []string{
		"qa-process-debt/regression_catch/regression-catch-gap-deadbeef01",
		"qa-process-debt/coverage_discipline/coverage-below-floor-internal-bar",
	} {
		if !seen[want] {
			t.Errorf("missing expected stable key %q in %v", want, keysOf(seen))
		}
	}
}

// TestKeyStableAcrossDetailDrift is the load-bearing dedup guarantee: the coverage percentage
// in the Detail changes run-to-run, but the dedup Key must not -- else a re-run opens a
// duplicate instead of updating in place.
func TestKeyStableAcrossDetailDrift(t *testing.T) {
	a := fixtureGaps(40.0)[1] // coverage gap at 40%
	b := fixtureGaps(55.0)[1] // same package, 55%
	if a.Key() != b.Key() {
		t.Fatalf("key must be stable across coverage drift: %q != %q", a.Key(), b.Key())
	}
	if a.Detail == b.Detail {
		t.Fatalf("fixture bug: the two runs should render different Detail (%q)", a.Detail)
	}
}

func TestGapInstructionsStayAlignedByKPI(t *testing.T) {
	tests := []struct {
		name string
		kpi  string
		want []string
	}{
		{"regression", RegressionCatchKey, []string{"reproduces the bug", "reproduces the reverted bug", "covering the reverted behavior"}},
		{"coverage", CoverageDisciplineKey, []string{"coverage ratchet", "go tool cover -func", "coverage floor/ratchet"}},
		{"flake", FlakeQuarantineKey, []string{"deterministic", "run the test in a loop / under -race", "without a rerun"}},
		{"fallback", "future-kpi", []string{"Close the qa-process gap", "Fix the underlying qa-process gap", "Close the named qa-process gap"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gap := Gap{KPI: tt.kpi}
			got := []string{gap.NextAction("internal/example"), gap.WorkingSpine("internal/example"), gap.InScope("internal/example")}
			for i, want := range tt.want {
				if !strings.Contains(got[i], want) || !strings.Contains(got[i], "internal/example") {
					t.Errorf("instruction %d = %q, want host and %q", i, got[i], want)
				}
			}
		})
	}
}

// TestRerunUpdatesInPlace proves the dedup convergence end-to-end through the backlog bridge:
// a first dry-run plans two creates; feeding the generated issues back (matched by their
// marker key) turns the same batch into two updates, never a duplicate create.
func TestRerunUpdatesInPlace(t *testing.T) {
	items := ActionItems(fixtureGaps(40.0), "fak score qa-process --json")
	opt := dogfoodissues.BuildOptions{DedupeChecked: true, DedupeCap: 10, ParentBaseline: 40, CompletionStandard: "development"}

	plan, skipped := dogfoodissues.BuildPlanWithOptions(items, nil, opt)
	if len(skipped) != 0 {
		t.Fatalf("items must pass the dispatchability review, got skipped=%+v", skipped)
	}
	if len(plan) != 2 {
		t.Fatalf("first run: got %d planned rows want 2", len(plan))
	}
	for _, r := range plan {
		if r.Action != "create" {
			t.Errorf("first run must create %s, got %q", r.Key, r.Action)
		}
	}

	// Re-run against the issues those items would have created (marker-key matched).
	var existing []dogfoodissues.Issue
	for i, it := range items {
		existing = append(existing, dogfoodissues.Issue{
			Number: 100 + i,
			Title:  it.Title,
			Body:   dogfoodissues.IssueBody(it),
			State:  "open",
		})
	}
	plan2, skipped2 := dogfoodissues.BuildPlanWithOptions(items, existing, opt)
	if len(skipped2) != 0 {
		t.Fatalf("rerun: unexpected skips %+v", skipped2)
	}
	if len(plan2) != 2 {
		t.Fatalf("rerun: got %d planned rows want 2", len(plan2))
	}
	for _, r := range plan2 {
		if r.Action != "update" {
			t.Errorf("rerun must update %s in place, got %q", r.Key, r.Action)
		}
	}
}

func keysOf(m map[string]bool) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}

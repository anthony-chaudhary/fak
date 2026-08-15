package issuecohort

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/issuepolicy"
)

// fullCandidate returns a candidate that passes the (non-live) issuecontract
// review as a dispatchable leaf. Tests mutate the returned value to exercise one
// axis at a time.
func fullCandidate(key string) issuepolicy.Candidate {
	return issuepolicy.Candidate{
		Schema:         issuepolicy.Schema,
		Key:            key,
		Title:          "leaf " + key,
		ParentRef:      "epic #1",
		CurrentState:   "the thing is not yet done",
		WhyNow:         "it unblocks the next leaf",
		WorkingSpine:   "make the working path more true",
		InScope:        "the one file",
		OutOfScope:     "everything else",
		DoneCondition:  "the file changes",
		Witness:        "go test ./... passes",
		AcceptanceGate: "make ci",
		ClosureBinding: "commit cites #1 and (fak leaf)",
		Paths:          []string{"internal/" + key + "/**"},
	}
}

func TestBuildDisjointPathsShareWave(t *testing.T) {
	a := fullCandidate("alpha")
	a.Paths = []string{"internal/foo/x.go"}
	b := fullCandidate("beta")
	b.Paths = []string{"internal/foo/y.go"}

	plan := Build([]issuepolicy.Candidate{a, b}, Options{})
	if plan.Dispatchable != 2 {
		t.Fatalf("dispatchable = %d, want 2", plan.Dispatchable)
	}
	if plan.CollisionPairs != 0 {
		t.Fatalf("collision pairs = %d, want 0 (distinct files in a dir do not overlap)", plan.CollisionPairs)
	}
	if plan.NumWaves != 1 || plan.PeakConcurrency != 2 {
		t.Fatalf("waves=%d peak=%d, want 1 wave of 2", plan.NumWaves, plan.PeakConcurrency)
	}
}

func TestBuildOverlappingPathsSplitWaves(t *testing.T) {
	a := fullCandidate("alpha")
	a.Paths = []string{"internal/foo/**"}
	b := fullCandidate("beta")
	b.Paths = []string{"internal/foo/bar.go"} // inside a's tree

	plan := Build([]issuepolicy.Candidate{a, b}, Options{})
	if plan.CollisionPairs != 1 {
		t.Fatalf("collision pairs = %d, want 1", plan.CollisionPairs)
	}
	if plan.NumWaves != 2 || plan.PeakConcurrency != 1 {
		t.Fatalf("waves=%d peak=%d, want 2 serial waves", plan.NumWaves, plan.PeakConcurrency)
	}
}

func TestBuildSameLaneNoPathsCollide(t *testing.T) {
	a := fullCandidate("alpha")
	a.Paths = nil
	a.Lane = "docs"
	b := fullCandidate("beta")
	b.Paths = nil
	b.Lane = "docs"

	plan := Build([]issuepolicy.Candidate{a, b}, Options{})
	if plan.Dispatchable != 2 {
		t.Fatalf("dispatchable = %d, want 2 (lane routes them)", plan.Dispatchable)
	}
	if plan.NumWaves != 2 {
		t.Fatalf("waves = %d, want 2 (whole-lane takers collide)", plan.NumWaves)
	}
}

func TestBuildDifferentLaneNoPathsShareWave(t *testing.T) {
	a := fullCandidate("alpha")
	a.Paths = nil
	a.Lane = "docs"
	b := fullCandidate("beta")
	b.Paths = nil
	b.Lane = "gateway"

	plan := Build([]issuepolicy.Candidate{a, b}, Options{})
	if plan.NumWaves != 1 || plan.PeakConcurrency != 2 {
		t.Fatalf("waves=%d peak=%d, want 1 wave of 2 (distinct lanes)", plan.NumWaves, plan.PeakConcurrency)
	}
}

func TestBuildOversizedSubdivides(t *testing.T) {
	a := fullCandidate("big")
	a.ExpectedSteps = 20

	plan := Build([]issuepolicy.Candidate{a}, Options{})
	if plan.Dispatchable != 0 || plan.Subdividable != 1 {
		t.Fatalf("dispatchable=%d subdividable=%d, want 0/1", plan.Dispatchable, plan.Subdividable)
	}
	if len(plan.Subdivide) != 1 {
		t.Fatalf("subdivide rows = %d, want 1", len(plan.Subdivide))
	}
	if got := plan.Subdivide[0].ChildIssueBudget; got != 3 { // ceil(20/8)
		t.Fatalf("child issue budget = %d, want 3", got)
	}
	if plan.ChildIssueTotal != 3 {
		t.Fatalf("child issue total = %d, want 3", plan.ChildIssueTotal)
	}
	if !hasReason(plan.Subdivide[0].Reasons, issuepolicy.ReasonOversizedSteps) {
		t.Fatalf("subdivide reasons = %v, want oversized", plan.Subdivide[0].Reasons)
	}
}

func TestBuildEpicSubdivides(t *testing.T) {
	a := fullCandidate("umbrella")
	a.WorkUnit = "epic"

	plan := Build([]issuepolicy.Candidate{a}, Options{})
	if plan.Subdividable != 1 || len(plan.Subdivide) != 1 {
		t.Fatalf("subdividable=%d rows=%d, want 1/1", plan.Subdividable, len(plan.Subdivide))
	}
	if plan.Subdivide[0].ChildIssueBudget != 1 {
		t.Fatalf("child budget = %d, want 1 (unknown steps)", plan.Subdivide[0].ChildIssueBudget)
	}
	if !hasReason(plan.Subdivide[0].Reasons, issuepolicy.ReasonNotDispatchLeaf) {
		t.Fatalf("reasons = %v, want not-dispatch-leaf", plan.Subdivide[0].Reasons)
	}
}

func TestBuildScopeGapTriaged(t *testing.T) {
	a := fullCandidate("vague")
	a.OutOfScope = ""    // scope incomplete
	a.DoneCondition = "" // and no done condition

	plan := Build([]issuepolicy.Candidate{a}, Options{})
	if plan.TriageOnly != 1 || len(plan.Triage) != 1 {
		t.Fatalf("triageOnly=%d rows=%d, want 1/1", plan.TriageOnly, len(plan.Triage))
	}
	if plan.Subdividable != 0 || plan.Dispatchable != 0 {
		t.Fatalf("dispatchable=%d subdividable=%d, want 0/0", plan.Dispatchable, plan.Subdividable)
	}
	if !hasReason(plan.Triage[0].Reasons, issuepolicy.ReasonScopeIncomplete) {
		t.Fatalf("triage reasons = %v, want scope-incomplete", plan.Triage[0].Reasons)
	}
}

func TestBuildDuplicateKeyDetected(t *testing.T) {
	a := fullCandidate("dupe")
	b := fullCandidate("dupe") // same key
	b.Paths = []string{"internal/other/**"}

	plan := Build([]issuepolicy.Candidate{a, b}, Options{})
	if plan.Dispatchable != 1 {
		t.Fatalf("dispatchable = %d, want 1 (duplicate not planned twice)", plan.Dispatchable)
	}
	if plan.DuplicateKeys != 1 {
		t.Fatalf("duplicate keys = %d, want 1", plan.DuplicateKeys)
	}
	if len(plan.Duplicates) != 1 || plan.Duplicates[0].Key != "dupe" || plan.Duplicates[0].Count != 2 {
		t.Fatalf("duplicates = %+v, want one dupe x2", plan.Duplicates)
	}
}

func TestBuildMaxWaveCap(t *testing.T) {
	var cands []issuepolicy.Candidate
	for _, k := range []string{"a", "b", "c"} {
		c := fullCandidate(k)
		c.Paths = []string{"internal/" + k + "/**"} // all disjoint
		cands = append(cands, c)
	}
	plan := Build(cands, Options{MaxWave: 2})
	if plan.CollisionPairs != 0 {
		t.Fatalf("collision pairs = %d, want 0", plan.CollisionPairs)
	}
	if plan.NumWaves != 2 || plan.PeakConcurrency != 2 {
		t.Fatalf("waves=%d peak=%d, want 2 waves capped at 2", plan.NumWaves, plan.PeakConcurrency)
	}
}

func TestBuildStepBudgetCountsUnknownAsOne(t *testing.T) {
	a := fullCandidate("stepless") // ExpectedSteps 0
	a.Paths = []string{"internal/foo/x.go"}
	b := fullCandidate("stepped")
	b.Paths = []string{"internal/foo/y.go"}
	b.ExpectedSteps = 4

	plan := Build([]issuepolicy.Candidate{a, b}, Options{})
	if plan.NumWaves != 1 {
		t.Fatalf("waves = %d, want 1", plan.NumWaves)
	}
	if got := plan.Waves[0].StepBudget; got != 5 { // 1 (unknown) + 4
		t.Fatalf("step budget = %d, want 5", got)
	}
}

func TestPortfolioCarriesCentralityWithoutReorderingOrScoring(t *testing.T) {
	core := fullCandidate("core")
	core.Priority = "P1"
	core.ProblemFrame = issuepolicy.ProblemFrame{Schema: issuepolicy.ProblemFrameSchema, Ready: true, Enforced: true, Centrality: issuepolicy.CentralityCore, Checks: map[string]issuepolicy.ProblemCheck{
		"p1": {ID: "p1", Status: issuepolicy.ProblemCheckAdvanced, Evidence: "shared context stays reusable", Valid: true},
		"p2": {ID: "p2", Status: issuepolicy.ProblemCheckPreserved, Evidence: "no new operating cost", Valid: true},
		"p3": {ID: "p3", Status: issuepolicy.ProblemCheckNA, Evidence: "fixed fixture has no adaptation", Valid: true},
		"p4": {ID: "p4", Status: issuepolicy.ProblemCheckAdvanced, Evidence: "dispatch carries the frame", Valid: true},
	}}

	stewardship := fullCandidate("stewardship")
	stewardship.Priority = "urgent"
	stewardship.Reversibility = "rollback the release manifest"
	stewardship.BoundaryNotes = []string{"signing deadline is hard"}
	stewardship.Dependencies = []issuepolicy.DependencyRef{{Relation: "blocked-by", Issue: 7, Blocking: true}}
	stewardship.ProblemFrame = issuepolicy.ProblemFrame{Schema: issuepolicy.ProblemFrameSchema, Ready: true, Enforced: true, Centrality: issuepolicy.CentralityStewardship, CentralityTarget: "release signing obligation", Checks: map[string]issuepolicy.ProblemCheck{}}

	enabling := fullCandidate("enabling")
	enabling.ProblemFrame = issuepolicy.ProblemFrame{Schema: issuepolicy.ProblemFrameSchema, Ready: true, Enforced: true, Centrality: issuepolicy.CentralityEnabling, CentralityTarget: "managed-context outcome", Checks: map[string]issuepolicy.ProblemCheck{}}

	peripheral := fullCandidate("peripheral")
	peripheral.ProblemFrame = issuepolicy.ProblemFrame{Schema: issuepolicy.ProblemFrameSchema, Ready: true, Enforced: true, Centrality: issuepolicy.CentralityPeripheral, Checks: map[string]issuepolicy.ProblemCheck{}}

	legacy := fullCandidate("legacy")
	plan := Build([]issuepolicy.Candidate{core, stewardship, enabling, peripheral, legacy}, Options{})
	if len(plan.Portfolio) != 5 {
		t.Fatalf("portfolio rows = %d, want 5", len(plan.Portfolio))
	}
	wantOrder := []string{"core", "stewardship", "enabling", "peripheral", "legacy"}
	for i, want := range wantOrder {
		if plan.Portfolio[i].Key != want {
			t.Fatalf("portfolio reordered at %d: got %q want %q", i, plan.Portfolio[i].Key, want)
		}
	}
	if got := plan.Centrality; got.Core != 1 || got.Enabling != 1 || got.Stewardship != 1 || got.Peripheral != 1 || got.Unclassified != 1 {
		t.Fatalf("centrality counts = %+v", got)
	}
	steward := plan.Portfolio[1]
	if steward.Priority != "urgent" || len(steward.Dependencies) != 1 || len(steward.RiskBoundaryNotes) != 1 || steward.Reversibility != "rollback the release manifest" {
		t.Fatalf("independent selection axes lost: %+v", steward)
	}
	if steward.CentralityTarget != "release signing obligation" || !strings.Contains(steward.SelectionNote, "may outrank ready Core") {
		t.Fatalf("stewardship row = %+v", steward)
	}
	if plan.Portfolio[2].CentralityTarget != "managed-context outcome" {
		t.Fatalf("enabling linkage lost: %+v", plan.Portfolio[2])
	}
	if plan.Portfolio[4].Centrality != issuepolicy.CentralityUnclassified {
		t.Fatalf("legacy row hidden or inferred: %+v", plan.Portfolio[4])
	}
	if got := plan.Portfolio[0].ProblemFrame.Checks["p1"].Evidence; got != "shared context stays reusable" {
		t.Fatalf("portfolio lost canonical P1 evidence: %q", got)
	}
	rendered := Render(plan)
	for _, want := range []string{
		"portfolio: core centrality=core checks=p1:advanced[shared context stays reusable]",
		"- core  centrality=core checks=p1:advanced[shared context stays reusable]",
		"centrality=stewardship(release signing obligation)",
	} {
		if !strings.Contains(rendered, want) {
			t.Fatalf("render missing %q:\n%s", want, rendered)
		}
	}
}

func TestPortfolioDoesNotInferCentralityFromMetadata(t *testing.T) {
	candidate := fullCandidate("internal/core")
	candidate.Title = "Core epic implementation"
	candidate.ParentRef = "Core outcome #99"
	candidate.Labels = []string{"core", "priority/P0"}
	plan := Build([]issuepolicy.Candidate{candidate}, Options{})
	if got := plan.Portfolio[0].Centrality; got != issuepolicy.CentralityUnclassified {
		t.Fatalf("metadata inferred centrality %q, want unclassified", got)
	}
}

func TestRenderSmoke(t *testing.T) {
	a := fullCandidate("alpha")
	b := fullCandidate("big")
	b.ExpectedSteps = 20
	plan := Build([]issuepolicy.Candidate{a, b}, Options{})
	out := Render(plan)
	for _, want := range []string{"issue-cohort:", "concurrency:", "centrality (non-scoring):", "portfolio: alpha centrality=unclassified", "wave 0:", "split-first"} {
		if !strings.Contains(out, want) {
			t.Fatalf("render missing %q in:\n%s", want, out)
		}
	}
}

func TestWaveLeaseRegionCoversMembersMinimally(t *testing.T) {
	a := fullCandidate("alpha")
	a.Paths = []string{"internal/foo/**", "internal/foo/bar.go"} // bar.go is inside foo
	b := fullCandidate("beta")
	b.Paths = []string{"internal/baz/x.go"} // disjoint from a

	plan := Build([]issuepolicy.Candidate{a, b}, Options{})
	if plan.NumWaves != 1 {
		t.Fatalf("waves = %d, want 1 (disjoint)", plan.NumWaves)
	}
	region := plan.Waves[0].LeaseRegion
	// foo/bar.go collapses under foo; baz/x.go stays. Minimal roots, sorted.
	want := []string{"internal/baz/x.go", "internal/foo"}
	if len(region) != len(want) {
		t.Fatalf("lease region = %v, want %v", region, want)
	}
	for i := range want {
		if region[i] != want[i] {
			t.Fatalf("lease region = %v, want %v", region, want)
		}
	}
	// Every member path must be covered by some root.
	for _, m := range plan.Waves[0].Members {
		for _, p := range m.Paths {
			np := normPath(p)
			covered := false
			for _, r := range region {
				if np == r || pathOverlap(r, np) {
					covered = true
					break
				}
			}
			if !covered {
				t.Fatalf("member path %q not covered by lease region %v", p, region)
			}
		}
	}
}

func TestWaveLeaseLanesForLaneOnlyMembers(t *testing.T) {
	a := fullCandidate("alpha")
	a.Paths = nil
	a.Lane = "docs"
	b := fullCandidate("beta")
	b.Paths = nil
	b.Lane = "gateway"

	plan := Build([]issuepolicy.Candidate{a, b}, Options{})
	if plan.NumWaves != 1 {
		t.Fatalf("waves = %d, want 1 (distinct lanes co-wave)", plan.NumWaves)
	}
	lanes := plan.Waves[0].LeaseLanes
	if len(lanes) != 2 || lanes[0] != "docs" || lanes[1] != "gateway" {
		t.Fatalf("lease lanes = %v, want [docs gateway]", lanes)
	}
	if len(plan.Waves[0].LeaseRegion) != 0 {
		t.Fatalf("lease region = %v, want empty (no path-scoped members)", plan.Waves[0].LeaseRegion)
	}
}

func hasReason(reasons []string, want string) bool {
	for _, r := range reasons {
		if r == want {
			return true
		}
	}
	return false
}

func TestIsSplitTargetCoversBothSplitReasons(t *testing.T) {
	// BOTH reasons make an issue a split target. `fak issue decompose` used to carry
	// its own copy of this predicate, so a test that only exercised one reason would
	// let the two surfaces drift into disagreeing about what "too big" means.
	for _, reason := range []string{issuepolicy.ReasonNotDispatchLeaf, issuepolicy.ReasonOversizedSteps} {
		if !IsSplitTarget(issuepolicy.Review{Reasons: []string{reason}}) {
			t.Fatalf("IsSplitTarget(%q) = false, want true", reason)
		}
	}
	if IsSplitTarget(issuepolicy.Review{Reasons: []string{"missing-witness"}}) {
		t.Fatalf("IsSplitTarget(unrelated reason) = true, want false")
	}
	if IsSplitTarget(issuepolicy.Review{}) {
		t.Fatalf("IsSplitTarget(no reasons) = true, want false")
	}
}

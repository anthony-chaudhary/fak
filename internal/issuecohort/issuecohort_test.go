package issuecohort

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/issuecontract"
)

// fullCandidate returns a candidate that passes the (non-live) issuecontract
// review as a dispatchable leaf. Tests mutate the returned value to exercise one
// axis at a time.
func fullCandidate(key string) issuecontract.Candidate {
	return issuecontract.Candidate{
		Schema:         issuecontract.Schema,
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

	plan := Build([]issuecontract.Candidate{a, b}, Options{})
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

	plan := Build([]issuecontract.Candidate{a, b}, Options{})
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

	plan := Build([]issuecontract.Candidate{a, b}, Options{})
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

	plan := Build([]issuecontract.Candidate{a, b}, Options{})
	if plan.NumWaves != 1 || plan.PeakConcurrency != 2 {
		t.Fatalf("waves=%d peak=%d, want 1 wave of 2 (distinct lanes)", plan.NumWaves, plan.PeakConcurrency)
	}
}

func TestBuildOversizedSubdivides(t *testing.T) {
	a := fullCandidate("big")
	a.ExpectedSteps = 20

	plan := Build([]issuecontract.Candidate{a}, Options{})
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
	if !hasReason(plan.Subdivide[0].Reasons, issuecontract.ReasonOversizedSteps) {
		t.Fatalf("subdivide reasons = %v, want oversized", plan.Subdivide[0].Reasons)
	}
}

func TestBuildEpicSubdivides(t *testing.T) {
	a := fullCandidate("umbrella")
	a.WorkUnit = "epic"

	plan := Build([]issuecontract.Candidate{a}, Options{})
	if plan.Subdividable != 1 || len(plan.Subdivide) != 1 {
		t.Fatalf("subdividable=%d rows=%d, want 1/1", plan.Subdividable, len(plan.Subdivide))
	}
	if plan.Subdivide[0].ChildIssueBudget != 1 {
		t.Fatalf("child budget = %d, want 1 (unknown steps)", plan.Subdivide[0].ChildIssueBudget)
	}
	if !hasReason(plan.Subdivide[0].Reasons, issuecontract.ReasonNotDispatchLeaf) {
		t.Fatalf("reasons = %v, want not-dispatch-leaf", plan.Subdivide[0].Reasons)
	}
}

func TestBuildScopeGapTriaged(t *testing.T) {
	a := fullCandidate("vague")
	a.OutOfScope = ""    // scope incomplete
	a.DoneCondition = "" // and no done condition

	plan := Build([]issuecontract.Candidate{a}, Options{})
	if plan.TriageOnly != 1 || len(plan.Triage) != 1 {
		t.Fatalf("triageOnly=%d rows=%d, want 1/1", plan.TriageOnly, len(plan.Triage))
	}
	if plan.Subdividable != 0 || plan.Dispatchable != 0 {
		t.Fatalf("dispatchable=%d subdividable=%d, want 0/0", plan.Dispatchable, plan.Subdividable)
	}
	if !hasReason(plan.Triage[0].Reasons, issuecontract.ReasonScopeIncomplete) {
		t.Fatalf("triage reasons = %v, want scope-incomplete", plan.Triage[0].Reasons)
	}
}

func TestBuildDuplicateKeyDetected(t *testing.T) {
	a := fullCandidate("dupe")
	b := fullCandidate("dupe") // same key
	b.Paths = []string{"internal/other/**"}

	plan := Build([]issuecontract.Candidate{a, b}, Options{})
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
	var cands []issuecontract.Candidate
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

	plan := Build([]issuecontract.Candidate{a, b}, Options{})
	if plan.NumWaves != 1 {
		t.Fatalf("waves = %d, want 1", plan.NumWaves)
	}
	if got := plan.Waves[0].StepBudget; got != 5 { // 1 (unknown) + 4
		t.Fatalf("step budget = %d, want 5", got)
	}
}

func TestRenderSmoke(t *testing.T) {
	a := fullCandidate("alpha")
	b := fullCandidate("big")
	b.ExpectedSteps = 20
	plan := Build([]issuecontract.Candidate{a, b}, Options{})
	out := Render(plan)
	for _, want := range []string{"issue-cohort:", "concurrency:", "wave 0:", "split-first"} {
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

	plan := Build([]issuecontract.Candidate{a, b}, Options{})
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

	plan := Build([]issuecontract.Candidate{a, b}, Options{})
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

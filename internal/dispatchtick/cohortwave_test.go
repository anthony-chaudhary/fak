package dispatchtick

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/issuecohort"
	"github.com/anthony-chaudhary/fak/internal/issuepolicy"
)

// cohortLeaf returns a candidate that passes the (non-live) issuecontract review
// as a dispatchable leaf, scoped to the given paths. Mirrors the fixture used in
// internal/issuecohort so the parity check runs on realistic dispatch leaves.
func cohortLeaf(key string, paths ...string) issuepolicy.Candidate {
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
		Paths:          paths,
	}
}

// TestCohortWaveParityAgreesOnFixture is the Done-condition witness for #2076: a
// cohort plan partitioned by issuecohort's creation-time overlap rule contains no
// intra-wave collision under dispatchtick's own dispatch-time rule. Empty
// conflicts == the two overlap rules agree on this fixture, so the cohort waves
// seed the dispatch planner directly.
func TestCohortWaveParityAgreesOnFixture(t *testing.T) {
	// alpha owns internal/foo/**; beta lives inside it (collides -> next wave);
	// gamma is disjoint (shares alpha's wave); delta/epsilon are whole-lane docs
	// takers (collide with each other -> serialized).
	candidates := []issuepolicy.Candidate{
		cohortLeaf("alpha", "internal/foo/**"),
		cohortLeaf("beta", "internal/foo/bar.go"),
		cohortLeaf("gamma", "internal/baz/x.go"),
		func() issuepolicy.Candidate { c := cohortLeaf("delta"); c.Lane = "docs"; return c }(),
		func() issuepolicy.Candidate { c := cohortLeaf("epsilon"); c.Lane = "docs"; return c }(),
	}

	plan := issuecohort.Build(candidates, issuecohort.Options{})
	if plan.Dispatchable != len(candidates) {
		t.Fatalf("dispatchable = %d, want %d", plan.Dispatchable, len(candidates))
	}

	// The overlap rules agree: dispatchtick finds no collision inside any wave
	// issuecohort declared concurrency-safe.
	if conflicts := CohortWaveConflicts(plan); len(conflicts) != 0 {
		t.Fatalf("cohort/dispatch overlap rules disagree on fixture: %+v", conflicts)
	}

	// The seed preserves every wave and every member key.
	seeds := SeedWavesFromCohort(plan)
	if len(seeds) != plan.NumWaves {
		t.Fatalf("seeded %d waves, want %d", len(seeds), plan.NumWaves)
	}
	seededKeys := map[string]bool{}
	for _, s := range seeds {
		for _, k := range s.Keys {
			if seededKeys[k] {
				t.Fatalf("key %q seeded into more than one wave", k)
			}
			seededKeys[k] = true
		}
	}
	for _, want := range []string{"alpha", "beta", "gamma", "delta", "epsilon"} {
		if !seededKeys[want] {
			t.Fatalf("key %q missing from seeded waves", want)
		}
	}
}

// TestCohortWaveConflictsFlagsOverlappingMembers is the negative control: the
// parity detector is not vacuously empty. A hand-built wave that WRONGLY co-holds
// two overlapping members is flagged by dispatchtick's rule.
func TestCohortWaveConflictsFlagsOverlappingMembers(t *testing.T) {
	bad := issuecohort.Plan{
		Waves: []issuecohort.Wave{{
			Index: 0,
			Members: []issuecohort.WaveMember{
				{Key: "a", Paths: []string{"internal/foo/**"}},
				{Key: "b", Paths: []string{"internal/foo/bar.go"}}, // inside a's tree
			},
		}},
	}
	conflicts := CohortWaveConflicts(bad)
	if len(conflicts) != 1 {
		t.Fatalf("conflicts = %d, want 1", len(conflicts))
	}
	if conflicts[0].A != "a" || conflicts[0].B != "b" || conflicts[0].WaveIndex != 0 {
		t.Fatalf("unexpected conflict: %+v", conflicts[0])
	}
}

// TestCohortMembersCollideLaneRule pins the whole-lane branch: two members that
// name no paths take their whole lane, so they collide iff they share a lane.
func TestCohortMembersCollideLaneRule(t *testing.T) {
	docsA := issuecohort.WaveMember{Key: "a", Lane: "docs"}
	docsB := issuecohort.WaveMember{Key: "b", Lane: "docs"}
	toolsC := issuecohort.WaveMember{Key: "c", Lane: "tools"}
	if !cohortMembersCollide(docsA, docsB) {
		t.Fatal("same-lane whole-lane takers should collide")
	}
	if cohortMembersCollide(docsA, toolsC) {
		t.Fatal("distinct-lane whole-lane takers should not collide")
	}
}

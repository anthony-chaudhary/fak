package trajctl

import (
	"path/filepath"
	"testing"
)

// Real-shaped 40-hex SHAs so the fixture ledger reads like a genuine one; the
// injected resolver stands in for the git resolver that would confirm them.
const (
	shaP1 = "1a2b3c4d5e6f70818293a4b5c6d7e8f901234567"
	shaP2 = "89abcdef0123456789abcdef0123456789abcdef"
	shaP3 = "fedcba9876543210fedcba9876543210fedcba98"
)

// fourPhaseObjective is the done-condition fixture: an objective with a 4-phase
// plan.
func fourPhaseObjective() Objective {
	return Objective{
		ID:        "obj-spine",
		Statement: "ship the trajctl scorer spine",
		Status:    StatusActive,
		Plan: []PlanPhase{
			{ID: "p1", Title: "types"},
			{ID: "p2", Title: "scorer"},
			{ID: "p3", Title: "curve"},
			{ID: "p4", Title: "steer"},
		},
	}
}

// TestCommitProgressScorer_FourPhaseTwoWitnessed is the issue #2536 done
// condition: a 4-phase plan with two phases bound to witnessed commits yields
// value 0.5, W3, with commit evidence pointers that resolve. The produced row is
// then round-tripped through a real ledger file to prove it is a valid, durable
// score record (the fixture-ledger witness).
func TestCommitProgressScorer_FourPhaseTwoWitnessed(t *testing.T) {
	obj := fourPhaseObjective()
	win := EvidenceWindow{
		PhaseCommits: map[string][]string{
			"p1": {shaP1}, // verified
			"p2": {shaP2}, // verified
			"p3": {shaP3}, // present but dangling -> no progress
			// p4 has no commit at all -> no progress
		},
		Resolve: fixtureResolver(shaP3), // shaP3 dangling; shaP1/shaP2 verified
	}

	rows := CommitProgressScorer{}.Score(obj, win)
	if len(rows) != 1 {
		t.Fatalf("Score returned %d rows, want 1", len(rows))
	}
	got := rows[0]

	if got.Value != 0.5 {
		t.Errorf("Value = %v, want 0.5 (2 of 4 phases witnessed)", got.Value)
	}
	if got.Witness != W3 {
		t.Errorf("Witness = %q, want W3 (a witnessed commit is deterministic)", got.Witness)
	}
	if got.Method != CommitScorerMethod || got.Version != CommitScorerVersion {
		t.Errorf("Method/Version = %q/%q, want %q/%q", got.Method, got.Version, CommitScorerMethod, CommitScorerVersion)
	}
	if len(got.Evidence) != 2 {
		t.Fatalf("Evidence has %d refs, want 2 (one per witnessed phase)", len(got.Evidence))
	}
	for _, ev := range got.Evidence {
		if ev.Kind != "commit" {
			t.Errorf("evidence kind = %q, want commit", ev.Kind)
		}
		if ev.Ref != shaP1 && ev.Ref != shaP2 {
			t.Errorf("evidence ref = %q, want a witnessed SHA (%s or %s)", ev.Ref, shaP1, shaP2)
		}
	}

	// The score must be a durable ledger row: append the objective and the score,
	// read them back, and confirm the fold surfaces exactly this value with its
	// W3 evidence pointers intact (Append validates, so a malformed row would
	// error here).
	path := filepath.Join(t.TempDir(), "traj.jsonl")
	if err := Append(path, ObjectiveRecord(obj)); err != nil {
		t.Fatalf("append objective: %v", err)
	}
	if err := Append(path, ScoreRecord(got)); err != nil {
		t.Fatalf("append score: %v", err)
	}
	st := Fold(ReadLedgerFile(path))
	scores := st.ScoresFor(obj.ID)
	if len(scores) != 1 {
		t.Fatalf("ScoresFor returned %d rows, want 1", len(scores))
	}
	if scores[0].Value != 0.5 || scores[0].Witness != W3 || len(scores[0].Evidence) != 2 {
		t.Errorf("folded score = %+v, want value 0.5 / W3 / 2 evidence refs", scores[0])
	}
}

// TestCommitProgressScorer_NilResolverFailsClosed proves the scorer is
// fail-closed: with no resolver it can verify nothing, so it credits no progress
// rather than trusting the claimed bindings.
func TestCommitProgressScorer_NilResolverFailsClosed(t *testing.T) {
	obj := fourPhaseObjective()
	win := EvidenceWindow{
		PhaseCommits: map[string][]string{"p1": {shaP1}, "p2": {shaP2}},
		Resolve:      nil,
	}
	rows := CommitProgressScorer{}.Score(obj, win)
	if len(rows) != 1 {
		t.Fatalf("Score returned %d rows, want 1", len(rows))
	}
	if rows[0].Value != 0 {
		t.Errorf("Value = %v, want 0 (nil resolver verifies nothing)", rows[0].Value)
	}
	if len(rows[0].Evidence) != 0 {
		t.Errorf("Evidence = %v, want none", rows[0].Evidence)
	}
}

// TestCommitProgressScorer_EmptyPlanScoresNothing: an objective with no plan has
// nothing to score against, so the scorer emits no row (rather than a 0/0 NaN).
func TestCommitProgressScorer_EmptyPlanScoresNothing(t *testing.T) {
	obj := Objective{ID: "obj-noplan", Statement: "no plan", Status: StatusActive}
	rows := CommitProgressScorer{}.Score(obj, EvidenceWindow{Resolve: fixtureResolver()})
	if rows != nil {
		t.Errorf("Score = %+v, want nil for an objective with no plan", rows)
	}
}

// TestCommitProgressScorer_AllPhasesWitnessed rounds out the fraction: every
// phase bound to a verified commit scores a full 1.0.
func TestCommitProgressScorer_AllPhasesWitnessed(t *testing.T) {
	obj := Objective{
		ID:        "obj-full",
		Statement: "fully witnessed",
		Status:    StatusActive,
		Plan:      []PlanPhase{{ID: "a"}, {ID: "b"}},
	}
	win := EvidenceWindow{
		PhaseCommits: map[string][]string{"a": {shaP1}, "b": {shaP2}},
		Resolve:      fixtureResolver(),
	}
	rows := CommitProgressScorer{}.Score(obj, win)
	if rows[0].Value != 1 {
		t.Errorf("Value = %v, want 1 (all phases witnessed)", rows[0].Value)
	}
}

// badScorer exercises the registry's guards.
type badScorer struct {
	method, version string
}

func (b badScorer) Method() string  { return b.method }
func (b badScorer) Version() string { return b.version }
func (b badScorer) Score(Objective, EvidenceWindow) []ScoreRow {
	return nil
}

// TestScorerRegistry_RegisterGetDupAndMethods proves the registry is the
// one-step extension seam and refuses silent overwrites and malformed scorers.
func TestScorerRegistry_RegisterGetDupAndMethods(t *testing.T) {
	r := NewRegistry()

	if err := r.Register(CommitProgressScorer{}); err != nil {
		t.Fatalf("Register commit scorer: %v", err)
	}
	got, ok := r.Get(CommitScorerMethod)
	if !ok {
		t.Fatalf("Get(%q) missing after Register", CommitScorerMethod)
	}
	if got.Version() != CommitScorerVersion {
		t.Errorf("registered version = %q, want %q", got.Version(), CommitScorerVersion)
	}

	// A second scorer claiming the same method is an error, not an overwrite.
	if err := r.Register(CommitProgressScorer{}); err == nil {
		t.Error("re-registering a live method: want error, got nil")
	}
	// Nil, empty method, and empty version are all refused.
	if err := r.Register(nil); err == nil {
		t.Error("Register(nil): want error, got nil")
	}
	if err := r.Register(badScorer{method: "", version: "1"}); err == nil {
		t.Error("Register(empty method): want error, got nil")
	}
	if err := r.Register(badScorer{method: "m", version: ""}); err == nil {
		t.Error("Register(empty version): want error, got nil")
	}

	// A second valid method registers and Methods() lists both in lexical order.
	if err := r.Register(badScorer{method: "aardvark", version: "1"}); err != nil {
		t.Fatalf("Register second method: %v", err)
	}
	methods := r.Methods()
	if len(methods) != 2 || methods[0] != "aardvark" || methods[1] != CommitScorerMethod {
		t.Errorf("Methods() = %v, want [aardvark %s]", methods, CommitScorerMethod)
	}
}

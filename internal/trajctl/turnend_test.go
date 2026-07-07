package trajctl

import (
	"path/filepath"
	"strings"
	"testing"
)

// panicScorer is a deliberately buggy scorer: it panics on every Score call, to
// prove the runner's fail-open recover shield.
type panicScorer struct{}

func (panicScorer) Method() string                             { return "boom" }
func (panicScorer) Version() string                            { return "1" }
func (panicScorer) Score(Objective, EvidenceWindow) []ScoreRow { panic("scorer bug") }

// TestSample_PlannedObjectivePointPerTurn is the issue #2539 done condition: a
// session with a declared (planned) objective accumulates at least one ScoreRow
// per turn end, and each row is attributed to the firing session and run.
func TestSample_PlannedObjectivePointPerTurn(t *testing.T) {
	obj := fourPhaseObjective() // active, 4-phase plan
	win := EvidenceWindow{
		PhaseCommits: map[string][]string{"p1": {shaP1}, "p2": {shaP2}},
		Resolve:      fixtureResolver(), // everything verified
		UnixMillis:   1234,
	}
	objs := map[string]Objective{obj.ID: obj}

	sample := Sample(objs, []Scorer{CommitProgressScorer{}}, win, Stamp{SessionID: "sess-1", RunID: "run-1"})

	if sample.Reason != ReasonTurnEnd {
		t.Errorf("Reason = %q, want %q", sample.Reason, ReasonTurnEnd)
	}
	if len(sample.Rows) < 1 {
		t.Fatalf("Sample produced %d rows, want >= 1 (the per-turn point)", len(sample.Rows))
	}
	got := sample.Rows[0]
	if got.Method != CommitScorerMethod {
		t.Errorf("Method = %q, want %q", got.Method, CommitScorerMethod)
	}
	if got.Value != 0.5 {
		t.Errorf("Value = %v, want 0.5 (2 of 4 phases witnessed)", got.Value)
	}
	if got.SessionID != "sess-1" || got.RunID != "run-1" {
		t.Errorf("attribution = (%q,%q), want (sess-1,run-1)", got.SessionID, got.RunID)
	}
	if got.UnixMillis != 1234 {
		t.Errorf("UnixMillis = %d, want 1234 (window stamp)", got.UnixMillis)
	}
	if len(sample.Failures) != 0 {
		t.Errorf("Failures = %v, want none", sample.Failures)
	}
}

// TestSample_ScoresOnlyOpenObjectives proves a met or abandoned objective's curve
// is closed: only active/paused objectives are sampled at turn end.
func TestSample_ScoresOnlyOpenObjectives(t *testing.T) {
	plan := []PlanPhase{{ID: "p1"}}
	objs := map[string]Objective{
		"open-active":      {ID: "open-active", Statement: "x", Status: StatusActive, Plan: plan},
		"open-paused":      {ID: "open-paused", Statement: "x", Status: StatusPaused, Plan: plan},
		"closed-met":       {ID: "closed-met", Statement: "x", Status: StatusMet, Plan: plan},
		"closed-abandoned": {ID: "closed-abandoned", Statement: "x", Status: StatusAbandoned, Plan: plan},
	}
	win := EvidenceWindow{Resolve: fixtureResolver(), UnixMillis: 1}

	sample := Sample(objs, []Scorer{CommitProgressScorer{}}, win, Stamp{})

	scored := map[string]bool{}
	for _, r := range sample.Rows {
		scored[r.ObjectiveID] = true
	}
	if !scored["open-active"] || !scored["open-paused"] {
		t.Errorf("open objectives not scored: %v", scored)
	}
	if scored["closed-met"] || scored["closed-abandoned"] {
		t.Errorf("closed objectives were scored: %v", scored)
	}
}

// TestSample_FailOpenOnScorerPanic is the fail-open rung: an injected scorer panic
// is swallowed into a SampleFailure, the turn does not unwind, and the healthy
// scorer alongside it still contributes its row.
func TestSample_FailOpenOnScorerPanic(t *testing.T) {
	obj := fourPhaseObjective()
	win := EvidenceWindow{
		PhaseCommits: map[string][]string{"p1": {shaP1}},
		Resolve:      fixtureResolver(),
		UnixMillis:   7,
	}
	objs := map[string]Objective{obj.ID: obj}

	// panicScorer first so the panic happens before the healthy scorer runs.
	sample := Sample(objs, []Scorer{panicScorer{}, CommitProgressScorer{}}, win, Stamp{})

	if len(sample.Failures) != 1 {
		t.Fatalf("Failures = %d, want 1 (the panicking scorer)", len(sample.Failures))
	}
	f := sample.Failures[0]
	if f.Method != "boom" || f.ObjectiveID != obj.ID {
		t.Errorf("failure = %+v, want method boom on %q", f, obj.ID)
	}
	if !strings.Contains(f.Detail, "panicked") {
		t.Errorf("Detail = %q, want it to mention the panic", f.Detail)
	}
	// The healthy scorer still produced its row — the panic cost only its own row.
	if len(sample.Rows) != 1 || sample.Rows[0].Method != CommitScorerMethod {
		t.Errorf("Rows = %+v, want the one witnessed-commit-progress row", sample.Rows)
	}
}

// TestCompactionBoundary_MarkerPerOpenObjective proves the PreCompact twin emits
// one valid, own-series boundary marker per open objective and none for closed
// ones.
func TestCompactionBoundary_MarkerPerOpenObjective(t *testing.T) {
	objs := map[string]Objective{
		"a": {ID: "a", Statement: "x", Status: StatusActive},
		"b": {ID: "b", Statement: "x", Status: StatusPaused},
		"c": {ID: "c", Statement: "x", Status: StatusMet}, // closed -> no marker
	}
	win := EvidenceWindow{UnixMillis: 999}

	sample := CompactionBoundary(objs, win, Stamp{SessionID: "s"})

	if sample.Reason != ReasonCompaction {
		t.Errorf("Reason = %q, want %q", sample.Reason, ReasonCompaction)
	}
	if len(sample.Rows) != 2 {
		t.Fatalf("markers = %d, want 2 (a and b, not c)", len(sample.Rows))
	}
	for _, r := range sample.Rows {
		if r.Method != CompactionBoundaryMethod {
			t.Errorf("marker method = %q, want %q", r.Method, CompactionBoundaryMethod)
		}
		if r.Witness != W0 || r.Value != 0 {
			t.Errorf("marker = (witness %q, value %v), want (W0, 0)", r.Witness, r.Value)
		}
		if r.UnixMillis != 999 {
			t.Errorf("UnixMillis = %d, want 999", r.UnixMillis)
		}
		// Every marker must be a durable, valid ledger row.
		if err := Validate(ScoreRecord(r)); err != nil {
			t.Errorf("marker fails Validate: %v", err)
		}
	}
}

// TestAppendSample_RoundTrip proves the thin I/O wrapper writes rows the ledger
// reader parses back — the stop-hook's one-liner is durable.
func TestAppendSample_RoundTrip(t *testing.T) {
	objs := map[string]Objective{
		"a": {ID: "a", Statement: "x", Status: StatusActive},
		"b": {ID: "b", Statement: "x", Status: StatusActive},
	}
	win := EvidenceWindow{UnixMillis: 42}
	sample := CompactionBoundary(objs, win, Stamp{SessionID: "s"})

	path := filepath.Join(t.TempDir(), "trajctl.jsonl")
	n, err := AppendSample(path, sample)
	if err != nil {
		t.Fatalf("AppendSample: %v", err)
	}
	if n != len(sample.Rows) {
		t.Fatalf("appended %d, want %d", n, len(sample.Rows))
	}

	rows := ReadLedgerFile(path)
	if len(rows) != n {
		t.Fatalf("read back %d rows, want %d", len(rows), n)
	}
	for _, row := range rows {
		if row.Kind != KindScore || row.Score == nil {
			t.Fatalf("row kind = %q, want a score row", row.Kind)
		}
		if row.Score.Method != CompactionBoundaryMethod {
			t.Errorf("round-tripped method = %q, want %q", row.Score.Method, CompactionBoundaryMethod)
		}
	}
}

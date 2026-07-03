package trajctl

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestLedgerRoundTripsObjectiveAndEveryWitnessRung(t *testing.T) {
	path := filepath.Join(t.TempDir(), "trajctl", "ledger.jsonl")
	obj := Objective{
		ID:        "obj-1",
		ParentID:  "epic-1",
		Statement: "ship the trajectory control spine",
		Plan: []PlanPhase{
			{ID: "p1", Title: "types"},
			{ID: "p2", Title: "declare"},
			{ID: "p3", Title: "score"},
			{ID: "p4", Title: "curve"},
		},
		Scorers: []string{"commit-progress"},
		Budget:  Budget{Turns: 12, Tokens: 50000},
		Status:  StatusActive,
	}

	if err := Append(path, ObjectiveRecord(obj)); err != nil {
		t.Fatalf("append objective: %v", err)
	}
	rungs := []WitnessRung{W3, W2, W1, W0}
	for i, rung := range rungs {
		row := ScoreRow{
			ObjectiveID: obj.ID,
			Value:       float64(i) / float64(len(rungs)),
			Method:      "commit-progress",
			Version:     "v1",
			Witness:     rung,
			Evidence:    []EvidenceRef{{Kind: "commit", Ref: "0123456789abcdef0123456789abcdef01234567"}},
			UnixMillis:  int64(1000 + i),
			SessionID:   "session-a",
			RunID:       "run-a",
		}
		if err := Append(path, ScoreRecord(row)); err != nil {
			t.Fatalf("append score %s: %v", rung, err)
		}
	}

	rows := ReadLedgerFile(path)
	if len(rows) != 5 {
		t.Fatalf("rows = %d, want 5", len(rows))
	}
	state := Fold(rows)
	got, ok := state.Objectives[obj.ID]
	if !ok {
		t.Fatalf("objective %q missing from fold", obj.ID)
	}
	if got.ParentID != obj.ParentID || got.Status != StatusActive || len(got.Plan) != 4 {
		t.Fatalf("folded objective = %+v", got)
	}
	scores := state.ScoresFor(obj.ID)
	if len(scores) != len(rungs) {
		t.Fatalf("scores = %d, want %d", len(scores), len(rungs))
	}
	for i, rung := range rungs {
		if scores[i].Witness != rung {
			t.Fatalf("score %d witness = %q, want %q", i, scores[i].Witness, rung)
		}
		if scores[i].Evidence[0].Ref == "" {
			t.Fatalf("score %d lost evidence ref", i)
		}
	}
}

func TestFoldKeepsLatestObjectiveAndScoreHistory(t *testing.T) {
	active := Objective{ID: "obj-1", Statement: "ship", Status: StatusActive}
	met := active
	met.Status = StatusMet
	rows := []Row{
		ObjectiveRecord(active),
		ScoreRecord(ScoreRow{ObjectiveID: "obj-1", Value: 0.25, Method: "m", Version: "v1", Witness: W3}),
		ObjectiveRecord(met),
		ScoreRecord(ScoreRow{ObjectiveID: "obj-1", Value: 1, Method: "m", Version: "v1", Witness: W3}),
		ObjectiveRecord(Objective{ID: "obj-2", Statement: "other", Status: StatusPaused}),
	}

	state := Fold(rows)
	if state.Objectives["obj-1"].Status != StatusMet {
		t.Fatalf("latest objective status = %q, want %q", state.Objectives["obj-1"].Status, StatusMet)
	}
	if got := state.ScoresFor("obj-1"); len(got) != 2 || got[0].Value != 0.25 || got[1].Value != 1 {
		t.Fatalf("score history = %+v", got)
	}
	ids := state.ObjectiveIDs()
	if len(ids) != 2 || ids[0] != "obj-1" || ids[1] != "obj-2" {
		t.Fatalf("objective ids = %v", ids)
	}
}

func TestParseLedgerSkipsMalformedForeignAndInvalidRows(t *testing.T) {
	valid := mustJSON(t, ObjectiveRecord(Objective{ID: "obj-1", Statement: "ship", Status: StatusActive}))
	foreign := `{"schema":"other","kind":"objective","objective":{"id":"obj-2"}}`
	invalid := mustJSON(t, ScoreRecord(ScoreRow{ObjectiveID: "obj-1", Value: 2, Method: "m", Version: "v1", Witness: W3}))
	rows := ParseLedger("\nnot json\n" + foreign + "\n" + invalid + "\n" + valid + "\n")
	if len(rows) != 1 {
		t.Fatalf("rows = %d, want only the valid row", len(rows))
	}
	if rows[0].Objective.ID != "obj-1" {
		t.Fatalf("parsed row = %+v", rows[0])
	}
}

func TestValidateRejectsInvalidRows(t *testing.T) {
	cases := []Row{
		ObjectiveRecord(Objective{ID: "", Statement: "ship", Status: StatusActive}),
		ObjectiveRecord(Objective{ID: "obj", Statement: "", Status: StatusActive}),
		ObjectiveRecord(Objective{ID: "obj", Statement: "ship", Status: "done"}),
		ObjectiveRecord(Objective{ID: "obj", Statement: "ship", Status: StatusActive, Plan: []PlanPhase{{ID: "p1"}, {ID: "p1"}}}),
		ScoreRecord(ScoreRow{ObjectiveID: "obj", Value: -0.1, Method: "m", Version: "v1", Witness: W3}),
		ScoreRecord(ScoreRow{ObjectiveID: "obj", Value: 0.5, Method: "", Version: "v1", Witness: W3}),
		ScoreRecord(ScoreRow{ObjectiveID: "obj", Value: 0.5, Method: "m", Version: "v1", Witness: "W9"}),
		ScoreRecord(ScoreRow{ObjectiveID: "obj", Value: 0.5, Method: "m", Version: "v1", Witness: W3, Evidence: []EvidenceRef{{Kind: "commit"}}}),
	}
	for i, row := range cases {
		if err := Validate(row); err == nil {
			t.Fatalf("case %d Validate succeeded, want error", i)
		}
	}
}

func TestReadLedgerFileMissingAndAppendCreatesParents(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "nested", "ledger.jsonl")
	if got := ReadLedgerFile(path); got != nil {
		t.Fatalf("missing ledger = %+v, want nil", got)
	}
	if err := Append(path, ObjectiveRecord(Objective{ID: "obj", Statement: "ship", Status: StatusActive})); err != nil {
		t.Fatalf("append: %v", err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("ledger not created: %v", err)
	}
}

func mustJSON(t *testing.T, row Row) string {
	t.Helper()
	b, err := json.Marshal(row)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

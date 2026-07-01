package fleetmon

import (
	"strings"
	"testing"
	"time"
)

func foldTranscript(t *testing.T, issue int, session string, lines []string, alive *bool) LedgerRow {
	t.Helper()
	sig := ReadTranscript(writeTranscript(t, lines...))
	return FoldWorker(FoldInput{
		Worker:     PlanWorker{Issue: issue, Session: session},
		Transcript: sig,
		PIDAlive:   alive,
		Now:        time.Now(),
	})
}

// TestFoldFourWitnessCases is the issue's witness: one read-only scoped issue, one
// patch-with-witness issue, one idle/no-final issue, and one crashed/stale busy
// issue — the fold must classify all four correctly from synthetic transcripts.
func TestFoldFourWitnessCases(t *testing.T) {
	now := time.Now()

	// 1) read-only audit: a final report, no file changes.
	readonly := foldTranscript(t, 1, "issue-1", []string{
		assistantToolUse(t, tsAt(now, 5), "Read", map[string]any{"file_path": "a.go"}),
		assistantToolUse(t, tsAt(now, 4), "Grep", map[string]any{"pattern": "x"}),
		assistantText(t, tsAt(now, 1), "Audited: the code is correct, no change needed.", "end_turn"),
	}, bptr(true))
	if readonly.Outcome != string(OutcomeReadOnlyAudit) {
		t.Errorf("read-only case: want %s, got %s", OutcomeReadOnlyAudit, readonly.Outcome)
	}

	// 2) patch-with-witness: changed files + a captured test witness + final report.
	patch := foldTranscript(t, 2, "issue-2", []string{
		assistantToolUse(t, tsAt(now, 8), "Edit", map[string]any{"file_path": "internal/x/x.go"}),
		assistantToolUse(t, tsAt(now, 5), "Bash", map[string]any{"command": "go test ./internal/x/"}),
		userToolResult(t, tsAt(now, 4), "ok  internal/x  0.2s", false),
		assistantText(t, tsAt(now, 1), "Fixed and verified with go test.", "end_turn"),
	}, bptr(true))
	if patch.Outcome != string(OutcomePatchWitness) {
		t.Errorf("patch case: want %s, got %s", OutcomePatchWitness, patch.Outcome)
	}
	if len(patch.ChangedFiles) == 0 || patch.Witness == "" {
		t.Errorf("patch-with-witness must carry changed files + witness, got %+v", patch)
	}

	// 3) idle/no-final: last turn is a pending tool_use — NOT complete.
	idle := foldTranscript(t, 3, "issue-3", []string{
		assistantText(t, tsAt(now, 20), "Working.", "end_turn"),
		assistantToolUse(t, tsAt(now, 2), "Bash", map[string]any{"command": "ls"}),
	}, bptr(true))
	if idle.Outcome != string(OutcomeStaleIncomplete) {
		t.Errorf("idle/no-final case: want %s, got %s", OutcomeStaleIncomplete, idle.Outcome)
	}

	// 4) crashed/stale busy: no final report AND the process is gone.
	crashed := foldTranscript(t, 4, "issue-4", []string{
		assistantToolUse(t, tsAt(now, 30), "Bash", map[string]any{"command": "go build ./..."}),
	}, bptr(false))
	if crashed.Outcome != string(OutcomeCrashedNoFinal) {
		t.Errorf("crashed case: want %s, got %s", OutcomeCrashedNoFinal, crashed.Outcome)
	}
}

func TestFoldBlockedScoped(t *testing.T) {
	now := time.Now()
	row := foldTranscript(t, 5, "issue-5", []string{
		assistantToolUse(t, tsAt(now, 5), "Read", map[string]any{"file_path": "a.go"}),
		assistantText(t, tsAt(now, 1), "Not yet — this is blocked on a design decision. Follow-up: confirm the API shape with the owner.", "end_turn"),
	}, bptr(true))
	if row.Outcome != string(OutcomeBlockedScoped) {
		t.Fatalf("blocked final report: want %s, got %s", OutcomeBlockedScoped, row.Outcome)
	}
	if row.FollowUp == "" {
		t.Error("blocked-scoped must carry the smallest follow-up")
	}
}

func TestFoldChangedFilesNoWitnessIsNotPatchWitness(t *testing.T) {
	now := time.Now()
	row := foldTranscript(t, 6, "issue-6", []string{
		assistantToolUse(t, tsAt(now, 5), "Edit", map[string]any{"file_path": "a.go"}),
		assistantText(t, tsAt(now, 1), "Changed a.go.", "end_turn"),
	}, bptr(true))
	if row.Outcome == string(OutcomePatchWitness) {
		t.Fatal("a change with no captured witness is a claim, not patch-with-witness")
	}
	if row.Outcome != string(OutcomeBlockedScoped) {
		t.Fatalf("want blocked-scoped (missing witness), got %s", row.Outcome)
	}
}

func TestFoldSuperseded(t *testing.T) {
	now := time.Now()
	row := FoldWorker(FoldInput{
		Worker:       PlanWorker{Issue: 7, Session: "issue-7"},
		Transcript:   TranscriptSignal{},
		Superseded:   true,
		SupersededBy: "issue-7-replacement-1",
		Now:          now,
	})
	if row.Outcome != string(OutcomeSuperseded) {
		t.Fatalf("want superseded, got %s", row.Outcome)
	}
	if row.SupersededBy != "issue-7-replacement-1" {
		t.Fatalf("superseded_by should link the replacement, got %q", row.SupersededBy)
	}
}

func TestFoldInterruptedThenRecovered(t *testing.T) {
	now := time.Now()
	// An interrupted tool result mid-run, then the worker recovered and produced a
	// final report with a witness — this is a completed patch, not a crash.
	row := foldTranscript(t, 8, "issue-8", []string{
		assistantToolUse(t, tsAt(now, 20), "Bash", map[string]any{"command": "go test ./..."}),
		userToolResult(t, tsAt(now, 19), "[Request interrupted by user]", true),
		assistantToolUse(t, tsAt(now, 10), "Edit", map[string]any{"file_path": "a.go"}),
		assistantToolUse(t, tsAt(now, 6), "Bash", map[string]any{"command": "go test ./..."}),
		userToolResult(t, tsAt(now, 5), "ok", false),
		assistantText(t, tsAt(now, 1), "Recovered and fixed; go test passes.", "end_turn"),
	}, bptr(true))
	if row.Outcome != string(OutcomePatchWitness) {
		t.Fatalf("interrupted-then-recovered with a witness should be patch-with-witness, got %s", row.Outcome)
	}
}

func TestLedgerRoundTrip(t *testing.T) {
	row := LedgerRow{Schema: RunLedgerSchema, Issue: 1, Session: "issue-1", Outcome: string(OutcomeReadOnlyAudit), RecordedAt: "2026-07-01T00:00:00Z"}
	line, err := AppendLedgerLine(row)
	if err != nil {
		t.Fatal(err)
	}
	got := ParseLedger(line + "\n")
	if len(got) != 1 || got[0].Session != "issue-1" {
		t.Fatalf("round trip failed: %+v", got)
	}
}

func TestLedgerParseSkipsGarbage(t *testing.T) {
	content := strings.Join([]string{
		`{"schema":"x","issue":1,"session":"a","outcome":"read-only-audit","recorded_at":"t"}`,
		``,
		`not json`,
		`{"session":"","issue":0}`, // no id — dropped
		`{"schema":"x","issue":2,"session":"b","outcome":"crashed-no-final","recorded_at":"t"}`,
	}, "\n")
	rows := ParseLedger(content)
	if len(rows) != 2 {
		t.Fatalf("want 2 valid rows, got %d: %+v", len(rows), rows)
	}
}

func TestValidateLedgerEvidenceInvariants(t *testing.T) {
	rows := []LedgerRow{
		{Outcome: string(OutcomePatchWitness)},                                                     // missing files + witness
		{Outcome: string(OutcomePatchWitness), ChangedFiles: []string{"a.go"}, Witness: "go test"}, // ok
		{Outcome: string(OutcomeBlockedScoped)},                                                    // missing blocker + follow-up
		{Outcome: string(OutcomeSuperseded)},                                                       // missing superseded_by
		{Outcome: "made-up-outcome"},                                                               // off-schema
	}
	defects := ValidateLedger(rows)
	if len(defects) == 0 {
		t.Fatal("expected defects for the invalid rows")
	}
	// The one valid row (index 1, line 2) must not appear in defects.
	for _, d := range defects {
		if d.Line == 2 {
			t.Errorf("the valid patch-with-witness row must not be a defect: %+v", d)
		}
	}
	// patch-with-witness missing both fields => two defects on line 1.
	line1 := 0
	for _, d := range defects {
		if d.Line == 1 {
			line1++
		}
	}
	if line1 != 2 {
		t.Errorf("row 1 should have 2 defects (no files, no witness), got %d", line1)
	}
}

func TestSummarizeHistogram(t *testing.T) {
	rows := []LedgerRow{
		{Outcome: string(OutcomeReadOnlyAudit), Session: "a", Issue: 1},
		{Outcome: string(OutcomeReadOnlyAudit), Session: "b", Issue: 2},
		{Outcome: string(OutcomeCrashedNoFinal), Session: "c", Issue: 3},
	}
	s := Summarize("run", rows)
	if s.ByOutcome[OutcomeReadOnlyAudit] != 2 || s.ByOutcome[OutcomeCrashedNoFinal] != 1 {
		t.Fatalf("histogram wrong: %+v", s.ByOutcome)
	}
	if s.Total != 3 {
		t.Fatalf("want total 3, got %d", s.Total)
	}
}

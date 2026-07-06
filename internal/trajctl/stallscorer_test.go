package trajctl

import (
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/sessionaudit"
)

func TestActivityDivergenceScorer_StalledSessionEmitsW2(t *testing.T) {
	obj := fourPhaseObjective()
	session := sessionaudit.Analyze(filepath.Join("testdata", "stalled-session.jsonl"))
	if session.Error != "" {
		t.Fatalf("analyze stalled fixture: %s", session.Error)
	}
	win := EvidenceWindow{
		PriorScores: []ScoreRow{
			commitScore(obj.ID, 0.25),
			commitScore(obj.ID, 0.25),
		},
		Sessions:   []sessionaudit.Session{session},
		UnixMillis: 123,
	}

	rows := (ActivityDivergenceScorer{}).Score(obj, win)
	if len(rows) != 1 {
		t.Fatalf("Score returned %d rows, want 1", len(rows))
	}
	got := rows[0]
	if got.Witness != W2 || got.Method != ActivityDivergenceScorerMethod || got.Version != ActivityDivergenceScorerVersion {
		t.Fatalf("row identity = witness %q method %q version %q", got.Witness, got.Method, got.Version)
	}
	if got.Value < defaultActivityDivergenceThreshold {
		t.Fatalf("Value = %.3f, want above threshold %.2f", got.Value, defaultActivityDivergenceThreshold)
	}
	if got.SessionID == "" {
		t.Fatalf("SessionID should be set from sessionaudit")
	}
	if len(got.Evidence) != 1 || got.Evidence[0].Kind != "transcript-span" || got.Evidence[0].Ref == "" {
		t.Fatalf("Evidence = %+v, want transcript-span ref", got.Evidence)
	}

	path := filepath.Join(t.TempDir(), "trajctl.jsonl")
	if err := Append(path, ObjectiveRecord(obj)); err != nil {
		t.Fatalf("append objective: %v", err)
	}
	if err := Append(path, ScoreRecord(got)); err != nil {
		t.Fatalf("append score: %v", err)
	}
	if scores := Fold(ReadLedgerFile(path)).ScoresFor(obj.ID); len(scores) != 1 || scores[0].Witness != W2 {
		t.Fatalf("folded scores = %+v, want one W2 score", scores)
	}
}

func TestActivityDivergenceScorer_HealthyProgressStaysSilent(t *testing.T) {
	obj := fourPhaseObjective()
	session := sessionaudit.Analyze(filepath.Join("testdata", "healthy-session.jsonl"))
	if session.Error != "" {
		t.Fatalf("analyze healthy fixture: %s", session.Error)
	}
	win := EvidenceWindow{
		PriorScores: []ScoreRow{
			commitScore(obj.ID, 0.25),
			commitScore(obj.ID, 0.75),
		},
		Sessions: []sessionaudit.Session{session},
	}

	if rows := (ActivityDivergenceScorer{}).Score(obj, win); len(rows) != 0 {
		t.Fatalf("healthy progressing session emitted rows: %+v", rows)
	}
}

func TestActivityDivergenceScorer_NoCurveOrLowActivityStaysSilent(t *testing.T) {
	obj := fourPhaseObjective()
	session := sessionaudit.Session{Session: "quiet", AssistantTurns: 1, NToolUse: 1}
	for _, tc := range []struct {
		name string
		win  EvidenceWindow
	}{
		{name: "no prior curve", win: EvidenceWindow{Sessions: []sessionaudit.Session{session}}},
		{name: "one prior point", win: EvidenceWindow{PriorScores: []ScoreRow{commitScore(obj.ID, 0)}, Sessions: []sessionaudit.Session{session}}},
		{name: "flat but quiet", win: EvidenceWindow{PriorScores: []ScoreRow{commitScore(obj.ID, 0), commitScore(obj.ID, 0)}, Sessions: []sessionaudit.Session{session}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if rows := (ActivityDivergenceScorer{}).Score(obj, tc.win); len(rows) != 0 {
				t.Fatalf("Score emitted rows: %+v", rows)
			}
		})
	}
}

func commitScore(objID string, value float64) ScoreRow {
	return ScoreRow{
		ObjectiveID: objID,
		Value:       value,
		Method:      CommitScorerMethod,
		Version:     CommitScorerVersion,
		Witness:     W3,
	}
}

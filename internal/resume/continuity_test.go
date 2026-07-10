package resume

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/trajctl"
)

// w3 builds a W3 commit-progress ScoreRow at a millis stamp for objective obj.
func w3(obj string, value float64, unixMillis int64) trajctl.ScoreRow {
	return trajctl.ScoreRow{
		ObjectiveID: obj,
		Value:       value,
		Method:      trajctl.CommitScorerMethod,
		Version:     "v1",
		Witness:     trajctl.W3,
		UnixMillis:  unixMillis,
	}
}

// TestFoldW3Continuity pins the verified-progress cursor compare across a launch boundary:
// advance, flat (witnessed but not advanced), un-witnessed, and the fail-closed handling of
// unstamped / non-W3 rows.
func TestFoldW3Continuity(t *testing.T) {
	const launch = 1_000
	tests := []struct {
		name          string
		rows          []trajctl.ScoreRow
		obj           string
		since         int64
		wantWitnessed bool
		wantAdvanced  bool
		wantW3Rows    int
	}{
		{
			name:          "no W3 rows is the un-witnessed floor",
			rows:          nil,
			since:         launch,
			wantWitnessed: false, wantAdvanced: false, wantW3Rows: 0,
		},
		{
			name:          "post cursor above pre cursor advances",
			rows:          []trajctl.ScoreRow{w3("o1", 0.25, 500), w3("o1", 0.5, 1_500)},
			obj:           "o1",
			since:         launch,
			wantWitnessed: true, wantAdvanced: true, wantW3Rows: 2,
		},
		{
			name:          "post cursor equal to pre cursor does NOT advance",
			rows:          []trajctl.ScoreRow{w3("o1", 0.5, 500), w3("o1", 0.5, 1_500)},
			obj:           "o1",
			since:         launch,
			wantWitnessed: true, wantAdvanced: false, wantW3Rows: 2,
		},
		{
			name:          "witnessed pre-curve with no post row is flat (took_no_progress shape)",
			rows:          []trajctl.ScoreRow{w3("o1", 0.5, 500)},
			obj:           "o1",
			since:         launch,
			wantWitnessed: true, wantAdvanced: false, wantW3Rows: 1,
		},
		{
			name:          "unstamped W3 row counts toward PRE, never post progress",
			rows:          []trajctl.ScoreRow{w3("o1", 0.75, 0), w3("o1", 0.75, 1_500)},
			obj:           "o1",
			since:         launch,
			wantWitnessed: true, wantAdvanced: false, wantW3Rows: 2,
		},
		{
			name: "non-W3 rows are ignored",
			rows: []trajctl.ScoreRow{
				{ObjectiveID: "o1", Value: 1, Witness: trajctl.W2, UnixMillis: 1_500},
				w3("o1", 0.2, 500),
			},
			obj:           "o1",
			since:         launch,
			wantWitnessed: true, wantAdvanced: false, wantW3Rows: 1,
		},
		{
			name:          "objectiveID scopes the compare",
			rows:          []trajctl.ScoreRow{w3("other", 0.9, 1_500), w3("o1", 0.1, 500)},
			obj:           "o1",
			since:         launch,
			wantWitnessed: true, wantAdvanced: false, wantW3Rows: 1,
		},
		{
			name:          "no launch boundary attributes nothing post",
			rows:          []trajctl.ScoreRow{w3("o1", 0.5, 1_500)},
			obj:           "o1",
			since:         0,
			wantWitnessed: true, wantAdvanced: false, wantW3Rows: 1,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := FoldW3Continuity(tc.rows, tc.obj, tc.since)
			if got.Witnessed != tc.wantWitnessed || got.Advanced != tc.wantAdvanced || got.W3Rows != tc.wantW3Rows {
				t.Fatalf("FoldW3Continuity = %+v, want Witnessed=%v Advanced=%v W3Rows=%d",
					got, tc.wantWitnessed, tc.wantAdvanced, tc.wantW3Rows)
			}
		})
	}
}

// TestFoldResumeStateContinuity: a clean-terminal resume splits into took (verified progress
// advanced), took_no_progress (witnessed curve did not move), and the legacy took when no W3
// curve exists — absence of a witness is never a progress claim.
func TestFoldResumeStateContinuity(t *testing.T) {
	base := ResumeFacts{Attempts: 1, NewTurns: 5, Outcome: OutcomeProgressed}
	adv := base
	adv.Continuity = ContinuityWitness{Witnessed: true, Advanced: true}
	flat := base
	flat.Continuity = ContinuityWitness{Witnessed: true, Advanced: false}

	if got := FoldResumeState(adv); got != ResumeTook {
		t.Errorf("advanced W3 curve = %q, want %q", got, ResumeTook)
	}
	if got := FoldResumeState(flat); got != ResumeTookNoProgress {
		t.Errorf("flat W3 curve = %q, want %q", got, ResumeTookNoProgress)
	}
	if got := FoldResumeState(base); got != ResumeTook {
		t.Errorf("un-witnessed (legacy) = %q, want %q", got, ResumeTook)
	}
}

// TestRetryGateContinuityKeepsGateOpen: a took_no_progress under the attempt cap keeps the
// gate OPEN (re-anchor and keep watching) instead of latching "resume took"; the plain
// RetryGate (no witness) still latches. A spent cap still blocks — the stay-open is bounded.
func TestRetryGateContinuityKeepsGateOpen(t *testing.T) {
	history := []Attempt{{UnixSeconds: 100, Phase: "launched"}}
	flat := ContinuityWitness{Witnessed: true, Advanced: false}

	openGate := RetryGateContinuity(history, OutcomeProgressed, 8, flat)
	if openGate.Blocked {
		t.Fatalf("witnessed took_no_progress under cap: gate Blocked=%v, want open (%s)", openGate.Blocked, openGate.Reason)
	}
	if legacy := RetryGate(history, OutcomeProgressed, 8); !legacy.Blocked {
		t.Fatalf("un-witnessed progressed: gate Blocked=%v, want blocked (legacy took)", legacy.Blocked)
	}
	// Cap spent: even a took_no_progress must block — bounded, never a resume loop.
	capped := []Attempt{{UnixSeconds: 100, Phase: "launched"}, {UnixSeconds: 200, Phase: "launched"}}
	if d := RetryGateContinuity(capped, OutcomeProgressed, 1, flat); !d.Blocked {
		t.Fatalf("cap-spent took_no_progress: gate Blocked=%v, want blocked (%s)", d.Blocked, d.Reason)
	}
}

// TestFoldNextActionTookNoProgress: an OPEN-gate took_no_progress routes to the reversible
// wait_progress (never a fresh forced resume); a cap-blocked took_no_progress reads gave_up.
func TestFoldNextActionTookNoProgress(t *testing.T) {
	open := FoldNextAction(NextInput{
		State:    ResumeTookNoProgress,
		Outcome:  OutcomeProgressed,
		Retry:    RetryDecision{Blocked: false, Reason: "re-anchor and keep watching"},
		Admitted: true,
	})
	if open.Action != ActWaitProgress || open.Fire {
		t.Fatalf("open took_no_progress = %q fire=%v, want wait_progress/false (%s)", open.Action, open.Fire, open.Reason)
	}
	blocked := FoldNextAction(NextInput{
		State:   ResumeTookNoProgress,
		Outcome: OutcomeProgressed,
		Retry:   RetryDecision{Blocked: true, Reason: "attempt cap reached (2/2)"},
	})
	if blocked.Action != ActGaveUp {
		t.Fatalf("cap-blocked took_no_progress = %q, want gave_up (%s)", blocked.Action, blocked.Reason)
	}
}

// TestSelfObservationTookNoProgress: selfobserve renders the un-recovered verdict and the
// keep-watching hint instead of "you are progressing".
func TestSelfObservationTookNoProgress(t *testing.T) {
	obs := FoldSelfObservation(SelfFacts{
		Session:     "s-flat",
		History:     []Attempt{{UnixSeconds: 100, Phase: "launched"}},
		Outcome:     OutcomeProgressed,
		NewTurns:    4,
		MaxAttempts: 8,
		Continuity:  ContinuityWitness{Witnessed: true, Advanced: false},
	})
	if obs.State != ResumeTookNoProgress {
		t.Fatalf("State = %q, want %q", obs.State, ResumeTookNoProgress)
	}
	if obs.RetryBlocked {
		t.Fatalf("RetryBlocked = true, want open (keep watching): %s", obs.RetryReason)
	}
	if strings.Contains(obs.NextHint, "you are progressing") || !strings.Contains(obs.NextHint, "NOT yet recovered") {
		t.Fatalf("NextHint = %q, want an un-recovered/keep-watching hint", obs.NextHint)
	}
}

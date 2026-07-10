package resume

import (
	"strconv"
	"testing"
)

// launch is a fired-launch ledger row at unix t; bookkeeping/settled rows are built inline.
func launchAt(t int64) Attempt { return Attempt{UnixSeconds: t, Phase: "launched"} }

// TestFoldSelfObservation pins the worker self-observation fold field-by-field across the
// whole ResumeState vocabulary, including the fail-closed empty record — the acceptance the
// CLI and the MCP hook both rest on (#3804).
func TestFoldSelfObservation(t *testing.T) {
	tests := []struct {
		name  string
		facts SelfFacts
		want  SelfObservation
	}{
		{
			name:  "empty history is the fail-closed floor",
			facts: SelfFacts{Session: "s-empty"},
			want: SelfObservation{
				Session:      "s-empty",
				HasHistory:   false,
				Attempts:     0,
				NewTurns:     0,
				Outcome:      "", // shell supplied none
				State:        ResumePending,
				RetryBlocked: false,
				RetryReason:  "first resume",
				EarnedBudget: DefaultMaxResumeAttempts,
				NextHint:     selfNextHint(ResumePending),
			},
		},
		{
			name: "bookkeeping-only history reads pending, not never-seen",
			facts: SelfFacts{
				Session: "s-book",
				History: []Attempt{{Phase: "deferred"}},
				Outcome: OutcomeUnknown,
			},
			want: SelfObservation{
				Session:      "s-book",
				HasHistory:   true, // a row exists...
				Attempts:     0,    // ...but none is a fired launch
				Outcome:      OutcomeUnknown,
				State:        ResumePending,
				RetryBlocked: false,
				RetryReason:  "first resume",
				EarnedBudget: DefaultMaxResumeAttempts,
				NextHint:     selfNextHint(ResumePending),
			},
		},
		{
			name: "one launch that progressed reads took, burns the gate",
			facts: SelfFacts{
				Session:  "s-took",
				History:  []Attempt{launchAt(1000)},
				Outcome:  OutcomeProgressed,
				NewTurns: 3,
			},
			want: SelfObservation{
				Session:        "s-took",
				HasHistory:     true,
				Attempts:       1,
				LastLaunchUnix: 1000,
				NewTurns:       3,
				Outcome:        OutcomeProgressed,
				State:          ResumeTook,
				RetryBlocked:   true,
				RetryReason:    "already resumed once (resume took)",
				EarnedBudget:   DefaultMaxResumeAttempts,
				NextHint:       selfNextHint(ResumeTook),
			},
		},
		{
			name: "one launch on a wall reads re-stranded, stays retriable",
			facts: SelfFacts{
				Session: "s-restrand",
				History: []Attempt{launchAt(2000)},
				Outcome: OutcomeRecoverable,
			},
			want: SelfObservation{
				Session:        "s-restrand",
				HasHistory:     true,
				Attempts:       1,
				LastLaunchUnix: 2000,
				Outcome:        OutcomeRecoverable,
				State:          ResumeReStranded,
				RetryBlocked:   false,
				RetryReason:    "last resume failed recoverably; attempt 2/8",
				EarnedBudget:   DefaultMaxResumeAttempts,
				NextHint:       selfNextHint(ResumeReStranded),
			},
		},
		{
			name: "auth wall reads gave-up",
			facts: SelfFacts{
				Session: "s-auth",
				History: []Attempt{launchAt(3000)},
				Outcome: OutcomeUnrecoverable,
			},
			want: SelfObservation{
				Session:        "s-auth",
				HasHistory:     true,
				Attempts:       1,
				LastLaunchUnix: 3000,
				Outcome:        OutcomeUnrecoverable,
				State:          ResumeGaveUp,
				RetryBlocked:   true,
				RetryReason:    "last resume hit an auth/access wall — a re-resume cannot fix it",
				EarnedBudget:   DefaultMaxResumeAttempts,
				NextHint:       selfNextHint(ResumeGaveUp),
			},
		},
		{
			name: "operator-settled row is authoritative",
			facts: SelfFacts{
				Session: "s-settled",
				History: []Attempt{launchAt(4000), {ManualOverride: true}},
				Outcome: OutcomeProgressed,
			},
			want: SelfObservation{
				Session:         "s-settled",
				HasHistory:      true,
				Attempts:        1,
				LastLaunchUnix:  4000,
				Outcome:         OutcomeProgressed,
				State:           ResumeSettled,
				RetryBlocked:    true,
				RetryReason:     "operator-settled (manual ledger override)",
				EarnedBudget:    DefaultMaxResumeAttempts,
				OperatorSettled: true,
				NextHint:        selfNextHint(ResumeSettled),
			},
		},
		{
			name: "steady progress earns budget above the base",
			facts: SelfFacts{
				Session:  "s-earned",
				History:  []Attempt{launchAt(10000), launchAt(10000 + ProgressGapSeconds + 1), launchAt(10000 + 2*(ProgressGapSeconds+1))},
				Outcome:  OutcomeRecoverable,
				NewTurns: 0,
			},
			want: SelfObservation{
				Session:        "s-earned",
				HasHistory:     true,
				Attempts:       3,
				LastLaunchUnix: 10000 + 2*(ProgressGapSeconds+1),
				Outcome:        OutcomeRecoverable,
				State:          ResumeReStranded,
				RetryBlocked:   false,
				RetryReason:    "last resume failed recoverably; attempt 4/10",
				EarnedBudget:   DefaultMaxResumeAttempts + 2, // two >=gap intervals → +2
				NextHint:       selfNextHint(ResumeReStranded),
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := FoldSelfObservation(tc.facts)
			if got != tc.want {
				t.Fatalf("FoldSelfObservation mismatch\n got: %+v\nwant: %+v", got, tc.want)
			}
		})
	}
}

func TestFoldSelfObservationDefaultMaxAttemptsIsConsistent(t *testing.T) {
	history := []Attempt{{UnixSeconds: 1}, {UnixSeconds: 2}, {UnixSeconds: 3}, {UnixSeconds: 4}}
	got := FoldSelfObservation(SelfFacts{
		Session:     "s",
		History:     history,
		Outcome:     OutcomeRecoverable,
		MaxAttempts: 0,
	})
	if got.RetryBlocked {
		t.Fatalf("RetryBlocked = true, want earned-budget retry open; observation=%+v", got)
	}
	if got.State != ResumeReStranded {
		t.Fatalf("State = %q, want %q when the same earned cap keeps retry open", got.State, ResumeReStranded)
	}
	wantReason := "last resume failed recoverably; attempt 5/" + strconv.Itoa(got.EarnedBudget)
	if got.RetryReason != wantReason {
		t.Fatalf("RetryReason = %q, want %q", got.RetryReason, wantReason)
	}
}

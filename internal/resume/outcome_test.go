package resume

import "testing"

// The outcome classification is the remediation-cost precedence: auth outranks
// limit/transient outranks clean; no terminal turn is unknown.
func TestClassifyOutcome(t *testing.T) {
	cases := []struct {
		name string
		sig  TerminalSignal
		want Outcome
	}{
		{"no terminal turn", TerminalSignal{}, OutcomeUnknown},
		{"clean turn", TerminalSignal{Found: true}, OutcomeProgressed},
		{"limit wall", TerminalSignal{Found: true, LimitWall: true}, OutcomeRecoverable},
		{"transient 529", TerminalSignal{Found: true, TransientAPIError: true}, OutcomeRecoverable},
		{"auth wall", TerminalSignal{Found: true, AuthWall: true}, OutcomeUnrecoverable},
		{"auth outranks limit", TerminalSignal{Found: true, AuthWall: true, LimitWall: true}, OutcomeUnrecoverable},
	}
	for _, tc := range cases {
		if got := ClassifyOutcome(tc.sig); got != tc.want {
			t.Errorf("%s: ClassifyOutcome = %q, want %q", tc.name, got, tc.want)
		}
	}
}

// Deferred/considered/skipped rows are bookkeeping, not attempts; phase-less and
// launched rows are fired launches. LastLaunchUnix keys only on fired launches.
func TestCountAttemptsAndLastLaunch(t *testing.T) {
	history := []Attempt{
		{UnixSeconds: 100, Phase: "launched"},
		{UnixSeconds: 200, Phase: "deferred"},
		{UnixSeconds: 300, Phase: ""}, // a phase-less launcher row records a real spawn
		{UnixSeconds: 400, Phase: "Considered"},
	}
	if got := CountAttempts(history); got != 2 {
		t.Errorf("CountAttempts = %d, want 2", got)
	}
	if got := LastLaunchUnix(history); got != 300 {
		t.Errorf("LastLaunchUnix = %d, want 300 (the deferred/considered rows must not win)", got)
	}
	if got := LastLaunchUnix(nil); got != 0 {
		t.Errorf("LastLaunchUnix(nil) = %d, want 0", got)
	}
}

func TestNewTurnsAfter(t *testing.T) {
	turns := []int64{10, 20, 30}
	if got := NewTurnsAfter(turns, 15); got != 2 {
		t.Errorf("NewTurnsAfter(15) = %d, want 2", got)
	}
	if got := NewTurnsAfter(turns, 30); got != 0 {
		t.Errorf("NewTurnsAfter(30) = %d, want 0 (strictly after)", got)
	}
	if got := NewTurnsAfter(turns, 0); got != 0 {
		t.Errorf("NewTurnsAfter(no launch) = %d, want 0", got)
	}
}

// RetryGate is the outcome-aware once-gate: blocked unless the last attempt failed
// recoverably and the cap has room; operator settles and auth walls are final.
func TestRetryGate(t *testing.T) {
	launched := []Attempt{{UnixSeconds: 100, Phase: "launched"}}
	cases := []struct {
		name    string
		history []Attempt
		outcome Outcome
		max     int
		blocked bool
	}{
		{"first resume is never blocked", nil, OutcomeUnknown, 8, false},
		{"recoverable wall retries", launched, OutcomeRecoverable, 8, false},
		{"auth wall blocks", launched, OutcomeUnrecoverable, 8, true},
		{"clean finish burns once", launched, OutcomeProgressed, 8, true},
		{"unknown burns once", launched, OutcomeUnknown, 8, true},
		{"cap blocks even a recoverable wall", launched, OutcomeRecoverable, 1, true},
		{"operator settle is final", []Attempt{{Action: "consolidate-manual"}}, OutcomeRecoverable, 8, true},
		{"manual override is final", []Attempt{{ManualOverride: true}}, OutcomeRecoverable, 8, true},
	}
	for _, tc := range cases {
		d := RetryGate(tc.history, tc.outcome, tc.max)
		if d.Blocked != tc.blocked {
			t.Errorf("%s: Blocked = %v (reason %q), want %v", tc.name, d.Blocked, d.Reason, tc.blocked)
		}
		if d.Reason == "" {
			t.Errorf("%s: want a non-empty closed reason", tc.name)
		}
	}
}

// A zero/negative cap takes the watchdog's default, so a caller passing an unset knob
// still gets the real gate, not an always-blocked one.
func TestRetryGateDefaultCap(t *testing.T) {
	history := []Attempt{{Phase: "launched"}, {Phase: "launched"}}
	d := RetryGate(history, OutcomeRecoverable, 0)
	if d.Blocked {
		t.Fatalf("2 attempts under the default cap (%d) must not block: %q", DefaultMaxResumeAttempts, d.Reason)
	}
}

// FoldResumeState's precedence: pending, settled, took (proven progress wins even at
// the cap), gave-up (auth wall or spent cap), re-stranded, else launched-unproven.
func TestFoldResumeState(t *testing.T) {
	cases := []struct {
		name string
		f    ResumeFacts
		want ResumeState
	}{
		{"no attempt yet", ResumeFacts{Attempts: 0, Outcome: OutcomeRecoverable}, ResumePending},
		{"operator settled", ResumeFacts{Attempts: 1, OperatorSettled: true, Outcome: OutcomeProgressed}, ResumeSettled},
		{"took: new turns + clean terminal", ResumeFacts{Attempts: 1, NewTurns: 5, Outcome: OutcomeProgressed}, ResumeTook},
		{"took wins even at the cap", ResumeFacts{Attempts: 8, NewTurns: 3, Outcome: OutcomeProgressed}, ResumeTook},
		{"auth wall gave up", ResumeFacts{Attempts: 1, NewTurns: 2, Outcome: OutcomeUnrecoverable}, ResumeGaveUp},
		{"cap spent gave up", ResumeFacts{Attempts: 8, Outcome: OutcomeRecoverable}, ResumeGaveUp},
		{"re-stranded on a wall", ResumeFacts{Attempts: 1, NewTurns: 4, Outcome: OutcomeRecoverable}, ResumeReStranded},
		{"re-stranded immediately (0 new turns)", ResumeFacts{Attempts: 1, Outcome: OutcomeRecoverable}, ResumeReStranded},
		{"launched, unproven (0 new turns, clean)", ResumeFacts{Attempts: 1, Outcome: OutcomeProgressed}, ResumeLaunched},
		{"launched, unproven (unknown outcome)", ResumeFacts{Attempts: 1, NewTurns: 1, Outcome: OutcomeUnknown}, ResumeLaunched},
	}
	for _, tc := range cases {
		if got := FoldResumeState(tc.f); got != tc.want {
			t.Errorf("%s: FoldResumeState = %q, want %q", tc.name, got, tc.want)
		}
	}
}

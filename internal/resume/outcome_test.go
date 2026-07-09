package resume

import (
	"strings"
	"testing"
)

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
		{UnixSeconds: 500, Phase: "queued"},
		{UnixSeconds: 600, Phase: "status"},
		{UnixSeconds: 700, Phase: "progress"},
		{UnixSeconds: 800, Phase: "settled", Action: "consolidate-auth-plan-row"},
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

// ResumeModels attributes the recovery turns' model: the latest post-launch model is what
// the resume is on now; the distinct set exposes a mid-resume drift; pre-launch turns and
// the no-launch floor attribute nothing.
func TestResumeModels(t *testing.T) {
	times := []int64{10, 20, 30, 40}
	models := []string{"claude-fable-5", "claude-opus-4-8", "claude-opus-4-8", "claude-sonnet-5"}

	// After 15: turns at 20/30/40 count. Latest is the sonnet turn at 40; distinct is
	// opus then sonnet, in first-appearance order (the pre-launch fable turn is excluded).
	latest, distinct := ResumeModels(times, models, 15)
	if latest != "claude-sonnet-5" {
		t.Errorf("latest = %q, want claude-sonnet-5", latest)
	}
	if len(distinct) != 2 || distinct[0] != "claude-opus-4-8" || distinct[1] != "claude-sonnet-5" {
		t.Errorf("distinct = %v, want [claude-opus-4-8 claude-sonnet-5]", distinct)
	}

	// A single-model recovery: one distinct entry, latest equal to it.
	latest, distinct = ResumeModels(times[:2], models[:2], 15)
	if latest != "claude-opus-4-8" || len(distinct) != 1 || distinct[0] != "claude-opus-4-8" {
		t.Errorf("single-model = %q %v, want claude-opus-4-8 [claude-opus-4-8]", latest, distinct)
	}

	// No launch on record (sinceUnix <= 0) attributes nothing.
	if latest, distinct = ResumeModels(times, models, 0); latest != "" || distinct != nil {
		t.Errorf("no-launch floor = %q %v, want \"\" nil", latest, distinct)
	}

	// Nothing landed after the last turn: no attribution.
	if latest, distinct = ResumeModels(times, models, 40); latest != "" || distinct != nil {
		t.Errorf("nothing-after = %q %v, want \"\" nil", latest, distinct)
	}

	// Mismatched slice lengths fail closed to no attribution rather than panicking.
	if latest, _ = ResumeModels(times, models[:1], 5); latest != "" {
		t.Errorf("mismatched-lengths latest = %q, want \"\"", latest)
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

func TestRetryGateIgnoresStatusOnlyLedgerRows(t *testing.T) {
	history := []Attempt{{Phase: "queued"}, {Phase: "status"}, {Phase: "progress"}}
	d := RetryGate(history, OutcomeUnknown, 8)
	if d.Blocked {
		t.Fatalf("status-only ledger rows must not burn the first launch: %q", d.Reason)
	}
}

// The earned-budget curve is the #3124 fix: instead of a flat DefaultMaxResumeAttempts
// for every session, the cap is read from the session's own launch spacing. A launch
// that ran a productive stretch before the next fired (gap >= ProgressGapSeconds) earns
// budget; one that re-stranded almost immediately (gap < ProgressGapSeconds) costs it.
func TestEarnedResumeBudgetCurve(t *testing.T) {
	const base = DefaultMaxResumeAttempts
	gap := ProgressGapSeconds // one full progress interval

	// A launch t seconds after the previous one.
	at := func(unix int64) Attempt { return Attempt{UnixSeconds: unix, Phase: "launched"} }
	// Build a history whose consecutive launch gaps are exactly the given seconds.
	fromGaps := func(gaps ...int64) []Attempt {
		var h []Attempt
		var t int64 = 1_000_000 // arbitrary non-zero epoch so every timestamp is measurable
		h = append(h, at(t))
		for _, g := range gaps {
			t += g
			h = append(h, at(t))
		}
		return h
	}

	cases := []struct {
		name    string
		history []Attempt
		want    int
	}{
		{"no history: base", nil, base},
		{"one launch: no interval evidence, base", fromGaps(), base},
		{"single progress gap earns +1", fromGaps(gap), base + 1},
		{"single thrash gap costs -1", fromGaps(5), base - 1},
		{"all-progress climbs toward the ceiling", fromGaps(gap, gap, gap), base + 3},
		{"sustained progress clamps at the ceiling", fromGaps(gap, gap, gap, gap, gap, gap, gap, gap), MaxEarnedResumeBudget},
		{"all-thrash sinks toward the floor", fromGaps(1, 1, 1, 1, 1), base - 5},
		{"sustained thrash clamps at the floor", fromGaps(1, 1, 1, 1, 1, 1, 1, 1, 1, 1), MinEarnedResumeBudget},
		{"mixed nets out", fromGaps(gap, 5, gap, 5), base}, // +1 -1 +1 -1 == base
		{"gap exactly at the threshold counts as progress", fromGaps(ProgressGapSeconds), base + 1},
		{"gap one under the threshold is thrash", fromGaps(ProgressGapSeconds - 1), base - 1},
	}
	for _, tc := range cases {
		if got := EarnedResumeBudget(tc.history); got != tc.want {
			t.Errorf("%s: EarnedResumeBudget = %d, want %d", tc.name, got, tc.want)
		}
	}
}

// Absence of timestamp evidence must never REDUCE a session's budget below the flat
// default — an untimestamped ledger (older launchers, or rows that dropped the field)
// folds back to exactly DefaultMaxResumeAttempts, not the thrash floor.
func TestEarnedResumeBudgetUntimestampedIsNeutral(t *testing.T) {
	history := []Attempt{
		{Phase: "launched"}, {Phase: "launched"}, {Phase: "launched"}, {Phase: "launched"},
	}
	if got := EarnedResumeBudget(history); got != DefaultMaxResumeAttempts {
		t.Fatalf("untimestamped launches must fold to the flat base %d, got %d", DefaultMaxResumeAttempts, got)
	}
	// A history mixing timestamped progress with untimestamped rows credits only the
	// measurable gap and treats the zero-timestamp boundaries as neutral, not as thrash.
	mixed := []Attempt{
		{UnixSeconds: 1_000_000, Phase: "launched"},
		{UnixSeconds: 1_000_000 + ProgressGapSeconds, Phase: "launched"}, // +1 measurable progress
		{Phase: "launched"}, // gap into/out of a zero-stamp row is unmeasurable → neutral
	}
	if got := EarnedResumeBudget(mixed); got != DefaultMaxResumeAttempts+1 {
		t.Fatalf("one measurable progress gap + neutral unmeasured gaps: want %d, got %d", DefaultMaxResumeAttempts+1, got)
	}
}

// Only fired launches shape the budget: bookkeeping rows (deferred/status/queued) carry
// no resume, so their timestamps must not be read as launch gaps.
func TestEarnedResumeBudgetCountsOnlyLaunches(t *testing.T) {
	// Two real launches a full progress interval apart, with dense bookkeeping rows
	// interleaved between them. If the bookkeeping timestamps were counted, the tight
	// gaps would register as thrash; they must be skipped, leaving one progress interval.
	history := []Attempt{
		{UnixSeconds: 1_000_000, Phase: "launched"},
		{UnixSeconds: 1_000_001, Phase: "status"},
		{UnixSeconds: 1_000_002, Phase: "deferred"},
		{UnixSeconds: 1_000_003, Phase: "queued"},
		{UnixSeconds: 1_000_000 + ProgressGapSeconds, Phase: "launched"},
	}
	if got := EarnedResumeBudget(history); got != DefaultMaxResumeAttempts+1 {
		t.Fatalf("bookkeeping rows must not count as launch gaps: want %d, got %d", DefaultMaxResumeAttempts+1, got)
	}
}

// RetryGate wires the earned budget in: with maxAttempts <= 0 (the "use the earned
// budget" sentinel), a thrashing session is cut off EARLIER than the flat default, while
// an explicit positive cap is honored literally and un-earned.
func TestRetryGateUsesEarnedBudget(t *testing.T) {
	// Four fired launches, each re-stranding within seconds → all-thrash → budget cut to
	// base-3 = 5. Attempts (4) is under 5, so a recoverable wall still retries.
	thrash := []Attempt{
		{UnixSeconds: 1_000_000, Phase: "launched"},
		{UnixSeconds: 1_000_003, Phase: "launched"},
		{UnixSeconds: 1_000_006, Phase: "launched"},
		{UnixSeconds: 1_000_009, Phase: "launched"},
	}
	if b := EarnedResumeBudget(thrash); b != DefaultMaxResumeAttempts-3 {
		t.Fatalf("precondition: earned budget = %d, want %d", b, DefaultMaxResumeAttempts-3)
	}
	if d := RetryGate(thrash, OutcomeRecoverable, 0); d.Blocked {
		t.Fatalf("4 thrash attempts under the earned cap (5) must still retry: %q", d.Reason)
	}
	// One more thrash launch (5 attempts) meets the earned cap of 5 → blocked, EARLIER
	// than the flat 8 a progress-blind gate would have allowed.
	thrash5 := append(thrash, Attempt{UnixSeconds: 1_000_012, Phase: "launched"})
	if b := EarnedResumeBudget(thrash5); b != DefaultMaxResumeAttempts-4 {
		t.Fatalf("precondition: earned budget = %d, want %d", b, DefaultMaxResumeAttempts-4)
	}
	d := RetryGate(thrash5, OutcomeRecoverable, 0)
	if !d.Blocked {
		t.Fatalf("a thrashing session at its earned cap must be blocked before the flat default: %q", d.Reason)
	}

	// The SAME attempt count on a PROGRESSING session (each resume ran a full interval)
	// earns budget instead, so the recoverable wall is still retried where the thrashing
	// one was cut off — progress-earned, not flat.
	prog5 := []Attempt{
		{UnixSeconds: 1_000_000, Phase: "launched"},
		{UnixSeconds: 1_000_000 + 1*ProgressGapSeconds, Phase: "launched"},
		{UnixSeconds: 1_000_000 + 2*ProgressGapSeconds, Phase: "launched"},
		{UnixSeconds: 1_000_000 + 3*ProgressGapSeconds, Phase: "launched"},
		{UnixSeconds: 1_000_000 + 4*ProgressGapSeconds, Phase: "launched"},
	}
	if d := RetryGate(prog5, OutcomeRecoverable, 0); d.Blocked {
		t.Fatalf("a progressing session at 5 attempts earned more budget and must still retry: %q", d.Reason)
	}

	// An explicit positive cap is an override: honored literally, NOT earned. Even a
	// strongly-progressing history is blocked once it meets the caller's explicit cap.
	if d := RetryGate(prog5, OutcomeRecoverable, 2); !d.Blocked {
		t.Fatalf("an explicit cap of 2 must block at 5 attempts regardless of earned progress: %q", d.Reason)
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

// #3354: the deny axis of the migratable/non-migratable split. Every terminal error
// class maps to exactly one relaunch disposition — relaunch now (transport/transient
// death), back off then relaunch (limit wall, recoverable after its reset), or never
// relaunch (intentional cancel, auth wall) — so a deliberate stop or a capacity refusal
// is never blindly relaunched the way a transport death is.
func TestCancelledRelaunchClassification(t *testing.T) {
	launched := []Attempt{{UnixSeconds: 100, Phase: "launched"}}

	// Classification: an intentional cancel is its own closed outcome, and it outranks
	// every environmental wall reading on the same turn — the operator's intent is never
	// masked by a cheaper (retryable) reading of the same terminal text.
	classify := []struct {
		name string
		sig  TerminalSignal
		want Outcome
	}{
		{"cancelled turn", TerminalSignal{Found: true, Cancelled: true}, OutcomeCancelled},
		{"cancel outranks limit wall", TerminalSignal{Found: true, Cancelled: true, LimitWall: true}, OutcomeCancelled},
		{"cancel outranks auth wall", TerminalSignal{Found: true, Cancelled: true, AuthWall: true}, OutcomeCancelled},
		{"cancel outranks transient", TerminalSignal{Found: true, Cancelled: true, TransientAPIError: true}, OutcomeCancelled},
	}
	for _, tc := range classify {
		if got := ClassifyOutcome(tc.sig); got != tc.want {
			t.Errorf("%s: ClassifyOutcome = %q, want %q", tc.name, got, tc.want)
		}
	}

	// The gate: each error class → relaunch / back-off-relaunch / block.
	gate := []struct {
		name    string
		outcome Outcome
		blocked bool
	}{
		{"transport/transient death relaunches", OutcomeRecoverable, false}, // MIGRATABLE
		{"limit wall backs off then relaunches", OutcomeRecoverable, false}, // back-off arm: same closed outcome, retried past its reset
		{"intentional cancel never relaunches", OutcomeCancelled, true},     // NON_MIGRATABLE: deliberate stop
		{"auth wall never relaunches", OutcomeUnrecoverable, true},          // NON_MIGRATABLE: needs a human
	}
	for _, tc := range gate {
		d := RetryGate(launched, tc.outcome, 8)
		if d.Blocked != tc.blocked {
			t.Errorf("%s: Blocked = %v (reason %q), want %v", tc.name, d.Blocked, d.Reason, tc.blocked)
		}
	}

	// The cancel refusal is its own closed reason — distinct from the operator-settled
	// override and from the auth-wall block, so a monitor can tell "the operator stopped
	// this on purpose" from "a wall no retry can fix".
	cancelled := RetryGate(launched, OutcomeCancelled, 8)
	if !cancelled.Blocked || !strings.Contains(cancelled.Reason, "intentionally cancelled") {
		t.Errorf("cancel block = %v %q, want blocked with an 'intentionally cancelled' reason", cancelled.Blocked, cancelled.Reason)
	}
	settled := RetryGate([]Attempt{{Action: "consolidate-manual"}}, OutcomeCancelled, 8)
	if cancelled.Reason == settled.Reason {
		t.Errorf("cancel reason %q must stay distinct from the operator-settled reason %q", cancelled.Reason, settled.Reason)
	}
	auth := RetryGate(launched, OutcomeUnrecoverable, 8)
	if cancelled.Reason == auth.Reason {
		t.Errorf("cancel reason %q must stay distinct from the auth-wall reason %q", cancelled.Reason, auth.Reason)
	}

	// The per-session label: a cancelled session is gave-up (no automatic resume will
	// fire again; a human owns it), never re-stranded or launched-unproven.
	if got := FoldResumeState(ResumeFacts{Attempts: 1, Outcome: OutcomeCancelled}); got != ResumeGaveUp {
		t.Errorf("FoldResumeState(cancelled) = %q, want %q", got, ResumeGaveUp)
	}
}

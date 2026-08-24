package resume

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

// The load-bearing facts these pin (from tools/fleet_resume_watchdog.py and its tests):
//   - the self-guard refuses the watchdog's own session unconditionally, and is inert
//     when the watchdog runs outside a Claude session (empty SelfSID);
//   - the worker-policy guard refuses a stale plan row for a tombstoned account, and is
//     inert (fail-open) when no roster could be read;
//   - the retry gate is the same outcome-aware once-gate resume_status uses: a deferred
//     ledger row does not burn the attempt budget, a recoverable failure stays eligible
//     under the cap, an auth wall or a clean finish blocks;
//   - a resumed child's env drops the parent's guard-gateway/model-API wiring and
//     harness identity, and pins CLAUDE_CONFIG_DIR to the resume target (the 2026-07-01
//     whole-wave-crash fix);
//   - probe mode "auto" resolves to a real probe only on a live tick;
//   - the operator-hold guard (drivestate.go) refuses a session an operator paused/
//     drained/stopped via the durable drive-state store, sits ABOVE the policy and retry
//     gates (so it beats a proven re-death revive, a spent cap, and a non-worker account),
//     yields only to the self-guard, and is inert (fail-open) for a running/absent/unknown
//     drive-state.

func TestDecideWatchdogRowSelfGuard(t *testing.T) {
	row := WatchdogPlanRow{Session: "sid-self", Account: ".claude-x"}
	d := DecideWatchdogRow(row, WatchdogGuards{SelfSID: "sid-self"}, nil, OutcomeUnknown)
	if d.Action != WatchdogSkipSelf {
		t.Fatalf("action = %s, want skip_self", d.Action)
	}
	// Outside a Claude session (empty SelfSID) the guard is inert.
	d = DecideWatchdogRow(row, WatchdogGuards{}, nil, OutcomeUnknown)
	if d.Action != WatchdogLaunch {
		t.Fatalf("empty SelfSID must leave the guard inert, got %s (%s)", d.Action, d.Reason)
	}
}

func TestDecideWatchdogRowLivenessGuard(t *testing.T) {
	// #3459 regression: the same session UUID exists as a stale/older copy the plan
	// classified STOPPED_APIERR (crashed, queued) AND a newer copy under another account dir
	// that a live `claude --resume` is actively advancing. With the session id in the live
	// census the row must be SKIPPED, never launched — no second driver onto a live
	// transcript, whatever disposition the stale copy carried.
	row := WatchdogPlanRow{Session: "94aea02a", Account: ".claude-day26NEW-netra", Disp: "STOPPED_APIERR"}
	g := WatchdogGuards{LiveSIDs: map[string]bool{"94aea02a": true}}
	if d := DecideWatchdogRow(row, g, nil, OutcomeRecoverable); d.Action != WatchdogSkipLive {
		t.Fatalf("live-driven session: action = %s, want skip_live", d.Action)
	}
	// The gate beats the retry gate: WITHOUT the live census the same row LAUNCHES (a
	// recoverable STOPPED_APIERR under the cap), so the skip is the liveness gate's doing.
	if d := DecideWatchdogRow(row, WatchdogGuards{}, nil, OutcomeRecoverable); d.Action != WatchdogLaunch {
		t.Fatalf("precondition: without the live census the row should launch, got %s (%s)", d.Action, d.Reason)
	}
	// Fail-open per key: a nil map and a census that names only OTHER sessions both leave
	// the guard inert — an unreadable/partial process table must never strand a real crash.
	for name, g := range map[string]WatchdogGuards{
		"nil census":   {},
		"other sids":   {LiveSIDs: map[string]bool{"someone-else": true}},
		"empty census": {LiveSIDs: map[string]bool{}},
	} {
		if d := DecideWatchdogRow(row, g, nil, OutcomeUnknown); d.Action == WatchdogSkipLive {
			t.Fatalf("%s: liveness guard must be inert (fail-open), got %s", name, d.Action)
		}
	}
	// The self-guard still comes first: a row that is BOTH self and live reports skip_self.
	self := WatchdogGuards{SelfSID: "94aea02a", LiveSIDs: map[string]bool{"94aea02a": true}}
	if d := DecideWatchdogRow(row, self, nil, OutcomeUnknown); d.Action != WatchdogSkipSelf {
		t.Fatalf("self vs live = %s, want skip_self (the self-guard is first)", d.Action)
	}
}

func TestDecideWatchdogRowWorkerPolicyGuard(t *testing.T) {
	row := WatchdogPlanRow{Session: "sid-1", Account: ".claude-tombstoned"}
	g := WatchdogGuards{WorkerAccounts: map[string]bool{".claude-x": true}}
	d := DecideWatchdogRow(row, g, nil, OutcomeUnknown)
	if d.Action != WatchdogSkipNonWorker {
		t.Fatalf("action = %s, want skip_non_worker", d.Action)
	}
	if !strings.Contains(d.Reason, ".claude-tombstoned") {
		t.Fatalf("reason must name the account, got %q", d.Reason)
	}
	// An empty roster (failed read) leaves the guard inert — fail-open.
	d = DecideWatchdogRow(row, WatchdogGuards{}, nil, OutcomeUnknown)
	if d.Action != WatchdogLaunch {
		t.Fatalf("empty roster must fail open, got %s", d.Action)
	}
}

func TestDecideWatchdogRowFirstResumeLaunchesWithAttemptOne(t *testing.T) {
	d := DecideWatchdogRow(WatchdogPlanRow{Session: "sid-2", Account: ".claude-x"},
		WatchdogGuards{}, nil, OutcomeUnknown)
	if d.Action != WatchdogLaunch || d.Attempt != 1 {
		t.Fatalf("first resume: action=%s attempt=%d, want launch/1", d.Action, d.Attempt)
	}
}

func TestDecideWatchdogRowRecoverableStaysEligibleAndCountsAttempts(t *testing.T) {
	hist := []Attempt{{Phase: "launched"}, {Phase: "deferred"}, {Phase: "launched"}}
	d := DecideWatchdogRow(WatchdogPlanRow{Session: "sid-3", Account: ".claude-x"},
		WatchdogGuards{MaxAttempts: 8}, hist, OutcomeRecoverable)
	if d.Action != WatchdogLaunch {
		t.Fatalf("recoverable under cap must launch, got %s (%s)", d.Action, d.Reason)
	}
	// The deferred row is bookkeeping, not an attempt: 2 fired + 1 = attempt 3.
	if d.Attempt != 3 {
		t.Fatalf("attempt = %d, want 3 (deferred rows must not burn the budget)", d.Attempt)
	}
}

func TestDecideWatchdogRowBlockedOutcomes(t *testing.T) {
	hist := []Attempt{{Phase: "launched"}}
	for _, tc := range []struct {
		outcome Outcome
		wantWhy string
	}{
		{OutcomeUnrecoverable, "auth"},
		{OutcomeProgressed, "already resumed once"},
		{OutcomeUnknown, "already resumed once"},
	} {
		d := DecideWatchdogRow(WatchdogPlanRow{Session: "sid-4", Account: ".claude-x"},
			WatchdogGuards{}, hist, tc.outcome)
		if d.Action != WatchdogSkipBlocked {
			t.Fatalf("outcome %s: action = %s, want skip_blocked", tc.outcome, d.Action)
		}
		if !strings.Contains(strings.ToLower(d.Reason), tc.wantWhy) {
			t.Fatalf("outcome %s: reason %q must mention %q", tc.outcome, d.Reason, tc.wantWhy)
		}
	}
}

func TestDecideWatchdogRowReplaySafetyConjunct(t *testing.T) {
	// A retryable (recoverable) outcome whose interrupted turn emitted replay-unsafe
	// output must NOT auto-retry (#5927) — and the refusal carries the observable reason.
	row := WatchdogPlanRow{Session: "sid-replay", Account: ".claude-x",
		PartialBlocks: []EmittedBlock{{Kind: BlockText, Text: "half an answer"}, {Kind: BlockToolCall, ToolCallID: "charged-card"}}}
	d := DecideWatchdogRow(row, WatchdogGuards{}, nil, OutcomeRecoverable)
	if d.Action != WatchdogSkipBlocked || !strings.Contains(d.Reason, ReasonReplayUnsafeOutput) {
		t.Fatalf("unsafe partial: action=%s reason=%q, want skip_blocked naming %s", d.Action, d.Reason, ReasonReplayUnsafeOutput)
	}

	// A thinking-only partial stays retryable: the conjunct narrows, it never widens or
	// replaces the error classification.
	row.PartialBlocks = []EmittedBlock{{Kind: BlockThinking}}
	if d := DecideWatchdogRow(row, WatchdogGuards{}, nil, OutcomeRecoverable); d.Action != WatchdogLaunch {
		t.Fatalf("thinking-only partial: action=%s (%s), want launch", d.Action, d.Reason)
	}

	// An interrupted turn whose tool calls all carry matching results is
	// preserve-and-continued (the resume continues the preserved transcript), never
	// discarded — and still counts its attempt.
	row.PartialBlocks = []EmittedBlock{
		{Kind: BlockToolCall, ToolCallID: "c1"}, {Kind: BlockToolResult, ToolCallID: "c1"}}
	d = DecideWatchdogRow(row, WatchdogGuards{}, []Attempt{}, OutcomeRecoverable)
	if d.Action != WatchdogLaunch || !strings.Contains(d.Reason, ReasonCompletedToolEffects) || d.Attempt != 1 {
		t.Fatalf("completed effects: got %#v, want launch naming %s at attempt 1", d, ReasonCompletedToolEffects)
	}

	// No blocks (clean tail / unreadable transcript) leaves the gate inert.
	row.PartialBlocks = nil
	if d := DecideWatchdogRow(row, WatchdogGuards{}, nil, OutcomeRecoverable); d.Action != WatchdogLaunch {
		t.Fatalf("nil blocks: action=%s (%s), want launch", d.Action, d.Reason)
	}

	// The conjunct never overturns a Blocked verdict: a spent cap stays blocked even
	// with completed tool effects on the row.
	row.PartialBlocks = []EmittedBlock{
		{Kind: BlockToolCall, ToolCallID: "c1"}, {Kind: BlockToolResult, ToolCallID: "c1"}}
	hist := []Attempt{{Phase: "launched"}, {Phase: "launched"}}
	if d := DecideWatchdogRow(row, WatchdogGuards{MaxAttempts: 2}, hist, OutcomeRecoverable); d.Action != WatchdogSkipBlocked {
		t.Fatalf("spent cap with completed effects: action=%s (%s), want skip_blocked", d.Action, d.Reason)
	}
}

func TestDecideWatchdogRowAttemptCapBlocks(t *testing.T) {
	hist := []Attempt{{Phase: "launched"}, {Phase: "launched"}}
	d := DecideWatchdogRow(WatchdogPlanRow{Session: "sid-5", Account: ".claude-x"},
		WatchdogGuards{MaxAttempts: 2}, hist, OutcomeRecoverable)
	if d.Action != WatchdogSkipBlocked || !strings.Contains(d.Reason, "cap") {
		t.Fatalf("cap spent: action=%s reason=%q, want skip_blocked with cap reason", d.Action, d.Reason)
	}
}

func TestWatchdogPlanRowTargets(t *testing.T) {
	r := WatchdogPlanRow{ConfigDir: "/home/.claude-a"}
	if r.ResumeTarget() != "/home/.claude-a" || r.RehomeSource() != "/home/.claude-a" {
		t.Fatalf("bare row must fall back to the owner dir")
	}
	r.ResumeConfigDir = "/home/.claude-b"
	r.SourceConfigDir = "/home/.claude-src"
	if r.ResumeTarget() != "/home/.claude-b" {
		t.Fatalf("ResumeTarget = %q, want the re-home target", r.ResumeTarget())
	}
	if r.RehomeSource() != "/home/.claude-src" {
		t.Fatalf("RehomeSource = %q, want the explicit source", r.RehomeSource())
	}
}

func TestWatchdogChildEnvStripsGuardWiringAndPinsConfigDir(t *testing.T) {
	env := WatchdogChildEnv([]string{
		"PATH=/usr/bin",
		"ANTHROPIC_API_KEY=sk-parent",
		"ANTHROPIC_AUTH_TOKEN=tok",
		"ANTHROPIC_BASE_URL=http://127.0.0.1:9999",
		"CLAUDE_CODE_SESSION_ID=parent-sid",
		"CLAUDE_CODE_CHILD_SESSION=1",
		"JOB_SUPERVISED_WORKER=1",
		"CLAUDE_CONFIG_DIR=/home/.claude-old",
		"HOME=/home/u",
	}, "/home/.claude-target")

	joined := strings.Join(env, "\n")
	for _, banned := range []string{"ANTHROPIC_", "CLAUDE_CODE_SESSION_ID", "CLAUDE_CODE_CHILD_SESSION", "JOB_SUPERVISED_WORKER"} {
		if strings.Contains(joined, banned) {
			t.Fatalf("child env still carries %s:\n%s", banned, joined)
		}
	}
	if !strings.Contains(joined, "PATH=/usr/bin") || !strings.Contains(joined, "HOME=/home/u") {
		t.Fatalf("unrelated env must survive:\n%s", joined)
	}
	// Exactly one CLAUDE_CONFIG_DIR, pinned to the target.
	n := 0
	for _, kv := range env {
		if strings.HasPrefix(kv, "CLAUDE_CONFIG_DIR=") {
			n++
			if kv != "CLAUDE_CONFIG_DIR=/home/.claude-target" {
				t.Fatalf("config dir = %q, want the resume target", kv)
			}
		}
	}
	if n != 1 {
		t.Fatalf("CLAUDE_CONFIG_DIR appears %d times, want exactly 1", n)
	}
}

func TestReviveOutcomeReleasesStaleTookLatchOnProvenDeath(t *testing.T) {
	dead := ReDeathEvidence{ProcessScanOK: true, TranscriptIdleSeconds: 3600, PostLaunchProgress: true}
	for _, tc := range []struct {
		in      Outcome
		want    Outcome
		revived bool
	}{
		{OutcomeProgressed, OutcomeRecoverable, true},
		{OutcomeUnknown, OutcomeRecoverable, true},
		{OutcomeUnrecoverable, OutcomeUnrecoverable, false}, // an auth wall is never revived
		{OutcomeRecoverable, OutcomeRecoverable, false},     // already open — nothing to release
	} {
		got, revived := ReviveOutcome(tc.in, dead)
		if got != tc.want || revived != tc.revived {
			t.Fatalf("ReviveOutcome(%s, dead) = (%s, %v), want (%s, %v)",
				tc.in, got, revived, tc.want, tc.revived)
		}
	}
}

func TestReviveOutcomeAnyUnprovenFactKeepsTheBurn(t *testing.T) {
	for name, ev := range map[string]ReDeathEvidence{
		"zero value":       {},
		"live process":     {ProcessScanOK: true, ProcessLive: true, TranscriptIdleSeconds: 3600, PostLaunchProgress: true},
		"unreadable table": {ProcessScanOK: false, TranscriptIdleSeconds: 3600, PostLaunchProgress: true},
		"no transcript":    {ProcessScanOK: true, TranscriptIdleSeconds: -1, LaunchAgeSeconds: 3600, PostLaunchProgress: true},
		"fresh transcript": {ProcessScanOK: true, TranscriptIdleSeconds: DeadTranscriptIdleFloorSeconds - 1, PostLaunchProgress: true},
	} {
		if got, revived := ReviveOutcome(OutcomeProgressed, ev); revived || got != OutcomeProgressed {
			t.Fatalf("%s: ReviveOutcome = (%s, %v), want the burn kept", name, got, revived)
		}
	}
}

func TestReviveOutcomeUnprovenLaunchRequiresGraceAndNoLiveDriver(t *testing.T) {
	for name, tc := range map[string]struct {
		ev      ReDeathEvidence
		revived bool
	}{
		"past grace and dead": {
			ev:      ReDeathEvidence{ProcessScanOK: true, TranscriptIdleSeconds: -1, LaunchAgeSeconds: DeadTranscriptIdleFloorSeconds},
			revived: true,
		},
		"inside grace": {
			// A stale pre-launch transcript must not collapse the launch's own startup grace.
			ev: ReDeathEvidence{ProcessScanOK: true, TranscriptIdleSeconds: 3600, LaunchAgeSeconds: DeadTranscriptIdleFloorSeconds - 1},
		},
		"past grace but live": {
			ev: ReDeathEvidence{ProcessScanOK: true, ProcessLive: true, TranscriptIdleSeconds: -1, LaunchAgeSeconds: DeadTranscriptIdleFloorSeconds},
		},
		"past grace but scan unreadable": {
			ev: ReDeathEvidence{ProcessScanOK: false, TranscriptIdleSeconds: -1, LaunchAgeSeconds: DeadTranscriptIdleFloorSeconds},
		},
		"launch timestamp unknown": {
			ev: ReDeathEvidence{ProcessScanOK: true, TranscriptIdleSeconds: -1, LaunchAgeSeconds: -1},
		},
	} {
		t.Run(name, func(t *testing.T) {
			got, revived := ReviveOutcome(OutcomeUnknown, tc.ev)
			want := OutcomeUnknown
			if tc.revived {
				want = OutcomeRecoverable
			}
			if got != want || revived != tc.revived {
				t.Fatalf("ReviveOutcome(unknown, %+v) = (%s, %v), want (%s, %v)",
					tc.ev, got, revived, want, tc.revived)
			}
		})
	}
}

func TestReviveOutcomeThroughWatchdogGate(t *testing.T) {
	// The #2368 acceptance chain, leaf half: a prior take marker (launched history,
	// progressed outcome) plus proof of a new death must reach LAUNCH, not the
	// "already resumed once" skip.
	dead := ReDeathEvidence{ProcessScanOK: true, TranscriptIdleSeconds: 3600, PostLaunchProgress: true}
	row := WatchdogPlanRow{Session: "sid-relatch", Account: ".claude-x"}
	hist := []Attempt{{Phase: "launched"}}

	outcome, revived := ReviveOutcome(OutcomeProgressed, dead)
	if !revived {
		t.Fatal("proven death must release the latch")
	}
	d := DecideWatchdogRow(row, WatchdogGuards{}, hist, outcome)
	if d.Action != WatchdogLaunch || d.Attempt != 2 {
		t.Fatalf("revived row: action=%s attempt=%d (%s), want launch/2", d.Action, d.Attempt, d.Reason)
	}

	// Higher-precedence blocks keep binding through a revive: an operator settle …
	settled := []Attempt{{Phase: "launched"}, {Action: "consolidate-operator"}}
	if d := DecideWatchdogRow(row, WatchdogGuards{}, settled, outcome); d.Action != WatchdogSkipBlocked ||
		!strings.Contains(d.Reason, "settled") {
		t.Fatalf("settled + revive = %s (%s), want skip_blocked settled", d.Action, d.Reason)
	}
	// … and the spent attempt cap.
	if d := DecideWatchdogRow(row, WatchdogGuards{MaxAttempts: 1}, hist, outcome); d.Action != WatchdogSkipBlocked ||
		!strings.Contains(d.Reason, "cap") {
		t.Fatalf("cap + revive = %s (%s), want skip_blocked cap", d.Action, d.Reason)
	}
}

func TestDecideWatchdogRowOperatorHoldGuard(t *testing.T) {
	row := WatchdogPlanRow{Session: "sid-held", Account: ".claude-x"}
	for _, st := range []WatchdogDriveState{DrivePaused, DriveDraining, DriveStopped} {
		g := WatchdogGuards{DriveStates: map[string]WatchdogDriveState{"sid-held": st}}
		d := DecideWatchdogRow(row, g, nil, OutcomeUnknown)
		if d.Action != WatchdogSkipOperatorHold {
			t.Fatalf("drive-state %s: action = %s, want skip_operator_hold", st, d.Action)
		}
		if !strings.Contains(strings.ToLower(d.Reason), "operator") {
			t.Fatalf("drive-state %s: reason %q must name the operator hold", st, d.Reason)
		}
	}
	// Inert (per-key fail-open) for a live / absent / unknown state: a nil map, a session with
	// no entry, running/throttled, and an unrecognized token all fall through to launch — an
	// absent hold must never strand a crashed session (matching the worker-policy guard).
	for name, g := range map[string]WatchdogGuards{
		"nil map":       {},
		"absent sid":    {DriveStates: map[string]WatchdogDriveState{"other": DriveStopped}},
		"running":       {DriveStates: map[string]WatchdogDriveState{"sid-held": DriveRunning}},
		"throttled":     {DriveStates: map[string]WatchdogDriveState{"sid-held": DriveThrottled}},
		"unknown token": {DriveStates: map[string]WatchdogDriveState{"sid-held": WatchdogDriveState("weird")}},
	} {
		if d := DecideWatchdogRow(row, g, nil, OutcomeUnknown); d.Action != WatchdogLaunch {
			t.Fatalf("%s: operator-hold guard must be inert (fail-open), got %s (%s)", name, d.Action, d.Reason)
		}
	}
}

func TestOperatorHoldOutranksReviveCapAndPolicy(t *testing.T) {
	// A stopped operator hold must beat every lower-precedence path: a revived re-death that
	// WOULD otherwise LAUNCH, a spent attempt cap, and a non-worker account — and must itself
	// yield to the self-guard (never race two `claude` on one transcript).
	dead := ReDeathEvidence{ProcessScanOK: true, TranscriptIdleSeconds: 3600, PostLaunchProgress: true}
	revived, ok := ReviveOutcome(OutcomeProgressed, dead)
	if !ok {
		t.Fatal("precondition: proven death must revive the latch (else this test is vacuous)")
	}
	held := map[string]WatchdogDriveState{"sid-h": DriveStopped}
	row := WatchdogPlanRow{Session: "sid-h", Account: ".claude-worker"}
	hist := []Attempt{{Phase: "launched"}}

	// Precondition (not vacuous): WITHOUT the hold, the revived row would LAUNCH.
	if d := DecideWatchdogRow(row, WatchdogGuards{}, hist, revived); d.Action != WatchdogLaunch {
		t.Fatalf("precondition: revived row without a hold should launch, got %s", d.Action)
	}
	// Beats the revived re-death latch.
	if d := DecideWatchdogRow(row, WatchdogGuards{DriveStates: held}, hist, revived); d.Action != WatchdogSkipOperatorHold {
		t.Fatalf("hold vs revive = %s (%s), want skip_operator_hold", d.Action, d.Reason)
	}
	// Beats a spent attempt cap.
	if d := DecideWatchdogRow(row, WatchdogGuards{DriveStates: held, MaxAttempts: 1}, hist, revived); d.Action != WatchdogSkipOperatorHold {
		t.Fatalf("hold vs cap = %s (%s), want skip_operator_hold", d.Action, d.Reason)
	}
	// Beats the worker-policy guard: a non-worker account that is ALSO held reports the hold,
	// not skip_non_worker (operator-hold sits above the policy guard).
	g := WatchdogGuards{DriveStates: held, WorkerAccounts: map[string]bool{".claude-other": true}}
	if d := DecideWatchdogRow(row, g, nil, OutcomeUnknown); d.Action != WatchdogSkipOperatorHold {
		t.Fatalf("hold vs policy = %s (%s), want skip_operator_hold (hold outranks policy)", d.Action, d.Reason)
	}
	// But the self-guard still comes first.
	self := WatchdogGuards{SelfSID: "sid-h", DriveStates: held}
	if d := DecideWatchdogRow(row, self, nil, OutcomeUnknown); d.Action != WatchdogSkipSelf {
		t.Fatalf("self vs hold = %s (%s), want skip_self (the self-guard is first)", d.Action, d.Reason)
	}
}

func TestFoldDriveStatesReleaseAndHeldPredicate(t *testing.T) {
	// Last row per session wins: a later `running` row RELEASES an earlier paused hold, and an
	// unknown token never clobbers a prior valid state. (Deliberately avoids the stopped-then-
	// running case, whose result differs under a sticky vs last-writer fold.)
	rows := []DriveStateRow{
		{Session: "a", State: "paused"},
		{Session: "b", State: "stopped"},
		{Session: "a", State: "running"},     // releases a's pause
		{Session: "b", State: "weird-token"}, // ignored — b stays stopped
		{Session: "", State: "paused"},       // no session id — dropped
	}
	got := FoldDriveStates(rows)
	if got["a"].HeldByOperator() {
		t.Fatalf("a should be released by the later running row, got %q (held)", got["a"])
	}
	if got["b"] != DriveStopped || !got["b"].HeldByOperator() {
		t.Fatalf("b should stay stopped despite the unknown later token, got %q", got["b"])
	}
	if _, ok := got[""]; ok {
		t.Fatal("a row with no session id must be dropped")
	}
	// The held vocabulary, and a reason lexically distinct from the account-level "tombstoned".
	if DriveRunning.HeldByOperator() || DriveThrottled.HeldByOperator() || WatchdogDriveState("").HeldByOperator() {
		t.Fatal("running / throttled / empty must not be holds")
	}
	for _, st := range []WatchdogDriveState{DrivePaused, DriveDraining, DriveStopped} {
		if !st.HeldByOperator() {
			t.Fatalf("%s must be a hold", st)
		}
		if r := strings.ToLower(st.HoldReason()); !strings.Contains(r, "operator") || strings.Contains(r, "tombstoned") {
			t.Fatalf("%s hold reason %q must name the operator and not collide with the account 'tombstoned' reason", st, st.HoldReason())
		}
	}
}

func TestResolveWatchdogProbeMode(t *testing.T) {
	if got := ResolveWatchdogProbeMode("auto", false); got != "none" {
		t.Fatalf("auto dry-run = %q, want none (a default tick must spend nothing)", got)
	}
	if got := ResolveWatchdogProbeMode("auto", true); got != "stale" {
		t.Fatalf("auto live = %q, want stale (idle seats need fresh evidence before taking rehomed load)", got)
	}
	if got := ResolveWatchdogProbeMode("ALL", false); got != "all" {
		t.Fatalf("explicit setting must be honored (normalized), got %q", got)
	}
	if got := ResolveWatchdogProbeMode("", true); got != "stale" {
		t.Fatalf("empty setting behaves as auto, got %q", got)
	}
}

func TestWatchdogPlanRowCodexHarnessRoundTrip(t *testing.T) {
	want := WatchdogPlanRow{Session: "thread", Harness: "codex", CWD: "/repo", Rollout: "rollout.jsonl", GoalFile: "goal.txt", ResultFile: "result.json"}
	raw, err := json.Marshal(want)
	if err != nil {
		t.Fatal(err)
	}
	var got WatchdogPlanRow
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got=%+v want=%+v", got, want)
	}
}

func TestCodexWatchdogChildEnvDropsClaudeHarnessAndKeepsCodexHome(t *testing.T) {
	got := CodexWatchdogChildEnv([]string{"ANTHROPIC_BASE_URL=http://parent", "CLAUDE_CONFIG_DIR=/claude", "CLAUDE_CODE_SESSION_ID=parent", "CODEX_HOME=/codex", "PATH=/bin"})
	want := []string{"CODEX_HOME=/codex", "PATH=/bin"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got=%q want=%q", got, want)
	}
}

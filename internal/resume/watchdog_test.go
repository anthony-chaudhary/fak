package resume

import (
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
//   - probe mode "auto" resolves to a real probe only on a live tick.

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
	dead := ReDeathEvidence{ProcessScanOK: true, TranscriptIdleSeconds: 3600}
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
		"live process":     {ProcessScanOK: true, ProcessLive: true, TranscriptIdleSeconds: 3600},
		"unreadable table": {ProcessScanOK: false, TranscriptIdleSeconds: 3600},
		"no transcript":    {ProcessScanOK: true, TranscriptIdleSeconds: -1},
		"fresh transcript": {ProcessScanOK: true, TranscriptIdleSeconds: DeadTranscriptIdleFloorSeconds - 1},
	} {
		if got, revived := ReviveOutcome(OutcomeProgressed, ev); revived || got != OutcomeProgressed {
			t.Fatalf("%s: ReviveOutcome = (%s, %v), want the burn kept", name, got, revived)
		}
	}
}

func TestReviveOutcomeThroughWatchdogGate(t *testing.T) {
	// The #2368 acceptance chain, leaf half: a prior take marker (launched history,
	// progressed outcome) plus proof of a new death must reach LAUNCH, not the
	// "already resumed once" skip.
	dead := ReDeathEvidence{ProcessScanOK: true, TranscriptIdleSeconds: 3600}
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

func TestResolveWatchdogProbeMode(t *testing.T) {
	if got := ResolveWatchdogProbeMode("auto", false); got != "none" {
		t.Fatalf("auto dry-run = %q, want none (a default tick must spend nothing)", got)
	}
	if got := ResolveWatchdogProbeMode("auto", true); got != "blocked" {
		t.Fatalf("auto live = %q, want blocked", got)
	}
	if got := ResolveWatchdogProbeMode("ALL", false); got != "all" {
		t.Fatalf("explicit setting must be honored (normalized), got %q", got)
	}
	if got := ResolveWatchdogProbeMode("", true); got != "blocked" {
		t.Fatalf("empty setting behaves as auto, got %q", got)
	}
}

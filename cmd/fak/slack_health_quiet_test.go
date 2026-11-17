package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
	"time"
)

// TestQuietBrakeRegradesFaultsAsQuiet is the contract that keeps the brake from
// manufacturing its own alarm: with FAK_SLACK_QUIET engaged, a stale cadence surface and
// a stalled outbox are the INTENDED state, so they re-grade QUIET (non-actionable) and
// the exit code stays 0. Without this, one budget window of silence would redden every
// surface and slack-watchdog.yml would file a GitHub issue about the silence the
// operator asked for.
func TestQuietBrakeRegradesFaultsAsQuiet(t *testing.T) {
	clearSlackEnv(t)
	t.Setenv("FAK_SCOREBOARD_TOKEN", "bottok-ok")
	t.Setenv("FAK_SCOREBOARD_CHANNEL", "C0SCORE") // daily cadence => 36h budget
	t.Setenv("FAK_SLACK_QUIET", "1")

	hub := newHealthHub(t)
	now := time.Date(2026, 6, 29, 12, 0, 0, 0, time.UTC)
	// Posted 10 days ago => stale against the 36h budget, i.e. exactly what a muted
	// fleet looks like after one budget window.
	hub.historyTS["C0SCORE"] = slackTS(now.Add(-10 * 24 * time.Hour))

	reports := buildSurfaceReports()
	runAuthChecks(reports, hub.base())
	health := foldSlackHealth(reports, hub.base(), now)
	health = applyQuietBrake(health, true, "env", "1")

	sb := healthByName(health, "scoreboard")
	if sb == nil || sb.Verdict != verdictQuiet {
		t.Fatalf("a stale surface under the brake should be QUIET: %+v", sb)
	}
	// The diagnosis is preserved, not thrown away — the operator can still see WHY it
	// would have been stale.
	if !strings.Contains(sb.Detail, "STALE") || !strings.Contains(sb.Detail, "FAK_SLACK_QUIET") {
		t.Fatalf("QUIET detail should preserve the original verdict and name the switch: %q", sb.Detail)
	}
	// The gate itself is asserted in TestQuietBrakeKeepsTheGateOpen; here the other
	// surfaces are INCOMPLETE (no channel in a cleared env), which legitimately exits 1.
}

// TestQuietBrakeKeepsTheGateOpen is the exit-code half, isolated from the INCOMPLETE
// surfaces a cleared env produces: a fold whose only non-OK rows are QUIET must pass the
// gate, because that is exactly the signal slack-watchdog.yml files an issue on.
func TestQuietBrakeKeepsTheGateOpen(t *testing.T) {
	health := applyQuietBrake([]healthReport{
		{Name: "scoreboard", Verdict: verdictStale, Detail: "10d old"},
		{Name: "outbox", Verdict: verdictOutboxStalled, Detail: "3h pending"},
		{Name: "product", Verdict: verdictOK},
		{Name: "marketing", Verdict: verdictDeferred},
	}, true, "env", "1")
	if code := healthExit(health); code != 0 {
		t.Fatalf("a fold of QUIET/OK/DEFERRED must exit 0, got %d: %+v", code, health)
	}
}

// TestQuietBrakeLeavesAuthFailAndIncompleteAlone: a rotated token and a config drift are
// real problems whether or not we are posting — the operator must learn about them BEFORE
// unmuting, or the first post after the quiet window silently goes nowhere.
func TestQuietBrakeLeavesAuthFailAndIncompleteAlone(t *testing.T) {
	clearSlackEnv(t)
	t.Setenv("FAK_SCOREBOARD_TOKEN", "bottok-bad")
	t.Setenv("FAK_SCOREBOARD_CHANNEL", "C0SCORE")
	t.Setenv("FAK_SLACK_QUIET", "1")

	hub := newHealthHub(t)
	hub.authOK = false
	now := time.Date(2026, 6, 29, 12, 0, 0, 0, time.UTC)

	reports := buildSurfaceReports()
	runAuthChecks(reports, hub.base())
	health := applyQuietBrake(foldSlackHealth(reports, hub.base(), now), true, "env", "1")

	sb := healthByName(health, "scoreboard")
	if sb == nil || sb.Verdict != verdictAuthFail {
		t.Fatalf("AUTH_FAIL must survive the brake (a rotated token is real): %+v", sb)
	}
	// Every other surface is INCOMPLETE (no channel configured) and must also survive.
	for _, h := range health {
		if h.Verdict == verdictQuiet {
			t.Fatalf("surface %q was re-graded QUIET; only stale/stalled/dead-rows may be: %+v", h.Name, h)
		}
	}
	if code := healthExit(health); code != 1 {
		t.Fatalf("a real fault under the brake must still exit 1, got %d", code)
	}
}

// TestQuietBrakeDisengagedIsIdentity pins the fail-safe direction: with the switch unset
// the fold is a no-op, so a fresh clone and a fresh CI runner grade exactly as before.
func TestQuietBrakeDisengagedIsIdentity(t *testing.T) {
	in := []healthReport{
		{Name: "scoreboard", Verdict: verdictStale, Detail: "old"},
		{Name: "outbox", Verdict: verdictOutboxStalled, Detail: "stuck"},
		{Name: "product", Verdict: verdictOK},
	}
	got := applyQuietBrake(in, false, "unset", "")
	for i := range got {
		if got[i].Verdict != in[i].Verdict || got[i].Detail != in[i].Detail {
			t.Fatalf("disengaged brake changed row %d: %+v -> %+v", i, in[i], got[i])
		}
	}
}

// TestQuietBrakeCoversOutboxRungs: the outbox rung is the one that would otherwise file
// an issue on its own (pending rows aging past the stall budget is exactly what a muted
// fleet produces), so both of its alarm verdicts must re-grade.
func TestQuietBrakeCoversOutboxRungs(t *testing.T) {
	in := []healthReport{
		{Name: "outbox", Verdict: verdictOutboxStalled, Detail: "oldest pending row is 3h old"},
		{Name: "outbox2", Verdict: verdictDeadRows, Detail: "2 dead row(s)"},
	}
	got := applyQuietBrake(in, true, "file", "yes")
	for _, h := range got {
		if h.Verdict != verdictQuiet {
			t.Errorf("%s = %q, want QUIET", h.Name, h.Verdict)
		}
		if !strings.Contains(h.Detail, ".env.slack.local") {
			t.Errorf("%s detail should name the file seam: %q", h.Name, h.Detail)
		}
		// The detail echoes the word the operator actually typed, not a hardcoded one:
		// a report that says "=1" while the switch reads "yes" is a report that disagrees
		// with the state it is describing.
		if !strings.Contains(h.Detail, "FAK_SLACK_QUIET=yes") {
			t.Errorf("%s detail should echo the resolved value: %q", h.Name, h.Detail)
		}
	}
}

// TestSlackHealthQuietEndToEnd drives the real verb with the switch engaged: the JSON
// report must carry QUIET rows and the process must exit 0, which is precisely the
// signal slack-watchdog.yml gates its issue-filing on.
func TestSlackHealthQuietEndToEnd(t *testing.T) {
	clearSlackEnv(t)
	t.Setenv("FAK_SCOREBOARD_TOKEN", "bottok-ok")
	t.Setenv("FAK_SCOREBOARD_CHANNEL", "C0SCORE")
	t.Setenv("FAK_SLACK_QUIET", "1")

	// An empty channel: without the brake this is the silent-dead-feeder alarm (STALE).
	hub := newHealthHub(t)

	var out, errb bytes.Buffer
	code := runSlackHealth(&out, &errb, []string{"--json", "--api-base", hub.base()})
	// The exit code is 1 here only because the OTHER surfaces are INCOMPLETE (no channel
	// configured in a cleared env) — a real fault the brake deliberately does not mask.
	// What matters is that the stale surface itself came back QUIET, not STALE.
	if code != 1 {
		t.Fatalf("exit = %d, want 1 (the unconfigured surfaces are genuinely INCOMPLETE)", code)
	}
	var health []healthReport
	if err := json.Unmarshal(out.Bytes(), &health); err != nil {
		t.Fatalf("decode health json: %v\n%s", err, out.String())
	}
	sb := healthByName(health, "scoreboard")
	if sb == nil || sb.Verdict != verdictQuiet {
		t.Fatalf("scoreboard should be QUIET with the brake engaged: %+v", sb)
	}
	// And the gate would be open if the INCOMPLETE surfaces were configured: no row is
	// left in an alarm verdict.
	for _, h := range health {
		switch h.Verdict {
		case verdictStale, verdictOutboxStalled, verdictDeadRows:
			t.Fatalf("surface %q still alarms under the brake: %+v", h.Name, h)
		}
	}
}

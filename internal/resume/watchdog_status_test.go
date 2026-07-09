package resume

import (
	"strings"
	"testing"
)

func TestFoldWatchdogStatusNonDrainingQueueTurnsRed(t *testing.T) {
	got := FoldWatchdogStatus(WatchdogStatusInput{
		Mode:           "LIVE",
		NowUnix:        20_000,
		SilentSeconds:  3600,
		MonotonicTicks: 3,
		Plan: []WatchdogPlanRow{
			{Session: "sid-stuck", Account: ".claude-a"},
			{Session: "sid-2", Account: ".claude-a"},
			{Session: "sid-3", Account: ".claude-a"},
			{Session: "sid-4", Account: ".claude-a"},
		},
		Events: []WatchdogStatusEvent{
			{UnixSeconds: 1_000, Phase: "status", Mode: "LIVE", AutoResumeDepth: 1},
			{UnixSeconds: 2_000, Phase: "status", Mode: "LIVE", AutoResumeDepth: 2},
			{UnixSeconds: 3_000, Phase: "status", Mode: "LIVE", AutoResumeDepth: 3},
			{UnixSeconds: 1_000, Session: "sid-stuck", Phase: "queued", Mode: "LIVE"},
		},
	})

	if got.Verdict != WatchdogDrainRed {
		t.Fatalf("verdict = %s, want red: %+v", got.Verdict, got)
	}
	if got.Mode != "LIVE" || got.AutoResumeMonotonicTicks != 3 {
		t.Fatalf("mode/ticks = %s/%d, want LIVE/3", got.Mode, got.AutoResumeMonotonicTicks)
	}
	joined := strings.Join(got.Reasons, "\n")
	if !strings.Contains(joined, "monotonically") || !strings.Contains(joined, "silent") {
		t.Fatalf("reasons must name monotonic growth and silent age, got %q", joined)
	}
	row := watchdogTestRow(got.MTTRSessions, "sid-stuck")
	if row.Status != WatchdogMTTRQueued {
		t.Fatalf("sid-stuck row = %+v, want queued", row)
	}
	if row.SilentSeconds <= 3600 {
		t.Fatalf("silent seconds = %d, want over the bound", row.SilentSeconds)
	}
}

func TestFoldWatchdogStatusDrainingQueueGreenWithMTTRWitness(t *testing.T) {
	got := FoldWatchdogStatus(WatchdogStatusInput{
		Mode:           "LIVE",
		NowUnix:        5_000,
		SilentSeconds:  3600,
		MonotonicTicks: 3,
		Events: []WatchdogStatusEvent{
			{UnixSeconds: 1_000, Phase: "status", Mode: "LIVE", AutoResumeDepth: 4},
			{UnixSeconds: 2_000, Phase: "status", Mode: "LIVE", AutoResumeDepth: 3},
			{UnixSeconds: 3_000, Phase: "status", Mode: "LIVE", AutoResumeDepth: 1},
			{UnixSeconds: 1_100, Session: "sid-drained", Phase: "queued", Mode: "LIVE"},
			{UnixSeconds: 1_200, Session: "sid-drained", Phase: "launched", Mode: "LIVE"},
			{UnixSeconds: 1_500, Session: "sid-drained", Phase: "progress", Mode: "LIVE", NewTurns: 2},
		},
	})

	if got.Verdict != WatchdogDrainGreen {
		t.Fatalf("verdict = %s, want green: %+v", got.Verdict, got)
	}
	if len(got.MTTRSessions) != 1 {
		t.Fatalf("mttr rows = %d, want 1", len(got.MTTRSessions))
	}
	row := got.MTTRSessions[0]
	if row.Status != WatchdogMTTRRecovered {
		t.Fatalf("status = %s, want recovered: %+v", row.Status, row)
	}
	if row.DetectedAt != 1_100 || row.ResumedAt != 1_200 || row.ProgressWitnessedAt != 1_500 {
		t.Fatalf("mttr timestamps = %+v, want detected/resumed/progress sequence", row)
	}
	if !strings.Contains(row.Evidence, "new_turns:2") {
		t.Fatalf("evidence = %q, want new-turn witness", row.Evidence)
	}
}

func TestFoldWatchdogStatusLaunchWithoutProgressIsNotRecovered(t *testing.T) {
	got := FoldWatchdogStatus(WatchdogStatusInput{
		Mode:           "LIVE",
		NowUnix:        1_300,
		SilentSeconds:  10_000,
		MonotonicTicks: 3,
		Plan:           []WatchdogPlanRow{{Session: "sid-launched", Account: ".claude-a"}},
		Events: []WatchdogStatusEvent{
			{UnixSeconds: 1_000, Session: "sid-launched", Phase: "queued", Mode: "LIVE"},
			{UnixSeconds: 1_100, Session: "sid-launched", Phase: "launched", Mode: "LIVE"},
		},
	})

	if got.Verdict != WatchdogDrainGreen {
		t.Fatalf("verdict = %s, want green when only the high-water alarms are quiet", got.Verdict)
	}
	if len(got.MTTRSessions) != 1 {
		t.Fatalf("mttr rows = %d, want 1", len(got.MTTRSessions))
	}
	row := got.MTTRSessions[0]
	if row.Status != WatchdogMTTRLaunchedUnproven {
		t.Fatalf("status = %s, want launched_unproven: %+v", row.Status, row)
	}
	if row.ProgressWitnessedAt != 0 || row.Evidence != "" {
		t.Fatalf("launch alone must not carry progress witness: %+v", row)
	}
	if row.UnprovenSeconds != 200 {
		t.Fatalf("unproven seconds = %d, want 200", row.UnprovenSeconds)
	}
}

func TestFoldWatchdogStatusOldLaunchWithoutProgressTurnsRed(t *testing.T) {
	got := FoldWatchdogStatus(WatchdogStatusInput{
		Mode:            "LIVE",
		NowUnix:         1_300,
		SilentSeconds:   10_000,
		UnprovenSeconds: 120,
		MonotonicTicks:  3,
		Plan:            []WatchdogPlanRow{{Session: "sid-launched", Account: ".claude-a"}},
		Events: []WatchdogStatusEvent{
			{UnixSeconds: 1_000, Session: "sid-launched", Phase: "queued", Mode: "LIVE"},
			{UnixSeconds: 1_100, Session: "sid-launched", Phase: "launched", Mode: "LIVE"},
		},
	})

	if got.Verdict != WatchdogDrainRed {
		t.Fatalf("verdict = %s, want red for old launched-unproven row: %+v", got.Verdict, got)
	}
	if got.UnprovenSeconds != 200 {
		t.Fatalf("max unproven seconds = %d, want 200", got.UnprovenSeconds)
	}
	joined := strings.Join(got.Reasons, "\n")
	if !strings.Contains(joined, "launched resume") || !strings.Contains(joined, "unproven") {
		t.Fatalf("reasons = %q, want launched-unproven alarm", joined)
	}
}

func TestFoldWatchdogStatusAuthPlanDoesNotMasqueradeAsUnproven(t *testing.T) {
	got := FoldWatchdogStatus(WatchdogStatusInput{
		Mode:            "LIVE",
		NowUnix:         1_300,
		SilentSeconds:   10_000,
		UnprovenSeconds: 120,
		MonotonicTicks:  3,
		Plan:            []WatchdogPlanRow{{Session: "sid-auth", Account: ".claude-a", Disp: "INFRA_AUTH"}},
		Events: []WatchdogStatusEvent{
			{UnixSeconds: 1_000, Session: "sid-auth", Phase: "queued", Mode: "LIVE"},
			{UnixSeconds: 1_100, Session: "sid-auth", Phase: "launched", Mode: "LIVE"},
		},
	})

	if got.Verdict != WatchdogDrainRed {
		t.Fatalf("verdict = %s, want red for auth-blocked row: %+v", got.Verdict, got)
	}
	row := watchdogTestRow(got.MTTRSessions, "sid-auth")
	if row.Status != WatchdogMTTRAuthBlocked {
		t.Fatalf("sid-auth row = %+v, want auth_blocked", row)
	}
	if row.UnprovenSeconds != 0 {
		t.Fatalf("auth-blocked row should not count as launched-unproven: %+v", row)
	}
	joined := strings.Join(got.Reasons, "\n")
	if !strings.Contains(joined, "auth/login") || strings.Contains(joined, "launched resume") {
		t.Fatalf("reasons = %q, want auth/login without launched-unproven alarm", joined)
	}
}

func TestFoldWatchdogStatusLegacyPhaseLessLaunchAndSettledRows(t *testing.T) {
	got := FoldWatchdogStatus(WatchdogStatusInput{
		Mode:           "LIVE",
		NowUnix:        2_000,
		SilentSeconds:  10_000,
		MonotonicTicks: 3,
		Events: []WatchdogStatusEvent{
			{UnixSeconds: 1_000, Session: "sid-legacy"},
			{UnixSeconds: 1_050, Session: "sid-settled", Phase: "settled"},
		},
	})

	if len(got.MTTRSessions) != 1 {
		t.Fatalf("mttr rows = %+v, want only the legacy launch row", got.MTTRSessions)
	}
	row := got.MTTRSessions[0]
	if row.Session != "sid-legacy" || row.Status != WatchdogMTTRLaunchedUnproven || row.ResumedAt != 1_000 {
		t.Fatalf("legacy launch row = %+v, want launched_unproven at 1000", row)
	}
}

func TestFoldWatchdogStatusCurrentPlanReopensRecoveredSession(t *testing.T) {
	got := FoldWatchdogStatus(WatchdogStatusInput{
		Mode:    "LIVE",
		NowUnix: 3_000,
		Plan:    []WatchdogPlanRow{{Session: "sid-reopened", Account: ".claude-a"}},
		Events: []WatchdogStatusEvent{
			{UnixSeconds: 1_000, Session: "sid-reopened", Phase: "queued", Mode: "LIVE"},
			{UnixSeconds: 1_100, Session: "sid-reopened", Phase: "launched", Mode: "LIVE"},
			{UnixSeconds: 1_200, Session: "sid-reopened", Phase: "progress", Mode: "LIVE", NewTurns: 1},
		},
	})

	// A recovered session re-appearing in the live plan reopens as queued, but its
	// detected/resumed times are NOT fabricated from now (#3460) — they stay 0 (—) until
	// a fresh ledger row lands, so a week-dead ledger can never read "resumed just now".
	row := watchdogTestRow(got.MTTRSessions, "sid-reopened")
	if row.Status != WatchdogMTTRQueued || row.DetectedAt != 0 || row.ResumedAt != 0 {
		t.Fatalf("reopened row = %+v, want queued with no fabricated detected/resumed time", row)
	}
}

// TestFoldWatchdogStatusStalePlanNoLaunchIsRedHeadline pins acceptance #2 of #3460:
// a non-empty plan whose sessions were never launched (empty ledger) turns RED with an
// "AUTO-RESUME NOT LAUNCHING" headline naming the queue depth — instead of reading
// healthy — and the queued rows carry NO fabricated detected/resumed times (acceptance #1).
func TestFoldWatchdogStatusStalePlanNoLaunchIsRedHeadline(t *testing.T) {
	got := FoldWatchdogStatus(WatchdogStatusInput{
		Mode:               "DRY-RUN", // a --status read is dry by nature; must not be the reason
		NowUnix:            1_000_000,
		SilentSeconds:      7200,
		LaunchStaleSeconds: 1800,
		MonotonicTicks:     3,
		Plan: []WatchdogPlanRow{
			{Session: "sid-q1", Account: ".claude-a"},
			{Session: "sid-q2", Account: ".claude-a"},
			{Session: "sid-q3", Account: ".claude-a"},
		},
	})

	if got.Verdict != WatchdogDrainRed {
		t.Fatalf("verdict = %s, want red for a queued-but-never-launched plan: %+v", got.Verdict, got)
	}
	if len(got.Reasons) == 0 || !strings.Contains(got.Reasons[0], "AUTO-RESUME NOT LAUNCHING") {
		t.Fatalf("reasons[0] must be the not-launching headline, got %q", got.Reasons)
	}
	if !strings.Contains(got.Reasons[0], "3 queued") {
		t.Fatalf("headline must name the queue depth, got %q", got.Reasons[0])
	}
	for _, row := range got.MTTRSessions {
		if row.DetectedAt != 0 || row.ResumedAt != 0 {
			t.Fatalf("queued row carries fabricated times: %+v", row)
		}
	}
}

// TestFoldWatchdogStatusStalePlanOldLaunchNamesAge proves the headline reports the age of
// the last REAL launch when one exists but is far in the past (a week-dead ledger), and
// that a launch for a since-settled, out-of-plan session still counts as the last launch.
func TestFoldWatchdogStatusStalePlanOldLaunchNamesAge(t *testing.T) {
	got := FoldWatchdogStatus(WatchdogStatusInput{
		Mode:               "DRY-RUN",
		NowUnix:            1_000 + 8*86400, // 8 days after the last launch
		LaunchStaleSeconds: 1800,
		Plan:               []WatchdogPlanRow{{Session: "sid-fresh", Account: ".claude-a"}},
		Events: []WatchdogStatusEvent{
			{UnixSeconds: 900, Session: "sid-old", Phase: "launched", Mode: "LIVE"},
			{UnixSeconds: 1_000, Session: "sid-old", Phase: "settled", Mode: "LIVE"},
		},
	})

	if got.Verdict != WatchdogDrainRed {
		t.Fatalf("verdict = %s, want red: %+v", got.Verdict, got)
	}
	if len(got.Reasons) == 0 || !strings.Contains(got.Reasons[0], "AUTO-RESUME NOT LAUNCHING") ||
		!strings.Contains(got.Reasons[0], "8.0d ago") || !strings.Contains(got.Reasons[0], "1 queued") {
		t.Fatalf("headline must name last-launch age and depth, got %q", got.Reasons)
	}
}

// TestFoldWatchdogStatusInstalledDryRunReasonFromLastTick proves the DRY-RUN reason keys
// off the INSTALLED watchdog's last real tick mode, not the --status invocation mode: a
// dry read of a LIVE-installed watchdog must NOT emit the DRY-RUN reason (the spurious one
// the audit skill warns about), while a genuinely DRY-RUN-installed watchdog must.
func TestFoldWatchdogStatusInstalledDryRunReasonFromLastTick(t *testing.T) {
	liveInstalled := FoldWatchdogStatus(WatchdogStatusInput{
		Mode:    "DRY-RUN", // the --status read is dry
		NowUnix: 5_000,
		Plan:    []WatchdogPlanRow{{Session: "sid-1", Account: ".claude-a"}},
		Events: []WatchdogStatusEvent{
			{UnixSeconds: 4_000, Phase: "status", Mode: "LIVE", AutoResumeDepth: 1}, // installed watchdog ticks LIVE
		},
	})
	if joined := strings.Join(liveInstalled.Reasons, "\n"); strings.Contains(joined, "DRY-RUN") {
		t.Fatalf("a dry --status read of a LIVE-installed watchdog must not emit a DRY-RUN reason, got %q", joined)
	}

	dryInstalled := FoldWatchdogStatus(WatchdogStatusInput{
		Mode:    "DRY-RUN",
		NowUnix: 5_000,
		Plan:    []WatchdogPlanRow{{Session: "sid-1", Account: ".claude-a"}},
		Events: []WatchdogStatusEvent{
			{UnixSeconds: 4_000, Phase: "status", Mode: "DRY-RUN", AutoResumeDepth: 1}, // installed watchdog ticks DRY-RUN
		},
	})
	if joined := strings.Join(dryInstalled.Reasons, "\n"); !strings.Contains(joined, "DRY-RUN") || !strings.Contains(joined, "-Live") {
		t.Fatalf("a DRY-RUN-installed watchdog with queued rows must flag it, got %q", joined)
	}
}

func TestFoldWatchdogStatusCurrentPlanBoundsDepthAndRows(t *testing.T) {
	got := FoldWatchdogStatus(WatchdogStatusInput{
		Mode:    "LIVE",
		NowUnix: 10_000,
		Plan: []WatchdogPlanRow{
			{Session: "sid-current-1", Account: ".claude-a"},
			{Session: "sid-current-2", Account: ".claude-a"},
		},
		Events: []WatchdogStatusEvent{
			{UnixSeconds: 1_000, Phase: "status", Mode: "LIVE", AutoResumeDepth: 61},
			{UnixSeconds: 1_100, Session: "sid-stale", Phase: "queued", Mode: "LIVE"},
			{UnixSeconds: 1_200, Session: "sid-stale", Phase: "launched", Mode: "LIVE"},
			{UnixSeconds: 2_000, Session: "sid-current-1", Phase: "queued", Mode: "LIVE"},
		},
	})

	if got.AutoResumeDepth != 2 {
		t.Fatalf("auto resume depth = %d, want current plan depth 2", got.AutoResumeDepth)
	}
	if watchdogTestRow(got.MTTRSessions, "sid-stale").Session != "" {
		t.Fatalf("stale row leaked into current plan report: %+v", got.MTTRSessions)
	}
	if len(got.MTTRSessions) != 2 {
		t.Fatalf("mttr rows = %+v, want only two current plan rows", got.MTTRSessions)
	}
}

func watchdogTestRow(rows []WatchdogMTTRRow, session string) WatchdogMTTRRow {
	for _, row := range rows {
		if row.Session == session {
			return row
		}
	}
	return WatchdogMTTRRow{}
}

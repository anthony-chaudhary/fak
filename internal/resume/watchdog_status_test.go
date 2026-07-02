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

	row := watchdogTestRow(got.MTTRSessions, "sid-reopened")
	if row.Status != WatchdogMTTRQueued || row.DetectedAt != 3_000 {
		t.Fatalf("reopened row = %+v, want current plan to reopen as queued at now", row)
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

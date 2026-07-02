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

func watchdogTestRow(rows []WatchdogMTTRRow, session string) WatchdogMTTRRow {
	for _, row := range rows {
		if row.Session == session {
			return row
		}
	}
	return WatchdogMTTRRow{}
}

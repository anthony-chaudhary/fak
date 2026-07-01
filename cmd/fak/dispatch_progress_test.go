package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestDispatchProgressHourlyProjectionFromRecentClosedLedger(t *testing.T) {
	now := time.Date(2026, 7, 1, 12, 0, 0, 0, time.UTC)
	runsDir := filepath.Join(t.TempDir(), dispatchProgressRunsDir)
	writeDispatchProgressRows(t, runsDir, []map[string]any{
		{"utc": "2026-07-01T10:00:00Z", "closed_now": 999}, // outside the one-hour window
		{"utc": "2026-07-01T11:30:00Z", "closed_now": 80},
		{"utc": "2026-07-01T11:45:00Z", "closed_now": 40},
		{"utc": "2026-07-01T11:50:00Z", "closed_now": 0},
	})

	got := dispatchProgressHourlyProjection(runsDir, now, map[string]any{
		"utc":        "2026-07-01T12:00:00Z",
		"closed_now": 0,
	})
	if got["current_issues_per_hour"] != 480.0 ||
		got["target_issues_per_hour"] != 400.0 ||
		got["issues_per_hour_gap"] != 0.0 ||
		got["projection_closed_count"] != 120 ||
		got["projection_window_hours"] != 0.25 {
		t.Fatalf("reaching projection = %+v, want 120 closes over 0.25h => 480/h gap 0", got)
	}

	runsDir = filepath.Join(t.TempDir(), dispatchProgressRunsDir)
	writeDispatchProgressRows(t, runsDir, []map[string]any{
		{"utc": "2026-07-01T11:30:00Z", "closed_now": 50},
		{"utc": "2026-07-01T12:00:00Z", "closed_now": 50},
	})
	got = dispatchProgressHourlyProjection(runsDir, now, nil)
	if got["current_issues_per_hour"] != 200.0 ||
		got["target_issues_per_hour"] != 400.0 ||
		got["issues_per_hour_gap"] != 200.0 ||
		got["projection_closed_count"] != 100 ||
		got["projection_window_hours"] != 0.5 {
		t.Fatalf("missing projection = %+v, want 100 closes over 0.5h => 200/h gap 200", got)
	}
}

func TestRenderDispatchProgressIncludesHourlyProjection(t *testing.T) {
	out := renderDispatchProgress(map[string]any{
		"target":                  50,
		"open_now":                479,
		"baseline_open":           483,
		"resolved_toward_target":  4,
		"target_remaining":        46,
		"witnessed_open":          2,
		"witnessed_numbers":       []int{491, 493},
		"closed_now":              0,
		"closed_by_loop_total":    120,
		"current_issues_per_hour": 480.0,
		"target_issues_per_hour":  400.0,
		"issues_per_hour_gap":     0.0,
		"projection_closed_count": 120,
		"projection_window_hours": 0.25,
	})
	for _, want := range []string{
		"hourly projection:",
		"current=480.0/h",
		"target=400.0/h",
		"gap=0.0/h",
		"closes=120",
		"window=0.25h",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("rendered progress missing %q:\n%s", want, out)
		}
	}
}

func TestDispatchWeeklyReportFromFixtureLedger(t *testing.T) {
	runsDir := filepath.Join(t.TempDir(), dispatchProgressRunsDir)
	writeDispatchProgressRows(t, runsDir, []map[string]any{
		{"utc": "2026-06-30T23:00:00Z", "closed_now": 99, "target_issues_per_hour": 40.0}, // outside window
		{"utc": "2026-07-01T00:10:00Z", "ok": true, "closed_now": 20, "target_issues_per_hour": 40.0},
		{"utc": "2026-07-01T00:30:00Z", "ok": false, "closed_now": 10, "audit_error": "commit audit unavailable"},
		{"utc": "2026-07-01T00:40:00Z", "ok": false, "closed_now": 0, "audit_error": "commit audit unavailable"},
		{"utc": "2026-07-01T00:50:00Z", "ok": false, "closed_now": 0, "open_error": "gh rate limit"},
	})

	since := time.Date(2026, 7, 1, 0, 0, 0, 0, time.UTC)
	until := time.Date(2026, 7, 1, 1, 0, 0, 0, time.UTC)
	got, err := buildDispatchWeeklyReport(runsDir, since, until)
	if err != nil {
		t.Fatal(err)
	}
	if got.Schema != dispatchWeeklySchema ||
		got.RowsConsidered != 4 ||
		got.WitnessedCloses != 30 ||
		got.TargetIssuesPerHour != 40.0 ||
		got.AchievedWitnessedClosesPerHour != 30.0 ||
		got.CapacityLossIssues != 10.0 {
		t.Fatalf("weekly report = %+v, want 30 closes over 1h against 40/h", got)
	}
	if len(got.TopBlockers) < 2 ||
		got.TopBlockers[0] != (dispatchWeeklyBlocker{Reason: "AUDIT_UNAVAILABLE", Count: 2}) ||
		got.TopBlockers[1] != (dispatchWeeklyBlocker{Reason: "OPEN_COUNT_UNAVAILABLE", Count: 1}) {
		t.Fatalf("top blockers = %+v, want audit before open-count", got.TopBlockers)
	}
	if got.NextSafeCapChange != "hold cap; clear AUDIT_UNAVAILABLE before raising" {
		t.Fatalf("next cap change = %q", got.NextSafeCapChange)
	}
}

func TestRenderDispatchWeeklyReportMarkdown(t *testing.T) {
	report := dispatchWeeklyReport{
		Schema:                         dispatchWeeklySchema,
		WindowStartUTC:                 "2026-07-01T00:00:00Z",
		WindowEndUTC:                   "2026-07-01T01:00:00Z",
		WindowHours:                    1.0,
		RowsConsidered:                 4,
		TargetIssuesPerHour:            40.0,
		WitnessedCloses:                30,
		AchievedWitnessedClosesPerHour: 30.0,
		CapacityLossIssues:             10.0,
		TopBlockers:                    []dispatchWeeklyBlocker{{Reason: "AUDIT_UNAVAILABLE", Count: 2}},
		NextSafeCapChange:              "hold cap; clear AUDIT_UNAVAILABLE before raising",
	}
	out := renderDispatchWeeklyReport(report)
	for _, want := range []string{
		"# Dispatch Weekly Throughput Retrospective",
		"| target witnessed closes/hour | 40.0 |",
		"| achieved witnessed closes/hour | 30.0 |",
		"| capacity loss | 10.0 issue(s) |",
		"- AUDIT_UNAVAILABLE: 2",
		"Next safe cap change: hold cap; clear AUDIT_UNAVAILABLE before raising",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("weekly markdown missing %q:\n%s", want, out)
		}
	}
}

func TestDispatchProgressWeeklyModeReadsLedgerWithoutAppending(t *testing.T) {
	root := t.TempDir()
	runsDir := filepath.Join(root, dispatchProgressRunsDir)
	writeDispatchProgressRows(t, runsDir, []map[string]any{
		{"utc": "2026-07-01T00:10:00Z", "ok": true, "closed_now": 20, "target_issues_per_hour": 40.0},
		{"utc": "2026-07-01T00:30:00Z", "ok": false, "closed_now": 10, "audit_error": "commit audit unavailable"},
	})
	logPath := filepath.Join(runsDir, dispatchProgressLogName)
	before, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	code := runDispatchProgress(&stdout, &stderr, []string{
		"--workspace", root,
		"--weekly",
		"--since", "2026-07-01T00:00:00Z",
		"--until", "2026-07-01T01:00:00Z",
		"--json",
	})
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (stderr: %s)", code, stderr.String())
	}
	var got dispatchWeeklyReport
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
		t.Fatalf("bad json: %v\n%s", err, stdout.String())
	}
	if got.WitnessedCloses != 30 || got.RowsConsidered != 2 {
		t.Fatalf("weekly cli report = %+v, want two ledger rows and 30 witnessed closes", got)
	}
	after, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(before, after) {
		t.Fatal("weekly report must not append or mutate the progress ledger")
	}
}

func writeDispatchProgressRows(t *testing.T, runsDir string, rows []map[string]any) {
	t.Helper()
	if err := os.MkdirAll(runsDir, 0o755); err != nil {
		t.Fatal(err)
	}
	var b strings.Builder
	for _, row := range rows {
		encoded, err := json.Marshal(row)
		if err != nil {
			t.Fatal(err)
		}
		b.Write(encoded)
		b.WriteByte('\n')
	}
	if err := os.WriteFile(filepath.Join(runsDir, dispatchProgressLogName), []byte(b.String()), 0o644); err != nil {
		t.Fatal(err)
	}
}

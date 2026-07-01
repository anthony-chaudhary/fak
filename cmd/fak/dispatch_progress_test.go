package main

import (
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

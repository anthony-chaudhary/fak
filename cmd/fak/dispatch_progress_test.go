package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
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

func TestDispatchWeeklyReportSurfacesSlowLaneWedges(t *testing.T) {
	runsDir := filepath.Join(t.TempDir(), dispatchProgressRunsDir)
	writeDispatchProgressRows(t, runsDir, []map[string]any{
		{"utc": "2026-07-01T00:05:00Z", "lane": "cmd", "ok": false, "closed_now": 0, "audit_error": "commit audit unavailable"},
		{"utc": "2026-07-01T00:10:00Z", "lane": "cmd", "ok": false, "closed_now": 0, "audit_error": "commit audit unavailable"},
		{"utc": "2026-07-01T00:15:00Z", "lane": "cmd", "ok": false, "closed_now": 0, "audit_error": "commit audit unavailable"},
		{"utc": "2026-07-01T00:20:00Z", "lane": "docs", "ok": true, "closed_now": 1},
		{"utc": "2026-07-01T00:25:00Z", "lane": "docs", "ok": true, "closed_now": 1},
	})

	since := time.Date(2026, 7, 1, 0, 0, 0, 0, time.UTC)
	until := time.Date(2026, 7, 1, 1, 0, 0, 0, time.UTC)
	got, err := buildDispatchWeeklyReport(runsDir, since, until)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.LaneWedges) != 1 {
		t.Fatalf("lane wedges = %+v, want exactly the cmd lane wedged", got.LaneWedges)
	}
	wedge := got.LaneWedges[0]
	if wedge.Lane != "cmd" ||
		wedge.Attempts != 3 ||
		wedge.WitnessedCloses != 0 ||
		wedge.CloseRate != 0 ||
		wedge.DominantFailureClass != "AUDIT_UNAVAILABLE" ||
		wedge.DominantFailureCount != 3 ||
		!strings.Contains(wedge.NextAction, "clear AUDIT_UNAVAILABLE") {
		t.Fatalf("cmd wedge = %+v, want repeated audit blocker and no witnessed closes", wedge)
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
		LaneWedges: []dispatchLaneWedge{{
			Lane: "cmd", Attempts: 3, WitnessedCloses: 0, CloseRate: 0,
			DominantFailureClass: "AUDIT_UNAVAILABLE", DominantFailureCount: 3,
			NextAction: "inspect cmd lane; clear AUDIT_UNAVAILABLE before adding workers",
		}},
		NextSafeCapChange: "hold cap; clear AUDIT_UNAVAILABLE before raising",
	}
	out := renderDispatchWeeklyReport(report)
	for _, want := range []string{
		"# Dispatch Weekly Throughput Retrospective",
		"| target witnessed closes/hour | 40.0 |",
		"| achieved witnessed closes/hour | 30.0 |",
		"| capacity loss | 10.0 issue(s) |",
		"- AUDIT_UNAVAILABLE: 2",
		"## Lane Wedges",
		"- cmd: attempts=3 closes=0 close_rate=0.00 dominant=AUDIT_UNAVAILABLE/3",
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

// TestDispatchProgressClosuresTowardTargetSurvivesBacklogDrift is the #2639
// witness: a synthetic ledger where the loop witnessed real closures
// (closed_now>0) while the open backlog ROSE in the same window (new issues
// filed/reopened). The net-open metric drifts to zero and pins target_remaining
// at the full target, but the close-N counter counts the witnessed closures.
func TestDispatchProgressClosuresTowardTargetSurvivesBacklogDrift(t *testing.T) {
	root := t.TempDir()
	runsDir := filepath.Join(root, dispatchProgressRunsDir)
	writeDispatchProgressRows(t, runsDir, []map[string]any{
		{"utc": "2026-07-01T10:00:00Z", "open_now": 480, "closed_now": 3, "closed_by_loop_total": 3},
		{"utc": "2026-07-01T10:20:00Z", "open_now": 486, "closed_now": 4, "closed_by_loop_total": 7},
		{"utc": "2026-07-01T10:40:00Z", "open_now": 490, "closed_now": 2, "closed_by_loop_total": 9},
	})
	if err := dispatchProgressSaveBaseline(runsDir, 480); err != nil {
		t.Fatal(err)
	}

	restoreOpen, restoreAudit, restoreNow := dispatchProgressOpenCount, dispatchProgressAudit, dispatchProgressNow
	t.Cleanup(func() {
		dispatchProgressOpenCount, dispatchProgressAudit, dispatchProgressNow = restoreOpen, restoreAudit, restoreNow
	})
	dispatchProgressOpenCount = func(string) (int, error) { return 492, nil } // backlog rose above baseline
	dispatchProgressAudit = func(string, io.Writer, int, string) (map[string]any, error) {
		return map[string]any{"issues": []any{}}, nil
	}
	dispatchProgressNow = func() time.Time { return time.Date(2026, 7, 1, 11, 0, 0, 0, time.UTC) }

	rec, err := evaluateDispatchProgress(dispatchProgressOptions{Workspace: root, Target: 50, MaxCommits: 10}, io.Discard)
	if err != nil {
		t.Fatal(err)
	}

	// Net-open reduction drifted: baseline 480 - open 492 < 0 -> clamped 0, so the
	// old target_remaining pins at the full target despite 9 real witnessed closures.
	if got := dispatchMapInt(rec, "resolved_toward_target"); got != 0 {
		t.Fatalf("resolved_toward_target = %d, want 0 (net-open reduction drifted away)", got)
	}
	if got := dispatchMapInt(rec, "target_remaining"); got != 50 {
		t.Fatalf("target_remaining = %d, want 50 (net-open pins at full target)", got)
	}
	// The close-N counter reflects the 9 witnessed closures, backlog-independent.
	if got := dispatchMapInt(rec, "closures_toward_target"); got != 9 {
		t.Fatalf("closures_toward_target = %d, want 9 witnessed closures", got)
	}
	if got := dispatchMapInt(rec, "closures_target_remaining"); got != 41 {
		t.Fatalf("closures_target_remaining = %d, want 41 (50-9), not backlog drift", got)
	}
}

func TestDispatchProgressClosuresTowardTargetClamps(t *testing.T) {
	cases := []struct {
		closed, target, wantToward, wantRemaining int
	}{
		{0, 50, 0, 50},
		{9, 50, 9, 41},
		{50, 50, 50, 0},
		{63, 50, 50, 0}, // over-target: bar caps at target, remaining floors at 0
		{-1, 50, 0, 50}, // defensive: a negative fold is treated as 0
	}
	for _, tc := range cases {
		toward, remaining := dispatchProgressClosuresTowardTarget(tc.closed, tc.target)
		if toward != tc.wantToward || remaining != tc.wantRemaining {
			t.Fatalf("closures(%d,%d) = (%d,%d), want (%d,%d)",
				tc.closed, tc.target, toward, remaining, tc.wantToward, tc.wantRemaining)
		}
	}
}

func TestRenderDispatchProgressDisambiguatesClosuresFromBacklogDrift(t *testing.T) {
	out := renderDispatchProgress(map[string]any{
		"target": 50, "open_now": 492, "baseline_open": 480,
		"resolved_toward_target": 0, "target_remaining": 50,
		"closures_toward_target": 9, "closures_target_remaining": 41,
		"witnessed_open": 2, "witnessed_numbers": []int{491, 493},
		"closed_now": 0, "closed_by_loop_total": 9,
	})
	for _, want := range []string{
		"witnessed closures toward 50:",
		"] 9/50", // the bar tracks witnessed closures, not net-open reduction
		"witnessed closures remaining to 50: 41",
		"net-open reduction (baseline-open, drifts with backlog): 0  net-open remaining: 50",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("rendered progress missing %q:\n%s", want, out)
		}
	}
}

// TestDispatchProgressJSONFoldsAgingCensusOverReadySet is the #3590 witness: with a
// --aging-candidates ready set supplied, the progress JSON carries the anti-starvation census
// folded through dispatchaging — starved_count and oldest_wait_seconds — over a fixture set with
// one unit parked past the 6h hard starvation deadline.
func TestDispatchProgressJSONFoldsAgingCensusOverReadySet(t *testing.T) {
	root := t.TempDir()
	now := time.Date(2026, 7, 10, 12, 0, 0, 0, time.UTC)
	restoreOpen, restoreAudit, restoreNow := dispatchProgressOpenCount, dispatchProgressAudit, dispatchProgressNow
	t.Cleanup(func() {
		dispatchProgressOpenCount, dispatchProgressAudit, dispatchProgressNow = restoreOpen, restoreAudit, restoreNow
	})
	dispatchProgressOpenCount = func(string) (int, error) { return 100, nil }
	dispatchProgressAudit = func(string, io.Writer, int, string) (map[string]any, error) {
		return map[string]any{"issues": []any{}}, nil
	}
	dispatchProgressNow = func() time.Time { return now }

	// #42 ready 7h ago (starved, past the 6h deadline); #7 ready 30s ago (fresh).
	starvedSince := now.Add(-7 * time.Hour).Unix()
	freshSince := now.Add(-30 * time.Second).Unix()
	candPath := filepath.Join(root, "ready.json")
	if err := os.WriteFile(candPath, []byte(fmt.Sprintf(
		`[{"id":"42","base_weight":150,"ready_since":%d},{"id":"7","base_weight":1000,"ready_since":%d}]`,
		starvedSince, freshSince)), 0o644); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	code := runDispatchProgress(&stdout, &stderr, []string{
		"--workspace", root,
		"--json", "--no-loop-ledger",
		"--aging-candidates", candPath,
	})
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (stderr: %s)", code, stderr.String())
	}
	var got map[string]any
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
		t.Fatalf("bad json: %v\n%s", err, stdout.String())
	}
	if dispatchMapInt(got, "starved_count") != 1 {
		t.Fatalf("starved_count = %v, want 1", got["starved_count"])
	}
	if dispatchMapInt(got, "aging_count") != 0 {
		t.Fatalf("aging_count = %v, want 0", got["aging_count"])
	}
	if dispatchMapInt(got, "oldest_wait_seconds") != 7*3600 {
		t.Fatalf("oldest_wait_seconds = %v, want %d", got["oldest_wait_seconds"], 7*3600)
	}
}

// TestRenderDispatchProgressShowsAgingCensus is the #3590 human-render witness: when the aging
// census is present the readout gains a `starved: K  aging: A  oldest-wait: Th` line, and stays
// silent (additive) when no ready set was folded.
func TestRenderDispatchProgressShowsAgingCensus(t *testing.T) {
	base := map[string]any{
		"target": 50, "open_now": 100, "baseline_open": 100,
		"closures_toward_target": 0, "closures_target_remaining": 50,
		"witnessed_open": 0, "witnessed_numbers": []int{},
		"closed_now": 0, "closed_by_loop_total": 0,
	}
	withCensus := map[string]any{}
	for k, v := range base {
		withCensus[k] = v
	}
	withCensus["starved_count"] = 2
	withCensus["aging_count"] = 3
	withCensus["oldest_wait_seconds"] = int64(21600)

	out := renderDispatchProgress(withCensus)
	for _, want := range []string{"starved: 2", "aging: 3", "oldest-wait: 6h"} {
		if !strings.Contains(out, want) {
			t.Fatalf("rendered progress missing %q:\n%s", want, out)
		}
	}
	if bare := renderDispatchProgress(base); strings.Contains(bare, "starved:") {
		t.Fatalf("no ready set folded, but render showed a starved line:\n%s", bare)
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

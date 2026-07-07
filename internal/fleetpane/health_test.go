package fleetpane

import (
	"context"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/fleetmon"
)

// TestFleetPaneWorkerHealthFold is the #2035 witness: the fleet control pane folds
// the `fak fleet monitor` class histogram into ONE health summary, and the pane's
// JSON (StatusDoc.WorkerHealth) and its human readout (PaneText) expose the SAME
// per-class counts. The fixture carries at least one stale-transcript worker and
// one completed-final worker, exactly as the issue's done-condition names.
func TestFleetPaneWorkerHealthFold(t *testing.T) {
	root := testRoot(t)
	mustWrite(t, filepath.Join(root, "tools", "control_pane.loops.json"), `{"loops": {}}`)
	mustWrite(t, filepath.Join(root, "tools", "_registry", "control_pane.local.json"), `{}`)

	monitorJSON := `{
  "schema": "fak-fleet-monitor/1",
  "generated_at": "2026-06-30T11:59:00Z",
  "total": 4,
  "by_class": {
    "healthy": 1,
    "completed-final": 1,
    "stale-transcript": 1,
    "dead": 1
  }
}`
	cfg, err := LoadConfig(root)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	runner := &fakeRunner{
		paths: map[string]bool{"fak": true},
		runs: map[string]RunResult{
			strings.Join(cfg.MonitorStatus, "\x00"): {ExitCode: 0, Stdout: monitorJSON},
		},
	}
	opts := fixedOptions(runner)

	status := CollectStatus(context.Background(), cfg, opts)
	h := status.WorkerHealth
	if !h.Available {
		t.Fatalf("worker health should be available: %+v", h)
	}
	if h.Total != 4 {
		t.Fatalf("total mismatch: got %d want 4", h.Total)
	}
	if h.Counts["stale-transcript"] != 1 || h.Counts["completed-final"] != 1 {
		t.Fatalf("witness classes missing: stale-transcript=%d completed-final=%d", h.Counts["stale-transcript"], h.Counts["completed-final"])
	}
	// replacement-needed = dead + auth-or-rate-blocked + stale-transcript = 1+0+1.
	if h.ReplacementNeeded != 2 {
		t.Fatalf("replacement-needed mismatch: got %d want 2", h.ReplacementNeeded)
	}

	// The done-condition: the human output exposes the SAME counts as the JSON.
	// Reconstruct the counts from the rendered health line and require an exact
	// match against the struct the JSON serializes — a drift is the bug this guards.
	humanCounts := parseHealthLine(t, PaneText(status))
	jsonCounts := map[string]int{"total": h.Total, "replacement-needed": h.ReplacementNeeded}
	for class, n := range h.Counts {
		jsonCounts[class] = n
	}
	if !reflect.DeepEqual(humanCounts, jsonCounts) {
		t.Fatalf("pane JSON and human counts diverge:\n  json=%v\n  human=%v", jsonCounts, humanCounts)
	}
}

// TestWorkerHealthUnavailableWhenMonitorMissing proves the honest degraded rung:
// with no monitor witness the fold reports unavailable (with a reason) in both the
// JSON and the human line, never a misleading all-zero histogram.
func TestWorkerHealthUnavailableWhenMonitorMissing(t *testing.T) {
	root := testRoot(t)
	mustWrite(t, filepath.Join(root, "tools", "control_pane.loops.json"), `{"loops": {}}`)
	mustWrite(t, filepath.Join(root, "tools", "_registry", "control_pane.local.json"), `{}`)
	cfg, err := LoadConfig(root)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	runner := &fakeRunner{paths: map[string]bool{}} // `fak` not resolvable
	status := CollectStatus(context.Background(), cfg, fixedOptions(runner))
	if status.WorkerHealth.Available {
		t.Fatalf("worker health should be unavailable without a monitor: %+v", status.WorkerHealth)
	}
	if !strings.Contains(PaneText(status), "health: unavailable") {
		t.Fatalf("pane text should carry an unavailable health line:\n%s", PaneText(status))
	}
}

// TestWorkerHealthClassesMirrorFleetmon is the drift guard the health.go doc
// promises: the pane's canonical class order and its replacement-eligible set must
// mirror internal/fleetmon (the source of truth). A rename there fails this test
// rather than silently rendering a wrong or missing class.
func TestWorkerHealthClassesMirrorFleetmon(t *testing.T) {
	wantClasses := []string{
		string(fleetmon.ClassHealthy),
		string(fleetmon.ClassCompletedFinal),
		string(fleetmon.ClassDead),
		string(fleetmon.ClassStaleTranscript),
		string(fleetmon.ClassAuthRateBlocked),
		string(fleetmon.ClassStaleChild),
		string(fleetmon.ClassAttention),
	}
	if !reflect.DeepEqual(healthClassOrder, wantClasses) {
		t.Fatalf("healthClassOrder drifted from fleetmon:\n  pane=%v\n  fleetmon=%v", healthClassOrder, wantClasses)
	}
	wantReplacement := []string{
		string(fleetmon.ClassDead),
		string(fleetmon.ClassAuthRateBlocked),
		string(fleetmon.ClassStaleTranscript),
	}
	if !reflect.DeepEqual(healthReplacementClasses, wantReplacement) {
		t.Fatalf("healthReplacementClasses drifted from fleetmon:\n  pane=%v\n  fleetmon=%v", healthReplacementClasses, wantReplacement)
	}
}

// parseHealthLine extracts the "health:" readout PaneText emits and re-parses its
// `k=v` tokens into a count map, so a test can compare the human line against the
// JSON struct without depending on token order.
func parseHealthLine(t *testing.T, paneText string) map[string]int {
	t.Helper()
	for _, line := range strings.Split(paneText, "\n") {
		rest, ok := strings.CutPrefix(line, "health: ")
		if !ok {
			continue
		}
		counts := map[string]int{}
		for _, tok := range strings.Fields(rest) {
			k, v, ok := strings.Cut(tok, "=")
			if !ok {
				t.Fatalf("health token %q is not k=v", tok)
			}
			n, err := strconv.Atoi(v)
			if err != nil {
				t.Fatalf("health token %q has non-int value: %v", tok, err)
			}
			counts[k] = n
		}
		return counts
	}
	t.Fatalf("no health line in pane text:\n%s", paneText)
	return nil
}

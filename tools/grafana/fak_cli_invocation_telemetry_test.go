package grafana

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// fakCLITelemetryPath is the hand-authored CLI/dev-tool invocation-telemetry
// board (uid fak-cli-invocation-telemetry). No generator owns this artifact, so
// this test is its contract: it pins the board to the TWO scrape jobs whose
// exporters actually emit the families it queries — fak_fleet (:9098) for the
// public fak_cli_* runtime plane and fak_ops (:9101) for the private fak_tool_*
// developer plane — and guards against the board silently querying an exporter
// that never emits the family (a panel that renders empty while looking healthy).
const fakCLITelemetryPath = "dashboards/fak-cli-invocation-telemetry.json"

type fakCLIPanel struct {
	ID          int    `json:"id"`
	Type        string `json:"type"`
	Title       string `json:"title"`
	Description string `json:"description"`
	Targets     []struct {
		Expr string `json:"expr"`
	} `json:"targets"`
}

type fakCLIDashboard struct {
	Title  string        `json:"title"`
	UID    string        `json:"uid"`
	Panels []fakCLIPanel `json:"panels"`
}

func loadFakCLITelemetry(t *testing.T) fakCLIDashboard {
	t.Helper()
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	path := filepath.Join(filepath.Dir(file), filepath.FromSlash(fakCLITelemetryPath))
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("dashboard artifact missing at %s: %v", path, err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	var d fakCLIDashboard
	if err := json.Unmarshal(raw, &d); err != nil {
		t.Fatalf("%s is not well-formed dashboard JSON: %v", path, err)
	}
	return d
}

// TestFakCLITelemetryDashboardContract pins the board identity and the scrape
// job every family must select. fak_cli_* is emitted ONLY by the fak_fleet
// exporter and fak_tool_* ONLY by the fak_ops exporter; a panel that queried the
// generic fak_gateway job (or the wrong one) would read "No data" forever.
func TestFakCLITelemetryDashboardContract(t *testing.T) {
	d := loadFakCLITelemetry(t)

	if d.UID != "fak-cli-invocation-telemetry" {
		t.Errorf("expected uid fak-cli-invocation-telemetry, got %q", d.UID)
	}
	if d.Title == "" {
		t.Error("dashboard must carry a title")
	}

	var sawCLI, sawTool, sawFleetJob, sawOpsJob bool
	for _, p := range d.Panels {
		for _, tgt := range p.Targets {
			expr := strings.TrimSpace(tgt.Expr)
			if expr == "" {
				continue
			}
			if strings.Contains(expr, "fak_cli_") {
				sawCLI = true
				if !strings.Contains(expr, `job="fak_fleet"`) {
					t.Errorf("panel %d %q queries fak_cli_* without selecting job=%q (expr: %s)",
						p.ID, p.Title, "fak_fleet", expr)
				}
			}
			if strings.Contains(expr, "fak_tool_") {
				sawTool = true
				if !strings.Contains(expr, `job="fak_ops"`) {
					t.Errorf("panel %d %q queries fak_tool_* without selecting job=%q (expr: %s)",
						p.ID, p.Title, "fak_ops", expr)
				}
			}
			if strings.Contains(expr, `job="fak_fleet"`) {
				sawFleetJob = true
			}
			if strings.Contains(expr, `job="fak_ops"`) {
				sawOpsJob = true
			}
		}
	}
	if !sawCLI || !sawTool {
		t.Errorf("%s must query both fak_cli_* (runtime) and fak_tool_* (developer) families", fakCLITelemetryPath)
	}
	if !sawFleetJob || !sawOpsJob {
		t.Errorf("%s must select the fak_fleet and fak_ops scrape jobs", fakCLITelemetryPath)
	}
}

// TestFakCLITelemetryHonestyPanels pins the operator-facing honesty contract:
// the readability gauges and truncation gauge must be present and explained, so
// an operator can tell an empty journal (honest zero) from an unreadable one,
// and a truncated panel from a complete one.
func TestFakCLITelemetryHonestyPanels(t *testing.T) {
	d := loadFakCLITelemetry(t)

	descByTitle := map[string]string{}
	exprs := []string{}
	for _, p := range d.Panels {
		if p.Title != "" {
			descByTitle[p.Title] = p.Description
		}
		for _, tgt := range p.Targets {
			exprs = append(exprs, tgt.Expr)
		}
	}
	joined := strings.Join(exprs, "\n")

	for _, want := range []string{
		"fak_cli_usage_log_readable",
		"fak_cli_usage_series_truncated",
		"fak_tool_usage_log_readable",
		"fak_tool_usage_series_truncated",
	} {
		if !strings.Contains(joined, want) {
			t.Errorf("%s must expose the %s honesty gauge", fakCLITelemetryPath, want)
		}
	}

	readable, ok := descByTitle["Usage log readable"]
	if !ok {
		t.Fatalf("%s is missing the 'Usage log readable' panel", fakCLITelemetryPath)
	}
	if !strings.Contains(readable, "NOT") {
		t.Errorf("readability panel must explain that 0 means unavailable, NOT that no verb ran; got: %q", readable)
	}
	truncated, ok := descByTitle["Series truncated"]
	if !ok {
		t.Fatalf("%s is missing the 'Series truncated' panel", fakCLITelemetryPath)
	}
	if !strings.Contains(truncated, "incomplete") {
		t.Errorf("truncation panel must explain that non-zero means an incomplete panel; got: %q", truncated)
	}
}

package grafanacontract

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
)

// opsWorkersQueuePath is the operator dashboard fed by the fak_ops_workers scrape
// job. Its "companion" panels read the separate fak_ops job (the Fak Ops statuses
// exporter).
const opsWorkersQueuePath = "../../tools/grafana/dashboards/fak-ops-workers-queue.json"

// TestOpsWorkersQueueRoutinesEnabledTargetsEmittedSeries is the regression guard
// for the "Ops routines enabled reads 0" defect: the panel queried
// fak_ops_schedule_enabled{job="fak_ops"} behind an `or vector(0)` fallback, so a
// missing/ambiguous series collapsed to a confident wrong zero. The fak_ops job
// emits fak_ops_routines{state="enabled"} (the schedule_enabled name is ambiguous
// across scrape jobs), and the panel must query that, with no zero-fallback.
func TestOpsWorkersQueueRoutinesEnabledTargetsEmittedSeries(t *testing.T) {
	panels := readOpsWorkersQueuePanels(t)
	p, ok := panelByTitle(panels, "Ops routines enabled")
	if !ok {
		t.Fatalf("panel %q not found in %s", "Ops routines enabled", opsWorkersQueuePath)
	}
	if len(p.Targets) == 0 {
		t.Fatalf("panel %q has no targets", p.Title)
	}
	expr := p.Targets[0].Expr
	if !strings.Contains(expr, `fak_ops_routines{job="fak_ops",state="enabled"}`) {
		t.Errorf("panel %q must query the series the fak_ops job emits; got expr %q", p.Title, expr)
	}
	if strings.Contains(expr, "or vector(0)") {
		t.Errorf("panel %q must not use `or vector(0)` (a missing scrape must read Unavailable, not a fabricated 0); got expr %q", p.Title, expr)
	}
}

// TestOpsWorkersQueueCompanionPanelsHaveNoZeroFallback pins that no fak_ops
// companion panel launders a missing scrape into zero: each must be a bare
// selector/aggregation with a `noValue` mapping so the stat reads Unavailable.
func TestOpsWorkersQueueCompanionPanelsHaveNoZeroFallback(t *testing.T) {
	panels := readOpsWorkersQueuePanels(t)
	for _, title := range []string{
		"Ops routines enabled",
		"Ops routines stale",
		"Stuck work",
		"Reports failure rate (24h)",
	} {
		p, ok := panelByTitle(panels, title)
		if !ok {
			t.Errorf("companion panel %q not found", title)
			continue
		}
		for _, tgt := range p.Targets {
			if strings.Contains(tgt.Expr, "or vector(0)") {
				t.Errorf("panel %q target %q keeps a zero-fallback; a missing series must read Unavailable", title, tgt.Expr)
			}
		}
	}
}

// TestOpsWorkersQueueAtRiskTargetsEmittedSeries is the regression guard for the
// "Workers at risk reads no data" defect: the dashboard queried
// fak_ops_workers_at_risk{}, a series the fak_ops_workers exporter never emits
// (it emits fak_ops_at_risk_worktrees), so both the "Workers at risk" stat and
// panel 6's "at risk" legend line were permanently empty. Every reference must
// name the emitted series.
func TestOpsWorkersQueueAtRiskTargetsEmittedSeries(t *testing.T) {
	panels := readOpsWorkersQueueRawPanels(t)

	// The stat panel must query the emitted series directly.
	var riskPanelFound bool
	for _, p := range panels {
		if p.ID != 3 || p.Title != "Workers at risk" {
			continue
		}
		riskPanelFound = true
		if len(p.Targets) == 0 {
			t.Errorf("panel %d %q has no targets", p.ID, p.Title)
			continue
		}
		if expr := p.Targets[0].Expr; !strings.Contains(expr, `fak_ops_at_risk_worktrees{job="fak_ops_workers"}`) {
			t.Errorf("panel %d %q must query the series the fak_ops_workers exporter emits; got expr %q", p.ID, p.Title, expr)
		}
	}
	if !riskPanelFound {
		t.Errorf("panel id 3 titled %q not found in %s", "Workers at risk", opsWorkersQueuePath)
	}

	// The timeseries panel's "at risk" legend line must track the same series.
	var atRiskLegendFound bool
	for _, p := range panels {
		if p.ID != 6 || p.Title != "Workers live (over time)" {
			continue
		}
		for _, tgt := range p.Targets {
			if tgt.LegendFormat != "at risk" {
				continue
			}
			atRiskLegendFound = true
			if !strings.Contains(tgt.Expr, "fak_ops_at_risk_worktrees") {
				t.Errorf("panel %d %q legend %q must track fak_ops_at_risk_worktrees; got expr %q", p.ID, p.Title, tgt.LegendFormat, tgt.Expr)
			}
		}
	}
	if !atRiskLegendFound {
		t.Errorf("panel id 6 titled %q has no target with legendFormat %q", "Workers live (over time)", "at risk")
	}

	// No panel anywhere may reference the never-emitted series name.
	for _, p := range panels {
		text := p.Title + "\n" + p.Description
		for _, tgt := range p.Targets {
			text += "\n" + tgt.Expr
		}
		if strings.Contains(text, "fak_ops_workers_at_risk") {
			t.Errorf("panel %d %q still references the never-emitted series fak_ops_workers_at_risk", p.ID, p.Title)
		}
	}
}

// opsWorkersQueueRawPanel is a local view of the dashboard panels carrying the
// fields the shared Panel/Target contract types omit (panel id, target
// legendFormat). It is deliberately private to this test file so the shared
// contract structs stay minimal.
type opsWorkersQueueRawPanel struct {
	ID          int                     `json:"id"`
	Title       string                  `json:"title"`
	Description string                  `json:"description"`
	Targets     []opsWorkersQueueRawTgt `json:"targets"`
}

type opsWorkersQueueRawTgt struct {
	Expr         string `json:"expr"`
	LegendFormat string `json:"legendFormat"`
}

// readOpsWorkersQueueRawPanels decodes the dashboard with the fuller local view
// so a test can assert on panel id and target legendFormat.
func readOpsWorkersQueueRawPanels(t *testing.T) []opsWorkersQueueRawPanel {
	t.Helper()
	b, err := os.ReadFile(opsWorkersQueuePath)
	if err != nil {
		t.Fatalf("read %s: %v", opsWorkersQueuePath, err)
	}
	var d struct {
		Panels []opsWorkersQueueRawPanel `json:"panels"`
	}
	if err := json.Unmarshal(b, &d); err != nil {
		t.Fatalf("unmarshal %s: %v", opsWorkersQueuePath, err)
	}
	return d.Panels
}

func readOpsWorkersQueuePanels(t *testing.T) []Panel {
	t.Helper()
	b, err := os.ReadFile(opsWorkersQueuePath)
	if err != nil {
		t.Fatalf("read %s: %v", opsWorkersQueuePath, err)
	}
	var d Dashboard
	if err := json.Unmarshal(b, &d); err != nil {
		t.Fatalf("unmarshal %s: %v", opsWorkersQueuePath, err)
	}
	return d.Panels
}

func panelByTitle(panels []Panel, title string) (Panel, bool) {
	for _, p := range panels {
		if p.Title == title {
			return p, true
		}
	}
	return Panel{}, false
}

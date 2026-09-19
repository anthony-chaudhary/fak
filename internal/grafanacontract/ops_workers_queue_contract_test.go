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

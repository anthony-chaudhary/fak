package gateway

import (
	"strings"
	"testing"
)

func TestMetricsRendersBoundedTrajctlScoreboard(t *testing.T) {
	srv := newTestServer(t)
	srv.trajctlMetrics = func() TrajctlMetrics {
		return TrajctlMetrics{Objectives: map[string]int{"active": 2, "paused": 1}, Scores: map[string]float64{"root": .75, "child": .25}, Signals: map[string]int{"HEALTHY": 1, "STALL": 1, "DRIFT": 1}, Nudges: map[string]int{"delivered": 3, "failed": 1}}
	}
	text := srv.renderMetrics()
	for _, want := range []string{`fak_trajctl_objectives{status="active"} 2`, `fak_trajctl_score{objective_kind="root"} 0.75`, `fak_trajctl_signals{signal="DRIFT"} 1`, `fak_trajctl_nudges_total{outcome="delivered"} 3`} {
		if !strings.Contains(text, want) {
			t.Fatalf("missing %q", want)
		}
	}
	if strings.Contains(text, "objective_id=") || strings.Contains(text, "trace_id=") {
		t.Fatal("high-cardinality identifier leaked into trajctl metrics")
	}
	if got := strings.Count(text, "fak_trajctl_objectives{status="); got != 4 {
		t.Fatalf("status series=%d", got)
	}
	if got := strings.Count(text, "fak_trajctl_score{objective_kind="); got != 3 {
		t.Fatalf("score series=%d", got)
	}
}

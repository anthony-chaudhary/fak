package modelroute

import (
	"math"
	"testing"
)

// TestRouterAPGR verifies the APGR router evaluation ported from
// lm-sys/RouteLLM@0b64fda (routellm/evals/evaluate.py, apgr_metric +
// auc_metric, Apache-2.0): the trapezoid AUC over cost, the normalized
// APGR score, and the APGR-descending ranking.
//
// Hand-computed from the source formula with weak_accuracy = 0.5 and
// strong_accuracy = 0.9 (so strong_auc - weak_auc = 0.4 over a [0,1] cost
// span):
//
//	good: points (0,.50) (0.25,.70) (0.50,.80) (0.75,.85) (1,.90)
//	    AUC = .25*(.50+.70)/2 + .25*(.70+.80)/2 + .25*(.80+.85)/2 + .25*(.85+.90)/2
//	        = .15 + .1875 + .20625 + .21875 = 0.7625
//	    APGR = (0.7625 - 0.5) / 0.4 = 0.65625
//
//	poor: points (0,.50) (0.5,.55) (1,.90)
//	    AUC = .5*(.50+.55)/2 + .5*(.55+.90)/2 = .2625 + .3625 = 0.625
//	    APGR = (0.625 - 0.5) / 0.4 = 0.3125
//
//	flat: points (0,.50) (0.5,.50) (1,.50)
//	    AUC = 0.5, APGR = (0.5 - 0.5) / 0.4 = 0.0
//
// Expected ranking (APGR descending): good, poor, flat.
func TestRouterAPGR(t *testing.T) {
	const (
		weakAcc   = 0.5
		strongAcc = 0.9
	)

	routes := []RouteCurve{
		// Deliberately out of order: the source sorts by strong_percentage
		// before integrating, so input order must not matter.
		{Name: "good", Points: []RoutePoint{
			{Cost: 0.75, Accuracy: 0.85},
			{Cost: 0.0, Accuracy: 0.50},
			{Cost: 1.0, Accuracy: 0.90},
			{Cost: 0.25, Accuracy: 0.70},
			{Cost: 0.5, Accuracy: 0.80},
		}},
		{Name: "poor", Points: []RoutePoint{
			{Cost: 0.0, Accuracy: 0.50},
			{Cost: 0.5, Accuracy: 0.55},
			{Cost: 1.0, Accuracy: 0.90},
		}},
		{Name: "flat", Points: []RoutePoint{
			{Cost: 0.0, Accuracy: 0.50},
			{Cost: 0.5, Accuracy: 0.50},
			{Cost: 1.0, Accuracy: 0.50},
		}},
	}

	got := RankRoutesByAPGR(routes, weakAcc, strongAcc)
	if len(got) != 3 {
		t.Fatalf("RankRoutesByAPGR returned %d results, want 3", len(got))
	}

	// Ranking matches the source's APGR-descending order.
	wantOrder := []string{"good", "poor", "flat"}
	for i, name := range wantOrder {
		if got[i].Route != name {
			t.Fatalf("rank %d = %q (APGR %v), want %q", i, got[i].Route, got[i].APGR, name)
		}
	}

	// Hand-computed source outputs.
	want := map[string]struct{ auc, apgr float64 }{
		"good": {0.7625, 0.65625},
		"poor": {0.625, 0.3125},
		"flat": {0.5, 0.0},
	}
	for _, r := range got {
		w, ok := want[r.Route]
		if !ok {
			t.Fatalf("unexpected route %q in results", r.Route)
		}
		if math.Abs(r.AUC-w.auc) > 1e-9 {
			t.Errorf("route %q AUC = %v, want %v", r.Route, r.AUC, w.auc)
		}
		if math.Abs(r.APGR-w.apgr) > 1e-9 {
			t.Errorf("route %q APGR = %v, want %v", r.Route, r.APGR, w.apgr)
		}
	}

	// Degenerate denominator: weak == strong makes strong_auc - weak_auc = 0,
	// so APGR is NaN (mirrors the source's numpy division).
	degen := RankRoutesByAPGR([]RouteCurve{{Name: "x", Points: []RoutePoint{{0, 0.5}, {1, 0.9}}}}, 0.7, 0.7)
	if !math.IsNaN(degen[0].APGR) {
		t.Errorf("weak==strong APGR = %v, want NaN", degen[0].APGR)
	}

	// Single-point curve: np.trapz over one sample is 0, so AUC = 0 and the
	// zero/zero APGR is NaN.
	single := RankRoutesByAPGR([]RouteCurve{{Name: "s", Points: []RoutePoint{{0.5, 0.7}}}}, weakAcc, strongAcc)
	if single[0].AUC != 0 {
		t.Errorf("single-point AUC = %v, want 0", single[0].AUC)
	}
	if !math.IsNaN(single[0].APGR) {
		t.Errorf("single-point APGR = %v, want NaN", single[0].APGR)
	}
}

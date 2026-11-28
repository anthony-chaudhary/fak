// APGR (accuracy-per-gpu-resource) router evaluation: rank candidate routes
// by the area under their accuracy-vs-cost curve, normalized between the
// always-weak and always-strong baselines.
//
// NOTICE / attribution: ported from lm-sys/RouteLLM, routellm/evals/evaluate.py
// (functions auc_metric + apgr_metric, and the APGR-descending ranking in
package modelroute

import (
	"math"
	"sort"
)

// RoutePoint is one per-threshold sample of a candidate route: the fraction
// of calls sent to the strong model (the source's strong_percentage/100 — the
// cost axis, in [0,1]) and the routed accuracy at that operating point.
type RoutePoint struct {
	Cost     float64
	Accuracy float64
}

// RouteCurve is a candidate route's per-threshold samples, in any order.
// RankRoutesByAPGR sorts them by Cost before integrating, mirroring the
// source's sort_values(by=["strong_percentage"]).
type RouteCurve struct {
	Name   string
	Points []RoutePoint
}

// APGRResult is the computed APGR score for one route.
type APGRResult struct {
	Route string
	AUC   float64 // trapezoid integral of accuracy over cost
	APGR  float64 // (AUC - weak_auc) / (strong_auc - weak_auc)
}

// trapezoid is the trapezoidal rule over paired (x, y) samples — the Go
// mirror of np.trapz(y, x). Fewer than two samples integrate to 0, exactly
// as np.trapz does.
func trapezoid(x, y []float64) float64 {
	if len(x) < 2 || len(y) != len(x) {
		return 0
	}
	sum := 0.0
	for i := 1; i < len(x); i++ {
		sum += (x[i] - x[i-1]) * (y[i] + y[i-1]) / 2
	}
	return sum
}

// apgrScore computes the source's APGR for one route's curve:
//
//	AUC        = trapz(accuracy, cost)
//	weak_auc   = trapz(constant weak_accuracy, cost)
//	strong_auc = trapz(constant strong_accuracy, cost)
//	APGR       = (AUC - weak_auc) / (strong_auc - weak_auc)
//
// A zero denominator (weak == strong, or a degenerate single-point curve)
// yields NaN, mirroring the source's numpy division.
func apgrScore(points []RoutePoint, weakAccuracy, strongAccuracy float64) (auc, apgr float64) {
	costs := make([]float64, len(points))
	accs := make([]float64, len(points))
	for i, p := range points {
		costs[i] = p.Cost
		accs[i] = p.Accuracy
	}
	auc = trapezoid(costs, accs)

	weakAUC := trapezoid(costs, makeConst(len(points), weakAccuracy))
	strongAUC := trapezoid(costs, makeConst(len(points), strongAccuracy))
	apgr = (auc - weakAUC) / (strongAUC - weakAUC)
	return auc, apgr
}

func makeConst(n int, v float64) []float64 {
	out := make([]float64, n)
	for i := range out {
		out[i] = v
	}
	return out
}

// RankRoutesByAPGR computes the APGR score for each candidate route and
// returns the routes ranked by APGR descending — the source's
// sort_values(by=["APGR"], ascending=False). Ties keep input order (stable),
// and NaN scores (degenerate curves) rank last, matching pandas'
// na_position="last" default.
func RankRoutesByAPGR(routes []RouteCurve, weakAccuracy, strongAccuracy float64) []APGRResult {
	results := make([]APGRResult, 0, len(routes))
	for _, r := range routes {
		points := make([]RoutePoint, len(r.Points))
		copy(points, r.Points)
		sort.SliceStable(points, func(i, j int) bool { return points[i].Cost < points[j].Cost })
		auc, apgr := apgrScore(points, weakAccuracy, strongAccuracy)
		results = append(results, APGRResult{Route: r.Name, AUC: auc, APGR: apgr})
	}
	sort.SliceStable(results, func(i, j int) bool {
		a, b := results[i].APGR, results[j].APGR
		if math.IsNaN(a) && math.IsNaN(b) {
			return false
		}
		if math.IsNaN(a) {
			return false // NaN ranks last
		}
		if math.IsNaN(b) {
			return true
		}
		return a > b
	})
	return results
}

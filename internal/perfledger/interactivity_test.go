// fak-test:runtime fast est=10ms
package perfledger

// Adversarial spec tests for the E2E normalized interactivity port (direct port
// of NVIDIA aiperf e2e_normalized_interactivity_metrics.py +
// e2e_normalized_interactivity_analyzer.py). Every expected number is
// hand-computed in comments so a reviewer can verify each vector without running
// numpy. The formula is the slow-tail convention: p = percentile of the
// seconds-per-output-token ratio (request_latency_s / OSL), value = 1 / p.

import (
	"math"
	"testing"
)

// intvSameFloat matches a finite want with a 1e-9 epsilon and a NaN want with
// math.IsNaN.
func intvSameFloat(t *testing.T, label string, got, want float64) {
	t.Helper()
	if math.IsNaN(want) {
		if !math.IsNaN(got) {
			t.Errorf("%s = %v, want NaN", label, got)
		}
		return
	}
	if math.Abs(got-want) > 1e-9 {
		t.Errorf("%s = %v, want %v", label, got, want)
	}
}

// TestInteractivity verifies the normalized interactivity metric returns the
// correct normalized value for known inputs and handles zero/missing reference.
//
// Known inputs: four requests whose seconds-per-output-token ratios are
// [0.25, 0.5, 0.75, 1.0] (latency 2.5/5/7.5/10 s over 10 output tokens each).
// Sorted population n=4, linear-interpolation rank r = q*(n-1) = q*3:
//
//	P75 r=2.25 -> 0.75 + 0.25*(1.0-0.75) = 0.8125 -> 1/0.8125 = 16/13
//	P90 r=2.70 -> 0.75 + 0.70*(1.0-0.75) = 0.925  -> 1/0.925  = 40/37
//
// 16/13 = 1.2307692307692308, 40/37 = 1.0810810810810811. Count = 4.
//
// Zero/missing reference: a request with a zero or NaN latency or token count is
// excluded from the sample (upstream "all fields > 0" filter). Only the first
// sample survives, so the population is the single ratio 0.25 and every rank
// collapses to it: p75 = p90 = 0.25 -> 1/0.25 = 4. Count = 1. When no request
// passes the filter (or the input is empty/nil) there is nothing to report and
// the result is nil.
func TestInteractivity(t *testing.T) {
	// Known inputs -> exact normalized values.
	samples := []InteractivitySample{
		{LatencyNS: 2_500_000_000, OutputTokens: 10},  // 2.5s / 10 = 0.25 s/token
		{LatencyNS: 5_000_000_000, OutputTokens: 10},  // 5s   / 10 = 0.5
		{LatencyNS: 7_500_000_000, OutputTokens: 10},  // 7.5s / 10 = 0.75
		{LatencyNS: 10_000_000_000, OutputTokens: 10}, // 10s  / 10 = 1.0
	}
	got := ComputeInteractivity(samples)
	if got == nil {
		t.Fatalf("known inputs: got nil, want a non-nil Interactivity")
	}
	if got.Count != 4 {
		t.Errorf("Count = %d, want 4", got.Count)
	}
	intvSameFloat(t, "P75", got.P75, 16.0/13.0)
	intvSameFloat(t, "P90", got.P90, 40.0/37.0)

	// Zero/missing reference: zero or NaN latency/tokens are excluded.
	mixed := []InteractivitySample{
		{LatencyNS: 2_500_000_000, OutputTokens: 10},         // valid: 0.25 s/token
		{LatencyNS: 0, OutputTokens: 10},                     // zero latency -> excluded
		{LatencyNS: 5_000_000_000, OutputTokens: 0},          // zero tokens -> excluded
		{LatencyNS: math.NaN(), OutputTokens: 10},            // NaN latency -> excluded
		{LatencyNS: 5_000_000_000, OutputTokens: math.NaN()}, // NaN tokens -> excluded
	}
	gotMixed := ComputeInteractivity(mixed)
	if gotMixed == nil {
		t.Fatalf("zero/missing: got nil, want a non-nil Interactivity")
	}
	if gotMixed.Count != 1 {
		t.Errorf("zero/missing Count = %d, want 1", gotMixed.Count)
	}
	intvSameFloat(t, "zero/missing P75", gotMixed.P75, 4.0)
	intvSameFloat(t, "zero/missing P90", gotMixed.P90, 4.0)

	// No request passes the filter -> nothing to report -> nil.
	allBad := []InteractivitySample{
		{LatencyNS: 0, OutputTokens: 10},
		{LatencyNS: 5_000_000_000, OutputTokens: 0},
		{LatencyNS: math.NaN(), OutputTokens: 10},
	}
	if g := ComputeInteractivity(allBad); g != nil {
		t.Errorf("all-bad: got %+v, want nil", *g)
	}
	if g := ComputeInteractivity(nil); g != nil {
		t.Errorf("nil input: got %+v, want nil", *g)
	}
	if g := ComputeInteractivity([]InteractivitySample{}); g != nil {
		t.Errorf("empty input: got %+v, want nil", *g)
	}
}

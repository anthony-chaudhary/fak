package dispatchtick

import (
	"strings"
	"testing"
)

// preflight_churn_rate_test.go — the admission side of the same calibration pin as
// internal/stallscan/spawn_rate_test.go.
//
// The host_churn term reads the stallscan spawn signal, so it inherits the same
// trap: a burst COUNT is only interpretable against the window it was counted
// over. Live capture on the reference box (2026-08-05, 101 one-second ticks under
// ordinary fleet load) measured median 22 gross births/sec, p95 63, max 83 — all
// of which clear the legacy count threshold of 8. Feeding gross births into the
// count path would refuse dispatch on ~95% of ticks of a healthy box.
//
// A gate that always refuses is the same defect as one that never fires, pointed
// the other way. These cases keep it pointed at neither.

// okPreflight is a SPAWN_OK preflight with headroom, so the churn fold is the only
// term that can bind.
func okPreflight() PreflightResult {
	return PreflightResult{OK: true, Live: 3, Cap: 8, Headroom: 5}
}

func TestChurnRateAbstainsOnMeasuredWorkingBox(t *testing.T) {
	for _, births := range []int{22, 63, 83} { // median, p95, observed max
		res := ApplyChurnBackpressure(okPreflight(), ChurnCheck{
			Recent:        births,
			WindowSeconds: 1,
		})
		if !res.OK {
			t.Errorf("%d births/sec refused with %s — that rate was measured on a WORKING host, "+
				"so this gate would stop dispatch during ordinary fleet load", births, res.Verdict)
		}
		if res.Cap != 8 {
			t.Errorf("%d births/sec lowered the cap to %d; the term must abstain below its threshold",
				births, res.Cap)
		}
	}
}

func TestChurnRateBindsAboveThreshold(t *testing.T) {
	res := ApplyChurnBackpressure(okPreflight(), ChurnCheck{
		Recent:        int(DefaultChurnBurstRate) + 50,
		WindowSeconds: 1,
	})
	if res.OK {
		t.Fatalf("a burst of %.0f/sec (>= %.0f/sec) did not bind; verdict=%s cap=%d",
			DefaultChurnBurstRate+50, DefaultChurnBurstRate, res.Verdict, res.Cap)
	}
	if res.Verdict != PreflightRefuseChurn {
		t.Fatalf("verdict = %s, want %s", res.Verdict, PreflightRefuseChurn)
	}
	if !strings.Contains(res.Reason, HostChurnBackoff) {
		t.Fatalf("reason missing the closed token %s: %s", HostChurnBackoff, res.Reason)
	}
	// The reason must cite the RATE and the raw count+window it came from, so the
	// derived number stays auditable rather than taken on trust.
	if !strings.Contains(res.Reason, "/sec") {
		t.Fatalf("reason does not cite a rate: %s", res.Reason)
	}
}

// TestChurnRateWindowChangesTheVerdict is the conflation guard: the SAME count is a
// refusal over a short window and an abstention over a long one. If these two ever
// agree, the window stopped being read and the gate is back to comparing
// incomparable numbers.
func TestChurnRateWindowChangesTheVerdict(t *testing.T) {
	const burst = 200
	short := ApplyChurnBackpressure(okPreflight(), ChurnCheck{Recent: burst, WindowSeconds: 1})
	long := ApplyChurnBackpressure(okPreflight(), ChurnCheck{Recent: burst, WindowSeconds: 30})
	if short.OK {
		t.Fatalf("%d births in 1s (=%d/sec) must bind", burst, burst)
	}
	if !long.OK {
		t.Fatalf("%d births in 30s (=%.1f/sec) must abstain; got %s",
			burst, float64(burst)/30, long.Verdict)
	}
}

// TestChurnCountPathUnchangedWithoutWindow pins backward compatibility: a caller
// that carries no window keeps the exact legacy count comparison, so wiring the
// window in cannot silently change any existing caller's behaviour.
func TestChurnCountPathUnchangedWithoutWindow(t *testing.T) {
	below := ApplyChurnBackpressure(okPreflight(), ChurnCheck{Recent: DefaultChurnBurstThreshold - 1})
	if !below.OK {
		t.Fatalf("count %d (< %d) must abstain on the window-less path",
			DefaultChurnBurstThreshold-1, DefaultChurnBurstThreshold)
	}
	at := ApplyChurnBackpressure(okPreflight(), ChurnCheck{Recent: DefaultChurnBurstThreshold})
	if at.OK {
		t.Fatalf("count %d (>= %d) must bind on the window-less path",
			DefaultChurnBurstThreshold, DefaultChurnBurstThreshold)
	}
}

// TestChurnRateThresholdClearsMeasuredMax keeps the two gates' calibrations from
// drifting apart, and keeps this one above the measured working-box maximum.
func TestChurnRateThresholdClearsMeasuredMax(t *testing.T) {
	const measuredMax = 83.0 // 2026-08-05 capture, 101 ticks
	if DefaultChurnBurstRate <= measuredMax {
		t.Fatalf("DefaultChurnBurstRate = %.0f/sec is at or below the measured working-box max of %.0f/sec",
			DefaultChurnBurstRate, measuredMax)
	}
}

// fak-test:runtime fast est=10ms lane=default
// fak-test:justify why=contract when=changed:internal/perfledger/**
package perfledger

import (
	"math"
	"testing"
)

func TestQueueDepthTokenPressure(t *testing.T) {
	cases := []struct {
		name         string
		conc         int
		isl, osl     int
		wantPressure float64
		wantDepth    int
	}{
		{"light_requests_hold_base_depth", 25, 512, 128, 1, 25},
		// The threshold itself is not pressure: the model switches strictly above it.
		{"exactly_at_threshold", 25, 8000, 2000, 1, 25},
		{"one_token_over", 25, 8000, 2001, 1.0001, 25},
		{"double_the_threshold", 25, 15000, 5000, 2, 50},
		// 7 * 1.5 = 10.5 records; a fractional record is not queued.
		{"truncates_to_whole_records", 7, 10000, 5000, 1.5, 10},
		{"cap_reached_exactly", 15, 60000, 40000, 10, 150},
		// The calibration point: 173K tokens saturates at ten times the base.
		{"past_the_cap", 15, 150000, 23000, 10, 150},
		{"output_tokens_alone_count", 4, 0, 30000, 3, 12},
		{"zero_concurrency", 0, 90000, 90000, 10, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := QueueTokenPressure(tc.isl, tc.osl); math.Abs(got-tc.wantPressure) > 1e-12 {
				t.Fatalf("QueueTokenPressure(%d, %d) = %v, want %v", tc.isl, tc.osl, got, tc.wantPressure)
			}
			if got := QueueDepth(tc.conc, tc.isl, tc.osl); got != tc.wantDepth {
				t.Fatalf("QueueDepth(%d, %d, %d) = %d, want %d", tc.conc, tc.isl, tc.osl, got, tc.wantDepth)
			}
		})
	}
}

func TestQueueDepthHardenedInputs(t *testing.T) {
	if got := QueueDepth(-5, 20000, 0); got != 0 {
		t.Fatalf("negative concurrency gave depth %d, want 0", got)
	}
	// A negative count must not cancel real tokens on the other side.
	if got := QueueDepth(10, -50000, 30000); got != 30 {
		t.Fatalf("negative input tokens gave depth %d, want 30", got)
	}
	if got := QueueTokenPressure(-50000, 30000); got != 3 {
		t.Fatalf("negative input tokens gave pressure %v, want 3", got)
	}
	// Two counts whose plain sum wraps negative still saturate at the cap.
	if got := QueueDepth(10, math.MaxInt, math.MaxInt); got != 100 {
		t.Fatalf("huge token counts gave depth %d, want the capped 100", got)
	}
	if got := QueueDepth(math.MaxInt, 500000, 0); got != 10<<52 {
		t.Fatalf("huge concurrency under full pressure gave depth %d, want ten times the 2^52 ceiling", got)
	}
	if got := QueueTokenPressure(math.MaxInt, math.MaxInt); got != QueueTokenPressureCap {
		t.Fatalf("huge token counts gave pressure %v, want the cap", got)
	}
}

func TestQueueDepthAgreesWithMemoryModel(t *testing.T) {
	for _, conc := range []int{0, 1, 7, 25, 400} {
		for _, isl := range []int{0, 512, 9999, 10000, 10001, 64000, 99999, 100000, 250000} {
			for _, osl := range []int{0, 1, 128, 5000, 100000} {
				want := rpQueueDepth(conc, isl, osl)
				if got := QueueDepth(conc, isl, osl); got != want {
					t.Fatalf("QueueDepth(%d, %d, %d) = %d, memory model uses %d", conc, isl, osl, got, want)
				}
				if byPressure := int(float64(conc) * QueueTokenPressure(isl, osl)); byPressure != want {
					t.Fatalf("pressure %v for (%d, %d) implies depth %d at concurrency %d, memory model uses %d",
						QueueTokenPressure(isl, osl), isl, osl, byPressure, conc, want)
				}
			}
		}
	}
}

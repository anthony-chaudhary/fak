package agent

import (
	"math"
	"testing"
)

// TestNativeDecodeRate pins the decode-throughput validity contract for the
// native in-kernel witness line (#13298). A generate call always produces its
// first token from the prefill block, so that token is NOT a sustained decode
// interval: the rate must be computed over the completed post-prefill intervals
// (gen-1) and reported as unavailable — never a finite success rate — when no
// such interval exists (gen < 2) or no elapsed decode time was measured.
func TestNativeDecodeRate(t *testing.T) {
	tests := []struct {
		name    string
		gen     int
		decodeS float64
		wantTPS float64
		wantOK  bool
	}{
		{
			// The fed6a6a37 defect: one output token, a near-zero decode window,
			// reported as 6957.7 tok/s. The single token is prefill-produced, so
			// there is no completed decode interval and no rate to report.
			name:    "prefill-produced single token is not a decode interval",
			gen:     1,
			decodeS: 0.0001437,
			wantTPS: 0,
			wantOK:  false,
		},
		{
			name:    "single token with zero elapsed time",
			gen:     1,
			decodeS: 0,
			wantTPS: 0,
			wantOK:  false,
		},
		{
			// Two tokens => exactly one completed post-prefill interval.
			name:    "two tokens is one timed interval",
			gen:     2,
			decodeS: 0.5,
			wantTPS: 2.0,
			wantOK:  true,
		},
		{
			// Ordinary multi-token decode: 128 generated tokens are 127 intervals.
			name:    "ordinary multi-token decode counts gen minus one intervals",
			gen:     128,
			decodeS: 25.6,
			wantTPS: 4.9609375,
			wantOK:  true,
		},
		{
			// A positive token count cannot resurrect a zero elapsed window.
			name:    "positive elapsed required even for many tokens",
			gen:     128,
			decodeS: 0,
			wantTPS: 0,
			wantOK:  false,
		},
		{
			name:    "negative elapsed is invalid",
			gen:     8,
			decodeS: -1,
			wantTPS: 0,
			wantOK:  false,
		},
		{
			name:    "no generated tokens",
			gen:     0,
			decodeS: 5,
			wantTPS: 0,
			wantOK:  false,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			gotTPS, gotOK := nativeDecodeRate(tc.gen, tc.decodeS)
			if gotOK != tc.wantOK {
				t.Fatalf("nativeDecodeRate(%d, %g) ok = %v, want %v", tc.gen, tc.decodeS, gotOK, tc.wantOK)
			}
			if !tc.wantOK {
				// An unavailable rate must never be a finite success value.
				if math.IsInf(gotTPS, 0) || math.IsNaN(gotTPS) {
					t.Fatalf("nativeDecodeRate(%d, %g) = %g, want non-finite-free unavailable value", tc.gen, tc.decodeS, gotTPS)
				}
				if gotTPS != 0 {
					t.Fatalf("nativeDecodeRate(%d, %g) = %g, want 0 when unavailable", tc.gen, tc.decodeS, gotTPS)
				}
				return
			}
			if gotTPS != tc.wantTPS {
				t.Fatalf("nativeDecodeRate(%d, %g) = %g, want %g", tc.gen, tc.decodeS, gotTPS, tc.wantTPS)
			}
		})
	}
}

// TestNativeDecodeRateInvalidIsSerializableSafe is the negative control: the
// unavailable verdict must never let an invalid rate be serialized as a finite
// success rate. The historical gen/decodeS form would have produced a huge
// finite number for the one-token warmup; the corrected helper must not.
func TestNativeDecodeRateInvalidIsSerializableSafe(t *testing.T) {
	legacyTPS := float64(1) / 0.0001437 // what the old gen/decodeS line reported
	if !math.IsInf(legacyTPS, 0) && legacyTPS < 1000 {
		t.Fatalf("fixture precondition: legacy rate %g should be the absurd single-token number", legacyTPS)
	}
	gotTPS, ok := nativeDecodeRate(1, 0.0001437)
	if ok {
		t.Fatalf("single prefill-produced token must be unavailable, got ok=true rate=%g", gotTPS)
	}
	if gotTPS >= 1000 {
		t.Fatalf("unavailable rate leaked a finite success value: %g", gotTPS)
	}
}

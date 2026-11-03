package regime

import (
	"strings"
	"testing"
)

// PHL-127-shaped profile: strix1 Qwen3.8 decode, GPU busy 71.9 of 75.2
// ms/token. The single-reading prose verdict ("GPU-bound") must come out as a
// reproducible COMPUTE_BOUND from the numbers alone.
func TestClassifyPHL127ComputeBound(t *testing.T) {
	t.Parallel()
	p, err := DecodeProfile([]byte(`{
		"schema": "fak.decode-step-profile/v1",
		"model": "qwen3.8",
		"device": "strix1",
		"device_busy_ms_per_token": 71.9,
		"total_ms_per_token": 75.2
	}`))
	if err != nil {
		t.Fatalf("DecodeProfile: %v", err)
	}
	v := Classify(p)
	if v.Regime != RegimeComputeBound {
		t.Fatalf("regime = %s, want COMPUTE_BOUND; evidence=%v missing=%v", v.Regime, v.Evidence, v.Missing)
	}
	wantFrac := 71.9 / 75.2 // 0.956...
	if !strings.Contains(v.Evidence[0], "busy_fraction=0.956") {
		t.Errorf("evidence %q does not name the computed fraction %.3f", v.Evidence[0], wantFrac)
	}
}

// Synthetic launch-bound profile: many tiny kernels with large inter-kernel
// gaps (per-layer host setup starves the launch queue).
func TestClassifyLaunchBoundGaps(t *testing.T) {
	t.Parallel()
	p, err := DecodeProfile([]byte(`{
		"kernels_per_step": 412,
		"kernel_execution_sum_us": 20.0,
		"inter_kernel_gap_sum_us": 180.0,
		"total_ms_per_token": 0.2
	}`))
	if err != nil {
		t.Fatalf("DecodeProfile: %v", err)
	}
	v := Classify(p)
	if v.Regime != RegimeLaunchBound {
		t.Fatalf("regime = %s, want LAUNCH_BOUND; evidence=%v", v.Regime, v.Evidence)
	}
	if !strings.Contains(v.Evidence[0], "gap_share=0.900") {
		t.Errorf("evidence %q does not name gap_share=0.900", v.Evidence[0])
	}
}

// Sparse data must classify UNKNOWN without failing: no signal, no guess.
func TestClassifySparseUnknown(t *testing.T) {
	t.Parallel()
	for _, doc := range []string{`{}`, `{"schema":"fak.decode-step-profile/v1","model":"qwen3.8"}`} {
		p, err := DecodeProfile([]byte(doc))
		if err != nil {
			t.Fatalf("DecodeProfile(%s): %v", doc, err)
		}
		v := Classify(p)
		if v.Regime != RegimeUnknown {
			t.Fatalf("sparse doc %s -> %s, want UNKNOWN; evidence=%v", doc, v.Regime, v.Evidence)
		}
		if len(v.Missing) == 0 {
			t.Errorf("sparse doc %s: expected missing-signal names, got none", doc)
		}
	}
}

// A v2 native-performance receipt carries end-to-end timing but no kernel or
// busy split: it must map what exists and stay UNKNOWN for the rest.
func TestClassifyV2ReceiptWithoutBusySignals(t *testing.T) {
	t.Parallel()
	doc := `{
		"schema": "fak-native-performance-receipt/v2",
		"role": "candidate",
		"controls": {"batch": 1, "decode_tokens": 32},
		"repetitions": [
			{"end_to_end_milliseconds": 100, "tokens_per_second": 10},
			{"end_to_end_milliseconds": 98, "tokens_per_second": 10.2}
		]
	}`
	p, err := DecodeProfile([]byte(doc))
	if err != nil {
		t.Fatalf("DecodeProfile: %v", err)
	}
	if p.TotalUsPerStep > 0 {
		t.Errorf("TotalUsPerStep = %.1f, want absent (no decode_milliseconds reported)", p.TotalUsPerStep)
	}
	v := Classify(p)
	if v.Regime != RegimeUnknown {
		t.Fatalf("regime = %s, want UNKNOWN; evidence=%v", v.Regime, v.Evidence)
	}
}

// A v2 receipt that also reports per-repetition decode time plus a device-busy
// reading classifies from the mapped total + busy pair.
func TestClassifyV2ReceiptWithDecodeMilliseconds(t *testing.T) {
	t.Parallel()
	doc := `{
		"schema": "fak-native-performance-receipt/v2",
		"role": "candidate",
		"repetitions": [
			{"decode_milliseconds": 12.5},
			{"decode_milliseconds": 13.5}
		],
		"device_busy_ms_per_token": 12.0
	}`
	p, err := DecodeProfile([]byte(doc))
	if err != nil {
		t.Fatalf("DecodeProfile: %v", err)
	}
	if p.TotalUsPerStep != 13000 { // mean(12.5, 13.5) ms in us
		t.Fatalf("TotalUsPerStep = %.1f, want 13000", p.TotalUsPerStep)
	}
	v := Classify(p)
	if v.Regime != RegimeComputeBound {
		t.Fatalf("regime = %s, want COMPUTE_BOUND (12/13 = 0.923); evidence=%v", v.Regime, v.Evidence)
	}
}

// Memory-bandwidth saturation from traffic counters: 4.2 GB moved in a 10 ms
// step on a 500 GB/s device is 0.84 utilization -> MEM_BOUND.
func TestClassifyMemBoundFromTraffic(t *testing.T) {
	t.Parallel()
	p, err := DecodeProfile([]byte(`{
		"mem_traffic_bytes_per_step": 4200000000,
		"device_peak_bandwidth_gbps": 500,
		"total_ms_per_token": 0.010
	}`))
	if err != nil {
		t.Fatalf("DecodeProfile: %v", err)
	}
	v := Classify(p)
	if v.Regime != RegimeMemBound {
		t.Fatalf("regime = %s, want MEM_BOUND; evidence=%v missing=%v", v.Regime, v.Evidence, v.Missing)
	}
}

// Threshold boundaries are inclusive on the decisive side only: exactly at
// LaunchGapShareMin the step is launch-bound; a busy fraction inside the mixed
// band stays UNKNOWN rather than forcing either label.
func TestClassifyThresholdBoundaries(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name string
		p    Profile
		want Regime
	}{
		{"gap share exactly 0.5", Profile{GapSumUs: 50, ExecSumUs: 50}, RegimeLaunchBound},
		{"busy fraction exactly 0.8", NewProfileAt(0.8), RegimeComputeBound},
		{"busy fraction 0.49 idle-dominated", NewProfileAt(0.49), RegimeLaunchBound},
		{"busy fraction 0.5 mixed band", NewProfileAt(0.5), RegimeUnknown},
		{"busy fraction 0.79 mixed band", NewProfileAt(0.79), RegimeUnknown},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			if got := Classify(tc.p).Regime; got != tc.want {
				t.Fatalf("Classify = %s, want %s", got, tc.want)
			}
		})
	}
}

// NewProfileAt builds an all-absent profile with one direct busy fraction.
func NewProfileAt(frac float64) Profile {
	p := NewProfile()
	p.BusyFraction = frac
	return p
}

// Malformed JSON is a hard error, distinct from sparse-but-valid data.
func TestDecodeProfileMalformedJSON(t *testing.T) {
	t.Parallel()
	if _, err := DecodeProfile([]byte(`{"device_busy_ms_per_token": 71.9`)); err == nil {
		t.Fatal("expected error for truncated JSON, got nil")
	}
}

// Out-of-range fractions are treated as absent, not clamped into a verdict.
func TestDecodeProfileRejectsOutOfRangeFractions(t *testing.T) {
	t.Parallel()
	p, err := DecodeProfile([]byte(`{"device_busy_fraction": 1.7, "total_ms_per_token": 10}`))
	if err != nil {
		t.Fatalf("DecodeProfile: %v", err)
	}
	if p.BusyFraction >= 0 {
		t.Fatalf("BusyFraction = %v, want absent sentinel for out-of-range input", p.BusyFraction)
	}
	if v := Classify(p); v.Regime != RegimeUnknown {
		t.Fatalf("regime = %s, want UNKNOWN when the only fraction is invalid", v.Regime)
	}
}

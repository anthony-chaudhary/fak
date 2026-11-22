package perfledger

import (
	"encoding/json"
	"math"
	"reflect"
	"testing"
)

func TestRoundSummaryValue(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name    string
		in      float64
		ndigits int
		want    float64
	}{
		// Golden values from CPython round(), which the upstream relies on.
		{"python 2.675 binary below half", 2.675, 2, 2.67},
		{"python half to even down", 0.125, 2, 0.12},
		{"python half to even up", 0.375, 2, 0.38},
		{"python 2.5 to even", 2.5, 0, 2},
		{"python 3.5 to even", 3.5, 0, 4},
		{"upstream error rate kept", 0.004, 9, 0.004},
		{"upstream error rate erased at 2", 0.004, 2, 0},
		{"upstream 1/21 at 9", 1.0 / 21.0, 9, 0.047619048},
		{"upstream 1/21 inflates at 2", 1.0 / 21.0, 2, 0.05},
		{"human scale", 568.8888, 2, 568.89},
		{"negative", -11.446, 2, -11.45},
		{"integer valued", 42, 2, 42},
		{"negative ndigits", 1234.5, -2, 1200},
		{"negative ndigits tie to even", 250, -2, 200},
		{"huge negative ndigits", 1234.5, -400, 0},
		{"max float", math.MaxFloat64, 2, math.MaxFloat64},
		{"pos inf passthrough", math.Inf(1), 2, math.Inf(1)},
		{"neg inf passthrough", math.Inf(-1), 9, math.Inf(-1)},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			if got := RoundSummaryValue(tc.in, tc.ndigits); got != tc.want {
				t.Fatalf("RoundSummaryValue(%v, %d) = %v, want %v", tc.in, tc.ndigits, got, tc.want)
			}
		})
	}
}

func TestRoundSummaryValueNaNAndNegativeZero(t *testing.T) {
	t.Parallel()
	if got := RoundSummaryValue(math.NaN(), 2); !math.IsNaN(got) {
		t.Fatalf("NaN must pass through, got %v", got)
	}
	if got := RoundSummaryValue(-0.001, 2); got != 0 || !math.Signbit(got) {
		t.Fatalf("-0.001 at 2 digits = %v, want -0 like Python", got)
	}
}

func TestRoundSummaryMetricAndFloats(t *testing.T) {
	t.Parallel()
	keys := DefaultSummaryFractionKeys()
	cases := []struct {
		key  string
		in   float64
		want float64
	}{
		{"error_rate", 0.0047619047619, 0.004761905},
		{"total_requests", 210, 210},
		{"ttft_p50_ms", 400.4567, 400.46},
		{"cache_hit_share", 0.11111, 0.11},
	}
	in := map[string]float64{}
	want := map[string]float64{}
	for _, tc := range cases {
		if got := RoundSummaryMetric(tc.key, tc.in, keys); got != tc.want {
			t.Errorf("RoundSummaryMetric(%q, %v) = %v, want %v", tc.key, tc.in, got, tc.want)
		}
		in[tc.key] = tc.in
		want[tc.key] = tc.want
	}
	got := RoundSummaryFloats(in, keys)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("RoundSummaryFloats = %v, want %v", got, want)
	}
	if in["ttft_p50_ms"] != 400.4567 {
		t.Fatal("RoundSummaryFloats mutated its input")
	}
	if RoundSummaryFloats(nil, keys) != nil {
		t.Fatal("nil map must stay nil")
	}
	if got := RoundSummaryMetric("error_rate", 0.123456789123, nil); got != 0.12 {
		t.Fatalf("nil fraction set must fall back to display digits, got %v", got)
	}
}

func TestRoundSummaryTree(t *testing.T) {
	t.Parallel()
	raw := `{
		"request_latency": {"avg": 123.4567, "p99": 999.999, "count": 7, "unit": "ms"},
		"error_rate": 0.0047619047619,
		"nested": {"error_rate": [0.123456789123, 0.5], "list": [1.005, 2.675, "x", null, true]},
		"total_requests": 210
	}`
	var in map[string]any
	if err := json.Unmarshal([]byte(raw), &in); err != nil {
		t.Fatal(err)
	}
	got := RoundSummaryTree(in, DefaultSummaryFractionKeys())
	want := map[string]any{
		"request_latency": map[string]any{"avg": 123.46, "p99": 1000.0, "count": 7.0, "unit": "ms"},
		"error_rate":      0.004761905,
		"nested": map[string]any{
			"error_rate": []any{0.123456789, 0.5},
			"list":       []any{1.0, 2.67, "x", nil, true},
		},
		"total_requests": 210.0,
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("RoundSummaryTree =\n%#v\nwant\n%#v", got, want)
	}
	if in["error_rate"] != 0.0047619047619 {
		t.Fatal("RoundSummaryTree mutated its input")
	}
}

func TestRoundSummaryTreeScalarsAndEdges(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name string
		in   any
		want any
	}{
		{"nil", nil, nil},
		{"int untouched", 7, 7},
		{"string untouched", "1.234", "1.234"},
		{"float32", float32(1.2345), float32(1.23)},
		{"bare float", 3.14159, 3.14},
		{"nil map", map[string]any(nil), map[string]any(nil)},
		{"nil slice", []any(nil), []any(nil)},
		{"inf leaf", map[string]any{"x": math.Inf(1)}, map[string]any{"x": math.Inf(1)}},
	}
	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := RoundSummaryTree(tc.in, DefaultSummaryFractionKeys())
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("RoundSummaryTree(%#v) = %#v, want %#v", tc.in, got, tc.want)
			}
		})
	}
}

func TestRoundSummaryDefaultKeysAreFresh(t *testing.T) {
	t.Parallel()
	a := DefaultSummaryFractionKeys()
	a["cache_hit_share"] = true
	if DefaultSummaryFractionKeys()["cache_hit_share"] {
		t.Fatal("DefaultSummaryFractionKeys shares state across calls")
	}
}

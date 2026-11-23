// fak-test:runtime fast est=10ms lane=default
// fak-test:justify why=contract when=changed:internal/perfledger/**
package perfledger

import (
	"math"
	"reflect"
	"strings"
	"testing"
)

// Latency is recorded in nanoseconds and stated in milliseconds; throughput
// shares one unit and is the larger-is-better metric.
var goodputTestMetrics = map[string]GoodputMetric{
	"request_latency":         {DisplayToInternal: 1e6},
	"time_to_first_token":     {DisplayToInternal: 1e6},
	"output_token_throughput": {LargerIsBetter: true},
}

func TestGoodputNormalizesDisplayUnits(t *testing.T) {
	slos, err := NewGoodputSLOs(map[string]float64{
		"request_latency":         250,
		"output_token_throughput": 40,
	}, goodputTestMetrics)
	if err != nil {
		t.Fatalf("NewGoodputSLOs: %v", err)
	}
	// Map order is random per call; a missing sort shows up within a few.
	for i := 0; i < 32; i++ {
		ordered, err := NewGoodputSLOs(map[string]float64{
			"time_to_first_token": 1, "request_latency": 1, "output_token_throughput": 1,
		}, goodputTestMetrics)
		if err != nil {
			t.Fatalf("NewGoodputSLOs: %v", err)
		}
		want := []string{"output_token_throughput", "request_latency", "time_to_first_token"}
		if got := ordered.RequiredTags(); !reflect.DeepEqual(got, want) {
			t.Fatalf("RequiredTags = %v, want sorted %v", got, want)
		}
	}
	if got, ok := slos.Threshold("request_latency"); !ok || got != 250e6 {
		t.Fatalf("request_latency threshold = %v, %v; want 2.5e8 ns", got, ok)
	}
	if got, ok := slos.Threshold("output_token_throughput"); !ok || got != 40 {
		t.Fatalf("throughput threshold = %v, %v; want 40 unconverted", got, ok)
	}
	if _, ok := slos.Threshold("time_to_first_token"); ok {
		t.Fatal("unconfigured metric reported a threshold")
	}
	want := []string{"output_token_throughput", "request_latency"}
	if got := slos.RequiredTags(); !reflect.DeepEqual(got, want) {
		t.Fatalf("RequiredTags = %v, want %v", got, want)
	}
}

func TestGoodputDirectionality(t *testing.T) {
	slos, err := NewGoodputSLOs(map[string]float64{
		"request_latency":         250,
		"output_token_throughput": 40,
	}, goodputTestMetrics)
	if err != nil {
		t.Fatalf("NewGoodputSLOs: %v", err)
	}
	cases := []struct {
		name       string
		latencyNS  float64
		throughput float64
		want       bool
	}{
		{"both_inside", 200e6, 55, true},
		{"both_exactly_at_threshold", 250e6, 40, true},
		{"latency_over", 250e6 + 1, 55, false},
		{"throughput_under", 200e6, 39.9, false},
		// A larger latency must fail and a larger throughput must pass; a
		// single shared comparison would get one of these two wrong.
		{"high_throughput_passes", 1, 1e9, true},
		{"low_latency_low_throughput_fails", 1, 1, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			record := map[string]float64{
				"request_latency":         tc.latencyNS,
				"output_token_throughput": tc.throughput,
			}
			if got := slos.IsGood(record); got != tc.want {
				t.Fatalf("IsGood(%v) = %v, want %v", record, got, tc.want)
			}
		})
	}
}

func TestGoodputMissingValueIsNotGood(t *testing.T) {
	slos, err := NewGoodputSLOs(map[string]float64{
		"request_latency":     250,
		"time_to_first_token": 80,
	}, goodputTestMetrics)
	if err != nil {
		t.Fatalf("NewGoodputSLOs: %v", err)
	}
	if slos.IsGood(map[string]float64{"request_latency": 100e6}) {
		t.Fatal("record without time_to_first_token counted as good")
	}
	if slos.IsGood(map[string]float64{"request_latency": 100e6, "time_to_first_token": math.NaN()}) {
		t.Fatal("record with NaN time_to_first_token counted as good")
	}
	if !slos.IsGood(map[string]float64{"request_latency": 100e6, "time_to_first_token": 50e6, "unrelated": -1}) {
		t.Fatal("complete record inside both SLOs was not good")
	}
}

func TestGoodputRefusals(t *testing.T) {
	cases := []struct {
		name    string
		slos    map[string]float64
		metrics map[string]GoodputMetric
		wantErr string
	}{
		{
			name:    "unknown_tags_listed_sorted",
			slos:    map[string]float64{"zzz": 1, "request_latency": 1, "aaa": 1},
			metrics: goodputTestMetrics,
			wantErr: "unknown metric tag(s) in goodput SLOs: aaa, zzz",
		},
		{
			name:    "nan_threshold",
			slos:    map[string]float64{"request_latency": math.NaN()},
			metrics: goodputTestMetrics,
			wantErr: "not finite",
		},
		{
			name:    "infinite_threshold",
			slos:    map[string]float64{"request_latency": math.Inf(1)},
			metrics: goodputTestMetrics,
			wantErr: "not finite",
		},
		{
			name:    "negative_infinite_threshold",
			slos:    map[string]float64{"output_token_throughput": math.Inf(-1)},
			metrics: goodputTestMetrics,
			wantErr: "not finite",
		},
		{
			name:    "infinite_unit_factor",
			slos:    map[string]float64{"m": 1},
			metrics: map[string]GoodputMetric{"m": {DisplayToInternal: math.Inf(1)}},
			wantErr: "cannot convert",
		},
		{
			name:    "negative_conversion_overflow",
			slos:    map[string]float64{"m": -math.MaxFloat64},
			metrics: map[string]GoodputMetric{"m": {DisplayToInternal: 10}},
			wantErr: "overflows",
		},
		{
			name:    "negative_unit_factor",
			slos:    map[string]float64{"m": 1},
			metrics: map[string]GoodputMetric{"m": {DisplayToInternal: -1}},
			wantErr: "cannot convert",
		},
		{
			name:    "nan_unit_factor",
			slos:    map[string]float64{"m": 1},
			metrics: map[string]GoodputMetric{"m": {DisplayToInternal: math.NaN()}},
			wantErr: "cannot convert",
		},
		{
			name:    "conversion_overflow",
			slos:    map[string]float64{"m": math.MaxFloat64},
			metrics: map[string]GoodputMetric{"m": {DisplayToInternal: 10}},
			wantErr: "overflows",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			slos, err := NewGoodputSLOs(tc.slos, tc.metrics)
			if err == nil {
				t.Fatalf("NewGoodputSLOs accepted %v, got %+v", tc.slos, slos)
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("error %q does not contain %q", err, tc.wantErr)
			}
		})
	}
}

func TestGoodputNoSLOsCountsNothing(t *testing.T) {
	records := []map[string]float64{{"request_latency": 1}, {"request_latency": 2}}
	for name, slos := range map[string]*GoodputSLOs{"nil": nil, "zero": {}} {
		if slos.IsGood(records[0]) || slos.GoodRequestCount(records) != 0 {
			t.Fatalf("%s SLO set counted a request as good", name)
		}
		if slos.RequiredTags() != nil {
			t.Fatalf("%s SLO set reported required tags", name)
		}
	}
	empty, err := NewGoodputSLOs(nil, goodputTestMetrics)
	if err != nil {
		t.Fatalf("NewGoodputSLOs(nil): %v", err)
	}
	if empty.GoodRequestCount(records) != 0 || empty.RequiredTags() != nil {
		t.Fatal("empty SLO map counted a request as good or required tags")
	}
}

func TestGoodputRequestCount(t *testing.T) {
	slos, err := NewGoodputSLOs(map[string]float64{
		"request_latency":         250,
		"output_token_throughput": 40,
	}, goodputTestMetrics)
	if err != nil {
		t.Fatalf("NewGoodputSLOs: %v", err)
	}
	records := []map[string]float64{
		{"request_latency": 100e6, "output_token_throughput": 60},
		{"request_latency": 300e6, "output_token_throughput": 60},
		{"request_latency": 100e6, "output_token_throughput": 10},
		{"request_latency": 250e6, "output_token_throughput": 40},
		{"output_token_throughput": 60},
	}
	if got := slos.GoodRequestCount(records); got != 2 {
		t.Fatalf("GoodRequestCount = %d, want 2", got)
	}
}

func TestGoodputNonPositiveThresholdIsApplied(t *testing.T) {
	slos, err := NewGoodputSLOs(map[string]float64{"request_latency": 0}, goodputTestMetrics)
	if err != nil {
		t.Fatalf("NewGoodputSLOs: %v", err)
	}
	if !slos.IsGood(map[string]float64{"request_latency": 0}) {
		t.Fatal("value equal to a zero threshold was not good")
	}
	if slos.IsGood(map[string]float64{"request_latency": 1}) {
		t.Fatal("a zero latency threshold was ignored instead of applied")
	}
}

func TestGoodputRecordWiring(t *testing.T) {
	// Stated in seconds and tok/s; Record carries milliseconds and tok/s.
	slos, err := NewGoodputSLOs(map[string]float64{
		GoodputTagE2E:       1,
		GoodputTagTTFT:      0.2,
		GoodputTagDecodeTPS: 30,
	}, RecordGoodputMetrics())
	if err != nil {
		t.Fatalf("NewGoodputSLOs: %v", err)
	}
	if got, _ := slos.Threshold(GoodputTagE2E); got != 1000 {
		t.Fatalf("e2e threshold = %v ms, want 1000", got)
	}
	records := []Record{
		{E2EMS: 900, TTFTMS: 100, DecodeTPS: 45},                         // good
		{E2EMS: 1000, TTFTMS: 200, DecodeTPS: 30},                        // good: every bound is inclusive
		{E2EMS: 1001, TTFTMS: 100, DecodeTPS: 45},                        // e2e over by 1 ms
		{E2EMS: 900, TTFTMS: 500, DecodeTPS: 45},                         // ttft over
		{E2EMS: 900, TTFTMS: 100, DecodeTPS: 29.9},                       // decode rate under
		{E2EMS: 900, TTFTMS: 0, DecodeTPS: 45},                           // first token never observed
		{E2EMS: 900, TTFTMS: 100},                                        // decode rate never observed
		{E2EMS: 900, TTFTMS: 100, DecodeTPS: 45, Error: "client_cancel"}, // failed turn
		{E2EMS: 900, TTFTMS: 100, DecodeTPS: math.Inf(1)},                // non-finite rate
	}
	if got := slos.GoodRecordCount(records); got != 2 {
		t.Fatalf("GoodRecordCount = %d, want 2", got)
	}
	if got := GoodputRecordValues(records[5]); len(got) != 2 {
		t.Fatalf("unobserved ttft was projected as a value: %v", got)
	}
	if got := GoodputRecordValues(records[7]); got != nil {
		t.Fatalf("failed turn projected values: %v", got)
	}

	// The same number is a ceiling for a latency and a floor for a rate.
	same, err := NewGoodputSLOs(map[string]float64{GoodputTagE2E: 0.05, GoodputTagDecodeTPS: 50}, RecordGoodputMetrics())
	if err != nil {
		t.Fatalf("NewGoodputSLOs: %v", err)
	}
	if !same.IsGood(GoodputRecordValues(Record{E2EMS: 40, DecodeTPS: 60})) {
		t.Fatal("40 ms under a 50 ms ceiling with 60 tok/s over a 50 tok/s floor was not good")
	}
	if same.IsGood(GoodputRecordValues(Record{E2EMS: 60, DecodeTPS: 60})) || same.IsGood(GoodputRecordValues(Record{E2EMS: 40, DecodeTPS: 40})) {
		t.Fatal("a value on the wrong side of its own direction was good")
	}
}

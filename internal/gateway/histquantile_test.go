package gateway

import (
	"math"
	"testing"
)

// TestHistQuantile verifies the histogram quantile estimator (ported from
// ai-dynamo/aiperf@ab610ad, src/aiperf/server_metrics/histogram_percentiles.py,
// Apache-2.0) returns the exact quantile the source's _prometheus_quantile
// produces for known histograms, and handles empty / zero-count histograms.
//
// The expected values below were produced by executing the source's
// _prometheus_quantile (Python float64) on each histogram; Go float64 is the
// same IEEE-754 binary64 type and the port preserves the source's arithmetic
// order, so the results are bit-identical and asserted with exact equality.
func TestHistQuantile(t *testing.T) {
	inf := math.Inf(1)

	cases := []struct {
		name    string
		buckets []HistogramBucket
		wantOK  bool
		want    map[float64]float64 // q -> expected value
	}{
		{
			name:    "empty_histogram",
			buckets: nil,
			wantOK:  false,
		},
		{
			name: "zero_total_count",
			buckets: []HistogramBucket{
				{Bound: 0.1, Count: 0},
				{Bound: 0.5, Count: 0},
				{Bound: inf, Count: 0},
			},
			wantOK: false,
		},
		{
			// The source's docstring example:
			// buckets = {"0.1": 20, "0.5": 60, "1.0": 90, "+Inf": 100}
			name: "docstring_example",
			buckets: []HistogramBucket{
				{Bound: 0.1, Count: 20},
				{Bound: 0.5, Count: 60},
				{Bound: 1.0, Count: 90},
				{Bound: inf, Count: 100},
			},
			wantOK: true,
			want: map[float64]float64{
				0.01: 0.005000000000000001,
				0.05: 0.025,
				0.10: 0.05,
				0.25: 0.15000000000000002,
				0.50: 0.4,
				0.75: 0.75,
				0.90: 1.0, // target rank 90 lands exactly on the 1.0 edge
				0.95: 1.0, // target rank falls in +Inf -> last finite bound
				0.99: 1.0,
			},
		},
		{
			// Power-of-two bounds: every intermediate value is exactly
			// representable in binary64, so the results are clean.
			name: "powers_of_two",
			buckets: []HistogramBucket{
				{Bound: 0.25, Count: 10},
				{Bound: 0.5, Count: 30},
				{Bound: 1.0, Count: 40},
				{Bound: inf, Count: 40},
			},
			wantOK: true,
			want: map[float64]float64{
				0.01: 0.01,
				0.05: 0.05,
				0.10: 0.1,
				0.25: 0.25,
				0.50: 0.375,
				0.75: 0.5,
				0.90: 0.8,
				0.95: 0.9,
				0.99: 0.9800000000000001,
			},
		},
		{
			// No +Inf bucket: total count comes from the last finite bucket
			// (90), mirroring bucket_cumulative[sorted_keys[-1]].
			name: "no_inf_bucket",
			buckets: []HistogramBucket{
				{Bound: 0.1, Count: 20},
				{Bound: 0.5, Count: 60},
				{Bound: 1.0, Count: 90},
			},
			wantOK: true,
			want: map[float64]float64{
				0.01: 0.0045,
				0.05: 0.022500000000000003,
				0.10: 0.045000000000000005,
				0.25: 0.125,
				0.50: 0.35,
				0.75: 0.625,
				0.90: 0.85,
				0.95: 0.925,
				0.99: 0.9849999999999999,
			},
		},
		{
			// All observations in one finite bucket: pure linear sweep.
			name: "single_bucket",
			buckets: []HistogramBucket{
				{Bound: 1.0, Count: 100},
				{Bound: inf, Count: 100},
			},
			wantOK: true,
			want: map[float64]float64{
				0.01: 0.01,
				0.50: 0.5,
				0.99: 0.99,
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if !tc.wantOK {
				got, ok := HistQuantile(tc.buckets, 0.5)
				if ok {
					t.Fatalf("HistQuantile(%v, 0.5) ok = true (value %v), want false", tc.buckets, got)
				}
				return
			}
			for q, want := range tc.want {
				got, ok := HistQuantile(tc.buckets, q)
				if !ok {
					t.Fatalf("HistQuantile(%v, %v) ok = false, want true", tc.buckets, q)
				}
				if got != want {
					t.Errorf("HistQuantile(%v, %v) = %v, want %v", tc.buckets, q, got, want)
				}
			}
		})
	}
}

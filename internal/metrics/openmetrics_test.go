package metrics

import (
	"strings"
	"testing"
)

func TestRenderOpenMetricsTextGolden(t *testing.T) {
	got, err := RenderOpenMetricsText([]OpenMetricFamily{
		{
			Name: "fak_requests_total",
			Help: "requests by tool\nand outcome",
			Type: OpenMetricCounter,
			Samples: []OpenMetricSample{
				{
					Labels: []OpenMetricLabel{
						{Name: "tool", Value: "search\nkb"},
						{Name: "tenant", Value: `a\b"c`},
					},
					Value: 7,
				},
				{
					Labels: []OpenMetricLabel{
						{Name: "tool", Value: "refund"},
						{Name: "tenant", Value: "alpha"},
					},
					Value: 2,
				},
			},
		},
		{
			Name: "fak_request_duration_seconds",
			Help: "preflight latency",
			Type: OpenMetricHistogram,
			Histograms: []OpenMetricHistogramSample{
				{
					Labels: []OpenMetricLabel{{Name: "route", Value: "/preflight"}},
					Buckets: []OpenMetricBucket{
						{UpperBound: 1, CumulativeCount: 5},
						{UpperBound: 0.1, CumulativeCount: 1},
						{UpperBound: 0.5, CumulativeCount: 4},
					},
					Count: 5,
					Sum:   1.7,
				},
			},
		},
		{
			Name: "fak_queue_depth",
			Help: `queued \ calls`,
			Type: OpenMetricGauge,
			Samples: []OpenMetricSample{
				{Labels: []OpenMetricLabel{{Name: "lane", Value: "metrics"}}, Value: 3.5},
			},
		},
	})
	if err != nil {
		t.Fatalf("RenderOpenMetricsText() error = %v", err)
	}

	want := "" +
		"# HELP fak_queue_depth queued \\\\ calls\n" +
		"# TYPE fak_queue_depth gauge\n" +
		"fak_queue_depth{lane=\"metrics\"} 3.5\n" +
		"# HELP fak_request_duration_seconds preflight latency\n" +
		"# TYPE fak_request_duration_seconds histogram\n" +
		"fak_request_duration_seconds_bucket{route=\"/preflight\",le=\"0.1\"} 1\n" +
		"fak_request_duration_seconds_bucket{route=\"/preflight\",le=\"0.5\"} 4\n" +
		"fak_request_duration_seconds_bucket{route=\"/preflight\",le=\"1\"} 5\n" +
		"fak_request_duration_seconds_bucket{route=\"/preflight\",le=\"+Inf\"} 5\n" +
		"fak_request_duration_seconds_sum{route=\"/preflight\"} 1.7\n" +
		"fak_request_duration_seconds_count{route=\"/preflight\"} 5\n" +
		"# HELP fak_requests_total requests by tool\\nand outcome\n" +
		"# TYPE fak_requests_total counter\n" +
		"fak_requests_total{tenant=\"a\\\\b\\\"c\",tool=\"search\\nkb\"} 7\n" +
		"fak_requests_total{tenant=\"alpha\",tool=\"refund\"} 2\n" +
		"# EOF\n"
	if string(got) != want {
		t.Fatalf("rendered OpenMetrics text mismatch\n--- got ---\n%s\n--- want ---\n%s", got, want)
	}
}

func TestRenderOpenMetricsTextRejectsInvalidNamesAndDuplicateLabels(t *testing.T) {
	tests := []struct {
		name    string
		family  OpenMetricFamily
		wantErr string
	}{
		{
			name: "invalid metric name",
			family: OpenMetricFamily{
				Name: "9bad",
				Type: OpenMetricGauge,
			},
			wantErr: "must start",
		},
		{
			name: "invalid label name",
			family: OpenMetricFamily{
				Name: "fak_ok",
				Type: OpenMetricGauge,
				Samples: []OpenMetricSample{{
					Labels: []OpenMetricLabel{{Name: "bad-label", Value: "x"}},
					Value:  1,
				}},
			},
			wantErr: "invalid byte",
		},
		{
			name: "duplicate labels",
			family: OpenMetricFamily{
				Name: "fak_ok",
				Type: OpenMetricGauge,
				Samples: []OpenMetricSample{{
					Labels: []OpenMetricLabel{{Name: "lane", Value: "a"}, {Name: "lane", Value: "b"}},
					Value:  1,
				}},
			},
			wantErr: "duplicate label",
		},
		{
			name: "reserved histogram label",
			family: OpenMetricFamily{
				Name: "fak_latency_seconds",
				Type: OpenMetricHistogram,
				Histograms: []OpenMetricHistogramSample{{
					Labels: []OpenMetricLabel{{Name: "le", Value: "bad"}},
					Count:  1,
				}},
			},
			wantErr: "reserved label",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := RenderOpenMetricsText([]OpenMetricFamily{tc.family})
			if err == nil {
				t.Fatalf("RenderOpenMetricsText() error = nil, want %q", tc.wantErr)
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("RenderOpenMetricsText() error = %q, want substring %q", err, tc.wantErr)
			}
		})
	}
}

func TestRenderOpenMetricsTextRejectsInvalidHistogramBuckets(t *testing.T) {
	tests := []struct {
		name    string
		buckets []OpenMetricBucket
		count   uint64
		wantErr string
	}{
		{
			name: "non monotonic counts",
			buckets: []OpenMetricBucket{
				{UpperBound: 1, CumulativeCount: 3},
				{UpperBound: 2, CumulativeCount: 2},
			},
			count:   3,
			wantErr: "non-monotonic",
		},
		{
			name: "bucket count exceeds total",
			buckets: []OpenMetricBucket{
				{UpperBound: 1, CumulativeCount: 4},
			},
			count:   3,
			wantErr: "exceeds total",
		},
		{
			name: "duplicate bound",
			buckets: []OpenMetricBucket{
				{UpperBound: 1, CumulativeCount: 1},
				{UpperBound: 1, CumulativeCount: 1},
			},
			count:   1,
			wantErr: "duplicate bucket bound",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := RenderOpenMetricsText([]OpenMetricFamily{{
				Name: "fak_latency_seconds",
				Type: OpenMetricHistogram,
				Histograms: []OpenMetricHistogramSample{{
					Buckets: tc.buckets,
					Count:   tc.count,
				}},
			}})
			if err == nil {
				t.Fatalf("RenderOpenMetricsText() error = nil, want %q", tc.wantErr)
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("RenderOpenMetricsText() error = %q, want substring %q", err, tc.wantErr)
			}
		})
	}
}

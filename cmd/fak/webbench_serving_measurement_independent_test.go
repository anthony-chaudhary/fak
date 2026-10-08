package main

import (
	"io"
	"os"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/webbench"
)

// fak-test:justify why=contract when=changed:cmd/fak/**
// fak-test:runtime fast est=5ms lane=default
func TestPrintServingSummaryMeasurementFields(t *testing.T) {
	for _, tc := range []struct {
		name   string
		stats  webbench.ServingStats
		want   []string
		reject []string
	}{
		{
			name:  "measured",
			stats: servingSummaryMeasuredStats(5, 3, 2),
			want:  []string{"wall_s=2.5", "output_exact=5/3", "goodput_tok_s=1.2", "slo_s=1", "observed_http_max=2", "basis=client_http"},
		},
		{
			name:  "legacy unknown",
			stats: webbench.ServingStats{Requests: 4},
			want:  []string{"wall_s=unmeasured", "output_exact=unmeasured/unmeasured", "goodput_tok_s=unmeasured", "slo_s=unmeasured", "observed_http_max=unmeasured", "basis=unmeasured"},
		},
		{
			name: "witnessed zero",
			stats: webbench.ServingStats{
				Requests: 4, WallSeconds: servingSummaryFloat(1), GoodputSLOSeconds: servingSummaryFloat(1),
				SuccessfulOutputTokensExact: servingSummaryInt64(0), SLOSuccessfulOutputTokensExact: servingSummaryInt64(0),
				GoodputTokensS:      webbench.ScalarMetric{Status: "measured", Value: servingSummaryFloat(0)},
				ObservedMaxInFlight: servingSummaryInt(0), ObservedInFlightBasis: "client_http",
			},
			want: []string{"wall_s=1", "output_exact=0/0", "goodput_tok_s=0", "slo_s=1", "observed_http_max=0", "basis=client_http"},
		},
		{
			name: "partial numerator without SLO",
			stats: webbench.ServingStats{
				Requests: 4, WallSeconds: servingSummaryFloat(2.5), SuccessfulOutputTokensExact: servingSummaryInt64(5),
			},
			want: []string{"wall_s=2.5", "output_exact=5/unmeasured", "goodput_tok_s=unmeasured", "slo_s=unmeasured"},
		},
		{
			name: "invalid overlap basis",
			stats: func() webbench.ServingStats {
				stats := servingSummaryMeasuredStats(5, 3, 2)
				stats.ObservedInFlightBasis = "server_gpu"
				return stats
			}(),
			want:   []string{"observed_http_max=unmeasured", "basis=unmeasured"},
			reject: []string{"observed_http_max=2", "basis=server_gpu"},
		},
		{
			name: "overlap above request count",
			stats: func() webbench.ServingStats {
				stats := servingSummaryMeasuredStats(5, 3, 5)
				return stats
			}(),
			want:   []string{"observed_http_max=unmeasured", "basis=unmeasured"},
			reject: []string{"observed_http_max=5"},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			report := &webbench.ServingParityReport{Tracks: []webbench.ServingTrackResult{{Track: webbench.TrackOurs, Status: "measured", Stats: tc.stats}}}
			output := servingSummaryCapture(t, func(file *os.File) { printServingSummary(file, report, "") })
			servingSummaryWantFields(t, output, tc.want, tc.reject)
		})
	}
}

// fak-test:justify why=contract when=changed:cmd/fak/**
// fak-test:runtime fast est=5ms lane=default
func TestPrintServingSummaryPreservesMeasuredThroughputUnits(t *testing.T) {
	for _, tc := range []struct {
		name   string
		metric webbench.ScalarMetric
		want   string
		reject []string
	}{
		{
			name:   "stream events remain events",
			metric: webbench.ScalarMetric{Status: "measured", Value: servingSummaryFloat(7), Unit: "stream_content_events/s"},
			want:   "throughput=7 stream_content_events/s",
			reject: []string{"throughput=7 usage.completion_tokens/s", "throughput=7 tok/s"},
		},
		{
			name:   "exact usage keeps exact basis",
			metric: webbench.ScalarMetric{Status: "measured", Value: servingSummaryFloat(7), Unit: "usage.completion_tokens/s"},
			want:   "throughput=7 usage.completion_tokens/s",
			reject: []string{"throughput=7 stream_content_events/s", "throughput=7 tok/s"},
		},
		{
			name:   "estimated content keeps estimated basis",
			metric: webbench.ScalarMetric{Status: "measured", Value: servingSummaryFloat(7), Unit: "estimated_content_tokens/s"},
			want:   "throughput=7 estimated_content_tokens/s",
		},
		{
			name:   "output estimate keeps estimate basis",
			metric: webbench.ScalarMetric{Status: "measured", Value: servingSummaryFloat(7), Unit: "output_token_estimate/s"},
			want:   "throughput=7 output_token_estimate/s",
		},
		{
			name:   "legacy missing rate",
			metric: webbench.ScalarMetric{},
			want:   "throughput=unmeasured",
		},
		{
			name:   "unsupported unit",
			metric: webbench.ScalarMetric{Status: "measured", Value: servingSummaryFloat(7), Unit: "tok/s"},
			want:   "throughput=unmeasured",
			reject: []string{"throughput=7 tok/s"},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			stats := servingSummaryMeasuredStats(5, 3, 2)
			stats.ThroughputTokensS = tc.metric
			report := &webbench.ServingParityReport{Tracks: []webbench.ServingTrackResult{{Track: webbench.TrackOurs, Status: "measured", Stats: stats}}}
			output := servingSummaryCapture(t, func(file *os.File) { printServingSummary(file, report, "") })
			servingSummaryWantFields(t, output, []string{tc.want}, tc.reject)
		})
	}
}

// fak-test:justify why=contract when=changed:cmd/fak/**
// fak-test:runtime fast est=5ms lane=default
func TestPrintServingSweepSummaryUsesSelectedPointStats(t *testing.T) {
	stats := servingSummaryMeasuredStats(5, 3, 2)
	report := servingSummarySweepReport([]webbench.ServingSweepTrackPoint{{
		Track: webbench.TrackOurs, Status: "valid", MeasurementStatus: "measured", Stats: stats,
	}})
	output := servingSummaryCapture(t, func(file *os.File) { printServingSweepSummary(file, report, "") })
	for _, field := range []string{"wall_s=2.5", "output_exact=5/3", "goodput_tok_s=1.2", "slo_s=1", "observed_http_max=2", "basis=client_http"} {
		if count := strings.Count(output, field); count != 2 {
			t.Errorf("selected peak/knee output contains %q %d times, want 2", field, count)
		}
	}
}

// fak-test:justify why=contract when=changed:cmd/fak/**
// fak-test:runtime fast est=5ms lane=default
func TestPrintServingSweepSummaryUsesSelectedSavedThroughputUnit(t *testing.T) {
	for _, tc := range []struct {
		name   string
		metric webbench.ScalarMetric
		want   []string
		reject []string
	}{
		{
			name:   "stream events remain events",
			metric: webbench.ScalarMetric{Status: "measured", Value: servingSummaryFloat(7), Unit: "stream_content_events/s"},
			want:   []string{"peak=c2/7 stream_content_events/s", "sla-knee=c2/7 stream_content_events/s"},
			reject: []string{"peak=c2/7 tok/s", "sla-knee=c2/7 tok/s", "c2/999"},
		},
		{
			name:   "exact usage keeps exact basis",
			metric: webbench.ScalarMetric{Status: "measured", Value: servingSummaryFloat(7), Unit: "usage.completion_tokens/s"},
			want:   []string{"peak=c2/7 usage.completion_tokens/s", "sla-knee=c2/7 usage.completion_tokens/s"},
			reject: []string{"peak=c2/7 tok/s", "sla-knee=c2/7 tok/s", "c2/999"},
		},
		{
			name:   "missing selected rate",
			metric: webbench.ScalarMetric{},
			want:   []string{"peak=c2/unmeasured", "sla-knee=c2/unmeasured"},
			reject: []string{"c2/999"},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			stats := servingSummaryMeasuredStats(5, 3, 2)
			stats.ThroughputTokensS = tc.metric
			report := servingSummarySweepReport([]webbench.ServingSweepTrackPoint{{
				Track: webbench.TrackOurs, Status: "valid", MeasurementStatus: "measured", Stats: stats,
			}})
			report.Tracks[0].Peak.ThroughputTokens = 999
			report.Tracks[0].SLAKnee.ThroughputTokens = 999
			output := servingSummaryCapture(t, func(file *os.File) { printServingSweepSummary(file, report, "") })
			servingSummaryWantFields(t, output, tc.want, tc.reject)
		})
	}
}

// fak-test:justify why=invariant when=changed:cmd/fak/**
// fak-test:runtime fast est=5ms lane=default
func TestPrintServingSweepSummaryRejectsUnqualifiedSelectedStats(t *testing.T) {
	measured := servingSummaryMeasuredStats(5, 3, 2)
	for _, tc := range []struct {
		name   string
		points []webbench.ServingSweepTrackPoint
	}{
		{name: "missing coordinate"},
		{
			name: "explicit not measured",
			points: []webbench.ServingSweepTrackPoint{{
				Track: webbench.TrackOurs, Status: "not_measured", MeasurementStatus: "not_measured", Stats: measured,
			}},
		},
		{
			name: "duplicate coordinate",
			points: []webbench.ServingSweepTrackPoint{
				{Track: webbench.TrackOurs, Status: "valid", MeasurementStatus: "measured", Stats: measured},
				{Track: webbench.TrackOurs, Status: "valid", MeasurementStatus: "measured", Stats: measured},
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			report := servingSummarySweepReport(tc.points)
			output := servingSummaryCapture(t, func(file *os.File) { printServingSweepSummary(file, report, "") })
			servingSummaryWantFields(t, output,
				[]string{"wall_s=unmeasured", "output_exact=unmeasured/unmeasured", "goodput_tok_s=unmeasured", "slo_s=unmeasured", "observed_http_max=unmeasured", "basis=unmeasured"},
				[]string{"wall_s=2.5", "output_exact=5/3", "observed_http_max=2"})
		})
	}
}

func servingSummaryMeasuredStats(successful, qualified int64, observed int) webbench.ServingStats {
	return webbench.ServingStats{
		Requests: 4, WallSeconds: servingSummaryFloat(2.5), GoodputSLOSeconds: servingSummaryFloat(1),
		SuccessfulOutputTokensExact: servingSummaryInt64(successful), SLOSuccessfulOutputTokensExact: servingSummaryInt64(qualified),
		GoodputTokensS:      webbench.ScalarMetric{Status: "measured", Value: servingSummaryFloat(1.2)},
		ObservedMaxInFlight: servingSummaryInt(observed), ObservedInFlightBasis: "client_http",
	}
}

func servingSummarySweepReport(points []webbench.ServingSweepTrackPoint) *webbench.ServingSweepReport {
	return &webbench.ServingSweepReport{
		Points: []webbench.ServingSweepPoint{{Concurrency: 2, Tracks: points}},
		Tracks: []webbench.ServingSweepTrackSummary{{
			Track: webbench.TrackOurs, PeakStatus: "measured", Peak: &webbench.ServingSweepSelection{Concurrency: 2},
			SLAStatus: "measured", SLAKnee: &webbench.ServingSweepSelection{Concurrency: 2},
		}},
	}
}

func servingSummaryCapture(t *testing.T, print func(*os.File)) string {
	t.Helper()
	file, err := os.CreateTemp(t.TempDir(), "serving-summary-*.txt")
	if err != nil {
		t.Fatalf("create output: %v", err)
	}
	print(file)
	if _, err := file.Seek(0, io.SeekStart); err != nil {
		file.Close()
		t.Fatalf("rewind output: %v", err)
	}
	raw, err := io.ReadAll(file)
	if closeErr := file.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		t.Fatalf("read output: %v", err)
	}
	return string(raw)
}

func servingSummaryWantFields(t *testing.T, output string, want, reject []string) {
	t.Helper()
	for _, field := range want {
		if !strings.Contains(output, field) {
			t.Errorf("summary missing %q", field)
		}
	}
	for _, field := range reject {
		if strings.Contains(output, field) {
			t.Errorf("summary invented %q", field)
		}
	}
}

func servingSummaryFloat(value float64) *float64 { return &value }
func servingSummaryInt(value int) *int           { return &value }
func servingSummaryInt64(value int64) *int64     { return &value }

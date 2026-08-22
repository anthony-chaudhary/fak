package modelperfobs

import (
	"strings"
	"testing"
	"time"
)

func TestAttributionRefusalsNameRecoveryEvidence(t *testing.T) {
	t0 := time.Unix(2_000, 0).UTC()
	tests := []struct {
		name    string
		mutate  func(*AttributionInput)
		grade   AttributionGrade
		wantCue string
	}{
		{
			name: "scrape failure",
			mutate: func(in *AttributionInput) {
				in.ScrapeFailure = "timeout"
			},
			grade:   GradeUnavailable,
			wantCue: "capture successful before and after metrics scrapes",
		},
		{
			name: "missing snapshot",
			mutate: func(in *AttributionInput) {
				in.Before = nil
			},
			grade:   GradeUnavailable,
			wantCue: "capture both snapshots around the request window",
		},
		{
			name: "missing instance ID",
			mutate: func(in *AttributionInput) {
				in.Before.ServerInstanceID = ""
			},
			grade:   GradeUnavailable,
			wantCue: "record the same non-empty server instance ID in both snapshots",
		},
		{
			name: "instance restart",
			mutate: func(in *AttributionInput) {
				in.After.ServerInstanceID = "server-b"
			},
			grade:   GradeUnavailable,
			wantCue: "capture a new before/after pair from one server instance",
		},
		{
			name: "invalid bounds",
			mutate: func(in *AttributionInput) {
				in.After.ScrapedAt = in.Before.ScrapedAt
			},
			grade:   GradeUnavailable,
			wantCue: "capture an after scrape timestamp later than the before scrape",
		},
		{
			name: "stale evidence",
			mutate: func(in *AttributionInput) {
				in.ObservedAt = in.After.ScrapedAt.Add(10 * time.Second)
				in.MaxScrapeAge = time.Second
			},
			grade:   GradeStale,
			wantCue: "refresh the after scrape",
		},
		{
			name: "reset causal counter",
			mutate: func(in *AttributionInput) {
				in.After.Counters[CounterPrefixCacheHits] = CounterSample{Value: 1}
			},
			grade:   GradeIsolatedWindow,
			wantCue: "capture a fresh monotonic prefix_cache_hits counter pair",
		},
		{
			name: "untrusted correlation source",
			mutate: func(in *AttributionInput) {
				in.After.Counters[CounterRequests] = CounterSample{Value: 12}
				in.Requests = append(in.Requests, RequestWindow{
					RequestID: "B",
					StartedAt: t0.Add(200 * time.Millisecond),
					EndedAt:   t0.Add(800 * time.Millisecond),
				})
				in.After.RequestCounters = map[string]CorrelatedCounterSet{
					"A": {
						Source: CorrelationSource("adapter-guess"),
						Counters: CounterSet{
							CounterPrefixCacheHits: {Value: 1},
						},
					},
				}
			},
			grade:   GradeCohortOnly,
			wantCue: "use request-label or trace correlation",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			input := refusalAttributionInput(t0)
			tt.mutate(&input)
			report := AttributeMetrics(input)

			if report.Delta.Grade != tt.grade {
				t.Fatalf("delta grade=%q, want %q: %+v", report.Delta.Grade, tt.grade, report.Delta)
			}
			for _, reason := range attributionReasons(report) {
				if !strings.Contains(reason, tt.wantCue) {
					t.Errorf("reason %q does not name recovery cue %q", reason, tt.wantCue)
				}
			}
			for _, row := range report.Requests {
				if row.Grade != tt.grade || len(row.Causes) != 0 || row.Correlation != "" {
					t.Errorf("request attribution did not fail closed: %+v", row)
				}
			}
		})
	}
}

func refusalAttributionInput(t0 time.Time) AttributionInput {
	return AttributionInput{
		Before: &MetricsSnapshot{
			ServerInstanceID: "server-a",
			ScrapedAt:        t0,
			Counters: CounterSet{
				CounterRequests:        {Value: 10},
				CounterPrefixCacheHits: {Value: 20},
			},
		},
		After: &MetricsSnapshot{
			ServerInstanceID: "server-a",
			ScrapedAt:        t0.Add(time.Second),
			Counters: CounterSet{
				CounterRequests:        {Value: 11},
				CounterPrefixCacheHits: {Value: 21},
			},
		},
		Requests: []RequestWindow{{
			RequestID: "A",
			StartedAt: t0.Add(100 * time.Millisecond),
			EndedAt:   t0.Add(900 * time.Millisecond),
		}},
	}
}

func attributionReasons(report AttributionReport) []string {
	reasons := []string{report.Delta.Reason}
	if report.Cohort != nil {
		reasons = append(reasons, report.Cohort.Reason)
	}
	for _, row := range report.Requests {
		reasons = append(reasons, row.Reason)
	}
	return reasons
}

package modelperfobs

import (
	"math"
	"slices"
	"strings"
	"testing"
	"time"
)

func TestAdversarialAttributionUnsupportedInputsStayFailClosed(t *testing.T) {
	t0 := time.Unix(1_000, 0).UTC()
	tests := []struct {
		name           string
		mutate         func(*AttributionInput)
		wantGrade      AttributionGrade
		wantRequestIDs []string
		wantReason     string
	}{
		{
			name:       "missing before snapshot",
			mutate:     func(in *AttributionInput) { in.Before = nil },
			wantGrade:  GradeUnavailable,
			wantReason: "both metrics scrapes are required",
		},
		{
			name:       "missing after snapshot",
			mutate:     func(in *AttributionInput) { in.After = nil },
			wantGrade:  GradeUnavailable,
			wantReason: "both metrics scrapes are required",
		},
		{
			name:           "empty before snapshot",
			mutate:         func(in *AttributionInput) { in.Before = &MetricsSnapshot{} },
			wantGrade:      GradeUnavailable,
			wantRequestIDs: []string{"A"},
			wantReason:     "server-instance identity is unavailable",
		},
		{
			name:       "empty after snapshot",
			mutate:     func(in *AttributionInput) { in.After = &MetricsSnapshot{} },
			wantGrade:  GradeUnavailable,
			wantReason: "scrape bounds are invalid",
		},
		{
			name: "equal scrape bounds",
			mutate: func(in *AttributionInput) {
				in.After.ScrapedAt = in.Before.ScrapedAt
			},
			wantGrade:  GradeUnavailable,
			wantReason: "scrape bounds are invalid",
		},
		{
			name: "reversed scrape bounds",
			mutate: func(in *AttributionInput) {
				in.After.ScrapedAt = in.Before.ScrapedAt.Add(-time.Nanosecond)
			},
			wantGrade:  GradeUnavailable,
			wantReason: "scrape bounds are invalid",
		},
		{
			name: "empty request ID",
			mutate: func(in *AttributionInput) {
				in.Requests = []RequestWindow{{StartedAt: t0.Add(100 * time.Millisecond), EndedAt: t0.Add(900 * time.Millisecond)}}
			},
			wantGrade:  GradeContaminated,
			wantReason: "unobserved request",
		},
		{
			name: "duplicate request ID",
			mutate: func(in *AttributionInput) {
				window := RequestWindow{RequestID: "duplicate", StartedAt: t0.Add(100 * time.Millisecond), EndedAt: t0.Add(900 * time.Millisecond)}
				in.Requests = []RequestWindow{window, window}
			},
			wantGrade:      GradeCohortOnly,
			wantRequestIDs: []string{"duplicate"},
			wantReason:     "duplicate request IDs",
		},
		{
			name: "request counter missing before",
			mutate: func(in *AttributionInput) {
				delete(in.Before.Counters, CounterRequests)
			},
			wantGrade:      GradeCohortOnly,
			wantRequestIDs: []string{"A"},
			wantReason:     "usable server request counter",
		},
		{
			name: "extreme monotonic counters",
			mutate: func(in *AttributionInput) {
				in.Before.Counters[CounterRequests] = CounterSample{Value: math.MaxUint64 - 1}
				in.After.Counters[CounterRequests] = CounterSample{Value: math.MaxUint64}
				in.Before.Counters[CounterPrefixCacheHits] = CounterSample{Value: math.MaxUint64 - 1}
				in.After.Counters[CounterPrefixCacheHits] = CounterSample{Value: math.MaxUint64}
				in.Requests = []RequestWindow{
					{RequestID: "A", StartedAt: t0.Add(100 * time.Millisecond), EndedAt: t0.Add(800 * time.Millisecond)},
					{RequestID: "B", StartedAt: t0.Add(200 * time.Millisecond), EndedAt: t0.Add(900 * time.Millisecond)},
				}
			},
			wantGrade:      GradeCohortOnly,
			wantRequestIDs: []string{"A", "B"},
			wantReason:     "multiple observed requests",
		},
		{
			name: "unknown correlation source",
			mutate: func(in *AttributionInput) {
				in.After.Counters[CounterRequests] = CounterSample{Value: 12}
				in.Requests = []RequestWindow{
					{RequestID: "A", StartedAt: t0.Add(100 * time.Millisecond), EndedAt: t0.Add(800 * time.Millisecond)},
					{RequestID: "B", StartedAt: t0.Add(200 * time.Millisecond), EndedAt: t0.Add(900 * time.Millisecond)},
				}
				in.After.RequestCounters = map[string]CorrelatedCounterSet{
					"A": {Source: CorrelationSource("adapter-guess"), Counters: CounterSet{CounterPrefixCacheHits: {Value: 1}}},
				}
			},
			wantGrade:      GradeCohortOnly,
			wantRequestIDs: []string{"A", "B"},
			wantReason:     "multiple observed requests",
		},
		{
			name: "correlation source changes between snapshots",
			mutate: func(in *AttributionInput) {
				in.After.Counters[CounterRequests] = CounterSample{Value: 12}
				in.Requests = []RequestWindow{
					{RequestID: "A", StartedAt: t0.Add(100 * time.Millisecond), EndedAt: t0.Add(800 * time.Millisecond)},
					{RequestID: "B", StartedAt: t0.Add(200 * time.Millisecond), EndedAt: t0.Add(900 * time.Millisecond)},
				}
				in.Before.RequestCounters = map[string]CorrelatedCounterSet{
					"A": {Source: CorrelationTrace, Counters: CounterSet{CounterPrefixCacheHits: {Value: 0}}},
				}
				in.After.RequestCounters = map[string]CorrelatedCounterSet{
					"A": {Source: CorrelationRequestLabel, Counters: CounterSet{CounterPrefixCacheHits: {Value: 1}}},
				}
			},
			wantGrade:      GradeCohortOnly,
			wantRequestIDs: []string{"A", "B"},
			wantReason:     "multiple observed requests",
		},
		{
			name: "stale trusted correlation",
			mutate: func(in *AttributionInput) {
				in.After.RequestCounters = map[string]CorrelatedCounterSet{
					"A": {Source: CorrelationTrace, Counters: CounterSet{CounterPrefixCacheHits: {Value: 1}}},
				}
				in.ObservedAt = in.After.ScrapedAt.Add(10 * time.Second)
				in.MaxScrapeAge = time.Second
			},
			wantGrade:      GradeStale,
			wantRequestIDs: []string{"A"},
			wantReason:     "latest scrape is older",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			input := adversarialAttributionInput(t0)
			tt.mutate(&input)
			report := AttributeMetrics(input)

			if report.Schema != AttributionSchema {
				t.Fatalf("schema=%q, want %q", report.Schema, AttributionSchema)
			}
			if report.Delta.Grade != tt.wantGrade || !strings.Contains(report.Delta.Reason, tt.wantReason) {
				t.Fatalf("delta=%+v, want grade=%q reason containing %q", report.Delta, tt.wantGrade, tt.wantReason)
			}
			if !slices.Equal(report.Delta.OverlappingRequestIDs, tt.wantRequestIDs) {
				t.Fatalf("overlapping request IDs=%v, want %v", report.Delta.OverlappingRequestIDs, tt.wantRequestIDs)
			}
			if !failClosedAttributionGrade(report.Delta.Grade) {
				t.Fatalf("unsupported input escaped fail-closed grades: %+v", report.Delta)
			}
			if report.Cohort == nil || report.Cohort.Grade != report.Delta.Grade {
				t.Fatalf("cohort=%+v, want grade %q", report.Cohort, report.Delta.Grade)
			}
			for _, row := range report.Requests {
				if len(row.Causes) != 0 || row.Grade != report.Delta.Grade || row.Correlation != "" {
					t.Fatalf("unsupported input fabricated request attribution: %+v", row)
				}
			}
		})
	}
}

func adversarialAttributionInput(t0 time.Time) AttributionInput {
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

func failClosedAttributionGrade(grade AttributionGrade) bool {
	switch grade {
	case GradeCohortOnly, GradeContaminated, GradeStale, GradeUnavailable:
		return true
	default:
		return false
	}
}

package modelperfobs

import (
	"bytes"
	"encoding/json"
	"reflect"
	"slices"
	"testing"
	"time"
)

func TestConcurrentTrafficDoesNotFabricateRequestCacheAttribution(t *testing.T) {
	t0 := time.Unix(100, 0).UTC()
	before := &MetricsSnapshot{
		ServerInstanceID: "server-a",
		ScrapedAt:        t0,
		Counters: CounterSet{
			CounterRequests:        {Value: 10},
			CounterPrefixCacheHits: {Value: 20},
		},
	}
	after := &MetricsSnapshot{
		ServerInstanceID: "server-a",
		ScrapedAt:        t0.Add(2 * time.Second),
		Counters: CounterSet{
			CounterRequests:        {Value: 12},
			CounterPrefixCacheHits: {Value: 21},
		},
	}
	requests := []RequestWindow{
		{RequestID: "A", StartedAt: t0.Add(100 * time.Millisecond), EndedAt: t0.Add(1500 * time.Millisecond)},
		{RequestID: "B", StartedAt: t0.Add(500 * time.Millisecond), EndedAt: t0.Add(1900 * time.Millisecond)},
	}

	report := AttributeMetrics(AttributionInput{Before: before, After: after, Requests: requests})
	if report.Delta.Grade != GradeCohortOnly {
		t.Fatalf("shared delta grade=%q, want %q", report.Delta.Grade, GradeCohortOnly)
	}
	if report.Delta.ServerInstanceID != "server-a" || !report.Delta.ScrapeStartedAt.Equal(t0) || !report.Delta.ScrapeEndedAt.Equal(t0.Add(2*time.Second)) || !slices.Equal(report.Delta.OverlappingRequestIDs, []string{"A", "B"}) || report.Delta.OverlappingObservedRequests != 2 {
		t.Fatalf("shared delta provenance=%+v", report.Delta)
	}
	if report.Cohort == nil || !slices.Contains(report.Cohort.Causes, CausePrefixCacheHit) {
		t.Fatalf("cohort evidence=%+v, want visible prefix-hit cause", report.Cohort)
	}
	assertRequestAttribution(t, report, "A", GradeCohortOnly, nil)
	assertRequestAttribution(t, report, "B", GradeCohortOnly, nil)

	after.RequestCounters = map[string]CorrelatedCounterSet{
		"A": {Source: CorrelationSource("adapter-guess"), Counters: CounterSet{CounterPrefixCacheHits: {Value: 1}}},
	}
	report = AttributeMetrics(AttributionInput{Before: before, After: after, Requests: requests})
	assertRequestAttribution(t, report, "A", GradeCohortOnly, nil)

	after.RequestCounters["A"] = CorrelatedCounterSet{Source: CorrelationTrace, Counters: CounterSet{CounterPrefixCacheHits: {Value: 1}}}
	report = AttributeMetrics(AttributionInput{Before: before, After: after, Requests: requests})
	assertRequestAttribution(t, report, "A", GradeRequestCorrelated, []Cause{CausePrefixCacheHit})
	assertRequestAttribution(t, report, "B", GradeCohortOnly, nil)
}

func TestAttributionRequiresProofOfAnIsolatedWindow(t *testing.T) {
	t0 := time.Unix(200, 0).UTC()
	request := []RequestWindow{{RequestID: "solo", StartedAt: t0.Add(time.Millisecond), EndedAt: t0.Add(900 * time.Millisecond)}}
	before := &MetricsSnapshot{ServerInstanceID: "server-a", ScrapedAt: t0, Counters: CounterSet{
		CounterRequests: {Value: 4}, CounterPrefixCacheHits: {Value: 8}, CounterCacheEvictions: {Value: 2}, CounterQueueEvents: {Value: 5}, CounterPreemptions: {Value: 1},
	}}
	after := &MetricsSnapshot{ServerInstanceID: "server-a", ScrapedAt: t0.Add(time.Second), Counters: CounterSet{
		CounterRequests: {Value: 5}, CounterPrefixCacheHits: {Value: 9}, CounterCacheEvictions: {Value: 3}, CounterQueueEvents: {Value: 6}, CounterPreemptions: {Value: 2},
	}}

	report := AttributeMetrics(AttributionInput{Before: before, After: after, Requests: request})
	assertRequestAttribution(t, report, "solo", GradeIsolatedWindow, []Cause{CausePrefixCacheHit, CauseCacheEviction, CauseQueue, CausePreemption})
	if report.Cohort != nil || report.Delta.BackgroundUnobserved == nil || *report.Delta.BackgroundUnobserved != 0 {
		t.Fatalf("isolated report provenance=%+v", report)
	}

	delete(before.Counters, CounterRequests)
	delete(after.Counters, CounterRequests)
	report = AttributeMetrics(AttributionInput{Before: before, After: after, Requests: request})
	assertRequestAttribution(t, report, "solo", GradeCohortOnly, nil)
	if report.Cohort == nil || len(report.Cohort.Causes) != 4 {
		t.Fatalf("generic no-label cohort=%+v", report.Cohort)
	}
}

func TestAttributionDetectsBackgroundUnobservedTrafficAtFourRequestEnvelope(t *testing.T) {
	t0 := time.Unix(300, 0).UTC()
	requests := make([]RequestWindow, 0, 4)
	for _, id := range []string{"A", "B", "C", "D"} {
		requests = append(requests, RequestWindow{RequestID: id, StartedAt: t0.Add(100 * time.Millisecond), EndedAt: t0.Add(900 * time.Millisecond)})
	}
	before := &MetricsSnapshot{ServerInstanceID: "server-a", ScrapedAt: t0, Counters: CounterSet{
		CounterRequests: {Value: 10}, CounterPrefixCacheHits: {Value: 20},
	}}
	after := &MetricsSnapshot{ServerInstanceID: "server-a", ScrapedAt: t0.Add(time.Second), Counters: CounterSet{
		CounterRequests: {Value: 15}, CounterPrefixCacheHits: {Value: 21},
	}}

	report := AttributeMetrics(AttributionInput{Before: before, After: after, Requests: requests})
	if report.Delta.Grade != GradeContaminated || report.Delta.OverlappingObservedRequests != 4 || report.Delta.BackgroundUnobserved == nil || *report.Delta.BackgroundUnobserved != 1 {
		t.Fatalf("contamination provenance=%+v", report.Delta)
	}
	for _, id := range []string{"A", "B", "C", "D"} {
		assertRequestAttribution(t, report, id, GradeContaminated, nil)
	}
	if report.Cohort == nil || !slices.Contains(report.Cohort.Causes, CausePrefixCacheHit) {
		t.Fatalf("contaminated cohort evidence=%+v", report.Cohort)
	}
}

func TestAttributionFailsClosedForScrapeRestartAndStaleness(t *testing.T) {
	t0 := time.Unix(400, 0).UTC()
	request := []RequestWindow{{RequestID: "A", StartedAt: t0.Add(time.Millisecond), EndedAt: t0.Add(900 * time.Millisecond)}}
	before := &MetricsSnapshot{ServerInstanceID: "server-a", ScrapedAt: t0, Counters: CounterSet{CounterRequests: {Value: 1}, CounterPrefixCacheHits: {Value: 1}}}
	after := &MetricsSnapshot{ServerInstanceID: "server-a", ScrapedAt: t0.Add(time.Second), Counters: CounterSet{CounterRequests: {Value: 2}, CounterPrefixCacheHits: {Value: 2}}}

	tests := []struct {
		name  string
		input AttributionInput
		grade AttributionGrade
	}{
		{name: "scrape failure", input: AttributionInput{Before: before, After: after, Requests: request, ScrapeFailure: "timeout"}, grade: GradeUnavailable},
		{name: "server restart", input: AttributionInput{Before: before, After: &MetricsSnapshot{ServerInstanceID: "server-b", ScrapedAt: after.ScrapedAt, Counters: after.Counters}, Requests: request}, grade: GradeUnavailable},
		{name: "stale", input: AttributionInput{Before: before, After: after, Requests: request, ObservedAt: after.ScrapedAt.Add(10 * time.Second), MaxScrapeAge: time.Second}, grade: GradeStale},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			report := AttributeMetrics(tt.input)
			if report.Delta.Grade != tt.grade || len(report.Delta.Counters) != 0 && tt.grade == GradeUnavailable {
				t.Fatalf("delta=%+v, want grade %q and no usable unavailable counters", report.Delta, tt.grade)
			}
			for _, row := range report.Requests {
				if len(row.Causes) != 0 {
					t.Fatalf("request received cause from %s evidence: %+v", tt.grade, row)
				}
			}
		})
	}
}

func TestAttributionDistinguishesCounterResetFromKnownWrap(t *testing.T) {
	t0 := time.Unix(500, 0).UTC()
	requests := []RequestWindow{{RequestID: "A", StartedAt: t0.Add(time.Millisecond), EndedAt: t0.Add(900 * time.Millisecond)}}
	before := &MetricsSnapshot{ServerInstanceID: "server-a", ScrapedAt: t0, Counters: CounterSet{
		CounterRequests: {Value: 10}, CounterPrefixCacheHits: {Value: 98},
	}}
	after := &MetricsSnapshot{ServerInstanceID: "server-a", ScrapedAt: t0.Add(time.Second), Counters: CounterSet{
		CounterRequests: {Value: 11}, CounterPrefixCacheHits: {Value: 3},
	}}

	report := AttributeMetrics(AttributionInput{Before: before, After: after, Requests: requests})
	if !slices.Contains(report.Delta.CounterResets, CounterPrefixCacheHits) || report.Delta.Counters[CounterPrefixCacheHits].State != CounterReset {
		t.Fatalf("reset provenance=%+v", report.Delta)
	}
	assertRequestAttribution(t, report, "A", GradeIsolatedWindow, nil)

	before.Counters[CounterPrefixCacheHits] = CounterSample{Value: 98, WrapAt: 100}
	after.Counters[CounterPrefixCacheHits] = CounterSample{Value: 3, WrapAt: 100}
	report = AttributeMetrics(AttributionInput{Before: before, After: after, Requests: requests})
	if !slices.Contains(report.Delta.CounterWraps, CounterPrefixCacheHits) || report.Delta.Counters[CounterPrefixCacheHits].Value != 5 {
		t.Fatalf("wrap provenance=%+v", report.Delta)
	}
	assertRequestAttribution(t, report, "A", GradeIsolatedWindow, []Cause{CausePrefixCacheHit})
}

func TestAttributionDeterminismUnderConcurrentReads(t *testing.T) {
	t0 := time.Unix(600, 0).UTC()
	input := AttributionInput{
		Before: &MetricsSnapshot{
			ServerInstanceID: "server-a",
			ScrapedAt:        t0,
			Counters: CounterSet{
				CounterRequests:        {Value: 40},
				CounterPrefixCacheHits: {Value: 100},
				CounterCacheEvictions:  {Value: 7},
				CounterQueueEvents:     {Value: 10},
				CounterPreemptions:     {Value: 1},
			},
			RequestCounters: map[string]CorrelatedCounterSet{
				"A": {Source: CorrelationTrace, Counters: CounterSet{CounterPrefixCacheHits: {Value: 10}}},
				"B": {Source: CorrelationRequestLabel, Counters: CounterSet{CounterQueueEvents: {Value: 20}}},
				"C": {Source: CorrelationSource("adapter-guess"), Counters: CounterSet{CounterCacheEvictions: {Value: 30}}},
			},
		},
		After: &MetricsSnapshot{
			ServerInstanceID: "server-a",
			ScrapedAt:        t0.Add(time.Second),
			Counters: CounterSet{
				CounterRequests:        {Value: 44},
				CounterPrefixCacheHits: {Value: 102},
				CounterCacheEvictions:  {Value: 8},
				CounterQueueEvents:     {Value: 13},
				CounterPreemptions:     {Value: 2},
			},
			RequestCounters: map[string]CorrelatedCounterSet{
				"A": {Source: CorrelationTrace, Counters: CounterSet{CounterPrefixCacheHits: {Value: 11}}},
				"B": {Source: CorrelationRequestLabel, Counters: CounterSet{CounterQueueEvents: {Value: 22}}},
				"C": {Source: CorrelationSource("adapter-guess"), Counters: CounterSet{CounterCacheEvictions: {Value: 31}}},
			},
		},
		Requests: []RequestWindow{
			{RequestID: "D", StartedAt: t0.Add(400 * time.Millisecond), EndedAt: t0.Add(900 * time.Millisecond)},
			{RequestID: "B", StartedAt: t0.Add(200 * time.Millisecond), EndedAt: t0.Add(700 * time.Millisecond)},
			{RequestID: "A", StartedAt: t0.Add(100 * time.Millisecond), EndedAt: t0.Add(600 * time.Millisecond)},
			{RequestID: "C", StartedAt: t0.Add(300 * time.Millisecond), EndedAt: t0.Add(800 * time.Millisecond)},
		},
	}

	baseline := AttributeMetrics(input)
	assertRequestAttribution(t, baseline, "A", GradeRequestCorrelated, []Cause{CausePrefixCacheHit})
	assertRequestAttribution(t, baseline, "B", GradeRequestCorrelated, []Cause{CauseQueue})
	assertRequestAttribution(t, baseline, "C", GradeCohortOnly, nil)
	assertRequestAttribution(t, baseline, "D", GradeCohortOnly, nil)
	baselineJSON, err := json.Marshal(baseline)
	if err != nil {
		t.Fatal(err)
	}

	type result struct {
		run    int
		report AttributionReport
		json   []byte
		err    error
	}
	const repetitions = 64
	start := make(chan struct{})
	results := make(chan result, repetitions)
	for run := range repetitions {
		go func() {
			<-start
			report := AttributeMetrics(input)
			encoded, err := json.Marshal(report)
			results <- result{run: run, report: report, json: encoded, err: err}
		}()
	}
	close(start)
	for range repetitions {
		got := <-results
		if got.err != nil {
			t.Fatalf("run %d: marshal report: %v", got.run, got.err)
		}
		if !reflect.DeepEqual(got.report, baseline) {
			t.Fatalf("run %d: report differs from baseline\ngot:  %+v\nwant: %+v", got.run, got.report, baseline)
		}
		if !bytes.Equal(got.json, baselineJSON) {
			t.Fatalf("run %d: JSON differs from baseline\ngot:  %s\nwant: %s", got.run, got.json, baselineJSON)
		}
	}
}

func assertRequestAttribution(t *testing.T, report AttributionReport, requestID string, grade AttributionGrade, causes []Cause) {
	t.Helper()
	for _, got := range report.Requests {
		if got.RequestID != requestID {
			continue
		}
		if got.Grade != grade || !slices.Equal(got.Causes, causes) {
			t.Fatalf("request %s attribution=%+v, want grade=%q causes=%v", requestID, got, grade, causes)
		}
		return
	}
	t.Fatalf("request %s missing from report %+v", requestID, report)
}

package modelperfobs

import (
	"fmt"
	"sort"
	"time"
)

const AttributionSchema = "fak-model-perf-attribution/1"

type AttributionGrade string

const (
	GradeRequestCorrelated AttributionGrade = "request-correlated"
	GradeIsolatedWindow    AttributionGrade = "isolated-window"
	GradeCohortOnly        AttributionGrade = "cohort-only"
	GradeContaminated      AttributionGrade = "contaminated"
	GradeStale             AttributionGrade = "stale"
	GradeUnavailable       AttributionGrade = "unavailable"
)

type Cause string

const (
	CausePrefixCacheHit Cause = "prefix-cache-hit"
	CauseCacheEviction  Cause = "cache-eviction"
	CauseQueue          Cause = "queue"
	CausePreemption     Cause = "preemption"
)

type CounterName string

const (
	CounterRequests        CounterName = "requests"
	CounterPrefixCacheHits CounterName = "prefix_cache_hits"
	CounterCacheEvictions  CounterName = "cache_evictions"
	CounterQueueEvents     CounterName = "queue_events"
	CounterPreemptions     CounterName = "preemptions"
)

// CounterSample represents a monotonic counter. WrapAt is its modulus; zero
// means a decrease is a reset whose delta cannot be recovered.
type CounterSample struct {
	Value  uint64 `json:"value"`
	WrapAt uint64 `json:"wrap_at,omitempty"`
}

type CounterSet map[CounterName]CounterSample

type CorrelationSource string

const (
	CorrelationRequestLabel CorrelationSource = "request-label"
	CorrelationTrace        CorrelationSource = "trace"
)

type CorrelatedCounterSet struct {
	Source   CorrelationSource `json:"source"`
	Counters CounterSet        `json:"counters"`
}

// MetricsSnapshot is one adapter-neutral scrape. RequestCounters is optional;
// generic Prometheus adapters can supply only Counters.
type MetricsSnapshot struct {
	ServerInstanceID string                          `json:"server_instance_id"`
	ScrapedAt        time.Time                       `json:"scraped_at"`
	Counters         CounterSet                      `json:"counters"`
	RequestCounters  map[string]CorrelatedCounterSet `json:"request_counters,omitempty"`
}

type RequestWindow struct {
	RequestID string    `json:"request_id"`
	StartedAt time.Time `json:"started_at"`
	EndedAt   time.Time `json:"ended_at"`
}

type AttributionInput struct {
	Before        *MetricsSnapshot
	After         *MetricsSnapshot
	Requests      []RequestWindow
	ScrapeFailure string
	ObservedAt    time.Time
	MaxScrapeAge  time.Duration
}

type CounterState string

const (
	CounterOK          CounterState = "ok"
	CounterReset       CounterState = "reset"
	CounterWrapped     CounterState = "wrapped"
	CounterUnavailable CounterState = "unavailable"
)

type CounterDelta struct {
	Value uint64       `json:"value,omitempty"`
	State CounterState `json:"state"`
}

type CorrelatedCounterDelta struct {
	Source   CorrelationSource            `json:"source,omitempty"`
	Counters map[CounterName]CounterDelta `json:"counters"`
}

type MetricsDelta struct {
	ServerInstanceID            string                            `json:"server_instance_id,omitempty"`
	ScrapeStartedAt             time.Time                         `json:"scrape_started_at"`
	ScrapeEndedAt               time.Time                         `json:"scrape_ended_at"`
	Counters                    map[CounterName]CounterDelta      `json:"counters,omitempty"`
	RequestCounters             map[string]CorrelatedCounterDelta `json:"request_counters,omitempty"`
	OverlappingRequestIDs       []string                          `json:"overlapping_request_ids,omitempty"`
	OverlappingObservedRequests int                               `json:"overlapping_observed_requests"`
	BackgroundUnobserved        *uint64                           `json:"background_unobserved_requests,omitempty"`
	CounterResets               []CounterName                     `json:"counter_resets,omitempty"`
	CounterWraps                []CounterName                     `json:"counter_wraps,omitempty"`
	Grade                       AttributionGrade                  `json:"grade"`
	Reason                      string                            `json:"reason"`
}

type RequestAttribution struct {
	RequestID       string            `json:"request_id"`
	Grade           AttributionGrade  `json:"grade"`
	Causes          []Cause           `json:"causes,omitempty"`
	Correlation     CorrelationSource `json:"correlation_source,omitempty"`
	ServerInstance  string            `json:"server_instance_id,omitempty"`
	ScrapeStartedAt time.Time         `json:"scrape_started_at"`
	ScrapeEndedAt   time.Time         `json:"scrape_ended_at"`
	Reason          string            `json:"reason"`
}

type CohortAttribution struct {
	RequestIDs      []string         `json:"request_ids,omitempty"`
	Grade           AttributionGrade `json:"grade"`
	Causes          []Cause          `json:"causes,omitempty"`
	ServerInstance  string           `json:"server_instance_id,omitempty"`
	ScrapeStartedAt time.Time        `json:"scrape_started_at"`
	ScrapeEndedAt   time.Time        `json:"scrape_ended_at"`
	Reason          string           `json:"reason"`
}

type AttributionReport struct {
	Schema   string               `json:"schema"`
	Delta    MetricsDelta         `json:"delta"`
	Requests []RequestAttribution `json:"requests"`
	Cohort   *CohortAttribution   `json:"cohort,omitempty"`
}

// AttributeMetrics joins one scrape interval to observed requests. Shared
// counters reach a request only when the interval is proven isolated.
func AttributeMetrics(in AttributionInput) AttributionReport {
	delta := buildMetricsDelta(in)
	report := AttributionReport{Schema: AttributionSchema, Delta: delta}
	sharedCauses := causesFromCounters(delta.Counters)
	if delta.Grade == GradeStale || delta.Grade == GradeUnavailable {
		sharedCauses = nil
	}

	for _, requestID := range delta.OverlappingRequestIDs {
		row := RequestAttribution{
			RequestID:       requestID,
			Grade:           delta.Grade,
			ServerInstance:  delta.ServerInstanceID,
			ScrapeStartedAt: delta.ScrapeStartedAt,
			ScrapeEndedAt:   delta.ScrapeEndedAt,
			Reason:          delta.Reason,
		}
		if correlated, ok := delta.RequestCounters[requestID]; ok && trustedCorrelation(correlated.Source) && delta.Grade != GradeStale && delta.Grade != GradeUnavailable {
			if causes := causesFromCounters(correlated.Counters); len(causes) > 0 {
				row.Grade = GradeRequestCorrelated
				row.Causes = causes
				row.Correlation = correlated.Source
				row.Reason = "backend request label or trace identifies the evidence owner"
			}
		}
		if row.Grade == GradeIsolatedWindow {
			row.Causes = append([]Cause(nil), sharedCauses...)
		}
		report.Requests = append(report.Requests, row)
	}

	if delta.Grade != GradeIsolatedWindow {
		report.Cohort = &CohortAttribution{
			RequestIDs:      append([]string(nil), delta.OverlappingRequestIDs...),
			Grade:           delta.Grade,
			Causes:          sharedCauses,
			ServerInstance:  delta.ServerInstanceID,
			ScrapeStartedAt: delta.ScrapeStartedAt,
			ScrapeEndedAt:   delta.ScrapeEndedAt,
			Reason:          delta.Reason,
		}
	}
	return report
}

func buildMetricsDelta(in AttributionInput) MetricsDelta {
	delta := MetricsDelta{Grade: GradeUnavailable}
	if in.ScrapeFailure != "" {
		delta.Reason = "metrics scrape failed: " + in.ScrapeFailure
		return delta
	}
	if in.Before == nil || in.After == nil {
		delta.Reason = "both metrics scrapes are required"
		return delta
	}
	delta.ServerInstanceID = in.After.ServerInstanceID
	delta.ScrapeStartedAt = in.Before.ScrapedAt.UTC()
	delta.ScrapeEndedAt = in.After.ScrapedAt.UTC()
	if !in.After.ScrapedAt.After(in.Before.ScrapedAt) {
		delta.Reason = "scrape bounds are invalid"
		return delta
	}
	delta.OverlappingRequestIDs = overlappingRequestIDs(in.Requests, in.Before.ScrapedAt, in.After.ScrapedAt)
	delta.OverlappingObservedRequests = len(delta.OverlappingRequestIDs)
	if in.Before.ServerInstanceID == "" || in.After.ServerInstanceID == "" {
		delta.Reason = "server-instance identity is unavailable"
		return delta
	}
	if in.Before.ServerInstanceID != in.After.ServerInstanceID {
		delta.ServerInstanceID = in.Before.ServerInstanceID + "->" + in.After.ServerInstanceID
		delta.Reason = "server instance changed between scrapes"
		return delta
	}
	delta.Counters, delta.CounterResets, delta.CounterWraps = subtractCounters(in.Before.Counters, in.After.Counters)
	delta.RequestCounters = subtractCorrelatedCounters(in.Before.RequestCounters, in.After.RequestCounters)

	if in.MaxScrapeAge > 0 && !in.ObservedAt.IsZero() && in.ObservedAt.Sub(in.After.ScrapedAt) > in.MaxScrapeAge {
		delta.Grade = GradeStale
		delta.Reason = fmt.Sprintf("latest scrape is older than %s", in.MaxScrapeAge)
		return delta
	}

	requestDelta, requestCountKnown := usableCounter(delta.Counters, CounterRequests)
	if requestCountKnown {
		var background uint64
		if requestDelta > uint64(delta.OverlappingObservedRequests) {
			background = requestDelta - uint64(delta.OverlappingObservedRequests)
		}
		delta.BackgroundUnobserved = &background
		if background > 0 {
			delta.Grade = GradeContaminated
			delta.Reason = fmt.Sprintf("server request counter proves %d unobserved request(s) in the scrape window", background)
			return delta
		}
		if delta.OverlappingObservedRequests == 1 && requestDelta == 1 {
			delta.Grade = GradeIsolatedWindow
			delta.Reason = "one observed request accounts for the complete server request delta"
			return delta
		}
	}

	delta.Grade = GradeCohortOnly
	if delta.OverlappingObservedRequests > 1 {
		delta.Reason = "multiple observed requests overlap the shared counter window"
	} else if !requestCountKnown {
		delta.Reason = "background traffic cannot be excluded without a usable server request counter"
	} else {
		delta.Reason = "the server request counter does not prove an isolated request window"
	}
	return delta
}

func subtractCounters(before, after CounterSet) (map[CounterName]CounterDelta, []CounterName, []CounterName) {
	names := make(map[CounterName]struct{}, len(before)+len(after))
	for name := range before {
		names[name] = struct{}{}
	}
	for name := range after {
		names[name] = struct{}{}
	}
	ordered := make([]CounterName, 0, len(names))
	for name := range names {
		ordered = append(ordered, name)
	}
	sort.Slice(ordered, func(i, j int) bool { return ordered[i] < ordered[j] })

	deltas := make(map[CounterName]CounterDelta, len(ordered))
	var resets, wraps []CounterName
	for _, name := range ordered {
		b, bok := before[name]
		a, aok := after[name]
		if !bok || !aok {
			deltas[name] = CounterDelta{State: CounterUnavailable}
			continue
		}
		if a.Value >= b.Value {
			deltas[name] = CounterDelta{Value: a.Value - b.Value, State: CounterOK}
			continue
		}
		if b.WrapAt > b.Value && a.WrapAt == b.WrapAt && a.Value < b.WrapAt {
			deltas[name] = CounterDelta{Value: (b.WrapAt - b.Value) + a.Value, State: CounterWrapped}
			wraps = append(wraps, name)
			continue
		}
		deltas[name] = CounterDelta{State: CounterReset}
		resets = append(resets, name)
	}
	return deltas, resets, wraps
}

func subtractCorrelatedCounters(before, after map[string]CorrelatedCounterSet) map[string]CorrelatedCounterDelta {
	ids := make(map[string]struct{}, len(before)+len(after))
	for id := range before {
		ids[id] = struct{}{}
	}
	for id := range after {
		ids[id] = struct{}{}
	}
	result := make(map[string]CorrelatedCounterDelta, len(ids))
	for id := range ids {
		b, bok := before[id]
		a, aok := after[id]
		row := CorrelatedCounterDelta{}
		if !bok && aok && a.Source != "" {
			b = CorrelatedCounterSet{Source: a.Source, Counters: zeroCounterSet(a.Counters)}
			bok = true
		}
		if bok && aok && b.Source != "" && b.Source == a.Source {
			row.Source = a.Source
			row.Counters, _, _ = subtractCounters(b.Counters, a.Counters)
		} else {
			row.Counters = map[CounterName]CounterDelta{}
		}
		result[id] = row
	}
	return result
}

func zeroCounterSet(counters CounterSet) CounterSet {
	zero := make(CounterSet, len(counters))
	for name, sample := range counters {
		zero[name] = CounterSample{WrapAt: sample.WrapAt}
	}
	return zero
}

func trustedCorrelation(source CorrelationSource) bool {
	return source == CorrelationRequestLabel || source == CorrelationTrace
}

func overlappingRequestIDs(requests []RequestWindow, started, ended time.Time) []string {
	seen := make(map[string]struct{}, len(requests))
	for _, request := range requests {
		if request.RequestID == "" || request.StartedAt.IsZero() {
			continue
		}
		requestEnd := request.EndedAt
		if requestEnd.IsZero() {
			requestEnd = ended
		}
		if request.StartedAt.Before(ended) && requestEnd.After(started) {
			seen[request.RequestID] = struct{}{}
		}
	}
	ids := make([]string, 0, len(seen))
	for id := range seen {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	return ids
}

func causesFromCounters(counters map[CounterName]CounterDelta) []Cause {
	mapping := []struct {
		counter CounterName
		cause   Cause
	}{
		{CounterPrefixCacheHits, CausePrefixCacheHit},
		{CounterCacheEvictions, CauseCacheEviction},
		{CounterQueueEvents, CauseQueue},
		{CounterPreemptions, CausePreemption},
	}
	var causes []Cause
	for _, candidate := range mapping {
		if value, ok := usableCounter(counters, candidate.counter); ok && value > 0 {
			causes = append(causes, candidate.cause)
		}
	}
	return causes
}

func usableCounter(counters map[CounterName]CounterDelta, name CounterName) (uint64, bool) {
	delta, ok := counters[name]
	return delta.Value, ok && (delta.State == CounterOK || delta.State == CounterWrapped)
}

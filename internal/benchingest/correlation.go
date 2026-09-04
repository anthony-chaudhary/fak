package benchingest

import (
	"fmt"
	"math"
)

// CorrelationReceiptSchema is the canonical schema version tag for benchmark telemetry correlation receipts.
const CorrelationReceiptSchema = "fak.benchmark-telemetry-correlation/v1"

// AccountingBucket identifies the terminal classification of an ingested telemetry sample.
type AccountingBucket = string

// AccountingBucket string constants.
const (
	BucketAccepted     AccountingBucket = "accepted"
	BucketMalformed    AccountingBucket = "malformed"
	BucketDuplicate    AccountingBucket = "duplicate"
	BucketMissing      AccountingBucket = "missing"
	BucketUnmatched    AccountingBucket = "unmatched"
	BucketAmbiguous    AccountingBucket = "ambiguous"
	BucketSkewRejected AccountingBucket = "skew_rejected"
	BucketReordered    AccountingBucket = "reordered"
)

// Disposition constants.
const (
	DispositionAdmissible = "admissible"
	DispositionRefused    = "refused"
)

// Closed RefusalReason tokens.
const (
	ReasonExcessiveAmbiguity    = "EXCESSIVE_AMBIGUITY"
	ReasonSkewThresholdExceeded = "SKEW_THRESHOLD_EXCEEDED"
	ReasonMalformedTelemetry    = "MALFORMED_TELEMETRY"
	ReasonDuplicateTelemetry    = "DUPLICATE_TELEMETRY"
	ReasonDuplicateEvents       = "DUPLICATE_EVENTS"
)

// MatchRule constants.
const (
	MatchRuleExactID                   = "exact_id"
	MatchRuleProximity                 = "proximity"
	MatchRuleProximityTieBreakEarliest = "proximity_tiebreak_earliest"
	MatchRuleProximityTieBreakLatest   = "proximity_tiebreak_latest"
)

// TieBreakRule constants.
const (
	TieBreakEarliest = "earliest"
	TieBreakLatest   = "latest"
)

// DefaultMaxClockSkewMS is the default clock skew tolerance in milliseconds when not specified.
const DefaultMaxClockSkewMS int64 = 1000

// WorkloadEvent captures a discrete workload execution boundary.
type WorkloadEvent struct {
	ID        string            `json:"id"`
	Timestamp int64             `json:"timestamp"` // nanoseconds or milliseconds
	Name      string            `json:"name"`
	Tags      map[string]string `json:"tags,omitempty"`
}

// TelemetrySample captures a timestamped single-metric observation.
type TelemetrySample struct {
	ID        string  `json:"id"`
	EventID   string  `json:"event_id,omitempty"` // optional key
	Timestamp int64   `json:"timestamp"`
	Metric    string  `json:"metric"`
	Value     float64 `json:"value"`
}

// CorrelationOptions configures correlation matching rules and thresholds.
type CorrelationOptions struct {
	MaxClockSkewMS      int64  `json:"max_clock_skew_ms"`
	TieBreakRule        string `json:"tie_break_rule"` // "earliest", "latest"
	PreferIDMatching    bool   `json:"prefer_id_matching"`
	TimestampUnit       string `json:"timestamp_unit,omitempty"` // "ms", "ns", "auto" (default)
	MaxMalformedAllowed int    `json:"max_malformed_allowed,omitempty"`
	MaxSkewAllowed      int    `json:"max_skew_allowed,omitempty"`
	MaxAmbiguousAllowed int    `json:"max_ambiguous_allowed,omitempty"`
	RefuseOnDuplicate   bool   `json:"refuse_on_duplicate,omitempty"`
}

// CorrelationPair records an accepted correlation between a workload event and a telemetry sample.
type CorrelationPair struct {
	EventID          string `json:"event_id"`
	SampleID         string `json:"sample_id"`
	ProximityDeltaMS int64  `json:"proximity_delta_ms"`
	MatchRule        string `json:"match_rule"`
}

// CorrelationReceipt provides a tamper-evident, conservation-checked record of workload telemetry correlation.
type CorrelationReceipt struct {
	Schema            string            `json:"schema"`
	TotalEvents       int               `json:"total_events"`
	TotalSamples      int               `json:"total_samples"`
	AcceptedCount     int               `json:"accepted_count"`
	MalformedCount    int               `json:"malformed_count"`
	DuplicateCount    int               `json:"duplicate_count"`
	MissingCount      int               `json:"missing_count"`
	UnmatchedCount    int               `json:"unmatched_count"`
	AmbiguousCount    int               `json:"ambiguous_count"`
	SkewRejectedCount int               `json:"skew_rejected_count"`
	ReorderedCount    int               `json:"reordered_count"`
	CorrelationPairs  []CorrelationPair `json:"correlation_pairs"`
	Disposition       string            `json:"disposition"`
	RefusalReason     string            `json:"refusal_reason,omitempty"`
}

// ConservationSatisfied asserts that every sample is accounted for in exactly one terminal bucket.
//
// Invariant:
// TotalSamples == AcceptedCount + MalformedCount + DuplicateCount + UnmatchedCount + AmbiguousCount + SkewRejectedCount + ReorderedCount.
func (r *CorrelationReceipt) ConservationSatisfied() bool {
	terminalSum := r.AcceptedCount + r.MalformedCount + r.DuplicateCount + r.UnmatchedCount +
		r.AmbiguousCount + r.SkewRejectedCount + r.ReorderedCount
	return r.TotalSamples == terminalSum
}

// Validate checks receipt schema and exact conservation accounting.
func (r *CorrelationReceipt) Validate() error {
	if r.Schema != CorrelationReceiptSchema {
		return fmt.Errorf("invalid schema: got %q, want %q", r.Schema, CorrelationReceiptSchema)
	}
	terminalSum := r.AcceptedCount + r.MalformedCount + r.DuplicateCount + r.UnmatchedCount +
		r.AmbiguousCount + r.SkewRejectedCount + r.ReorderedCount
	if r.TotalSamples != terminalSum {
		return fmt.Errorf("exact conservation violated: TotalSamples=%d != terminalSum=%d (accepted=%d malformed=%d dup=%d unmatched=%d ambiguous=%d skew=%d reordered=%d)",
			r.TotalSamples, terminalSum, r.AcceptedCount, r.MalformedCount, r.DuplicateCount,
			r.UnmatchedCount, r.AmbiguousCount, r.SkewRejectedCount, r.ReorderedCount)
	}
	return nil
}

func normalizeTimestamp(ts int64, unit string) int64 {
	if unit == "ns" {
		return ts / 1_000_000
	}
	if unit == "ms" {
		return ts
	}
	// "auto" detection:
	// Unix timestamps in nanoseconds exceed 10^14 (e.g. 1.7e18).
	// Unix timestamps in milliseconds are ~1.7e12.
	if ts > 100_000_000_000_000 {
		return ts / 1_000_000
	}
	return ts
}

func absInt64(v int64) int64 {
	if v < 0 {
		return -v
	}
	return v
}

func indexWorkloadEvents(events []WorkloadEvent) ([]WorkloadEvent, map[string]WorkloadEvent, bool, bool) {
	hasMalformedEvent := false
	hasDuplicateEvent := false
	eventIDSeen := make(map[string]bool, len(events))
	validEventsByID := make(map[string]WorkloadEvent, len(events))
	validEvents := make([]WorkloadEvent, 0, len(events))

	for _, ev := range events {
		if ev.ID == "" || ev.Timestamp <= 0 || ev.Name == "" {
			hasMalformedEvent = true
			continue
		}
		if eventIDSeen[ev.ID] {
			hasDuplicateEvent = true
			continue
		}
		eventIDSeen[ev.ID] = true
		validEventsByID[ev.ID] = ev
		validEvents = append(validEvents, ev)
	}
	return validEvents, validEventsByID, hasMalformedEvent, hasDuplicateEvent
}

func matchProximityEvent(sMS int64, validEvents []WorkloadEvent, maxSkew int64, opts CorrelationOptions) (*WorkloadEvent, string, int64, AccountingBucket) {
	if len(validEvents) == 0 {
		return nil, "", 0, BucketUnmatched
	}

	var minDelta int64 = math.MaxInt64
	for _, ev := range validEvents {
		evMS := normalizeTimestamp(ev.Timestamp, opts.TimestampUnit)
		d := absInt64(sMS - evMS)
		if d < minDelta {
			minDelta = d
		}
	}

	if minDelta > maxSkew {
		return nil, "", minDelta, BucketSkewRejected
	}

	var nearestEvents []WorkloadEvent
	for _, ev := range validEvents {
		evMS := normalizeTimestamp(ev.Timestamp, opts.TimestampUnit)
		d := absInt64(sMS - evMS)
		if d == minDelta {
			nearestEvents = append(nearestEvents, ev)
		}
	}

	if len(nearestEvents) == 1 {
		return &nearestEvents[0], MatchRuleProximity, minDelta, BucketAccepted
	}

	// Multiple equidistant candidates: apply TieBreakRule
	if opts.TieBreakRule == TieBreakEarliest {
		var earliestEv WorkloadEvent
		var earliestTS int64 = math.MaxInt64
		tieCount := 0
		for _, ev := range nearestEvents {
			evMS := normalizeTimestamp(ev.Timestamp, opts.TimestampUnit)
			if evMS < earliestTS {
				earliestTS = evMS
				earliestEv = ev
				tieCount = 1
			} else if evMS == earliestTS {
				tieCount++
			}
		}
		if tieCount == 1 {
			return &earliestEv, MatchRuleProximityTieBreakEarliest, minDelta, BucketAccepted
		}
	} else if opts.TieBreakRule == TieBreakLatest {
		var latestEv WorkloadEvent
		var latestTS int64 = math.MinInt64
		tieCount := 0
		for _, ev := range nearestEvents {
			evMS := normalizeTimestamp(ev.Timestamp, opts.TimestampUnit)
			if evMS > latestTS {
				latestTS = evMS
				latestEv = ev
				tieCount = 1
			} else if evMS == latestTS {
				tieCount++
			}
		}
		if tieCount == 1 {
			return &latestEv, MatchRuleProximityTieBreakLatest, minDelta, BucketAccepted
		}
	}

	return nil, "", minDelta, BucketAmbiguous
}

func enforceDisposition(rcpt *CorrelationReceipt, opts CorrelationOptions, hasMalformedEvent, hasDuplicateEvent bool) {
	if hasMalformedEvent || rcpt.MalformedCount > opts.MaxMalformedAllowed {
		rcpt.Disposition = DispositionRefused
		rcpt.RefusalReason = ReasonMalformedTelemetry
	} else if rcpt.SkewRejectedCount > opts.MaxSkewAllowed {
		rcpt.Disposition = DispositionRefused
		rcpt.RefusalReason = ReasonSkewThresholdExceeded
	} else if rcpt.AmbiguousCount > opts.MaxAmbiguousAllowed {
		rcpt.Disposition = DispositionRefused
		rcpt.RefusalReason = ReasonExcessiveAmbiguity
	} else if opts.RefuseOnDuplicate && (hasDuplicateEvent || rcpt.DuplicateCount > 0) {
		rcpt.Disposition = DispositionRefused
		rcpt.RefusalReason = ReasonDuplicateTelemetry
	}
}

// Correlate correlates workload events with external telemetry samples, enforcing exact conservation accounting
// and evaluating claim eligibility against closed refusal criteria.
func Correlate(events []WorkloadEvent, samples []TelemetrySample, opts CorrelationOptions) (*CorrelationReceipt, error) {
	maxSkew := opts.MaxClockSkewMS
	if maxSkew <= 0 {
		maxSkew = DefaultMaxClockSkewMS
	}

	rcpt := &CorrelationReceipt{
		Schema:           CorrelationReceiptSchema,
		TotalEvents:      len(events),
		TotalSamples:     len(samples),
		CorrelationPairs: make([]CorrelationPair, 0),
		Disposition:      DispositionAdmissible,
	}

	// 1. Validate and index events.
	validEvents, validEventsByID, hasMalformedEvent, hasDuplicateEvent := indexWorkloadEvents(events)

	// 2. Classify each sample into exactly one terminal bucket.
	sampleIDSeen := make(map[string]bool)
	var maxSampleTS int64 = math.MinInt64
	eventMatchedCount := make(map[string]int)

	for _, s := range samples {
		// Terminal Bucket: Malformed
		if s.ID == "" || s.Timestamp <= 0 || s.Metric == "" || math.IsNaN(s.Value) || math.IsInf(s.Value, 0) {
			rcpt.MalformedCount++
			continue
		}

		// Terminal Bucket: Duplicate
		if sampleIDSeen[s.ID] {
			rcpt.DuplicateCount++
			continue
		}
		sampleIDSeen[s.ID] = true

		sMS := normalizeTimestamp(s.Timestamp, opts.TimestampUnit)

		// Terminal Bucket: Reordered (non-monotonic timestamp arrival in telemetry stream)
		if maxSampleTS != math.MinInt64 && sMS < maxSampleTS {
			rcpt.ReorderedCount++
			continue
		}
		if sMS > maxSampleTS {
			maxSampleTS = sMS
		}

		// Matching candidate selection:
		// Case A: EventID provided (exact key match)
		if s.EventID != "" {
			ev, found := validEventsByID[s.EventID]
			if !found {
				rcpt.UnmatchedCount++
				continue
			}
			evMS := normalizeTimestamp(ev.Timestamp, opts.TimestampUnit)
			delta := absInt64(sMS - evMS)
			if delta > maxSkew {
				rcpt.SkewRejectedCount++
				continue
			}
			rcpt.AcceptedCount++
			rcpt.CorrelationPairs = append(rcpt.CorrelationPairs, CorrelationPair{
				EventID:          ev.ID,
				SampleID:         s.ID,
				ProximityDeltaMS: delta,
				MatchRule:        MatchRuleExactID,
			})
			eventMatchedCount[ev.ID]++
			continue
		}

		// Case B: Missing EventID key -> Fallback to timestamp proximity
		ev, matchRule, delta, bucket := matchProximityEvent(sMS, validEvents, maxSkew, opts)
		switch bucket {
		case BucketUnmatched:
			rcpt.UnmatchedCount++
		case BucketSkewRejected:
			rcpt.SkewRejectedCount++
		case BucketAccepted:
			rcpt.AcceptedCount++
			rcpt.CorrelationPairs = append(rcpt.CorrelationPairs, CorrelationPair{
				EventID:          ev.ID,
				SampleID:         s.ID,
				ProximityDeltaMS: delta,
				MatchRule:        matchRule,
			})
			eventMatchedCount[ev.ID]++
		default:
			rcpt.AmbiguousCount++
		}
	}

	// 3. Compute MissingCount: valid events with zero accepted samples
	for _, ev := range validEvents {
		if eventMatchedCount[ev.ID] == 0 {
			rcpt.MissingCount++
		}
	}

	// 4. Enforce claim eligibility & disposition
	enforceDisposition(rcpt, opts, hasMalformedEvent, hasDuplicateEvent)

	// 5. Invariant assertion before return
	if err := rcpt.Validate(); err != nil {
		return nil, err
	}

	return rcpt, nil
}

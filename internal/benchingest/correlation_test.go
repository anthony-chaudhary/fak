package benchingest

import (
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/benchsnapshot"
)

func TestCorrelate_ExactKeyMatches(t *testing.T) {
	events := []WorkloadEvent{
		{ID: "ev-1", Timestamp: 1000, Name: "prefill"},
		{ID: "ev-2", Timestamp: 2000, Name: "decode"},
	}

	samples := []TelemetrySample{
		{ID: "s-1", EventID: "ev-1", Timestamp: 1010, Metric: "power_w", Value: 120.5},
		{ID: "s-2", EventID: "ev-2", Timestamp: 2020, Metric: "gpu_util", Value: 95.0},
	}

	opts := CorrelationOptions{
		MaxClockSkewMS:   50,
		PreferIDMatching: true,
	}

	rcpt, err := Correlate(events, samples, opts)
	if err != nil {
		t.Fatalf("Correlate failed: %v", err)
	}

	if rcpt.TotalEvents != 2 {
		t.Errorf("expected TotalEvents=2, got %d", rcpt.TotalEvents)
	}
	if rcpt.TotalSamples != 2 {
		t.Errorf("expected TotalSamples=2, got %d", rcpt.TotalSamples)
	}
	if rcpt.AcceptedCount != 2 {
		t.Errorf("expected AcceptedCount=2, got %d", rcpt.AcceptedCount)
	}
	if rcpt.MalformedCount != 0 || rcpt.DuplicateCount != 0 || rcpt.UnmatchedCount != 0 ||
		rcpt.AmbiguousCount != 0 || rcpt.SkewRejectedCount != 0 || rcpt.ReorderedCount != 0 {
		t.Errorf("unexpected non-zero bucket counts: %+v", rcpt)
	}
	if rcpt.MissingCount != 0 {
		t.Errorf("expected MissingCount=0, got %d", rcpt.MissingCount)
	}
	if rcpt.Disposition != DispositionAdmissible {
		t.Errorf("expected disposition admissible, got %s", rcpt.Disposition)
	}
	if len(rcpt.CorrelationPairs) != 2 {
		t.Fatalf("expected 2 correlation pairs, got %d", len(rcpt.CorrelationPairs))
	}

	p0 := rcpt.CorrelationPairs[0]
	if p0.EventID != "ev-1" || p0.SampleID != "s-1" || p0.ProximityDeltaMS != 10 || p0.MatchRule != MatchRuleExactID {
		t.Errorf("unexpected pair 0: %+v", p0)
	}

	p1 := rcpt.CorrelationPairs[1]
	if p1.EventID != "ev-2" || p1.SampleID != "s-2" || p1.ProximityDeltaMS != 20 || p1.MatchRule != MatchRuleExactID {
		t.Errorf("unexpected pair 1: %+v", p1)
	}

	if !rcpt.ConservationSatisfied() {
		t.Errorf("exact conservation violated")
	}
}

func TestCorrelate_MissingKeys_FallbackToTimestampProximity(t *testing.T) {
	events := []WorkloadEvent{
		{ID: "ev-prefill", Timestamp: 1000, Name: "prefill"},
		{ID: "ev-decode", Timestamp: 2000, Name: "decode"},
	}

	samples := []TelemetrySample{
		{ID: "s-1", EventID: "", Timestamp: 1025, Metric: "power_w", Value: 110.0},
		{ID: "s-2", EventID: "", Timestamp: 1980, Metric: "power_w", Value: 90.0},
	}

	opts := CorrelationOptions{
		MaxClockSkewMS: 100,
	}

	rcpt, err := Correlate(events, samples, opts)
	if err != nil {
		t.Fatalf("Correlate failed: %v", err)
	}

	if rcpt.AcceptedCount != 2 {
		t.Errorf("expected AcceptedCount=2, got %d", rcpt.AcceptedCount)
	}
	if len(rcpt.CorrelationPairs) != 2 {
		t.Fatalf("expected 2 correlation pairs, got %d", len(rcpt.CorrelationPairs))
	}

	if rcpt.CorrelationPairs[0].EventID != "ev-prefill" || rcpt.CorrelationPairs[0].MatchRule != MatchRuleProximity || rcpt.CorrelationPairs[0].ProximityDeltaMS != 25 {
		t.Errorf("unexpected pair 0: %+v", rcpt.CorrelationPairs[0])
	}
	if rcpt.CorrelationPairs[1].EventID != "ev-decode" || rcpt.CorrelationPairs[1].MatchRule != MatchRuleProximity || rcpt.CorrelationPairs[1].ProximityDeltaMS != 20 {
		t.Errorf("unexpected pair 1: %+v", rcpt.CorrelationPairs[1])
	}

	if !rcpt.ConservationSatisfied() {
		t.Errorf("exact conservation violated")
	}
}

func TestCorrelate_DuplicateEventsAndSamples(t *testing.T) {
	events := []WorkloadEvent{
		{ID: "ev-1", Timestamp: 1000, Name: "event_1"},
		{ID: "ev-1", Timestamp: 1050, Name: "event_1_dup"},
		{ID: "ev-2", Timestamp: 2000, Name: "event_2"},
	}

	samples := []TelemetrySample{
		{ID: "s-1", Timestamp: 1010, Metric: "cpu", Value: 50.0},
		{ID: "s-1", Timestamp: 1020, Metric: "cpu", Value: 55.0}, // Duplicate sample ID
		{ID: "s-2", Timestamp: 2010, Metric: "cpu", Value: 60.0},
	}

	opts := CorrelationOptions{
		MaxClockSkewMS: 100,
	}

	rcpt, err := Correlate(events, samples, opts)
	if err != nil {
		t.Fatalf("Correlate failed: %v", err)
	}

	if rcpt.TotalEvents != 3 {
		t.Errorf("expected TotalEvents=3, got %d", rcpt.TotalEvents)
	}
	if rcpt.TotalSamples != 3 {
		t.Errorf("expected TotalSamples=3, got %d", rcpt.TotalSamples)
	}
	if rcpt.DuplicateCount != 1 {
		t.Errorf("expected DuplicateCount=1, got %d", rcpt.DuplicateCount)
	}
	if rcpt.AcceptedCount != 2 {
		t.Errorf("expected AcceptedCount=2, got %d", rcpt.AcceptedCount)
	}

	if !rcpt.ConservationSatisfied() {
		t.Errorf("exact conservation violated: %+v", rcpt)
	}
}

func TestCorrelate_MalformedRows(t *testing.T) {
	events := []WorkloadEvent{
		{ID: "ev-1", Timestamp: 1000, Name: "event_1"},
	}

	samples := []TelemetrySample{
		{ID: "s-ok", Timestamp: 1000, Metric: "val", Value: 1.0},
		{ID: "", Timestamp: 1000, Metric: "val", Value: 1.0},              // Missing ID
		{ID: "s-badts", Timestamp: -10, Metric: "val", Value: 1.0},        // Non-positive timestamp
		{ID: "s-nometric", Timestamp: 1000, Metric: "", Value: 1.0},       // Empty metric
		{ID: "s-nan", Timestamp: 1000, Metric: "val", Value: math.NaN()},  // NaN value
		{ID: "s-inf", Timestamp: 1000, Metric: "val", Value: math.Inf(1)}, // Inf value
	}

	opts := CorrelationOptions{
		MaxClockSkewMS: 100,
	}

	rcpt, err := Correlate(events, samples, opts)
	if err != nil {
		t.Fatalf("Correlate failed: %v", err)
	}

	if rcpt.TotalSamples != 6 {
		t.Errorf("expected TotalSamples=6, got %d", rcpt.TotalSamples)
	}
	if rcpt.AcceptedCount != 1 {
		t.Errorf("expected AcceptedCount=1, got %d", rcpt.AcceptedCount)
	}
	if rcpt.MalformedCount != 5 {
		t.Errorf("expected MalformedCount=5, got %d", rcpt.MalformedCount)
	}
	if rcpt.Disposition != DispositionRefused {
		t.Errorf("expected disposition refused, got %s", rcpt.Disposition)
	}
	if rcpt.RefusalReason != ReasonMalformedTelemetry {
		t.Errorf("expected refusal reason %s, got %s", ReasonMalformedTelemetry, rcpt.RefusalReason)
	}

	if !rcpt.ConservationSatisfied() {
		t.Errorf("exact conservation violated: %+v", rcpt)
	}

	// Test malformed event
	malformedEvents := []WorkloadEvent{
		{ID: "", Timestamp: 1000, Name: "missing_id"},
	}
	rcpt2, err := Correlate(malformedEvents, samples[:1], opts)
	if err != nil {
		t.Fatalf("Correlate failed: %v", err)
	}
	if rcpt2.Disposition != DispositionRefused || rcpt2.RefusalReason != ReasonMalformedTelemetry {
		t.Errorf("expected malformed event refusal, got %+v", rcpt2)
	}
}

func TestCorrelate_ClockSkewExceedingMaxClockSkewMS(t *testing.T) {
	events := []WorkloadEvent{
		{ID: "ev-1", Timestamp: 1000, Name: "event_1"},
	}

	samples := []TelemetrySample{
		{ID: "s-exact-skewed", EventID: "ev-1", Timestamp: 1500, Metric: "power", Value: 10.0},
		{ID: "s-prox-skewed", EventID: "", Timestamp: 1600, Metric: "power", Value: 20.0},
	}

	opts := CorrelationOptions{
		MaxClockSkewMS: 100,
	}

	rcpt, err := Correlate(events, samples, opts)
	if err != nil {
		t.Fatalf("Correlate failed: %v", err)
	}

	if rcpt.AcceptedCount != 0 {
		t.Errorf("expected AcceptedCount=0, got %d", rcpt.AcceptedCount)
	}
	if rcpt.SkewRejectedCount != 2 {
		t.Errorf("expected SkewRejectedCount=2, got %d", rcpt.SkewRejectedCount)
	}
	if rcpt.Disposition != DispositionRefused {
		t.Errorf("expected disposition refused, got %s", rcpt.Disposition)
	}
	if rcpt.RefusalReason != ReasonSkewThresholdExceeded {
		t.Errorf("expected refusal reason %s, got %s", ReasonSkewThresholdExceeded, rcpt.RefusalReason)
	}

	if !rcpt.ConservationSatisfied() {
		t.Errorf("exact conservation violated: %+v", rcpt)
	}
}

func TestCorrelate_OutOfOrderInput(t *testing.T) {
	events := []WorkloadEvent{
		{ID: "ev-1", Timestamp: 1000, Name: "event_1"},
	}

	samples := []TelemetrySample{
		{ID: "s-1", Timestamp: 2000, Metric: "power", Value: 10.0},
		{ID: "s-2", Timestamp: 1000, Metric: "power", Value: 20.0}, // Out of order: 1000 < 2000
		{ID: "s-3", Timestamp: 2500, Metric: "power", Value: 30.0}, // In order: 2500 > 2000
	}

	opts := CorrelationOptions{
		MaxClockSkewMS: 2000,
	}

	rcpt, err := Correlate(events, samples, opts)
	if err != nil {
		t.Fatalf("Correlate failed: %v", err)
	}

	if rcpt.TotalSamples != 3 {
		t.Errorf("expected TotalSamples=3, got %d", rcpt.TotalSamples)
	}
	if rcpt.ReorderedCount != 1 {
		t.Errorf("expected ReorderedCount=1, got %d", rcpt.ReorderedCount)
	}
	if rcpt.AcceptedCount != 2 {
		t.Errorf("expected AcceptedCount=2, got %d", rcpt.AcceptedCount)
	}

	if !rcpt.ConservationSatisfied() {
		t.Errorf("exact conservation violated: %+v", rcpt)
	}
}

func TestCorrelate_AmbiguousNearestCandidates(t *testing.T) {
	events := []WorkloadEvent{
		{ID: "ev-early", Timestamp: 1000, Name: "early"},
		{ID: "ev-late", Timestamp: 2000, Name: "late"},
	}

	sample := TelemetrySample{
		ID:        "s-mid",
		EventID:   "",
		Timestamp: 1500, // Equidistant (delta 500) to both events
		Metric:    "power",
		Value:     42.0,
	}

	t.Run("AmbiguousWithoutTieBreak", func(t *testing.T) {
		opts := CorrelationOptions{
			MaxClockSkewMS: 1000,
			TieBreakRule:   "",
		}

		rcpt, err := Correlate(events, []TelemetrySample{sample}, opts)
		if err != nil {
			t.Fatalf("Correlate failed: %v", err)
		}

		if rcpt.AmbiguousCount != 1 {
			t.Errorf("expected AmbiguousCount=1, got %d", rcpt.AmbiguousCount)
		}
		if rcpt.AcceptedCount != 0 {
			t.Errorf("expected AcceptedCount=0, got %d", rcpt.AcceptedCount)
		}
		if rcpt.Disposition != DispositionRefused {
			t.Errorf("expected disposition refused, got %s", rcpt.Disposition)
		}
		if rcpt.RefusalReason != ReasonExcessiveAmbiguity {
			t.Errorf("expected refusal reason %s, got %s", ReasonExcessiveAmbiguity, rcpt.RefusalReason)
		}
		if !rcpt.ConservationSatisfied() {
			t.Errorf("exact conservation violated: %+v", rcpt)
		}
	})

	t.Run("TieBreakEarliest", func(t *testing.T) {
		opts := CorrelationOptions{
			MaxClockSkewMS: 1000,
			TieBreakRule:   TieBreakEarliest,
		}

		rcpt, err := Correlate(events, []TelemetrySample{sample}, opts)
		if err != nil {
			t.Fatalf("Correlate failed: %v", err)
		}

		if rcpt.AcceptedCount != 1 {
			t.Fatalf("expected AcceptedCount=1, got %d", rcpt.AcceptedCount)
		}
		if rcpt.AmbiguousCount != 0 {
			t.Errorf("expected AmbiguousCount=0, got %d", rcpt.AmbiguousCount)
		}
		if rcpt.CorrelationPairs[0].EventID != "ev-early" {
			t.Errorf("expected match ev-early, got %s", rcpt.CorrelationPairs[0].EventID)
		}
		if rcpt.CorrelationPairs[0].MatchRule != MatchRuleProximityTieBreakEarliest {
			t.Errorf("expected match rule %s, got %s", MatchRuleProximityTieBreakEarliest, rcpt.CorrelationPairs[0].MatchRule)
		}
		if !rcpt.ConservationSatisfied() {
			t.Errorf("exact conservation violated: %+v", rcpt)
		}
	})

	t.Run("TieBreakLatest", func(t *testing.T) {
		opts := CorrelationOptions{
			MaxClockSkewMS: 1000,
			TieBreakRule:   TieBreakLatest,
		}

		rcpt, err := Correlate(events, []TelemetrySample{sample}, opts)
		if err != nil {
			t.Fatalf("Correlate failed: %v", err)
		}

		if rcpt.AcceptedCount != 1 {
			t.Fatalf("expected AcceptedCount=1, got %d", rcpt.AcceptedCount)
		}
		if rcpt.AmbiguousCount != 0 {
			t.Errorf("expected AmbiguousCount=0, got %d", rcpt.AmbiguousCount)
		}
		if rcpt.CorrelationPairs[0].EventID != "ev-late" {
			t.Errorf("expected match ev-late, got %s", rcpt.CorrelationPairs[0].EventID)
		}
		if rcpt.CorrelationPairs[0].MatchRule != MatchRuleProximityTieBreakLatest {
			t.Errorf("expected match rule %s, got %s", MatchRuleProximityTieBreakLatest, rcpt.CorrelationPairs[0].MatchRule)
		}
		if !rcpt.ConservationSatisfied() {
			t.Errorf("exact conservation violated: %+v", rcpt)
		}
	})
}

func TestCorrelate_InputRowConservationAssertion(t *testing.T) {
	events := []WorkloadEvent{
		{ID: "ev-1", Timestamp: 1000, Name: "ev1"},
		{ID: "ev-2", Timestamp: 2000, Name: "ev2"},
		{ID: "ev-3", Timestamp: 3000, Name: "ev3"},
		{ID: "ev-unmatched", Timestamp: 9000, Name: "ev_unmatched"}, // will have missing telemetry
	}

	samples := []TelemetrySample{
		// 1. Accepted by exact key
		{ID: "s-exact", EventID: "ev-1", Timestamp: 1010, Metric: "m1", Value: 1.0},
		// 2. Accepted by proximity
		{ID: "s-prox", EventID: "", Timestamp: 1990, Metric: "m2", Value: 2.0},
		// 3. Malformed (empty ID)
		{ID: "", EventID: "", Timestamp: 2000, Metric: "m3", Value: 3.0},
		// 4. Malformed (NaN)
		{ID: "s-nan", EventID: "", Timestamp: 2005, Metric: "m4", Value: math.NaN()},
		// 5. Duplicate of s-exact
		{ID: "s-exact", EventID: "ev-1", Timestamp: 2010, Metric: "m1", Value: 1.0},
		// 6. Out-of-order arrival
		{ID: "s-reorder", EventID: "", Timestamp: 500, Metric: "m5", Value: 5.0},
		// 7. Skew rejected (delta 5000 > maxSkew 100)
		{ID: "s-skew", EventID: "ev-1", Timestamp: 6000, Metric: "m6", Value: 6.0},
		// 8. Ambiguous (midpoint between ev-2 at 2000 and ev-3 at 3000 -> ts 2500)
		// but wait: sample timestamp must be >= 6000 to avoid being marked reordered,
		// or placed earlier. Let's create an ambiguous pair further along or test with appropriate ts:
	}

	// Re-sequence timestamps so they don't accidentally trip reordered:
	samples = []TelemetrySample{
		{ID: "s-exact", EventID: "ev-1", Timestamp: 1010, Metric: "m1", Value: 1.0},                // Accepted (exact)
		{ID: "s-exact", EventID: "ev-1", Timestamp: 1015, Metric: "m1", Value: 1.0},                // Duplicate
		{ID: "s-prox", EventID: "", Timestamp: 1990, Metric: "m2", Value: 2.0},                     // Accepted (prox)
		{ID: "s-reorder", EventID: "", Timestamp: 500, Metric: "m5", Value: 5.0},                   // Reordered
		{ID: "", EventID: "", Timestamp: 2000, Metric: "m3", Value: 3.0},                           // Malformed
		{ID: "s-nan", EventID: "", Timestamp: 2005, Metric: "m4", Value: math.NaN()},               // Malformed
		{ID: "s-ambig", EventID: "", Timestamp: 2500, Metric: "m_ambig", Value: 4.0},               // Ambiguous (delta 500 to ev-2 & ev-3)
		{ID: "s-skew", EventID: "ev-1", Timestamp: 6000, Metric: "m6", Value: 6.0},                 // Skew rejected (delta 5000 > 500)
		{ID: "s-unmatched-key", EventID: "nonexistent", Timestamp: 7000, Metric: "m7", Value: 7.0}, // Unmatched
	}

	opts := CorrelationOptions{
		MaxClockSkewMS:      500,
		TieBreakRule:        "", // Ambiguity not resolved
		MaxMalformedAllowed: 10,
		MaxSkewAllowed:      10,
		MaxAmbiguousAllowed: 10,
	}

	rcpt, err := Correlate(events, samples, opts)
	if err != nil {
		t.Fatalf("Correlate failed: %v", err)
	}

	if rcpt.TotalSamples != len(samples) {
		t.Errorf("expected TotalSamples=%d, got %d", len(samples), rcpt.TotalSamples)
	}
	if rcpt.AcceptedCount != 2 {
		t.Errorf("expected AcceptedCount=2, got %d", rcpt.AcceptedCount)
	}
	if rcpt.DuplicateCount != 1 {
		t.Errorf("expected DuplicateCount=1, got %d", rcpt.DuplicateCount)
	}
	if rcpt.ReorderedCount != 1 {
		t.Errorf("expected ReorderedCount=1, got %d", rcpt.ReorderedCount)
	}
	if rcpt.MalformedCount != 2 {
		t.Errorf("expected MalformedCount=2, got %d", rcpt.MalformedCount)
	}
	if rcpt.AmbiguousCount != 1 {
		t.Errorf("expected AmbiguousCount=1, got %d", rcpt.AmbiguousCount)
	}
	if rcpt.SkewRejectedCount != 1 {
		t.Errorf("expected SkewRejectedCount=1, got %d", rcpt.SkewRejectedCount)
	}
	if rcpt.UnmatchedCount != 1 {
		t.Errorf("expected UnmatchedCount=1, got %d", rcpt.UnmatchedCount)
	}

	// Exact row conservation assertion
	expectedTotal := rcpt.AcceptedCount + rcpt.MalformedCount + rcpt.DuplicateCount +
		rcpt.UnmatchedCount + rcpt.AmbiguousCount + rcpt.SkewRejectedCount + rcpt.ReorderedCount

	if rcpt.TotalSamples != expectedTotal {
		t.Errorf("exact conservation failed: TotalSamples (%d) != sum of buckets (%d)", rcpt.TotalSamples, expectedTotal)
	}

	if !rcpt.ConservationSatisfied() {
		t.Errorf("ConservationSatisfied returned false")
	}

	if err := rcpt.Validate(); err != nil {
		t.Errorf("Validate failed: %v", err)
	}

	if rcpt.MissingCount != 2 { // ev-3 and ev-unmatched had no accepted samples
		t.Errorf("expected MissingCount=2, got %d", rcpt.MissingCount)
	}
}

func TestCorrelate_NanosecondTimestamps(t *testing.T) {
	events := []WorkloadEvent{
		{ID: "ev-ns", Timestamp: 1_700_000_000_000_000_000, Name: "ns_event"}, // Unix nano
	}

	samples := []TelemetrySample{
		{ID: "s-ns", EventID: "ev-ns", Timestamp: 1_700_000_000_025_000_000, Metric: "power", Value: 50.0}, // +25ms
	}

	opts := CorrelationOptions{
		MaxClockSkewMS: 100,
	}

	rcpt, err := Correlate(events, samples, opts)
	if err != nil {
		t.Fatalf("Correlate failed: %v", err)
	}

	if rcpt.AcceptedCount != 1 {
		t.Fatalf("expected AcceptedCount=1, got %d", rcpt.AcceptedCount)
	}
	if rcpt.CorrelationPairs[0].ProximityDeltaMS != 25 {
		t.Errorf("expected ProximityDeltaMS=25, got %d", rcpt.CorrelationPairs[0].ProximityDeltaMS)
	}
}

func TestSnapshotPublisher_SequenceAddressedReceipts(t *testing.T) {
	tmpDir := t.TempDir()
	writer, err := benchsnapshot.NewWriter(tmpDir, "run-alpha", "benchmark")
	if err != nil {
		t.Fatalf("NewWriter failed: %v", err)
	}
	publisher := NewSnapshotPublisher(writer)

	events := []WorkloadEvent{
		{ID: "ev-1", Timestamp: 1000, Name: "trace"},
	}
	samples := []TelemetrySample{
		{ID: "s-1", EventID: "ev-1", Timestamp: 1010, Metric: "m", Value: 1.0},
	}

	rcpt, err := Correlate(events, samples, CorrelationOptions{MaxClockSkewMS: 100})
	if err != nil {
		t.Fatalf("Correlate failed: %v", err)
	}

	snap1, err := publisher.PublishCorrelationReceipt(rcpt)
	if err != nil {
		t.Fatalf("Publish 1: %v", err)
	}
	if snap1.Sequence != 1 {
		t.Errorf("expected sequence 1, got %d", snap1.Sequence)
	}

	snap2, err := publisher.PublishCorrelationReceipt(rcpt)
	if err != nil {
		t.Fatalf("Publish 2: %v", err)
	}
	if snap2.Sequence != 2 {
		t.Errorf("expected sequence 2, got %d", snap2.Sequence)
	}
}

package timeaware

import (
	"strings"
	"testing"
	"unicode/utf8"
)

func TestSpanValidate(t *testing.T) {
	valid := Span{
		Schema:  SpanSchema,
		ID:      "span-1",
		StartNS: 100,
		EndNS:   200,
		Phase:   PhaseActive,
	}
	if err := valid.Validate(); err != nil {
		t.Fatalf("valid.Validate() failed: %v", err)
	}

	invalidSchema := valid
	invalidSchema.Schema = "bad-schema"
	if err := invalidSchema.Validate(); err == nil {
		t.Fatalf("expected error for invalid schema")
	}

	missingID := valid
	missingID.ID = ""
	if err := missingID.Validate(); err == nil {
		t.Fatalf("expected error for missing ID")
	}

	negStart := valid
	negStart.StartNS = -1
	if err := negStart.Validate(); err == nil {
		t.Fatalf("expected error for negative start")
	}

	negEnd := valid
	negEnd.EndNS = -1
	if err := negEnd.Validate(); err == nil {
		t.Fatalf("expected error for negative end")
	}

	inverted := valid
	inverted.StartNS = 200
	inverted.EndNS = 100
	if err := inverted.Validate(); err == nil {
		t.Fatalf("expected error for end preceding start")
	}

	unknownPhase := valid
	unknownPhase.Phase = "unrecognized"
	if err := unknownPhase.Validate(); err == nil {
		t.Fatalf("expected error for unknown phase")
	}
}

func TestSpanDurationNS(t *testing.T) {
	s := Span{
		Schema:  SpanSchema,
		ID:      "span-dur",
		StartNS: 50,
		EndNS:   175,
		Phase:   PhaseActive,
	}
	if got, want := s.DurationNS(), int64(125); got != want {
		t.Fatalf("Span.DurationNS() = %d, want %d", got, want)
	}
}

func TestMetadataValidateAndCohortKey(t *testing.T) {
	m := Metadata{
		Schema:        MetadataSchema,
		BuildVersion:  "v1.0.0",
		ModuleVersion: "r100+gabc",
		SchemaVersion: "1.0",
		ConfigDigest:  "cfg123",
		PolicyDigest:  "pol456",
		Platform:      "linux",
		Architecture:  "amd64",
		Model:         "fak-model",
		Engine:        "fak-native",
		Component:     "agent",
		Runtime:       "go1.26",
	}

	if err := m.Validate(); err != nil {
		t.Fatalf("Metadata.Validate() failed: %v", err)
	}

	badSchema := m
	badSchema.Schema = "wrong"
	if err := badSchema.Validate(); err == nil {
		t.Fatalf("expected error for invalid metadata schema")
	}

	k1 := m.CohortKey()
	k2 := m.CohortKey()
	if k1 != k2 {
		t.Fatalf("CohortKey should be deterministic: %q != %q", k1, k2)
	}

	if !strings.Contains(k1, "build=v1.0.0") || !strings.Contains(k1, "engine=fak-native") {
		t.Fatalf("CohortKey missing expected elements: %s", k1)
	}

	other := m
	other.Model = "other-model"
	if m.CohortKey() == other.CohortKey() {
		t.Fatalf("CohortKey should differ for different model: %s", m.CohortKey())
	}
}

func TestAggregateInvalidAndDuplicateSpans(t *testing.T) {
	spans := []Span{
		{
			Schema:  SpanSchema,
			ID:      "valid-1",
			StartNS: 0,
			EndNS:   100,
			Phase:   PhaseActive,
		},
		{
			Schema:  "invalid-schema",
			ID:      "bad-1",
			StartNS: 10,
			EndNS:   50,
			Phase:   PhaseActive,
		},
		{
			Schema:  SpanSchema,
			ID:      "valid-1", // duplicate
			StartNS: 20,
			EndNS:   80,
			Phase:   PhaseActive,
		},
	}

	r := Aggregate(spans, nil)
	if r.SpanCount != 1 {
		t.Fatalf("SpanCount = %d, want 1", r.SpanCount)
	}
	if r.InvalidSpanCount != 1 {
		t.Fatalf("InvalidSpanCount = %d, want 1", r.InvalidSpanCount)
	}
	if r.DuplicateSpanCount != 1 {
		t.Fatalf("DuplicateSpanCount = %d, want 1", r.DuplicateSpanCount)
	}
	if r.Measures.EffortNS != 100 {
		t.Fatalf("EffortNS = %d, want 100", r.Measures.EffortNS)
	}
}

func TestAggregateSequentialOperations(t *testing.T) {
	spans := []Span{
		{
			Schema:  SpanSchema,
			ID:      "step-1",
			StartNS: 0,
			EndNS:   100,
			Phase:   PhaseActive,
		},
		{
			Schema:  SpanSchema,
			ID:      "step-2",
			StartNS: 100,
			EndNS:   250,
			Phase:   PhaseActive,
		},
		{
			Schema:  SpanSchema,
			ID:      "step-3",
			StartNS: 250,
			EndNS:   400,
			Phase:   PhaseActive,
		},
	}
	edges := []Edge{
		{From: "step-1", To: "step-2", Kind: EdgeDependsOn},
		{From: "step-2", To: "step-3", Kind: EdgeDependsOn},
	}

	r := Aggregate(spans, edges)
	if r.SpanCount != 3 {
		t.Fatalf("SpanCount = %d, want 3", r.SpanCount)
	}
	if r.Measures.WallNS != 400 {
		t.Fatalf("WallNS = %d, want 400", r.Measures.WallNS)
	}
	if r.Measures.EffortNS != 400 {
		t.Fatalf("EffortNS = %d, want 400", r.Measures.EffortNS)
	}
	if r.Measures.UnionActiveNS != 400 {
		t.Fatalf("UnionActiveNS = %d, want 400", r.Measures.UnionActiveNS)
	}
	if r.Measures.CriticalPathNS != 400 {
		t.Fatalf("CriticalPathNS = %d, want 400", r.Measures.CriticalPathNS)
	}
}

func TestAggregateParallelWorkers(t *testing.T) {
	const tenSec = int64(10_000_000_000)
	spans := []Span{
		{Schema: SpanSchema, ID: "w1", StartNS: 0, EndNS: tenSec, Phase: PhaseActive},
		{Schema: SpanSchema, ID: "w2", StartNS: 0, EndNS: tenSec, Phase: PhaseActive},
		{Schema: SpanSchema, ID: "w3", StartNS: 0, EndNS: tenSec, Phase: PhaseActive},
		{Schema: SpanSchema, ID: "w4", StartNS: 0, EndNS: tenSec, Phase: PhaseActive},
	}

	r := Aggregate(spans, nil)
	if r.SpanCount != 4 {
		t.Fatalf("SpanCount = %d, want 4", r.SpanCount)
	}
	if r.Measures.WallNS != tenSec {
		t.Fatalf("WallNS = %d, want %d", r.Measures.WallNS, tenSec)
	}
	if wantEffort := 4 * tenSec; r.Measures.EffortNS != wantEffort {
		t.Fatalf("EffortNS = %d, want %d", r.Measures.EffortNS, wantEffort)
	}
	if r.Measures.UnionActiveNS != tenSec {
		t.Fatalf("UnionActiveNS = %d, want %d", r.Measures.UnionActiveNS, tenSec)
	}
	if r.Measures.CriticalPathNS != tenSec {
		t.Fatalf("CriticalPathNS = %d, want %d", r.Measures.CriticalPathNS, tenSec)
	}
}

func TestAggregatePhasesAndEffortSeparation(t *testing.T) {
	spans := []Span{
		{Schema: SpanSchema, ID: "act", StartNS: 0, EndNS: 100, Phase: PhaseActive},
		{Schema: SpanSchema, ID: "q", StartNS: 100, EndNS: 150, Phase: PhaseQueue},
		{Schema: SpanSchema, ID: "wait", StartNS: 150, EndNS: 220, Phase: PhaseWait},
		{Schema: SpanSchema, ID: "stall", StartNS: 220, EndNS: 260, Phase: PhaseStall},
		{Schema: SpanSchema, ID: "idle", StartNS: 260, EndNS: 300, Phase: PhaseIdle},
		{Schema: SpanSchema, ID: "unk", StartNS: 300, EndNS: 330, Phase: PhaseUnknown},
		{Schema: SpanSchema, ID: "spec", StartNS: 330, EndNS: 370, Phase: PhaseSpeculative},
		{Schema: SpanSchema, ID: "cancel", StartNS: 370, EndNS: 400, Phase: PhasePostCancel},
	}

	r := Aggregate(spans, nil)
	if r.Measures.QueueNS != 50 {
		t.Fatalf("QueueNS = %d, want 50", r.Measures.QueueNS)
	}
	if r.Measures.WaitNS != 70 {
		t.Fatalf("WaitNS = %d, want 70", r.Measures.WaitNS)
	}
	if r.Measures.StallNS != 40 {
		t.Fatalf("StallNS = %d, want 40", r.Measures.StallNS)
	}
	if r.Measures.IdleNS != 40 {
		t.Fatalf("IdleNS = %d, want 40", r.Measures.IdleNS)
	}
	if r.Measures.UnknownNS != 30 {
		t.Fatalf("UnknownNS = %d, want 30", r.Measures.UnknownNS)
	}
	if r.Measures.SpeculativeNS != 40 {
		t.Fatalf("SpeculativeNS = %d, want 40", r.Measures.SpeculativeNS)
	}
	if r.Measures.PostCancelNS != 30 {
		t.Fatalf("PostCancelNS = %d, want 30", r.Measures.PostCancelNS)
	}

	// Effort is active (100) + speculative (40) + post-cancel (30) = 170.
	if wantEffort := int64(170); r.Measures.EffortNS != wantEffort {
		t.Fatalf("EffortNS = %d, want %d", r.Measures.EffortNS, wantEffort)
	}
}

func TestAggregateRetryAndPollCounts(t *testing.T) {
	spans := []Span{
		{
			Schema:     SpanSchema,
			ID:         "s1",
			StartNS:    0,
			EndNS:      50,
			Phase:      PhaseActive,
			Dimensions: Dimensions{Retry: 1},
		},
		{
			Schema:     SpanSchema,
			ID:         "s2",
			StartNS:    50,
			EndNS:      100,
			Phase:      PhaseWait,
			Dimensions: Dimensions{Poll: 2},
		},
		{
			Schema:     SpanSchema,
			ID:         "s3",
			StartNS:    100,
			EndNS:      150,
			Phase:      PhaseActive,
			Dimensions: Dimensions{Retry: 2, Poll: 1},
		},
	}

	r := Aggregate(spans, nil)
	if r.RetryCount != 2 {
		t.Fatalf("RetryCount = %d, want 2", r.RetryCount)
	}
	if r.PollCount != 2 {
		t.Fatalf("PollCount = %d, want 2", r.PollCount)
	}
}

func TestAggregateCriticalPathWithDAG(t *testing.T) {
	spans := []Span{
		{Schema: SpanSchema, ID: "A", StartNS: 0, EndNS: 50, Phase: PhaseActive},
		{Schema: SpanSchema, ID: "B", StartNS: 50, EndNS: 150, Phase: PhaseActive},
		{Schema: SpanSchema, ID: "C", StartNS: 0, EndNS: 70, Phase: PhaseActive},
		{Schema: SpanSchema, ID: "D", StartNS: 150, EndNS: 180, Phase: PhaseActive},
	}
	// A (50) -> B (100) -> D (30) = 180.
	// C (70) -> D (30) = 100.
	edges := []Edge{
		{From: "A", To: "B", Kind: EdgeDependsOn},
		{From: "B", To: "D", Kind: EdgeDependsOn},
		{From: "C", To: "D", Kind: EdgeDependsOn},
	}

	r := Aggregate(spans, edges)
	if r.Measures.CriticalPathNS != 180 {
		t.Fatalf("CriticalPathNS = %d, want 180", r.Measures.CriticalPathNS)
	}
}

func TestUnionDurationOverlapping(t *testing.T) {
	spans := []Span{
		{Schema: SpanSchema, ID: "s1", StartNS: 0, EndNS: 100, Phase: PhaseActive},
		{Schema: SpanSchema, ID: "s2", StartNS: 50, EndNS: 150, Phase: PhaseActive},
		{Schema: SpanSchema, ID: "s3", StartNS: 200, EndNS: 300, Phase: PhaseActive},
		{Schema: SpanSchema, ID: "s4", StartNS: 250, EndNS: 350, Phase: PhaseActive},
	}
	r := Aggregate(spans, nil)
	// [0, 150] -> 150. [200, 350] -> 150. Total = 300.
	if r.Measures.UnionActiveNS != 300 {
		t.Fatalf("UnionActiveNS = %d, want 300", r.Measures.UnionActiveNS)
	}
}

func TestActivityScopeRevisionAndDenominatorClass(t *testing.T) {
	snapshot := ActivitySnapshot{
		State:  StateWorking,
		Motion: MotionAdvancing,
		Scope: Scope{
			Completed:        4,
			Total:            KnownCount(8),
			DenominatorClass: DenominatorDiscoveredWork,
			Revision:         3,
		},
		Queued:   KnownCount(1),
		InFlight: KnownCount(2),
		Current:  Summary{Text: "checking new scope", Provenance: ProvenanceFact},
		Next:     Summary{Text: "finish remaining checks", Provenance: ProvenanceForecast},
	}

	got := FormatActivitySnapshot(snapshot, 0)
	for _, want := range []string{"scope 4/8 discovered_work @r3", "current checking new scope [FACT]", "next finish remaining checks [FORECAST]"} {
		if !strings.Contains(got, want) {
			t.Fatalf("FormatActivitySnapshot() = %q, want %q", got, want)
		}
	}
	if gotMethod := snapshot.Render(0); gotMethod != got {
		t.Fatalf("snapshot.Render() = %q, want %q", gotMethod, got)
	}
}

func TestActivitySnapshotUnknownTotalDoesNotRenderZero(t *testing.T) {
	snapshot := ActivitySnapshot{
		Scope: Scope{Completed: 4, Total: UnavailableCount(), DenominatorClass: DenominatorDeclaredWork, Revision: 2},
	}
	got := FormatActivitySnapshot(snapshot, 0)
	if !strings.Contains(got, "scope 4/? declared_work @r2") {
		t.Fatalf("FormatActivitySnapshot() = %q, want explicit unknown total", got)
	}
	if strings.Contains(got, "scope 4/0") {
		t.Fatalf("FormatActivitySnapshot() = %q, unavailable total rendered as zero", got)
	}
}

func TestActivitySnapshotKnownAndUnavailableLifecycleCounts(t *testing.T) {
	snapshot := ActivitySnapshot{
		Queued:   UnavailableCount(),
		InFlight: KnownCount(0),
	}
	got := FormatActivitySnapshot(snapshot, 0)
	if !strings.Contains(got, "queued ? · in-flight 0") {
		t.Fatalf("FormatActivitySnapshot() = %q, want unavailable queue and known-zero in-flight", got)
	}
}

func TestActivitySnapshotRenderingIsDeterministicAndWidthBounded(t *testing.T) {
	snapshot := ActivitySnapshot{
		State:    StateRecovering,
		Motion:   MotionOscillating,
		Scope:    Scope{Completed: 3, Total: KnownCount(5), DenominatorClass: DenominatorAttempt, Revision: 7},
		Queued:   KnownCount(2),
		InFlight: KnownCount(1),
		Current:  Summary{Text: "retrying a deliberately long semantic operation", Provenance: ProvenanceInference},
		Next:     Summary{Text: "observe material delta", Provenance: ProvenanceForecast},
	}
	const width = 72
	first := FormatActivitySnapshot(snapshot, width)
	second := snapshot.Render(width)
	if first != second {
		t.Fatalf("rendering is not deterministic: %q != %q", first, second)
	}
	if got := utf8.RuneCountInString(first); got > width {
		t.Fatalf("render width = %d, want <= %d: %q", got, width, first)
	}
	if !strings.HasSuffix(first, "…") {
		t.Fatalf("truncated render = %q, want ellipsis", first)
	}
}

func TestActivityEpisodeRepresentsSemanticTransition(t *testing.T) {
	episode := Episode{
		Intent: "understand current work",
		Transition: Transition{
			Operation:        "inspect activity ledger",
			MaterialEvidence: "scope revision advanced to r3",
			Provenance:       ProvenanceFact,
		},
		Age: "12s",
	}
	if episode.Transition.Provenance != ProvenanceFact || episode.Transition.MaterialEvidence == "" {
		t.Fatalf("episode lost semantic transition evidence: %#v", episode)
	}
}

package nativeperfslo

import (
	"strings"
	"testing"
	"time"
)

func TestTransitionsAreMatchedAndDebounced(t *testing.T) {
	now := time.Date(2026, 8, 28, 20, 0, 0, 0, time.UTC)
	envelope := Envelope{ModuleRev: "internal/nativeperf@r41+gbbdda8f04", Benchmark: "qwen38-4b-in128-out128-b1-quality-v3", Model: "Qwen3.8-4B", Backend: "cuda"}
	baseline := observation(now, envelope, 1)
	evaluator := New(DefaultThresholds())

	assertState(t, evaluator, now, baseline, observation(now, envelope, 1), StateMissing)
	assertState(t, evaluator, now.Add(time.Second), baseline, observation(now.Add(time.Second), envelope, 1), StateGood)

	bad := observation(now.Add(2*time.Second), envelope, 1)
	bad.Values[TTFT] = Value{Available: true, Value: 1.25}
	assertState(t, evaluator, now.Add(2*time.Second), baseline, bad, StateGood)
	assertState(t, evaluator, now.Add(3*time.Second), baseline, bad, StateRegression)

	recovered := observation(now.Add(4*time.Second), envelope, 1)
	assertState(t, evaluator, now.Add(4*time.Second), baseline, recovered, StateRegression)
	result := assertState(t, evaluator, now.Add(5*time.Second), baseline, recovered, StateGood)
	if result.Ratios[TTFT].Value != 1 {
		t.Fatalf("TTFT ratio = %v, want 1", result.Ratios[TTFT].Value)
	}
}

func TestMissingEvidenceIsImmediateAndNotZeroCoerced(t *testing.T) {
	now := time.Date(2026, 8, 28, 20, 0, 0, 0, time.UTC)
	envelope := Envelope{ModuleRev: "internal/nativeperf@r41+gbbdda8f04", Benchmark: "qwen38-4b-in128-out128-b1-quality-v3", Model: "Qwen3.8-4B", Backend: "metal"}
	baseline := observation(now, envelope, 1)
	candidate := observation(now, envelope, 1)
	candidate.Values[TransferShare] = Value{}

	result, err := New(DefaultThresholds()).Observe(now, baseline, candidate)
	if err != nil {
		t.Fatal(err)
	}
	if result.State != StateMissing {
		t.Fatalf("state = %q, want %q", result.State, StateMissing)
	}
	metrics := RenderPrometheus(result)
	if strings.Contains(metrics, `objective="transfer_share"`) {
		t.Fatalf("unavailable transfer share was rendered as a value:\n%s", metrics)
	}
	if !strings.Contains(metrics, `state="missing_evidence"} 1`) {
		t.Fatalf("missing state absent:\n%s", metrics)
	}
}

func TestUnmatchedEnvelopeIsRejected(t *testing.T) {
	now := time.Date(2026, 8, 28, 20, 0, 0, 0, time.UTC)
	baselineEnvelope := Envelope{ModuleRev: "internal/nativeperf@r41+gbbdda8f04", Benchmark: "qwen38-4b-in128-out128-b1-quality-v3", Model: "Qwen3.8-4B", Backend: "cuda"}
	candidateEnvelope := baselineEnvelope
	candidateEnvelope.Benchmark = "qwen38-4b-in512-out128-b1-quality-v3"
	_, err := New(DefaultThresholds()).Observe(now, observation(now, baselineEnvelope, 1), observation(now, candidateEnvelope, 1))
	if err == nil || !strings.Contains(err.Error(), "unmatched benchmark envelopes") {
		t.Fatalf("error = %v, want unmatched-envelope refusal", err)
	}
}

func TestFreshnessAndCoverageBecomeActionableViolations(t *testing.T) {
	now := time.Date(2026, 8, 28, 20, 0, 0, 0, time.UTC)
	envelope := Envelope{ModuleRev: "internal/nativeperf@r41+gbbdda8f04", Benchmark: "qwen38-4b-in128-out128-b1-quality-v3", Model: "Qwen3.8-4B", Backend: "cuda"}
	baseline := observation(now, envelope, 1)
	candidate := observation(now.Add(-10*time.Minute), envelope, 1)
	candidate.Receipts = 8
	candidate.Expected = 10
	evaluator := New(DefaultThresholds())
	result, err := evaluator.Observe(now, baseline, candidate)
	if err != nil {
		t.Fatal(err)
	}
	if !result.Violations[EvidenceFresh] || !result.Violations[ReceiptCoverage] {
		t.Fatalf("violations = %#v", result.Violations)
	}
	result, err = evaluator.Observe(now.Add(time.Second), baseline, candidate)
	if err != nil {
		t.Fatal(err)
	}
	if result.State != StateRegression {
		t.Fatalf("state = %q, want regression", result.State)
	}
}

func assertState(t *testing.T, evaluator *Evaluator, now time.Time, baseline, candidate Observation, want State) Result {
	t.Helper()
	result, err := evaluator.Observe(now, baseline, candidate)
	if err != nil {
		t.Fatal(err)
	}
	if result.State != want {
		t.Fatalf("state = %q, want %q (good=%d bad=%d)", result.State, want, result.GoodStreak, result.BadStreak)
	}
	return result
}

func observation(at time.Time, envelope Envelope, scale float64) Observation {
	return Observation{
		At: at, Envelope: envelope, Receipts: 100, Expected: 100,
		//enumlint:exempt EvidenceFresh and ReceiptCoverage are derived by Observe from At and receipt counts; callers must not supply them in Values.
		Values: map[Objective]Value{
			TTFT:            {Available: true, Value: 1.0 * scale},
			TPOT:            {Available: true, Value: 0.03 * scale},
			Throughput:      {Available: true, Value: 100 / scale},
			QueueDelay:      {Available: true, Value: 0.1 * scale},
			CacheEfficiency: {Available: true, Value: 0.8 / scale},
			TransferShare:   {Available: true, Value: 0.08 * scale},
			KernelShare:     {Available: true, Value: 0.60 * scale},
			MemoryPressure:  {Available: true, Value: 0.70 * scale},
		},
	}
}

func TestProduceLiveSeriesGoodAndLifecycleAnnotation(t *testing.T) {
	now := time.Date(2026, 8, 29, 20, 0, 0, 0, time.UTC)
	envelope := Envelope{
		ModuleRev: "internal/nativeperf@r41+gbbdda8f04",
		Benchmark: "qwen38-4b-in128-out128-b1-quality-v3",
		Model:     "Qwen3.8-4B",
		Backend:   "metal",
	}
	baseline := observation(now.Add(-10*time.Second), envelope, 1)
	candidate := observation(now.Add(-2*time.Second), envelope, 1)

	evaluator := New(DefaultThresholds())
	// Prime evaluator to Good state (requires 2 consecutive complete samples)
	evaluator.Observe(now.Add(-5*time.Second), baseline, candidate)
	evaluator.Observe(now.Add(-3*time.Second), baseline, candidate)
	evaluator.Observe(now.Add(-1*time.Second), baseline, candidate)

	input := LiveRunInput{
		ObservedAt:        now.Add(-2 * time.Second),
		Baseline:          baseline,
		Candidate:         candidate,
		SupervisionStatus: SupervisionActive,
		Lifecycle: LifecycleAnnotation{
			Event:   EventRelease,
			Release: "v0.45.1",
		},
	}

	res := ProduceLiveSeries(now, 5*time.Minute, evaluator, input)
	if res.Status != SeriesProduced {
		t.Fatalf("expected status %v, got %v (reason: %v)", SeriesProduced, res.Status, res.UnavailableReason)
	}
	if res.LifecycleStatus != LifecycleGood {
		t.Fatalf("expected lifecycle status %v, got %v", LifecycleGood, res.LifecycleStatus)
	}
	if res.UnavailableReason != ReasonNone {
		t.Fatalf("expected reason none, got %v", res.UnavailableReason)
	}
	if !strings.Contains(res.Prometheus, `fak_native_slo_state{engine="fak-native",module_rev="internal/nativeperf@r41+gbbdda8f04",benchmark_envelope="qwen38-4b-in128-out128-b1-quality-v3",model="Qwen3.8-4B",backend="metal",state="good"} 1`) {
		t.Fatalf("missing good state metric in prometheus output:\n%s", res.Prometheus)
	}
	if !strings.Contains(res.Prometheus, `fak_native_lifecycle_event_info{engine="fak-native",module_rev="internal/nativeperf@r41+gbbdda8f04",benchmark_envelope="qwen38-4b-in128-out128-b1-quality-v3",model="Qwen3.8-4B",backend="metal",event="release",release="v0.45.1"} 1`) {
		t.Fatalf("missing lifecycle event info metric in prometheus output:\n%s", res.Prometheus)
	}
}

func TestProduceLiveSeriesRegressed(t *testing.T) {
	now := time.Date(2026, 8, 29, 20, 0, 0, 0, time.UTC)
	envelope := Envelope{
		ModuleRev: "internal/nativeperf@r42+gbad",
		Benchmark: "qwen38-4b-in128-out128-b1-quality-v3",
		Model:     "Qwen3.8-4B",
		Backend:   "metal",
	}
	baseline := observation(now.Add(-10*time.Second), envelope, 1)
	badCandidatePre := observation(now.Add(-3*time.Second), envelope, 1)
	badCandidatePre.Values[TTFT] = Value{Available: true, Value: 1.5}
	badCandidate := observation(now.Add(-2*time.Second), envelope, 1)
	badCandidate.Values[TTFT] = Value{Available: true, Value: 1.5}

	evaluator := New(DefaultThresholds())
	// Prime evaluator to Good, then transition to regression (2 consecutive bad samples)
	evaluator.Observe(now.Add(-5*time.Second), baseline, baseline)
	evaluator.Observe(now.Add(-4*time.Second), baseline, baseline)
	evaluator.Observe(now.Add(-3*time.Second), baseline, badCandidatePre)

	input := LiveRunInput{
		ObservedAt:        now.Add(-2 * time.Second),
		Baseline:          baseline,
		Candidate:         badCandidate,
		SupervisionStatus: SupervisionActive,
		Lifecycle: LifecycleAnnotation{
			Event:   EventRevert,
			Release: "v0.45.0",
		},
	}

	res := ProduceLiveSeries(now, 5*time.Minute, evaluator, input)
	if res.Status != SeriesProduced {
		t.Fatalf("expected status %v, got %v", SeriesProduced, res.Status)
	}
	if res.LifecycleStatus != LifecycleRegressed {
		t.Fatalf("expected lifecycle status %v, got %v", LifecycleRegressed, res.LifecycleStatus)
	}
	if !strings.Contains(res.Prometheus, `fak_native_slo_state{engine="fak-native",module_rev="internal/nativeperf@r42+gbad",benchmark_envelope="qwen38-4b-in128-out128-b1-quality-v3",model="Qwen3.8-4B",backend="metal",state="regression"} 1`) {
		t.Fatalf("missing regression state metric in prometheus output:\n%s", res.Prometheus)
	}
	if !strings.Contains(res.Prometheus, `fak_native_slo_violation{engine="fak-native",module_rev="internal/nativeperf@r42+gbad",benchmark_envelope="qwen38-4b-in128-out128-b1-quality-v3",model="Qwen3.8-4B",backend="metal",objective="ttft"} 1`) {
		t.Fatalf("missing ttft violation metric in prometheus output:\n%s", res.Prometheus)
	}
}

func TestProduceLiveSeriesStale(t *testing.T) {
	now := time.Date(2026, 8, 29, 20, 0, 0, 0, time.UTC)
	envelope := Envelope{
		ModuleRev: "internal/nativeperf@r41+gbbdda8f04",
		Benchmark: "qwen38-4b-in128-out128-b1-quality-v3",
		Model:     "Qwen3.8-4B",
		Backend:   "metal",
	}
	baseline := observation(now.Add(-10*time.Minute), envelope, 1)
	candidate := observation(now.Add(-10*time.Minute), envelope, 1)

	input := LiveRunInput{
		ObservedAt:        now.Add(-10 * time.Minute),
		Baseline:          baseline,
		Candidate:         candidate,
		SupervisionStatus: SupervisionActive,
	}

	res := ProduceLiveSeries(now, 5*time.Minute, nil, input)
	if res.Status != SeriesUnavailable {
		t.Fatalf("expected status %v, got %v", SeriesUnavailable, res.Status)
	}
	if res.UnavailableReason != ReasonStale {
		t.Fatalf("expected reason %v, got %v", ReasonStale, res.UnavailableReason)
	}
	if res.LifecycleStatus != LifecycleStale {
		t.Fatalf("expected lifecycle status %v, got %v", LifecycleStale, res.LifecycleStatus)
	}
	if res.Prometheus != "" {
		t.Fatalf("stale observation produced prometheus metrics: %s", res.Prometheus)
	}
}

func TestProduceLiveSeriesMismatchedEnvelope(t *testing.T) {
	now := time.Date(2026, 8, 29, 20, 0, 0, 0, time.UTC)
	baselineEnvelope := Envelope{
		ModuleRev: "internal/nativeperf@r41+gbbdda8f04",
		Benchmark: "qwen38-4b-in128-out128-b1-quality-v3",
		Model:     "Qwen3.8-4B",
		Backend:   "metal",
	}
	candidateEnvelope := baselineEnvelope
	candidateEnvelope.Benchmark = "qwen38-4b-in512-out128-b1-quality-v3"

	input := LiveRunInput{
		ObservedAt:        now.Add(-10 * time.Second),
		Baseline:          observation(now.Add(-10*time.Second), baselineEnvelope, 1),
		Candidate:         observation(now.Add(-10*time.Second), candidateEnvelope, 1),
		SupervisionStatus: SupervisionActive,
	}

	res := ProduceLiveSeries(now, 5*time.Minute, nil, input)
	if res.Status != SeriesUnavailable {
		t.Fatalf("expected status %v, got %v", SeriesUnavailable, res.Status)
	}
	if res.UnavailableReason != ReasonMismatched {
		t.Fatalf("expected reason %v, got %v", ReasonMismatched, res.UnavailableReason)
	}
	if res.LifecycleStatus != LifecycleMismatched {
		t.Fatalf("expected lifecycle status %v, got %v", LifecycleMismatched, res.LifecycleStatus)
	}
	if res.Prometheus != "" {
		t.Fatalf("mismatched envelope produced prometheus metrics: %s", res.Prometheus)
	}
}

func TestProduceLiveSeriesSupervisionFailed(t *testing.T) {
	now := time.Date(2026, 8, 29, 20, 0, 0, 0, time.UTC)
	envelope := Envelope{
		ModuleRev: "internal/nativeperf@r41+gbbdda8f04",
		Benchmark: "qwen38-4b-in128-out128-b1-quality-v3",
		Model:     "Qwen3.8-4B",
		Backend:   "metal",
	}
	baseline := observation(now.Add(-10*time.Second), envelope, 1)
	candidate := observation(now.Add(-2*time.Second), envelope, 1)

	input := LiveRunInput{
		ObservedAt:        now.Add(-2 * time.Second),
		Baseline:          baseline,
		Candidate:         candidate,
		SupervisionStatus: SupervisionFailed,
	}

	res := ProduceLiveSeries(now, 5*time.Minute, nil, input)
	if res.Status != SeriesUnavailable {
		t.Fatalf("expected status %v, got %v", SeriesUnavailable, res.Status)
	}
	if res.UnavailableReason != ReasonIncomplete {
		t.Fatalf("expected reason %v, got %v", ReasonIncomplete, res.UnavailableReason)
	}
}

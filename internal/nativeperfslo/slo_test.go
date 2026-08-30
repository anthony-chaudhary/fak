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

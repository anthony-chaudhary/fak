package nativeperfslo

import (
	"math"
	"strings"
	"testing"
	"time"
)

func makeTestEnvelope(backend string) Envelope {
	return Envelope{
		ModuleRev: "internal/nativeperf@r50+gabcdef123",
		Benchmark: "qwen38-4b-in128-out128-b1-quality-v3",
		Model:     "Qwen3.8-4B",
		Backend:   backend,
	}
}

func makeTestObservation(at time.Time, env Envelope, scale float64) Observation {
	return Observation{
		At:       at,
		Envelope: env,
		Receipts: 100,
		Expected: 100,
		Values: map[Objective]Value{
			TTFT:            {Available: true, Value: 1.0 * scale},
			TPOT:            {Available: true, Value: 0.03 * scale},
			Throughput:      {Available: true, Value: 100.0 / scale},
			QueueDelay:      {Available: true, Value: 0.10 * scale},
			CacheEfficiency: {Available: true, Value: 0.80 / scale},
			TransferShare:   {Available: true, Value: 0.08 * scale},
			KernelShare:     {Available: true, Value: 0.60 * scale},
			MemoryPressure:  {Available: true, Value: 0.70 * scale},
		},
	}
}

func TestSLOEvaluationLifecycleTransitions(t *testing.T) {
	now := time.Date(2026, 9, 4, 12, 0, 0, 0, time.UTC)
	env := makeTestEnvelope("cuda")
	baseline := makeTestObservation(now, env, 1.0)
	evaluator := New(DefaultThresholds())

	// Step 1: Initial state is StateMissing.
	res1, err := evaluator.Observe(now, baseline, makeTestObservation(now, env, 1.0))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res1.State != StateMissing || res1.GoodStreak != 1 || res1.BadStreak != 0 {
		t.Fatalf("sample 1: state=%v good=%d bad=%d, want state=%v good=1 bad=0",
			res1.State, res1.GoodStreak, res1.BadStreak, StateMissing)
	}

	// Step 2: Second consecutive healthy observation reaches StateGood.
	res2, err := evaluator.Observe(now.Add(time.Second), baseline, makeTestObservation(now.Add(time.Second), env, 1.0))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res2.State != StateGood || res2.GoodStreak != 2 || res2.BadStreak != 0 {
		t.Fatalf("sample 2: state=%v good=%d bad=%d, want state=%v good=2 bad=0",
			res2.State, res2.GoodStreak, res2.BadStreak, StateGood)
	}

	// Step 3: Continued healthy observation maintains StateGood and increments GoodStreak.
	res3, err := evaluator.Observe(now.Add(2*time.Second), baseline, makeTestObservation(now.Add(2*time.Second), env, 1.0))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res3.State != StateGood || res3.GoodStreak != 3 || res3.BadStreak != 0 {
		t.Fatalf("sample 3: state=%v good=%d bad=%d, want state=%v good=3 bad=0",
			res3.State, res3.GoodStreak, res3.BadStreak, StateGood)
	}

	// Step 4: First regressed sample (TTFT regressed by 25%). Debouncing holds state at StateGood.
	badCandidate1 := makeTestObservation(now.Add(3*time.Second), env, 1.0)
	badCandidate1.Values[TTFT] = Value{Available: true, Value: 1.25}
	res4, err := evaluator.Observe(now.Add(3*time.Second), baseline, badCandidate1)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res4.State != StateGood || res4.GoodStreak != 0 || res4.BadStreak != 1 {
		t.Fatalf("sample 4: state=%v good=%d bad=%d, want state=%v good=0 bad=1",
			res4.State, res4.GoodStreak, res4.BadStreak, StateGood)
	}

	// Step 5: Second consecutive regressed sample triggers transition to StateRegression.
	badCandidate2 := makeTestObservation(now.Add(4*time.Second), env, 1.0)
	badCandidate2.Values[TTFT] = Value{Available: true, Value: 1.25}
	res5, err := evaluator.Observe(now.Add(4*time.Second), baseline, badCandidate2)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res5.State != StateRegression || res5.GoodStreak != 0 || res5.BadStreak != 2 {
		t.Fatalf("sample 5: state=%v good=%d bad=%d, want state=%v good=0 bad=2",
			res5.State, res5.GoodStreak, res5.BadStreak, StateRegression)
	}

	// Step 6: First recovery sample. Debouncing holds state in StateRegression.
	recCandidate1 := makeTestObservation(now.Add(5*time.Second), env, 1.0)
	res6, err := evaluator.Observe(now.Add(5*time.Second), baseline, recCandidate1)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res6.State != StateRegression || res6.GoodStreak != 1 || res6.BadStreak != 0 {
		t.Fatalf("sample 6: state=%v good=%d bad=%d, want state=%v good=1 bad=0",
			res6.State, res6.GoodStreak, res6.BadStreak, StateRegression)
	}

	// Step 7: Second recovery sample confirms recovery to StateGood.
	recCandidate2 := makeTestObservation(now.Add(6*time.Second), env, 1.0)
	res7, err := evaluator.Observe(now.Add(6*time.Second), baseline, recCandidate2)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res7.State != StateGood || res7.GoodStreak != 2 || res7.BadStreak != 0 {
		t.Fatalf("sample 7: state=%v good=%d bad=%d, want state=%v good=2 bad=0",
			res7.State, res7.GoodStreak, res7.BadStreak, StateGood)
	}

	// Step 8: Missing evidence immediately drops state to StateMissing and resets streaks.
	incompleteCandidate := makeTestObservation(now.Add(7*time.Second), env, 1.0)
	incompleteCandidate.Values[KernelShare] = Value{Available: false}
	res8, err := evaluator.Observe(now.Add(7*time.Second), baseline, incompleteCandidate)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res8.State != StateMissing || res8.GoodStreak != 0 || res8.BadStreak != 0 {
		t.Fatalf("sample 8: state=%v good=%d bad=%d, want state=%v good=0 bad=0",
			res8.State, res8.GoodStreak, res8.BadStreak, StateMissing)
	}

	// Step 9: Non-finite value (NaN) also causes immediate StateMissing and resets streaks.
	nanCandidate := makeTestObservation(now.Add(8*time.Second), env, 1.0)
	nanCandidate.Values[TPOT] = Value{Available: true, Value: math.NaN()}
	res9, err := evaluator.Observe(now.Add(8*time.Second), baseline, nanCandidate)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res9.State != StateMissing || res9.GoodStreak != 0 || res9.BadStreak != 0 {
		t.Fatalf("sample 9: state=%v good=%d bad=%d, want state=%v good=0 bad=0",
			res9.State, res9.GoodStreak, res9.BadStreak, StateMissing)
	}
}

func TestObjectiveComparisonRules(t *testing.T) {
	now := time.Date(2026, 9, 4, 12, 0, 0, 0, time.UTC)
	env := makeTestEnvelope("metal")
	baseline := makeTestObservation(now, env, 1.0)

	thresh := DefaultThresholds()
	thresh.MaxMemoryPressure = 0.85

	t.Run("LowerIsBetterObjectives", func(t *testing.T) {
		lowerObjectives := []Objective{TTFT, TPOT, QueueDelay, TransferShare, KernelShare}
		for _, obj := range lowerObjectives {
			evaluator := New(thresh)
			// Candidate with 9% regression (within 10% allowance) -> no violation
			candOk := makeTestObservation(now, env, 1.0)
			candOk.Values[obj] = Value{Available: true, Value: baseline.Values[obj].Value * 1.09}
			resOk, err := evaluator.Observe(now, baseline, candOk)
			if err != nil {
				t.Fatalf("%s ok check error: %v", obj, err)
			}
			if resOk.Violations[obj] {
				t.Fatalf("objective %s unexpectedly violated at ratio 1.09", obj)
			}

			// Candidate with 11% regression (exceeds 10% allowance) -> violation
			candBad := makeTestObservation(now, env, 1.0)
			candBad.Values[obj] = Value{Available: true, Value: baseline.Values[obj].Value * 1.11}
			resBad, err := evaluator.Observe(now, baseline, candBad)
			if err != nil {
				t.Fatalf("%s bad check error: %v", obj, err)
			}
			if !resBad.Violations[obj] {
				t.Fatalf("objective %s failed to report violation at ratio 1.11", obj)
			}
		}
	})

	t.Run("HigherIsBetterObjectives", func(t *testing.T) {
		higherObjectives := []Objective{Throughput, CacheEfficiency}
		for _, obj := range higherObjectives {
			evaluator := New(thresh)
			// Candidate retains 91% (within 90% allowance) -> no violation
			candOk := makeTestObservation(now, env, 1.0)
			candOk.Values[obj] = Value{Available: true, Value: baseline.Values[obj].Value * 0.91}
			resOk, err := evaluator.Observe(now, baseline, candOk)
			if err != nil {
				t.Fatalf("%s ok check error: %v", obj, err)
			}
			if resOk.Violations[obj] {
				t.Fatalf("objective %s unexpectedly violated at retention 0.91", obj)
			}

			// Candidate retains 89% (below 90% allowance) -> violation
			candBad := makeTestObservation(now, env, 1.0)
			candBad.Values[obj] = Value{Available: true, Value: baseline.Values[obj].Value * 0.89}
			resBad, err := evaluator.Observe(now, baseline, candBad)
			if err != nil {
				t.Fatalf("%s bad check error: %v", obj, err)
			}
			if !resBad.Violations[obj] {
				t.Fatalf("objective %s failed to report violation at retention 0.89", obj)
			}
		}
	})

	t.Run("AbsoluteMemoryPressure", func(t *testing.T) {
		evaluator := New(thresh)
		// Memory pressure 0.80 <= 0.85 -> no violation
		candOk := makeTestObservation(now, env, 1.0)
		candOk.Values[MemoryPressure] = Value{Available: true, Value: 0.80}
		resOk, err := evaluator.Observe(now, baseline, candOk)
		if err != nil {
			t.Fatalf("memory pressure ok check error: %v", err)
		}
		if resOk.Violations[MemoryPressure] {
			t.Fatalf("memory pressure 0.80 unexpectedly flagged as violation")
		}

		// Memory pressure 0.86 > 0.85 -> violation
		candBad := makeTestObservation(now, env, 1.0)
		candBad.Values[MemoryPressure] = Value{Available: true, Value: 0.86}
		resBad, err := evaluator.Observe(now, baseline, candBad)
		if err != nil {
			t.Fatalf("memory pressure bad check error: %v", err)
		}
		if !resBad.Violations[MemoryPressure] {
			t.Fatalf("memory pressure 0.86 failed to flag violation")
		}
	})

	t.Run("DerivedEvidenceFreshnessAndCoverage", func(t *testing.T) {
		evaluator := New(thresh)
		// Candidate observed 4 minutes ago (<= 5 minute max age) -> no fresh violation
		candOk := makeTestObservation(now.Add(-4*time.Minute), env, 1.0)
		candOk.Receipts = 96
		candOk.Expected = 100
		resOk, err := evaluator.Observe(now, baseline, candOk)
		if err != nil {
			t.Fatalf("freshness ok check error: %v", err)
		}
		if resOk.Violations[EvidenceFresh] {
			t.Fatalf("4-minute age unexpectedly flagged as fresh violation")
		}
		if resOk.Violations[ReceiptCoverage] {
			t.Fatalf("96%% coverage unexpectedly flagged as coverage violation")
		}

		// Candidate observed 6 minutes ago (> 5 minute max age) and 90% coverage (< 95%) -> violations
		candBad := makeTestObservation(now.Add(-6*time.Minute), env, 1.0)
		candBad.Receipts = 90
		candBad.Expected = 100
		resBad, err := evaluator.Observe(now, baseline, candBad)
		if err != nil {
			t.Fatalf("freshness bad check error: %v", err)
		}
		if !resBad.Violations[EvidenceFresh] {
			t.Fatalf("6-minute age failed to flag fresh violation")
		}
		if !resBad.Violations[ReceiptCoverage] {
			t.Fatalf("90%% coverage failed to flag coverage violation")
		}
	})

	t.Run("InvalidBaselineValuesFail", func(t *testing.T) {
		evaluator := New(thresh)
		badBaseline := makeTestObservation(now, env, 1.0)
		badBaseline.Values[TTFT] = Value{Available: true, Value: 0.0}
		_, err := evaluator.Observe(now, badBaseline, makeTestObservation(now, env, 1.0))
		if err == nil || !strings.Contains(err.Error(), "unavailable or non-positive") {
			t.Fatalf("expected non-positive error, got %v", err)
		}

		badBaseline.Values[TTFT] = Value{Available: false, Value: 1.0}
		_, err = evaluator.Observe(now, badBaseline, makeTestObservation(now, env, 1.0))
		if err == nil || !strings.Contains(err.Error(), "unavailable or non-positive") {
			t.Fatalf("expected unavailable error, got %v", err)
		}
	})
}

func TestProduceLiveSeriesComprehensive(t *testing.T) {
	now := time.Date(2026, 9, 4, 14, 0, 0, 0, time.UTC)
	env := makeTestEnvelope("cpu")
	baseline := makeTestObservation(now.Add(-10*time.Second), env, 1.0)
	candidate := makeTestObservation(now.Add(-2*time.Second), env, 1.0)

	t.Run("ZeroTimestampOrInvalidMaxAge", func(t *testing.T) {
		input := LiveRunInput{
			ObservedAt:        now.Add(-2 * time.Second),
			Baseline:          baseline,
			Candidate:         candidate,
			SupervisionStatus: SupervisionActive,
		}
		resZeroNow := ProduceLiveSeries(time.Time{}, 5*time.Minute, nil, input)
		if resZeroNow.Status != SeriesUnavailable || resZeroNow.UnavailableReason != ReasonIncomplete {
			t.Fatalf("zero now: got status=%v reason=%v", resZeroNow.Status, resZeroNow.UnavailableReason)
		}

		resZeroAge := ProduceLiveSeries(now, 0, nil, input)
		if resZeroAge.Status != SeriesUnavailable || resZeroAge.UnavailableReason != ReasonIncomplete {
			t.Fatalf("zero maxAge: got status=%v reason=%v", resZeroAge.Status, resZeroAge.UnavailableReason)
		}
	})

	t.Run("FutureObservedAt", func(t *testing.T) {
		input := LiveRunInput{
			ObservedAt:        now.Add(10 * time.Second),
			Baseline:          baseline,
			Candidate:         candidate,
			SupervisionStatus: SupervisionActive,
		}
		res := ProduceLiveSeries(now, 5*time.Minute, nil, input)
		if res.Status != SeriesUnavailable || res.UnavailableReason != ReasonIncomplete {
			t.Fatalf("future observed: got status=%v reason=%v", res.Status, res.UnavailableReason)
		}
	})

	t.Run("ZeroObservedAt", func(t *testing.T) {
		input := LiveRunInput{
			ObservedAt:        time.Time{},
			Baseline:          baseline,
			Candidate:         candidate,
			SupervisionStatus: SupervisionActive,
		}
		res := ProduceLiveSeries(now, 5*time.Minute, nil, input)
		if res.Status != SeriesUnavailable || res.UnavailableReason != ReasonIncomplete {
			t.Fatalf("zero observed: got status=%v reason=%v", res.Status, res.UnavailableReason)
		}
	})

	t.Run("InvalidEnvelopes", func(t *testing.T) {
		badEnv := env
		badEnv.ModuleRev = "invalid-no-rev"
		badBase := makeTestObservation(now.Add(-10*time.Second), badEnv, 1.0)

		inputBadBase := LiveRunInput{
			ObservedAt:        now.Add(-2 * time.Second),
			Baseline:          badBase,
			Candidate:         candidate,
			SupervisionStatus: SupervisionActive,
		}
		resBase := ProduceLiveSeries(now, 5*time.Minute, nil, inputBadBase)
		if resBase.Status != SeriesUnavailable || resBase.UnavailableReason != ReasonIncomplete {
			t.Fatalf("invalid base envelope: got status=%v reason=%v", resBase.Status, resBase.UnavailableReason)
		}

		inputBadCand := LiveRunInput{
			ObservedAt:        now.Add(-2 * time.Second),
			Baseline:          baseline,
			Candidate:         badBase,
			SupervisionStatus: SupervisionActive,
		}
		resCand := ProduceLiveSeries(now, 5*time.Minute, nil, inputBadCand)
		if resCand.Status != SeriesUnavailable || resCand.UnavailableReason != ReasonIncomplete {
			t.Fatalf("invalid candidate envelope: got status=%v reason=%v", resCand.Status, resCand.UnavailableReason)
		}
	})

	t.Run("DegradedSupervisionProducesSeries", func(t *testing.T) {
		evaluator := New(DefaultThresholds())
		cand1 := makeTestObservation(now.Add(-4*time.Second), env, 1.0)
		evaluator.Observe(now.Add(-3*time.Second), baseline, cand1)
		cand2 := makeTestObservation(now.Add(-3*time.Second), env, 1.0)
		evaluator.Observe(now.Add(-2*time.Second), baseline, cand2)

		input := LiveRunInput{
			ObservedAt:        now.Add(-2 * time.Second),
			Baseline:          baseline,
			Candidate:         candidate,
			SupervisionStatus: SupervisionDegraded,
		}
		res := ProduceLiveSeries(now, 5*time.Minute, evaluator, input)
		if res.Status != SeriesProduced {
			t.Fatalf("degraded supervision expected produced status, got %v (reason: %v)", res.Status, res.UnavailableReason)
		}
		if res.LifecycleStatus != LifecycleGood {
			t.Fatalf("expected lifecycle status good, got %v", res.LifecycleStatus)
		}
	})

	t.Run("LifecycleAnnotationsRendered", func(t *testing.T) {
		events := []LifecycleEvent{EventPromotion, EventRevert, EventRelease}
		for _, evt := range events {
			evaluator := New(DefaultThresholds())
			cand1 := makeTestObservation(now.Add(-4*time.Second), env, 1.0)
			evaluator.Observe(now.Add(-3*time.Second), baseline, cand1)
			cand2 := makeTestObservation(now.Add(-3*time.Second), env, 1.0)
			evaluator.Observe(now.Add(-2*time.Second), baseline, cand2)

			input := LiveRunInput{
				ObservedAt:        now.Add(-2 * time.Second),
				Baseline:          baseline,
				Candidate:         candidate,
				SupervisionStatus: SupervisionActive,
				Lifecycle: LifecycleAnnotation{
					Event:   evt,
					Release: "v1.0.0-rc1",
				},
			}
			res := ProduceLiveSeries(now, 5*time.Minute, evaluator, input)
			if res.Status != SeriesProduced {
				t.Fatalf("event %s: expected produced status, got %v", evt, res.Status)
			}
			expectedSub := string(evt)
			if !strings.Contains(res.Prometheus, `event="`+expectedSub+`"`) {
				t.Fatalf("event %s metric missing from output:\n%s", evt, res.Prometheus)
			}
			if !strings.Contains(res.Prometheus, `release="v1.0.0-rc1"`) {
				t.Fatalf("release tag missing from output:\n%s", res.Prometheus)
			}
		}
	})
}

func TestPrometheusOutputFormatAndStructure(t *testing.T) {
	now := time.Date(2026, 9, 4, 14, 0, 0, 0, time.UTC)
	env := makeTestEnvelope("cuda")
	baseline := makeTestObservation(now.Add(-10*time.Second), env, 1.0)
	candidate := makeTestObservation(now, env, 1.0)

	evaluator := New(DefaultThresholds())
	cand1 := makeTestObservation(now.Add(-3*time.Second), env, 1.0)
	evaluator.Observe(now.Add(-2*time.Second), baseline, cand1)
	cand2 := makeTestObservation(now.Add(-2*time.Second), env, 1.0)
	evaluator.Observe(now.Add(-1*time.Second), baseline, cand2)
	res, err := evaluator.Observe(now, baseline, candidate)
	if err != nil {
		t.Fatalf("observe failed: %v", err)
	}

	rendered := RenderPrometheus(res)

	expectedHeaders := []string{
		"# HELP fak_native_slo_state Current debounced native performance SLO state.",
		"# TYPE fak_native_slo_state gauge",
		"# HELP fak_native_slo_value Latest available native performance objective value.",
		"# TYPE fak_native_slo_value gauge",
		"# HELP fak_native_slo_ratio Candidate divided by matched-envelope baseline.",
		"# TYPE fak_native_slo_ratio gauge",
		"# HELP fak_native_slo_violation Whether an available objective currently violates its SLO.",
		"# TYPE fak_native_slo_violation gauge",
	}
	for _, hdr := range expectedHeaders {
		if !strings.Contains(rendered, hdr) {
			t.Fatalf("rendered output missing header: %s\nFull output:\n%s", hdr, rendered)
		}
	}

	expectedLabels := `engine="fak-native",module_rev="internal/nativeperf@r50+gabcdef123",benchmark_envelope="qwen38-4b-in128-out128-b1-quality-v3",model="Qwen3.8-4B",backend="cuda"`
	if !strings.Contains(rendered, expectedLabels) {
		t.Fatalf("rendered output missing expected labels: %s\nFull output:\n%s", expectedLabels, rendered)
	}

	if !strings.Contains(rendered, `state="good"} 1`) {
		t.Fatalf("expected state=good to be 1")
	}
	if !strings.Contains(rendered, `state="regression"} 0`) {
		t.Fatalf("expected state=regression to be 0")
	}
	if !strings.Contains(rendered, `state="missing_evidence"} 0`) {
		t.Fatalf("expected state=missing_evidence to be 0")
	}

	idxCacheEff := strings.Index(rendered, `objective="cache_efficiency"`)
	idxTTFT := strings.Index(rendered, `objective="ttft"`)
	if idxCacheEff == -1 || idxTTFT == -1 || idxCacheEff > idxTTFT {
		t.Fatalf("expected alphabetical ordering of objectives (cache_efficiency before ttft)")
	}
}

func TestEnvelopeValidationConstraints(t *testing.T) {
	cases := []struct {
		name    string
		env     Envelope
		wantErr bool
		errMsg  string
	}{
		{
			name: "valid metal",
			env: Envelope{
				ModuleRev: "internal/nativeperf@r41+g123",
				Benchmark: "envelope-v1",
				Model:     "Qwen3.8-7B",
				Backend:   "metal",
			},
			wantErr: false,
		},
		{
			name: "valid cuda",
			env: Envelope{
				ModuleRev: "internal/nativeperf@r41+g123",
				Benchmark: "envelope-v1",
				Model:     "Qwen3.8-14B",
				Backend:   "cuda",
			},
			wantErr: false,
		},
		{
			name: "valid cpu",
			env: Envelope{
				ModuleRev: "internal/nativeperf@r41+g123",
				Benchmark: "envelope-v1",
				Model:     "Qwen3.8-0.5B",
				Backend:   "cpu",
			},
			wantErr: false,
		},
		{
			name: "missing at-r in module rev",
			env: Envelope{
				ModuleRev: "internal/nativeperf-plain-sha",
				Benchmark: "envelope-v1",
				Model:     "Qwen3.8-7B",
				Backend:   "cuda",
			},
			wantErr: true,
			errMsg:  "module_rev must be module@rev",
		},
		{
			name: "empty benchmark",
			env: Envelope{
				ModuleRev: "internal/nativeperf@r41+g123",
				Benchmark: "   ",
				Model:     "Qwen3.8-7B",
				Backend:   "cuda",
			},
			wantErr: true,
			errMsg:  "benchmark envelope is required",
		},
		{
			name: "non-qwen38 model",
			env: Envelope{
				ModuleRev: "internal/nativeperf@r41+g123",
				Benchmark: "envelope-v1",
				Model:     "Qwen2.5-7B",
				Backend:   "cuda",
			},
			wantErr: true,
			errMsg:  "native performance evidence must use Qwen3.8",
		},
		{
			name: "unsupported backend rocm",
			env: Envelope{
				ModuleRev: "internal/nativeperf@r41+g123",
				Benchmark: "envelope-v1",
				Model:     "Qwen3.8-7B",
				Backend:   "rocm",
			},
			wantErr: true,
			errMsg:  "unsupported fak-native backend",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.env.Validate()
			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected error, got nil")
				}
				if !strings.Contains(err.Error(), tc.errMsg) {
					t.Fatalf("error %q does not contain %q", err.Error(), tc.errMsg)
				}
			} else {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
			}
		})
	}
}

func TestThresholdSanitizationDefaults(t *testing.T) {
	dirty := Thresholds{
		MaxRegression:      -1.0,
		MinRetention:       -0.5,
		MaxMemoryPressure:  0.0,
		MaxEvidenceAge:     -time.Second,
		MinReceiptCoverage: 2.0,
		Consecutive:        0,
	}
	evaluator := New(dirty)
	def := DefaultThresholds()

	if evaluator.thresholds.MaxRegression != def.MaxRegression {
		t.Fatalf("MaxRegression = %v, want %v", evaluator.thresholds.MaxRegression, def.MaxRegression)
	}
	if evaluator.thresholds.MinRetention != def.MinRetention {
		t.Fatalf("MinRetention = %v, want %v", evaluator.thresholds.MinRetention, def.MinRetention)
	}
	if evaluator.thresholds.MaxMemoryPressure != def.MaxMemoryPressure {
		t.Fatalf("MaxMemoryPressure = %v, want %v", evaluator.thresholds.MaxMemoryPressure, def.MaxMemoryPressure)
	}
	if evaluator.thresholds.MaxEvidenceAge != def.MaxEvidenceAge {
		t.Fatalf("MaxEvidenceAge = %v, want %v", evaluator.thresholds.MaxEvidenceAge, def.MaxEvidenceAge)
	}
	if evaluator.thresholds.MinReceiptCoverage != def.MinReceiptCoverage {
		t.Fatalf("MinReceiptCoverage = %v, want %v", evaluator.thresholds.MinReceiptCoverage, def.MinReceiptCoverage)
	}
	if evaluator.thresholds.Consecutive != def.Consecutive {
		t.Fatalf("Consecutive = %v, want %v", evaluator.thresholds.Consecutive, def.Consecutive)
	}
}

func BenchmarkNativePerfSLOLifecycle(b *testing.B) {
	now := time.Date(2026, 9, 4, 12, 0, 0, 0, time.UTC)
	env := makeTestEnvelope("cuda")
	baseline := makeTestObservation(now, env, 1.0)
	candidate := makeTestObservation(now.Add(time.Second), env, 1.0)
	evaluator := New(DefaultThresholds())

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		ts := now.Add(time.Duration(i) * time.Second)
		candidate.At = ts
		_, err := evaluator.Observe(ts, baseline, candidate)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkProduceLiveSeries(b *testing.B) {
	now := time.Date(2026, 9, 4, 12, 0, 0, 0, time.UTC)
	env := makeTestEnvelope("metal")
	baseline := makeTestObservation(now.Add(-10*time.Second), env, 1.0)
	candidate := makeTestObservation(now.Add(-time.Second), env, 1.0)
	evaluator := New(DefaultThresholds())
	input := LiveRunInput{
		ObservedAt:        now.Add(-time.Second),
		Baseline:          baseline,
		Candidate:         candidate,
		SupervisionStatus: SupervisionActive,
		Lifecycle: LifecycleAnnotation{
			Event:   EventRelease,
			Release: "v0.50.0",
		},
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		res := ProduceLiveSeries(now, 5*time.Minute, evaluator, input)
		if res.Status != SeriesProduced {
			b.Fatalf("benchmark failed: %v", res.UnavailableReason)
		}
	}
}

func BenchmarkRenderPrometheus(b *testing.B) {
	now := time.Date(2026, 9, 4, 12, 0, 0, 0, time.UTC)
	env := makeTestEnvelope("cuda")
	baseline := makeTestObservation(now, env, 1.0)
	candidate := makeTestObservation(now, env, 1.0)
	evaluator := New(DefaultThresholds())
	res, err := evaluator.Observe(now, baseline, candidate)
	if err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		text := RenderPrometheus(res)
		if len(text) == 0 {
			b.Fatal("rendered empty metrics")
		}
	}
}

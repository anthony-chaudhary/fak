package headroom

import (
	"encoding/json"
	"math"
	"reflect"
	"strings"
	"testing"
)

func validLiveEvidence() LiveComparisonEvidence {
	metrics := LiveArmMetrics{
		TaskSuccess: 1, MetricFactRecall: 1, ProviderInputTokens: 100,
		TTFTMilliseconds: 10, RegrowthTaxTokens: 0, TotalCostUSD: 0.01,
	}
	return LiveComparisonEvidence{
		Schema: "fak-headroom-live-evidence/1", Witness: "ledger://independent/run-1",
		WorkloadDigest: "sha256:abc", Model: "model-v1", Provider: "provider-v1",
		CacheState: "warm-prefix", Grader: "grader-v1",
		Arms: map[string]LiveArmMetrics{"none": metrics, NativeName: metrics},
	}
}

func ptr[T any](value T) *T { return &value }

func validPromotionEvidence() LiveComparisonEvidence {
	provenance := func(witness string) PromotionProvenance {
		return PromotionProvenance{
			Witness: witness, WorkloadDigest: "sha256:" + strings.Repeat("a", 64), Model: "model-v1", Provider: "provider-v1",
			Seed: ptr[int64](0), Temperature: ptr(0.0), OutputLimit: ptr[int64](1024), CacheState: "warm",
			GraderID: "fact-task-grader", GraderVersion: "v1",
		}
	}
	controlLegacy := LiveArmMetrics{TaskSuccess: 0.9, MetricFactRecall: 0.8, ProviderInputTokens: 1000, TTFTMilliseconds: 100}
	nativeLegacy := LiveArmMetrics{TaskSuccess: 0.9, MetricFactRecall: 0.8, ProviderInputTokens: 800, TTFTMilliseconds: 100, RegrowthTaxTokens: 50}
	return LiveComparisonEvidence{
		Schema: "fak-headroom-live-evidence/1", Witness: "ledger://matched/run", WorkloadDigest: "sha256:" + strings.Repeat("a", 64),
		Model: "model-v1", Provider: "provider-v1", CacheState: "warm", Grader: "fact-task-grader@v1",
		Arms: map[string]LiveArmMetrics{PromotionControlName: controlLegacy, NativeName: nativeLegacy}, PromotionArms: []PromotionArmEvidence{
			{Name: PromotionControlName, CaseIDs: []string{"case-a", "case-b"}, Provenance: provenance("ledger://none"), Metrics: PromotionArmMetrics{
				TaskSuccess: 0.9, RetainedFactRecall: 0.8, InitialInputTokens: 1000, EffectiveInputTokens: 1000,
				P95ResultToResponseMilliseconds: 100,
			}},
			{Name: NativeName, CaseIDs: []string{"case-a", "case-b"}, Provenance: provenance("ledger://native"), Metrics: PromotionArmMetrics{
				TaskSuccess: 0.9, RetainedFactRecall: 0.8, InitialInputTokens: 800, RegrowthInputTokens: 50,
				RefetchInputTokens: 25, OverrideInputTokens: 25, EffectiveInputTokens: 900,
				P95ResultToResponseMilliseconds: 105,
			}},
		},
	}
}

func TestHeadroomPromotionDecisionMatchedPromote(t *testing.T) {
	got := DecideNativePromotion(validPromotionEvidence())
	if got.Verdict != PromotionPromote || len(got.Reasons) != 0 {
		t.Fatalf("decision=%+v, want promote at exact token/latency boundaries", got)
	}
}

func TestHeadroomPromotionDecisionMutationHolds(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*LiveComparisonEvidence)
		want   PromotionHoldReason
	}{
		{"reordered_arms", func(e *LiveComparisonEvidence) {
			e.PromotionArms[0], e.PromotionArms[1] = e.PromotionArms[1], e.PromotionArms[0]
		}, HoldReorderedArms},
		{"duplicate_arms", func(e *LiveComparisonEvidence) { e.PromotionArms[1].Name = PromotionControlName }, HoldDuplicateArms},
		{"incomplete_arms", func(e *LiveComparisonEvidence) { e.PromotionArms = e.PromotionArms[:1] }, HoldIncompleteArms},
		{"reordered_cases", func(e *LiveComparisonEvidence) {
			e.PromotionArms[1].CaseIDs[0], e.PromotionArms[1].CaseIDs[1] = e.PromotionArms[1].CaseIDs[1], e.PromotionArms[1].CaseIDs[0]
		}, HoldReorderedCases},
		{"missing_cases", func(e *LiveComparisonEvidence) { e.PromotionArms[1].CaseIDs = nil }, HoldMissingCases},
		{"partial_cases", func(e *LiveComparisonEvidence) { e.PromotionArms[1].CaseIDs = e.PromotionArms[1].CaseIDs[:1] }, HoldMissingCases},
		{"duplicate_cases", func(e *LiveComparisonEvidence) { e.PromotionArms[1].CaseIDs[1] = e.PromotionArms[1].CaseIDs[0] }, HoldDuplicateCases},
		{"oversized_case_id", func(e *LiveComparisonEvidence) { e.PromotionArms[1].CaseIDs[0] = strings.Repeat("x", 129) }, HoldInvalidCases},
		{"control_case_id", func(e *LiveComparisonEvidence) { e.PromotionArms[1].CaseIDs[0] = "case\nraw-output" }, HoldInvalidCases},
		{"whitespace_case_id", func(e *LiveComparisonEvidence) { e.PromotionArms[1].CaseIDs[0] = "case raw output" }, HoldInvalidCases},
		{"missing_provenance", func(e *LiveComparisonEvidence) { e.PromotionArms[1].Provenance.GraderVersion = "" }, HoldMissingProvenance},
		{"provenance_mismatch", func(e *LiveComparisonEvidence) { e.PromotionArms[1].Provenance.CacheState = "cold" }, HoldProvenanceMismatch},
		{"negative_counter", func(e *LiveComparisonEvidence) { e.PromotionArms[1].Metrics.RefetchInputTokens = -1 }, HoldNegativeCounter},
		{"non_finite", func(e *LiveComparisonEvidence) {
			e.PromotionArms[1].Metrics.P95ResultToResponseMilliseconds = math.NaN()
		}, HoldNonFiniteMetric},
		{"rate_out_of_range", func(e *LiveComparisonEvidence) { e.PromotionArms[1].Metrics.TaskSuccess = 1.01 }, HoldMetricOutOfRange},
		{"overflow", func(e *LiveComparisonEvidence) {
			e.PromotionArms[1].Metrics.InitialInputTokens = math.MaxInt64
			e.PromotionArms[1].Metrics.RegrowthInputTokens = 1
			e.PromotionArms[1].Metrics.EffectiveInputTokens = math.MaxInt64
		}, HoldCounterOverflow},
		{"non_conserving", func(e *LiveComparisonEvidence) { e.PromotionArms[1].Metrics.EffectiveInputTokens++ }, HoldCounterNonConserving},
		{"recovery_tax_reversal", func(e *LiveComparisonEvidence) {
			e.PromotionArms[1].Metrics.RegrowthInputTokens = 200
			e.PromotionArms[1].Metrics.RefetchInputTokens = 0
			e.PromotionArms[1].Metrics.OverrideInputTokens = 0
			e.PromotionArms[1].Metrics.EffectiveInputTokens = 1000
		}, HoldEffectiveInputTooHigh},
		{"quality_regression", func(e *LiveComparisonEvidence) { e.PromotionArms[1].Metrics.TaskSuccess = 0.89 }, HoldTaskSuccessRegression},
		{"recall_regression", func(e *LiveComparisonEvidence) { e.PromotionArms[1].Metrics.RetainedFactRecall = 0.79 }, HoldFactRecallRegression},
		{"latency_regression", func(e *LiveComparisonEvidence) { e.PromotionArms[1].Metrics.P95ResultToResponseMilliseconds = 105.01 }, HoldLatencyRegression},
		{"restoration_failure", func(e *LiveComparisonEvidence) { e.PromotionArms[1].Metrics.ExactOriginalRestorationFailures = 1 }, HoldRestorationFailure},
		{"control_restoration_failure", func(e *LiveComparisonEvidence) { e.PromotionArms[0].Metrics.ExactOriginalRestorationFailures = 1 }, HoldRestorationFailure},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			evidence := validPromotionEvidence()
			tc.mutate(&evidence)
			got := DecideNativePromotion(evidence)
			if got.Verdict != PromotionHold || len(got.Reasons) == 0 || got.Reasons[0] != tc.want {
				t.Fatalf("decision=%+v, want first reason %q", got, tc.want)
			}
		})
	}
}

func TestHeadroomPromotionDecisionStrictDeterministicPayloadFreeJSON(t *testing.T) {
	evidence := validPromotionEvidence()
	raw, err := json.Marshal(evidence)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(strings.ToLower(string(raw)), "payload") || strings.Contains(string(raw), "raw result") {
		t.Fatalf("receipt persisted a raw-payload surface: %s", raw)
	}
	var decoded LiveComparisonEvidence
	err = json.Unmarshal(raw, &decoded)
	if err != nil || !reflect.DeepEqual(decoded, evidence) {
		t.Fatalf("strict round trip: decoded=%+v err=%v", decoded, err)
	}
	a, _ := json.Marshal(DecideNativePromotion(decoded))
	b, _ := json.Marshal(DecideNativePromotion(decoded))
	if !reflect.DeepEqual(a, b) {
		t.Fatalf("decision JSON is nondeterministic:\n%s\n%s", a, b)
	}
	for name, bad := range map[string][]byte{
		"unknown_payload": []byte(`{"schema":"fak-headroom-live-evidence/1","raw_result_payload":"secret","arms":{}}`),
		"duplicate_key":   []byte(`{"schema":"fak-headroom-live-evidence/1","schema":"fak-headroom-live-evidence/1","arms":{}}`),
	} {
		t.Run(name, func(t *testing.T) {
			var got LiveComparisonEvidence
			if err := json.Unmarshal(bad, &got); err == nil {
				t.Fatal("strict decoder accepted invalid receipt")
			}
		})
	}
}

func TestHeadroomPromotionDecisionDecodeRequiresEveryField(t *testing.T) {
	deleteAndDecode := func(t *testing.T, objectName, field string) {
		t.Helper()
		raw, _ := json.Marshal(validPromotionEvidence())
		var root map[string]any
		if err := json.Unmarshal(raw, &root); err != nil {
			t.Fatal(err)
		}
		arm := root["promotion_arms"].([]any)[0].(map[string]any)
		var object map[string]any
		switch objectName {
		case "arm":
			object = arm
		case "provenance":
			object = arm["provenance"].(map[string]any)
		case "metrics":
			object = arm["metrics"].(map[string]any)
		default:
			t.Fatalf("unknown object %q", objectName)
		}
		delete(object, field)
		bad, _ := json.Marshal(root)
		var got LiveComparisonEvidence
		if err := json.Unmarshal(bad, &got); err == nil {
			t.Fatalf("decoder accepted omitted %s.%s", objectName, field)
		}
	}

	for _, field := range []string{"provenance", "metrics"} {
		t.Run("object_"+field, func(t *testing.T) { deleteAndDecode(t, "arm", field) })
	}
	for _, field := range []string{"witness", "workload_digest", "model", "provider", "seed", "temperature", "output_limit", "cache_state", "grader_id", "grader_version"} {
		field := field
		t.Run("provenance_"+field, func(t *testing.T) { deleteAndDecode(t, "provenance", field) })
	}
	for _, field := range []string{
		"task_success", "retained_fact_recall", "initial_input_tokens", "regrowth_input_tokens",
		"refetch_input_tokens", "override_input_tokens", "effective_input_tokens",
		"p95_result_to_response_ms", "exact_original_restoration_failures",
	} {
		field := field
		t.Run("metrics_"+field, func(t *testing.T) { deleteAndDecode(t, "metrics", field) })
	}
}

func TestHeadroomPromotionDecisionDecodeAcceptsTypedPerformanceHold(t *testing.T) {
	evidence := validPromotionEvidence()
	evidence.PromotionArms[1].Metrics.P95ResultToResponseMilliseconds = 106
	raw, _ := json.Marshal(evidence)
	var decoded LiveComparisonEvidence
	err := json.Unmarshal(raw, &decoded)
	if err != nil {
		t.Fatalf("valid matched hold receipt was rejected: %v", err)
	}
	got := DecideNativePromotion(decoded)
	if got.Verdict != PromotionHold || !reflect.DeepEqual(got.Reasons, []PromotionHoldReason{HoldLatencyRegression}) {
		t.Fatalf("decision=%+v, want typed latency hold", got)
	}
}

func TestHeadroomPromotionDecisionRejectsProvenanceStuffing(t *testing.T) {
	tests := map[string]func(*LiveComparisonEvidence){
		"oversized_witness": func(e *LiveComparisonEvidence) {
			e.PromotionArms[1].Provenance.Witness = "ledger://" + strings.Repeat("x", 300)
		},
		"bad_digest":        func(e *LiveComparisonEvidence) { e.PromotionArms[1].Provenance.WorkloadDigest = "sha256:not-a-digest" },
		"control_character": func(e *LiveComparisonEvidence) { e.PromotionArms[1].Provenance.Model = "model\nraw-result-stuffing" },
	}
	for name, mutate := range tests {
		t.Run(name, func(t *testing.T) {
			evidence := validPromotionEvidence()
			mutate(&evidence)
			got := DecideNativePromotion(evidence)
			if got.Verdict != PromotionHold || len(got.Reasons) == 0 || got.Reasons[0] != HoldInvalidProvenance {
				t.Fatalf("decision=%+v, want invalid provenance", got)
			}
		})
	}
}

func TestHeadroomPromotionDecisionAbsentEvidenceHolds(t *testing.T) {
	got := DecideNativePromotion(LiveComparisonEvidence{Schema: "fak-headroom-live-evidence/1"})
	if got.Verdict != PromotionHold || !reflect.DeepEqual(got.Reasons, []PromotionHoldReason{HoldIncompleteArms}) {
		t.Fatalf("absent evidence decision=%+v", got)
	}
}

func TestApplyLiveEvidenceCompletesOnlyFullIndependentReadback(t *testing.T) {
	report := CompareBench([]string{"none", NativeName}, BenchCorpus())
	got, err := ApplyLiveEvidence(report, validLiveEvidence())
	if err != nil {
		t.Fatal(err)
	}
	if !got.Complete || len(got.Pending) != 0 || len(got.Measured) != len(requiredComparisonMetrics) {
		t.Fatalf("joined report=%+v", got)
	}
}

func TestApplyLiveEvidencePromotionIntegration(t *testing.T) {
	report := func() ComparisonReport { return CompareBench([]string{"none", NativeName}, BenchCorpus()) }
	t.Run("promote", func(t *testing.T) {
		got, err := ApplyLiveEvidence(report(), validPromotionEvidence())
		if err != nil || got.LiveEvidence == nil || got.LiveEvidence.Decision == nil || got.LiveEvidence.Decision.Verdict != PromotionPromote {
			t.Fatalf("report=%+v err=%v", got, err)
		}
	})
	t.Run("typed_hold", func(t *testing.T) {
		evidence := validPromotionEvidence()
		evidence.PromotionArms[1].Metrics.P95ResultToResponseMilliseconds = 106
		got, err := ApplyLiveEvidence(report(), evidence)
		if err != nil || got.LiveEvidence == nil || got.LiveEvidence.Decision == nil ||
			!reflect.DeepEqual(got.LiveEvidence.Decision.Reasons, []PromotionHoldReason{HoldLatencyRegression}) {
			t.Fatalf("report=%+v err=%v", got, err)
		}
	})
	t.Run("malformed_refused", func(t *testing.T) {
		evidence := validPromotionEvidence()
		evidence.PromotionArms[1].Metrics.EffectiveInputTokens++
		if got, err := ApplyLiveEvidence(report(), evidence); err == nil || got.Complete {
			t.Fatalf("malformed promotion evidence accepted: report=%+v err=%v", got, err)
		}
	})
	t.Run("legacy_promotion_coherence", func(t *testing.T) {
		tests := map[string]func(*LiveComparisonEvidence){
			"task_success": func(e *LiveComparisonEvidence) {
				m := e.Arms[NativeName]
				m.TaskSuccess = 0.8
				e.Arms[NativeName] = m
			},
			"retained_fact_recall": func(e *LiveComparisonEvidence) {
				m := e.Arms[NativeName]
				m.MetricFactRecall = 0.7
				e.Arms[NativeName] = m
			},
			"initial_input_tokens": func(e *LiveComparisonEvidence) {
				m := e.Arms[NativeName]
				m.ProviderInputTokens++
				e.Arms[NativeName] = m
			},
			"regrowth_input_tokens": func(e *LiveComparisonEvidence) {
				m := e.Arms[NativeName]
				m.RegrowthTaxTokens++
				e.Arms[NativeName] = m
			},
			"different_arm_set": func(e *LiveComparisonEvidence) {
				delete(e.Arms, NativeName)
				e.Arms["other"] = LiveArmMetrics{}
			},
		}
		for name, mutate := range tests {
			t.Run(name, func(t *testing.T) {
				evidence := validPromotionEvidence()
				mutate(&evidence)
				if got, err := ApplyLiveEvidence(report(), evidence); err == nil || got.Complete {
					t.Fatalf("incoherent evidence accepted: report=%+v err=%v", got, err)
				}
			})
		}
	})
	t.Run("runtime_noop", func(t *testing.T) {
		withSelected(t, NoopName)
		if _, err := ApplyLiveEvidence(report(), validPromotionEvidence()); err != nil {
			t.Fatal(err)
		}
		if got := Selected().Name(); got != NoopName {
			t.Fatalf("runtime selected %q", got)
		}
	})
}

func TestApplyLiveEvidenceRejectsPartialOrInvalidReadback(t *testing.T) {
	report := CompareBench([]string{"none", NativeName}, BenchCorpus())
	for name, mutate := range map[string]func(*LiveComparisonEvidence){
		"no witness":  func(e *LiveComparisonEvidence) { e.Witness = "" },
		"missing arm": func(e *LiveComparisonEvidence) { delete(e.Arms, NativeName) },
		"invalid recall": func(e *LiveComparisonEvidence) {
			m := e.Arms[NativeName]
			m.MetricFactRecall = 1.1
			e.Arms[NativeName] = m
		},
	} {
		t.Run(name, func(t *testing.T) {
			evidence := validLiveEvidence()
			mutate(&evidence)
			got, err := ApplyLiveEvidence(report, evidence)
			if err == nil || got.Complete {
				t.Fatalf("err=%v report=%+v", err, got)
			}
			if strings.TrimSpace(err.Error()) == "" {
				t.Fatal("empty diagnostic")
			}
		})
	}
}

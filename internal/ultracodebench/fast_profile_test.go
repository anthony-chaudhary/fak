package ultracodebench

import "testing"

func fastSample(id, tier string, latency int64) FastSample {
	return FastSample{RunID: id, RequestedTier: tier, ResolvedTier: tier, RealizedTier: tier, Model: "eligible-model", ReasoningEffort: "high", OutputMode: "concise", TTFTMS: 100, TPOTMS: 10, OutputTokensPerSecond: 100, EndToEndMS: latency, CriticalPathMS: latency - 10, PromptTokens: 1000, CacheReadTokens: 500, CachePosture: "warm", CostUSD: .1, CostAuthority: "provider-receipt", TimingAuthority: "harness-monotonic-clock", WorkerWidth: 2, TotalWorkerMS: latency * 2, LeaseWaitMS: 5, InvalidationMS: 2, ReconcileMS: 8, AcceptancePassed: true, OutcomeDigest: "sha256:outcome", WitnessDigest: "sha256:witness-" + id, AcceptanceAuthority: "independent-test-runner"}
}
func validFastBundle() FastProfileBundle {
	return FastProfileBundle{Schema: FastProfileSchema, Scenario: "fast-profile", Task: FastTask{ID: "pinned-task", Digest: "sha256:task", Environment: "linux-amd64", ModelEligibility: "same-capability-class", HardBudgetMS: 10000, AcceptanceTests: []string{"go test ./..."}}, Noise: RepeatRule{MinimumSamples: 3, RelativeThreshold: .05, Statistic: "median", StoppingRule: "three runs per arm"}, Comparisons: []FastComparison{{Binding: FastBinding{ID: "binding-a", Provider: "provider-a", Harness: "harness-a"}, Standard: FastArm{Profile: "tuned-standard", Samples: []FastSample{fastSample("s1", "standard", 1000), fastSample("s2", "standard", 1020), fastSample("s3", "standard", 980)}}, Fast: FastArm{Profile: "resolved-fast", Samples: []FastSample{fastSample("f1", "fast", 700), fastSample("f2", "fast", 720), fastSample("f3", "fast", 680)}}}}, FactorCells: []FastFactorCell{{ID: "tier", ProviderTier: "fast", ModelMode: "fixed", CachePosture: "warm", WorkerWidth: 2}, {ID: "model", ProviderTier: "standard", ModelMode: "high-concise", CachePosture: "warm", WorkerWidth: 2}, {ID: "cache", ProviderTier: "standard", ModelMode: "fixed", CachePosture: "cold", WorkerWidth: 2}, {ID: "width", ProviderTier: "standard", ModelMode: "fixed", CachePosture: "warm", WorkerWidth: 4}}}
}
func TestEvaluateFastProfileGainAndDeterministicNormalization(t *testing.T) {
	b := validFastBundle()
	got := EvaluateFastProfile(b)
	if got.Verdict != "GAIN" {
		t.Fatalf("verdict=%s reasons=%v", got.Verdict, got.Reasons)
	}
	if got.Comparisons[0].Standard.MedianEndToEndMS != 1000 || got.Comparisons[0].Fast.MedianEndToEndMS != 700 {
		t.Fatalf("distributions=%+v", got.Comparisons[0])
	}
	if got.BundleDigest != EvaluateFastProfile(b).BundleDigest {
		t.Fatal("bundle digest is not deterministic")
	}
}
func TestEvaluateFastProfileAbstainsForIncompleteOrUnequalEnvelope(t *testing.T) {
	b := validFastBundle()
	b.Comparisons[0].Fast.Samples[0].CostAuthority = ""
	b.Comparisons[0].Fast.Samples[0].OutcomeDigest = "other"
	b.Comparisons[0].Fast.Samples[0].CachePosture = "cold"
	got := EvaluateFastProfile(b)
	if got.Verdict != "ABSTAIN" {
		t.Fatalf("verdict=%s", got.Verdict)
	}
	if len(got.Reasons) < 3 {
		t.Fatalf("missing abstain reasons: %v", got.Reasons)
	}
}

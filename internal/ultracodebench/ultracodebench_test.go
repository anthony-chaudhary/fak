package ultracodebench

import (
	"slices"
	"testing"
)

func TestEvaluateGain(t *testing.T) {
	p := fixture()
	r, err := Evaluate(p)
	if err != nil {
		t.Fatal(err)
	}
	if r.Verdict != "GAIN" {
		t.Fatalf("verdict=%s reasons=%v", r.Verdict, r.Reasons)
	}
	if r.Gains.ConcurrencySpeedup != 2 {
		t.Fatalf("speedup=%v", r.Gains.ConcurrencySpeedup)
	}
	if r.Gains.BilledTokenReduction <= 0 {
		t.Fatalf("token reduction=%v", r.Gains.BilledTokenReduction)
	}
	if r.Fleet.CacheReadTokens == 0 || r.Fleet.AcceptedEffects != r.Single.AcceptedEffects {
		t.Fatalf("report=%+v", r)
	}
}

func TestEvaluateAbstainsOnUnequalOrNoisyPair(t *testing.T) {
	for name, mutate := range map[string]func(*Pair){
		"identity":          func(p *Pair) { p.Fleet.Identity.Model = "other" },
		"retry":             func(p *Pair) { p.Fleet.Retries = 1 },
		"failed acceptance": func(p *Pair) { p.Fleet.AcceptancePassed = false },
		"missing witness":   func(p *Pair) { p.Fleet.WitnessDigest = "" },
	} {
		t.Run(name, func(t *testing.T) {
			p := fixture()
			mutate(&p)
			r, err := Evaluate(p)
			if name == "identity" {
				if err == nil {
					t.Fatal("expected identity error")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if r.Verdict != "ABSTAIN" {
				t.Fatalf("verdict=%s", r.Verdict)
			}
		})
	}
}

func TestEvaluateAbstainsWhenTreatmentActivationIsUnverified(t *testing.T) {
	p := fixture()
	p.Fleet.Activation.Receipts[0].Observable = ObservableUnknown
	p.Fleet.Activation.Receipts[0].ObservationSource = ""
	r, err := Evaluate(p)
	if err != nil {
		t.Fatal(err)
	}
	if r.Verdict != "ABSTAIN" || r.Attribution != AttributionUnverified || !slices.Contains(r.Reasons, AttributionUnverified) {
		t.Fatalf("report=%+v", r)
	}
}

func TestEvaluateAbstainsWithoutDeclaredActivationThreshold(t *testing.T) {
	p := fixture()
	p.Fleet.Activation.MinimumActiveRatio = 0
	r, err := Evaluate(p)
	if err != nil {
		t.Fatal(err)
	}
	if r.Verdict != "ABSTAIN" || r.Attribution != AttributionUnverified || r.Activation.Treatment.Declared {
		t.Fatalf("report=%+v", r)
	}
}

func TestEvaluateNoGainWhenQualityDrops(t *testing.T) {
	p := fixture()
	p.Fleet.AcceptedEffects = 2
	r, err := Evaluate(p)
	if err != nil {
		t.Fatal(err)
	}
	if r.Verdict != "NO_GAIN" {
		t.Fatalf("verdict=%s", r.Verdict)
	}
}

func fixture() Pair {
	id := Identity{Task: "fix three independent defects", TaskDigest: "sha256:task", Model: "qwen3-coder:27b", Environment: "lab-l4", WallBudgetMS: 120000, TokenBudget: 20000, SpendBudgetUSD: 1}
	control, err := BeforeSpawn(BeforeSpawnInput{RunID: "pair-control", ChildID: "control", Harness: "codex", Requested: SettingOff, Resolved: SettingOff})
	if err != nil {
		panic(err)
	}
	treatment, err := BeforeSpawn(BeforeSpawnInput{RunID: "pair-treatment", ChildID: "treatment", Harness: "codex", Requested: SettingOn, Resolved: SettingOn, Injected: true})
	if err != nil {
		panic(err)
	}
	treatment, err = Acknowledge(treatment, ObservableActive, SourceRuntimeObservation)
	if err != nil {
		panic(err)
	}
	return Pair{Schema: Schema,
		Single: Run{Mode: "single", Identity: id, CriticalPathMS: 10000, TotalWorkerMS: 10000, InputTokens: 6000, OutputTokens: 1000, BilledTokens: 7000, SpendUSD: .07, ExpectedEffects: 3, AcceptedEffects: 3, AcceptancePassed: true, WitnessDigest: "sha256:single", Activation: ActivationCohort{Receipts: []ActivationReceipt{control}}},
		Fleet:  Run{Mode: "fleet", Identity: id, CriticalPathMS: 5000, TotalWorkerMS: 13000, InputTokens: 3000, OutputTokens: 900, CacheReadTokens: 5000, CacheWriteTokens: 100, BilledTokens: 4000, SpendUSD: .04, ExpectedEffects: 3, AcceptedEffects: 3, AcceptancePassed: true, WitnessDigest: "sha256:fleet", Activation: ActivationCohort{MinimumActiveRatio: 1, Receipts: []ActivationReceipt{treatment}}},
	}
}

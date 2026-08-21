package ultracodebench

import "testing"

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
	return Pair{Schema: Schema,
		Single: Run{Mode: "single", Identity: id, CriticalPathMS: 10000, TotalWorkerMS: 10000, InputTokens: 6000, OutputTokens: 1000, BilledTokens: 7000, SpendUSD: .07, ExpectedEffects: 3, AcceptedEffects: 3, AcceptancePassed: true, WitnessDigest: "sha256:single"},
		Fleet:  Run{Mode: "fleet", Identity: id, CriticalPathMS: 5000, TotalWorkerMS: 13000, InputTokens: 3000, OutputTokens: 900, CacheReadTokens: 5000, CacheWriteTokens: 100, BilledTokens: 4000, SpendUSD: .04, ExpectedEffects: 3, AcceptedEffects: 3, AcceptancePassed: true, WitnessDigest: "sha256:fleet"},
	}
}

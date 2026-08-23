package ultracodebench

import (
	"encoding/json"
	"os"
	"slices"
	"strings"
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
	if r.Gains.BilledTokenReduction == nil || *r.Gains.BilledTokenReduction <= 0 {
		t.Fatalf("token reduction=%v", r.Gains.BilledTokenReduction)
	}
	if r.Gains.OutcomePerUSDGain == nil || *r.Gains.OutcomePerUSDGain <= 0 {
		t.Fatalf("spend gain=%v", r.Gains.OutcomePerUSDGain)
	}
	if r.Fleet.CacheReadTokens == 0 || r.Fleet.AcceptedEffects != r.Single.AcceptedEffects {
		t.Fatalf("report=%+v", r)
	}
}

func TestEvaluateAbstainsWhenSubscriptionAccountingIsUnavailable(t *testing.T) {
	raw, err := os.ReadFile("testdata/accounting_subscription_unavailable.json")
	if err != nil {
		t.Fatal(err)
	}
	var pair Pair
	if err := json.Unmarshal(raw, &pair); err != nil {
		t.Fatal(err)
	}
	report, err := Evaluate(pair)
	if err != nil {
		t.Fatal(err)
	}
	if report.Verdict != "ABSTAIN" || !strings.Contains(strings.Join(report.Reasons, "\n"), "accounting") {
		t.Fatalf("unsupported cost comparison admitted: verdict=%s reasons=%v", report.Verdict, report.Reasons)
	}
	if report.Single.AcceptedPerBilledKToken != nil || report.Fleet.AcceptedPerBilledKToken != nil || report.Gains.BilledTokenReduction != nil || report.Gains.OutcomePerUSDGain != nil {
		t.Fatalf("unavailable accounting produced numeric cost metrics: %+v", report)
	}
}

func TestEvaluateAccountingCoverageAndAuthorityFailClosed(t *testing.T) {
	for name, test := range map[string]struct {
		mutate func(*Pair)
		reason string
	}{
		"partially covered fleet": {
			mutate: func(p *Pair) {
				p.CostComparison = CompareBilledTokens
				p.Fleet.Accounting.BilledTokens.Availability = AccountingPartial
				p.Fleet.Accounting.BilledTokens.Coverage = .5
				p.Fleet.Accounting.BilledTokens.Reason = "only one of two provider billing rows joined"
			},
			reason: "accounting_billed_tokens_partial",
		},
		"mismatched authority": {
			mutate: func(p *Pair) {
				p.CostComparison = CompareBilledTokens
				p.Fleet.Accounting.BilledTokens.Authority = AuthorityProviderUsage
			},
			reason: "accounting_billed_tokens_authority_mismatch",
		},
	} {
		t.Run(name, func(t *testing.T) {
			pair := fixture()
			test.mutate(&pair)
			report, err := Evaluate(pair)
			if err != nil {
				t.Fatal(err)
			}
			if report.Verdict != "ABSTAIN" || !slices.Contains(report.Reasons, test.reason) {
				t.Fatalf("report=%+v", report)
			}
		})
	}
}

func TestAccountingOutcomeCountsAreQueryableFromReport(t *testing.T) {
	pair := fixture()
	pair.Single.Accounting.CacheWriteTokens = TokenAccounting{
		Availability: AccountingUnavailable,
		Authority:    AuthorityUnreported,
		Reason:       "provider did not expose cache writes",
	}
	partialValue := int64(50)
	pair.Fleet.Accounting.CacheWriteTokens = TokenAccounting{
		Availability:   AccountingPartial,
		Authority:      AuthorityProviderUsage,
		ArtifactDigest: "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
		Coverage:       0.5,
		Value:          &partialValue,
		Reason:         "provider returned incomplete cache-write usage",
	}

	report, err := Evaluate(pair)
	if err != nil {
		t.Fatal(err)
	}
	want := AccountingOutcomeCounts{Refusal: 1, Error: 1}
	if report.Accounting.Outcomes != want {
		t.Fatalf("accounting outcomes = %+v, want %+v", report.Accounting.Outcomes, want)
	}
	raw, err := json.Marshal(report)
	if err != nil {
		t.Fatal(err)
	}
	field := `"outcomes":{"success":0,"refusal":1,"error":1}`
	if !strings.Contains(string(raw), field) {
		t.Fatalf("report missing queryable outcome counts %s: %s", field, raw)
	}
}
func TestEvaluateRequestsOnlyNamedCostAxis(t *testing.T) {
	pair := fixture()
	pair.CostComparison = CompareBilledTokens
	pair.Single.Accounting.SpendUSD = SpendAccounting{Availability: AccountingUnavailable, Authority: AuthorityProviderUsage, ArtifactDigest: "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", Reason: "spend not emitted"}
	pair.Fleet.Accounting.SpendUSD = SpendAccounting{Availability: AccountingUnavailable, Authority: AuthorityProviderUsage, ArtifactDigest: "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", Reason: "spend not emitted"}
	report, err := Evaluate(pair)
	if err != nil {
		t.Fatal(err)
	}
	if report.Verdict != "GAIN" || report.Accounting.Availability != AccountingAvailable || report.Gains.OutcomePerUSDGain != nil {
		t.Fatalf("report=%+v", report)
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
	return Pair{Schema: Schema, CostComparison: CompareBilledTokensAndSpend,
		Single: Run{Mode: "single", Identity: id, CriticalPathMS: 10000, TotalWorkerMS: 10000, InputTokens: 6000, OutputTokens: 1000, BilledTokens: 7000, SpendUSD: .07, Accounting: knownAccounting(6000, 1000, 0, 0, 7000, .07, "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"), ExpectedEffects: 3, AcceptedEffects: 3, AcceptancePassed: true, WitnessDigest: "sha256:single", Activation: ActivationCohort{Receipts: []ActivationReceipt{control}}},
		Fleet:  Run{Mode: "fleet", Identity: id, CriticalPathMS: 5000, TotalWorkerMS: 13000, InputTokens: 3000, OutputTokens: 900, CacheReadTokens: 5000, CacheWriteTokens: 100, BilledTokens: 4000, SpendUSD: .04, Accounting: knownAccounting(3000, 900, 5000, 100, 4000, .04, "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"), ExpectedEffects: 3, AcceptedEffects: 3, AcceptancePassed: true, WitnessDigest: "sha256:fleet", Activation: ActivationCohort{MinimumActiveRatio: 1, Receipts: []ActivationReceipt{treatment}}},
	}
}

// Package ultracodebench evaluates paired single-agent and fleet coding runs.
package ultracodebench

import (
	"errors"
	"fmt"
	"math"
)

const Schema = "fak-ultracode-paired/1"

type Identity struct {
	Task           string  `json:"task"`
	TaskDigest     string  `json:"task_digest"`
	Model          string  `json:"model"`
	Environment    string  `json:"environment"`
	WallBudgetMS   int64   `json:"wall_budget_ms"`
	TokenBudget    int64   `json:"token_budget"`
	SpendBudgetUSD float64 `json:"spend_budget_usd"`
}

type Run struct {
	Mode             string            `json:"mode"`
	Identity         Identity          `json:"identity"`
	CriticalPathMS   int64             `json:"critical_path_ms"`
	TotalWorkerMS    int64             `json:"total_worker_ms"`
	InputTokens      int64             `json:"input_tokens"`
	OutputTokens     int64             `json:"output_tokens"`
	CacheReadTokens  int64             `json:"cache_read_tokens"`
	CacheWriteTokens int64             `json:"cache_write_tokens"`
	BilledTokens     int64             `json:"billed_tokens"`
	SpendUSD         float64           `json:"spend_usd"`
	Accounting       AccountingReceipt `json:"accounting"`
	ExpectedEffects  int               `json:"expected_effects"`
	AcceptedEffects  int               `json:"accepted_effects"`
	Contradictions   int               `json:"contradictions"`
	AcceptancePassed bool              `json:"acceptance_passed"`
	Retries          int               `json:"retries"`
	WitnessDigest    string            `json:"witness_digest"`
	Activation       ActivationCohort  `json:"activation"`
}

type ActivationCohort struct {
	MinimumActiveRatio float64             `json:"minimum_active_ratio"`
	Receipts           []ActivationReceipt `json:"receipts"`
}

type Pair struct {
	Schema         string         `json:"schema"`
	CostComparison CostComparison `json:"cost_comparison,omitempty"`
	Single         Run            `json:"single"`
	Fleet          Run            `json:"fleet"`
}

type CostComparison string

const (
	CompareBilledTokens         CostComparison = "billed_tokens"
	CompareSpendUSD             CostComparison = "spend_usd"
	CompareBilledTokensAndSpend CostComparison = "billed_tokens_and_spend"
)

type ModeMetrics struct {
	CriticalPathMS          int64             `json:"critical_path_ms"`
	TotalWorkerMS           int64             `json:"total_worker_ms"`
	InputTokens             int64             `json:"input_tokens"`
	OutputTokens            int64             `json:"output_tokens"`
	CacheReadTokens         int64             `json:"cache_read_tokens"`
	CacheWriteTokens        int64             `json:"cache_write_tokens"`
	Accounting              AccountingReceipt `json:"accounting"`
	AcceptedEffects         int               `json:"accepted_effects"`
	ExpectedEffects         int               `json:"expected_effects"`
	PassRate                float64           `json:"pass_rate"`
	ContradictionRate       float64           `json:"contradiction_rate"`
	AcceptedPerWallSecond   float64           `json:"accepted_per_wall_second"`
	AcceptedPerBilledKToken *float64          `json:"accepted_per_billed_ktoken,omitempty"`
	AcceptedPerUSD          *float64          `json:"accepted_per_usd,omitempty"`
}

type Gains struct {
	ConcurrencySpeedup        float64  `json:"concurrency_speedup"`
	BilledTokenReduction      *float64 `json:"billed_token_reduction,omitempty"`
	CachedInputShare          float64  `json:"cached_input_share"`
	AcceptedEffectDelta       int      `json:"accepted_effect_delta"`
	OutcomePerWallGain        float64  `json:"outcome_per_wall_gain"`
	OutcomePerBilledTokenGain *float64 `json:"outcome_per_billed_token_gain,omitempty"`
	OutcomePerUSDGain         *float64 `json:"outcome_per_usd_gain,omitempty"`
}

type AccountingOutcomeCounts struct {
	Success int `json:"success"`
	Refusal int `json:"refusal"`
	Error   int `json:"error"`
}

type AccountingAssessment struct {
	Comparison   CostComparison          `json:"comparison"`
	Availability AccountingAvailability  `json:"availability"`
	Outcomes     AccountingOutcomeCounts `json:"outcomes"`
	Reasons      []string                `json:"reasons,omitempty"`
}

type Report struct {
	Schema      string               `json:"schema"`
	Verdict     string               `json:"verdict"`
	Reasons     []string             `json:"reasons"`
	Attribution string               `json:"attribution"`
	Identity    Identity             `json:"identity"`
	Activation  BenchActivation      `json:"activation"`
	Accounting  AccountingAssessment `json:"accounting"`
	Single      ModeMetrics          `json:"single"`
	Fleet       ModeMetrics          `json:"fleet"`
	Gains       Gains                `json:"gains"`
}

type TreatmentActivation struct {
	Declared           bool    `json:"declared"`
	MinimumActiveRatio float64 `json:"minimum_active_ratio"`
	Active             int     `json:"active"`
	Total              int     `json:"total"`
	Coverage           float64 `json:"coverage"`
	Satisfied          bool    `json:"satisfied"`
}

type BenchActivation struct {
	Control   ActivationSummary   `json:"control"`
	Treatment TreatmentActivation `json:"treatment"`
}

func Evaluate(p Pair) (Report, error) {
	if p.Schema != Schema {
		return Report{}, fmt.Errorf("schema must be %q", Schema)
	}
	if err := validateRun("single", p.Single); err != nil {
		return Report{}, err
	}
	if err := validateRun("fleet", p.Fleet); err != nil {
		return Report{}, err
	}
	if p.Single.Mode != "single" || p.Fleet.Mode != "fleet" {
		return Report{}, errors.New("modes must be single and fleet")
	}
	if p.Single.Identity != p.Fleet.Identity {
		return Report{}, errors.New("single and fleet identity/budgets differ")
	}
	comparison := p.CostComparison
	if comparison == "" {
		comparison = CompareBilledTokensAndSpend
	}
	if comparison != CompareBilledTokens && comparison != CompareSpendUSD && comparison != CompareBilledTokensAndSpend {
		return Report{}, fmt.Errorf("cost_comparison must be %q, %q, or %q", CompareBilledTokens, CompareSpendUSD, CompareBilledTokensAndSpend)
	}
	singleAccounting, err := normalizeAccounting(p.Single.Accounting)
	if err != nil {
		return Report{}, fmt.Errorf("single accounting: %w", err)
	}
	fleetAccounting, err := normalizeAccounting(p.Fleet.Accounting)
	if err != nil {
		return Report{}, fmt.Errorf("fleet accounting: %w", err)
	}
	if err := validateAccountingJoin("single", p.Single, singleAccounting); err != nil {
		return Report{}, err
	}
	if err := validateAccountingJoin("fleet", p.Fleet, fleetAccounting); err != nil {
		return Report{}, err
	}

	s := metrics(p.Single, singleAccounting)
	f := metrics(p.Fleet, fleetAccounting)
	activation, err := assessActivation(p.Single.Activation, p.Fleet.Activation)
	if err != nil {
		return Report{}, err
	}
	accounting := assessAccounting(comparison, singleAccounting, fleetAccounting)
	r := Report{Schema: Schema, Identity: p.Single.Identity, Activation: activation, Attribution: AttributionVerified, Accounting: accounting, Single: s, Fleet: f}
	r.Gains = Gains{
		ConcurrencySpeedup:  ratio(float64(s.CriticalPathMS), float64(f.CriticalPathMS)),
		CachedInputShare:    ratio(float64(f.CacheReadTokens), float64(f.InputTokens+f.CacheReadTokens)),
		AcceptedEffectDelta: f.AcceptedEffects - s.AcceptedEffects,
		OutcomePerWallGain:  gain(s.AcceptedPerWallSecond, f.AcceptedPerWallSecond),
	}
	if s.AcceptedPerBilledKToken != nil && f.AcceptedPerBilledKToken != nil {
		r.Gains.BilledTokenReduction = reductionPtr(float64(*singleAccounting.BilledTokens.Value), float64(*fleetAccounting.BilledTokens.Value))
		r.Gains.OutcomePerBilledTokenGain = gainPtr(*s.AcceptedPerBilledKToken, *f.AcceptedPerBilledKToken)
	}
	if s.AcceptedPerUSD != nil && f.AcceptedPerUSD != nil {
		r.Gains.OutcomePerUSDGain = gainPtr(*s.AcceptedPerUSD, *f.AcceptedPerUSD)
	}

	if !activation.Treatment.Satisfied {
		r.Attribution = AttributionUnverified
		r.Reasons = append(r.Reasons, AttributionUnverified)
	}
	if !p.Single.AcceptancePassed || !p.Fleet.AcceptancePassed {
		r.Reasons = append(r.Reasons, "acceptance outcome is not equal and passing")
	}
	if p.Single.Retries > 0 || p.Fleet.Retries > 0 {
		r.Reasons = append(r.Reasons, "retries make the paired comparison non-equivalent")
	}
	if p.Single.WitnessDigest == "" || p.Fleet.WitnessDigest == "" {
		r.Reasons = append(r.Reasons, "both modes require independent witness digests")
	}
	r.Reasons = append(r.Reasons, accounting.Reasons...)
	if len(r.Reasons) > 0 {
		r.Verdict = "ABSTAIN"
		return r, nil
	}
	if f.AcceptedEffects < s.AcceptedEffects || f.PassRate < s.PassRate || f.ContradictionRate > s.ContradictionRate {
		r.Verdict, r.Reasons = "NO_GAIN", []string{"fleet quality is worse than the single-agent baseline"}
	} else if r.Gains.OutcomePerWallGain > 0 && requestedCostGainPositive(comparison, r.Gains) {
		r.Verdict, r.Reasons = "GAIN", []string{"fleet improves accepted outcome per wall second and the requested authoritative cost axis without lowering quality"}
	} else {
		r.Verdict, r.Reasons = "NO_GAIN", []string{"fleet does not improve both accepted outcome efficiency axes"}
	}
	return r, nil
}

func requestedCostGainPositive(comparison CostComparison, gains Gains) bool {
	billed := gains.OutcomePerBilledTokenGain != nil && *gains.OutcomePerBilledTokenGain > 0
	spend := gains.OutcomePerUSDGain != nil && *gains.OutcomePerUSDGain > 0
	switch comparison {
	case CompareBilledTokens:
		return billed
	case CompareSpendUSD:
		return spend
	default:
		return billed && spend
	}
}

func assessActivation(control, treatment ActivationCohort) (BenchActivation, error) {
	controlSummary, err := SummarizeActivation(control.Receipts)
	if err != nil {
		return BenchActivation{}, fmt.Errorf("control activation: %w", err)
	}
	treatmentSummary, err := SummarizeActivation(treatment.Receipts)
	if err != nil {
		return BenchActivation{}, fmt.Errorf("treatment activation: %w", err)
	}
	result := BenchActivation{Control: controlSummary}
	result.Treatment = TreatmentActivation{
		Declared: treatment.MinimumActiveRatio > 0, MinimumActiveRatio: treatment.MinimumActiveRatio,
		Active: treatmentSummary.Active, Total: treatmentSummary.Total,
	}
	if treatment.MinimumActiveRatio < 0 || treatment.MinimumActiveRatio > 1 {
		return BenchActivation{}, fmt.Errorf("treatment activation minimum_active_ratio must be in (0,1]")
	}
	if result.Treatment.Total > 0 {
		result.Treatment.Coverage = round(float64(result.Treatment.Active) / float64(result.Treatment.Total))
	}
	result.Treatment.Satisfied = result.Treatment.Declared && result.Treatment.Total > 0 && result.Treatment.Coverage >= result.Treatment.MinimumActiveRatio
	return result, nil
}

func validateRun(name string, r Run) error {
	if r.Identity.Task == "" || r.Identity.TaskDigest == "" || r.Identity.Model == "" || r.Identity.Environment == "" {
		return fmt.Errorf("%s identity requires task, task_digest, model, and environment", name)
	}
	if r.CriticalPathMS <= 0 || r.TotalWorkerMS <= 0 || r.InputTokens < 0 || r.OutputTokens < 0 || r.CacheReadTokens < 0 || r.CacheWriteTokens < 0 || r.BilledTokens < 0 || r.SpendUSD < 0 {
		return fmt.Errorf("%s contains invalid timing, token, or spend values", name)
	}
	if r.ExpectedEffects <= 0 || r.AcceptedEffects < 0 || r.AcceptedEffects > r.ExpectedEffects || r.Contradictions < 0 || r.Retries < 0 {
		return fmt.Errorf("%s contains invalid quality counts", name)
	}
	return nil
}

func metrics(r Run, accounting AccountingReceipt) ModeMetrics {
	m := ModeMetrics{
		CriticalPathMS: r.CriticalPathMS, TotalWorkerMS: r.TotalWorkerMS,
		InputTokens: r.InputTokens, OutputTokens: r.OutputTokens,
		CacheReadTokens: r.CacheReadTokens, CacheWriteTokens: r.CacheWriteTokens,
		Accounting:      accounting,
		AcceptedEffects: r.AcceptedEffects, ExpectedEffects: r.ExpectedEffects,
		PassRate:              ratio(float64(r.AcceptedEffects), float64(r.ExpectedEffects)),
		ContradictionRate:     ratio(float64(r.Contradictions), float64(r.ExpectedEffects)),
		AcceptedPerWallSecond: ratio(float64(r.AcceptedEffects)*1000, float64(r.CriticalPathMS)),
	}
	if authoritativeToken(accounting.BilledTokens) {
		value := ratio(float64(r.AcceptedEffects)*1000, float64(*accounting.BilledTokens.Value))
		m.AcceptedPerBilledKToken = &value
	}
	if authoritativeSpend(accounting.SpendUSD) {
		value := ratio(float64(r.AcceptedEffects), *accounting.SpendUSD.ValueUSD)
		m.AcceptedPerUSD = &value
	}
	return m
}

func validateAccountingJoin(name string, run Run, accounting AccountingReceipt) error {
	for axis, values := range map[string]struct {
		receipt TokenAccounting
		run     int64
	}{
		"input_tokens":       {accounting.InputTokens, run.InputTokens},
		"output_tokens":      {accounting.OutputTokens, run.OutputTokens},
		"cache_read_tokens":  {accounting.CacheReadTokens, run.CacheReadTokens},
		"cache_write_tokens": {accounting.CacheWriteTokens, run.CacheWriteTokens},
	} {
		if values.receipt.Availability == AccountingAvailable && *values.receipt.Value != values.run {
			return fmt.Errorf("%s accounting %s value does not match the raw token axis", name, axis)
		}
	}
	if accounting.BilledTokens.Availability == AccountingAvailable && run.BilledTokens != 0 && *accounting.BilledTokens.Value != run.BilledTokens {
		return fmt.Errorf("%s accounting billed_tokens value does not match the compatibility field", name)
	}
	if accounting.SpendUSD.Availability == AccountingAvailable && run.SpendUSD != 0 && *accounting.SpendUSD.ValueUSD != run.SpendUSD {
		return fmt.Errorf("%s accounting spend_usd value does not match the compatibility field", name)
	}
	return nil
}

func accountingOutcomeCounts(receipts ...AccountingReceipt) AccountingOutcomeCounts {
	var counts AccountingOutcomeCounts
	for _, receipt := range receipts {
		hasRefusal, hasError := false, false
		for _, availability := range []AccountingAvailability{
			receipt.InputTokens.Availability,
			receipt.OutputTokens.Availability,
			receipt.CacheReadTokens.Availability,
			receipt.CacheWriteTokens.Availability,
			receipt.BilledTokens.Availability,
			receipt.SpendUSD.Availability,
		} {
			hasRefusal = hasRefusal || availability == AccountingUnavailable
			hasError = hasError || availability == AccountingPartial
		}
		switch {
		case hasError:
			counts.Error++
		case hasRefusal:
			counts.Refusal++
		default:
			counts.Success++
		}
	}
	return counts
}
func assessAccounting(comparison CostComparison, single, fleet AccountingReceipt) AccountingAssessment {
	assessment := AccountingAssessment{Comparison: comparison, Availability: AccountingAvailable, Outcomes: accountingOutcomeCounts(single, fleet)}
	if comparison == CompareBilledTokens || comparison == CompareBilledTokensAndSpend {
		assessment.Reasons = append(assessment.Reasons, tokenAxisReasons("billed_tokens", single.BilledTokens, fleet.BilledTokens)...)
	}
	if comparison == CompareSpendUSD || comparison == CompareBilledTokensAndSpend {
		assessment.Reasons = append(assessment.Reasons, spendAxisReasons(single.SpendUSD, fleet.SpendUSD)...)
	}
	if len(assessment.Reasons) > 0 {
		assessment.Availability = AccountingUnavailable
	}
	return assessment
}

func tokenAxisReasons(name string, single, fleet TokenAccounting) []string {
	prefix := "accounting_" + name
	if single.Availability == AccountingPartial || fleet.Availability == AccountingPartial || single.Coverage > 0 && single.Coverage < 1 || fleet.Coverage > 0 && fleet.Coverage < 1 {
		return []string{prefix + "_partial"}
	}
	if single.Availability != AccountingAvailable || fleet.Availability != AccountingAvailable {
		return []string{prefix + "_unavailable"}
	}
	if single.Authority != fleet.Authority {
		return []string{prefix + "_authority_mismatch"}
	}
	if !authoritativeToken(single) || !authoritativeToken(fleet) {
		return []string{prefix + "_authority_not_provider_billing"}
	}
	return nil
}

func spendAxisReasons(single, fleet SpendAccounting) []string {
	const prefix = "accounting_spend_usd"
	if single.Availability == AccountingPartial || fleet.Availability == AccountingPartial || single.Coverage > 0 && single.Coverage < 1 || fleet.Coverage > 0 && fleet.Coverage < 1 {
		return []string{prefix + "_partial"}
	}
	if single.Availability != AccountingAvailable || fleet.Availability != AccountingAvailable {
		return []string{prefix + "_unavailable"}
	}
	if single.Authority != fleet.Authority {
		return []string{prefix + "_authority_mismatch"}
	}
	if !authoritativeSpend(single) || !authoritativeSpend(fleet) {
		return []string{prefix + "_authority_not_provider_billing"}
	}
	return nil
}

func authoritativeToken(axis TokenAccounting) bool {
	return axis.Availability == AccountingAvailable && axis.Authority == AuthorityProviderBilling && axis.Coverage == 1 && axis.Value != nil && *axis.Value > 0
}

func authoritativeSpend(axis SpendAccounting) bool {
	return axis.Availability == AccountingAvailable && axis.Authority == AuthorityProviderBilling && axis.Coverage == 1 && axis.ValueUSD != nil && *axis.ValueUSD > 0
}

func ratio(a, b float64) float64 {
	if b == 0 {
		return 0
	}
	return round(a / b)
}
func reduction(base, next float64) float64 {
	if base == 0 {
		return 0
	}
	return round((base - next) / base)
}
func reductionPtr(base, next float64) *float64 {
	v := reduction(base, next)
	return &v
}
func gain(base, next float64) float64 {
	if base == 0 {
		return 0
	}
	return round(next/base - 1)
}
func gainPtr(base, next float64) *float64 {
	v := gain(base, next)
	return &v
}
func round(v float64) float64 { return math.Round(v*1e6) / 1e6 }

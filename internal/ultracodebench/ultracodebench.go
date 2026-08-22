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
	Mode             string           `json:"mode"`
	Identity         Identity         `json:"identity"`
	CriticalPathMS   int64            `json:"critical_path_ms"`
	TotalWorkerMS    int64            `json:"total_worker_ms"`
	InputTokens      int64            `json:"input_tokens"`
	OutputTokens     int64            `json:"output_tokens"`
	CacheReadTokens  int64            `json:"cache_read_tokens"`
	CacheWriteTokens int64            `json:"cache_write_tokens"`
	BilledTokens     int64            `json:"billed_tokens"`
	SpendUSD         float64          `json:"spend_usd"`
	ExpectedEffects  int              `json:"expected_effects"`
	AcceptedEffects  int              `json:"accepted_effects"`
	Contradictions   int              `json:"contradictions"`
	AcceptancePassed bool             `json:"acceptance_passed"`
	Retries          int              `json:"retries"`
	WitnessDigest    string           `json:"witness_digest"`
	Activation       ActivationCohort `json:"activation"`
}

type ActivationCohort struct {
	MinimumActiveRatio float64             `json:"minimum_active_ratio"`
	Receipts           []ActivationReceipt `json:"receipts"`
}

type Pair struct {
	Schema string `json:"schema"`
	Single Run    `json:"single"`
	Fleet  Run    `json:"fleet"`
}

type ModeMetrics struct {
	CriticalPathMS          int64   `json:"critical_path_ms"`
	TotalWorkerMS           int64   `json:"total_worker_ms"`
	InputTokens             int64   `json:"input_tokens"`
	OutputTokens            int64   `json:"output_tokens"`
	CacheReadTokens         int64   `json:"cache_read_tokens"`
	CacheWriteTokens        int64   `json:"cache_write_tokens"`
	BilledTokens            int64   `json:"billed_tokens"`
	SpendUSD                float64 `json:"spend_usd"`
	AcceptedEffects         int     `json:"accepted_effects"`
	ExpectedEffects         int     `json:"expected_effects"`
	PassRate                float64 `json:"pass_rate"`
	ContradictionRate       float64 `json:"contradiction_rate"`
	AcceptedPerWallSecond   float64 `json:"accepted_per_wall_second"`
	AcceptedPerBilledKToken float64 `json:"accepted_per_billed_ktoken"`
}

type Gains struct {
	ConcurrencySpeedup        float64 `json:"concurrency_speedup"`
	BilledTokenReduction      float64 `json:"billed_token_reduction"`
	CachedInputShare          float64 `json:"cached_input_share"`
	AcceptedEffectDelta       int     `json:"accepted_effect_delta"`
	OutcomePerWallGain        float64 `json:"outcome_per_wall_gain"`
	OutcomePerBilledTokenGain float64 `json:"outcome_per_billed_token_gain"`
}

type Report struct {
	Schema      string          `json:"schema"`
	Verdict     string          `json:"verdict"`
	Reasons     []string        `json:"reasons"`
	Attribution string          `json:"attribution"`
	Identity    Identity        `json:"identity"`
	Activation  BenchActivation `json:"activation"`
	Single      ModeMetrics     `json:"single"`
	Fleet       ModeMetrics     `json:"fleet"`
	Gains       Gains           `json:"gains"`
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

	s := metrics(p.Single)
	f := metrics(p.Fleet)
	activation, err := assessActivation(p.Single.Activation, p.Fleet.Activation)
	if err != nil {
		return Report{}, err
	}
	r := Report{Schema: Schema, Identity: p.Single.Identity, Activation: activation, Attribution: AttributionVerified, Single: s, Fleet: f}
	r.Gains = Gains{
		ConcurrencySpeedup:        ratio(float64(s.CriticalPathMS), float64(f.CriticalPathMS)),
		BilledTokenReduction:      reduction(float64(s.BilledTokens), float64(f.BilledTokens)),
		CachedInputShare:          ratio(float64(f.CacheReadTokens), float64(f.InputTokens+f.CacheReadTokens)),
		AcceptedEffectDelta:       f.AcceptedEffects - s.AcceptedEffects,
		OutcomePerWallGain:        gain(s.AcceptedPerWallSecond, f.AcceptedPerWallSecond),
		OutcomePerBilledTokenGain: gain(s.AcceptedPerBilledKToken, f.AcceptedPerBilledKToken),
	}

	switch {
	case !activation.Treatment.Satisfied:
		r.Verdict, r.Attribution, r.Reasons = "ABSTAIN", AttributionUnverified, []string{AttributionUnverified}
	case !p.Single.AcceptancePassed || !p.Fleet.AcceptancePassed:
		r.Verdict, r.Reasons = "ABSTAIN", []string{"acceptance outcome is not equal and passing"}
	case p.Single.Retries > 0 || p.Fleet.Retries > 0:
		r.Verdict, r.Reasons = "ABSTAIN", []string{"retries make the paired comparison non-equivalent"}
	case p.Single.WitnessDigest == "" || p.Fleet.WitnessDigest == "":
		r.Verdict, r.Reasons = "ABSTAIN", []string{"both modes require independent witness digests"}
	case f.AcceptedEffects < s.AcceptedEffects || f.PassRate < s.PassRate || f.ContradictionRate > s.ContradictionRate:
		r.Verdict, r.Reasons = "NO_GAIN", []string{"fleet quality is worse than the single-agent baseline"}
	case r.Gains.OutcomePerWallGain > 0 && r.Gains.OutcomePerBilledTokenGain > 0:
		r.Verdict, r.Reasons = "GAIN", []string{"fleet improves accepted outcome per wall second and per billed token without lowering quality"}
	default:
		r.Verdict, r.Reasons = "NO_GAIN", []string{"fleet does not improve both accepted outcome efficiency axes"}
	}
	return r, nil
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
	if r.CriticalPathMS <= 0 || r.TotalWorkerMS <= 0 || r.InputTokens < 0 || r.OutputTokens < 0 || r.CacheReadTokens < 0 || r.CacheWriteTokens < 0 || r.BilledTokens <= 0 || r.SpendUSD < 0 {
		return fmt.Errorf("%s contains invalid timing, token, or spend values", name)
	}
	if r.ExpectedEffects <= 0 || r.AcceptedEffects < 0 || r.AcceptedEffects > r.ExpectedEffects || r.Contradictions < 0 || r.Retries < 0 {
		return fmt.Errorf("%s contains invalid quality counts", name)
	}
	return nil
}

func metrics(r Run) ModeMetrics {
	billed := r.BilledTokens
	m := ModeMetrics{
		CriticalPathMS: r.CriticalPathMS, TotalWorkerMS: r.TotalWorkerMS,
		InputTokens: r.InputTokens, OutputTokens: r.OutputTokens,
		CacheReadTokens: r.CacheReadTokens, CacheWriteTokens: r.CacheWriteTokens,
		BilledTokens: billed, SpendUSD: r.SpendUSD,
		AcceptedEffects: r.AcceptedEffects, ExpectedEffects: r.ExpectedEffects,
		PassRate:                ratio(float64(r.AcceptedEffects), float64(r.ExpectedEffects)),
		ContradictionRate:       ratio(float64(r.Contradictions), float64(r.ExpectedEffects)),
		AcceptedPerWallSecond:   ratio(float64(r.AcceptedEffects)*1000, float64(r.CriticalPathMS)),
		AcceptedPerBilledKToken: ratio(float64(r.AcceptedEffects)*1000, float64(billed)),
	}
	return m
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
func gain(base, next float64) float64 {
	if base == 0 {
		return 0
	}
	return round(next/base - 1)
}
func round(v float64) float64 { return math.Round(v*1e6) / 1e6 }

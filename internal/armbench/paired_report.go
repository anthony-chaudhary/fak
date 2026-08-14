package armbench

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"strings"
)

const PairedReceiptsSchema = "fak.armbench.paired-receipts/1"
const PairedReportSchema = "fak.armbench.paired-report/1"

type PairedReceipts struct {
	Schema            string          `json:"schema"`
	Benchmark         string          `json:"benchmark"`
	TunedBaseline     string          `json:"tuned_baseline"`
	CorrectnessMargin float64         `json:"correctness_noninferiority_margin"`
	SafetyMargin      float64         `json:"safety_noninferiority_margin"`
	BootstrapSamples  int             `json:"bootstrap_samples,omitempty"`
	Provenance        string          `json:"provenance"`
	Witness           string          `json:"witness"`
	Setup             []ArmSetupCost  `json:"setup"`
	Prices            []PriceScenario `json:"price_scenarios"`
	Trials            []PairedTrial   `json:"trials"`
}

type ArmSetupCost struct {
	Arm             string  `json:"arm"`
	WallMS          float64 `json:"wall_ms"`
	LocalComputeUSD float64 `json:"local_compute_usd"`
}
type PriceScenario struct {
	Name                   string  `json:"name"`
	InputUSDPerMillion     float64 `json:"input_usd_per_million"`
	OutputUSDPerMillion    float64 `json:"output_usd_per_million"`
	LocalComputeMultiplier float64 `json:"local_compute_multiplier"`
}
type PairedTrial struct {
	PairID       string  `json:"pair_id"`
	Task         string  `json:"task"`
	Model        string  `json:"model"`
	Temperature  string  `json:"temperature"`
	Arm          string  `json:"arm"`
	Success      bool    `json:"success"`
	Safe         bool    `json:"safe"`
	InputTokens  float64 `json:"input_tokens"`
	OutputTokens float64 `json:"output_tokens"`
	WallMS       float64 `json:"wall_ms"`
	TTFTMS       float64 `json:"ttft_ms"`
	Retries      float64 `json:"retries"`
	Failed       bool    `json:"failed"`
	Cold         bool    `json:"cold"`
}

type Interval struct {
	Estimate   float64 `json:"estimate"`
	Lower      float64 `json:"lower"`
	Upper      float64 `json:"upper"`
	Confidence float64 `json:"confidence"`
}
type MetricComparison struct {
	Metric        string   `json:"metric"`
	Unit          string   `json:"unit"`
	BaselineMean  float64  `json:"baseline_mean"`
	TreatmentMean float64  `json:"treatment_mean"`
	Delta         Interval `json:"paired_delta_treatment_minus_baseline"`
}
type CostSensitivity struct {
	Scenario             string   `json:"scenario"`
	ProviderBaselineUSD  float64  `json:"provider_baseline_usd"`
	ProviderTreatmentUSD float64  `json:"provider_treatment_usd"`
	SteadyDeltaUSD       Interval `json:"steady_state_paired_delta_usd"`
	SetupDeltaUSD        float64  `json:"one_time_setup_delta_usd"`
	SetupDeltaMS         float64  `json:"one_time_setup_delta_ms"`
	Amortized100DeltaUSD float64  `json:"amortized_delta_usd_at_100_trials"`
	BreakEvenTrials      *float64 `json:"break_even_trials,omitempty"`
}
type PairedComparison struct {
	Task                    string             `json:"task"`
	Model                   string             `json:"model"`
	Temperature             string             `json:"temperature"`
	Treatment               string             `json:"treatment"`
	Pairs                   int                `json:"pairs"`
	ColdPairs               int                `json:"cold_pairs"`
	WarmPairs               int                `json:"warm_pairs"`
	BaselineFailures        int                `json:"baseline_failures"`
	TreatmentFailures       int                `json:"treatment_failures"`
	Correctness             Interval           `json:"success_rate_paired_delta"`
	Safety                  Interval           `json:"safety_rate_paired_delta"`
	CorrectnessGate         string             `json:"correctness_gate"`
	SafetyGate              string             `json:"safety_gate"`
	EfficiencyClaimsAllowed bool               `json:"efficiency_claims_allowed"`
	Headline                string             `json:"headline"`
	Metrics                 []MetricComparison `json:"metrics"`
	ColdMetrics             []MetricComparison `json:"cold_metrics"`
	WarmMetrics             []MetricComparison `json:"warm_metrics"`
	Costs                   []CostSensitivity  `json:"cost_sensitivity"`
}
type ClaimCheckInput struct {
	Claim         string `json:"claim"`
	TunedBaseline string `json:"tuned_baseline"`
	Scope         string `json:"scope"`
	Provenance    string `json:"provenance"`
	Witness       string `json:"witness"`
	Verdict       string `json:"verdict"`
}
type PairedReport struct {
	Schema          string             `json:"schema"`
	Benchmark       string             `json:"benchmark"`
	TunedBaseline   string             `json:"tuned_baseline"`
	Confidence      float64            `json:"confidence"`
	FamilywiseAlpha float64            `json:"familywise_alpha"`
	Correction      string             `json:"multiple_comparison_correction"`
	Comparisons     []PairedComparison `json:"comparisons"`
	ClaimCheck      []ClaimCheckInput  `json:"claim_check_input"`
}

type pair struct{ b, t PairedTrial }
type groupKey struct{ task, model, temp, arm string }

func BuildPairedReport(in *PairedReceipts) (*PairedReport, error) {
	if in == nil || in.Schema != PairedReceiptsSchema || in.TunedBaseline == "" || len(in.Trials) == 0 {
		return nil, errors.New("paired receipts require schema, tuned_baseline, and trials")
	}
	if in.BootstrapSamples == 0 {
		in.BootstrapSamples = 10000
	}
	if in.BootstrapSamples < 1000 {
		return nil, errors.New("bootstrap_samples must be at least 1000")
	}
	byPair := map[string]map[string]PairedTrial{}
	arms := map[string]bool{}
	for _, t := range in.Trials {
		if t.PairID == "" || t.Task == "" || t.Model == "" || t.Temperature == "" || t.Arm == "" {
			return nil, errors.New("every trial requires pair_id, task, model, temperature, and arm")
		}
		k := t.Task + "\x00" + t.Model + "\x00" + t.Temperature + "\x00" + t.PairID
		if byPair[k] == nil {
			byPair[k] = map[string]PairedTrial{}
		}
		if _, ok := byPair[k][t.Arm]; ok {
			return nil, fmt.Errorf("duplicate arm %q in pair %q", t.Arm, t.PairID)
		}
		byPair[k][t.Arm] = t
		arms[t.Arm] = true
	}
	if !arms[in.TunedBaseline] {
		return nil, errors.New("tuned baseline has no receipts")
	}
	groups := map[groupKey][]pair{}
	for _, m := range byPair {
		b, ok := m[in.TunedBaseline]
		if !ok {
			continue
		}
		for arm, t := range m {
			if arm == in.TunedBaseline {
				continue
			}
			if b.Cold != t.Cold {
				return nil, fmt.Errorf("pair %q mixes cold and warm state", b.PairID)
			}
			k := groupKey{b.Task, b.Model, b.Temperature, arm}
			groups[k] = append(groups[k], pair{b, t})
		}
	}
	if len(groups) == 0 {
		return nil, errors.New("no complete baseline-treatment pairs")
	}
	hypotheses := len(groups) * 2 // correctness and safety gates form the claim family.
	alpha := 0.05 / float64(hypotheses)
	confidence := 1 - alpha
	keys := make([]groupKey, 0, len(groups))
	for k := range groups {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool { return fmt.Sprint(keys[i]) < fmt.Sprint(keys[j]) })
	rep := &PairedReport{Schema: PairedReportSchema, Benchmark: in.Benchmark, TunedBaseline: in.TunedBaseline, Confidence: confidence, FamilywiseAlpha: 0.05, Correction: "Bonferroni across task/model/arm correctness+safety gates"}
	setup := map[string]ArmSetupCost{}
	for _, s := range in.Setup {
		setup[s.Arm] = s
	}
	for _, k := range keys {
		ps := groups[k]
		sort.Slice(ps, func(i, j int) bool { return ps[i].b.PairID < ps[j].b.PairID })
		c := PairedComparison{Task: k.task, Model: k.model, Temperature: k.temp, Treatment: k.arm, Pairs: len(ps)}
		for _, p := range ps {
			if p.b.Cold {
				c.ColdPairs++
			} else {
				c.WarmPairs++
			}
			if p.b.Failed {
				c.BaselineFailures++
			}
			if p.t.Failed {
				c.TreatmentFailures++
			}
		}
		c.Correctness = boot(ps, in.BootstrapSamples, alpha, k, "success", func(x PairedTrial) float64 { return boolf(x.Success) })
		c.Safety = boot(ps, in.BootstrapSamples, alpha, k, "safe", func(x PairedTrial) float64 { return boolf(x.Safe) })
		c.CorrectnessGate = gate(c.Correctness.Lower, -in.CorrectnessMargin)
		c.SafetyGate = gate(c.Safety.Lower, -in.SafetyMargin)
		c.EfficiencyClaimsAllowed = c.CorrectnessGate == "pass" && c.SafetyGate == "pass"
		c.Metrics = metrics(ps, in.BootstrapSamples, alpha, k, nil)
		c.ColdMetrics = metrics(ps, in.BootstrapSamples, alpha, k, func(p pair) bool { return p.b.Cold })
		c.WarmMetrics = metrics(ps, in.BootstrapSamples, alpha, k, func(p pair) bool { return !p.b.Cold })
		for _, price := range in.Prices {
			fn := func(x PairedTrial) float64 {
				return x.InputTokens*price.InputUSDPerMillion/1e6 + x.OutputTokens*price.OutputUSDPerMillion/1e6
			}
			iv := boot(ps, in.BootstrapSamples, alpha, k, "cost-"+price.Name, fn)
			bm, tm := means(ps, fn)
			sd := (setup[k.arm].LocalComputeUSD - setup[in.TunedBaseline].LocalComputeUSD) * price.LocalComputeMultiplier
			one := sd
			var be *float64
			if iv.Estimate < 0 && one > 0 {
				x := one / -iv.Estimate
				be = &x
			}
			c.Costs = append(c.Costs, CostSensitivity{Scenario: price.Name, ProviderBaselineUSD: bm, ProviderTreatmentUSD: tm, SteadyDeltaUSD: iv, SetupDeltaUSD: one, SetupDeltaMS: setup[k.arm].WallMS - setup[in.TunedBaseline].WallMS, Amortized100DeltaUSD: iv.Estimate + one/100, BreakEvenTrials: be})
		}
		total := c.Metrics[2].Delta
		c.Headline = fmt.Sprintf("%s vs tuned baseline %s: total tokens delta %.3f (%.3f, %.3f), failures %d vs %d", k.arm, in.TunedBaseline, total.Estimate, total.Lower, total.Upper, c.TreatmentFailures, c.BaselineFailures)
		verdict := "not-yet"
		claim := fmt.Sprintf("%s versus tuned baseline %s for %s/%s/%s", k.arm, in.TunedBaseline, k.task, k.model, k.temp)
		efficiencyImproved := total.Upper < 0 || c.Metrics[3].Delta.Upper < 0
		for _, cost := range c.Costs {
			efficiencyImproved = efficiencyImproved || cost.SteadyDeltaUSD.Upper < 0
		}
		if c.EfficiencyClaimsAllowed && efficiencyImproved {
			verdict = "net-true"
		} else {
			claim = "efficiency claim withheld: " + claim
		}
		rep.ClaimCheck = append(rep.ClaimCheck, ClaimCheckInput{claim, in.TunedBaseline, fmt.Sprintf("task=%s model=%s temperature=%s paired_n=%d cold=%d warm=%d", k.task, k.model, k.temp, len(ps), c.ColdPairs, c.WarmPairs), in.Provenance, in.Witness, verdict})
		rep.Comparisons = append(rep.Comparisons, c)
	}
	return rep, nil
}
func boolf(v bool) float64 {
	if v {
		return 1
	}
	return 0
}
func gate(lower, margin float64) string {
	if lower >= margin {
		return "pass"
	}
	return "fail"
}
func means(ps []pair, fn func(PairedTrial) float64) (float64, float64) {
	var b, t float64
	for _, p := range ps {
		b += fn(p.b)
		t += fn(p.t)
	}
	return b / float64(len(ps)), t / float64(len(ps))
}
func boot(ps []pair, n int, alpha float64, k groupKey, label string, fn func(PairedTrial) float64) Interval {
	ds := make([]float64, len(ps))
	var sum float64
	for i, p := range ps {
		ds[i] = fn(p.t) - fn(p.b)
		sum += ds[i]
	}
	h := sha256.Sum256([]byte(fmt.Sprintf("%v/%s", k, label)))
	r := rand.New(rand.NewSource(int64(binary.LittleEndian.Uint64(h[:8]))))
	samples := make([]float64, n)
	for i := range samples {
		var s float64
		for range ds {
			s += ds[r.Intn(len(ds))]
		}
		samples[i] = s / float64(len(ds))
	}
	sort.Float64s(samples)
	lo := int(math.Floor(alpha / 2 * float64(n)))
	hi := int(math.Ceil((1-alpha/2)*float64(n))) - 1
	if lo < 0 {
		lo = 0
	}
	if hi >= n {
		hi = n - 1
	}
	return Interval{sum / float64(len(ds)), samples[lo], samples[hi], 1 - alpha}
}

type metricDef struct {
	name, unit string
	fn         func(PairedTrial) float64
}

var pairedMetrics = []metricDef{{"input_tokens", "tokens", func(x PairedTrial) float64 { return x.InputTokens }}, {"output_tokens", "tokens", func(x PairedTrial) float64 { return x.OutputTokens }}, {"total_tokens", "tokens", func(x PairedTrial) float64 { return x.InputTokens + x.OutputTokens }}, {"wall_latency", "ms", func(x PairedTrial) float64 { return x.WallMS }}, {"ttft", "ms", func(x PairedTrial) float64 { return x.TTFTMS }}, {"retry_rate", "retries/trial", func(x PairedTrial) float64 { return x.Retries }}, {"failure_rate", "failures/trial", func(x PairedTrial) float64 { return boolf(x.Failed) }}}

func metrics(all []pair, n int, alpha float64, k groupKey, keep func(pair) bool) []MetricComparison {
	ps := all
	if keep != nil {
		ps = nil
		for _, p := range all {
			if keep(p) {
				ps = append(ps, p)
			}
		}
	}
	if len(ps) == 0 {
		return []MetricComparison{}
	}
	out := make([]MetricComparison, 0, len(pairedMetrics))
	for _, m := range pairedMetrics {
		b, t := means(ps, m.fn)
		out = append(out, MetricComparison{m.name, m.unit, b, t, boot(ps, n, alpha, k, m.name, m.fn)})
	}
	return out
}
func MarshalPairedReport(r *PairedReport) ([]byte, error) {
	b, e := json.MarshalIndent(r, "", "  ")
	if e == nil {
		b = append(b, '\n')
	}
	return b, e
}
func UnmarshalPairedReceipts(b []byte) (*PairedReceipts, error) {
	var x PairedReceipts
	d := json.NewDecoder(strings.NewReader(string(b)))
	d.DisallowUnknownFields()
	if err := d.Decode(&x); err != nil {
		return nil, err
	}
	return &x, nil
}

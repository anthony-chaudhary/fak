package ultracodebench

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strings"
)

const FastProfileSchema = "fak-ultracode-fast-profile/1"

type FastProfileBundle struct {
	Schema      string           `json:"schema"`
	Scenario    string           `json:"scenario"`
	Task        FastTask         `json:"task"`
	Noise       RepeatRule       `json:"repeat_rule"`
	Comparisons []FastComparison `json:"comparisons"`
	FactorCells []FastFactorCell `json:"factor_cells"`
}
type FastTask struct {
	ID               string   `json:"id"`
	Digest           string   `json:"digest"`
	Environment      string   `json:"environment"`
	ModelEligibility string   `json:"model_eligibility"`
	HardBudgetMS     int64    `json:"hard_budget_ms"`
	AcceptanceTests  []string `json:"acceptance_tests"`
}
type RepeatRule struct {
	MinimumSamples    int     `json:"minimum_samples"`
	RelativeThreshold float64 `json:"relative_threshold"`
	Statistic         string  `json:"statistic"`
	StoppingRule      string  `json:"stopping_rule"`
}
type FastComparison struct {
	Binding  FastBinding `json:"binding"`
	Standard FastArm     `json:"standard"`
	Fast     FastArm     `json:"fast"`
}
type FastBinding struct {
	ID       string `json:"id"`
	Provider string `json:"provider"`
	Harness  string `json:"harness"`
}
type FastArm struct {
	Profile string       `json:"profile"`
	Samples []FastSample `json:"samples"`
}
type FastSample struct {
	RunID                 string  `json:"run_id"`
	RequestedTier         string  `json:"requested_tier"`
	ResolvedTier          string  `json:"resolved_tier"`
	RealizedTier          string  `json:"realized_tier"`
	Model                 string  `json:"model"`
	ReasoningEffort       string  `json:"reasoning_effort"`
	OutputMode            string  `json:"output_mode"`
	TTFTMS                int64   `json:"ttft_ms"`
	TPOTMS                float64 `json:"tpot_ms"`
	OutputTokensPerSecond float64 `json:"output_tokens_per_second"`
	EndToEndMS            int64   `json:"end_to_end_ms"`
	CriticalPathMS        int64   `json:"critical_path_ms"`
	PromptTokens          int64   `json:"prompt_tokens"`
	CacheReadTokens       int64   `json:"cache_read_tokens"`
	CacheWriteTokens      int64   `json:"cache_write_tokens"`
	CachePosture          string  `json:"cache_posture"`
	Retries               int     `json:"retries"`
	Fallbacks             int     `json:"fallbacks"`
	DiscardedCalls        int     `json:"discarded_calls"`
	CostUSD               float64 `json:"cost_usd"`
	CostAuthority         string  `json:"cost_authority"`
	TimingAuthority       string  `json:"timing_authority"`
	WorkerWidth           int     `json:"worker_width"`
	TotalWorkerMS         int64   `json:"total_worker_ms"`
	LeaseWaitMS           int64   `json:"lease_wait_ms"`
	InvalidationMS        int64   `json:"invalidation_ms"`
	ReconcileMS           int64   `json:"reconcile_ms"`
	AcceptancePassed      bool    `json:"acceptance_passed"`
	OutcomeDigest         string  `json:"outcome_digest"`
	WitnessDigest         string  `json:"witness_digest"`
	AcceptanceAuthority   string  `json:"acceptance_authority"`
}
type FastFactorCell struct {
	ID           string   `json:"id"`
	BindingID    string   `json:"binding_id"`
	ProviderTier string   `json:"provider_tier"`
	ModelMode    string   `json:"model_mode"`
	CachePosture string   `json:"cache_posture"`
	WorkerWidth  int      `json:"worker_width"`
	SampleRunIDs []string `json:"sample_run_ids"`
}
type FastProfileReport struct {
	Schema       string                 `json:"schema"`
	Scenario     string                 `json:"scenario"`
	BundleDigest string                 `json:"bundle_digest"`
	Verdict      string                 `json:"verdict"`
	Reasons      []string               `json:"reasons,omitempty"`
	Comparisons  []FastComparisonReport `json:"comparisons"`
	FactorCells  []FastFactorCell       `json:"factor_cells"`
	Noise        RepeatRule             `json:"repeat_rule"`
}
type FastComparisonReport struct {
	Binding          FastBinding      `json:"binding"`
	Verdict          string           `json:"verdict"`
	Reasons          []string         `json:"reasons,omitempty"`
	Standard         FastDistribution `json:"standard"`
	Fast             FastDistribution `json:"fast"`
	LatencyGainRatio float64          `json:"latency_gain_ratio"`
	CostDeltaUSD     float64          `json:"cost_delta_usd"`
}
type FastDistribution struct {
	SampleCount      int     `json:"sample_count"`
	EndToEndMS       []int64 `json:"end_to_end_ms"`
	MedianEndToEndMS int64   `json:"median_end_to_end_ms"`
	TotalCostUSD     float64 `json:"total_cost_usd"`
	TotalWorkerMS    int64   `json:"total_worker_ms"`
}

func EvaluateFastProfile(b FastProfileBundle) FastProfileReport {
	r := FastProfileReport{Schema: FastProfileSchema, Scenario: b.Scenario, Verdict: "GAIN", FactorCells: b.FactorCells, Noise: b.Noise}
	canonical, _ := json.Marshal(b)
	sum := sha256.Sum256(canonical)
	r.BundleDigest = "sha256:" + hex.EncodeToString(sum[:])
	common := validateFastBundle(b)
	for _, p := range b.Comparisons {
		cr := evaluateFastComparison(p, b.Noise, common)
		r.Comparisons = append(r.Comparisons, cr)
		if cr.Verdict == "ABSTAIN" {
			r.Verdict = "ABSTAIN"
		} else if cr.Verdict == "NO_GAIN" && r.Verdict != "ABSTAIN" {
			r.Verdict = "NO_GAIN"
		}
		r.Reasons = append(r.Reasons, cr.Reasons...)
	}
	if len(b.Comparisons) == 0 {
		r.Verdict = "ABSTAIN"
		r.Reasons = append(r.Reasons, "no comparisons")
	}
	r.Reasons = uniqueStrings(r.Reasons)
	return r
}
func evaluateFastComparison(p FastComparison, n RepeatRule, common []string) FastComparisonReport {
	r := FastComparisonReport{Binding: p.Binding, Verdict: "ABSTAIN"}
	r.Standard = fastDistribution(p.Standard.Samples)
	r.Fast = fastDistribution(p.Fast.Samples)
	reasons := append([]string{}, common...)
	reasons = append(reasons, validateFastArm("standard", p.Standard, n)...)
	reasons = append(reasons, validateFastArm("fast", p.Fast, n)...)
	if len(p.Standard.Samples) > 0 && len(p.Fast.Samples) > 0 {
		a, b := p.Standard.Samples[0], p.Fast.Samples[0]
		if a.CachePosture != b.CachePosture {
			reasons = append(reasons, "cache-posture mismatch")
		}
		if a.OutcomeDigest != b.OutcomeDigest {
			reasons = append(reasons, "unequal accepted outcomes")
		}
	}
	if len(reasons) > 0 {
		r.Reasons = uniqueStrings(reasons)
		return r
	}
	r.LatencyGainRatio = 1 - float64(r.Fast.MedianEndToEndMS)/float64(r.Standard.MedianEndToEndMS)
	r.CostDeltaUSD = r.Fast.TotalCostUSD - r.Standard.TotalCostUSD
	if r.LatencyGainRatio > n.RelativeThreshold {
		r.Verdict = "GAIN"
		r.Reasons = []string{"median latency gain exceeds declared noise threshold at equal accepted outcome"}
	} else {
		r.Verdict = "NO_GAIN"
		r.Reasons = []string{"median latency gain does not exceed declared noise threshold"}
	}
	return r
}
func validateFastBundle(b FastProfileBundle) []string {
	var x []string
	if b.Schema != FastProfileSchema || b.Scenario != "fast-profile" {
		x = append(x, "unsupported fast-profile schema or scenario")
	}
	if b.Task.ID == "" || b.Task.Digest == "" || b.Task.Environment == "" || b.Task.ModelEligibility == "" || b.Task.HardBudgetMS <= 0 || len(b.Task.AcceptanceTests) == 0 {
		x = append(x, "incomplete immutable task envelope")
	}
	if b.Noise.MinimumSamples < 2 || b.Noise.RelativeThreshold < 0 || b.Noise.Statistic != "median" || b.Noise.StoppingRule == "" {
		x = append(x, "incomplete repeated-run noise policy")
	}
	factors := map[string]bool{}
	for _, c := range b.FactorCells {
		factors[c.ProviderTier] = true
		factors[c.ModelMode] = true
		factors[c.CachePosture] = true
		if c.WorkerWidth > 0 {
			factors[fmt.Sprintf("width:%d", c.WorkerWidth)] = true
		}
	}
	if len(b.FactorCells) < 4 {
		x = append(x, "bounded factor matrix must contain at least four controlled cells")
	}
	return x
}
func validateFastArm(name string, a FastArm, n RepeatRule) []string {
	var x []string
	if len(a.Samples) < n.MinimumSamples {
		x = append(x, name+" has fewer samples than noise policy")
	}
	for _, s := range a.Samples {
		prefix := name + " run " + s.RunID
		if s.RunID == "" || s.RequestedTier == "" || s.ResolvedTier == "" || s.RealizedTier == "" || s.Model == "" || s.ReasoningEffort == "" || s.OutputMode == "" {
			x = append(x, prefix+" missing requested/resolved/realized mode")
		}
		if s.TTFTMS <= 0 || s.TPOTMS <= 0 || s.OutputTokensPerSecond <= 0 || s.EndToEndMS <= 0 || s.CriticalPathMS <= 0 || s.TimingAuthority == "" {
			x = append(x, prefix+" missing authoritative timing")
		}
		if s.PromptTokens <= 0 || s.CachePosture == "" {
			x = append(x, prefix+" missing token/cache accounting")
		}
		if s.CostAuthority == "" || s.CostUSD < 0 {
			x = append(x, prefix+" missing authoritative cost")
		}
		if s.WorkerWidth <= 0 || s.TotalWorkerMS <= 0 {
			x = append(x, prefix+" missing coordination accounting")
		}
		if !s.AcceptancePassed || s.OutcomeDigest == "" || s.WitnessDigest == "" || s.AcceptanceAuthority == "" {
			x = append(x, prefix+" missing independent accepted-outcome evidence")
		}
	}
	return x
}
func fastDistribution(s []FastSample) FastDistribution {
	d := FastDistribution{SampleCount: len(s)}
	for _, v := range s {
		d.EndToEndMS = append(d.EndToEndMS, v.EndToEndMS)
		d.TotalCostUSD += v.CostUSD
		d.TotalWorkerMS += v.TotalWorkerMS
	}
	sort.Slice(d.EndToEndMS, func(i, j int) bool { return d.EndToEndMS[i] < d.EndToEndMS[j] })
	if len(d.EndToEndMS) > 0 {
		m := len(d.EndToEndMS) / 2
		if len(d.EndToEndMS)%2 == 1 {
			d.MedianEndToEndMS = d.EndToEndMS[m]
		} else {
			d.MedianEndToEndMS = int64(math.Round(float64(d.EndToEndMS[m-1]+d.EndToEndMS[m]) / 2))
		}
	}
	return d
}
func uniqueStrings(v []string) []string {
	seen := map[string]bool{}
	out := []string{}
	for _, s := range v {
		s = strings.TrimSpace(s)
		if s != "" && !seen[s] {
			seen[s] = true
			out = append(out, s)
		}
	}
	sort.Strings(out)
	return out
}

package qwen38quantrun

import (
	"cmp"
	"errors"
	"math"
	"slices"
)

const (
	AMDScoreboardInputSchema  = "fak.qwen38.amd-scoreboard-input.v1"
	AMDScoreboardReportSchema = "fak.qwen38.amd-scoreboard-report.v1"
)

// AMDScoreboardInput compares already-captured model executions. The reference
// remains an explicitly selected comparator; it is never an execution fallback.
type AMDScoreboardInput struct {
	Schema         string        `json:"schema"`
	LogitTolerance float64       `json:"logit_tolerance"`
	Candidate      AMDArmReceipt `json:"candidate"`
	Reference      AMDArmReceipt `json:"reference"`
}

type AMDArmReceipt struct {
	Name               string               `json:"name"`
	Engine             string               `json:"engine"`
	Backend            string               `json:"backend"`
	Runtime            string               `json:"runtime"`
	ComparatorOnly     bool                 `json:"comparator_only"`
	FallbackActive     bool                 `json:"fallback_active"`
	ArtifactSHA256     string               `json:"artifact_sha256"`
	PromptSHA256       string               `json:"prompt_sha256"`
	PromptTokenIDs     []int                `json:"prompt_token_ids"`
	ContextTokens      int                  `json:"context_tokens"`
	ContextBudgetBytes uint64               `json:"context_budget_bytes"`
	KVTypeK            string               `json:"kv_type_k"`
	KVTypeV            string               `json:"kv_type_v"`
	KVOffload          string               `json:"kv_offload"`
	FlashAttention     bool                 `json:"flash_attention"`
	GPUMemoryBudget    uint64               `json:"gpu_memory_budget_bytes"`
	HostSpillPolicy    string               `json:"host_spill_policy"`
	Temperature        float64              `json:"temperature"`
	PrefillTokens      int                  `json:"prefill_tokens"`
	DecodeTokens       int                  `json:"decode_tokens"`
	Hardware           string               `json:"hardware"`
	SoftwareRevision   string               `json:"software_revision"`
	BuildFlags         []string             `json:"build_flags"`
	PeakRSSBytes       uint64               `json:"peak_rss_bytes"`
	PeakVRAMBytes      uint64               `json:"peak_vram_bytes"`
	ResidentModelBytes uint64               `json:"resident_model_bytes"`
	Trials             []AMDScoreboardTrial `json:"trials"`
}

type AMDScoreboardTrial struct {
	Repetition                int       `json:"repetition"`
	ColdSetupSeconds          float64   `json:"cold_setup_seconds"`
	PrefillSeconds            float64   `json:"prefill_seconds"`
	PrefillTokensPerSecond    float64   `json:"prefill_tokens_per_second"`
	WarmDecodeSeconds         float64   `json:"warm_decode_seconds"`
	WarmDecodeTokensPerSecond float64   `json:"warm_decode_tokens_per_second"`
	OutputTokenIDs            []int     `json:"output_token_ids"`
	Logits                    []float64 `json:"logits"`
	H2DBytes                  uint64    `json:"h2d_bytes"`
	D2HBytes                  uint64    `json:"d2h_bytes"`
	D2DBytes                  uint64    `json:"d2d_bytes"`
	QueueSubmissions          uint64    `json:"queue_submissions"`
}

type AMDScoreboardReport struct {
	Schema                 string        `json:"schema"`
	Verdict                string        `json:"verdict"`
	Comparable             bool          `json:"comparable"`
	Reasons                []string      `json:"reasons,omitempty"`
	Candidate              AMDArmSummary `json:"candidate"`
	Reference              AMDArmSummary `json:"reference"`
	ReferenceOverCandidate *AMDRatios    `json:"reference_over_candidate,omitempty"`
}

type AMDArmSummary struct {
	Name                            string  `json:"name"`
	Engine                          string  `json:"engine"`
	Backend                         string  `json:"backend"`
	MedianColdSetupSeconds          float64 `json:"median_cold_setup_seconds"`
	MedianPrefillTokensPerSecond    float64 `json:"median_prefill_tokens_per_second"`
	MedianWarmDecodeTokensPerSecond float64 `json:"median_warm_decode_tokens_per_second"`
	PeakRSSBytes                    uint64  `json:"peak_rss_bytes"`
	PeakVRAMBytes                   uint64  `json:"peak_vram_bytes"`
	ResidentModelBytes              uint64  `json:"resident_model_bytes"`
}

type AMDRatios struct {
	Prefill float64 `json:"prefill"`
	Decode  float64 `json:"decode"`
}

func BuildAMDScoreboard(in AMDScoreboardInput) AMDScoreboardReport {
	reasons := validateAMDScoreboard(in)
	report := AMDScoreboardReport{Schema: AMDScoreboardReportSchema, Verdict: "not-comparable", Reasons: reasons,
		Candidate: summarizeAMDArm(in.Candidate), Reference: summarizeAMDArm(in.Reference)}
	if len(reasons) != 0 {
		return report
	}
	report.Comparable = true
	report.Verdict = "comparable"
	report.Reasons = nil
	report.ReferenceOverCandidate = &AMDRatios{
		Prefill: report.Reference.MedianPrefillTokensPerSecond / report.Candidate.MedianPrefillTokensPerSecond,
		Decode:  report.Reference.MedianWarmDecodeTokensPerSecond / report.Candidate.MedianWarmDecodeTokensPerSecond,
	}
	return report
}

func validateAMDScoreboard(in AMDScoreboardInput) []string {
	var reasons []string
	add := func(reason string) {
		if !slices.Contains(reasons, reason) {
			reasons = append(reasons, reason)
		}
	}
	if in.Schema != AMDScoreboardInputSchema {
		add("schema-mismatch")
	}
	if !finitePositive(in.LogitTolerance) {
		add("invalid-logit-tolerance")
	}
	validateAMDArm(in.Candidate, "candidate", add)
	validateAMDArm(in.Reference, "reference", add)
	if in.Candidate.Engine != "fak-native" || in.Candidate.ComparatorOnly || in.Candidate.FallbackActive {
		add("candidate-not-fak-native-no-fallback")
	}
	if in.Reference.Engine != "llama.cpp" || !in.Reference.ComparatorOnly || in.Reference.FallbackActive {
		add("reference-not-explicit-llamacpp-comparator")
	}
	if in.Candidate.ArtifactSHA256 != in.Reference.ArtifactSHA256 {
		add("artifact-mismatch")
	}
	if in.Candidate.PromptSHA256 != in.Reference.PromptSHA256 || !slices.Equal(in.Candidate.PromptTokenIDs, in.Reference.PromptTokenIDs) {
		add("prompt-or-tokenization-mismatch")
	}
	if in.Candidate.ContextTokens != in.Reference.ContextTokens {
		add("context-mismatch")
	}
	if in.Candidate.ContextBudgetBytes != in.Reference.ContextBudgetBytes {
		add("context-budget-mismatch")
	}
	if in.Candidate.KVTypeK != in.Reference.KVTypeK || in.Candidate.KVTypeV != in.Reference.KVTypeV {
		add("kv-type-mismatch")
	}
	if in.Candidate.KVOffload != in.Reference.KVOffload {
		add("kv-offload-mismatch")
	}
	if in.Candidate.FlashAttention != in.Reference.FlashAttention {
		add("flash-attention-mismatch")
	}
	if in.Candidate.GPUMemoryBudget != in.Reference.GPUMemoryBudget || in.Candidate.HostSpillPolicy != in.Reference.HostSpillPolicy {
		add("memory-placement-envelope-mismatch")
	}
	if in.Candidate.Temperature != in.Reference.Temperature || in.Candidate.PrefillTokens != in.Reference.PrefillTokens || in.Candidate.DecodeTokens != in.Reference.DecodeTokens {
		add("generation-envelope-mismatch")
	}
	if in.Candidate.Hardware != in.Reference.Hardware {
		add("hardware-mismatch")
	}
	if len(in.Candidate.Trials) == len(in.Reference.Trials) {
		for i := range in.Candidate.Trials {
			c, r := in.Candidate.Trials[i], in.Reference.Trials[i]
			if c.Repetition != r.Repetition || !slices.Equal(c.OutputTokenIDs, r.OutputTokenIDs) {
				add("output-token-mismatch")
				continue
			}
			if len(c.Logits) != len(r.Logits) {
				add("logit-shape-mismatch")
				continue
			}
			for j := range c.Logits {
				if math.Abs(c.Logits[j]-r.Logits[j]) > in.LogitTolerance {
					add("logit-tolerance-exceeded")
					break
				}
			}
		}
	} else {
		add("trial-count-mismatch")
	}
	slices.Sort(reasons)
	return reasons
}

func validateAMDArm(arm AMDArmReceipt, role string, add func(string)) {
	prefix := role + "-"
	if arm.Name == "" || arm.Engine == "" || arm.Backend == "" || arm.Runtime == "" || arm.Hardware == "" || arm.SoftwareRevision == "" || len(arm.BuildFlags) == 0 {
		add(prefix + "identity-incomplete")
	}
	if !validOracleSHA256(arm.ArtifactSHA256) || !validOracleSHA256(arm.PromptSHA256) || len(arm.PromptTokenIDs) == 0 {
		add(prefix + "artifact-or-prompt-incomplete")
	}
	if arm.ContextTokens <= 0 || arm.ContextBudgetBytes == 0 || arm.KVTypeK == "" || arm.KVTypeV == "" || arm.KVOffload == "" || arm.GPUMemoryBudget == 0 || arm.HostSpillPolicy == "" || arm.Temperature != 0 || arm.PrefillTokens <= 0 || arm.DecodeTokens <= 0 {
		add(prefix + "envelope-incomplete")
	}
	if arm.PeakRSSBytes == 0 || arm.PeakVRAMBytes == 0 || arm.ResidentModelBytes == 0 {
		add(prefix + "memory-evidence-missing")
	}
	if len(arm.Trials) < 3 {
		add(prefix + "three-trials-required")
	}
	seen := map[int]bool{}
	for _, t := range arm.Trials {
		if t.Repetition <= 0 || seen[t.Repetition] {
			add(prefix + "trial-index-invalid")
		}
		seen[t.Repetition] = true
		if !finitePositive(t.ColdSetupSeconds) || !finitePositive(t.PrefillSeconds) || !finitePositive(t.PrefillTokensPerSecond) || !finitePositive(t.WarmDecodeSeconds) || !finitePositive(t.WarmDecodeTokensPerSecond) || len(t.OutputTokenIDs) != arm.DecodeTokens || len(t.Logits) == 0 {
			add(prefix + "trial-evidence-incomplete")
		}
		if t.H2DBytes == 0 || t.D2HBytes == 0 || t.QueueSubmissions == 0 {
			add(prefix + "transfer-or-submission-accounting-missing")
		}
		for _, v := range t.Logits {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				add(prefix + "non-finite-logit")
			}
		}
	}
}

func summarizeAMDArm(arm AMDArmReceipt) AMDArmSummary {
	cold, prefill, decode := make([]float64, 0, len(arm.Trials)), make([]float64, 0, len(arm.Trials)), make([]float64, 0, len(arm.Trials))
	for _, t := range arm.Trials {
		cold = append(cold, t.ColdSetupSeconds)
		prefill = append(prefill, t.PrefillTokensPerSecond)
		decode = append(decode, t.WarmDecodeTokensPerSecond)
	}
	return AMDArmSummary{Name: arm.Name, Engine: arm.Engine, Backend: arm.Backend, MedianColdSetupSeconds: medianAMD(cold), MedianPrefillTokensPerSecond: medianAMD(prefill), MedianWarmDecodeTokensPerSecond: medianAMD(decode), PeakRSSBytes: arm.PeakRSSBytes, PeakVRAMBytes: arm.PeakVRAMBytes, ResidentModelBytes: arm.ResidentModelBytes}
}

func medianAMD(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	values = slices.Clone(values)
	slices.SortFunc(values, func(a, b float64) int { return cmp.Compare(a, b) })
	m := len(values) / 2
	if len(values)%2 == 1 {
		return values[m]
	}
	return (values[m-1] + values[m]) / 2
}

func ValidateAMDScoreboardReport(report AMDScoreboardReport) error {
	if report.Schema != AMDScoreboardReportSchema {
		return errors.New("scoreboard report schema mismatch")
	}
	if report.Comparable != (report.Verdict == "comparable") {
		return errors.New("scoreboard verdict and comparable flag disagree")
	}
	if report.Comparable && (len(report.Reasons) != 0 || report.ReferenceOverCandidate == nil || !finitePositive(report.ReferenceOverCandidate.Prefill) || !finitePositive(report.ReferenceOverCandidate.Decode)) {
		return errors.New("comparable scoreboard lacks valid ratios")
	}
	if !report.Comparable && (report.Verdict != "not-comparable" || len(report.Reasons) == 0 || report.ReferenceOverCandidate != nil) {
		return errors.New("not-comparable scoreboard must carry reasons and suppress ratios")
	}
	return nil
}

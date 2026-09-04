package qwen38quantrun

import (
	"cmp"
	"errors"
	"fmt"
	"math"
	"slices"

	"github.com/anthony-chaudhary/fak/internal/model"
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
	Name                string               `json:"name"`
	Engine              string               `json:"engine"`
	Backend             string               `json:"backend"`
	Runtime             string               `json:"runtime"`
	ComparatorOnly      bool                 `json:"comparator_only"`
	FallbackActive      bool                 `json:"fallback_active"`
	ArtifactSHA256      string               `json:"artifact_sha256"`
	PromptSHA256        string               `json:"prompt_sha256"`
	PromptTokenIDs      []int                `json:"prompt_token_ids"`
	ContextTokens       int                  `json:"context_tokens"`
	ContextBudgetBytes  uint64               `json:"context_budget_bytes"`
	KVTypeK             string               `json:"kv_type_k"`
	KVTypeV             string               `json:"kv_type_v"`
	KVOffload           string               `json:"kv_offload"`
	FlashAttention      bool                 `json:"flash_attention"`
	GPUMemoryBudget     uint64               `json:"gpu_memory_budget_bytes"`
	HostSpillPolicy     string               `json:"host_spill_policy"`
	Temperature         float64              `json:"temperature"`
	PrefillTokens       int                  `json:"prefill_tokens"`
	DecodeTokens        int                  `json:"decode_tokens"`
	Hardware            string               `json:"hardware"`
	SoftwareRevision    string               `json:"software_revision"`
	BuildFlags          []string             `json:"build_flags"`
	PeakRSSBytes        uint64               `json:"peak_rss_bytes"`
	PeakVRAMBytes       uint64               `json:"peak_vram_bytes"`
	ResidentModelBytes  uint64               `json:"resident_model_bytes"`
	DeterministicTokens bool                 `json:"deterministic_tokens,omitempty"`
	SelectedTokenLogits bool                 `json:"selected_token_logits,omitempty"`
	TokenizerDigest     string               `json:"tokenizer_digest,omitempty"`
	PromptPacketDigest  string               `json:"prompt_packet_digest,omitempty"`
	StopTokens          []string             `json:"stop_tokens,omitempty"`
	StopTokenIDs        []int                `json:"stop_token_ids,omitempty"`
	TopP                float64              `json:"top_p,omitempty"`
	PromptPacket        *PromptTokenPacket   `json:"prompt_packet,omitempty"`
	Trials              []AMDScoreboardTrial `json:"trials"`
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
	SelectedTokenIDs          []int     `json:"selected_token_ids,omitempty"`
	SelectedTokenLogits       []float64 `json:"selected_token_logits,omitempty"`
	H2DBytes                  uint64    `json:"h2d_bytes"`
	D2HBytes                  uint64    `json:"d2h_bytes"`
	D2DBytes                  uint64    `json:"d2d_bytes"`
	QueueSubmissions          uint64    `json:"queue_submissions"`
}

func (t AMDScoreboardTrial) EffectiveTokenIDs() []int {
	if len(t.OutputTokenIDs) > 0 {
		return t.OutputTokenIDs
	}
	return t.SelectedTokenIDs
}

func (t AMDScoreboardTrial) EffectiveLogits() []float64 {
	if len(t.Logits) > 0 {
		return t.Logits
	}
	return t.SelectedTokenLogits
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

func isDivergedOnly(reasons []string) bool {
	if len(reasons) == 0 {
		return false
	}
	for _, r := range reasons {
		if r != "output-token-mismatch" && r != "logit-shape-mismatch" && r != "logit-tolerance-exceeded" {
			return false
		}
	}
	return true
}

func BuildAMDScoreboard(in AMDScoreboardInput) AMDScoreboardReport {
	reasons := validateAMDScoreboard(in)
	verdict := "not-comparable"
	if isDivergedOnly(reasons) {
		verdict = "diverged"
	}
	report := AMDScoreboardReport{Schema: AMDScoreboardReportSchema, Verdict: verdict, Reasons: reasons,
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
	if in.Candidate.TokenizerDigest != in.Reference.TokenizerDigest {
		add("tokenizer-digest-mismatch")
	}
	if in.Candidate.PromptPacketDigest != in.Reference.PromptPacketDigest {
		add("prompt-packet-mismatch")
	}
	if in.Candidate.PromptPacket != nil && in.Reference.PromptPacket != nil {
		if err := ValidatePromptPacketAttestation(*in.Candidate.PromptPacket, *in.Reference.PromptPacket); err != nil {
			add("prompt-packet-mismatch")
		}
	} else if (in.Candidate.PromptPacket != nil) != (in.Reference.PromptPacket != nil) {
		add("prompt-packet-mismatch")
	}
	if in.Candidate.PromptSHA256 != in.Reference.PromptSHA256 || !slices.Equal(in.Candidate.PromptTokenIDs, in.Reference.PromptTokenIDs) {
		add("prompt-or-tokenization-mismatch")
	}
	if in.Candidate.TokenizerDigest != in.Reference.TokenizerDigest {
		add("tokenizer-digest-mismatch")
	}
	if in.Candidate.PromptPacketDigest != in.Reference.PromptPacketDigest {
		add("prompt-packet-digest-mismatch")
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
	if in.Candidate.Temperature != in.Reference.Temperature || in.Candidate.PrefillTokens != in.Reference.PrefillTokens || in.Candidate.DecodeTokens != in.Reference.DecodeTokens || in.Candidate.TopP != in.Reference.TopP {
		add("generation-envelope-mismatch")
	}
	if !slices.Equal(in.Candidate.StopTokens, in.Reference.StopTokens) || !slices.Equal(in.Candidate.StopTokenIDs, in.Reference.StopTokenIDs) {
		add("stop-tokens-mismatch")
	}
	if in.Candidate.PromptPacket != nil || in.Reference.PromptPacket != nil {
		if in.Candidate.PromptPacket == nil || in.Reference.PromptPacket == nil {
			add("prompt-packet-mismatch")
		} else if err := ValidatePromptPacketAttestation(*in.Candidate.PromptPacket, *in.Reference.PromptPacket); err != nil {
			add("prompt-packet-mismatch")
		}
	}
	if in.Candidate.Hardware != in.Reference.Hardware {
		add("hardware-mismatch")
	}
	if len(in.Candidate.Trials) == len(in.Reference.Trials) {
		for i := range in.Candidate.Trials {
			c, r := in.Candidate.Trials[i], in.Reference.Trials[i]
			cTokens, rTokens := c.EffectiveTokenIDs(), r.EffectiveTokenIDs()
			if c.Repetition != r.Repetition || !slices.Equal(cTokens, rTokens) {
				add("output-token-mismatch")
				continue
			}
			cLogits, rLogits := c.EffectiveLogits(), r.EffectiveLogits()
			if len(cLogits) != len(rLogits) {
				add("logit-shape-mismatch")
				continue
			}
			for j := range cLogits {
				if math.Abs(cLogits[j]-rLogits[j]) > in.LogitTolerance {
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
	if arm.PromptPacket != nil {
		if err := VerifyPromptPacket(*arm.PromptPacket); err != nil {
			add(prefix + "prompt-packet-invalid")
		}
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
		tokens := t.EffectiveTokenIDs()
		logits := t.EffectiveLogits()
		if !finitePositive(t.ColdSetupSeconds) || !finitePositive(t.PrefillSeconds) || !finitePositive(t.PrefillTokensPerSecond) || !finitePositive(t.WarmDecodeSeconds) || !finitePositive(t.WarmDecodeTokensPerSecond) || len(tokens) != arm.DecodeTokens || len(logits) == 0 {
			add(prefix + "trial-evidence-incomplete")
		}
		if t.H2DBytes == 0 || t.D2HBytes == 0 || t.QueueSubmissions == 0 {
			add(prefix + "transfer-or-submission-accounting-missing")
		}
		for _, v := range logits {
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
	if !report.Comparable && ((report.Verdict != "not-comparable" && report.Verdict != "diverged") || len(report.Reasons) == 0 || report.ReferenceOverCandidate != nil) {
		return errors.New("not-comparable or diverged scoreboard must carry reasons and suppress ratios")
	}
	return nil
}

// ValidateTokenEquivalence asserts that candidate and reference emitted identical ordered output token IDs.
func ValidateTokenEquivalence(candidateTokens, referenceTokens []int) error {
	if len(candidateTokens) != len(referenceTokens) {
		return fmt.Errorf("token count mismatch: candidate=%d reference=%d", len(candidateTokens), len(referenceTokens))
	}
	for i := range candidateTokens {
		if candidateTokens[i] != referenceTokens[i] {
			return fmt.Errorf("token ID mismatch at index %d: candidate=%d reference=%d", i, candidateTokens[i], referenceTokens[i])
		}
	}
	return nil
}

// ValidateLogitsTolerance asserts that candidate and reference logits match within the specified tolerance.
func ValidateLogitsTolerance(candidateLogits, referenceLogits []float64, tolerance float64) error {
	if !finitePositive(tolerance) {
		return errors.New("invalid logit tolerance: must be finite positive")
	}
	if len(candidateLogits) != len(referenceLogits) {
		return fmt.Errorf("logit shape mismatch: candidate=%d reference=%d", len(candidateLogits), len(referenceLogits))
	}
	for i := range candidateLogits {
		c, r := candidateLogits[i], referenceLogits[i]
		if math.IsNaN(c) || math.IsInf(c, 0) || math.IsNaN(r) || math.IsInf(r, 0) {
			return fmt.Errorf("non-finite logit at index %d: candidate=%v reference=%v", i, c, r)
		}
		diff := math.Abs(c - r)
		if diff > tolerance {
			return fmt.Errorf("logit tolerance exceeded at index %d: |%f - %f| = %f > %f", i, c, r, diff, tolerance)
		}
	}
	return nil
}

// ValidateTrialTokensAndLogits validates that trial tokens match exactly and logits are within tolerance.
func ValidateTrialTokensAndLogits(candidate, reference AMDScoreboardTrial, tolerance float64) error {
	if candidate.Repetition != reference.Repetition {
		return fmt.Errorf("trial repetition mismatch: candidate=%d reference=%d", candidate.Repetition, reference.Repetition)
	}
	if err := ValidateTokenEquivalence(candidate.EffectiveTokenIDs(), reference.EffectiveTokenIDs()); err != nil {
		return err
	}
	return ValidateLogitsTolerance(candidate.EffectiveLogits(), reference.EffectiveLogits(), tolerance)
}

// ValidateAMDScoreboardDivergence verifies that a scoreboard report represents a diverged or not-comparable outcome with token/logit divergence reasons.
func ValidateAMDScoreboardDivergence(report AMDScoreboardReport) error {
	if report.Comparable {
		return errors.New("expected diverged or not-comparable scoreboard report, got comparable")
	}
	if report.Verdict != "diverged" && report.Verdict != "not-comparable" {
		return fmt.Errorf("unexpected verdict %q: expected 'diverged' or 'not-comparable'", report.Verdict)
	}
	hasDivergenceReason := false
	for _, r := range report.Reasons {
		if r == "output-token-mismatch" || r == "logit-shape-mismatch" || r == "logit-tolerance-exceeded" {
			hasDivergenceReason = true
			break
		}
	}
	if !hasDivergenceReason {
		return fmt.Errorf("report reasons %v do not carry token or logit divergence", report.Reasons)
	}
	return nil
}

// CaptureAMDScoreboardTrial converts a native inference receipt and timing data into an AMDScoreboardTrial.
func CaptureAMDScoreboardTrial(repetition int, receipt *model.NativeInferenceReceipt, coldSetupS, prefillTokensPerS, warmDecodeTokensPerS float64, h2d, d2h, d2d, queueSubmissions uint64) (AMDScoreboardTrial, error) {
	if repetition <= 0 {
		return AMDScoreboardTrial{}, errors.New("repetition must be positive")
	}
	if receipt == nil {
		return AMDScoreboardTrial{}, errors.New("native inference receipt is required")
	}
	tokenIDs := append([]int(nil), receipt.TokenIDs...)
	logits := append([]float64(nil), receipt.TokenLogprobs...)
	return AMDScoreboardTrial{
		Repetition:                repetition,
		ColdSetupSeconds:          coldSetupS,
		PrefillSeconds:            receipt.PrefillSeconds,
		PrefillTokensPerSecond:    prefillTokensPerS,
		WarmDecodeSeconds:         receipt.DecodeSeconds,
		WarmDecodeTokensPerSecond: warmDecodeTokensPerS,
		OutputTokenIDs:            tokenIDs,
		Logits:                    logits,
		SelectedTokenIDs:          tokenIDs,
		SelectedTokenLogits:       logits,
		H2DBytes:                  h2d,
		D2HBytes:                  d2h,
		D2DBytes:                  d2d,
		QueueSubmissions:          queueSubmissions,
	}, nil
}

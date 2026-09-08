package qwen38quantrun

import (
	"cmp"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"reflect"
	"slices"

	"github.com/anthony-chaudhary/fak/internal/model"
)

const (
	AMDScoreboardInputSchema  = "fak.qwen38.amd-scoreboard-input.v2"
	AMDScoreboardReportSchema = "fak.qwen38.amd-scoreboard-report.v2"
	AMDStatisticalAssumptions = "Conditional on independent, identically distributed approximately normal paired trial observations; CV alone does not establish these assumptions. Arithmetic mean of per-trial accepted-output TPS; non-overlapping prefill and decode phases."
)

// AMDScoreboardInput compares already-captured model executions. The reference
// remains an explicitly selected comparator; it is never an execution fallback.
type AMDScoreboardInput struct {
	Schema         string        `json:"schema"`
	LogitTolerance float64       `json:"logit_tolerance"`
	Concurrency    int           `json:"concurrency"`
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
	TemplateDigest      string               `json:"template_digest,omitempty"`
	PromptPacketDigest  string               `json:"prompt_packet_digest,omitempty"`
	StopTokens          []string             `json:"stop_tokens,omitempty"`
	StopTokenIDs        []int                `json:"stop_token_ids,omitempty"`
	TopP                float64              `json:"top_p,omitempty"`
	TopK                int                  `json:"top_k,omitempty"`
	PromptPacket        *PromptTokenPacket   `json:"prompt_packet,omitempty"`
	Trials              []AMDScoreboardTrial `json:"trials"`
}

type AMDScoreboardTrial struct {
	// Empty evidence_kind retains the legacy raw-logit contract. Native receipts
	// contain log_softmax values, never raw logits; both arms must name that basis.
	// "selected-token-logprobs" means one normalized log probability per output
	// token, in token order. The legacy logits/selected_token_logits JSON fields
	// remain unchanged; evidence_kind declares the numeric basis of their values.
	EvidenceKind           string                        `json:"evidence_kind,omitempty"`
	NativeInferenceReceipt *model.NativeInferenceReceipt `json:"native_inference_receipt,omitempty"`

	Repetition                int       `json:"repetition"`
	Sequence                  int       `json:"sequence"`
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
	RawInput                AMDScoreboardInput `json:"raw_input"`
	PairedComparable        bool               `json:"paired_comparable"`
	StrixAbsoluteEligible   bool               `json:"strix_absolute_eligible"`
	StrixEligibilityReasons []string           `json:"strix_eligibility_reasons"`
	OverallWin              bool               `json:"overall_win"`
	Statistics              *AMDStatistics     `json:"statistics,omitempty"`
	Schema                  string             `json:"schema"`
	Verdict                 string             `json:"verdict"`
	Comparable              bool               `json:"comparable"`
	Reasons                 []string           `json:"reasons,omitempty"`
	Candidate               AMDArmSummary      `json:"candidate"`
	Reference               AMDArmSummary      `json:"reference"`
	ReferenceOverCandidate  *AMDRatios         `json:"reference_over_candidate,omitempty"`
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
	GPUMemoryBudget                 uint64  `json:"gpu_memory_budget_bytes"`
}

type AMDStatistics struct {
	Assumptions      string  `json:"assumptions"`
	Trials           int     `json:"trials"`
	CandidateMeanTPS float64 `json:"candidate_mean_tps"`
	ReferenceMeanTPS float64 `json:"reference_mean_tps"`
	CandidateCV      float64 `json:"candidate_cv"`
	ReferenceCV      float64 `json:"reference_cv"`
	CandidateLCB95   float64 `json:"candidate_lcb95"`
	PairedRatioLCB95 float64 `json:"paired_ratio_lcb95"`
	AbsoluteBar      float64 `json:"absolute_bar"`
	AbsolutePass     bool    `json:"absolute_pass"`
	PairedPass       bool    `json:"paired_pass"`
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
	// Snapshot JSON-representable evidence so later caller mutations cannot
	// rewrite the raw trials retained by this report. Non-finite inputs cannot
	// be serialized and are rejected by the arithmetic gates below.
	if encoded, err := json.Marshal(in); err == nil {
		var snapshot AMDScoreboardInput
		if json.Unmarshal(encoded, &snapshot) == nil {
			in = snapshot
		}
	}
	reasons := validateAMDScoreboard(in)
	verdict := "not-comparable"
	if isDivergedOnly(reasons) {
		verdict = "diverged"
	}
	report := AMDScoreboardReport{Schema: AMDScoreboardReportSchema, Verdict: verdict, Reasons: reasons,
		RawInput: in, StrixEligibilityReasons: []string{"trusted-sealed-strix-cell-unavailable"},
		Candidate: summarizeAMDArm(in.Candidate), Reference: summarizeAMDArm(in.Reference)}
	if len(reasons) != 0 {
		return report
	}
	stats, ok := statisticsAMD(in)
	if !ok {
		report.Reasons = []string{"statistical-arithmetic-invalid"}
		return report
	}
	if !cvAdmittedAMD(stats.CandidateCV) || !cvAdmittedAMD(stats.ReferenceCV) {
		report.Reasons = []string{"coefficient-of-variation-exceeded"}
		return report
	}
	report.Statistics = &stats
	report.Comparable = true
	report.PairedComparable = true
	report.Verdict = "comparable"
	report.Reasons = nil
	report.ReferenceOverCandidate = &AMDRatios{
		Prefill: report.Reference.MedianPrefillTokensPerSecond / report.Candidate.MedianPrefillTokensPerSecond,
		Decode:  report.Reference.MedianWarmDecodeTokensPerSecond / report.Candidate.MedianWarmDecodeTokensPerSecond,
	}
	if !finitePositive(report.ReferenceOverCandidate.Prefill) || !finitePositive(report.ReferenceOverCandidate.Decode) {
		report.Comparable, report.PairedComparable = false, false
		report.Verdict = "not-comparable"
		report.Reasons = []string{"ratio-arithmetic-invalid"}
		report.ReferenceOverCandidate, report.Statistics = nil, nil
	}
	return report
}

func closeRateAMD(got, want float64) bool {
	return finitePositive(got) && finitePositive(want) && math.Abs(got/want-1) <= 1e-9
}

// Admit the exactly 5% sample CV despite at most one floating-point rounding
// step. This tolerance is never applied to either strict confidence bound.
func cvAdmittedAMD(cv float64) bool {
	return cv >= 0 && cv <= math.Nextafter(0.05, math.Inf(1))
}

// oneSided95LCBAMDChecked uses sample SD and a fixed t(0.95,4). For n >= 5
// this is conservative relative to t(0.95,n-1), conditional on the sampling
// assumptions stated in AMDStatisticalAssumptions; it is not distribution free.
func oneSided95LCBAMDChecked(values []float64) (mean, cv, lcb float64, ok bool) {
	if len(values) < 5 {
		return
	}
	m2 := 0.0
	for i, x := range values {
		if !finitePositive(x) {
			return 0, 0, 0, false
		}
		delta := x - mean
		mean += delta / float64(i+1)
		term := delta * (x - mean)
		if term == 0 && delta != 0 && x != mean {
			return 0, 0, 0, false
		}
		m2 += term
		if !finitePositive(mean) || math.IsNaN(m2) || math.IsInf(m2, 0) || m2 < 0 {
			return 0, 0, 0, false
		}
	}
	sd := math.Sqrt(m2 / float64(len(values)-1))
	if sd == 0 && m2 != 0 {
		return 0, 0, 0, false
	}
	cv = sd / mean
	lcb = mean - 2.13184678632665*(sd/math.Sqrt(float64(len(values))))
	ok = !math.IsNaN(cv) && !math.IsInf(cv, 0) && !math.IsNaN(lcb) && !math.IsInf(lcb, 0)
	return
}

func frozenStrixBarAMD(concurrency int) float64 {
	switch concurrency {
	case 1:
		return 16.65
	case 4:
		return 24.13
	case 8:
		return 29.42
	}
	return 0
}

func statisticsAMD(in AMDScoreboardInput) (AMDStatistics, bool) {
	n := len(in.Candidate.Trials)
	if n != len(in.Reference.Trials) {
		return AMDStatistics{}, false
	}
	c, r, ratios := make([]float64, n), make([]float64, n), make([]float64, n)
	for i := range c {
		ct, rt := in.Candidate.Trials[i], in.Reference.Trials[i]
		c[i] = float64(len(ct.EffectiveTokenIDs())) / (ct.PrefillSeconds + ct.WarmDecodeSeconds)
		r[i] = float64(len(rt.EffectiveTokenIDs())) / (rt.PrefillSeconds + rt.WarmDecodeSeconds)
		ratios[i] = c[i] / r[i]
	}
	cm, cc, cl, cok := oneSided95LCBAMDChecked(c)
	rm, rc, _, rok := oneSided95LCBAMDChecked(r)
	_, _, pl, pok := oneSided95LCBAMDChecked(ratios)
	return AMDStatistics{Assumptions: AMDStatisticalAssumptions, Trials: n, CandidateMeanTPS: cm, ReferenceMeanTPS: rm, CandidateCV: cc, ReferenceCV: rc, CandidateLCB95: cl, PairedRatioLCB95: pl, PairedPass: pl > 1}, cok && rok && pok
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
	// Multi-request credit needs request-level admission/completion observations;
	// a concurrency label cannot multiply a single-request receipt into a batch.
	if in.Concurrency != 1 {
		add("concurrent-request-observation-required")
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
	if err := ValidateArmPromptPacketAttestation(in.Candidate, in.Reference); err != nil {
		add("prompt-packet-mismatch")
	}
	if in.Candidate.PromptSHA256 != in.Reference.PromptSHA256 || !slices.Equal(in.Candidate.PromptTokenIDs, in.Reference.PromptTokenIDs) {
		add("prompt-or-tokenization-mismatch")
	}
	if in.Candidate.TokenizerDigest != in.Reference.TokenizerDigest {
		add("tokenizer-digest-mismatch")
	}
	if in.Candidate.TemplateDigest != in.Reference.TemplateDigest {
		add("template-digest-mismatch")
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
	if in.Candidate.Temperature != in.Reference.Temperature || in.Candidate.PrefillTokens != in.Reference.PrefillTokens || in.Candidate.DecodeTokens != in.Reference.DecodeTokens || in.Candidate.TopP != in.Reference.TopP || in.Candidate.TopK != in.Reference.TopK {
		add("generation-envelope-mismatch")
	}
	if !slices.Equal(in.Candidate.StopTokens, in.Reference.StopTokens) || !slices.Equal(in.Candidate.StopTokenIDs, in.Reference.StopTokenIDs) {
		add("stop-tokens-mismatch")
	}
	if in.Candidate.Hardware != in.Reference.Hardware {
		add("hardware-mismatch")
	}
	if len(in.Candidate.Trials) == len(in.Reference.Trials) {
		for i := range in.Candidate.Trials {
			c, r := in.Candidate.Trials[i], in.Reference.Trials[i]
			cs, rs := 2*i+1, 2*i+2
			if i%2 != 0 {
				cs, rs = rs, cs
			}
			if c.Repetition != i+1 || r.Repetition != i+1 || c.Sequence != cs || r.Sequence != rs {
				add("alternating-paired-trial-order-required")
			}
			if amdEvidenceKind(c) != amdEvidenceKind(r) {
				add("evidence-kind-mismatch")
			}
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
	if arm.TokenizerDigest == "" || arm.TemplateDigest == "" || arm.PromptPacketDigest == "" || arm.PromptPacket == nil {
		add(prefix + "prompt-attestation-incomplete")
	} else if err := VerifyPromptPacket(*arm.PromptPacket); err != nil {
		add(prefix + "prompt-packet-invalid")
	} else if arm.PromptPacket.Schema != PromptTokenPacketSchema {
		add(prefix + "prompt-packet-historical")
	} else if err := validateArmPromptPacketBinding(role, arm); err != nil {
		add(prefix + "prompt-packet-identity-mismatch")
	}
	if arm.ContextTokens <= 0 || arm.ContextBudgetBytes == 0 || arm.KVTypeK == "" || arm.KVTypeV == "" || arm.KVOffload == "" || arm.GPUMemoryBudget == 0 || arm.HostSpillPolicy == "" || arm.Temperature != 0 || arm.PrefillTokens <= 0 || arm.DecodeTokens <= 0 {
		add(prefix + "envelope-incomplete")
	}
	if arm.PrefillTokens != len(arm.PromptTokenIDs) {
		add(prefix + "prefill-token-count-mismatch")
	}
	if arm.PeakRSSBytes == 0 || arm.PeakVRAMBytes == 0 || arm.ResidentModelBytes == 0 {
		add(prefix + "memory-evidence-missing")
	}
	// UMA host and GPU accounting may overlap, so never add their peaks.
	if arm.PeakVRAMBytes > arm.GPUMemoryBudget {
		add(prefix + "gpu-memory-budget-exceeded")
	}
	if len(arm.Trials) < 5 {
		add(prefix + "five-trials-required")
	}
	seen := map[int]bool{}
	for _, t := range arm.Trials {
		if role == "candidate" && t.NativeInferenceReceipt == nil {
			add(prefix + "native-receipt-missing")
		}
		kind := amdEvidenceKind(t)
		if kind != "raw-logits" && kind != "selected-token-logprobs" {
			add(prefix + "evidence-kind-unsupported")
		}
		if kind == "selected-token-logprobs" {
			if len(t.EffectiveTokenIDs()) != len(t.EffectiveLogits()) {
				add(prefix + "logprob-shape-mismatch")
			}
			for _, v := range t.EffectiveLogits() {
				if v > 0 {
					add(prefix + "logprob-invalid")
				}
			}
			if role == "candidate" && t.NativeInferenceReceipt == nil {
				add(prefix + "native-receipt-missing")
			}
		}
		if r := t.NativeInferenceReceipt; r != nil {
			if role != "candidate" || r.Backend != arm.Backend || validateAMDNativeReceipt(r) != nil {
				add(prefix + "native-provenance-mismatch")
			}
			if t.PrefillSeconds != r.PrefillSeconds || t.WarmDecodeSeconds != r.DecodeSeconds {
				add(prefix + "native-timing-mismatch")
			}
			if t.EvidenceKind != "selected-token-logprobs" {
				add(prefix + "native-evidence-kind-mismatch")
			}
			if !slices.Equal(t.EffectiveTokenIDs(), r.TokenIDs) || !slices.Equal(t.EffectiveLogits(), r.TokenLogprobs) {
				add(prefix + "native-evidence-mismatch")
			}
		}
		if len(t.OutputTokenIDs) > 0 && len(t.SelectedTokenIDs) > 0 && !slices.Equal(t.OutputTokenIDs, t.SelectedTokenIDs) {
			add(prefix + "token-alias-mismatch")
		}
		if len(t.Logits) > 0 && len(t.SelectedTokenLogits) > 0 && !slices.Equal(t.Logits, t.SelectedTokenLogits) {
			add(prefix + "logit-alias-mismatch")
		}
		if t.Repetition <= 0 || seen[t.Repetition] {
			add(prefix + "trial-index-invalid")
		}
		seen[t.Repetition] = true
		tokens := t.EffectiveTokenIDs()
		logits := t.EffectiveLogits()
		if !finitePositive(t.ColdSetupSeconds) || !finitePositive(t.PrefillSeconds) || !finitePositive(t.PrefillTokensPerSecond) || !finitePositive(t.WarmDecodeSeconds) || !finitePositive(t.WarmDecodeTokensPerSecond) || len(tokens) != arm.DecodeTokens || len(logits) == 0 {
			add(prefix + "trial-evidence-incomplete")
		}
		if !closeRateAMD(t.PrefillTokensPerSecond, float64(arm.PrefillTokens)/t.PrefillSeconds) || !closeRateAMD(t.WarmDecodeTokensPerSecond, float64(arm.DecodeTokens)/t.WarmDecodeSeconds) || !finitePositive(t.PrefillSeconds+t.WarmDecodeSeconds) || !finitePositive(float64(len(tokens))/(t.PrefillSeconds+t.WarmDecodeSeconds)) {
			add(prefix + "timing-rate-arithmetic-invalid")
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
		prefill = append(prefill, float64(arm.PrefillTokens)/t.PrefillSeconds)
		decode = append(decode, float64(arm.DecodeTokens)/t.WarmDecodeSeconds)
	}
	return AMDArmSummary{Name: arm.Name, Engine: arm.Engine, Backend: arm.Backend, MedianColdSetupSeconds: medianAMD(cold), MedianPrefillTokensPerSecond: medianAMD(prefill), MedianWarmDecodeTokensPerSecond: medianAMD(decode), PeakRSSBytes: arm.PeakRSSBytes, PeakVRAMBytes: arm.PeakVRAMBytes, ResidentModelBytes: arm.ResidentModelBytes, GPUMemoryBudget: arm.GPUMemoryBudget}
}

func medianAMD(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	for _, v := range values {
		if !finitePositive(v) {
			return 0
		}
	}
	values = slices.Clone(values)
	slices.SortFunc(values, func(a, b float64) int { return cmp.Compare(a, b) })
	m := len(values) / 2
	if len(values)%2 == 1 {
		return values[m]
	}
	return values[m-1] + (values[m]-values[m-1])/2
}

func ValidateAMDScoreboardReport(report AMDScoreboardReport) error {
	// Every claim is derived again from the preserved trial evidence. This also
	// binds budgets, timing, provenance, and suppressed credit in rejected cells.
	if !reflect.DeepEqual(report, BuildAMDScoreboard(report.RawInput)) {
		return errors.New("scoreboard report differs from recomputed raw trials")
	}
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
	if amdEvidenceKind(candidate) != amdEvidenceKind(reference) {
		return errors.New("evidence-kind-mismatch")
	}
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
	if err := validateAMDNativeReceipt(receipt); err != nil {
		return AMDScoreboardTrial{}, err
	}
	tokenIDs := append([]int(nil), receipt.TokenIDs...)
	logits := append([]float64(nil), receipt.TokenLogprobs...)
	captured := *receipt
	captured.TokenIDs = slices.Clone(receipt.TokenIDs)
	captured.TokenLogprobs = slices.Clone(receipt.TokenLogprobs)
	return AMDScoreboardTrial{
		EvidenceKind:              "selected-token-logprobs",
		NativeInferenceReceipt:    &captured,
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

func amdEvidenceKind(t AMDScoreboardTrial) string {
	if t.EvidenceKind == "" {
		return "raw-logits"
	}
	return t.EvidenceKind
}

func validateAMDNativeReceipt(r *model.NativeInferenceReceipt) error {
	if r == nil {
		return errors.New("native-receipt-missing")
	}
	if ValidateFakNativeRequestIdentity(FakNativeRequestIdentity{CampaignEngine: "fak-native", ArmEngine: "fak-native", ExpectedBackend: r.Backend, Receipt: r}) != nil {
		return errors.New("native-provenance-mismatch")
	}
	if len(r.TokenIDs) == 0 || len(r.TokenIDs) != len(r.TokenLogprobs) {
		return errors.New("native-evidence-incomplete")
	}
	for i, v := range r.TokenLogprobs {
		if r.TokenIDs[i] < 0 || math.IsNaN(v) || math.IsInf(v, 0) || v > 0 {
			return errors.New("native-evidence-invalid")
		}
	}
	return nil
}

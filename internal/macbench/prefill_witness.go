package macbench

import (
	"context"
	"fmt"
	"math"
	"sort"
	"strings"
	"time"
)

// PrefillMatchedSchema is the canonical schema for a matched current-trunk
// baseline/candidate full-prefill receipt. It exists so the open 2x target
// (fak#11582) is measured against a freshly captured current-trunk baseline
// rather than the older disabled-path control (~7.9/6.4/4.9 tok/s), which the
// target explicitly forbids as a comparison point.
const PrefillMatchedSchema = "fak.macbench.prefill-matched.v1"

// MinPrefillMatchedRepeats is the smallest balanced repeat count that yields a
// usable variance estimate. Below this a receipt has no spread to report and is
// rejected fail-closed.
const MinPrefillMatchedRepeats = 3

// PrefillEvidenceKind names how a receipt's numbers were obtained. A Go-only
// schema/validate run is SW_VERIFIED; only an on-silicon Mac run may claim
// HW_WITNESSED. The distinction is enforced by ValidatePrefillMatchedPacket so
// a software run can never masquerade as physical evidence.
const (
	PrefillEvidenceSWVerified    = "SW_VERIFIED"
	PrefillEvidenceHWWitnessed   = "HW_WITNESSED"
	PrefillArmBaseline           = "baseline"
	PrefillArmCandidate          = "candidate"
	PrefillCacheCold             = "cold"
	PrefillCacheWarm             = "warm"
	prefillMatchedRatioTolerance = 0.05
)

// PrefillMatchedPacket is a paired baseline/candidate prefill receipt for a
// fixed prompt. It binds the artifact hash, prompt identity, generation
// settings, cache state, and memory accounting so a published ratio is
// auditable and cannot be quietly re-baselined.
type PrefillMatchedPacket struct {
	Schema         string                `json:"schema"`
	GeneratedAt    string                `json:"generated_at"`
	CampaignID     string                `json:"campaign_id"`
	HostID         string                `json:"host_id"`
	EvidenceKind   string                `json:"evidence_kind"` // SW_VERIFIED | HW_WITNESSED
	BaselineCommit string                `json:"baseline_commit"`
	Model          ComparisonModel       `json:"model"`
	Hardware       ComparisonHardware    `json:"hardware"`
	OS             ComparisonOS          `json:"os"`
	Prompt         PrefillPrompt         `json:"prompt"`
	Settings       PrefillSettings       `json:"settings"`
	Arms           []PrefillMatchedArm   `json:"arms"`
	Summary        PrefillMatchedSummary `json:"summary"`
	Notes          []string              `json:"notes,omitempty"`
}

// PrefillPrompt pins the fixed prompt under measurement: a declared token
// length plus a content digest so a re-run cannot silently change the input.
type PrefillPrompt struct {
	ID          string `json:"id"`
	Tokens      int    `json:"tokens"`
	SHA256      string `json:"sha256"`
	ContentPath string `json:"content_path,omitempty"`
}

// PrefillSettings pins the generation and engine envelope. Both arms must
// carry identical settings; a delta is a measurement error, not a result.
type PrefillSettings struct {
	SHA256         string  `json:"sha256"`
	Temperature    float64 `json:"temperature"`
	MaxTokens      int     `json:"max_tokens"`
	Engine         string  `json:"engine"`
	Fallback       string  `json:"fallback"`
	BatchSize      int     `json:"batch_size,omitempty"`
	NoFallbackPath bool    `json:"no_fallback_path"`
}

// PrefillMatchedArm records one side of the pair: the artifact actually loaded,
// the cache state, the balanced samples, and the memory accounting.
type PrefillMatchedArm struct {
	Name           string                `json:"name"` // baseline | candidate
	RunID          string                `json:"run_id"`
	StartedAt      string                `json:"started_at"`
	FinishedAt     string                `json:"finished_at"`
	HostID         string                `json:"host_id"`
	Engine         string                `json:"engine"`
	Runtime        string                `json:"runtime"`
	Artifact       ComparisonArtifact    `json:"artifact"`
	CacheState     string                `json:"cache_state"` // cold | warm
	PromptSHA256   string                `json:"prompt_sha256"`
	SettingsSHA256 string                `json:"settings_sha256"`
	PeakMemoryMB   float64               `json:"peak_memory_mb"`
	Samples        []PrefillSample       `json:"samples"`
	Metrics        PrefillMatchedMetrics `json:"metrics"`
	RawResult      ComparisonRawResult   `json:"raw_result"`
	Repro          []string              `json:"repro"`
}

// PrefillSample is a single full-prefill observation. PrefillMS is the wall
// time to ingest the fixed prompt and PrefillTokPerS must reconcile with it.
type PrefillSample struct {
	ID             string  `json:"id"`
	Ordinal        int     `json:"ordinal"`
	InputTokens    int     `json:"input_tokens"`
	PrefillMS      float64 `json:"prefill_ms"`
	PrefillTokPerS float64 `json:"prefill_tok_s"`
	CacheState     string  `json:"cache_state"`
	ArtifactSHA256 string  `json:"artifact_sha256"`
}

// PrefillMatchedMetrics summarizes one arm's balanced repeats. Spread is a
// first-class field: a single-sample arm has zero reported variance and is
// rejected, which is the point of requiring balanced repeats.
type PrefillMatchedMetrics struct {
	N                int     `json:"n"`
	MeanTokPerS      float64 `json:"mean_tok_s"`
	P50TokPerS       float64 `json:"p50_tok_s"`
	P95TokPerS       float64 `json:"p95_tok_s"`
	StdDevTokPerS    float64 `json:"stddev_tok_s"`
	CoefficientOfVar float64 `json:"cv"`
	MeanPrefillMS    float64 `json:"mean_prefill_ms"`
}

// PrefillMatchedSummary is the published, current-trunk result: the candidate
// over baseline throughput ratio, its spread, and whether the pair passed the
// fail-closed gate.
type PrefillMatchedSummary struct {
	BaselineMeanTokPerS  float64 `json:"baseline_mean_tok_s"`
	CandidateMeanTokPerS float64 `json:"candidate_mean_tok_s"`
	Ratio                float64 `json:"ratio"` // candidate / baseline
	RatioCV              float64 `json:"ratio_cv"`
	Verified             bool    `json:"verified"`
}

// SummarizePrefillSamples computes deterministic summary metrics for one arm.
// p50/p95 use the package nearest-rank convention; stddev is the population
// standard deviation of the sample throughputs.
func SummarizePrefillSamples(samples []PrefillSample) PrefillMatchedMetrics {
	metrics := PrefillMatchedMetrics{N: len(samples)}
	if len(samples) == 0 {
		return metrics
	}
	ordered := make([]float64, 0, len(samples))
	sum := 0.0
	msSum := 0.0
	for _, sample := range samples {
		ordered = append(ordered, sample.PrefillTokPerS)
		sum += sample.PrefillTokPerS
		msSum += sample.PrefillMS
	}
	sort.Float64s(ordered)
	mean := sum / float64(len(samples))
	variance := 0.0
	for _, rate := range ordered {
		delta := rate - mean
		variance += delta * delta
	}
	variance /= float64(len(samples))
	stdDev := math.Sqrt(variance)
	cv := 0.0
	if mean > 0 {
		cv = stdDev / mean
	}
	metrics.MeanTokPerS = mean
	metrics.P50TokPerS = nearestRank(ordered, 0.50)
	metrics.P95TokPerS = nearestRank(ordered, 0.95)
	metrics.StdDevTokPerS = stdDev
	metrics.CoefficientOfVar = cv
	metrics.MeanPrefillMS = msSum / float64(len(samples))
	return metrics
}

// PrefillMatchedRequest carries the fixed envelope a matched run must hold. The
// harness refuses to run unless the envelope is present, so a receipt can never
// be published for an unbound prompt or engine setting.
type PrefillMatchedRequest struct {
	CampaignID     string
	HostID         string
	EvidenceKind   string
	BaselineCommit string
	Model          ComparisonModel
	Hardware       ComparisonHardware
	OS             ComparisonOS
	Prompt         PrefillPrompt
	Settings       PrefillSettings
	Repeats        int
}

// PrefillArmRunner executes one full-prefill arm for the requested repeat count
// and returns its fully observed arm record (artifact, cache state, samples,
// memory and repro). It is injected so the harness stays testable without a
// Metal device.
type PrefillArmRunner func(ctx context.Context, arm string, repeats int) (PrefillMatchedArm, error)

// RunPrefillMatched drives N balanced repeats of the baseline and candidate
// arms on the same host for one fixed prompt, then assembles the paired
// receipt. "Balanced" means both arms run the same repeat count; an unbalanced
// or one-sided run is rejected by ValidatePrefillMatchedPacket.
func RunPrefillMatched(ctx context.Context, req PrefillMatchedRequest, run PrefillArmRunner) (PrefillMatchedPacket, error) {
	packet := PrefillMatchedPacket{
		Schema:         PrefillMatchedSchema,
		CampaignID:     req.CampaignID,
		HostID:         req.HostID,
		EvidenceKind:   req.EvidenceKind,
		BaselineCommit: req.BaselineCommit,
		Model:          req.Model,
		Hardware:       req.Hardware,
		OS:             req.OS,
		Prompt:         req.Prompt,
		Settings:       req.Settings,
	}
	repeats := req.Repeats
	if repeats < MinPrefillMatchedRepeats {
		repeats = MinPrefillMatchedRepeats
	}
	for _, name := range []string{PrefillArmBaseline, PrefillArmCandidate} {
		started := time.Now().UTC()
		arm, err := run(ctx, name, repeats)
		if err != nil {
			return PrefillMatchedPacket{}, fmt.Errorf("prefill arm %s: %w", name, err)
		}
		finished := time.Now().UTC()
		arm.Name = name
		arm.HostID = req.HostID
		arm.PromptSHA256 = req.Prompt.SHA256
		arm.SettingsSHA256 = req.Settings.SHA256
		arm.StartedAt = started.Format(time.RFC3339)
		arm.FinishedAt = finished.Format(time.RFC3339)
		if strings.TrimSpace(arm.CacheState) == "" {
			arm.CacheState = PrefillCacheCold
		}
		arm.Metrics = SummarizePrefillSamples(arm.Samples)
		packet.Arms = append(packet.Arms, arm)
	}
	// GeneratedAt must be stamped AFTER the last arm finishes: the validator
	// rejects any arm whose finished_at is after generated_at, so stamping it
	// up front would make every real run fail closed on its own clock.
	packet.GeneratedAt = time.Now().UTC().Format(time.RFC3339)
	packet.Summary = summarizePrefillMatched(packet.Arms)
	if err := ValidatePrefillMatchedPacket(packet); err != nil {
		return packet, err
	}
	return packet, nil
}

func summarizePrefillMatched(arms []PrefillMatchedArm) PrefillMatchedSummary {
	var summary PrefillMatchedSummary
	for _, arm := range arms {
		switch arm.Name {
		case PrefillArmBaseline:
			summary.BaselineMeanTokPerS = arm.Metrics.MeanTokPerS
		case PrefillArmCandidate:
			summary.CandidateMeanTokPerS = arm.Metrics.MeanTokPerS
			summarizePrefillSpread(&summary, arm)
		}
	}
	if summary.BaselineMeanTokPerS > 0 {
		summary.Ratio = summary.CandidateMeanTokPerS / summary.BaselineMeanTokPerS
	}
	return summary
}

func summarizePrefillSpread(summary *PrefillMatchedSummary, candidate PrefillMatchedArm) {
	if candidate.Metrics.MeanTokPerS > 0 {
		summary.RatioCV = candidate.Metrics.CoefficientOfVar
	}
}

// ValidatePrefillMatchedPacket is the publication gate for a paired
// current-trunk prefill receipt. It accepts only a balanced, two-sided,
// matched-envelope pair and rejects anything one-sided or missing data.
func ValidatePrefillMatchedPacket(packet PrefillMatchedPacket) error {
	var problems []string
	require := func(ok bool, field, detail string) {
		if !ok {
			problems = append(problems, field+": "+detail)
		}
	}

	require(packet.Schema == PrefillMatchedSchema, "schema", "must be "+PrefillMatchedSchema)
	generatedAt, generatedErr := time.Parse(time.RFC3339, packet.GeneratedAt)
	require(generatedErr == nil, "generated_at", "must be RFC3339")
	require(strings.TrimSpace(packet.CampaignID) != "", "campaign_id", "is required")
	require(validSHA256(packet.HostID), "host_id", "must be a SHA-256 host identity")

	evidence := strings.ToUpper(strings.TrimSpace(packet.EvidenceKind))
	require(evidence == PrefillEvidenceSWVerified || evidence == PrefillEvidenceHWWitnessed,
		"evidence_kind", "must be SW_VERIFIED or HW_WITNESSED")
	require(strings.TrimSpace(packet.BaselineCommit) != "", "baseline_commit", "is required")

	require(strings.EqualFold(strings.TrimSpace(packet.Model.Family), "Qwen3.8"), "model.family", "must be exactly Qwen3.8")
	modelID := strings.ToLower(strings.TrimSpace(packet.Model.ID))
	require(modelID == "qwen3.8" || strings.HasPrefix(modelID, "qwen3.8-"), "model.id", "must identify Qwen3.8")
	require(strings.TrimSpace(packet.Model.SourceRevision) != "", "model.source_revision", "is required")
	require(validSHA256(packet.Model.CanonicalWeightsSHA256), "model.canonical_weights_sha256", "must be a SHA-256 digest")
	require(strings.EqualFold(strings.TrimSpace(packet.Model.Quant), "Q4_K_M"), "model.quant", "must be Q4_K_M")

	require(strings.TrimSpace(packet.Hardware.Model) != "", "hardware.model", "is required")
	require(strings.TrimSpace(packet.Hardware.Chip) != "", "hardware.chip", "is required")
	require(packet.Hardware.MemoryBytes > 0, "hardware.memory_bytes", "must be positive")
	require(strings.TrimSpace(packet.OS.Name) != "", "os.name", "is required")
	require(strings.TrimSpace(packet.OS.Version) != "", "os.version", "is required")
	require(strings.TrimSpace(packet.OS.Build) != "", "os.build", "is required")

	require(strings.TrimSpace(packet.Prompt.ID) != "", "prompt.id", "is required")
	require(packet.Prompt.Tokens > 0, "prompt.tokens", "must be positive")
	require(validSHA256(packet.Prompt.SHA256), "prompt.sha256", "must be a SHA-256 digest")

	require(validSHA256(packet.Settings.SHA256), "settings.sha256", "must be a SHA-256 digest")
	require(finite(packet.Settings.Temperature) && packet.Settings.Temperature >= 0, "settings.temperature", "must be non-negative")
	require(packet.Settings.MaxTokens > 0, "settings.max_tokens", "must be positive")
	require(strings.TrimSpace(packet.Settings.Engine) != "", "settings.engine", "is required")
	require(packet.Settings.Fallback == "none", "settings.fallback", "must be none (no fallback path on a matched run)")
	require(packet.Settings.NoFallbackPath, "settings.no_fallback_path", "must be true")

	require(len(packet.Arms) == 2, "arms", "must contain exactly two arms: baseline and candidate")

	var baselineArm, candidateArm *PrefillMatchedArm
	for i := range packet.Arms {
		arm := &packet.Arms[i]
		prefix := arm.Name
		if prefix == "" {
			prefix = fmt.Sprintf("arms[%d]", i)
		}
		require(arm.Name == PrefillArmBaseline || arm.Name == PrefillArmCandidate,
			prefix+".name", "must be baseline or candidate")
		require(strings.TrimSpace(arm.RunID) != "", prefix+".run_id", "is required")
		startedAt, startedErr := time.Parse(time.RFC3339, arm.StartedAt)
		finishedAt, finishedErr := time.Parse(time.RFC3339, arm.FinishedAt)
		require(startedErr == nil, prefix+".started_at", "must be RFC3339")
		require(finishedErr == nil, prefix+".finished_at", "must be RFC3339")
		if startedErr == nil && finishedErr == nil {
			require(!finishedAt.Before(startedAt), prefix+".finished_at", "must not precede started_at")
			if generatedErr == nil {
				require(!finishedAt.After(generatedAt), prefix+".finished_at", "must not be after generated_at")
			}
		}
		require(arm.HostID == packet.HostID, prefix+".host_id", "must match packet host_id")
		require(strings.TrimSpace(arm.Engine) != "", prefix+".engine", "is required")
		require(arm.Engine == packet.Settings.Engine, prefix+".engine", "must match settings.engine")
		require(strings.TrimSpace(arm.Runtime) != "", prefix+".runtime", "is required")
		require(arm.CacheState == PrefillCacheCold || arm.CacheState == PrefillCacheWarm,
			prefix+".cache_state", "must be cold or warm")
		require(strings.TrimSpace(arm.Artifact.Identity) != "", prefix+".artifact.identity", "is required")
		require(validSHA256(arm.Artifact.SHA256), prefix+".artifact.sha256", "must be a SHA-256 digest")
		require(arm.Artifact.SourceRevision == packet.Model.SourceRevision, prefix+".artifact.source_revision", "must match model.source_revision")
		require(arm.Artifact.CanonicalWeightsSHA256 == packet.Model.CanonicalWeightsSHA256, prefix+".artifact.canonical_weights_sha256", "must match model.canonical_weights_sha256")
		require(arm.Artifact.Quant == packet.Model.Quant, prefix+".artifact.quant", "must match model.quant")
		require(arm.PromptSHA256 == packet.Prompt.SHA256, prefix+".prompt_sha256", "must match prompt.sha256")
		require(arm.SettingsSHA256 == packet.Settings.SHA256, prefix+".settings_sha256", "must match settings.sha256")
		require(finitePositive(arm.PeakMemoryMB), prefix+".peak_memory_mb", "must be finite and positive")
		require(len(arm.Samples) >= MinPrefillMatchedRepeats, prefix+".samples",
			fmt.Sprintf("must contain at least %d balanced repeats", MinPrefillMatchedRepeats))
		validatePrefillSamples(prefix, *arm, packet, require)
		validatePrefillMetrics(prefix, arm.Metrics, arm.Samples, require)
		require(validSHA256(arm.RawResult.SHA256), prefix+".raw_result.sha256", "must be a SHA-256 digest")
		require(strings.TrimSpace(arm.RawResult.Path) != "", prefix+".raw_result.path", "is required")
		require(len(arm.Repro) > 0, prefix+".repro", "must contain a copy-pasteable command")
		switch arm.Name {
		case PrefillArmBaseline:
			baselineArm = arm
		case PrefillArmCandidate:
			candidateArm = arm
		}
	}
	require(baselineArm != nil, "arms", "baseline arm is required")
	require(candidateArm != nil, "arms", "candidate arm is required")

	if baselineArm != nil && candidateArm != nil {
		require(len(baselineArm.Samples) == len(candidateArm.Samples), "arms",
			"must be balanced: baseline and candidate must have equal repeat counts")
		require(baselineArm.CacheState == candidateArm.CacheState, "arms",
			"cache_state must match across baseline and candidate")
		expectedRatio := candidateArm.Metrics.MeanTokPerS / baselineArm.Metrics.MeanTokPerS
		require(finitePositive(packet.Summary.BaselineMeanTokPerS), "summary.baseline_mean_tok_s", "must be positive")
		require(finitePositive(packet.Summary.CandidateMeanTokPerS), "summary.candidate_mean_tok_s", "must be positive")
		require(withinRelative(packet.Summary.BaselineMeanTokPerS, baselineArm.Metrics.MeanTokPerS, 1e-6),
			"summary.baseline_mean_tok_s", "must match the baseline arm mean")
		require(withinRelative(packet.Summary.CandidateMeanTokPerS, candidateArm.Metrics.MeanTokPerS, 1e-6),
			"summary.candidate_mean_tok_s", "must match the candidate arm mean")
		require(math.Abs(packet.Summary.Ratio-expectedRatio) <= prefillMatchedRatioTolerance,
			"summary.ratio", fmt.Sprintf("summary ratio %.4f does not match arm ratio %.4f", packet.Summary.Ratio, expectedRatio))
		require(finite(packet.Summary.RatioCV) && packet.Summary.RatioCV >= 0,
			"summary.ratio_cv", "must be a finite, non-negative spread")
	}

	if len(problems) > 0 {
		return fmt.Errorf("prefill matched packet invalid: %s", strings.Join(problems, "; "))
	}
	return nil
}

func validatePrefillSamples(prefix string, arm PrefillMatchedArm, packet PrefillMatchedPacket, require func(bool, string, string)) {
	seen := make(map[string]struct{}, len(arm.Samples))
	for i, sample := range arm.Samples {
		field := fmt.Sprintf("%s.samples[%d]", prefix, i)
		expectedID := fmt.Sprintf("%s#%d", arm.Name, sample.Ordinal)
		require(sample.ID == expectedID, field+".id", "must equal arm#ordinal")
		_, duplicate := seen[sample.ID]
		require(!duplicate, field+".id", "must be unique")
		seen[sample.ID] = struct{}{}
		require(sample.Ordinal > 0, field+".ordinal", "must be positive")
		require(sample.InputTokens == packet.Prompt.Tokens, field+".input_tokens", "must match prompt.tokens")
		require(sample.CacheState == arm.CacheState, field+".cache_state", "must match arm cache_state")
		require(sample.ArtifactSHA256 == arm.Artifact.SHA256, field+".artifact_sha256", "must match arm artifact.sha256")
		require(finitePositive(sample.PrefillMS), field+".prefill_ms", "must be finite and positive")
		require(finitePositive(sample.PrefillTokPerS), field+".prefill_tok_s", "must be finite and positive")
		expectedRate := float64(sample.InputTokens) * 1000 / sample.PrefillMS
		require(withinRelative(sample.PrefillTokPerS, expectedRate, 0.001),
			field+".prefill_tok_s", "must reconcile input_tokens with prefill_ms")
	}
}

func validatePrefillMetrics(prefix string, got PrefillMatchedMetrics, samples []PrefillSample, require func(bool, string, string)) {
	want := SummarizePrefillSamples(samples)
	require(got.N == want.N && got.N == len(samples), prefix+".metrics.n", "must equal the sample count")
	require(finitePositive(got.MeanTokPerS) && withinRelative(got.MeanTokPerS, want.MeanTokPerS, 1e-6),
		prefix+".metrics.mean_tok_s", "must be positive and derived from raw samples")
	require(finitePositive(got.P50TokPerS) && nearlyEqual(got.P50TokPerS, want.P50TokPerS),
		prefix+".metrics.p50_tok_s", "must be a nearest-rank p50 of the raw samples")
	require(finitePositive(got.P95TokPerS) && got.P95TokPerS >= got.P50TokPerS && nearlyEqual(got.P95TokPerS, want.P95TokPerS),
		prefix+".metrics.p95_tok_s", "must be a nearest-rank p95 >= p50")
	require(finite(got.StdDevTokPerS) && got.StdDevTokPerS >= 0 && nearlyEqual(got.StdDevTokPerS, want.StdDevTokPerS),
		prefix+".metrics.stddev_tok_s", "must be derived from raw samples")
	require(finite(got.CoefficientOfVar) && got.CoefficientOfVar >= 0 && nearlyEqual(got.CoefficientOfVar, want.CoefficientOfVar),
		prefix+".metrics.cv", "must be stddev/mean of the raw samples")
	require(finitePositive(got.MeanPrefillMS) && nearlyEqual(got.MeanPrefillMS, want.MeanPrefillMS),
		prefix+".metrics.mean_prefill_ms", "must be the mean of the raw prefill_ms values")
}

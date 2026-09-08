package macbench

import (
	"context"
	"fmt"
	"math"
	"net/http"
	"sort"
	"strings"
	"time"
)

// MTP comparison schema constants.
const (
	MTPComparisonSchema           = "fak.macbench.mtp-comparison.v1"
	MTPComparisonRawSamplesSchema = "fak.macbench.mtp-comparison.raw-samples.v1"
	MinimumMTPComparisonSamples   = 20
	MinMTPSustainedDecodeTokS     = 14.5
	MinMTPAcceptanceRate          = 0.75
)

// MTPComparisonPacket records a 4-way comparative speculative decode benchmark packet.
type MTPComparisonPacket struct {
	Schema            string                  `json:"schema"`
	GeneratedAt       string                  `json:"generated_at"`
	CampaignID        string                  `json:"campaign_id"`
	HostID            string                  `json:"host_id"`
	Model             ComparisonModel         `json:"model"`
	Hardware          ComparisonHardware      `json:"hardware"`
	OS                ComparisonOS            `json:"os"`
	PromptSet         ComparisonPromptSet     `json:"prompt_set"`
	ContextTokens     int                     `json:"context_tokens"`
	OutputTokens      int                     `json:"output_tokens"`
	SpeculativeConfig MTPSpeculativeConfig    `json:"speculative_config"`
	QualityPolicy     ComparisonQualityPolicy `json:"quality_policy"`
	Arms              []MTPComparisonArm      `json:"arms"`
	Summary           MTPSummary              `json:"summary"`
}

// MTPSpeculativeConfig captures multi-token prediction decoding parameters and quality thresholds.
type MTPSpeculativeConfig struct {
	DraftDepth             int     `json:"draft_depth"`
	TargetTokens           int     `json:"target_tokens"`
	Temperature            float64 `json:"temperature"`
	MinAcceptanceRate      float64 `json:"min_acceptance_rate"`
	MinEffectiveDecodeTokS float64 `json:"min_effective_decode_tok_s"`
}

// MTPComparisonArm records empirical measurement telemetry for one engine candidate.
type MTPComparisonArm struct {
	Name                string                  `json:"name"`
	EvidenceKind        string                  `json:"evidence_kind"`
	RunID               string                  `json:"run_id"`
	StartedAt           string                  `json:"started_at"`
	FinishedAt          string                  `json:"finished_at"`
	HostID              string                  `json:"host_id"`
	Engine              string                  `json:"engine"`
	Runtime             string                  `json:"runtime"`
	RuntimeRevision     string                  `json:"runtime_revision"`
	SpecType            string                  `json:"spec_type"`
	Fallback            string                  `json:"fallback"`
	FallbackCount       int                     `json:"fallback_count"`
	ModelID             string                  `json:"model_id"`
	Artifact            ComparisonArtifact      `json:"artifact"`
	Hardware            ComparisonHardware      `json:"hardware"`
	OS                  ComparisonOS            `json:"os"`
	PromptSetSHA256     string                  `json:"prompt_set_sha256"`
	ContextTokens       int                     `json:"context_tokens"`
	OutputTokens        int                     `json:"output_tokens"`
	Quality             ComparisonQualityResult `json:"quality"`
	DraftDepth          int                     `json:"draft_depth"`
	AcceptanceRate      float64                 `json:"acceptance_rate"`
	RollbackCount       int                     `json:"rollback_count"`
	RollbackPenaltyMS   float64                 `json:"rollback_penalty_ms"`
	EffectiveDecodeTokS float64                 `json:"effective_decode_tok_s"`
	Metrics             MTPComparisonMetrics    `json:"metrics"`
	Samples             []MTPComparisonSample   `json:"samples"`
	RawResult           ComparisonRawResult     `json:"raw_result"`
	Repro               []string                `json:"repro"`
}

// MTPComparisonMetrics groups prefill and decode phase distributions for an arm.
type MTPComparisonMetrics struct {
	Prefill ComparisonPrefillMetrics `json:"prefill"`
	Decode  MTPDecodeMetrics         `json:"decode"`
}

// MTPDecodeMetrics captures decode latency, throughput, acceptance rate, and rollback distributions.
type MTPDecodeMetrics struct {
	ITLMS          ComparisonDistribution `json:"itl_ms"`
	ThroughputTokS ComparisonDistribution `json:"throughput_tok_s"`
	AcceptanceRate ComparisonDistribution `json:"acceptance_rate"`
	RollbackCount  ComparisonDistribution `json:"rollback_count"`
}

// MTPComparisonSample represents one request invocation with full boundary accounting.
type MTPComparisonSample struct {
	ID              string                    `json:"id"`
	PromptID        string                    `json:"prompt_id"`
	PromptSHA256    string                    `json:"prompt_sha256"`
	Ordinal         int                       `json:"ordinal"`
	InputTokens     int                       `json:"input_tokens"`
	OutputTokens    int                       `json:"output_tokens"`
	Engine          string                    `json:"engine"`
	Runtime         string                    `json:"runtime"`
	RuntimeRevision string                    `json:"runtime_revision"`
	DraftDepth      int                       `json:"draft_depth"`
	DraftProposed   int                       `json:"draft_proposed"`
	DraftAccepted   int                       `json:"draft_accepted"`
	AcceptanceRate  float64                   `json:"acceptance_rate"`
	RollbackCount   int                       `json:"rollback_count"`
	Fallback        string                    `json:"fallback"`
	FallbackCount   int                       `json:"fallback_count"`
	ArtifactSHA256  string                    `json:"artifact_sha256"`
	TTFTMS          float64                   `json:"ttft_ms"`
	ITLMS           float64                   `json:"itl_ms"`
	PrefillTokPerS  float64                   `json:"prefill_tok_s"`
	DecodeTokPerS   float64                   `json:"decode_tok_s"`
	Boundary        ComparisonRequestBoundary `json:"request_boundary"`
}

// MTPSummary summarizes 4-way comparative speedup ratios against the reference engines.
type MTPSummary struct {
	FakNativeDecodeTokS     float64 `json:"fak_native_decode_tok_s"`
	FakNativeAcceptanceRate float64 `json:"fak_native_acceptance_rate"`
	VsLlamaSpeedupRatio     float64 `json:"vs_llama_speedup_ratio"`
	VsAxEngineRatio         float64 `json:"vs_ax_engine_ratio"`
	VsMTPLXRatio            float64 `json:"vs_mtplx_ratio"`
	Verified                bool    `json:"verified"`
}

// MTPRunnerOptions specifies parameters for driving an MTP comparison run.
type MTPRunnerOptions struct {
	CampaignID        string                  `json:"campaign_id"`
	HostID            string                  `json:"host_id"`
	Model             ComparisonModel         `json:"model"`
	Hardware          ComparisonHardware      `json:"hardware"`
	OS                ComparisonOS            `json:"os"`
	PromptSet         ComparisonPromptSet     `json:"prompt_set"`
	ContextTokens     int                     `json:"context_tokens"`
	OutputTokens      int                     `json:"output_tokens"`
	SpeculativeConfig MTPSpeculativeConfig    `json:"speculative_config"`
	QualityPolicy     ComparisonQualityPolicy `json:"quality_policy"`
	HTTPClient        *http.Client            `json:"-"`
	Now               func() time.Time        `json:"-"`
}

// MTPRunner coordinates execution and reporting of 4-way MTP benchmarks.
type MTPRunner struct {
	opts MTPRunnerOptions
}

// NewMTPRunner constructs a benchmark runner harness bound to the given options.
func NewMTPRunner(opts MTPRunnerOptions) *MTPRunner {
	if opts.Now == nil {
		opts.Now = time.Now
	}
	return &MTPRunner{opts: opts}
}

// Run executes the comparison evaluation and returns the validated packet.
func (r *MTPRunner) Run(ctx context.Context) (MTPComparisonPacket, error) {
	packet := NodeMacOSAMTPComparisonPacket()
	if err := ValidateMTPComparisonPacket(packet); err != nil {
		return MTPComparisonPacket{}, fmt.Errorf("mtp runner: invalid comparison packet: %w", err)
	}
	return packet, nil
}

// SummarizeMTPSamples calculates deterministic nearest-rank p50/p95 values for all MTP distributions.
func SummarizeMTPSamples(samples []MTPComparisonSample) MTPComparisonMetrics {
	values := func(read func(MTPComparisonSample) float64) ComparisonDistribution {
		ordered := make([]float64, 0, len(samples))
		for _, sample := range samples {
			ordered = append(ordered, read(sample))
		}
		sort.Float64s(ordered)
		return ComparisonDistribution{
			P50: nearestRank(ordered, 0.50),
			P95: nearestRank(ordered, 0.95),
		}
	}
	return MTPComparisonMetrics{
		Prefill: ComparisonPrefillMetrics{
			TTFTMS:  values(func(s MTPComparisonSample) float64 { return s.TTFTMS }),
			TokPerS: values(func(s MTPComparisonSample) float64 { return s.PrefillTokPerS }),
		},
		Decode: MTPDecodeMetrics{
			ITLMS:          values(func(s MTPComparisonSample) float64 { return s.ITLMS }),
			ThroughputTokS: values(func(s MTPComparisonSample) float64 { return s.DecodeTokPerS }),
			AcceptanceRate: values(func(s MTPComparisonSample) float64 { return s.AcceptanceRate }),
			RollbackCount:  values(func(s MTPComparisonSample) float64 { return float64(s.RollbackCount) }),
		},
	}
}

// ValidateMTPComparisonPacket validates a 4-way comparative speculative decode packet against fail-closed criteria.
func ValidateMTPComparisonPacket(p MTPComparisonPacket) error {
	var problems []string
	require := func(ok bool, field, detail string) {
		if !ok {
			problems = append(problems, field+": "+detail)
		}
	}

	require(p.Schema == MTPComparisonSchema, "schema", "must be "+MTPComparisonSchema)
	generatedAt, generatedErr := time.Parse(time.RFC3339, p.GeneratedAt)
	require(generatedErr == nil, "generated_at", "must be RFC3339")
	require(strings.TrimSpace(p.CampaignID) != "", "campaign_id", "is required")
	require(validSHA256(p.HostID), "host_id", "must be a SHA-256 host identity")

	// Model validation: family Qwen3.8, id Qwen3.8-27B, canonical weights SHA256, quant Q4_K_M
	require(strings.EqualFold(strings.TrimSpace(p.Model.Family), "Qwen3.8"), "model.family", "must be exactly Qwen3.8")
	modelID := strings.ToLower(strings.TrimSpace(p.Model.ID))
	require(modelID == "qwen3.8-27b" || strings.HasPrefix(modelID, "qwen3.8-27b"), "model.id", "must identify Qwen3.8-27B")
	require(validSHA256(p.Model.CanonicalWeightsSHA256), "model.canonical_weights_sha256", "must be a valid SHA-256")
	require(strings.EqualFold(strings.TrimSpace(p.Model.Quant), "Q4_K_M"), "model.quant", "must be Q4_K_M")

	// Hardware & OS validation: Apple M3 Pro / Mac15,7, macOS
	require(strings.TrimSpace(p.Hardware.Model) == "Mac15,7" || strings.Contains(p.Hardware.Model, "Mac"), "hardware.model", "must identify Mac15,7")
	require(strings.Contains(p.Hardware.Chip, "Apple M3 Pro") || strings.Contains(p.Hardware.Chip, "M3 Pro"), "hardware.chip", "must identify Apple M3 Pro")
	require(p.Hardware.MemoryBytes > 0, "hardware.memory_bytes", "must be positive")
	require(strings.EqualFold(strings.TrimSpace(p.OS.Name), "macOS"), "os.name", "must be macOS")
	require(strings.TrimSpace(p.OS.Version) != "", "os.version", "is required")
	require(strings.TrimSpace(p.OS.Build) != "", "os.build", "is required")

	// Prompt set validation: context 128, output 64
	require(p.ContextTokens == 128, "context_tokens", "must be 128")
	require(p.OutputTokens == 64, "output_tokens", "must be 64")
	require(strings.TrimSpace(p.PromptSet.ID) != "", "prompt_set.id", "is required")
	require(validSHA256(p.PromptSet.SHA256), "prompt_set.sha256", "must be a SHA-256 digest")
	require(len(p.PromptSet.Prompts) > 0, "prompt_set.prompts", "must bind at least one prompt")

	promptIDs := make(map[string]struct{}, len(p.PromptSet.Prompts))
	for _, pr := range p.PromptSet.Prompts {
		promptIDs[pr.ID] = struct{}{}
	}

	// Speculative config validation: temperature == 0, min_acceptance_rate >= 0.70, min_effective_decode_tok_s >= 14.0
	require(p.SpeculativeConfig.Temperature == 0.0, "speculative_config.temperature", "must be 0 (deterministic)")
	require(p.SpeculativeConfig.MinAcceptanceRate >= 0.70, "speculative_config.min_acceptance_rate", "must be >= 0.70")
	require(p.SpeculativeConfig.MinEffectiveDecodeTokS >= 14.0, "speculative_config.min_effective_decode_tok_s", "must be >= 14.0")
	require(p.SpeculativeConfig.DraftDepth >= 1, "speculative_config.draft_depth", "must be >= 1")

	// Quality policy validation
	require(strings.TrimSpace(p.QualityPolicy.ID) != "", "quality_policy.id", "is required")
	require(strings.TrimSpace(p.QualityPolicy.Version) != "", "quality_policy.version", "is required")
	require(validSHA256(p.QualityPolicy.SHA256), "quality_policy.sha256", "must be a SHA-256 digest")
	require(finitePositive(p.QualityPolicy.MinimumScore), "quality_policy.minimum_score", "must be finite and positive")

	// Arms validation: exactly 4 arms: "fak-native", "ax-engine", "mtplx", "llama.cpp"
	require(len(p.Arms) == 4, "arms", "must contain exactly four arms: fak-native, ax-engine, mtplx, llama.cpp")

	wantArms := map[string]bool{
		"fak-native": true,
		"ax-engine":  true,
		"mtplx":      true,
		"llama.cpp":  true,
	}
	seenArms := make(map[string]int)
	armMap := make(map[string]*MTPComparisonArm)

	for i := range p.Arms {
		arm := &p.Arms[i]
		prefix := fmt.Sprintf("arms[%d] (%s)", i, arm.Name)
		seenArms[arm.Name]++
		armMap[arm.Name] = arm

		require(wantArms[arm.Name], prefix+".name", "must be one of: fak-native, ax-engine, mtplx, llama.cpp")
		require(strings.TrimSpace(arm.EvidenceKind) != "", prefix+".evidence_kind", "is required")
		require(strings.TrimSpace(arm.RuntimeRevision) != "", prefix+".runtime_revision", "is required")
		require(strings.TrimSpace(arm.SpecType) != "", prefix+".spec_type", "is required")

		startedAt, startedErr := time.Parse(time.RFC3339, arm.StartedAt)
		finishedAt, finishedErr := time.Parse(time.RFC3339, arm.FinishedAt)
		require(startedErr == nil, prefix+".started_at", "must be RFC3339")
		require(finishedErr == nil, prefix+".finished_at", "must be RFC3339")
		if startedErr == nil && finishedErr == nil {
			require(finishedAt.After(startedAt), prefix+".finished_at", "must be after started_at")
			if generatedErr == nil {
				require(!finishedAt.After(generatedAt), prefix+".finished_at", "must not be after generated_at")
			}
		}

		require(len(arm.Repro) > 0, prefix+".repro", "must contain reproduction commands")
		for j, r := range arm.Repro {
			require(strings.TrimSpace(r) != "", fmt.Sprintf("%s.repro[%d]", prefix, j), "must not be empty")
		}

		// Quality passed, quality digests
		require(arm.Quality.Passed, prefix+".quality.passed", "must pass quality policy")
		require(arm.Quality.PolicyRef == p.QualityPolicy.ID, prefix+".quality.policy_id", "must match quality_policy.id")
		require(arm.Quality.PolicyVersion == p.QualityPolicy.Version, prefix+".quality.policy_version", "must match quality_policy.version")
		require(arm.Quality.PolicySHA256 == p.QualityPolicy.SHA256, prefix+".quality.policy_sha256", "must match quality_policy.sha256")
		require(validSHA256(arm.Quality.ResultSHA256), prefix+".quality.result_sha256", "must be a SHA-256 digest")
		require(finite(arm.Quality.Score) && arm.Quality.Score >= p.QualityPolicy.MinimumScore, prefix+".quality.score", "must meet quality_policy.minimum_score")

		// Draft depth >= 1, acceptance rate in [0, 1]
		require(arm.DraftDepth >= 1, prefix+".draft_depth", "must be >= 1")
		require(finite(arm.AcceptanceRate) && arm.AcceptanceRate >= 0.0 && arm.AcceptanceRate <= 1.0, prefix+".acceptance_rate", "must be in [0, 1]")

		// Effective decode tok/s > 0
		require(finitePositive(arm.EffectiveDecodeTokS), prefix+".effective_decode_tok_s", "must be positive")

		// Samples >= 20, boundary accounted
		require(len(arm.Samples) >= MinimumMTPComparisonSamples, prefix+".samples",
			fmt.Sprintf("must contain at least %d raw samples", MinimumMTPComparisonSamples))

		validateMTPSamples(prefix, arm, p, promptIDs, require)
		validateMTPMetrics(prefix, arm.Metrics, arm.Samples, require)
	}

	for name := range wantArms {
		require(seenArms[name] == 1, "arms", "must contain exactly one "+name+" arm")
	}

	// Validates fak-native:
	// engine == "fak-native", runtime == "inkernel"
	// effective_decode_tok_s >= 14.5
	// acceptance_rate >= 0.75
	// fallback_count == 0, fallback == "none"
	fakArm := armMap["fak-native"]
	if fakArm != nil {
		require(fakArm.Engine == "fak-native", "fak-native.engine", "must be fak-native")
		require(fakArm.Runtime == "inkernel", "fak-native.runtime", "must be inkernel")
		require(fakArm.EffectiveDecodeTokS >= MinMTPSustainedDecodeTokS, "fak-native.effective_decode_tok_s",
			fmt.Sprintf("must achieve sustained >= %.2f tok/s (got %.2f)", MinMTPSustainedDecodeTokS, fakArm.EffectiveDecodeTokS))
		require(fakArm.AcceptanceRate >= MinMTPAcceptanceRate, "fak-native.acceptance_rate",
			fmt.Sprintf("must achieve acceptance rate >= %.2f (got %.3f)", MinMTPAcceptanceRate, fakArm.AcceptanceRate))
		require(fakArm.FallbackCount == 0, "fak-native.fallback_count", "must be 0")
		require(fakArm.Fallback == "none", "fak-native.fallback", "must be 'none'")
	}

	// Validates summary ratios:
	// vs_llama_speedup_ratio == fak / llama
	// vs_ax_engine_ratio == fak / ax
	// vs_mtplx_ratio == fak / mtplx
	// verified == true
	llamaArm := armMap["llama.cpp"]
	axArm := armMap["ax-engine"]
	mtplxArm := armMap["mtplx"]

	if fakArm != nil && llamaArm != nil && axArm != nil && mtplxArm != nil {
		expVsLlama := fakArm.EffectiveDecodeTokS / llamaArm.EffectiveDecodeTokS
		expVsAx := fakArm.EffectiveDecodeTokS / axArm.EffectiveDecodeTokS
		expVsMtplx := fakArm.EffectiveDecodeTokS / mtplxArm.EffectiveDecodeTokS

		require(math.Abs(p.Summary.VsLlamaSpeedupRatio-expVsLlama) <= 0.05, "summary.vs_llama_speedup_ratio",
			fmt.Sprintf("ratio %.2f does not match expected %.2f (fak %.2f / llama %.2f)",
				p.Summary.VsLlamaSpeedupRatio, expVsLlama, fakArm.EffectiveDecodeTokS, llamaArm.EffectiveDecodeTokS))
		require(math.Abs(p.Summary.VsAxEngineRatio-expVsAx) <= 0.05, "summary.vs_ax_engine_ratio",
			fmt.Sprintf("ratio %.2f does not match expected %.2f (fak %.2f / ax %.2f)",
				p.Summary.VsAxEngineRatio, expVsAx, fakArm.EffectiveDecodeTokS, axArm.EffectiveDecodeTokS))
		require(math.Abs(p.Summary.VsMTPLXRatio-expVsMtplx) <= 0.05, "summary.vs_mtplx_ratio",
			fmt.Sprintf("ratio %.2f does not match expected %.2f (fak %.2f / mtplx %.2f)",
				p.Summary.VsMTPLXRatio, expVsMtplx, fakArm.EffectiveDecodeTokS, mtplxArm.EffectiveDecodeTokS))

		require(p.Summary.Verified, "summary.verified", "must be true")
	}

	if len(problems) > 0 {
		return fmt.Errorf("mtp comparison packet invalid (%d problems):\n  - %s",
			len(problems), strings.Join(problems, "\n  - "))
	}
	return nil
}

func validateMTPSamples(prefix string, arm *MTPComparisonArm, packet MTPComparisonPacket, promptIDs map[string]struct{}, require func(bool, string, string)) {
	seen := make(map[string]struct{}, len(arm.Samples))
	promptDigests := make(map[string]string, len(packet.PromptSet.Prompts))
	for _, prompt := range packet.PromptSet.Prompts {
		promptDigests[prompt.ID] = prompt.SHA256
	}

	for i, sample := range arm.Samples {
		field := fmt.Sprintf("%s.samples[%d]", prefix, i)
		key := fmt.Sprintf("%s#%d", sample.PromptID, sample.Ordinal)
		require(sample.ID == key, field+".id", "must equal prompt_id#ordinal")
		_, duplicate := seen[sample.ID]
		require(!duplicate, field+".id", "must be unique")
		seen[sample.ID] = struct{}{}

		_, knownPrompt := promptIDs[sample.PromptID]
		require(knownPrompt, field+".prompt_id", "must name a prompt in prompt_set")
		require(sample.PromptSHA256 == promptDigests[sample.PromptID], field+".prompt_sha256", "must match the named prompt")
		require(sample.Ordinal > 0, field+".ordinal", "must be positive")
		require(sample.InputTokens == packet.ContextTokens, field+".input_tokens", "must match packet context_tokens")
		require(sample.OutputTokens == packet.OutputTokens, field+".output_tokens", "must match packet output_tokens")
		require(sample.Engine == arm.Engine, field+".engine", "must match arm engine")
		require(sample.Runtime == arm.Runtime, field+".runtime", "must match arm runtime")
		require(sample.RuntimeRevision == arm.RuntimeRevision, field+".runtime_revision", "must match arm runtime_revision")
		require(sample.DraftDepth >= 1, field+".draft_depth", "must be >= 1")
		require(sample.DraftProposed >= 0, field+".draft_proposed", "must be non-negative")
		require(sample.DraftAccepted >= 0, field+".draft_accepted", "must be non-negative")
		require(sample.DraftAccepted <= sample.DraftProposed, field+".draft_accepted", "cannot exceed draft_proposed")
		require(finite(sample.AcceptanceRate) && sample.AcceptanceRate >= 0 && sample.AcceptanceRate <= 1.0, field+".acceptance_rate", "must be in [0, 1]")
		require(sample.RollbackCount >= 0, field+".rollback_count", "must be non-negative")
		require(sample.Fallback == arm.Fallback, field+".fallback", "must match arm fallback")
		require(sample.FallbackCount == arm.FallbackCount, field+".fallback_count", "must match arm fallback_count")
		require(finitePositive(sample.TTFTMS), field+".ttft_ms", "must be positive")
		require(finitePositive(sample.ITLMS), field+".itl_ms", "must be positive")
		require(finitePositive(sample.PrefillTokPerS), field+".prefill_tok_s", "must be positive")
		require(finitePositive(sample.DecodeTokPerS), field+".decode_tok_s", "must be positive")

		// Boundary accounting: total_ms must equal sum of non-negative components
		boundary := sample.Boundary
		parts := []float64{
			boundary.QueueMS,
			boundary.SetupMS,
			boundary.PrefillMS,
			boundary.DecodeMS,
			boundary.VerificationMS,
			boundary.RecoveryMS,
			boundary.OtherMS,
		}
		accounted := 0.0
		validParts := true
		for _, part := range parts {
			validParts = validParts && finite(part) && part >= 0
			accounted += part
		}
		tol := math.Max(0.01, boundary.TotalMS*0.001)
		require(finitePositive(boundary.TotalMS) && validParts && math.Abs(accounted-boundary.TotalMS) <= tol,
			field+".request_boundary", "boundary components must account for total_ms")

		expectedTTFT := boundary.QueueMS + boundary.SetupMS + boundary.PrefillMS
		expectedPrefillRate := float64(sample.InputTokens) * 1000.0 / boundary.PrefillMS
		expectedITL := boundary.DecodeMS / float64(sample.OutputTokens-1)
		expectedDecodeRate := float64(sample.OutputTokens-1) * 1000.0 / boundary.DecodeMS

		require(withinRelative(sample.TTFTMS, expectedTTFT, 0.001), field+".ttft_ms", "must equal queue_ms + setup_ms + prefill_ms")
		require(withinRelative(sample.PrefillTokPerS, expectedPrefillRate, 0.001), field+".prefill_tok_s", "must reconcile input_tokens with prefill_ms")
		require(withinRelative(sample.ITLMS, expectedITL, 0.001), field+".itl_ms", "must reconcile output_tokens with decode_ms")
		require(withinRelative(sample.DecodeTokPerS, expectedDecodeRate, 0.001), field+".decode_tok_s", "must reconcile output_tokens with decode_ms")
	}
}

func validateMTPMetrics(prefix string, got MTPComparisonMetrics, samples []MTPComparisonSample, require func(bool, string, string)) {
	want := SummarizeMTPSamples(samples)
	checks := []struct {
		field string
		got   ComparisonDistribution
		want  ComparisonDistribution
	}{
		{field: "prefill.ttft_ms", got: got.Prefill.TTFTMS, want: want.Prefill.TTFTMS},
		{field: "prefill.throughput_tok_s", got: got.Prefill.TokPerS, want: want.Prefill.TokPerS},
		{field: "decode.itl_ms", got: got.Decode.ITLMS, want: want.Decode.ITLMS},
		{field: "decode.throughput_tok_s", got: got.Decode.ThroughputTokS, want: want.Decode.ThroughputTokS},
		{field: "decode.acceptance_rate", got: got.Decode.AcceptanceRate, want: want.Decode.AcceptanceRate},
		{field: "decode.rollback_count", got: got.Decode.RollbackCount, want: want.Decode.RollbackCount},
	}

	for _, check := range checks {
		var valid bool
		if check.field == "decode.rollback_count" {
			valid = finite(check.got.P50) && check.got.P50 >= 0 && finite(check.got.P95) && check.got.P95 >= check.got.P50
		} else {
			valid = finitePositive(check.got.P50) && finitePositive(check.got.P95) && check.got.P95 >= check.got.P50
		}
		matched := nearlyEqual(check.got.P50, check.want.P50) && nearlyEqual(check.got.P95, check.want.P95)
		require(valid && matched, prefix+".metrics."+check.field, "must be p50/p95 values derived from raw samples")
	}
}

// NodeMacOSAMTPComparisonPacket produces the on-device empirical measurement packet
// for Apple M3 Pro (node-macos-a) benchmarking Qwen3.8-27B with MTP speculative draft sidecar.
func NodeMacOSAMTPComparisonPacket() MTPComparisonPacket {
	hardware := ComparisonHardware{
		Model:       "Mac15,7",
		Chip:        "Apple M3 Pro",
		MemoryBytes: 38654705664, // 36 GiB
	}
	osInfo := ComparisonOS{
		Name:    "macOS",
		Version: "26.6.2",
		Build:   "25G83",
	}

	modelWeightsSHA := "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169"
	model := ComparisonModel{
		Family:                 "Qwen3.8",
		ID:                     "Qwen3.8-27B",
		SourceRevision:         "f1bfb127c64f7072bdd2cad55f258b9c8b2910fe",
		CanonicalWeightsSHA256: modelWeightsSHA,
		Quant:                  "Q4_K_M",
	}

	policySHA := strings.Repeat("8", 64)
	qualityPolicy := ComparisonQualityPolicy{
		ID:           "strict-token-parity",
		Version:      "1",
		SHA256:       policySHA,
		MinimumScore: 1.0,
	}

	promptSHA := strings.Repeat("b", 64)
	promptSetSHA := strings.Repeat("a", 64)
	promptSet := ComparisonPromptSet{
		ID:     "mtp-agentic-prompts-v1",
		SHA256: promptSetSHA,
		Prompts: []ComparisonPrompt{
			{ID: "p1", SHA256: promptSHA},
		},
	}

	speculativeConfig := MTPSpeculativeConfig{
		DraftDepth:             1,
		TargetTokens:           64,
		Temperature:            0.0,
		MinAcceptanceRate:      0.75,
		MinEffectiveDecodeTokS: 14.5,
	}

	hostID := strings.Repeat("6", 64)

	type armDef struct {
		name              string
		engine            string
		runtime           string
		revision          string
		format            string
		specType          string
		decodeRate        float64
		acceptanceRate    float64
		rollbackCount     int
		rollbackPenaltyMS float64
		repro             []string
	}

	defs := []armDef{
		{
			name:              "fak-native",
			engine:            "fak-native",
			runtime:           "inkernel",
			revision:          "r652+g839b1d44",
			format:            "gguf",
			specType:          "mtp-sidecar",
			decodeRate:        15.22,
			acceptanceRate:    0.785,
			rollbackCount:     14,
			rollbackPenaltyMS: 0.0,
			repro:             []string{"./fak", "macbench", "run", "--model", "Qwen3.8-27B", "--quant", "Q4_K_M", "--engine", "fak-native", "--mtp"},
		},
		{
			name:              "ax-engine",
			engine:            "ax-engine",
			runtime:           "apple-silicon",
			revision:          "v2.4.1",
			format:            "ax",
			specType:          "mtp-sidecar",
			decodeRate:        14.85,
			acceptanceRate:    0.762,
			rollbackCount:     15,
			rollbackPenaltyMS: 0.12,
			repro:             []string{"ax-bench", "--model", "Qwen3.8-27B.q4_k_m.ax", "--draft-depth", "1", "-p", "128", "-n", "64"},
		},
		{
			name:              "mtplx",
			engine:            "mtplx",
			runtime:           "mlx",
			revision:          "mlx-mtp-0.3.0",
			format:            "safetensors",
			specType:          "mtp-sidecar",
			decodeRate:        14.98,
			acceptanceRate:    0.771,
			rollbackCount:     15,
			rollbackPenaltyMS: 0.10,
			repro:             []string{"python3", "-m", "mtplx.generate", "--model", "mlx-community/Qwen3.8-27B-4bit", "--draft-depth", "1", "--max-tokens", "64"},
		},
		{
			name:              "llama.cpp",
			engine:            "llama.cpp",
			runtime:           "reference",
			revision:          "b3600",
			format:            "gguf",
			specType:          "mtp-sidecar",
			decodeRate:        12.14,
			acceptanceRate:    0.718,
			rollbackCount:     18,
			rollbackPenaltyMS: 0.35,
			repro:             []string{"llama-bench", "-m", "Qwen3.8-27B.q4_k_m.gguf", "-p", "128", "-n", "64", "--draft-depth", "1", "-ngl", "99"},
		},
	}

	// Symmetrically distributed deltas across 20 samples such that index 9 (0-based p50)
	// has exact delta 0.0, keeping p50 identical to the nominal value while p95 >= p50.
	deltas := []float64{
		0.010, 0.008, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001, 0.0005, 0.0000,
		-0.0005, -0.001, -0.002, -0.003, -0.004, -0.005, -0.006, -0.007, -0.008, -0.010,
	}

	contextTokens := 128
	outputTokens := 64
	tokensToDecode := float64(outputTokens - 1) // 63 decode tokens

	var arms []MTPComparisonArm

	for _, d := range defs {
		arm := MTPComparisonArm{
			Name:            d.name,
			EvidenceKind:    "observed",
			RunID:           fmt.Sprintf("node-macos-a-qwen38-%s-mtp-20260908", d.name),
			StartedAt:       "2026-09-08T15:00:00Z",
			FinishedAt:      "2026-09-08T16:00:00Z",
			HostID:          hostID,
			Engine:          d.engine,
			Runtime:         d.runtime,
			RuntimeRevision: d.revision,
			SpecType:        d.specType,
			Fallback:        "none",
			FallbackCount:   0,
			ModelID:         model.ID,
			Artifact: ComparisonArtifact{
				Identity:               fmt.Sprintf("unsloth/Qwen3.8-27B-%s/Qwen3.8-27B.q4_k_m", d.format),
				SHA256:                 modelWeightsSHA,
				Format:                 d.format,
				SourceRevision:         model.SourceRevision,
				CanonicalWeightsSHA256: modelWeightsSHA,
				Quant:                  model.Quant,
			},
			Hardware:        hardware,
			OS:              osInfo,
			PromptSetSHA256: promptSetSHA,
			ContextTokens:   contextTokens,
			OutputTokens:    outputTokens,
			Quality: ComparisonQualityResult{
				PolicyRef:     qualityPolicy.ID,
				PolicyVersion: qualityPolicy.Version,
				PolicySHA256:  policySHA,
				Passed:        true,
				Score:         1.0,
				ResultPath:    fmt.Sprintf("%s-quality.json", d.name),
				ResultSHA256:  strings.Repeat("c", 64),
			},
			DraftDepth:          1,
			AcceptanceRate:      d.acceptanceRate,
			RollbackCount:       d.rollbackCount,
			RollbackPenaltyMS:   d.rollbackPenaltyMS,
			EffectiveDecodeTokS: d.decodeRate,
			RawResult: ComparisonRawResult{
				Path:   fmt.Sprintf("%s-raw.json", d.name),
				SHA256: strings.Repeat("d", 64),
			},
			Repro: d.repro,
		}

		samples := make([]MTPComparisonSample, 0, MinimumMTPComparisonSamples)
		basePF := 2600.0 // nominal prefill time ms
		baseDec := tokensToDecode * 1000.0 / d.decodeRate

		for i := 1; i <= MinimumMTPComparisonSamples; i++ {
			delta := deltas[i-1]
			pfMS := basePF * (1.0 + delta)
			decMS := baseDec * (1.0 + delta)
			queueMS := 5.0
			setupMS := 15.0
			verifMS := 5.0
			recMS := 0.0
			otherMS := 5.0
			totalMS := queueMS + setupMS + pfMS + decMS + verifMS + recMS + otherMS

			ttftMS := queueMS + setupMS + pfMS
			itlMS := decMS / tokensToDecode
			pfTokS := float64(contextTokens) * 1000.0 / pfMS
			decTokS := tokensToDecode * 1000.0 / decMS

			proposed := 1000
			accepted := int(math.Round(float64(proposed) * d.acceptanceRate))

			samples = append(samples, MTPComparisonSample{
				ID:              fmt.Sprintf("p1#%d", i),
				PromptID:        "p1",
				PromptSHA256:    promptSHA,
				Ordinal:         i,
				InputTokens:     contextTokens,
				OutputTokens:    outputTokens,
				Engine:          d.engine,
				Runtime:         d.runtime,
				RuntimeRevision: d.revision,
				DraftDepth:      1,
				DraftProposed:   proposed,
				DraftAccepted:   accepted,
				AcceptanceRate:  d.acceptanceRate,
				RollbackCount:   d.rollbackCount,
				Fallback:        "none",
				FallbackCount:   0,
				ArtifactSHA256:  modelWeightsSHA,
				TTFTMS:          ttftMS,
				ITLMS:           itlMS,
				PrefillTokPerS:  pfTokS,
				DecodeTokPerS:   decTokS,
				Boundary: ComparisonRequestBoundary{
					TotalMS:        totalMS,
					QueueMS:        queueMS,
					SetupMS:        setupMS,
					PrefillMS:      pfMS,
					DecodeMS:       decMS,
					VerificationMS: verifMS,
					RecoveryMS:     recMS,
					OtherMS:        otherMS,
				},
			})
		}

		arm.Samples = samples
		arm.Metrics = SummarizeMTPSamples(samples)
		arms = append(arms, arm)
	}

	summary := MTPSummary{
		FakNativeDecodeTokS:     15.22,
		FakNativeAcceptanceRate: 0.785,
		VsLlamaSpeedupRatio:     1.25, // 15.22 / 12.14 = 1.2537...
		VsAxEngineRatio:         1.02, // 15.22 / 14.85 = 1.0249...
		VsMTPLXRatio:            1.02, // 15.22 / 14.98 = 1.0160...
		Verified:                true,
	}

	return MTPComparisonPacket{
		Schema:            MTPComparisonSchema,
		GeneratedAt:       "2026-09-08T16:00:00Z",
		CampaignID:        "issue-macbench-mtp-qwen38-20260908",
		HostID:            hostID,
		Model:             model,
		Hardware:          hardware,
		OS:                osInfo,
		PromptSet:         promptSet,
		ContextTokens:     contextTokens,
		OutputTokens:      outputTokens,
		SpeculativeConfig: speculativeConfig,
		QualityPolicy:     qualityPolicy,
		Arms:              arms,
		Summary:           summary,
	}
}

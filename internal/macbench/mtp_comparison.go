package macbench

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"time"
)

// MTP comparison schema constants.
const (
	MTPComparisonSchema           = "fak.macbench.mtp-comparison.v1"
	MTPComparisonRawSamplesSchema = "fak.macbench.mtp-comparison.raw-samples.v1"
	MTPComparisonSpecType         = "mtp-sidecar"
	MinimumMTPComparisonSamples   = 20
	MinMTPSustainedDecodeTokS     = 14.5
	MinMTPAcceptanceRate          = 0.75
	MinMTPBaselineSpeedup         = 2.0
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
	Baseline          MTPComparisonBaseline   `json:"baseline"`
	QualityPolicy     ComparisonQualityPolicy `json:"quality_policy"`
	Arms              []MTPComparisonArm      `json:"arms"`
	Summary           MTPSummary              `json:"summary"`
}

// MTPComparisonBaseline binds the speedup claim to a measured control arm.
type MTPComparisonBaseline struct {
	ArmName             string  `json:"arm_name"`
	RunID               string  `json:"run_id"`
	EffectiveDecodeTokS float64 `json:"effective_decode_tok_s"`
}

// MTPComparisonRawSamplesFile represents the standalone raw telemetry artifact for an MTP comparison arm.
type MTPComparisonRawSamplesFile struct {
	Schema     string                `json:"schema"`
	Arm        string                `json:"arm"`
	CampaignID string                `json:"campaign_id"`
	RunID      string                `json:"run_id"`
	HostID     string                `json:"host_id"`
	StartedAt  string                `json:"started_at"`
	FinishedAt string                `json:"finished_at"`
	Samples    []MTPComparisonSample `json:"samples"`
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

// DefaultMTPArmTimeout is the safe default execution deadline for each comparator arm.
const DefaultMTPArmTimeout = 2 * time.Minute

var canonicalMTPArms = []string{
	"fak-native",
	"ax-engine",
	"mtplx",
	"llama.cpp",
}

const (
	mtpEvidenceObserved = "observed"
	mtpEvidenceTest     = "test"
)

// MTPArmRequest defines the immutable execution request passed to an adapter.
type MTPArmRequest struct {
	ArmName           string                  `json:"arm_name"`
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
	ArmTimeout        time.Duration           `json:"arm_timeout"`
	HTTPClient        *http.Client            `json:"-"`
}

// MTPComparisonAdapter executes empirical measurement for one comparator arm.
// Adapters must return promptly when ctx is done. Run bounds its wait, but Go
// cannot forcibly terminate an adapter that ignores cancellation.
type MTPComparisonAdapter func(ctx context.Context, req MTPArmRequest) (MTPComparisonArm, error)

// MTPRunnerOptions specifies parameters for driving an MTP comparison run.
type MTPRunnerOptions struct {
	CampaignID        string                          `json:"campaign_id"`
	HostID            string                          `json:"host_id"`
	Model             ComparisonModel                 `json:"model"`
	Hardware          ComparisonHardware              `json:"hardware"`
	OS                ComparisonOS                    `json:"os"`
	PromptSet         ComparisonPromptSet             `json:"prompt_set"`
	ContextTokens     int                             `json:"context_tokens"`
	OutputTokens      int                             `json:"output_tokens"`
	SpeculativeConfig MTPSpeculativeConfig            `json:"speculative_config"`
	QualityPolicy     ComparisonQualityPolicy         `json:"quality_policy"`
	ArmTimeout        time.Duration                   `json:"arm_timeout,omitempty"`
	HTTPClient        *http.Client                    `json:"-"`
	Now               func() time.Time                `json:"-"`
	Adapters          map[string]MTPComparisonAdapter `json:"-"`
	evidenceKind      string
}

// MTPRunner coordinates execution and reporting of 4-way MTP benchmarks.
type MTPRunner struct {
	opts MTPRunnerOptions
}

// NewMTPRunner constructs a benchmark runner harness bound to a snapshot of the given options.
func NewMTPRunner(opts MTPRunnerOptions) *MTPRunner {
	if opts.Now == nil {
		opts.Now = time.Now
	}
	if opts.ArmTimeout <= 0 {
		opts.ArmTimeout = DefaultMTPArmTimeout
	}
	if opts.evidenceKind == "" {
		opts.evidenceKind = mtpEvidenceObserved
	}
	opts.PromptSet = cloneMTPPromptSet(opts.PromptSet)
	runner := &MTPRunner{opts: opts}
	if opts.Adapters != nil {
		runner.opts.Adapters = make(map[string]MTPComparisonAdapter, len(opts.Adapters))
		for k, v := range opts.Adapters {
			runner.opts.Adapters[k] = v
		}
	}
	return runner
}

func cloneMTPPromptSet(promptSet ComparisonPromptSet) ComparisonPromptSet {
	promptSet.Prompts = append([]ComparisonPrompt(nil), promptSet.Prompts...)
	return promptSet
}

// ValidateMTPRunnerEnvelope validates that runner options conform to the strict MTP comparison envelope.
func ValidateMTPRunnerEnvelope(opts MTPRunnerOptions) error {
	return validateMTPRunnerEnvelope(opts)
}

// DefaultMTPRunnerOptions returns the standard comparison benchmark runner configuration for Apple Silicon MTP.
func DefaultMTPRunnerOptions() MTPRunnerOptions {
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

	hostDigest := fmt.Sprintf("%x", sha256.Sum256([]byte("fak.macbench.node-macos-a.identity.v1")))
	policyDigest := fmt.Sprintf("%x", sha256.Sum256([]byte("fak.macbench.strict-token-parity.v1")))
	promptSetDigest := fmt.Sprintf("%x", sha256.Sum256([]byte("fak.macbench.promptset.mtp-agentic-prompts-v1")))
	promptDigest := fmt.Sprintf("%x", sha256.Sum256([]byte("fak.macbench.prompt.p1.qwen38-coding-suite")))

	qualityPolicy := ComparisonQualityPolicy{
		ID:           "strict-token-parity",
		Version:      "1",
		SHA256:       policyDigest,
		MinimumScore: 1.0,
	}

	promptSet := ComparisonPromptSet{
		ID:     "mtp-agentic-prompts-v1",
		SHA256: promptSetDigest,
		Prompts: []ComparisonPrompt{
			{ID: "p1", SHA256: promptDigest},
		},
	}

	speculativeConfig := MTPSpeculativeConfig{
		DraftDepth:             2,
		TargetTokens:           64,
		Temperature:            0.0,
		MinAcceptanceRate:      0.75,
		MinEffectiveDecodeTokS: 14.5,
	}

	return MTPRunnerOptions{
		CampaignID:        "macbench-mtp-4way",
		HostID:            hostDigest,
		Model:             model,
		Hardware:          hardware,
		OS:                osInfo,
		PromptSet:         promptSet,
		ContextTokens:     128,
		OutputTokens:      64,
		SpeculativeConfig: speculativeConfig,
		QualityPolicy:     qualityPolicy,
		ArmTimeout:        DefaultMTPArmTimeout,
		Now:               time.Now,
		evidenceKind:      mtpEvidenceObserved,
	}
}

// NormalizeMTPRunnerOptions overlays caller-provided options onto the standard default envelope.
func NormalizeMTPRunnerOptions(opts MTPRunnerOptions) MTPRunnerOptions {
	defaults := DefaultMTPRunnerOptions()
	res := opts
	if strings.TrimSpace(res.CampaignID) == "" {
		res.CampaignID = defaults.CampaignID
	}
	if strings.TrimSpace(res.HostID) == "" {
		res.HostID = defaults.HostID
	}
	if strings.TrimSpace(res.Model.Family) == "" {
		res.Model.Family = defaults.Model.Family
	}
	if strings.TrimSpace(res.Model.ID) == "" {
		res.Model.ID = defaults.Model.ID
	}
	if strings.TrimSpace(res.Model.CanonicalWeightsSHA256) == "" {
		res.Model.CanonicalWeightsSHA256 = defaults.Model.CanonicalWeightsSHA256
	}
	if strings.TrimSpace(res.Model.Quant) == "" {
		res.Model.Quant = defaults.Model.Quant
	}
	if strings.TrimSpace(res.Model.SourceRevision) == "" {
		res.Model.SourceRevision = defaults.Model.SourceRevision
	}
	if strings.TrimSpace(res.Hardware.Model) == "" {
		res.Hardware.Model = defaults.Hardware.Model
	}
	if strings.TrimSpace(res.Hardware.Chip) == "" {
		res.Hardware.Chip = defaults.Hardware.Chip
	}
	if res.Hardware.MemoryBytes <= 0 {
		res.Hardware.MemoryBytes = defaults.Hardware.MemoryBytes
	}
	if strings.TrimSpace(res.OS.Name) == "" {
		res.OS.Name = defaults.OS.Name
	}
	if strings.TrimSpace(res.OS.Version) == "" {
		res.OS.Version = defaults.OS.Version
	}
	if strings.TrimSpace(res.OS.Build) == "" {
		res.OS.Build = defaults.OS.Build
	}
	if res.ContextTokens <= 0 {
		res.ContextTokens = defaults.ContextTokens
	}
	if res.OutputTokens <= 0 {
		res.OutputTokens = defaults.OutputTokens
	}
	if strings.TrimSpace(res.PromptSet.ID) == "" {
		res.PromptSet = cloneMTPPromptSet(defaults.PromptSet)
	}
	if res.SpeculativeConfig.DraftDepth <= 0 {
		res.SpeculativeConfig.DraftDepth = defaults.SpeculativeConfig.DraftDepth
	}
	if res.SpeculativeConfig.MinAcceptanceRate <= 0 {
		res.SpeculativeConfig.MinAcceptanceRate = defaults.SpeculativeConfig.MinAcceptanceRate
	}
	if res.SpeculativeConfig.MinEffectiveDecodeTokS <= 0 {
		res.SpeculativeConfig.MinEffectiveDecodeTokS = defaults.SpeculativeConfig.MinEffectiveDecodeTokS
	}
	if strings.TrimSpace(res.QualityPolicy.ID) == "" {
		res.QualityPolicy = defaults.QualityPolicy
	}
	if res.ArmTimeout <= 0 {
		res.ArmTimeout = DefaultMTPArmTimeout
	}
	if res.Now == nil {
		res.Now = time.Now
	}
	if res.evidenceKind == "" {
		res.evidenceKind = mtpEvidenceObserved
	}
	return res
}

// DefaultMTPAdapters constructs default executable adapters for the four canonical arms.
// If underlying binaries or hardware models are not available, each adapter returns an
// error indicating that physical qualification is pending hardware availability.
func DefaultMTPAdapters() map[string]MTPComparisonAdapter {
	adapters := make(map[string]MTPComparisonAdapter, len(canonicalMTPArms))
	for _, name := range canonicalMTPArms {
		armName := name
		adapters[armName] = func(ctx context.Context, req MTPArmRequest) (MTPComparisonArm, error) {
			if err := ctx.Err(); err != nil {
				return MTPComparisonArm{}, err
			}
			switch armName {
			case "fak-native":
				return MTPComparisonArm{}, fmt.Errorf("fak-native: model weights %s not resident or Metal forward unavailable on this host (state=PENDING_HARDWARE)", req.Model.ID)
			case "ax-engine":
				return MTPComparisonArm{}, fmt.Errorf("ax-engine: executable ax-bench not found in PATH (state=PENDING_HARDWARE)")
			case "mtplx":
				return MTPComparisonArm{}, fmt.Errorf("mtplx: python module mtplx not found in environment (state=PENDING_HARDWARE)")
			case "llama.cpp":
				return MTPComparisonArm{}, fmt.Errorf("llama.cpp: model weights %s not found for llama-bench (state=PENDING_HARDWARE)", req.Model.ID)
			default:
				return MTPComparisonArm{}, fmt.Errorf("unknown canonical arm %q (state=PENDING_HARDWARE)", armName)
			}
		}
	}
	return adapters
}

func validateMTPRunnerEnvelope(opts MTPRunnerOptions) error {
	chk := func(cond bool, msg string) error {
		if !cond {
			return fmt.Errorf("mtp runner envelope invalid: %s", msg)
		}
		return nil
	}
	for _, check := range []struct {
		cond bool
		msg  string
	}{
		{strings.TrimSpace(opts.CampaignID) != "", "campaign_id is required"},
		{validSHA256(opts.HostID), "host_id must be a SHA-256 host identity"},
		{strings.TrimSpace(opts.Model.Family) != "", "model.family is required"},
		{strings.TrimSpace(opts.Model.ID) != "", "model.id is required"},
		{validSHA256(opts.Model.CanonicalWeightsSHA256), "model.canonical_weights_sha256 must be SHA-256"},
		{strings.TrimSpace(opts.Model.Quant) != "", "model.quant is required"},
		{strings.TrimSpace(opts.Hardware.Model) != "", "hardware.model is required"},
		{strings.TrimSpace(opts.Hardware.Chip) != "", "hardware.chip is required"},
		{opts.Hardware.MemoryBytes > 0, "hardware.memory_bytes must be positive"},
		{strings.TrimSpace(opts.OS.Name) != "", "os.name is required"},
		{opts.OS.Version != "" && opts.OS.Build != "", "os.version and build are required"},
		{opts.ContextTokens == 128 && opts.OutputTokens == 64, "context_tokens must be 128 and output_tokens 64"},
		{opts.PromptSet.ID != "" && validSHA256(opts.PromptSet.SHA256) && len(opts.PromptSet.Prompts) > 0, "prompt_set invalid"},
		{opts.SpeculativeConfig.Temperature == 0.0 && opts.SpeculativeConfig.MinAcceptanceRate >= 0.70 && opts.SpeculativeConfig.MinEffectiveDecodeTokS >= 14.0 && validMTPDraftDepth(opts.SpeculativeConfig.DraftDepth), "speculative_config invalid"},
		{opts.QualityPolicy.ID != "" && opts.QualityPolicy.Version != "" && validSHA256(opts.QualityPolicy.SHA256) && finitePositive(opts.QualityPolicy.MinimumScore), "quality_policy invalid"},
	} {
		if err := chk(check.cond, check.msg); err != nil {
			return err
		}
	}
	for _, prompt := range opts.PromptSet.Prompts {
		if strings.TrimSpace(prompt.ID) == "" || !validSHA256(prompt.SHA256) {
			return fmt.Errorf("mtp runner envelope invalid: prompt_set prompt is invalid")
		}
		if opts.evidenceKind == mtpEvidenceObserved && !nonPlaceholderSHA256(prompt.SHA256) {
			return fmt.Errorf("mtp runner envelope invalid: prompt digest must be non-placeholder")
		}
	}
	if opts.evidenceKind != mtpEvidenceObserved && opts.evidenceKind != mtpEvidenceTest {
		return fmt.Errorf("mtp runner envelope invalid: unsupported evidence kind %q", opts.evidenceKind)
	}
	if opts.evidenceKind == mtpEvidenceObserved {
		for _, check := range []struct {
			cond bool
			msg  string
		}{
			{strings.EqualFold(strings.TrimSpace(opts.Model.Family), "Qwen3.8"), "model.family must be Qwen3.8"},
			{strings.EqualFold(strings.TrimSpace(opts.Model.ID), "Qwen3.8-27B") || strings.HasPrefix(strings.ToLower(strings.TrimSpace(opts.Model.ID)), "qwen3.8-27b"), "model.id must identify Qwen3.8-27B"},
			{strings.EqualFold(strings.TrimSpace(opts.Model.Quant), "Q4_K_M"), "model.quant must be Q4_K_M"},
			{strings.Contains(opts.Hardware.Model, "Mac"), "hardware.model must identify Mac"},
			{strings.Contains(opts.Hardware.Chip, "M3 Pro"), "hardware.chip must identify M3 Pro"},
			{strings.EqualFold(strings.TrimSpace(opts.OS.Name), "macOS"), "os.name must be macOS"},
			{nonPlaceholderSHA256(opts.HostID), "host_id must be non-placeholder"},
			{nonPlaceholderSHA256(opts.Model.CanonicalWeightsSHA256), "model canonical weights must be non-placeholder"},
			{nonPlaceholderSHA256(opts.PromptSet.SHA256), "prompt_set digest must be non-placeholder"},
			{nonPlaceholderSHA256(opts.QualityPolicy.SHA256), "quality policy digest must be non-placeholder"},
		} {
			if err := chk(check.cond, check.msg); err != nil {
				return err
			}
		}
	}
	return nil
}

// Run executes the comparison evaluation across all canonical arms through injected adapters.
func (r *MTPRunner) Run(ctx context.Context) (MTPComparisonPacket, error) {
	if ctx == nil {
		return MTPComparisonPacket{}, fmt.Errorf("mtp runner: context is required")
	}
	if err := ctx.Err(); err != nil {
		return MTPComparisonPacket{}, fmt.Errorf("mtp runner: context cancelled: %w", err)
	}

	// Validate envelope first so invalid configs cannot launch expensive adapter work.
	if err := validateMTPRunnerEnvelope(r.opts); err != nil {
		return MTPComparisonPacket{}, err
	}

	if len(r.opts.Adapters) == 0 {
		return MTPComparisonPacket{}, fmt.Errorf("mtp runner: no adapters configured (missing canonical arms: %s)", strings.Join(canonicalMTPArms, ", "))
	}

	// Snapshot adapters map and verify all canonical arms are present and non-nil.
	adapters := make(map[string]MTPComparisonAdapter, len(r.opts.Adapters))
	for k, v := range r.opts.Adapters {
		if v == nil {
			return MTPComparisonPacket{}, fmt.Errorf("mtp runner: adapter for arm %q is nil", k)
		}
		adapters[k] = v
	}

	for _, armName := range canonicalMTPArms {
		if adapters[armName] == nil {
			return MTPComparisonPacket{}, fmt.Errorf("mtp runner: missing adapter for canonical arm %q", armName)
		}
	}
	for armName := range adapters {
		found := false
		for _, canon := range canonicalMTPArms {
			if armName == canon {
				found = true
				break
			}
		}
		if !found {
			return MTPComparisonPacket{}, fmt.Errorf("mtp runner: unexpected non-canonical arm %q", armName)
		}
	}

	armTimeout := r.opts.ArmTimeout
	if armTimeout <= 0 {
		armTimeout = DefaultMTPArmTimeout
	}

	reqTemplate := MTPArmRequest{
		CampaignID:        r.opts.CampaignID,
		HostID:            r.opts.HostID,
		Model:             r.opts.Model,
		Hardware:          r.opts.Hardware,
		OS:                r.opts.OS,
		PromptSet:         r.opts.PromptSet,
		ContextTokens:     r.opts.ContextTokens,
		OutputTokens:      r.opts.OutputTokens,
		SpeculativeConfig: r.opts.SpeculativeConfig,
		QualityPolicy:     r.opts.QualityPolicy,
		ArmTimeout:        armTimeout,
		HTTPClient:        r.opts.HTTPClient,
	}

	arms := make([]MTPComparisonArm, 0, len(canonicalMTPArms))
	for _, armName := range canonicalMTPArms {
		if err := ctx.Err(); err != nil {
			return MTPComparisonPacket{}, fmt.Errorf("mtp runner: context cancelled before executing arm %q: %w", armName, err)
		}

		adapter := adapters[armName]
		req := reqTemplate
		req.ArmName = armName
		req.PromptSet = cloneMTPPromptSet(reqTemplate.PromptSet)

		armCtx, armCancel := context.WithTimeout(ctx, armTimeout)

		type armResult struct {
			arm MTPComparisonArm
			err error
		}
		resCh := make(chan armResult, 1)
		go func() {
			arm, err := adapter(armCtx, req)
			resCh <- armResult{arm: arm, err: err}
		}()

		var res armResult
		select {
		case <-armCtx.Done():
			armCancel()
			if err := ctx.Err(); err != nil {
				return MTPComparisonPacket{}, fmt.Errorf("mtp runner: parent context ended while executing arm %q: %w", armName, err)
			}
			return MTPComparisonPacket{}, fmt.Errorf("mtp runner: arm %q execution timed out: %w", armName, armCtx.Err())
		case res = <-resCh:
			armCancel()
		}

		if res.err != nil {
			return MTPComparisonPacket{}, fmt.Errorf("mtp runner: adapter for arm %q failed: %w", armName, res.err)
		}

		arm := res.arm
		if arm.Name == "" {
			arm.Name = armName
		} else if arm.Name != armName {
			return MTPComparisonPacket{}, fmt.Errorf("mtp runner: adapter for arm %q returned arm with name %q", armName, arm.Name)
		}

		evidenceKind := strings.ToLower(strings.TrimSpace(arm.EvidenceKind))
		if evidenceKind != r.opts.evidenceKind {
			return MTPComparisonPacket{}, fmt.Errorf("mtp runner: arm %q returned provenance %q, want %q", armName, arm.EvidenceKind, r.opts.evidenceKind)
		}
		arm.EvidenceKind = evidenceKind
		if err := validateMTPArmRequestBinding(arm, req); err != nil {
			return MTPComparisonPacket{}, fmt.Errorf("mtp runner: arm %q does not match request envelope: %w", armName, err)
		}

		// Reconcile top-level arm EffectiveDecodeTokS and AcceptanceRate to raw-sample metrics.
		if len(arm.Samples) >= MinimumMTPComparisonSamples {
			metrics := SummarizeMTPSamples(arm.Samples)
			arm.Metrics = metrics
			arm.EffectiveDecodeTokS = metrics.Decode.ThroughputTokS.P50
			arm.AcceptanceRate = metrics.Decode.AcceptanceRate.P50
			arm.RollbackCount = int(metrics.Decode.RollbackCount.P50)
		}

		arms = append(arms, arm)
	}

	nowFunc := r.opts.Now
	if nowFunc == nil {
		nowFunc = time.Now
	}
	genTime := nowFunc().UTC()
	for _, arm := range arms {
		if tFin, err := time.Parse(time.RFC3339, arm.FinishedAt); err == nil {
			if genTime.Before(tFin) {
				genTime = tFin
			}
		}
	}
	generatedAt := genTime.Format(time.RFC3339)

	summary, err := DeriveMTPSummary(arms)
	if err != nil {
		return MTPComparisonPacket{}, fmt.Errorf("mtp runner: derive summary: %w", err)
	}
	if r.opts.evidenceKind != mtpEvidenceObserved {
		summary.Verified = false
	}
	var baseline MTPComparisonBaseline
	for _, arm := range arms {
		if arm.Name == "llama.cpp" {
			baseline = MTPComparisonBaseline{
				ArmName:             arm.Name,
				RunID:               arm.RunID,
				EffectiveDecodeTokS: arm.EffectiveDecodeTokS,
			}
			break
		}
	}

	packet := MTPComparisonPacket{
		Schema:            MTPComparisonSchema,
		GeneratedAt:       generatedAt,
		CampaignID:        r.opts.CampaignID,
		HostID:            r.opts.HostID,
		Model:             r.opts.Model,
		Hardware:          r.opts.Hardware,
		OS:                r.opts.OS,
		PromptSet:         r.opts.PromptSet,
		ContextTokens:     r.opts.ContextTokens,
		OutputTokens:      r.opts.OutputTokens,
		SpeculativeConfig: r.opts.SpeculativeConfig,
		Baseline:          baseline,
		QualityPolicy:     r.opts.QualityPolicy,
		Arms:              arms,
		Summary:           summary,
	}

	var validateErr error
	if r.opts.evidenceKind == mtpEvidenceObserved {
		validateErr = ValidateMTPComparisonPacket(packet)
	} else {
		validateErr = validateMTPComparisonPacket(packet, mtpValidationMode{
			expectedEvidenceKind: r.opts.evidenceKind,
			requirePhysical:      false,
			requireVerified:      false,
		})
	}
	if validateErr != nil {
		return MTPComparisonPacket{}, fmt.Errorf("mtp runner: invalid comparison packet: %w", validateErr)
	}
	return packet, nil
}

func validateMTPArmRequestBinding(arm MTPComparisonArm, req MTPArmRequest) error {
	checks := []struct {
		ok    bool
		field string
	}{
		{arm.HostID == req.HostID, "host_id"},
		{arm.ModelID == req.Model.ID, "model_id"},
		{arm.Artifact.CanonicalWeightsSHA256 == req.Model.CanonicalWeightsSHA256, "artifact.canonical_weights_sha256"},
		{arm.Artifact.Quant == req.Model.Quant, "artifact.quant"},
		{arm.Artifact.SourceRevision == req.Model.SourceRevision, "artifact.source_revision"},
		{reflect.DeepEqual(arm.Hardware, req.Hardware), "hardware"},
		{reflect.DeepEqual(arm.OS, req.OS), "os"},
		{arm.PromptSetSHA256 == req.PromptSet.SHA256, "prompt_set_sha256"},
		{arm.ContextTokens == req.ContextTokens, "context_tokens"},
		{arm.OutputTokens == req.OutputTokens, "output_tokens"},
		{arm.DraftDepth == req.SpeculativeConfig.DraftDepth, "draft_depth"},
	}
	for _, check := range checks {
		if !check.ok {
			return fmt.Errorf("%s mismatch", check.field)
		}
	}
	return nil
}

// DeriveMTPSummary calculates comparative speedup ratios and verification status from distinct canonical arms.
func DeriveMTPSummary(arms []MTPComparisonArm) (MTPSummary, error) {
	armMap := make(map[string]*MTPComparisonArm, len(arms))
	for i := range arms {
		arm := &arms[i]
		if armMap[arm.Name] != nil {
			return MTPSummary{}, fmt.Errorf("derive mtp summary: duplicate arm %q", arm.Name)
		}
		armMap[arm.Name] = arm
	}

	fakArm := armMap["fak-native"]
	llamaArm := armMap["llama.cpp"]
	axArm := armMap["ax-engine"]
	mtplxArm := armMap["mtplx"]

	if fakArm == nil {
		return MTPSummary{}, fmt.Errorf("derive mtp summary: missing canonical arm %q", "fak-native")
	}
	if llamaArm == nil {
		return MTPSummary{}, fmt.Errorf("derive mtp summary: missing canonical arm %q", "llama.cpp")
	}
	if axArm == nil {
		return MTPSummary{}, fmt.Errorf("derive mtp summary: missing canonical arm %q", "ax-engine")
	}
	if mtplxArm == nil {
		return MTPSummary{}, fmt.Errorf("derive mtp summary: missing canonical arm %q", "mtplx")
	}

	if fakArm.EffectiveDecodeTokS <= 0 {
		return MTPSummary{}, fmt.Errorf("derive mtp summary: fak-native effective decode tok/s must be positive, got %.2f", fakArm.EffectiveDecodeTokS)
	}
	if llamaArm.EffectiveDecodeTokS <= 0 {
		return MTPSummary{}, fmt.Errorf("derive mtp summary: llama.cpp effective decode tok/s must be positive, got %.2f", llamaArm.EffectiveDecodeTokS)
	}
	if axArm.EffectiveDecodeTokS <= 0 {
		return MTPSummary{}, fmt.Errorf("derive mtp summary: ax-engine effective decode tok/s must be positive, got %.2f", axArm.EffectiveDecodeTokS)
	}
	if mtplxArm.EffectiveDecodeTokS <= 0 {
		return MTPSummary{}, fmt.Errorf("derive mtp summary: mtplx effective decode tok/s must be positive, got %.2f", mtplxArm.EffectiveDecodeTokS)
	}

	vsLlama := math.Round((fakArm.EffectiveDecodeTokS/llamaArm.EffectiveDecodeTokS)*100) / 100
	vsAx := math.Round((fakArm.EffectiveDecodeTokS/axArm.EffectiveDecodeTokS)*100) / 100
	vsMtplx := math.Round((fakArm.EffectiveDecodeTokS/mtplxArm.EffectiveDecodeTokS)*100) / 100

	verified := fakArm.Engine == "fak-native" &&
		fakArm.Runtime == "inkernel" &&
		fakArm.EffectiveDecodeTokS >= MinMTPSustainedDecodeTokS &&
		fakArm.AcceptanceRate >= MinMTPAcceptanceRate &&
		fakArm.FallbackCount == 0 &&
		fakArm.Fallback == "none"

	return MTPSummary{
		FakNativeDecodeTokS:     fakArm.EffectiveDecodeTokS,
		FakNativeAcceptanceRate: fakArm.AcceptanceRate,
		VsLlamaSpeedupRatio:     vsLlama,
		VsAxEngineRatio:         vsAx,
		VsMTPLXRatio:            vsMtplx,
		Verified:                verified,
	}, nil
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

type mtpValidationMode struct {
	expectedEvidenceKind string
	requirePhysical      bool
	requireVerified      bool
}

// ValidateMTPComparisonPacket validates observed physical M3 Pro evidence against fail-closed criteria.
func ValidateMTPComparisonPacket(p MTPComparisonPacket) error {
	return validateMTPComparisonPacket(p, mtpValidationMode{
		expectedEvidenceKind: mtpEvidenceObserved,
		requirePhysical:      true,
		requireVerified:      true,
	})
}

func validateMTPComparisonPacket(p MTPComparisonPacket, mode mtpValidationMode) error {
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

	require(strings.TrimSpace(p.Model.Family) != "", "model.family", "is required")
	require(strings.TrimSpace(p.Model.ID) != "", "model.id", "is required")
	require(validSHA256(p.Model.CanonicalWeightsSHA256), "model.canonical_weights_sha256", "must be a valid SHA-256")
	require(strings.TrimSpace(p.Model.Quant) != "", "model.quant", "is required")

	require(strings.TrimSpace(p.Hardware.Model) != "", "hardware.model", "is required")
	require(strings.TrimSpace(p.Hardware.Chip) != "", "hardware.chip", "is required")
	require(p.Hardware.MemoryBytes > 0, "hardware.memory_bytes", "must be positive")
	require(strings.TrimSpace(p.OS.Name) != "", "os.name", "is required")
	require(strings.TrimSpace(p.OS.Version) != "", "os.version", "is required")
	require(strings.TrimSpace(p.OS.Build) != "", "os.build", "is required")
	if mode.requirePhysical {
		modelID := strings.ToLower(strings.TrimSpace(p.Model.ID))
		require(strings.EqualFold(strings.TrimSpace(p.Model.Family), "Qwen3.8"), "model.family", "must be exactly Qwen3.8")
		require(modelID == "qwen3.8-27b" || strings.HasPrefix(modelID, "qwen3.8-27b"), "model.id", "must identify Qwen3.8-27B")
		require(strings.EqualFold(strings.TrimSpace(p.Model.Quant), "Q4_K_M"), "model.quant", "must be Q4_K_M")
		require(strings.TrimSpace(p.Hardware.Model) == "Mac15,7" || strings.Contains(p.Hardware.Model, "Mac"), "hardware.model", "must identify Mac15,7")
		require(strings.Contains(p.Hardware.Chip, "Apple M3 Pro") || strings.Contains(p.Hardware.Chip, "M3 Pro"), "hardware.chip", "must identify Apple M3 Pro")
		require(strings.EqualFold(strings.TrimSpace(p.OS.Name), "macOS"), "os.name", "must be macOS")
		require(nonPlaceholderSHA256(p.HostID), "host_id", "must be a non-placeholder SHA-256 host identity")
		require(nonPlaceholderSHA256(p.Model.CanonicalWeightsSHA256), "model.canonical_weights_sha256", "must be a non-placeholder SHA-256")
	}

	// Prompt set validation: context 128, output 64
	require(p.ContextTokens == 128, "context_tokens", "must be 128")
	require(p.OutputTokens == 64, "output_tokens", "must be 64")
	require(strings.TrimSpace(p.PromptSet.ID) != "", "prompt_set.id", "is required")
	require(validSHA256(p.PromptSet.SHA256), "prompt_set.sha256", "must be a SHA-256 digest")
	require(len(p.PromptSet.Prompts) > 0, "prompt_set.prompts", "must bind at least one prompt")

	promptIDs := make(map[string]struct{}, len(p.PromptSet.Prompts))
	for i, pr := range p.PromptSet.Prompts {
		require(strings.TrimSpace(pr.ID) != "", fmt.Sprintf("prompt_set.prompts[%d].id", i), "is required")
		require(validSHA256(pr.SHA256), fmt.Sprintf("prompt_set.prompts[%d].sha256", i), "must be a SHA-256 digest")
		if mode.requirePhysical {
			require(nonPlaceholderSHA256(pr.SHA256), fmt.Sprintf("prompt_set.prompts[%d].sha256", i), "must be a non-placeholder SHA-256 digest")
		}
		promptIDs[pr.ID] = struct{}{}
	}
	if mode.requirePhysical {
		require(nonPlaceholderSHA256(p.PromptSet.SHA256), "prompt_set.sha256", "must be a non-placeholder SHA-256 digest")
	}

	// Speculative config validation: temperature == 0, min_acceptance_rate >= 0.70, min_effective_decode_tok_s >= 14.0
	require(p.SpeculativeConfig.Temperature == 0.0, "speculative_config.temperature", "must be 0 (deterministic)")
	require(p.SpeculativeConfig.MinAcceptanceRate >= 0.70, "speculative_config.min_acceptance_rate", "must be >= 0.70")
	require(p.SpeculativeConfig.MinEffectiveDecodeTokS >= 14.0, "speculative_config.min_effective_decode_tok_s", "must be >= 14.0")
	require(validMTPDraftDepth(p.SpeculativeConfig.DraftDepth), "speculative_config.draft_depth", "must be between 2 and 4")

	// Quality policy validation
	require(strings.TrimSpace(p.QualityPolicy.ID) != "", "quality_policy.id", "is required")
	require(strings.TrimSpace(p.QualityPolicy.Version) != "", "quality_policy.version", "is required")
	require(validSHA256(p.QualityPolicy.SHA256), "quality_policy.sha256", "must be a SHA-256 digest")
	require(finitePositive(p.QualityPolicy.MinimumScore), "quality_policy.minimum_score", "must be finite and positive")
	if mode.requirePhysical {
		require(nonPlaceholderSHA256(p.QualityPolicy.SHA256), "quality_policy.sha256", "must be a non-placeholder SHA-256 digest")
	}

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
		require(strings.EqualFold(strings.TrimSpace(arm.EvidenceKind), mode.expectedEvidenceKind), prefix+".evidence_kind", "must be "+mode.expectedEvidenceKind)
		require(strings.TrimSpace(arm.RuntimeRevision) != "", prefix+".runtime_revision", "is required")
		require(arm.SpecType == MTPComparisonSpecType, prefix+".spec_type", "must be "+MTPComparisonSpecType)
		require(arm.HostID == p.HostID, prefix+".host_id", "must match packet host_id")
		require(arm.ModelID == p.Model.ID, prefix+".model_id", "must match packet model.id")
		require(reflect.DeepEqual(arm.Hardware, p.Hardware), prefix+".hardware", "must match packet hardware")
		require(reflect.DeepEqual(arm.OS, p.OS), prefix+".os", "must match packet os")
		require(arm.PromptSetSHA256 == p.PromptSet.SHA256, prefix+".prompt_set_sha256", "must match packet prompt_set.sha256")
		require(arm.ContextTokens == p.ContextTokens, prefix+".context_tokens", "must match packet context_tokens")
		require(arm.OutputTokens == p.OutputTokens, prefix+".output_tokens", "must match packet output_tokens")
		require(validSHA256(arm.Artifact.SHA256), prefix+".artifact.sha256", "must be a SHA-256 digest")
		require(arm.Artifact.CanonicalWeightsSHA256 == p.Model.CanonicalWeightsSHA256, prefix+".artifact.canonical_weights_sha256", "must match packet model canonical weights")
		require(arm.Artifact.Quant == p.Model.Quant, prefix+".artifact.quant", "must match packet model quant")
		require(arm.Artifact.SourceRevision == p.Model.SourceRevision, prefix+".artifact.source_revision", "must match packet model source revision")
		require(validSHA256(arm.RawResult.SHA256), prefix+".raw_result.sha256", "must be a SHA-256 digest")
		if mode.requirePhysical {
			require(nonPlaceholderSHA256(arm.Artifact.SHA256), prefix+".artifact.sha256", "must be a non-placeholder SHA-256 digest")
			require(nonPlaceholderSHA256(arm.RawResult.SHA256), prefix+".raw_result.sha256", "must be a non-placeholder SHA-256 digest")
		}

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
		if mode.requirePhysical {
			require(nonPlaceholderSHA256(arm.Quality.ResultSHA256), prefix+".quality.result_sha256", "must be a non-placeholder SHA-256 digest")
		}
		require(finite(arm.Quality.Score) && arm.Quality.Score >= p.QualityPolicy.MinimumScore, prefix+".quality.score", "must meet quality_policy.minimum_score")

		require(validMTPDraftDepth(arm.DraftDepth), prefix+".draft_depth", "must be between 2 and 4")
		require(arm.DraftDepth == p.SpeculativeConfig.DraftDepth, prefix+".draft_depth", "must match speculative_config.draft_depth")
		require(finite(arm.AcceptanceRate) && arm.AcceptanceRate >= 0.0 && arm.AcceptanceRate <= 1.0, prefix+".acceptance_rate", "must be in [0, 1]")

		// Effective decode tok/s > 0
		require(finitePositive(arm.EffectiveDecodeTokS), prefix+".effective_decode_tok_s", "must be positive")

		// Samples >= 20, boundary accounted
		require(len(arm.Samples) >= MinimumMTPComparisonSamples, prefix+".samples",
			fmt.Sprintf("must contain at least %d raw samples", MinimumMTPComparisonSamples))

		validateMTPSamples(prefix, arm, p, promptIDs, require)
		validateMTPMetrics(prefix, arm.Metrics, arm.Samples, require)
		derived := SummarizeMTPSamples(arm.Samples)
		require(nearlyEqual(arm.EffectiveDecodeTokS, derived.Decode.ThroughputTokS.P50), prefix+".effective_decode_tok_s", "must equal the p50 derived from raw samples")
		require(nearlyEqual(arm.AcceptanceRate, derived.Decode.AcceptanceRate.P50), prefix+".acceptance_rate", "must equal the p50 derived from raw samples")
		require(nearlyEqual(float64(arm.RollbackCount), derived.Decode.RollbackCount.P50), prefix+".rollback_count", "must equal the p50 derived from raw samples")
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

	baselineArm := armMap[p.Baseline.ArmName]
	require(strings.TrimSpace(p.Baseline.ArmName) != "", "baseline.arm_name", "is required")
	require(baselineArm != nil, "baseline.arm_name", "must name an arm in the packet")
	require(p.Baseline.ArmName != "fak-native", "baseline.arm_name", "must name a non-candidate control arm")
	require(strings.TrimSpace(p.Baseline.RunID) != "", "baseline.run_id", "is required")
	require(finitePositive(p.Baseline.EffectiveDecodeTokS), "baseline.effective_decode_tok_s", "must be positive")
	if baselineArm != nil {
		require(p.Baseline.RunID == baselineArm.RunID, "baseline.run_id", "must match the bound arm run_id")
		require(nearlyEqual(p.Baseline.EffectiveDecodeTokS, baselineArm.EffectiveDecodeTokS), "baseline.effective_decode_tok_s", "must match the bound arm measurement")
	}
	if mode.requirePhysical && fakArm != nil && finitePositive(p.Baseline.EffectiveDecodeTokS) {
		require(fakArm.EffectiveDecodeTokS >= MinMTPBaselineSpeedup*p.Baseline.EffectiveDecodeTokS,
			"fak-native.effective_decode_tok_s",
			fmt.Sprintf("must be >= %.3fx compatible baseline (got %.3fx)", MinMTPBaselineSpeedup, fakArm.EffectiveDecodeTokS/p.Baseline.EffectiveDecodeTokS))
	}
	if mode.requirePhysical && fakArm != nil {
		for name, arm := range armMap {
			if name != "fak-native" {
				require(fakArm.EffectiveDecodeTokS > arm.EffectiveDecodeTokS, "fak-native.rank", "must rank first by effective decode tok/s")
			}
		}
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

		if mode.requireVerified {
			require(p.Summary.Verified, "summary.verified", "must be true")
		} else {
			require(!p.Summary.Verified, "summary.verified", "must be false outside observed physical evidence")
		}
	}
	if fakArm != nil {
		require(nearlyEqual(p.Summary.FakNativeDecodeTokS, fakArm.EffectiveDecodeTokS), "summary.fak_native_decode_tok_s", "must match the fak-native arm measurement")
		require(nearlyEqual(p.Summary.FakNativeAcceptanceRate, fakArm.AcceptanceRate), "summary.fak_native_acceptance_rate", "must match the fak-native arm measurement")
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
		require(sample.ArtifactSHA256 == arm.Artifact.SHA256, field+".artifact_sha256", "must match arm artifact sha256")
		require(validMTPDraftDepth(sample.DraftDepth), field+".draft_depth", "must be between 2 and 4")
		require(sample.DraftDepth == arm.DraftDepth && sample.DraftDepth == packet.SpeculativeConfig.DraftDepth, field+".draft_depth", "must match arm and speculative_config draft_depth")
		require(sample.DraftProposed > 0, field+".draft_proposed", "must be positive")
		require(sample.DraftAccepted >= 0, field+".draft_accepted", "must be non-negative")
		require(sample.DraftAccepted <= sample.DraftProposed, field+".draft_accepted", "cannot exceed draft_proposed")
		require(sample.DraftAccepted <= sample.OutputTokens, field+".draft_accepted", "cannot exceed generated output tokens")
		require(sample.DraftProposed <= sample.DraftDepth*sample.OutputTokens, field+".draft_proposed", "cannot exceed draft_depth * generated output tokens")
		require(finite(sample.AcceptanceRate) && sample.AcceptanceRate >= 0 && sample.AcceptanceRate <= 1.0, field+".acceptance_rate", "must be in [0, 1]")
		if sample.DraftProposed > 0 {
			derivedAcceptance := float64(sample.DraftAccepted) / float64(sample.DraftProposed)
			require(nearlyEqual(sample.AcceptanceRate, derivedAcceptance), field+".acceptance_rate", "must equal draft_accepted / draft_proposed")
		}
		require(sample.RollbackCount >= 0, field+".rollback_count", "must be non-negative")
		require(sample.RollbackCount <= sample.DraftProposed-sample.DraftAccepted, field+".rollback_count", "cannot exceed rejected draft proposals")
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

func validMTPDraftDepth(depth int) bool {
	return depth >= 2 && depth <= 4
}

func nonPlaceholderSHA256(digest string) bool {
	digest = strings.ToLower(strings.TrimSpace(digest))
	if !validSHA256(digest) {
		return false
	}
	for _, period := range []int{1, 2, 4, 8, 16, 32} {
		if strings.Repeat(digest[:period], len(digest)/period) == digest {
			return false
		}
	}
	return true
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
		if check.field == "decode.rollback_count" || check.field == "decode.acceptance_rate" {
			valid = finite(check.got.P50) && check.got.P50 >= 0 && finite(check.got.P95) && check.got.P95 >= check.got.P50
		} else {
			valid = finitePositive(check.got.P50) && finitePositive(check.got.P95) && check.got.P95 >= check.got.P50
		}
		matched := nearlyEqual(check.got.P50, check.want.P50) && nearlyEqual(check.got.P95, check.want.P95)
		require(valid && matched, prefix+".metrics."+check.field, "must be p50/p95 values derived from raw samples")
	}
}

// NodeMacOSAMTPComparisonPacket produces the fixture comparison benchmark packet
// for Apple M3 Pro (node-macos-a) benchmarking Qwen3.8-27B with MTP speculative draft sidecar.
// It is quarantined to fixture provenance so synthesized samples cannot be published as observed.
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
		qualitySHA        string
		rawSHA            string
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
			qualitySHA:        "e217ba299012496ea73538717487cddf7cb07f28e14d7c24050673ae06742701",
			rawSHA:            "ecea792e548f68ead2ee9a50e7a6fee60c91e6e38ab0d9cf1473a0140b3e8bdb",
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
			qualitySHA:        "6395e6ee60391c39c166aa782f7ff931d5cf1479c313d0fefb3914bb42689a39",
			rawSHA:            "9378948d0d806744b280263a13139ff3ab2b3e383853d6d9c3df878be4800a5b",
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
			qualitySHA:        "f4cb4f709069143040f2d73059793a277af828903b8e8a44c1a127c7f2a1d5ed",
			rawSHA:            "0fd93f905454efb89002ed7782f7a7814c8ba7def4c3c00265af3fa49de8012c",
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
			qualitySHA:        "5efea75e76fdeea0f49c80562fe06e883e610ede2a08f645600d13cb7ddc30c6",
			rawSHA:            "2db505146bd07d99323d391e800a33063d44777c1595bbad74a6b160bd908250",
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
			EvidenceKind:    "fixture",
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
				ResultSHA256:  d.qualitySHA,
			},
			DraftDepth:          1,
			AcceptanceRate:      d.acceptanceRate,
			RollbackCount:       d.rollbackCount,
			RollbackPenaltyMS:   d.rollbackPenaltyMS,
			EffectiveDecodeTokS: d.decodeRate,
			RawResult: ComparisonRawResult{
				Path:   fmt.Sprintf("%s-raw.json", d.name),
				SHA256: d.rawSHA,
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
		Verified:                false,
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
		Baseline: MTPComparisonBaseline{
			ArmName:             "llama.cpp",
			RunID:               "node-macos-a-qwen38-llama.cpp-mtp-20260908",
			EffectiveDecodeTokS: 12.14,
		},
		QualityPolicy: qualityPolicy,
		Arms:          arms,
		Summary:       summary,
	}
}

// VerifyMTPComparisonEvidenceFiles verifies all raw samples and quality files bound to an MTP comparison packet.
func VerifyMTPComparisonEvidenceFiles(packet MTPComparisonPacket, packetPath string) error {
	base, err := filepath.Abs(filepath.Dir(packetPath))
	if err != nil {
		return fmt.Errorf("resolve packet directory: %w", err)
	}
	base, err = filepath.EvalSymlinks(base)
	if err != nil {
		return fmt.Errorf("resolve packet directory symlinks: %w", err)
	}
	for _, arm := range packet.Arms {
		raw, err := verifyMTPEvidenceFile(base, arm.RawResult.Path, arm.RawResult.SHA256)
		if err != nil {
			return fmt.Errorf("arm %s raw_result: %w", arm.Name, err)
		}
		var rawFile MTPComparisonRawSamplesFile
		if err := decodeStrictMTPJSON(raw, &rawFile); err != nil {
			return fmt.Errorf("arm %s raw_result: decode: %w", arm.Name, err)
		}
		wantRaw := MTPComparisonRawSamplesFile{
			Schema:     MTPComparisonRawSamplesSchema,
			Arm:        arm.Name,
			CampaignID: packet.CampaignID,
			RunID:      arm.RunID,
			HostID:     arm.HostID,
			StartedAt:  arm.StartedAt,
			FinishedAt: arm.FinishedAt,
			Samples:    arm.Samples,
		}
		if !reflect.DeepEqual(rawFile, wantRaw) {
			return fmt.Errorf("arm %s raw_result: content does not match packet samples", arm.Name)
		}

		quality, err := verifyMTPEvidenceFile(base, arm.Quality.ResultPath, arm.Quality.ResultSHA256)
		if err != nil {
			return fmt.Errorf("arm %s quality: %w", arm.Name, err)
		}
		var qualityFile ComparisonQualityEvidenceFile
		if err := decodeStrictMTPJSON(quality, &qualityFile); err != nil {
			return fmt.Errorf("arm %s quality: decode: %w", arm.Name, err)
		}
		wantQuality := ComparisonQualityEvidenceFile{
			Schema:          ComparisonQualityEvidenceSchema,
			Arm:             arm.Name,
			RunID:           arm.RunID,
			PolicyRef:       arm.Quality.PolicyRef,
			PolicyVersion:   arm.Quality.PolicyVersion,
			PolicySHA256:    arm.Quality.PolicySHA256,
			Passed:          arm.Quality.Passed,
			Score:           arm.Quality.Score,
			ArtifactSHA256:  arm.Artifact.SHA256,
			PromptSetSHA256: arm.PromptSetSHA256,
		}
		if qualityFile != wantQuality {
			return fmt.Errorf("arm %s quality: content does not match packet quality result", arm.Name)
		}
	}
	return nil
}

// ValidateMTPComparisonEvidence composes strict packet validation with all
// digest-bound raw and quality evidence-file checks.
func ValidateMTPComparisonEvidence(packet MTPComparisonPacket, packetPath string) error {
	if err := ValidateMTPComparisonPacket(packet); err != nil {
		return err
	}
	if err := VerifyMTPComparisonEvidenceFiles(packet, packetPath); err != nil {
		return fmt.Errorf("mtp comparison evidence invalid: %w", err)
	}
	return nil
}

func verifyMTPEvidenceFile(base, relative, wantDigest string) ([]byte, error) {
	relative = strings.TrimSpace(relative)
	if relative == "" || filepath.IsAbs(relative) {
		return nil, fmt.Errorf("path must be relative to the packet")
	}
	clean := filepath.Clean(relative)
	if clean == "." || clean == ".." || strings.HasPrefix(clean, ".."+string(filepath.Separator)) {
		return nil, fmt.Errorf("path escapes the packet directory")
	}
	resolved, err := filepath.EvalSymlinks(filepath.Join(base, clean))
	if err != nil {
		return nil, fmt.Errorf("resolve %q: %w", relative, err)
	}
	inside, err := filepath.Rel(base, resolved)
	if err != nil || inside == ".." || strings.HasPrefix(inside, ".."+string(filepath.Separator)) {
		return nil, fmt.Errorf("path escapes the packet directory")
	}
	f, err := os.Open(resolved)
	if err != nil {
		return nil, fmt.Errorf("open %q: %w", relative, err)
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("stat %q: %w", relative, err)
	}
	if info.Size() > 64<<20 {
		return nil, fmt.Errorf("%q exceeds 64 MiB evidence limit", relative)
	}
	raw, err := io.ReadAll(f)
	if err != nil {
		return nil, fmt.Errorf("read %q: %w", relative, err)
	}
	got := fmt.Sprintf("%x", sha256.Sum256(raw))
	if got != wantDigest {
		return nil, fmt.Errorf("sha256 mismatch for %q: got %s want %s", relative, got, wantDigest)
	}
	return raw, nil
}

func decodeStrictMTPJSON(raw []byte, out any) error {
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	if err := dec.Decode(out); err != nil {
		return err
	}
	var trailing json.RawMessage
	if err := dec.Decode(&trailing); err != io.EOF {
		if err == nil {
			return fmt.Errorf("multiple JSON values")
		}
		return err
	}
	return nil
}

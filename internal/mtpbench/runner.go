// Package mtpbench runs a fail-closed, two-arm physical benchmark of fak's
// ordinary native Metal decode and its Qwen3.8 MTP depth-four candidate.
package mtpbench

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"slices"
	"sort"
	"strings"
	"time"
)

const (
	Schema            = "fak.native-qwen38-mtp-abba/1"
	BaselineArm       = "baseline"
	CandidateArm      = "mtp-depth4"
	MinimumSamples    = 20
	MaximumCV         = 0.05
	MinimumSpeedup    = 2.0
	ExpectedPanelPath = "fak-native/metal/qwen3.8-mtp-p4-target-verify-v1"
)

// Config fixes the complete comparison envelope. PromptIDs are tokenizer output
// supplied by the operator; TokenizerPath binds those IDs to the exact tokenizer
// artifact used to produce them.
type Config struct {
	ArtifactPath                   string
	ExpectedArtifactShardSetSHA256 string
	TokenizerPath                  string
	ExpectedTokenizerSHA256        string
	PromptIDs                      []int
	GeneratedTokens                int
	SamplesPerArm                  int
	RequiredSpeedup                float64
	MaximumCoefficientVariation    float64
	BaselineIndexPath              string
	ExpectedBaselineIndexSHA256    string
	BaselineIndex                  *BaselineIndex
}

func (c Config) normalized() Config {
	c.PromptIDs = slices.Clone(c.PromptIDs)
	if c.GeneratedTokens == 0 {
		c.GeneratedTokens = 64
	}
	if c.SamplesPerArm == 0 {
		c.SamplesPerArm = MinimumSamples
	}
	if c.RequiredSpeedup == 0 {
		c.RequiredSpeedup = MinimumSpeedup
	}
	if c.MaximumCoefficientVariation == 0 {
		c.MaximumCoefficientVariation = MaximumCV
	}
	return c
}

func (c Config) validate() error {
	if strings.TrimSpace(c.ArtifactPath) == "" || strings.TrimSpace(c.TokenizerPath) == "" {
		return errors.New("mtpbench: artifact and tokenizer paths are required")
	}
	if !validSHA256(c.ExpectedArtifactShardSetSHA256) || !validSHA256(c.ExpectedTokenizerSHA256) {
		return errors.New("mtpbench: expected artifact shard-set and tokenizer SHA-256 digests are required")
	}
	if len(c.PromptIDs) == 0 || c.GeneratedTokens < 2 {
		return errors.New("mtpbench: a non-empty prompt and at least two generated tokens are required")
	}
	for _, id := range c.PromptIDs {
		if id < 0 {
			return errors.New("mtpbench: prompt token IDs must be non-negative")
		}
	}
	if c.SamplesPerArm < MinimumSamples || c.SamplesPerArm%2 != 0 {
		return fmt.Errorf("mtpbench: samples per arm must be even and >= %d", MinimumSamples)
	}
	if !finitePositive(c.RequiredSpeedup) || c.RequiredSpeedup < MinimumSpeedup {
		return fmt.Errorf("mtpbench: required speedup must be >= %.1f", MinimumSpeedup)
	}
	if !finitePositive(c.MaximumCoefficientVariation) || c.MaximumCoefficientVariation > MaximumCV {
		return fmt.Errorf("mtpbench: maximum CV must be in (0, %.2f]", MaximumCV)
	}
	if !validSHA256(c.ExpectedBaselineIndexSHA256) || (c.BaselineIndex == nil && strings.TrimSpace(c.BaselineIndexPath) == "") {
		return errors.New("mtpbench: pinned external baseline index is required")
	}
	return nil
}

// Preflight is derived from the GGUF header and filesystem metadata only. It
// never reads tensor payloads or constructs a model.
type Preflight struct {
	Schema             string `json:"schema"`
	ArtifactPath       string `json:"artifact_path"`
	ArtifactFileBytes  int64  `json:"artifact_file_bytes"`
	HeaderPayloadBytes int64  `json:"header_payload_bytes"`
	EstimatedPeakBytes int64  `json:"estimated_peak_bytes"`
	TensorCount        int    `json:"tensor_count"`
	ModelType          string `json:"model_type"`
	ModelName          string `json:"model_name"`
	QuantRecipe        string `json:"quant_recipe"`
	Layers             int    `json:"layers"`
	MTPLayers          int    `json:"mtp_layers"`
	LoadPayload        bool   `json:"load_payload"`
}

type MemorySnapshot struct {
	CurrentRSSBytes uint64 `json:"current_rss_bytes"`
	SwapUsedBytes   uint64 `json:"swap_used_bytes"`
	RSSAvailable    bool   `json:"rss_available"`
	SwapAvailable   bool   `json:"swap_available"`
}

type HardwareIdentity struct {
	Hostname         string `json:"hostname"`
	MetalDevice      string `json:"metal_device"`
	OSVersion        string `json:"os_version"`
	OSBuild          string `json:"os_build"`
	Arch             string `json:"arch"`
	CPUs             uint64 `json:"cpus"`
	PhysicalRAMBytes uint64 `json:"physical_ram_bytes"`
	MetalMemoryBytes uint64 `json:"metal_memory_bytes"`
}

type ExternalBaseline struct {
	Engine       string  `json:"engine"`
	P50TokensSec float64 `json:"p50_tokens_per_second"`
}

type BaselineIndex struct {
	Schema                 string             `json:"schema"`
	CapturedAt             time.Time          `json:"captured_at"`
	ArtifactShardSetSHA256 string             `json:"artifact_shard_set_sha256"`
	TokenizerSHA256        string             `json:"tokenizer_sha256"`
	WorkloadSHA256         string             `json:"workload_sha256"`
	EnvelopeSHA256         string             `json:"envelope_sha256"`
	HostSHA256             string             `json:"host_sha256"`
	Entries                []ExternalBaseline `json:"entries"`
}

type CleanupReceipt struct {
	RetainMTPRestored  bool `json:"retain_mtp_restored"`
	ModelWeightsClosed bool `json:"model_weights_closed"`
	Q6LiveBefore       int  `json:"q6_live_before"`
	Q6LiveAfter        int  `json:"q6_live_after"`
}

// NativeReceipt is the exact aggregate of every candidate StepRound in a sample.
type NativeReceipt struct {
	Engine                       string   `json:"engine"`
	Rounds                       int      `json:"rounds"`
	Paths                        []string `json:"paths"`
	OneOperationRounds           int      `json:"one_operation_rounds"`
	TargetVerificationOperations int      `json:"target_verification_operations"`
	TargetDecodeSteps            int      `json:"target_decode_steps"`
	CommandBuffers               int      `json:"command_buffers"`
	Q6KDownProjectionOperations  int      `json:"q6k_down_projection_operations"`
	Q6KHeadOperations            int      `json:"q6k_head_operations"`
	FallbackCount                int      `json:"fallback_count"`
	Proposed                     int      `json:"proposed"`
	Accepted                     int      `json:"accepted"`
	Rollbacks                    int      `json:"rollbacks"`
	StateSHA256                  []string `json:"state_sha256"`
	TransactionSHA256            []string `json:"transaction_sha256"`
}

type Observation struct {
	Tokens  []int          `json:"tokens"`
	Elapsed time.Duration  `json:"-"`
	Receipt *NativeReceipt `json:"receipt,omitempty"`
	Metal   *MetalProfile  `json:"metal"`
}

type MetalProfile struct {
	Available            bool   `json:"available"`
	TargetFallbacks      int    `json:"target_cpu_fallbacks"`
	DraftObserved        bool   `json:"draft_observed"`
	DraftFallbacks       int    `json:"draft_cpu_fallbacks"`
	Q4Operations         int    `json:"q4_operations"`
	Q6Operations         int    `json:"q6_operations"`
	Q8Operations         int    `json:"q8_operations"`
	ExecutionSHA256      string `json:"execution_sha256"`
	FallbackSHA256       string `json:"fallback_sha256"`
	DraftQ4Operations    int    `json:"draft_q4_operations"`
	DraftQ6Operations    int    `json:"draft_q6_operations"`
	DraftQ8Operations    int    `json:"draft_q8_operations"`
	DraftExecutionSHA256 string `json:"draft_execution_sha256,omitempty"`
	DraftFallbackSHA256  string `json:"draft_fallback_sha256,omitempty"`
}

type Sample struct {
	Sequence        int            `json:"sequence"`
	Arm             string         `json:"arm"`
	Nanoseconds     int64          `json:"nanoseconds"`
	TokensPerSecond float64        `json:"tokens_per_second"`
	Tokens          []int          `json:"tokens"`
	Receipt         *NativeReceipt `json:"receipt,omitempty"`
	Metal           *MetalProfile  `json:"metal"`
	MemoryAfter     MemorySnapshot `json:"memory_after"`
}

type ArmSummary struct {
	Arm               string  `json:"arm"`
	Samples           int     `json:"samples"`
	P50Nanoseconds    int64   `json:"p50_nanoseconds"`
	P50TokensPerSec   float64 `json:"p50_tokens_per_second"`
	CoefficientVar    float64 `json:"coefficient_of_variation"`
	Rank              int     `json:"rank"`
	MetalQ4Operations int     `json:"metal_q4_operations"`
	MetalQ6Operations int     `json:"metal_q6_operations"`
	MetalQ8Operations int     `json:"metal_q8_operations"`
	MetalCPUFallbacks int     `json:"metal_cpu_fallbacks"`
}

type Identity struct {
	ArtifactShardSetSHA256 string           `json:"artifact_shard_set_sha256"`
	TokenizerSHA256        string           `json:"tokenizer_sha256"`
	RuntimeSHA256          string           `json:"runtime_sha256"`
	WorkloadSHA256         string           `json:"workload_sha256"`
	EnvelopeSHA256         string           `json:"envelope_sha256"`
	HostSHA256             string           `json:"host_sha256"`
	Host                   string           `json:"host"`
	Hardware               HardwareIdentity `json:"hardware"`
	ArtifactShards         []ArtifactShard  `json:"artifact_shards"`
}

type ArtifactShard struct {
	Path          string `json:"path"`
	Size          int64  `json:"size"`
	RawFileSHA256 string `json:"raw_file_sha256"`
}

type Report struct {
	Schema              string             `json:"schema"`
	Status              string             `json:"status"`
	Failure             string             `json:"failure,omitempty"`
	Identity            Identity           `json:"identity"`
	PromptIDs           []int              `json:"prompt_ids"`
	GeneratedTokens     int                `json:"generated_tokens"`
	Schedule            []string           `json:"schedule"`
	Samples             []Sample           `json:"samples"`
	Arms                []ArmSummary       `json:"arms"`
	SpeedupRatio        float64            `json:"speedup_ratio"`
	TokenVectorsEqual   bool               `json:"token_vectors_equal"`
	MemoryBefore        MemorySnapshot     `json:"memory_before"`
	MemoryAfterPrepare  MemorySnapshot     `json:"memory_after_prepare"`
	MemoryPeak          MemorySnapshot     `json:"memory_peak"`
	MemoryAfter         MemorySnapshot     `json:"memory_after"`
	Cleanup             CleanupReceipt     `json:"cleanup"`
	BaselineIndexSHA256 string             `json:"baseline_index_sha256"`
	ExternalBaselines   []ExternalBaseline `json:"external_baselines"`
}

// Prepared owns the single loaded model and creates a fresh session per call.
type Prepared interface {
	Warm(ctx context.Context, arm string, prompt []int, generated int) error
	Run(ctx context.Context, arm string, prompt []int, generated int) (Observation, error)
	Memory() (MemorySnapshot, error)
	Identity() Identity
	Close() (CleanupReceipt, error)
}

type Executor interface {
	Preflight(context.Context, Config) (Preflight, error)
	Memory() (MemorySnapshot, error)
	Prepare(context.Context, Config) (Prepared, error)
}

type Runner struct {
	Executor Executor
	Now      func() time.Time
}

func (r Runner) Preflight(ctx context.Context, cfg Config) (Preflight, error) {
	cfg = cfg.normalized()
	if strings.TrimSpace(cfg.ArtifactPath) == "" {
		return Preflight{}, errors.New("mtpbench: artifact path is required")
	}
	if r.Executor == nil {
		return Preflight{}, errors.New("mtpbench: nil executor")
	}
	return r.Executor.Preflight(ctx, cfg)
}

func (r Runner) Run(ctx context.Context, cfg Config) (report Report, err error) {
	cfg = cfg.normalized()
	report = Report{Schema: Schema, Status: "FAIL", PromptIDs: slices.Clone(cfg.PromptIDs), GeneratedTokens: cfg.GeneratedTokens}
	if err = cfg.validate(); err != nil {
		report.Failure = err.Error()
		return report, err
	}
	if r.Executor == nil {
		err = errors.New("mtpbench: nil executor")
		report.Failure = err.Error()
		return report, err
	}
	report.MemoryBefore, err = r.Executor.Memory()
	if err != nil {
		report.Failure = fmt.Sprintf("mtpbench: pre-prepare memory observation: %v", err)
		return report, errors.New(report.Failure)
	}
	if memoryErr := validateMemory(report.MemoryBefore); memoryErr != nil {
		report.Failure = memoryErr.Error()
		return report, memoryErr
	}
	prepared, prepErr := r.Executor.Prepare(ctx, cfg)
	if prepErr != nil {
		report.Failure = prepErr.Error()
		return report, prepErr
	}
	defer func() {
		cleanup, closeErr := prepared.Close()
		report.Cleanup = cleanup
		gateErr := cleanupError(cleanup, closeErr)
		if gateErr != nil {
			err = errors.Join(err, gateErr)
			report.Status = "FAIL"
			report.Failure = err.Error()
		}
	}()
	report.MemoryAfterPrepare, err = prepared.Memory()
	if err != nil {
		report.Failure = fmt.Sprintf("mtpbench: post-prepare memory observation: %v", err)
		return report, errors.New(report.Failure)
	}
	if memoryErr := validateMemory(report.MemoryAfterPrepare); memoryErr != nil {
		report.Failure = memoryErr.Error()
		return report, memoryErr
	}
	report.MemoryPeak = maxMemory(report.MemoryBefore, report.MemoryAfterPrepare)
	if report.MemoryAfterPrepare.SwapUsedBytes > report.MemoryBefore.SwapUsedBytes {
		err = errors.New("mtpbench: swap grew during model preparation")
		report.Failure = err.Error()
		return report, err
	}
	report.Identity = prepared.Identity()
	if identityErr := validateIdentity(report.Identity); identityErr != nil {
		report.Failure = identityErr.Error()
		return report, identityErr
	}
	now := time.Now()
	if r.Now != nil {
		now = r.Now()
	}
	index, indexHash, indexErr := loadBaselineIndex(cfg, report.Identity, now)
	if indexErr != nil {
		report.Failure = indexErr.Error()
		return report, indexErr
	}
	report.BaselineIndexSHA256, report.ExternalBaselines = indexHash, slices.Clone(index.Entries)

	for _, arm := range []string{BaselineArm, CandidateArm} {
		if warmErr := prepared.Warm(ctx, arm, cfg.PromptIDs, cfg.GeneratedTokens); warmErr != nil {
			report.Failure = fmt.Sprintf("mtpbench: %s warmup: %v", arm, warmErr)
			return report, errors.New(report.Failure)
		}
	}
	report.Schedule = abbaSchedule(cfg.SamplesPerArm)
	for i, arm := range report.Schedule {
		if ctxErr := ctx.Err(); ctxErr != nil {
			report.Failure = ctxErr.Error()
			return report, ctxErr
		}
		obs, runErr := prepared.Run(ctx, arm, cfg.PromptIDs, cfg.GeneratedTokens)
		if runErr != nil {
			report.Failure = fmt.Sprintf("mtpbench: sample %d %s: %v", i+1, arm, runErr)
			return report, errors.New(report.Failure)
		}
		if obs.Elapsed <= 0 || len(obs.Tokens) != cfg.GeneratedTokens {
			report.Failure = fmt.Sprintf("mtpbench: sample %d %s has invalid timing/token count", i+1, arm)
			return report, errors.New(report.Failure)
		}
		if !finitePositive(obs.Elapsed.Seconds()) {
			report.Failure = fmt.Sprintf("mtpbench: sample %d %s has non-finite timing", i+1, arm)
			return report, errors.New(report.Failure)
		}
		if arm == CandidateArm {
			if receiptErr := validateReceipt(obs.Receipt); receiptErr != nil {
				report.Failure = fmt.Sprintf("mtpbench: sample %d candidate receipt: %v", i+1, receiptErr)
				return report, errors.New(report.Failure)
			}
		} else if obs.Receipt != nil {
			report.Failure = fmt.Sprintf("mtpbench: sample %d baseline unexpectedly supplied MTP receipt", i+1)
			return report, errors.New(report.Failure)
		}
		if metalErr := validateMetalProfile(obs.Metal, arm == CandidateArm); metalErr != nil {
			report.Failure = fmt.Sprintf("mtpbench: sample %d %s Metal profile: %v", i+1, arm, metalErr)
			return report, errors.New(report.Failure)
		}
		memory, memoryErr := prepared.Memory()
		if memoryErr != nil {
			report.Failure = fmt.Sprintf("mtpbench: sample %d memory: %v", i+1, memoryErr)
			return report, errors.New(report.Failure)
		}
		if gateErr := validateMemory(memory); gateErr != nil {
			report.Failure = gateErr.Error()
			return report, gateErr
		}
		report.MemoryPeak = maxMemory(report.MemoryPeak, memory)
		report.Samples = append(report.Samples, Sample{
			Sequence: i + 1, Arm: arm, Nanoseconds: obs.Elapsed.Nanoseconds(),
			TokensPerSecond: float64(cfg.GeneratedTokens) / obs.Elapsed.Seconds(),
			Tokens:          slices.Clone(obs.Tokens), Receipt: cloneReceipt(obs.Receipt), Metal: obs.Metal, MemoryAfter: memory,
		})
	}

	cleanup, closeErr := prepared.Close()
	report.Cleanup = cleanup
	report.MemoryAfter, err = prepared.Memory()
	if err != nil {
		report.Failure = fmt.Sprintf("mtpbench: final memory observation: %v", err)
		return report, errors.New(report.Failure)
	}
	if memoryErr := validateMemory(report.MemoryAfter); memoryErr != nil {
		report.Failure = memoryErr.Error()
		return report, memoryErr
	}
	if report.MemoryAfter.CurrentRSSBytes > report.MemoryAfterPrepare.CurrentRSSBytes || report.MemoryAfter.SwapUsedBytes > report.MemoryBefore.SwapUsedBytes {
		err = errors.New("mtpbench: RSS did not restore to the post-prepare envelope or swap to the pre-prepare envelope")
		report.Failure = err.Error()
		return report, err
	}
	report.MemoryPeak = maxMemory(report.MemoryPeak, report.MemoryAfter)
	if cleanupErr := cleanupError(cleanup, closeErr); cleanupErr != nil {
		err = cleanupErr
		report.Failure = err.Error()
		return report, err
	}

	report.Arms, report.SpeedupRatio, report.TokenVectorsEqual, err = summarizeAndGate(report.Samples, cfg)
	if err != nil {
		report.Failure = err.Error()
		return report, err
	}
	for _, external := range index.Entries {
		if report.Arms[1].P50TokensPerSec <= external.P50TokensSec {
			err = fmt.Errorf("mtpbench: candidate %.6f tok/s does not exceed indexed %s %.6f tok/s", report.Arms[1].P50TokensPerSec, external.Engine, external.P50TokensSec)
			report.Failure = err.Error()
			return report, err
		}
	}
	report.Status = "PASS"
	return report, nil
}

func validateMemory(m MemorySnapshot) error {
	if !m.RSSAvailable || !m.SwapAvailable || m.CurrentRSSBytes == 0 {
		return errors.New("mtpbench: current RSS/swap observation unavailable")
	}
	return nil
}

func validateMetalProfile(p *MetalProfile, requireDraft bool) error {
	// The target artifact has observable native Q4_K and Q6_K operations. The
	// event taxonomy has no standalone Q5_K event, and target Q8_0 is optional:
	// this artifact's Q8_0 fc tensors belong to the MTP draft role.
	if p == nil || !p.Available || !validSHA256(p.ExecutionSHA256) || !validSHA256(p.FallbackSHA256) || p.TargetFallbacks != 0 || p.DraftFallbacks != 0 || p.Q4Operations <= 0 || p.Q6Operations <= 0 || p.Q8Operations < 0 || (requireDraft && (!p.DraftObserved || !validSHA256(p.DraftExecutionSHA256) || !validSHA256(p.DraftFallbackSHA256) || p.DraftQ4Operations <= 0 || p.DraftQ6Operations <= 0 || p.DraftQ8Operations <= 0)) {
		return errors.New("role-required Metal execution/fallback profile incomplete")
	}
	if !requireDraft && (p.DraftObserved || p.DraftFallbacks != 0 || p.DraftQ4Operations != 0 || p.DraftQ6Operations != 0 || p.DraftQ8Operations != 0 || p.DraftExecutionSHA256 != "" || p.DraftFallbackSHA256 != "") {
		return errors.New("baseline Metal profile contains draft contamination")
	}
	return nil
}

func cleanupError(c CleanupReceipt, closeErr error) error {
	if closeErr != nil {
		return fmt.Errorf("mtpbench: close model: %w", closeErr)
	}
	if !c.RetainMTPRestored || !c.ModelWeightsClosed || c.Q6LiveAfter != c.Q6LiveBefore {
		return fmt.Errorf("mtpbench: cleanup gate failed: %+v", c)
	}
	return nil
}

func loadBaselineIndex(cfg Config, id Identity, now time.Time) (BaselineIndex, string, error) {
	var raw []byte
	var err error
	if cfg.BaselineIndex != nil {
		raw, err = json.Marshal(cfg.BaselineIndex)
	} else {
		raw, err = os.ReadFile(cfg.BaselineIndexPath)
	}
	if err != nil {
		return BaselineIndex{}, "", fmt.Errorf("mtpbench: baseline index: %w", err)
	}
	sum := sha256.Sum256(raw)
	digest := hex.EncodeToString(sum[:])
	if digest != cfg.ExpectedBaselineIndexSHA256 {
		return BaselineIndex{}, "", errors.New("mtpbench: baseline index digest mismatch")
	}
	var index BaselineIndex
	if err := json.Unmarshal(raw, &index); err != nil {
		return index, "", err
	}
	if index.Schema != Schema+".baseline-index" || index.CapturedAt.IsZero() || now.Before(index.CapturedAt) || now.Sub(index.CapturedAt) > 30*24*time.Hour ||
		index.ArtifactShardSetSHA256 != id.ArtifactShardSetSHA256 || index.TokenizerSHA256 != id.TokenizerSHA256 || index.WorkloadSHA256 != id.WorkloadSHA256 || index.EnvelopeSHA256 != id.EnvelopeSHA256 || index.HostSHA256 != id.HostSHA256 || len(index.Entries) == 0 {
		return index, "", errors.New("mtpbench: incompatible, stale, or empty baseline index")
	}
	seen := map[string]bool{}
	families := map[string]bool{}
	for _, e := range index.Entries {
		if strings.TrimSpace(e.Engine) == "" || seen[e.Engine] || !finitePositive(e.P50TokensSec) {
			return index, "", errors.New("mtpbench: invalid baseline index entry")
		}
		seen[e.Engine] = true
		lower := strings.ToLower(e.Engine)
		for _, family := range []string{"fak", "llama", "mlx"} {
			if strings.Contains(lower, family) {
				families[family] = true
			}
		}
	}
	if len(families) != 3 {
		return index, "", errors.New("mtpbench: baseline index must contain recent fak, llama, and MLX engines")
	}
	return index, digest, nil
}

func abbaSchedule(samplesPerArm int) []string {
	out := make([]string, 0, 2*samplesPerArm)
	for i := 0; i < samplesPerArm/2; i++ {
		out = append(out, CandidateArm, BaselineArm, BaselineArm, CandidateArm)
	}
	return out
}

func validateReceipt(r *NativeReceipt) error {
	if r == nil || r.Rounds == 0 {
		return errors.New("absent target verification receipt")
	}
	if r.Engine != "fak-native" {
		return fmt.Errorf("unexpected target verification engine %q", r.Engine)
	}
	if len(r.Paths) != r.Rounds || r.OneOperationRounds != r.Rounds || r.TargetVerificationOperations != r.Rounds || r.TargetDecodeSteps != 0 {
		return fmt.Errorf("not one target operation per round: %+v", *r)
	}
	for _, path := range r.Paths {
		if path != ExpectedPanelPath {
			return fmt.Errorf("unexpected verification path %q", path)
		}
	}
	if r.CommandBuffers != r.Rounds || r.Q6KDownProjectionOperations <= 0 || r.Q6KHeadOperations != r.Rounds || r.FallbackCount != 0 {
		return fmt.Errorf("native/Q6/fallback counters failed: %+v", *r)
	}
	if r.Proposed < 0 || r.Accepted < 0 || r.Accepted > r.Proposed || r.Rollbacks < 0 {
		return errors.New("invalid MTP acceptance counters")
	}
	if len(r.StateSHA256) != r.Rounds || len(r.TransactionSHA256) != r.Rounds {
		return errors.New("missing per-round state/transaction binding")
	}
	for _, digest := range append(slices.Clone(r.StateSHA256), r.TransactionSHA256...) {
		if !validSHA256(digest) {
			return errors.New("invalid per-round state/transaction digest")
		}
	}
	return nil
}

func summarizeAndGate(samples []Sample, cfg Config) ([]ArmSummary, float64, bool, error) {
	byArm := map[string][]Sample{BaselineArm: {}, CandidateArm: {}}
	for _, sample := range samples {
		if _, ok := byArm[sample.Arm]; !ok {
			return nil, 0, false, fmt.Errorf("mtpbench: unknown arm %q", sample.Arm)
		}
		byArm[sample.Arm] = append(byArm[sample.Arm], sample)
	}
	var canonical []int
	equal := true
	for _, sample := range samples {
		if canonical == nil {
			canonical = sample.Tokens
		} else if !slices.Equal(canonical, sample.Tokens) {
			equal = false
		}
	}
	if !equal {
		return nil, 0, false, errors.New("mtpbench: generated token vectors differ")
	}
	summaries := make([]ArmSummary, 0, 2)
	for _, arm := range []string{BaselineArm, CandidateArm} {
		ss := byArm[arm]
		if len(ss) != cfg.SamplesPerArm {
			return nil, 0, equal, fmt.Errorf("mtpbench: %s has %d samples, want %d", arm, len(ss), cfg.SamplesPerArm)
		}
		rates := make([]float64, len(ss))
		durations := make([]int64, len(ss))
		for i := range ss {
			rates[i], durations[i] = ss[i].TokensPerSecond, ss[i].Nanoseconds
			if !finitePositive(rates[i]) || durations[i] <= 0 {
				return nil, 0, equal, errors.New("mtpbench: non-finite sample measurement")
			}
			if ss[i].Metal == nil {
				return nil, 0, equal, errors.New("mtpbench: missing Metal sample profile")
			}
		}
		summary := ArmSummary{Arm: arm, Samples: len(ss), P50TokensPerSec: medianFloat(rates), P50Nanoseconds: medianInt64(durations), CoefficientVar: coefficientVariation(rates)}
		for _, sample := range ss {
			summary.MetalQ4Operations += sample.Metal.Q4Operations + sample.Metal.DraftQ4Operations
			summary.MetalQ6Operations += sample.Metal.Q6Operations + sample.Metal.DraftQ6Operations
			summary.MetalQ8Operations += sample.Metal.Q8Operations + sample.Metal.DraftQ8Operations
			summary.MetalCPUFallbacks += sample.Metal.TargetFallbacks + sample.Metal.DraftFallbacks
		}
		if summary.MetalCPUFallbacks != 0 || summary.MetalQ4Operations <= 0 || summary.MetalQ6Operations <= 0 || (arm == CandidateArm && summary.MetalQ8Operations <= 0) {
			return nil, 0, equal, errors.New("mtpbench: aggregate Metal mechanism/fallback gate failed")
		}
		if summary.CoefficientVar > cfg.MaximumCoefficientVariation {
			return nil, 0, equal, fmt.Errorf("mtpbench: %s CV %.6f exceeds %.6f", arm, summary.CoefficientVar, cfg.MaximumCoefficientVariation)
		}
		summaries = append(summaries, summary)
	}
	baseline, candidate := summaries[0].P50TokensPerSec, summaries[1].P50TokensPerSec
	if baseline <= 0 {
		return nil, 0, equal, errors.New("mtpbench: baseline p50 throughput is not positive")
	}
	ratio := candidate / baseline
	if candidate > baseline {
		summaries[0].Rank, summaries[1].Rank = 2, 1
	} else {
		summaries[0].Rank, summaries[1].Rank = 1, 2
	}
	if summaries[1].Rank != 1 {
		return summaries, ratio, equal, errors.New("mtpbench: candidate is not highest-performing arm")
	}
	if ratio < cfg.RequiredSpeedup {
		return summaries, ratio, equal, fmt.Errorf("mtpbench: candidate speedup %.6fx is below %.6fx", ratio, cfg.RequiredSpeedup)
	}
	return summaries, ratio, equal, nil
}

func validateIdentity(id Identity) error {
	if id.Host == "" || !validSHA256(id.ArtifactShardSetSHA256) || !validSHA256(id.TokenizerSHA256) ||
		!validSHA256(id.RuntimeSHA256) || !validSHA256(id.WorkloadSHA256) ||
		!validSHA256(id.EnvelopeSHA256) || !validSHA256(id.HostSHA256) || len(id.ArtifactShards) == 0 ||
		id.Hardware.MetalDevice == "" || id.Hardware.OSVersion == "" || id.Hardware.OSBuild == "" || id.Hardware.Arch == "" || id.Hardware.CPUs == 0 || id.Hardware.PhysicalRAMBytes == 0 || id.Hardware.MetalMemoryBytes == 0 {
		return errors.New("mtpbench: incomplete physical identity/digest gate")
	}
	for _, shard := range id.ArtifactShards {
		if shard.Path == "" || shard.Size <= 0 || !validSHA256(shard.RawFileSHA256) {
			return errors.New("mtpbench: invalid artifact shard identity")
		}
	}
	setHash, err := digestJSON(id.ArtifactShards)
	if err != nil || setHash != id.ArtifactShardSetSHA256 {
		return errors.New("mtpbench: artifact shard-set identity mismatch")
	}
	hostHash, err := digestJSON(id.Hardware)
	if err != nil || hostHash != id.HostSHA256 {
		return errors.New("mtpbench: hardware identity digest mismatch")
	}
	hostRaw, err := json.Marshal(id.Hardware)
	if err != nil || string(hostRaw) != id.Host {
		return errors.New("mtpbench: hardware identity rendering mismatch")
	}
	return nil
}

func medianFloat(values []float64) float64 {
	v := slices.Clone(values)
	sort.Float64s(v)
	if len(v)%2 == 1 {
		return v[len(v)/2]
	}
	return (v[len(v)/2-1] + v[len(v)/2]) / 2
}

func medianInt64(values []int64) int64 {
	v := slices.Clone(values)
	sort.Slice(v, func(i, j int) bool { return v[i] < v[j] })
	if len(v)%2 == 1 {
		return v[len(v)/2]
	}
	return v[len(v)/2-1]/2 + v[len(v)/2]/2 + (v[len(v)/2-1]%2+v[len(v)/2]%2)/2
}

func coefficientVariation(values []float64) float64 {
	var sum float64
	for _, v := range values {
		sum += v
	}
	mean := sum / float64(len(values))
	if mean <= 0 {
		return math.Inf(1)
	}
	var sq float64
	for _, v := range values {
		d := v - mean
		sq += d * d
	}
	return math.Sqrt(sq/float64(len(values))) / mean
}

func digestJSON(v any) (string, error) {
	raw, err := json.Marshal(v)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(raw)
	return hex.EncodeToString(sum[:]), nil
}

func finitePositive(v float64) bool { return v > 0 && !math.IsNaN(v) && !math.IsInf(v, 0) }

func validSHA256(v string) bool {
	if len(v) != sha256.Size*2 {
		return false
	}
	_, err := hex.DecodeString(v)
	return err == nil && strings.ToLower(v) == v
}

func cloneReceipt(in *NativeReceipt) *NativeReceipt {
	if in == nil {
		return nil
	}
	out := *in
	out.Paths = slices.Clone(in.Paths)
	out.StateSHA256 = slices.Clone(in.StateSHA256)
	out.TransactionSHA256 = slices.Clone(in.TransactionSHA256)
	return &out
}

func maxMemory(a, b MemorySnapshot) MemorySnapshot {
	if b.CurrentRSSBytes > a.CurrentRSSBytes {
		a.CurrentRSSBytes = b.CurrentRSSBytes
	}
	if b.SwapUsedBytes > a.SwapUsedBytes {
		a.SwapUsedBytes = b.SwapUsedBytes
	}
	a.RSSAvailable = a.RSSAvailable && b.RSSAvailable
	a.SwapAvailable = a.SwapAvailable && b.SwapAvailable
	return a
}

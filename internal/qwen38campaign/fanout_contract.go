package qwen38campaign

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strings"
)

// HardwareInfo records targeted architectural characteristics for AMD Strix Halo APU.
type HardwareInfo struct {
	Architecture                 string  `json:"architecture"`
	Device                       string  `json:"device"`
	DeviceName                   string  `json:"device_name"`
	MALLCacheSizeMB              int     `json:"mall_cache_size_mb"`
	DRAMBandwidthSustainableGBps float64 `json:"dram_bandwidth_sustainable_gbps"`
	DRAMBandwidthTheoreticalGBps float64 `json:"dram_bandwidth_theoretical_gbps"`
	MALLBandwidthSustainedGBps   float64 `json:"mall_bandwidth_sustained_gbps"`
}

// DefaultHardwareInfo returns standard hardware descriptors for AMD Strix Halo gfx1151.
func DefaultHardwareInfo() HardwareInfo {
	return HardwareInfo{
		Architecture:                 DefaultArchStrixHalo,
		Device:                       DefaultDeviceStrixHalo,
		DeviceName:                   DefaultDeviceNameStrixHalo,
		MALLCacheSizeMB:              StrixHaloMALLCacheSizeMB,
		DRAMBandwidthSustainableGBps: StrixHaloSustainableDRAMBandwidthGBps,
		DRAMBandwidthTheoreticalGBps: StrixHaloTheoreticalPeakDRAMBandwidthGBps,
		MALLBandwidthSustainedGBps:   StrixHaloSustainedMALLBandwidthGBps,
	}
}

// FanoutConfig specifies benchmark parameters for a multi-agent fan-out campaign.
type FanoutConfig struct {
	Scenario                   string  `json:"scenario"`
	Concurrency                int     `json:"concurrency"` // Batch B in {1, 2, 4, 8}
	Runs                       int     `json:"runs"`        // >= 5 repetitions
	PrefixTokens               int     `json:"prefix_tokens"`
	GeneratedTokensPerSubagent int     `json:"generated_tokens_per_subagent"`
	Simulated                  bool    `json:"simulated"`
	ParityThreshold            float64 `json:"parity_threshold"`
	Seed                       int64   `json:"seed,omitempty"`
}

// Validate ensures the benchmark configuration satisfies the requirements of issue #11620.
func (c *FanoutConfig) Validate() error {
	switch c.Scenario {
	case ScenarioCold, ScenarioWarmSamePrefix, ScenarioSharedPrefixForked:
		// valid
	default:
		return fmt.Errorf("qwen38campaign: invalid scenario %q (must be %q, %q, or %q)",
			c.Scenario, ScenarioCold, ScenarioWarmSamePrefix, ScenarioSharedPrefixForked)
	}

	switch c.Concurrency {
	case 1, 2, 4, 8:
		// valid
	default:
		return fmt.Errorf("qwen38campaign: invalid concurrency %d (must be 1, 2, 4, or 8)", c.Concurrency)
	}

	if c.Runs < DefaultMinRuns {
		return fmt.Errorf("qwen38campaign: runs count %d is less than required minimum %d", c.Runs, DefaultMinRuns)
	}

	if c.PrefixTokens < 0 {
		return fmt.Errorf("qwen38campaign: prefix tokens cannot be negative: %d", c.PrefixTokens)
	}
	if c.GeneratedTokensPerSubagent <= 0 {
		return fmt.Errorf("qwen38campaign: generated tokens per subagent must be positive: %d", c.GeneratedTokensPerSubagent)
	}

	if c.ParityThreshold <= 0 {
		c.ParityThreshold = DefaultLogitCosineParityThreshold
	}

	return nil
}

// ExecutionIdentity binds source commit, source archive, binary, model artifact, and token packet hashes.
type ExecutionIdentity struct {
	SourceCommit        string `json:"source_commit"`
	SourceArchiveSHA256 string `json:"source_archive_sha256,omitempty"`
	BinarySHA256        string `json:"binary_sha256"`
	ModelGGUFSHA256     string `json:"model_gguf_sha256"`
	TokenPacketSHA256   string `json:"token_packet_sha256"`
}

// BackendExecutionEvidence is the lossless receipt form of one backend-owned
// execution observation. It deliberately keeps transfer bytes separate from
// DRAM traffic and reports closing memory gauges rather than peak memory.
type BackendExecutionEvidence struct {
	Backend                 string  `json:"backend"`
	Device                  string  `json:"device"`
	Driver                  string  `json:"driver"`
	Runtime                 string  `json:"runtime"`
	ComputeDispatches       uint64  `json:"compute_dispatches"`
	Q4KMatmulDispatches     uint64  `json:"q4_k_matmul_dispatches"`
	OtherDispatches         uint64  `json:"other_dispatches"`
	DispatchSubmits         uint64  `json:"dispatch_submits"`
	H2DBytes                uint64  `json:"h2d_bytes"`
	D2HBytes                uint64  `json:"d2h_bytes"`
	D2DCopies               uint64  `json:"d2d_copies"`
	Q4KStageCalls           uint64  `json:"q4_k_stage_calls"`
	Q4KStageBytes           uint64  `json:"q4_k_stage_bytes"`
	Fallbacks               uint64  `json:"fallbacks"`
	TensorHomeHits          uint64  `json:"tensor_home_hits"`
	TensorHomeAdmissions    uint64  `json:"tensor_home_admissions"`
	TensorHomeBypasses      uint64  `json:"tensor_home_bypasses"`
	TensorHomeCopiedBytes   uint64  `json:"tensor_home_copied_bytes"`
	TensorHomeEntries       uint64  `json:"tensor_home_entries"`
	TensorHomeResidentBytes uint64  `json:"tensor_home_resident_bytes"`
	DeviceMemoryTotalBytes  *uint64 `json:"device_memory_total_bytes,omitempty"`
	DeviceMemoryFreeBytes   *uint64 `json:"device_memory_free_bytes,omitempty"`
}

// Validate ensures the evidence is complete, internally consistent, and bound
// to the backend named by the physical trial.
func (e BackendExecutionEvidence) Validate(selectedBackend string) error {
	if strings.TrimSpace(e.Backend) == "" || e.Backend != selectedBackend {
		return fmt.Errorf("qwen38campaign: observed backend %q does not match selected backend %q", e.Backend, selectedBackend)
	}
	if e.Backend != "vulkan" {
		return fmt.Errorf("qwen38campaign: physical backend observation requires vulkan, got %q", e.Backend)
	}
	if strings.TrimSpace(e.Device) == "" || strings.TrimSpace(e.Driver) == "" || strings.TrimSpace(e.Runtime) == "" {
		return errors.New("qwen38campaign: backend observation requires device, driver, and runtime")
	}
	if e.ComputeDispatches == 0 || e.DispatchSubmits == 0 {
		return errors.New("qwen38campaign: backend observation requires executed dispatches and submits")
	}
	if e.ComputeDispatches != e.Q4KMatmulDispatches+e.OtherDispatches {
		return fmt.Errorf("qwen38campaign: compute dispatch total %d does not match q4_k %d plus other %d", e.ComputeDispatches, e.Q4KMatmulDispatches, e.OtherDispatches)
	}
	if (e.DeviceMemoryTotalBytes == nil) != (e.DeviceMemoryFreeBytes == nil) {
		return errors.New("qwen38campaign: backend device-memory observation is incomplete")
	}
	if e.DeviceMemoryTotalBytes == nil {
		return errors.New("qwen38campaign: backend device-memory observation is unavailable")
	}
	if *e.DeviceMemoryTotalBytes == 0 || *e.DeviceMemoryFreeBytes > *e.DeviceMemoryTotalBytes {
		return errors.New("qwen38campaign: backend device-memory observation is invalid")
	}
	return nil
}

// Validate ensures all required cryptographic identities are non-empty.
func (id ExecutionIdentity) Validate() error {
	if strings.TrimSpace(id.SourceCommit) == "" {
		return errors.New("qwen38campaign: physical execution missing source commit")
	}
	if strings.TrimSpace(id.SourceArchiveSHA256) == "" {
		return errors.New("qwen38campaign: physical execution missing source archive sha256")
	}
	if strings.TrimSpace(id.BinarySHA256) == "" {
		return errors.New("qwen38campaign: physical execution missing binary sha256")
	}
	if strings.TrimSpace(id.ModelGGUFSHA256) == "" {
		return errors.New("qwen38campaign: physical execution missing model gguf sha256")
	}
	if strings.TrimSpace(id.TokenPacketSHA256) == "" {
		return errors.New("qwen38campaign: physical execution missing token packet sha256")
	}
	return nil
}

// PhysicalTrialRequest defines input parameters for a single physical subagent trial.
type PhysicalTrialRequest struct {
	RunIndex                   int     `json:"run_index"`
	Scenario                   string  `json:"scenario"`
	Concurrency                int     `json:"concurrency"`
	PrefixTokens               int     `json:"prefix_tokens"`
	GeneratedTokensPerSubagent int     `json:"generated_tokens_per_subagent"`
	ParityThreshold            float64 `json:"parity_threshold"`
}

// PhysicalTrialResult captures real observed telemetry from a physical device execution trial.
type PhysicalTrialResult struct {
	RunIndex          int                       `json:"run_index"`
	Concurrency       int                       `json:"concurrency"`
	Scenario          string                    `json:"scenario"`
	Backend           string                    `json:"backend"`
	ExecutionPath     string                    `json:"execution_path"`
	Identity          ExecutionIdentity         `json:"identity"`
	WallDurationMS    float64                   `json:"wall_duration_ms"`
	UsefulTokens      int                       `json:"useful_tokens"`
	TokensPerSec      float64                   `json:"tokens_per_sec"`
	QueueLatencyMS    float64                   `json:"queue_latency_ms"`
	TTFTMS            float64                   `json:"ttft_ms,omitempty"`
	TPOTMS            float64                   `json:"tpot_ms,omitempty"`
	PrefixReuseRate   float64                   `json:"prefix_reuse_rate,omitempty"`
	PeakMemoryBytes   uint64                    `json:"peak_memory_bytes,omitempty"`
	CounterSource     string                    `json:"counter_source,omitempty"`
	PhysicalDRAMBytes int64                     `json:"physical_dram_bytes,omitempty"`
	DRAMBandwidthGBps float64                   `json:"dram_bandwidth_gbps,omitempty"`
	MALLHitBytes      int64                     `json:"mall_hit_bytes,omitempty"`
	MALLTotalBytes    int64                     `json:"mall_total_bytes,omitempty"`
	MALLHitRate       float64                   `json:"mall_hit_rate,omitempty"`
	PhasesMS          map[string]float64        `json:"phases_ms"`
	PhasesUS          map[string]float64        `json:"phases_us,omitempty"`
	LogitCosineParity float64                   `json:"logit_cosine_parity"`
	ParityPassed      bool                      `json:"parity_passed"`
	OutputTokenIDs    []int32                   `json:"output_token_ids,omitempty"`
	OutputText        string                    `json:"output_text,omitempty"`
	FallbackCount     int                       `json:"fallback_count"`
	FailureCount      int                       `json:"failure_count"`
	BackendExecution  *BackendExecutionEvidence `json:"backend_execution,omitempty"`
}

// Validate ensures physical trial results satisfy acceptance contracts.
func (res PhysicalTrialResult) Validate(threshold float64) error {
	if strings.TrimSpace(res.Backend) == "" {
		return errors.New("qwen38campaign: physical trial missing backend")
	}
	if strings.TrimSpace(res.ExecutionPath) == "" {
		return errors.New("qwen38campaign: physical trial missing execution path")
	}
	if err := res.Identity.Validate(); err != nil {
		return err
	}
	if res.WallDurationMS <= 0 || math.IsNaN(res.WallDurationMS) {
		return errors.New("qwen38campaign: physical trial non-positive wall duration")
	}
	if res.UsefulTokens <= 0 {
		return errors.New("qwen38campaign: physical trial non-positive useful tokens")
	}
	if res.TokensPerSec <= 0 || math.IsNaN(res.TokensPerSec) {
		return errors.New("qwen38campaign: physical trial non-positive tokens per second")
	}
	if res.FallbackCount != 0 {
		return fmt.Errorf("qwen38campaign: physical trial fallback count must be 0, got %d", res.FallbackCount)
	}
	if res.FailureCount != 0 {
		return fmt.Errorf("qwen38campaign: physical trial reported %d failures", res.FailureCount)
	}
	if res.BackendExecution == nil {
		return errors.New("qwen38campaign: physical trial missing backend execution observation")
	}
	if err := res.BackendExecution.Validate(res.Backend); err != nil {
		return err
	}
	if uint64(res.FallbackCount) != res.BackendExecution.Fallbacks {
		return fmt.Errorf("qwen38campaign: physical trial fallback count %d does not match observed backend fallback count %d", res.FallbackCount, res.BackendExecution.Fallbacks)
	}
	if !res.ParityPassed || res.LogitCosineParity < threshold {
		return fmt.Errorf("qwen38campaign: physical trial logit cosine parity %.6f below threshold %.6f", res.LogitCosineParity, threshold)
	}
	// Hardware counters must only be present when read from a named counter source.
	if (res.PhysicalDRAMBytes > 0 || res.MALLHitBytes > 0 || res.DRAMBandwidthGBps > 0) &&
		(res.CounterSource == "" || res.CounterSource == CountersUnavailable || strings.Contains(strings.ToLower(res.CounterSource), "synthetic")) {
		return errors.New("qwen38campaign: hardware counters present without a named counter source")
	}
	return nil
}

// PhysicalRunner defines the interface for executing real product/raw model subagent fanout trials.
type PhysicalRunner interface {
	ExecuteFanoutTrial(req PhysicalTrialRequest) (PhysicalTrialResult, error)
}

// ModeledTrialRequest captures parameters for simulated trial execution.
type ModeledTrialRequest struct {
	RunIndex int
	Config   FanoutConfig
}

// ModeledRunner defines the interface for modeled/simulated execution.
type ModeledRunner interface {
	ExecuteModeledTrial(req ModeledTrialRequest) (RunMetric, error)
}

// RunMetric captures the performance, memory traffic, and parity of one benchmark trial.
type RunMetric struct {
	RunIndex          int                       `json:"run_index"`
	Concurrency       int                       `json:"concurrency"`
	Scenario          string                    `json:"scenario"`
	WallDurationMS    float64                   `json:"wall_duration_ms"`
	UsefulTokens      int                       `json:"useful_tokens"`
	TokensPerSec      float64                   `json:"tokens_per_sec"`
	PhysicalDRAMBytes int64                     `json:"physical_dram_bytes,omitempty"`
	DRAMBandwidthGBps float64                   `json:"dram_bandwidth_gbps,omitempty"`
	MALLHitBytes      int64                     `json:"mall_hit_bytes,omitempty"`
	MALLTotalBytes    int64                     `json:"mall_total_bytes,omitempty"`
	MALLHitRate       float64                   `json:"mall_hit_rate,omitempty"`
	QueueLatencyMS    float64                   `json:"queue_latency_ms"`
	TTFTMS            float64                   `json:"ttft_ms,omitempty"`
	TPOTMS            float64                   `json:"tpot_ms,omitempty"`
	PrefixReuseRate   float64                   `json:"prefix_reuse_rate,omitempty"`
	PeakMemoryBytes   uint64                    `json:"peak_memory_bytes,omitempty"`
	CounterSource     string                    `json:"counter_source,omitempty"`
	PhasesMS          map[string]float64        `json:"phases_ms"`
	PhasesUS          map[string]float64        `json:"phases_us,omitempty"`
	LogitCosineParity float64                   `json:"logit_cosine_parity"`
	ParityPassed      bool                      `json:"parity_passed"`
	FallbackCount     int                       `json:"fallback_count,omitempty"`
	FailureCount      int                       `json:"failure_count,omitempty"`
	BackendExecution  *BackendExecutionEvidence `json:"backend_execution,omitempty"`
}

// StatisticalSummary aggregates distribution metrics across all repetitions.
type StatisticalSummary struct {
	RunsCount             int     `json:"runs_count"`
	MeanTokensPerSec      float64 `json:"mean_tokens_per_sec"`
	P50TokensPerSec       float64 `json:"p50_tokens_per_sec"`
	P95TokensPerSec       float64 `json:"p95_tokens_per_sec"`
	StdDevTokensPerSec    float64 `json:"stddev_tokens_per_sec"`
	NoisePercent          float64 `json:"noise_percent"`
	MeanDRAMBandwidthGBps float64 `json:"mean_dram_bandwidth_gbps,omitempty"`
	MeanMALLHitRate       float64 `json:"mean_mall_hit_rate,omitempty"`
	MeanQueueLatencyMS    float64 `json:"mean_queue_latency_ms"`
	MeanLogitCosineParity float64 `json:"mean_logit_cosine_parity"`
	ParityThreshold       float64 `json:"parity_threshold"`
	ParityPassed          bool    `json:"parity_passed"`
	CountersStatus        string  `json:"counters_status,omitempty"`
}

// SubagentFanoutReceipt represents the final validated receipt emitted by the benchmark harness.
type SubagentFanoutReceipt struct {
	Schema            string             `json:"schema"`
	Engine            string             `json:"engine"`
	PrimaryEngine     string             `json:"primary_engine,omitempty"`
	Backend           string             `json:"backend,omitempty"`
	ExecutionPath     string             `json:"execution_path,omitempty"`
	ZeroFallback      bool               `json:"zero_fallback"`
	FallbackCount     int                `json:"fallback_count"`
	Provenance        string             `json:"provenance"`
	ExecutionIdentity *ExecutionIdentity `json:"execution_identity,omitempty"`
	Hardware          HardwareInfo       `json:"hardware"`
	Config            FanoutConfig       `json:"config"`
	Summary           StatisticalSummary `json:"summary"`
	PhaseSummaryMS    map[string]float64 `json:"phase_summary_ms"`
	Runs              []RunMetric        `json:"runs"`
	Digest            string             `json:"digest,omitempty"`
}

// Validate validates the structure, invariants, and numbers of the subagent fan-out receipt.
func (r SubagentFanoutReceipt) Validate() error {
	if r.Schema != SubagentFanoutSchema {
		return fmt.Errorf("qwen38campaign: invalid schema %q, want %q", r.Schema, SubagentFanoutSchema)
	}
	if r.Engine != CanonicalEngineName {
		return fmt.Errorf("qwen38campaign: invalid engine %q, want %q", r.Engine, CanonicalEngineName)
	}
	if err := r.Config.Validate(); err != nil {
		return fmt.Errorf("qwen38campaign: invalid config: %w", err)
	}
	if len(r.Runs) != r.Config.Runs {
		return fmt.Errorf("qwen38campaign: expected %d runs, got %d", r.Config.Runs, len(r.Runs))
	}
	if r.Summary.RunsCount != len(r.Runs) {
		return fmt.Errorf("qwen38campaign: summary runs count %d does not match runs length %d",
			r.Summary.RunsCount, len(r.Runs))
	}

	// Provenance class enforcement: simulation cannot be relabeled as physical.
	if r.Config.Simulated {
		if r.Provenance != "" && r.Provenance != ProvenanceSimulation {
			return fmt.Errorf("qwen38campaign: simulated receipt cannot declare provenance %q (must be %q)", r.Provenance, ProvenanceSimulation)
		}
	} else {
		if r.Provenance != ProvenancePhysical {
			return fmt.Errorf("qwen38campaign: physical receipt requires provenance %q, got %q", ProvenancePhysical, r.Provenance)
		}
		if r.PrimaryEngine != CanonicalEngineName {
			return fmt.Errorf("qwen38campaign: physical receipt requires primary engine %q, got %q", CanonicalEngineName, r.PrimaryEngine)
		}
		if strings.TrimSpace(r.Backend) == "" {
			return errors.New("qwen38campaign: physical receipt requires non-empty backend")
		}
		if strings.TrimSpace(r.ExecutionPath) == "" {
			return errors.New("qwen38campaign: physical receipt requires non-empty execution_path")
		}
		if r.FallbackCount != 0 || !r.ZeroFallback {
			return fmt.Errorf("qwen38campaign: physical receipt requires zero fallback, got count=%d zero_fallback=%v", r.FallbackCount, r.ZeroFallback)
		}
		if r.ExecutionIdentity == nil {
			return errors.New("qwen38campaign: physical receipt missing execution identity")
		}
		if err := r.ExecutionIdentity.Validate(); err != nil {
			return fmt.Errorf("qwen38campaign: invalid execution identity: %w", err)
		}
	}

	for _, phase := range CanonicalPhaseBuckets {
		if _, ok := r.PhaseSummaryMS[phase]; !ok {
			return fmt.Errorf("qwen38campaign: missing canonical phase %q in phase_summary_ms", phase)
		}
	}

	for i, run := range r.Runs {
		if run.RunIndex != i+1 {
			return fmt.Errorf("qwen38campaign: run %d has mismatching index %d", i, run.RunIndex)
		}
		if run.Concurrency != r.Config.Concurrency {
			return fmt.Errorf("qwen38campaign: run %d concurrency %d does not match config %d",
				i, run.Concurrency, r.Config.Concurrency)
		}
		if run.Scenario != r.Config.Scenario {
			return fmt.Errorf("qwen38campaign: run %d scenario %q does not match config %q",
				i, run.Scenario, r.Config.Scenario)
		}
		if run.WallDurationMS <= 0 || math.IsNaN(run.WallDurationMS) {
			return fmt.Errorf("qwen38campaign: run %d non-positive wall duration: %f", i, run.WallDurationMS)
		}
		if run.TokensPerSec <= 0 || math.IsNaN(run.TokensPerSec) {
			return fmt.Errorf("qwen38campaign: run %d non-positive tokens per second: %f", i, run.TokensPerSec)
		}
		if run.MALLHitRate < 0.0 || run.MALLHitRate > 1.0 {
			return fmt.Errorf("qwen38campaign: run %d invalid mall hit rate: %f", i, run.MALLHitRate)
		}
		if run.LogitCosineParity < r.Config.ParityThreshold {
			return fmt.Errorf("qwen38campaign: run %d logit cosine parity %.6f below threshold %.6f",
				i, run.LogitCosineParity, r.Config.ParityThreshold)
		}
		for _, phase := range CanonicalPhaseBuckets {
			if _, ok := run.PhasesMS[phase]; !ok {
				return fmt.Errorf("qwen38campaign: run %d missing canonical phase %q in phases_ms", i, phase)
			}
		}
		if !r.Config.Simulated {
			if run.BackendExecution == nil {
				return fmt.Errorf("qwen38campaign: run %d missing backend execution observation", i)
			}
			if err := run.BackendExecution.Validate(r.Backend); err != nil {
				return fmt.Errorf("qwen38campaign: run %d invalid backend execution observation: %w", i, err)
			}
			if run.FallbackCount != 0 {
				return fmt.Errorf("qwen38campaign: run %d fallback count %d != 0", i, run.FallbackCount)
			}
			if run.FailureCount != 0 {
				return fmt.Errorf("qwen38campaign: run %d reported %d failures", i, run.FailureCount)
			}
			if (run.PhysicalDRAMBytes > 0 || run.MALLHitBytes > 0 || run.DRAMBandwidthGBps > 0) &&
				(run.CounterSource == "" || run.CounterSource == CountersUnavailable || strings.Contains(strings.ToLower(run.CounterSource), "synthetic")) {
				return fmt.Errorf("qwen38campaign: run %d hardware counters present without named counter source", i)
			}
		}
	}

	if !r.Summary.ParityPassed {
		return errors.New("qwen38campaign: receipt summary indicates logit parity failure")
	}

	return nil
}

// ComputeDigest computes a canonical SHA-256 digest over the receipt payload.
func (r *SubagentFanoutReceipt) ComputeDigest() (string, error) {
	clone := *r
	clone.Digest = ""
	data, err := json.Marshal(clone)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(data)
	return fmt.Sprintf("sha256:%s", hex.EncodeToString(sum[:])), nil
}

// JSON serializes the receipt to pretty-printed JSON.
func (r SubagentFanoutReceipt) JSON() ([]byte, error) {
	return json.MarshalIndent(r, "", "  ")
}

// String renders an operator-readable summary table.
func (r SubagentFanoutReceipt) String() string {
	var b bytes.Buffer
	fmt.Fprintf(&b, "================================================================================\n")
	fmt.Fprintf(&b, "Strix Halo Subagent Fan-Out Benchmark Receipt (%s)\n", r.Schema)
	fmt.Fprintf(&b, "================================================================================\n")
	fmt.Fprintf(&b, "Engine:              %s\n", r.Engine)
	fmt.Fprintf(&b, "Architecture:        %s\n", r.Hardware.Architecture)
	fmt.Fprintf(&b, "Target Device:       %s (%s)\n", r.Hardware.Device, r.Hardware.DeviceName)
	fmt.Fprintf(&b, "MALL Cache Size:     %d MB\n", r.Hardware.MALLCacheSizeMB)
	fmt.Fprintf(&b, "Sustainable DRAM BW: %.2f GB/s (Peak: %.2f GB/s)\n",
		r.Hardware.DRAMBandwidthSustainableGBps, r.Hardware.DRAMBandwidthTheoreticalGBps)
	fmt.Fprintf(&b, "Sustainable MALL BW: %.2f GB/s\n\n", r.Hardware.MALLBandwidthSustainedGBps)

	modeStr := "Physical Device Execution"
	if r.Config.Simulated {
		modeStr = "Calibrated Architecture Simulation"
	}
	fmt.Fprintf(&b, "Benchmark Parameters:\n")
	fmt.Fprintf(&b, "  Scenario:          %s\n", r.Config.Scenario)
	fmt.Fprintf(&b, "  Concurrency (B):   %d subagents\n", r.Config.Concurrency)
	fmt.Fprintf(&b, "  Repetitions:       %d runs\n", r.Config.Runs)
	fmt.Fprintf(&b, "  Prefix Tokens:     %d tokens\n", r.Config.PrefixTokens)
	fmt.Fprintf(&b, "  Gen Tokens/Sub:    %d tokens\n", r.Config.GeneratedTokensPerSubagent)
	fmt.Fprintf(&b, "  Execution Mode:    %s\n", modeStr)
	if r.Provenance != "" {
		fmt.Fprintf(&b, "  Provenance:        %s\n", r.Provenance)
	}
	if r.ExecutionIdentity != nil {
		fmt.Fprintf(&b, "  Source Commit:     %s\n", r.ExecutionIdentity.SourceCommit)
		fmt.Fprintf(&b, "  Binary SHA256:     %s\n", r.ExecutionIdentity.BinarySHA256)
		fmt.Fprintf(&b, "  Model GGUF SHA256: %s\n", r.ExecutionIdentity.ModelGGUFSHA256)
		fmt.Fprintf(&b, "  Token Packet SHA:  %s\n", r.ExecutionIdentity.TokenPacketSHA256)
	}
	fmt.Fprintf(&b, "\n")

	fmt.Fprintf(&b, "Statistical Performance Summary:\n")
	fmt.Fprintf(&b, "  Mean Throughput:   %10.2f tokens/sec\n", r.Summary.MeanTokensPerSec)
	fmt.Fprintf(&b, "  P50  Throughput:   %10.2f tokens/sec\n", r.Summary.P50TokensPerSec)
	fmt.Fprintf(&b, "  P95  Throughput:   %10.2f tokens/sec\n", r.Summary.P95TokensPerSec)
	fmt.Fprintf(&b, "  StdDev Throughput: %10.2f tokens/sec (Noise: %.2f%%)\n",
		r.Summary.StdDevTokensPerSec, r.Summary.NoisePercent)
	if r.Summary.CountersStatus == CountersUnavailable {
		fmt.Fprintf(&b, "  Mean DRAM Traffic: UNAVAILABLE\n")
		fmt.Fprintf(&b, "  Mean MALL Hit Rate:UNAVAILABLE\n")
	} else {
		fmt.Fprintf(&b, "  Mean DRAM Traffic: %10.2f GB/s\n", r.Summary.MeanDRAMBandwidthGBps)
		fmt.Fprintf(&b, "  Mean MALL Hit Rate:%9.2f%%\n", r.Summary.MeanMALLHitRate*100)
	}
	fmt.Fprintf(&b, "  Mean Queue Latency:%9.3f ms\n", r.Summary.MeanQueueLatencyMS)
	fmt.Fprintf(&b, "  Logit Cosine:      %10.6f (Threshold: >= %.6f) [%s]\n\n",
		r.Summary.MeanLogitCosineParity, r.Summary.ParityThreshold,
		func() string {
			if r.Summary.ParityPassed {
				return "PASS"
			}
			return "FAIL"
		}())

	fmt.Fprintf(&b, "Canonical Phase Timing Breakdown (Mean):\n")
	for _, phase := range CanonicalPhaseBuckets {
		ms := r.PhaseSummaryMS[phase]
		var pct float64
		totalWall := 0.0
		for _, p := range CanonicalPhaseBuckets {
			totalWall += r.PhaseSummaryMS[p]
		}
		if totalWall > 0 {
			pct = (ms / totalWall) * 100.0
		}
		fmt.Fprintf(&b, "  %-20s %10.3f ms (%6.2f%%)\n", phase+":", ms, pct)
	}

	fmt.Fprintf(&b, "\nPer-Run Execution Log:\n")
	fmt.Fprintf(&b, "  %-4s %10s %10s %12s %12s %10s %12s\n",
		"Run", "Wall (ms)", "Tokens", "Tokens/sec", "DRAM (GB/s)", "MALL Hit%", "Parity")
	for _, run := range r.Runs {
		if r.Summary.CountersStatus == CountersUnavailable {
			fmt.Fprintf(&b, "  #%-3d %10.2f %10d %12.2f %12s %10s %12.6f\n",
				run.RunIndex, run.WallDurationMS, run.UsefulTokens, run.TokensPerSec,
				"UNAVAILABLE", "UNAVAILABLE", run.LogitCosineParity)
		} else {
			fmt.Fprintf(&b, "  #%-3d %10.2f %10d %12.2f %12.2f %9.2f%% %12.6f\n",
				run.RunIndex, run.WallDurationMS, run.UsefulTokens, run.TokensPerSec,
				run.DRAMBandwidthGBps, run.MALLHitRate*100, run.LogitCosineParity)
		}
	}

	if r.Digest != "" {
		fmt.Fprintf(&b, "\nVerification Digest: %s\n", r.Digest)
	}
	return b.String()
}

// CosineSimilarity computes the vector cosine similarity between two float64 slices.
func CosineSimilarity(a, b []float64) float64 {
	if len(a) == 0 || len(a) != len(b) {
		return 0.0
	}
	var dot, normA, normB float64
	for i := 0; i < len(a); i++ {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA <= 0 || normB <= 0 {
		return 0.0
	}
	cos := dot / (math.Sqrt(normA) * math.Sqrt(normB))
	if cos > 1.0 {
		cos = 1.0
	}
	return cos
}

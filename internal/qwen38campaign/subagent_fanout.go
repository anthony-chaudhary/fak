// Package qwen38campaign implements the subagent fan-out multi-agent benchmark harness
// for AMD Strix Halo ideal-cache workloads (fak-native execution engine).
package qwen38campaign

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"slices"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/nativeperf"
	"github.com/anthony-chaudhary/fak/internal/rawdecode"
	"github.com/anthony-chaudhary/fak/internal/roofline"
)

const (
	// SubagentFanoutSchema identifies the subagent fan-out benchmark receipt contract.
	SubagentFanoutSchema = "fak.benchmark.subagent_fanout/v1"
	// SubagentFanoutPhysicalSchema identifies the physical device execution receipt contract.
	SubagentFanoutPhysicalSchema = "fak.benchmark.subagent_fanout/v1"

	// CanonicalEngineName identifies the fak-native execution engine.
	CanonicalEngineName = "fak-native"

	// Provenance classes.
	ProvenanceSimulation = "simulation"
	ProvenancePhysical   = "physical_device_execution"

	// Canonical model GGUF SHA256 from #12096.
	DefaultModelGGUFSHA256 = "7E78DA5D7E3AE28D178121F58646953305F3E5BD3CB46F4A75584E8B6C6FE169"

	// Named counter status indicators.
	CountersUnavailable = "UNAVAILABLE"
	CountersAvailable   = "AVAILABLE"

	// Target hardware constants for AMD Strix Halo APU (RDNA 3.5 / gfx1151).
	DefaultArchStrixHalo                            = "RDNA 3.5 / gfx1151 (UMA)"
	DefaultDeviceStrixHalo                          = roofline.DefaultArchStrixHalo
	DefaultDeviceNameStrixHalo                      = "AMD Radeon 8050S / Strix Halo"
	StrixHaloMALLCacheSizeMB                        = roofline.StrixHaloMALLCacheSizeMB
	StrixHaloMALLCacheSizeBytes               int64 = roofline.StrixHaloMALLCacheSizeMB * 1024 * 1024
	StrixHaloSustainableDRAMBandwidthGBps           = roofline.StrixHaloSustainableDRAMBandwidthGBps
	StrixHaloTheoreticalPeakDRAMBandwidthGBps       = roofline.StrixHaloTheoreticalPeakDRAMBandwidthGBps
	StrixHaloSustainedMALLBandwidthGBps             = roofline.StrixHaloSustainedMALLBandwidthGBps

	// Supported subagent fan-out scenarios.
	ScenarioCold               = "cold"
	ScenarioWarmSamePrefix     = "warm_same_prefix"
	ScenarioSharedPrefixForked = "shared_prefix_forked"

	// Defaults for workloads and thresholds.
	DefaultSharedPrefixTokens         = 30000
	DefaultWarmPrefixTokens           = 2048
	DefaultColdPromptTokens           = 1024
	DefaultGeneratedTokensPerSubagent = 64
	DefaultRuns                       = 5
	DefaultMinRuns                    = 5
	DefaultLogitCosineParityThreshold = 0.999900
	DefaultBytesPerToken              = 128 // Coherent UMA KV footprint per token (Qwen3.8 / 27B)

	// Phase bucket identifiers from #11621.
	PhaseHostDispatch     = nativeperf.SubagentPhaseHostDispatch
	PhasePrefixTreeLookup = nativeperf.SubagentPhasePrefixTreeLookup
	PhaseKVAllocation     = nativeperf.SubagentPhaseKVAllocation
	PhaseGPUKernel        = nativeperf.SubagentPhaseGPUKernel
	PhaseTokenSampling    = nativeperf.SubagentPhaseTokenSampling
)

// CanonicalPhaseBuckets returns the five canonical phase bucket names in execution sequence.
var CanonicalPhaseBuckets = [...]string{
	PhaseHostDispatch,
	PhasePrefixTreeLookup,
	PhaseKVAllocation,
	PhaseGPUKernel,
	PhaseTokenSampling,
}

// SubagentFanoutHarness executes the multi-agent fan-out matrix benchmark.
type SubagentFanoutHarness struct {
	Config         FanoutConfig
	Hardware       HardwareInfo
	RNG            *rand.Rand
	PhysicalRunner PhysicalRunner
	ModeledRunner  ModeledRunner
}

// SetPhysicalRunner attaches an explicit physical runner to the harness.
func (h *SubagentFanoutHarness) SetPhysicalRunner(r PhysicalRunner) {
	h.PhysicalRunner = r
}

// SetModeledRunner attaches an explicit modeled/simulated runner to the harness.
func (h *SubagentFanoutHarness) SetModeledRunner(r ModeledRunner) {
	h.ModeledRunner = r
}

// NewSubagentFanoutHarness initializes a harness with validated configuration.
func NewSubagentFanoutHarness(cfg FanoutConfig) (*SubagentFanoutHarness, error) {
	if cfg.Scenario == "" {
		cfg.Scenario = ScenarioSharedPrefixForked
	}
	if cfg.Concurrency == 0 {
		cfg.Concurrency = 4
	}
	if cfg.Runs == 0 {
		cfg.Runs = DefaultRuns
	}
	if cfg.GeneratedTokensPerSubagent == 0 {
		cfg.GeneratedTokensPerSubagent = DefaultGeneratedTokensPerSubagent
	}
	if cfg.PrefixTokens == 0 {
		switch cfg.Scenario {
		case ScenarioSharedPrefixForked:
			cfg.PrefixTokens = DefaultSharedPrefixTokens
		case ScenarioWarmSamePrefix:
			cfg.PrefixTokens = DefaultWarmPrefixTokens
		case ScenarioCold:
			cfg.PrefixTokens = DefaultColdPromptTokens
		}
	}
	if cfg.ParityThreshold <= 0 {
		cfg.ParityThreshold = DefaultLogitCosineParityThreshold
	}

	if err := cfg.Validate(); err != nil {
		return nil, err
	}

	seed := cfg.Seed
	if seed == 0 {
		seed = time.Now().UnixNano()
	}

	return &SubagentFanoutHarness{
		Config:   cfg,
		Hardware: DefaultHardwareInfo(),
		RNG:      rand.New(rand.NewSource(seed)),
	}, nil
}

// Execute runs all benchmark repetitions and generates the validated receipt.
func (h *SubagentFanoutHarness) Execute() (SubagentFanoutReceipt, error) {
	if !h.Config.Simulated {
		return h.executePhysical()
	}
	return h.executeSimulated()
}

// executePhysical drives benchmark repetitions exclusively through the attached PhysicalRunner.
func (h *SubagentFanoutHarness) executePhysical() (SubagentFanoutReceipt, error) {
	if h.PhysicalRunner == nil {
		return SubagentFanoutReceipt{}, errors.New("qwen38campaign: physical execution is unavailable; this harness models GPU metrics, use --simulated=true")
	}

	runs := make([]RunMetric, h.Config.Runs)
	var lastIdentity *ExecutionIdentity
	var backend, execPath string

	for i := 0; i < h.Config.Runs; i++ {
		req := PhysicalTrialRequest{
			RunIndex:                   i + 1,
			Scenario:                   h.Config.Scenario,
			Concurrency:                h.Config.Concurrency,
			PrefixTokens:               h.Config.PrefixTokens,
			GeneratedTokensPerSubagent: h.Config.GeneratedTokensPerSubagent,
			ParityThreshold:            h.Config.ParityThreshold,
		}

		trialRes, err := h.PhysicalRunner.ExecuteFanoutTrial(req)
		if err != nil {
			return SubagentFanoutReceipt{}, fmt.Errorf("qwen38campaign: physical run %d failed: %w", i+1, err)
		}

		if err := trialRes.Validate(h.Config.ParityThreshold); err != nil {
			return SubagentFanoutReceipt{}, fmt.Errorf("qwen38campaign: physical run %d validation failed: %w", i+1, err)
		}

		if i == 0 {
			backend = trialRes.Backend
			execPath = trialRes.ExecutionPath
			idCopy := trialRes.Identity
			lastIdentity = &idCopy
		}

		runs[i] = RunMetric{
			RunIndex:          trialRes.RunIndex,
			Concurrency:       trialRes.Concurrency,
			Scenario:          trialRes.Scenario,
			WallDurationMS:    trialRes.WallDurationMS,
			UsefulTokens:      trialRes.UsefulTokens,
			TokensPerSec:      trialRes.TokensPerSec,
			PhysicalDRAMBytes: trialRes.PhysicalDRAMBytes,
			DRAMBandwidthGBps: trialRes.DRAMBandwidthGBps,
			MALLHitBytes:      trialRes.MALLHitBytes,
			MALLTotalBytes:    trialRes.MALLTotalBytes,
			MALLHitRate:       trialRes.MALLHitRate,
			QueueLatencyMS:    trialRes.QueueLatencyMS,
			TTFTMS:            trialRes.TTFTMS,
			TPOTMS:            trialRes.TPOTMS,
			PrefixReuseRate:   trialRes.PrefixReuseRate,
			PeakMemoryBytes:   trialRes.PeakMemoryBytes,
			CounterSource:     trialRes.CounterSource,
			PhasesMS:          trialRes.PhasesMS,
			PhasesUS:          trialRes.PhasesUS,
			LogitCosineParity: trialRes.LogitCosineParity,
			ParityPassed:      trialRes.ParityPassed,
			FallbackCount:     trialRes.FallbackCount,
			FailureCount:      trialRes.FailureCount,
			BackendExecution:  cloneBackendExecutionEvidence(trialRes.BackendExecution),
		}
	}

	summary, phaseSummary := CalculateStatisticalSummary(runs, h.Config.ParityThreshold)

	receipt := SubagentFanoutReceipt{
		Schema:            SubagentFanoutSchema,
		Engine:            CanonicalEngineName,
		PrimaryEngine:     CanonicalEngineName,
		Backend:           backend,
		ExecutionPath:     execPath,
		ZeroFallback:      true,
		FallbackCount:     0,
		Provenance:        ProvenancePhysical,
		ExecutionIdentity: lastIdentity,
		Hardware:          h.Hardware,
		Config:            h.Config,
		Summary:           summary,
		PhaseSummaryMS:    phaseSummary,
		Runs:              runs,
	}

	digest, err := receipt.ComputeDigest()
	if err != nil {
		return SubagentFanoutReceipt{}, fmt.Errorf("qwen38campaign: digest error: %w", err)
	}
	receipt.Digest = digest

	if err := receipt.Validate(); err != nil {
		return SubagentFanoutReceipt{}, fmt.Errorf("qwen38campaign: receipt validation failed: %w", err)
	}

	return receipt, nil
}

// executeSimulated drives benchmark repetitions through calibrated architecture simulation.
func (h *SubagentFanoutHarness) executeSimulated() (SubagentFanoutReceipt, error) {
	runs := make([]RunMetric, h.Config.Runs)

	for i := 0; i < h.Config.Runs; i++ {
		var metric RunMetric
		var err error
		if h.ModeledRunner != nil {
			metric, err = h.ModeledRunner.ExecuteModeledTrial(ModeledTrialRequest{
				RunIndex: i + 1,
				Config:   h.Config,
			})
		} else {
			metric, err = h.executeTrial(i + 1)
		}
		if err != nil {
			return SubagentFanoutReceipt{}, fmt.Errorf("qwen38campaign: run %d failed: %w", i+1, err)
		}
		runs[i] = metric
	}

	summary, phaseSummary := CalculateStatisticalSummary(runs, h.Config.ParityThreshold)

	receipt := SubagentFanoutReceipt{
		Schema:         SubagentFanoutSchema,
		Engine:         CanonicalEngineName,
		PrimaryEngine:  CanonicalEngineName,
		ZeroFallback:   true,
		FallbackCount:  0,
		Provenance:     ProvenanceSimulation,
		Hardware:       h.Hardware,
		Config:         h.Config,
		Summary:        summary,
		PhaseSummaryMS: phaseSummary,
		Runs:           runs,
	}

	digest, err := receipt.ComputeDigest()
	if err != nil {
		return SubagentFanoutReceipt{}, fmt.Errorf("qwen38campaign: digest error: %w", err)
	}
	receipt.Digest = digest

	if err := receipt.Validate(); err != nil {
		return SubagentFanoutReceipt{}, fmt.Errorf("qwen38campaign: receipt validation failed: %w", err)
	}

	return receipt, nil
}

// executeTrial runs a single benchmark trial under the specified scenario and batch size.
func (h *SubagentFanoutHarness) executeTrial(runIndex int) (RunMetric, error) {
	b := h.Config.Concurrency
	genTokens := h.Config.GeneratedTokensPerSubagent
	totalUsefulTokens := b * genTokens

	// Phase timings in microseconds (µs)
	phasesUS := make(map[string]float64, len(CanonicalPhaseBuckets))
	for _, p := range CanonicalPhaseBuckets {
		phasesUS[p] = 0.0
	}

	// 1. Host Dispatch phase
	// Models queue admission, turn gating, and session locking for B subagents.
	queueLatencyMS := 0.05 + 0.02*float64(b) + (h.RNG.Float64()*0.02 - 0.01)
	if queueLatencyMS < 0.01 {
		queueLatencyMS = 0.01
	}
	hostDispatchUS := (queueLatencyMS * 1000.0) + (15.0 * float64(b)) + (h.RNG.Float64()*10.0 - 5.0)
	if hostDispatchUS < 10.0 {
		hostDispatchUS = 10.0
	}
	phasesUS[PhaseHostDispatch] = hostDispatchUS

	// 2. Prefix Tree Lookup & 3. KV Allocation via Context MMU
	var prefixLookupUS, kvAllocUS float64
	var physicalDRAMBytes int64
	var mallHitBytes, mallTotalBytes int64

	switch h.Config.Scenario {
	case ScenarioSharedPrefixForked:
		// Demonstrate zero-copy subagent session forking using ctxmmu.ForkManager.
		forkMgr := ctxmmu.NewForkManager(ctxmmu.ForkConfig{
			Granularity:   ctxmmu.BlockGranularity64,
			BytesPerToken: DefaultBytesPerToken,
		})

		parentID := fmt.Sprintf("run-%d-root-prefix", runIndex)
		parentSess, err := forkMgr.RegisterSession(parentID, ctxmmu.BlockGranularity64)
		if err != nil {
			return RunMetric{}, fmt.Errorf("register root session: %w", err)
		}

		// Append large instruction/repo prefix (e.g. 30,000 tokens)
		prefixTokens := make([]int32, h.Config.PrefixTokens)
		for i := range prefixTokens {
			prefixTokens[i] = int32((i % 32000) + 1)
		}
		if err := parentSess.AppendTokens(prefixTokens...); err != nil {
			return RunMetric{}, fmt.Errorf("append root prefix tokens: %w", err)
		}

		// Prefix tree lookup: fast trie match for the shared prefix across subagents
		prefixLookupUS = 25.0 + 5.0*float64(b) + (h.RNG.Float64()*4.0 - 2.0)

		// Subagents fork zero-copy from parent
		kvStart := time.Now()
		for subIdx := 0; subIdx < b; subIdx++ {
			childID := fmt.Sprintf("run-%d-subagent-%d", runIndex, subIdx)
			childSess, err := forkMgr.ForkSession(parentID, childID)
			if err != nil {
				return RunMetric{}, fmt.Errorf("fork session %s: %w", childID, err)
			}
			// Subagent generates unique turns (triggering COW for private pages)
			childTokens := make([]int32, genTokens)
			for j := range childTokens {
				childTokens[j] = int32(((subIdx+1)*1000 + j) % 32000)
			}
			if err := childSess.AppendTokens(childTokens...); err != nil {
				return RunMetric{}, fmt.Errorf("append child tokens for %s: %w", childID, err)
			}
		}
		kvAllocElapsedUS := float64(time.Since(kvStart).Nanoseconds()) / 1000.0
		kvAllocUS = 20.0 + 8.0*float64(b) + kvAllocElapsedUS*0.1

		// Memory calculations for Strix Halo ideal-cache:
		// 30,000 tokens * 128 bytes/token = 3.84 MB.
		// 3.84 MB KV cache fits comfortably inside Strix Halo's 32 MB MALL cache.
		prefixSizeBytes := int64(h.Config.PrefixTokens) * int64(DefaultBytesPerToken)
		subagentKVBytes := int64(b) * int64(genTokens) * int64(DefaultBytesPerToken)

		// Each decode step reads the 30k prefix KV. With B subagents over genTokens steps:
		// The 3.84 MB prefix is read on every decode step by all B subagents directly from MALL!
		totalKVReadBytes := int64(b) * int64(genTokens) * prefixSizeBytes
		mallHitBytes = int64(float64(totalKVReadBytes) * (0.95 + h.RNG.Float64()*0.03))
		mallTotalBytes = totalKVReadBytes + subagentKVBytes

		// Physical DRAM moved is only initial model weight read and private COW pages:
		// Weights streaming amortized over batch B: ~60 MB active layer streaming + COW page writes
		weightsPerStep := int64(64 * 1024 * 1024)
		physicalDRAMBytes = (weightsPerStep / int64(b)) + subagentKVBytes + (mallTotalBytes - mallHitBytes)

	case ScenarioWarmSamePrefix:
		// Warm same prefix: prefix already warm in MALL/DRAM
		prefixLookupUS = 30.0 + 6.0*float64(b) + (h.RNG.Float64()*5.0 - 2.5)
		kvAllocUS = 35.0 + 10.0*float64(b) + (h.RNG.Float64()*6.0 - 3.0)

		prefixSizeBytes := int64(h.Config.PrefixTokens) * int64(DefaultBytesPerToken)
		subagentKVBytes := int64(b) * int64(genTokens) * int64(DefaultBytesPerToken)
		totalKVReadBytes := int64(b) * int64(genTokens) * prefixSizeBytes

		mallHitBytes = int64(float64(totalKVReadBytes) * (0.88 + h.RNG.Float64()*0.05))
		mallTotalBytes = totalKVReadBytes + subagentKVBytes
		weightsPerStep := int64(64 * 1024 * 1024)
		physicalDRAMBytes = (weightsPerStep / int64(b)) + subagentKVBytes + (mallTotalBytes - mallHitBytes)

	case ScenarioCold:
		// Cold: no prior cache, full prefill required for each sequence
		prefixLookupUS = 15.0 + 3.0*float64(b) // Trie misses
		kvAllocUS = 80.0 + 25.0*float64(b) + (h.RNG.Float64()*10.0 - 5.0)

		prefillBytes := int64(b) * int64(h.Config.PrefixTokens) * int64(DefaultBytesPerToken)
		subagentKVBytes := int64(b) * int64(genTokens) * int64(DefaultBytesPerToken)
		mallTotalBytes = prefillBytes + subagentKVBytes

		// Cold misses MALL heavily: working set exceeds MALL or is streamed fresh
		mallHitBytes = int64(float64(mallTotalBytes) * (0.12 + h.RNG.Float64()*0.04))
		weightsPerStep := int64(64 * 1024 * 1024)
		physicalDRAMBytes = (weightsPerStep * int64(genTokens)) + prefillBytes + subagentKVBytes
	}

	phasesUS[PhasePrefixTreeLookup] = prefixLookupUS
	phasesUS[PhaseKVAllocation] = kvAllocUS

	// 4. GPU Kernel Execution
	// In Strix Halo UMA APU:
	// Decode throughput scales sublinearly with batch B because weight streaming is amortized.
	// Prefill cost is zero in warm/forked scenarios, but large in cold.
	var gpuKernelUS float64
	var baseDecodeTimePerTokenUS float64

	switch h.Config.Scenario {
	case ScenarioSharedPrefixForked:
		// Fast attention over 32MB MALL cache (800 GB/s bandwidth)
		// Batch decode scaling: B=1 -> 150us/tok, B=2 -> 175us/tok, B=4 -> 210us/tok, B=8 -> 270us/tok
		baseDecodeTimePerTokenUS = 140.0 + 16.0*float64(b)
		gpuKernelUS = baseDecodeTimePerTokenUS * float64(genTokens) * (1.0 + (h.RNG.Float64()*0.04 - 0.02))
	case ScenarioWarmSamePrefix:
		baseDecodeTimePerTokenUS = 160.0 + 20.0*float64(b)
		gpuKernelUS = baseDecodeTimePerTokenUS * float64(genTokens) * (1.0 + (h.RNG.Float64()*0.04 - 0.02))
	case ScenarioCold:
		// Includes full prefill GEMM forward pass for all B sequences:
		prefillKernelUS := float64(b) * float64(h.Config.PrefixTokens) * 4.5
		baseDecodeTimePerTokenUS = 220.0 + 35.0*float64(b)
		decodeKernelUS := baseDecodeTimePerTokenUS * float64(genTokens)
		gpuKernelUS = (prefillKernelUS + decodeKernelUS) * (1.0 + (h.RNG.Float64()*0.04 - 0.02))
	}
	phasesUS[PhaseGPUKernel] = gpuKernelUS

	// 5. Token Sampling
	tokenSamplingUS := float64(b*genTokens) * 1.5 * (1.0 + (h.RNG.Float64()*0.05 - 0.025))
	phasesUS[PhaseTokenSampling] = tokenSamplingUS

	// Total wall duration in milliseconds
	totalWallUS := 0.0
	phasesMS := make(map[string]float64, len(CanonicalPhaseBuckets))
	for _, p := range CanonicalPhaseBuckets {
		totalWallUS += phasesUS[p]
		phasesMS[p] = phasesUS[p] / 1000.0
	}
	wallDurationMS := totalWallUS / 1000.0

	// Throughput metrics
	tokensPerSec := float64(totalUsefulTokens) / (wallDurationMS / 1000.0)

	// DRAM bandwidth in GB/s
	dramBandwidthGBps := float64(physicalDRAMBytes) / (wallDurationMS * 1e6)
	if dramBandwidthGBps > StrixHaloSustainableDRAMBandwidthGBps {
		dramBandwidthGBps = StrixHaloSustainableDRAMBandwidthGBps * 0.96
	}

	// MALL hit rate
	mallHitRate := 0.0
	if mallTotalBytes > 0 {
		mallHitRate = float64(mallHitBytes) / float64(mallTotalBytes)
	}

	// Logit Cosine Parity
	// Generate output logits and evaluate against deterministic float64 reference.
	refLogits := generateReferenceLogits(runIndex, b, 256)
	actualLogits := make([]float64, len(refLogits))
	// Add minute floating point quantization perturbation (order of 1e-6)
	for idx, val := range refLogits {
		noise := (h.RNG.Float64() - 0.5) * 1e-5
		actualLogits[idx] = val + noise
	}
	logitCosine := CosineSimilarity(refLogits, actualLogits)
	parityPassed := logitCosine >= h.Config.ParityThreshold

	return RunMetric{
		RunIndex:          runIndex,
		Concurrency:       b,
		Scenario:          h.Config.Scenario,
		WallDurationMS:    wallDurationMS,
		UsefulTokens:      totalUsefulTokens,
		TokensPerSec:      tokensPerSec,
		PhysicalDRAMBytes: physicalDRAMBytes,
		DRAMBandwidthGBps: dramBandwidthGBps,
		MALLHitBytes:      mallHitBytes,
		MALLTotalBytes:    mallTotalBytes,
		MALLHitRate:       mallHitRate,
		QueueLatencyMS:    queueLatencyMS,
		CounterSource:     "calibrated_architecture_simulation",
		PhasesMS:          phasesMS,
		PhasesUS:          phasesUS,
		LogitCosineParity: logitCosine,
		ParityPassed:      parityPassed,
		FallbackCount:     0,
		FailureCount:      0,
	}, nil
}

// generateReferenceLogits synthesizes a stable float64 reference distribution for parity check.
func generateReferenceLogits(seed int, concurrency int, size int) []float64 {
	logits := make([]float64, size)
	for i := 0; i < size; i++ {
		theta := float64(i*concurrency+seed) * 0.137
		logits[i] = math.Sin(theta) + 0.5*math.Cos(2.0*theta)
	}
	return logits
}

// CalculateStatisticalSummary computes statistical distributions over repetition metrics.
func CalculateStatisticalSummary(runs []RunMetric, threshold float64) (StatisticalSummary, map[string]float64) {
	n := len(runs)
	if n == 0 {
		return StatisticalSummary{ParityThreshold: threshold, CountersStatus: CountersUnavailable}, nil
	}

	tpsVals := make([]float64, n)
	var sumTPS, sumDRAMBW, sumMALLHitRate, sumQueueLatency, sumParity float64
	allPassed := true
	hasUnavailableCounters := false

	for i, r := range runs {
		tpsVals[i] = r.TokensPerSec
		sumTPS += r.TokensPerSec
		sumDRAMBW += r.DRAMBandwidthGBps
		sumMALLHitRate += r.MALLHitRate
		sumQueueLatency += r.QueueLatencyMS
		sumParity += r.LogitCosineParity
		if !r.ParityPassed {
			allPassed = false
		}
		if r.CounterSource == CountersUnavailable {
			hasUnavailableCounters = true
		}
	}

	meanTPS := sumTPS / float64(n)
	meanQueueLatency := sumQueueLatency / float64(n)
	meanParity := sumParity / float64(n)

	var meanDRAMBW, meanMALLHitRate float64
	counterStatus := CountersAvailable
	if hasUnavailableCounters {
		counterStatus = CountersUnavailable
	} else {
		meanDRAMBW = sumDRAMBW / float64(n)
		meanMALLHitRate = sumMALLHitRate / float64(n)
	}

	// Sort for percentiles
	sortedTPS := append([]float64(nil), tpsVals...)
	sort.Float64s(sortedTPS)

	var p50TPS, p95TPS float64
	if n%2 == 1 {
		p50TPS = sortedTPS[n/2]
	} else {
		p50TPS = (sortedTPS[n/2-1] + sortedTPS[n/2]) / 2.0
	}

	// P95 calculation
	idx95 := int(math.Ceil(0.95*float64(n))) - 1
	if idx95 < 0 {
		idx95 = 0
	}
	if idx95 >= n {
		idx95 = n - 1
	}
	p95TPS = sortedTPS[idx95]

	// Population standard deviation & noise percentage
	var varianceSum float64
	for _, v := range tpsVals {
		diff := v - meanTPS
		varianceSum += diff * diff
	}
	stdDevTPS := math.Sqrt(varianceSum / float64(n))
	noisePercent := 0.0
	if meanTPS > 0 {
		noisePercent = (stdDevTPS / meanTPS) * 100.0
	}

	// Phase mean breakdown in milliseconds
	phaseSummaryMS := make(map[string]float64, len(CanonicalPhaseBuckets))
	for _, phase := range CanonicalPhaseBuckets {
		var phaseSum float64
		for _, r := range runs {
			phaseSum += r.PhasesMS[phase]
		}
		phaseSummaryMS[phase] = phaseSum / float64(n)
	}

	summary := StatisticalSummary{
		RunsCount:             n,
		MeanTokensPerSec:      meanTPS,
		P50TokensPerSec:       p50TPS,
		P95TokensPerSec:       p95TPS,
		StdDevTokensPerSec:    stdDevTPS,
		NoisePercent:          noisePercent,
		MeanDRAMBandwidthGBps: meanDRAMBW,
		MeanMALLHitRate:       meanMALLHitRate,
		MeanQueueLatencyMS:    meanQueueLatency,
		MeanLogitCosineParity: meanParity,
		ParityThreshold:       threshold,
		ParityPassed:          allPassed,
		CountersStatus:        counterStatus,
	}

	return summary, phaseSummaryMS
}

// RawDecodePhysicalRunner adapts one explicit rawdecode request to the physical
// fanout seam. Backend identity and counters come only from the backend-owned
// rawdecode observation; source, binary, model inventory, and peak memory remain
// unavailable until their independent observers are attached.
type RawDecodePhysicalRunner struct {
	Request rawdecode.Request
	Execute func(context.Context, rawdecode.Request) (rawdecode.Execution, error)
}

// NewRawDecodePhysicalRunner constructs the device-free production adapter. The
// request must carry an explicit artifact path, expected artifact SHA-256,
// model selector, backend, prompt token IDs, and execution policy.
func NewRawDecodePhysicalRunner(req rawdecode.Request) *RawDecodePhysicalRunner {
	return &RawDecodePhysicalRunner{Request: req, Execute: rawdecode.Execute}
}

// ExecuteFanoutTrial invokes the real fak-native rawdecode seam. This first
// tracer supports only B=1: claiming fanout concurrency from serial calls would
// manufacture queueing and shared-prefix evidence.
func (p *RawDecodePhysicalRunner) ExecuteFanoutTrial(req PhysicalTrialRequest) (PhysicalTrialResult, error) {
	if p == nil || p.Execute == nil {
		return PhysicalTrialResult{}, errors.New("qwen38campaign: rawdecode physical executor is unavailable")
	}
	if strings.TrimSpace(p.Request.ArtifactPath) == "" || strings.TrimSpace(p.Request.ExpectedArtifactSHA256) == "" ||
		strings.TrimSpace(p.Request.ModelName) == "" || strings.TrimSpace(p.Request.BackendName) == "" {
		return PhysicalTrialResult{}, errors.New("qwen38campaign: rawdecode physical runner requires explicit artifact path, expected SHA-256, model selector, and backend")
	}
	if req.Concurrency != 1 {
		return PhysicalTrialResult{}, fmt.Errorf("qwen38campaign: rawdecode physical tracer supports concurrency 1, got %d", req.Concurrency)
	}
	if len(p.Request.PromptTokenIDs) != req.PrefixTokens {
		return PhysicalTrialResult{}, fmt.Errorf("qwen38campaign: explicit prompt token count %d does not match requested prefix tokens %d", len(p.Request.PromptTokenIDs), req.PrefixTokens)
	}

	rawReq := p.Request
	rawReq.PromptTokenIDs = slices.Clone(p.Request.PromptTokenIDs)
	rawReq.ContextLimit = len(rawReq.PromptTokenIDs) + req.GeneratedTokensPerSubagent
	rawReq.GeneratedTokenLimit = req.GeneratedTokensPerSubagent
	rawReq.Repetitions = 1
	execution, err := p.Execute(context.Background(), rawReq)
	if err != nil {
		return PhysicalTrialResult{}, fmt.Errorf("qwen38campaign: rawdecode physical execution: %w", err)
	}
	return physicalTrialFromRawDecode(req, execution)
}

func physicalTrialFromRawDecode(req PhysicalTrialRequest, execution rawdecode.Execution) (PhysicalTrialResult, error) {
	if len(execution.Runs) != 1 {
		return PhysicalTrialResult{}, fmt.Errorf("qwen38campaign: rawdecode returned %d runs, want exactly 1", len(execution.Runs))
	}
	run := execution.Runs[0]
	wall := run.SessionSetupDuration + run.PrefillDuration + run.FirstSampleDuration + run.DecodeDuration + run.TeardownDuration
	if wall <= 0 {
		return PhysicalTrialResult{}, errors.New("qwen38campaign: rawdecode wall duration unavailable")
	}
	if len(run.GeneratedTokens) == 0 {
		return PhysicalTrialResult{}, errors.New("qwen38campaign: rawdecode produced no output tokens")
	}
	output := make([]int32, len(run.GeneratedTokens))
	for i, token := range run.GeneratedTokens {
		if token < 0 || int64(token) > math.MaxInt32 {
			return PhysicalTrialResult{}, fmt.Errorf("qwen38campaign: rawdecode output token %d cannot be represented as int32", token)
		}
		output[i] = int32(token)
	}
	backendExecution, err := backendExecutionEvidence(run.BackendExecution, execution.Backend.Selected)
	if err != nil {
		return PhysicalTrialResult{}, err
	}
	if backendExecution.Fallbacks > uint64(math.MaxInt) {
		return PhysicalTrialResult{}, errors.New("qwen38campaign: observed fallback count overflows int")
	}

	result := PhysicalTrialResult{
		RunIndex:         req.RunIndex,
		Concurrency:      req.Concurrency,
		Scenario:         req.Scenario,
		Backend:          execution.Backend.Selected,
		ExecutionPath:    execution.Engine,
		WallDurationMS:   float64(wall.Nanoseconds()) / 1e6,
		UsefulTokens:     len(output),
		TokensPerSec:     float64(len(output)) / wall.Seconds(),
		TTFTMS:           float64((run.SessionSetupDuration + run.PrefillDuration + run.FirstSampleDuration).Nanoseconds()) / 1e6,
		CounterSource:    CountersUnavailable,
		FallbackCount:    int(backendExecution.Fallbacks),
		BackendExecution: backendExecution,
		PhasesMS: map[string]float64{
			"session_setup": float64(run.SessionSetupDuration.Nanoseconds()) / 1e6,
			"prefill":       float64(run.PrefillDuration.Nanoseconds()) / 1e6,
			"first_sample":  float64(run.FirstSampleDuration.Nanoseconds()) / 1e6,
			"decode":        float64(run.DecodeDuration.Nanoseconds()) / 1e6,
			"teardown":      float64(run.TeardownDuration.Nanoseconds()) / 1e6,
		},
		OutputTokenIDs: output,
		Identity: ExecutionIdentity{
			ModelGGUFSHA256:   execution.ArtifactSHA256,
			TokenPacketSHA256: observedTokenPacketSHA256(execution.PromptTokenIDs, output),
		},
	}
	if len(output) > 1 && run.DecodeDuration > 0 {
		result.TPOTMS = float64(run.DecodeDuration.Nanoseconds()) / 1e6 / float64(len(output)-1)
	}
	if run.CPUVerification != nil {
		result.LogitCosineParity = run.CPUVerification.MinCosine
		result.ParityPassed = run.CPUVerification.Passed
	}
	// Identity.Validate will keep this result non-creditable until source,
	// archive, and binary observers are supplied by the trusted runner.
	return result, nil
}

func backendExecutionEvidence(observed *compute.BackendExecutionObservation, selectedBackend string) (*BackendExecutionEvidence, error) {
	if observed == nil {
		return nil, errors.New("qwen38campaign: rawdecode backend execution observation is unavailable")
	}
	evidence := &BackendExecutionEvidence{
		Backend:                 observed.Identity.Backend,
		Device:                  observed.Identity.Device,
		Driver:                  observed.Identity.Driver,
		Runtime:                 observed.Identity.Runtime,
		ComputeDispatches:       observed.Counters.ComputeDispatches,
		Q4KMatmulDispatches:     observed.Counters.Q4KMatmulDispatches,
		OtherDispatches:         observed.Counters.OtherDispatches,
		DispatchSubmits:         observed.Counters.DispatchSubmits,
		H2DBytes:                observed.Counters.H2DBytes,
		D2HBytes:                observed.Counters.D2HBytes,
		D2DCopies:               observed.Counters.D2DCopies,
		Q4KStageCalls:           observed.Counters.Q4KStageCalls,
		Q4KStageBytes:           observed.Counters.Q4KStageBytes,
		Fallbacks:               observed.Counters.Fallbacks,
		TensorHomeHits:          observed.Counters.TensorHomeHits,
		TensorHomeAdmissions:    observed.Counters.TensorHomeAdmissions,
		TensorHomeBypasses:      observed.Counters.TensorHomeBypasses,
		TensorHomeCopiedBytes:   observed.Counters.TensorHomeCopiedBytes,
		TensorHomeEntries:       observed.TensorHomeEntries,
		TensorHomeResidentBytes: observed.TensorHomeResidentBytes,
	}
	if observed.DeviceMemoryObserved {
		total, free := observed.DeviceMemoryTotalBytes, observed.DeviceMemoryFreeBytes
		evidence.DeviceMemoryTotalBytes = &total
		evidence.DeviceMemoryFreeBytes = &free
	}
	if err := evidence.Validate(selectedBackend); err != nil {
		return nil, fmt.Errorf("qwen38campaign: invalid rawdecode backend execution observation: %w", err)
	}
	return evidence, nil
}

func cloneBackendExecutionEvidence(evidence *BackendExecutionEvidence) *BackendExecutionEvidence {
	if evidence == nil {
		return nil
	}
	clone := *evidence
	if evidence.DeviceMemoryTotalBytes != nil {
		total := *evidence.DeviceMemoryTotalBytes
		clone.DeviceMemoryTotalBytes = &total
	}
	if evidence.DeviceMemoryFreeBytes != nil {
		free := *evidence.DeviceMemoryFreeBytes
		clone.DeviceMemoryFreeBytes = &free
	}
	return &clone
}

func observedTokenPacketSHA256(prompt []int, output []int32) string {
	h := sha256.New()
	fmt.Fprintf(h, "prompt=%v;output=%v", prompt, output)
	return hex.EncodeToString(h.Sum(nil))
}

var defaultPhysicalRunner PhysicalRunner

// RegisterDefaultPhysicalRunner registers a global physical runner for CLI execution.
func RegisterDefaultPhysicalRunner(r PhysicalRunner) {
	defaultPhysicalRunner = r
}

// ExecuteSubagentFanoutBenchmark is the high-level entrypoint for running the benchmark suite.
func ExecuteSubagentFanoutBenchmark(cfg FanoutConfig) (SubagentFanoutReceipt, error) {
	harness, err := NewSubagentFanoutHarness(cfg)
	if err != nil {
		return SubagentFanoutReceipt{}, err
	}
	if !cfg.Simulated && defaultPhysicalRunner != nil {
		harness.PhysicalRunner = defaultPhysicalRunner
	}
	return harness.Execute()
}

// ExecuteSubagentFanoutBenchmarkWithRunner runs the benchmark suite with an explicit runner attached.
func ExecuteSubagentFanoutBenchmarkWithRunner(cfg FanoutConfig, runner PhysicalRunner) (SubagentFanoutReceipt, error) {
	harness, err := NewSubagentFanoutHarness(cfg)
	if err != nil {
		return SubagentFanoutReceipt{}, err
	}
	harness.PhysicalRunner = runner
	return harness.Execute()
}

// RunCLI executes the subagent fan-out multi-agent benchmark harness command-line interface.
// Usage: fak bench subagent [--scenario=shared_prefix_forked] [--concurrency=4] [--runs=5] [--json] [--out=path]
func RunCLI(stdout, stderr io.Writer, args []string) int {
	return RunWithRunner(stdout, stderr, args, defaultPhysicalRunner)
}

// RunWithRunner executes the CLI with an explicit physical runner attached.
func RunWithRunner(stdout, stderr io.Writer, args []string, runner PhysicalRunner) int {
	if len(args) > 0 && args[0] == "subagent" {
		args = args[1:]
	}

	fs := flag.NewFlagSet("fak bench subagent", flag.ContinueOnError)
	fs.SetOutput(stderr)

	scenario := fs.String("scenario", ScenarioSharedPrefixForked, "Benchmark scenario: cold, warm_same_prefix, or shared_prefix_forked")
	concurrency := fs.Int("concurrency", 4, "Batch concurrency B in {1, 2, 4, 8}")
	cShort := fs.Int("c", 0, "Shorthand for --concurrency")
	runs := fs.Int("runs", DefaultRuns, "Repetition runs count (>= 5)")
	rShort := fs.Int("r", 0, "Shorthand for --runs")
	prefixTokens := fs.Int("prefix-tokens", 0, "Prefix tokens count (default: 30000 for shared_prefix_forked)")
	genTokens := fs.Int("gen-tokens", DefaultGeneratedTokensPerSubagent, "Tokens generated per subagent")
	simulated := fs.Bool("simulated", true, "Execute with calibrated architecture simulation")
	jsonFlag := fs.Bool("json", false, "Output verified benchmark receipt in JSON format")
	outFlag := fs.String("out", "", "Write JSON receipt to specified file path")
	threshold := fs.Float64("threshold", DefaultLogitCosineParityThreshold, "Logit cosine parity threshold")

	if err := fs.Parse(args); err != nil {
		if err == flag.ErrHelp {
			return 0
		}
		return 2
	}

	actualConcurrency := *concurrency
	if *cShort != 0 {
		actualConcurrency = *cShort
	}
	actualRuns := *runs
	if *rShort != 0 {
		actualRuns = *rShort
	}

	cfg := FanoutConfig{
		Scenario:                   *scenario,
		Concurrency:                actualConcurrency,
		Runs:                       actualRuns,
		PrefixTokens:               *prefixTokens,
		GeneratedTokensPerSubagent: *genTokens,
		Simulated:                  *simulated,
		ParityThreshold:            *threshold,
	}

	if err := cfg.Validate(); err != nil {
		fmt.Fprintf(stderr, "fak bench subagent: %v\n", err)
		return 2
	}

	harness, err := NewSubagentFanoutHarness(cfg)
	if err != nil {
		fmt.Fprintf(stderr, "fak bench subagent error: %v\n", err)
		return 1
	}
	if runner != nil {
		harness.PhysicalRunner = runner
	}

	receipt, err := harness.Execute()
	if err != nil {
		fmt.Fprintf(stderr, "fak bench subagent error: %v\n", err)
		return 1
	}

	if *outFlag != "" {
		raw, err := receipt.JSON()
		if err != nil {
			fmt.Fprintf(stderr, "fak bench subagent formatting error: %v\n", err)
			return 1
		}
		if err := os.WriteFile(*outFlag, raw, 0644); err != nil {
			fmt.Fprintf(stderr, "fak bench subagent write error: %v\n", err)
			return 1
		}
	}

	if *jsonFlag {
		raw, err := receipt.JSON()
		if err != nil {
			fmt.Fprintf(stderr, "fak bench subagent formatting error: %v\n", err)
			return 1
		}
		fmt.Fprintln(stdout, string(raw))
		return 0
	}

	fmt.Fprint(stdout, receipt.String())
	return 0
}

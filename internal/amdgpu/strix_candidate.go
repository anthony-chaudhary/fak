package amdgpu

import (
	"fmt"
	"math"
	"strings"
	"sync"
	"time"
)

// Canonical candidate IDs for Strix Halo performance optimizations.
const (
	CandidateIDTargetQ4KGEMV   = "target.q4k_gemv"
	CandidateIDTopologyNormMM  = "topology.norm_matmul"
	CandidateIDQuantQ4KvsF32   = "quant.q4k_vs_f32"
	CandidateIDResidencyDevLoc = "residency.device_local"
	CandidateIDLayoutF16Contig = "layout.f16_contiguize"
)

// Default thresholds for candidate promotion and classification.
const (
	DefaultMinParity        = 0.999900 // Minimum cosine parity against reference
	DefaultNoiseBand        = 0.05     // 5% noise tolerance band (0.05)
	DefaultSpeedupThreshold = 1.05     // Minimum speedup ratio for promotion (5% lift)
)

// StrixCandidateVerdict represents the outcome classification of a candidate evaluation.
type StrixCandidateVerdict string

const (
	VerdictPromoted  StrixCandidateVerdict = "PROMOTED"  // Speedup >= threshold, parity >= 0.999900, noise <= 5%
	VerdictNeutral   StrixCandidateVerdict = "NEUTRAL"   // Within noise band or noise too high
	VerdictRegressed StrixCandidateVerdict = "REGRESSED" // Slower or parity violated
)

// StrixCandidateBaseline represents a pinned reference baseline for a specific candidate optimization.
type StrixCandidateBaseline struct {
	CandidateID      string         `json:"candidate_id"`
	Dimension        string         `json:"dimension"`
	Feature          string         `json:"feature"`
	Description      string         `json:"description,omitempty"`
	BaselineArm      StrixArmResult `json:"baseline_arm"`      // Pinned control reference
	PinnedCandidate  StrixArmResult `json:"pinned_candidate"`  // Pinned treatment reference
	SpeedupThreshold float64        `json:"speedup_threshold"` // Minimum speedup ratio for promotion (default 1.05)
	MinParity        float64        `json:"min_parity"`        // Minimum cosine parity (default 0.999900)
	NoiseBand        float64        `json:"noise_band"`        // Noise band ratio (default 0.05)
}

// ReferenceSpeedup returns the speedup ratio of the pinned candidate against the baseline arm.
func (b *StrixCandidateBaseline) ReferenceSpeedup() float64 {
	if b.PinnedCandidate.LatencyUS <= 0 || b.BaselineArm.LatencyUS <= 0 {
		return 0
	}
	return float64(b.BaselineArm.LatencyUS) / float64(b.PinnedCandidate.LatencyUS)
}

// ReferenceCompressionRatio returns the memory compression ratio of baseline bytes over candidate bytes.
func (b *StrixCandidateBaseline) ReferenceCompressionRatio() float64 {
	if b.PinnedCandidate.AllocatedBytes <= 0 || b.BaselineArm.AllocatedBytes <= 0 {
		return 1.0
	}
	return float64(b.BaselineArm.AllocatedBytes) / float64(b.PinnedCandidate.AllocatedBytes)
}

// StrixCandidateComparison compares a candidate run against the baseline.
type StrixCandidateComparison struct {
	CandidateID             string                `json:"candidate_id"`
	Dimension               string                `json:"dimension"`
	Feature                 string                `json:"feature"`
	BaselineArmName         string                `json:"baseline_arm_name"`
	CandidateArmName        string                `json:"candidate_arm_name"`
	BaselineLatencyUS       int64                 `json:"baseline_latency_us"`
	CandidateLatencyUS      int64                 `json:"candidate_latency_us"`
	LatencyDeltaUS          int64                 `json:"latency_delta_us"` // T_candidate - T_baseline (< 0 indicates faster)
	Speedup                 float64               `json:"speedup"`          // T_baseline / T_candidate
	BaselineThroughputTokS  float64               `json:"baseline_throughput_tok_s"`
	CandidateThroughputTokS float64               `json:"candidate_throughput_tok_s"`
	ThroughputDelta         float64               `json:"throughput_delta"` // R_candidate - R_baseline (> 0 indicates higher throughput)
	LiftRatio               float64               `json:"lift_ratio"`       // R_candidate / R_baseline
	BaselineAllocatedBytes  int64                 `json:"baseline_allocated_bytes"`
	CandidateAllocatedBytes int64                 `json:"candidate_allocated_bytes"`
	AllocatedBytesDelta     int64                 `json:"allocated_bytes_delta"` // M_candidate - M_baseline (< 0 indicates memory savings)
	CompressionRatio        float64               `json:"compression_ratio"`     // M_baseline / M_candidate
	CosineParity            float64               `json:"cosine_parity"`         // Numerical parity against reference
	NoiseRatio              float64               `json:"noise_ratio"`           // Measurement noise/variance ratio (e.g. 0.02 = 2%)
	Verdict                 StrixCandidateVerdict `json:"verdict"`               // PROMOTED | NEUTRAL | REGRESSED
	Reason                  string                `json:"reason"`                // Detailed classification justification
	EvaluatedAt             time.Time             `json:"evaluated_at"`
}

// CandidateEvalOption specifies functional options when evaluating a candidate.
type CandidateEvalOption func(*candidateEvalConfig)

type candidateEvalConfig struct {
	noiseRatio       float64
	speedupThreshold float64
	minParity        float64
	noiseBand        float64
}

// WithNoise configures the measurement noise/variance ratio for the candidate evaluation.
func WithNoise(noiseRatio float64) CandidateEvalOption {
	return func(c *candidateEvalConfig) {
		c.noiseRatio = noiseRatio
	}
}

// WithSpeedupThreshold overrides the baseline's promotion speedup threshold.
func WithSpeedupThreshold(threshold float64) CandidateEvalOption {
	return func(c *candidateEvalConfig) {
		c.speedupThreshold = threshold
	}
}

// WithMinParity overrides the baseline's minimum cosine parity.
func WithMinParity(minParity float64) CandidateEvalOption {
	return func(c *candidateEvalConfig) {
		c.minParity = minParity
	}
}

// StrixCandidateRegistry indexes candidate baselines and maintains evaluation scoreboards.
type StrixCandidateRegistry struct {
	mu             sync.RWMutex
	baselines      map[string]*StrixCandidateBaseline
	aliases        map[string]string // alias -> candidateID
	canonicalOrder []string
	scoreboard     map[string]*StrixCandidateComparison
}

// NewStrixCandidateRegistry creates and pre-seeds the registry with the 5 canonical baselines
// from docs/benchmarks/strix-halo-validation-11940.json.
func NewStrixCandidateRegistry() *StrixCandidateRegistry {
	r := &StrixCandidateRegistry{
		baselines:  make(map[string]*StrixCandidateBaseline),
		aliases:    make(map[string]string),
		scoreboard: make(map[string]*StrixCandidateComparison),
	}

	r.seedCanonicalBaselines()
	return r
}

// seedCanonicalBaselines initializes the 5 canonical baselines and their initial comparison states.
func (r *StrixCandidateRegistry) seedCanonicalBaselines() {
	canonical := []struct {
		baseline StrixCandidateBaseline
		aliases  []string
	}{
		// 1. target.q4k_gemv (baseline: cpu_q4_reference 75.56ms, candidate: vulkan_gpu_q4k 451µs)
		{
			baseline: StrixCandidateBaseline{
				CandidateID: CandidateIDTargetQ4KGEMV,
				Dimension:   "target",
				Feature:     "cpu_vs_vulkan_gpu",
				Description: "Q4_K GEMV CPU oracle reference vs AMD Radeon 8060S Vulkan compute dispatch",
				BaselineArm: StrixArmResult{
					Name:           "cpu_q4_reference",
					LatencyUS:      75561,
					AllocatedBytes: 50135040,
				},
				PinnedCandidate: StrixArmResult{
					Name:            "vulkan_gpu_q4k",
					LatencyUS:       451,
					DRAMBandwidthGB: 117.1,
					AllocatedBytes:  50135040,
				},
				SpeedupThreshold: DefaultSpeedupThreshold,
				MinParity:        DefaultMinParity,
				NoiseBand:        DefaultNoiseBand,
			},
			aliases: []string{
				"target.q4k_gemv",
				"cpu_vs_vulkan_gpu",
				"target.cpu_vs_vulkan_gpu",
				"q4k_gemv",
				"vulkan_gpu_q4k",
			},
		},
		// 2. topology.norm_matmul (baseline: discrete_rmsnorm_then_matmul 28.28ms, candidate: fused_rmsnorm_matmul 17.40ms)
		{
			baseline: StrixCandidateBaseline{
				CandidateID: CandidateIDTopologyNormMM,
				Dimension:   "topology",
				Feature:     "fused_vs_discrete_norm_matmul",
				Description: "Fused RMSNormMatMul vs chained RMSNorm + MatMul",
				BaselineArm: StrixArmResult{
					Name:      "discrete_rmsnorm_then_matmul",
					LatencyUS: 28275,
				},
				PinnedCandidate: StrixArmResult{
					Name:      "fused_rmsnorm_matmul",
					LatencyUS: 17400,
				},
				SpeedupThreshold: DefaultSpeedupThreshold,
				MinParity:        DefaultMinParity,
				NoiseBand:        DefaultNoiseBand,
			},
			aliases: []string{
				"topology.norm_matmul",
				"fused_vs_discrete_norm_matmul",
				"topology.fused_vs_discrete_norm_matmul",
				"norm_matmul",
				"fused_rmsnorm_matmul",
			},
		},
		// 3. quant.q4k_vs_f32 (baseline: f32_dense_weights 1820µs, candidate: q4k_super_blocks 428µs)
		{
			baseline: StrixCandidateBaseline{
				CandidateID: CandidateIDQuantQ4KvsF32,
				Dimension:   "quantization",
				Feature:     "quant_q4k_vs_q8_vs_f32",
				Description: "Precision & memory footprint comparison across F32, Q8_0, and Q4_K",
				BaselineArm: StrixArmResult{
					Name:           "f32_dense_weights",
					LatencyUS:      1820,
					AllocatedBytes: 356515840,
				},
				PinnedCandidate: StrixArmResult{
					Name:           "q4k_super_blocks",
					LatencyUS:      428,
					AllocatedBytes: 50135040,
				},
				SpeedupThreshold: DefaultSpeedupThreshold,
				MinParity:        DefaultMinParity,
				NoiseBand:        DefaultNoiseBand,
			},
			aliases: []string{
				"quant.q4k_vs_f32",
				"quant_q4k_vs_q8_vs_f32",
				"quantization.quant_q4k_vs_q8_vs_f32",
				"q4k_vs_f32",
				"q4k_super_blocks",
			},
		},
		// 4. residency.device_local (baseline: host_visible_streaming 1420µs, candidate: device_local_pool 428µs)
		{
			baseline: StrixCandidateBaseline{
				CandidateID: CandidateIDResidencyDevLoc,
				Dimension:   "residency",
				Feature:     "device_local_vs_host_visible",
				Description: "VRAM/GTT resident tensor vs host-visible streaming over UMA bus",
				BaselineArm: StrixArmResult{
					Name:           "host_visible_streaming",
					LatencyUS:      1420,
					AllocatedBytes: 50135040,
				},
				PinnedCandidate: StrixArmResult{
					Name:           "device_local_pool",
					LatencyUS:      428,
					AllocatedBytes: 50135040,
				},
				SpeedupThreshold: DefaultSpeedupThreshold,
				MinParity:        DefaultMinParity,
				NoiseBand:        DefaultNoiseBand,
			},
			aliases: []string{
				"residency.device_local",
				"device_local_vs_host_visible",
				"residency.device_local_vs_host_visible",
				"device_local",
				"device_local_pool",
			},
		},
		// 5. layout.f16_contiguize (baseline: strided_f16_kv_camping 44.87ms, candidate: contiguized_f16_kv_scratch 16.68ms)
		{
			baseline: StrixCandidateBaseline{
				CandidateID: CandidateIDLayoutF16Contig,
				Dimension:   "layout",
				Feature:     "strided_vs_contiguized_f16_kv",
				Description: "Strided f16 KV cache (channel camping) vs head-contiguized scratch transposition",
				BaselineArm: StrixArmResult{
					Name:            "strided_f16_kv_camping",
					LatencyUS:       44869,
					DRAMBandwidthGB: 28.4,
					AllocatedBytes:  67108864,
				},
				PinnedCandidate: StrixArmResult{
					Name:            "contiguized_f16_kv_scratch",
					LatencyUS:       16680,
					DRAMBandwidthGB: 184.2,
					AllocatedBytes:  134217728,
				},
				SpeedupThreshold: DefaultSpeedupThreshold,
				MinParity:        DefaultMinParity,
				NoiseBand:        DefaultNoiseBand,
			},
			aliases: []string{
				"layout.f16_contiguize",
				"strided_vs_contiguized_f16_kv",
				"layout.strided_vs_contiguized_f16_kv",
				"f16_contiguize",
				"contiguized_f16_kv_scratch",
			},
		},
	}

	canonicalParities := map[string]float64{
		CandidateIDTargetQ4KGEMV:   0.9999999999986565,
		CandidateIDTopologyNormMM:  0.999999,
		CandidateIDQuantQ4KvsF32:   0.999998,
		CandidateIDResidencyDevLoc: 1.0,
		CandidateIDLayoutF16Contig: 1.0,
	}

	for _, entry := range canonical {
		b := entry.baseline
		r.baselines[b.CandidateID] = &b
		r.canonicalOrder = append(r.canonicalOrder, b.CandidateID)

		for _, alias := range entry.aliases {
			r.aliases[strings.ToLower(strings.TrimSpace(alias))] = b.CandidateID
		}

		parity := canonicalParities[b.CandidateID]
		initialResult := StrixAblationResult{
			Dimension:    b.Dimension,
			Feature:      b.Feature,
			BaselineArm:  b.BaselineArm,
			CandidateArm: b.PinnedCandidate,
			Speedup:      b.ReferenceSpeedup(),
			LiftRatio:    b.ReferenceSpeedup(),
			CosineParity: parity,
			Verdict:      "VERIFIED_LIFT",
		}
		comp, _ := r.evaluateCandidateInternal(&b, initialResult, candidateEvalConfig{
			noiseRatio:       0.0,
			speedupThreshold: b.SpeedupThreshold,
			minParity:        b.MinParity,
			noiseBand:        b.NoiseBand,
		})
		r.scoreboard[b.CandidateID] = comp
	}
}

// RegisterBaseline adds or updates a candidate baseline in the registry.
func (r *StrixCandidateRegistry) RegisterBaseline(baseline StrixCandidateBaseline) error {
	if strings.TrimSpace(baseline.CandidateID) == "" {
		return fmt.Errorf("amdgpu: candidate baseline must specify a non-empty CandidateID")
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	bCopy := baseline
	if bCopy.SpeedupThreshold <= 0 {
		bCopy.SpeedupThreshold = DefaultSpeedupThreshold
	}
	if bCopy.MinParity <= 0 {
		bCopy.MinParity = DefaultMinParity
	}
	if bCopy.NoiseBand <= 0 {
		bCopy.NoiseBand = DefaultNoiseBand
	}

	id := bCopy.CandidateID
	if _, exists := r.baselines[id]; !exists {
		r.canonicalOrder = append(r.canonicalOrder, id)
	}
	r.baselines[id] = &bCopy
	r.aliases[strings.ToLower(id)] = id
	if bCopy.Feature != "" {
		r.aliases[strings.ToLower(bCopy.Feature)] = id
		if bCopy.Dimension != "" {
			r.aliases[strings.ToLower(bCopy.Dimension+"."+bCopy.Feature)] = id
		}
	}
	if bCopy.PinnedCandidate.Name != "" {
		r.aliases[strings.ToLower(bCopy.PinnedCandidate.Name)] = id
	}

	return nil
}

// GetBaseline retrieves a baseline by candidateID or registered alias.
func (r *StrixCandidateRegistry) GetBaseline(candidateID string) (*StrixCandidateBaseline, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	base, ok := r.findBaselineLocked(candidateID, "", "")
	if !ok {
		return nil, false
	}
	copyBase := *base
	return &copyBase, true
}

// EvaluateCandidate evaluates an ablation result against its corresponding baseline,
// computes comparative metrics, determines the verdict, updates the scoreboard, and returns the comparison.
func (r *StrixCandidateRegistry) EvaluateCandidate(result StrixAblationResult) (*StrixCandidateComparison, error) {
	return r.EvaluateCandidateWithOptions(result)
}

// EvaluateCandidateWithOptions evaluates an ablation result with optional override configuration.
func (r *StrixCandidateRegistry) EvaluateCandidateWithOptions(result StrixAblationResult, opts ...CandidateEvalOption) (*StrixCandidateComparison, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	baseline, ok := r.findBaselineLocked(result.Feature, result.Dimension, result.CandidateArm.Name)
	if !ok {
		return nil, fmt.Errorf("amdgpu: candidate feature %q (dimension %q, arm %q) not found in registry",
			result.Feature, result.Dimension, result.CandidateArm.Name)
	}

	cfg := candidateEvalConfig{
		noiseRatio:       0.0,
		speedupThreshold: baseline.SpeedupThreshold,
		minParity:        baseline.MinParity,
		noiseBand:        baseline.NoiseBand,
	}
	if cfg.speedupThreshold <= 0 {
		cfg.speedupThreshold = DefaultSpeedupThreshold
	}
	if cfg.minParity <= 0 {
		cfg.minParity = DefaultMinParity
	}
	if cfg.noiseBand <= 0 {
		cfg.noiseBand = DefaultNoiseBand
	}

	for _, opt := range opts {
		if opt != nil {
			opt(&cfg)
		}
	}

	comp, err := r.evaluateCandidateInternal(baseline, result, cfg)
	if err != nil {
		return nil, err
	}

	r.scoreboard[baseline.CandidateID] = comp
	return comp, nil
}

// EvaluateReceipt evaluates all ablations contained in a validation receipt against the registry.
func (r *StrixCandidateRegistry) EvaluateReceipt(receipt *StrixValidationReceipt) ([]StrixCandidateComparison, error) {
	if receipt == nil {
		return nil, fmt.Errorf("amdgpu: receipt is nil")
	}

	var comparisons []StrixCandidateComparison
	for _, ab := range receipt.Ablations {
		comp, err := r.EvaluateCandidate(ab)
		if err != nil {
			continue // Skip unindexed ablations
		}
		comparisons = append(comparisons, *comp)
	}
	return comparisons, nil
}

// Scoreboard returns all current candidate comparisons in canonical order.
func (r *StrixCandidateRegistry) Scoreboard() []StrixCandidateComparison {
	r.mu.RLock()
	defer r.mu.RUnlock()

	out := make([]StrixCandidateComparison, 0, len(r.canonicalOrder))
	for _, id := range r.canonicalOrder {
		if comp, exists := r.scoreboard[id]; exists {
			out = append(out, *comp)
		}
	}
	return out
}

// GetComparison retrieves the latest comparison for a specific candidate.
func (r *StrixCandidateRegistry) GetComparison(candidateID string) (*StrixCandidateComparison, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	targetID := candidateID
	if mapped, exists := r.aliases[strings.ToLower(strings.TrimSpace(candidateID))]; exists {
		targetID = mapped
	}

	comp, ok := r.scoreboard[targetID]
	if !ok {
		return nil, false
	}
	copyComp := *comp
	return &copyComp, true
}

func (r *StrixCandidateRegistry) findBaselineLocked(feature, dimension, armName string) (*StrixCandidateBaseline, bool) {
	// 1. Direct ID lookup
	if base, ok := r.baselines[feature]; ok {
		return base, true
	}

	// 2. Alias lookup
	cleanFeature := strings.ToLower(strings.TrimSpace(feature))
	if mappedID, ok := r.aliases[cleanFeature]; ok {
		if base, ok := r.baselines[mappedID]; ok {
			return base, true
		}
	}

	// 3. Dimension-qualified feature lookup
	if dimension != "" {
		qualified := strings.ToLower(strings.TrimSpace(dimension + "." + feature))
		if mappedID, ok := r.aliases[qualified]; ok {
			if base, ok := r.baselines[mappedID]; ok {
				return base, true
			}
		}
		if base, ok := r.baselines[qualified]; ok {
			return base, true
		}
	}

	// 4. Candidate arm name alias lookup
	if armName != "" {
		cleanArm := strings.ToLower(strings.TrimSpace(armName))
		if mappedID, ok := r.aliases[cleanArm]; ok {
			if base, ok := r.baselines[mappedID]; ok {
				return base, true
			}
		}
	}

	// 5. Scan baselines for feature or arm match
	for _, b := range r.baselines {
		if strings.EqualFold(b.Feature, feature) ||
			strings.EqualFold(b.PinnedCandidate.Name, armName) ||
			strings.EqualFold(b.CandidateID, feature) {
			return b, true
		}
	}

	return nil, false
}

func (r *StrixCandidateRegistry) evaluateCandidateInternal(
	baseline *StrixCandidateBaseline,
	result StrixAblationResult,
	cfg candidateEvalConfig,
) (*StrixCandidateComparison, error) {
	candArm := result.CandidateArm
	candLatency := candArm.LatencyUS

	// If baseline arm is provided in result, prefer it; otherwise fall back to pinned baseline arm.
	baseArm := result.BaselineArm
	if baseArm.LatencyUS <= 0 {
		baseArm = baseline.BaselineArm
	}
	baseLatency := baseArm.LatencyUS

	if candLatency <= 0 {
		return &StrixCandidateComparison{
			CandidateID:        baseline.CandidateID,
			Dimension:          baseline.Dimension,
			Feature:            baseline.Feature,
			BaselineArmName:    baseArm.Name,
			CandidateArmName:   candArm.Name,
			BaselineLatencyUS:  baseLatency,
			CandidateLatencyUS: candLatency,
			Verdict:            VerdictRegressed,
			Reason:             fmt.Sprintf("invalid non-positive candidate latency: %dµs", candLatency),
			EvaluatedAt:        time.Now().UTC(),
		}, nil
	}

	// 1. Latency delta & speedup ratio (T_baseline / T_candidate)
	latencyDelta := candLatency - baseLatency
	speedup := 0.0
	if baseLatency > 0 {
		speedup = float64(baseLatency) / float64(candLatency)
	} else if result.Speedup > 0 {
		speedup = result.Speedup
	}

	// 2. Throughput delta & lift ratio (R_candidate / R_baseline)
	candThroughput := candArm.ThroughputTokS
	baseThroughput := baseArm.ThroughputTokS
	if baseThroughput <= 0 && baseline.BaselineArm.ThroughputTokS > 0 {
		baseThroughput = baseline.BaselineArm.ThroughputTokS
	}

	throughputDelta := candThroughput - baseThroughput
	liftRatio := 0.0
	if candThroughput > 0 && baseThroughput > 0 {
		liftRatio = candThroughput / baseThroughput
	} else if result.LiftRatio > 0 {
		liftRatio = result.LiftRatio
	} else {
		liftRatio = speedup
	}

	// 3. Memory allocation delta & compression ratio (M_baseline / M_candidate)
	candAlloc := candArm.AllocatedBytes
	baseAlloc := baseArm.AllocatedBytes
	if baseAlloc <= 0 && baseline.BaselineArm.AllocatedBytes > 0 {
		baseAlloc = baseline.BaselineArm.AllocatedBytes
	}

	allocDelta := candAlloc - baseAlloc
	compressionRatio := 1.0
	if candAlloc > 0 && baseAlloc > 0 {
		compressionRatio = float64(baseAlloc) / float64(candAlloc)
	}

	// 4. Numerical parity against reference
	cosineParity := result.CosineParity

	// 5. Verdict classification:
	// - PROMOTED: speedup >= threshold, parity >= 0.999900, noise <= 5%
	// - NEUTRAL: within noise band [1.0 - noiseBand, threshold) or noise > 5%
	// - REGRESSED: slower (speedup < 1.0 - noiseBand) or parity violated (< minParity)
	threshold := cfg.speedupThreshold
	minParity := cfg.minParity
	noiseBand := cfg.noiseBand
	noiseRatio := cfg.noiseRatio

	var verdict StrixCandidateVerdict
	var reason string

	if math.IsNaN(cosineParity) || cosineParity < minParity {
		verdict = VerdictRegressed
		reason = fmt.Sprintf("cosine parity %.6f below required minimum %.6f (numerical parity violated)",
			cosineParity, minParity)
	} else if speedup < (1.0 - noiseBand) {
		verdict = VerdictRegressed
		reason = fmt.Sprintf("speedup %.2fx is slower than baseline (threshold floor %.2fx)",
			speedup, 1.0-noiseBand)
	} else if speedup >= threshold && noiseRatio <= noiseBand {
		verdict = VerdictPromoted
		reason = fmt.Sprintf("promoted: speedup %.2fx >= %.2fx, parity %.6f >= %.6f, noise %.1f%% <= %.1f%%",
			speedup, threshold, cosineParity, minParity, noiseRatio*100, noiseBand*100)
	} else {
		verdict = VerdictNeutral
		if noiseRatio > noiseBand {
			reason = fmt.Sprintf("neutral: speedup %.2fx exceeds threshold but measurement noise %.1f%% exceeds %.1f%% limit",
				speedup, noiseRatio*100, noiseBand*100)
		} else {
			reason = fmt.Sprintf("neutral: speedup %.2fx is within the noise band [%.2fx, %.2fx)",
				speedup, 1.0-noiseBand, threshold)
		}
	}

	return &StrixCandidateComparison{
		CandidateID:             baseline.CandidateID,
		Dimension:               baseline.Dimension,
		Feature:                 baseline.Feature,
		BaselineArmName:         baseArm.Name,
		CandidateArmName:        candArm.Name,
		BaselineLatencyUS:       baseLatency,
		CandidateLatencyUS:      candLatency,
		LatencyDeltaUS:          latencyDelta,
		Speedup:                 speedup,
		BaselineThroughputTokS:  baseThroughput,
		CandidateThroughputTokS: candThroughput,
		ThroughputDelta:         throughputDelta,
		LiftRatio:               liftRatio,
		BaselineAllocatedBytes:  baseAlloc,
		CandidateAllocatedBytes: candAlloc,
		AllocatedBytesDelta:     allocDelta,
		CompressionRatio:        compressionRatio,
		CosineParity:            cosineParity,
		NoiseRatio:              noiseRatio,
		Verdict:                 verdict,
		Reason:                  reason,
		EvaluatedAt:             time.Now().UTC(),
	}, nil
}

// FormatScoreboard renders the current scoreboard as a human-readable table.
func (r *StrixCandidateRegistry) FormatScoreboard() string {
	sb := r.Scoreboard()
	var b strings.Builder
	b.WriteString("| Candidate ID | Dimension | Baseline Arm | Candidate Arm | Baseline Latency | Candidate Latency | Speedup | Compression | Parity | Verdict |\n")
	b.WriteString("|---|---|---|---|---|---|---|---|---|---|\n")

	for _, c := range sb {
		b.WriteString(fmt.Sprintf("| %s | %s | %s | %s | %s | %s | %.2fx | %.2fx | %.6f | %s |\n",
			c.CandidateID,
			c.Dimension,
			c.BaselineArmName,
			c.CandidateArmName,
			formatLatencyUS(c.BaselineLatencyUS),
			formatLatencyUS(c.CandidateLatencyUS),
			c.Speedup,
			c.CompressionRatio,
			c.CosineParity,
			c.Verdict,
		))
	}
	return b.String()
}

func formatLatencyUS(us int64) string {
	if us >= 1000000 {
		return fmt.Sprintf("%.2fs", float64(us)/1000000.0)
	}
	if us >= 1000 {
		return fmt.Sprintf("%.2fms", float64(us)/1000.0)
	}
	return fmt.Sprintf("%dµs", us)
}

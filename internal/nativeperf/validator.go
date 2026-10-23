package nativeperf

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"
)

const (
	// DefaultAMDStrixHaloDevice is the canonical AMD Strix Halo architecture identifier.
	DefaultAMDStrixHaloDevice = "gfx1151"

	// StrixHaloTheoreticalPeakDRAMBandwidthGBps is the theoretical peak DRAM bandwidth of AMD Strix Halo (273.056 GB/s).
	StrixHaloTheoreticalPeakDRAMBandwidthGBps = 273.056

	// MinimumRooflineAttainmentRatio is the strict 80% roofline attainment acceptance gate threshold (0.80).
	MinimumRooflineAttainmentRatio = 0.80

	// MinimumLogitCosineSimilarity is the FP16/BF16 numerical accuracy gate threshold (0.999900).
	MinimumLogitCosineSimilarity = 0.999900
)

var (
	// ErrNilReceipt indicates the provided benchmark receipt pointer was nil.
	ErrNilReceipt = errors.New("nativeperf: benchmark receipt is nil")

	// ErrMissingDenominator indicates a full ceiling was missing, nonfinite or nonpositive.
	ErrMissingDenominator = errors.New("nativeperf: missing denominator or zero bandwidth in receipt")

	// ErrZeroBandwidth indicates achieved bandwidth was zero or negative.
	ErrZeroBandwidth = errors.New("nativeperf: achieved bandwidth must be positive")

	// ErrSub80PercentAttainment indicates roofline attainment fell below the strict 80% floor.
	ErrSub80PercentAttainment = errors.New("nativeperf: roofline attainment ratio below 80% requirement")

	// ErrLogitDivergence indicates logit similarity is outside the selected policy's range.
	ErrLogitDivergence = errors.New("nativeperf: logit cosine similarity outside numerical accuracy policy")

	// ErrSimulatedWithoutProof indicates a simulated benchmark run lacked a proof token or signature.
	ErrSimulatedWithoutProof = errors.New("nativeperf: simulated benchmark receipt missing proof token or signature")

	// ErrMismatchedTargetDevice indicates the receipt targets hardware outside the selected policy.
	ErrMismatchedTargetDevice = errors.New("nativeperf: target device does not match roofline policy")

	// ErrMismatchedModelArch indicates the receipt lacks a valid model architecture.
	ErrMismatchedModelArch = errors.New("nativeperf: missing or mismatched model architecture")

	// ErrInvalidMetrics indicates negative timing or invalid counter metrics were present.
	ErrInvalidMetrics = errors.New("nativeperf: negative timing or invalid counter metrics")
)

// BenchmarkReceipt represents a hardware or simulated benchmark execution receipt
// capturing bandwidth throughput, numerical fidelity, and execution provenance.
type BenchmarkReceipt struct {
	Schema                       string               `json:"schema,omitempty"`
	ReceiptID                    string               `json:"receipt_id,omitempty"`
	Device                       string               `json:"device,omitempty"`
	TargetDevice                 string               `json:"target_device,omitempty"`
	ModelArchitecture            string               `json:"model_architecture"`
	TheoreticalPeakBandwidthGBps float64              `json:"theoretical_peak_bandwidth_gbps"`
	MeasuredPeakBandwidthGBps    float64              `json:"measured_peak_bandwidth_gbps,omitempty"`
	RooflineDenominator          *RooflineDenominator `json:"roofline_denominator,omitempty"`
	BandwidthMetric              string               `json:"bandwidth_metric,omitempty"`
	BandwidthUnits               string               `json:"bandwidth_units,omitempty"`
	BandwidthScope               string               `json:"bandwidth_scope,omitempty"`
	AchievedBandwidthGBps        float64              `json:"achieved_bandwidth_gbps"`
	Target80PctBandwidthGBps     float64              `json:"target_80pct_bandwidth_gbps,omitempty"`
	LogitCosineSimilarity        float64              `json:"logit_cosine_similarity"`
	Simulated                    bool                 `json:"simulated"`
	IsSimulated                  bool                 `json:"is_simulated,omitempty"`
	ProofToken                   string               `json:"proof_token,omitempty"`
	Signature                    string               `json:"signature,omitempty"`
	ExecutionTimeMs              float64              `json:"execution_time_ms"`
	LatencyMs                    float64              `json:"latency_ms,omitempty"`
	TokensPerSecond              float64              `json:"tokens_per_second,omitempty"`
	PromptTokens                 int                  `json:"prompt_tokens,omitempty"`
	DecodeTokens                 int                  `json:"decode_tokens,omitempty"`
	TotalTokens                  int                  `json:"total_tokens,omitempty"`
	PeakMemoryBytes              uint64               `json:"peak_memory_bytes,omitempty"`
	Timestamp                    string               `json:"timestamp,omitempty"`
	GitRevision                  string               `json:"git_revision,omitempty"`
	Environment                  map[string]string    `json:"environment,omitempty"`
	Metadata                     map[string]any       `json:"metadata,omitempty"`
}

// GetDevice resolves the canonical device identifier.
func (r *BenchmarkReceipt) GetDevice() string {
	if r == nil {
		return ""
	}
	if r.Device != "" {
		return r.Device
	}
	return r.TargetDevice
}

// IsSim reports whether the benchmark execution was simulated.
func (r *BenchmarkReceipt) IsSim() bool {
	if r == nil {
		return false
	}
	return r.Simulated || r.IsSimulated
}

// UnmarshalJSON implements custom JSON deserialization supporting flat and nested schemas.
func (dst *BenchmarkReceipt) UnmarshalJSON(data []byte) error {
	if err := rejectDuplicateRooflineFields(data, ""); err != nil {
		return err
	}
	var next BenchmarkReceipt
	r := &next
	type Alias BenchmarkReceipt
	aux := &struct {
		*Alias
		TheoreticalPeak   *float64 `json:"theoretical_peak_bandwidth_gbps"`
		MeasuredPeak      *float64 `json:"measured_peak_bandwidth_gbps"`
		AchievedBandwidth *float64 `json:"achieved_bandwidth_gbps"`
		LogitSimilarity   *float64 `json:"logit_cosine_similarity"`
		Hardware          *struct {
			TargetISA             string   `json:"target_isa"`
			Platform              string   `json:"platform"`
			PeakDRAMBandwidthGBps *float64 `json:"peak_dram_bandwidth_gbps"`
		} `json:"hardware,omitempty"`
		Workload *struct {
			Model        string `json:"model"`
			WorkloadType string `json:"workload_type"`
		} `json:"workload,omitempty"`
		NumericalParity *struct {
			LogitCosineSimilarity *float64 `json:"logit_cosine_similarity"`
		} `json:"numerical_parity,omitempty"`
		RooflineAttainment *struct {
			MeasuredRoofline *float64 `json:"measured_roofline"`
			UsefulThroughput *float64 `json:"useful_throughput"`
			AttainmentRatio  float64  `json:"attainment_ratio"`
		} `json:"roofline_attainment,omitempty"`
		TimingMetrics *struct {
			ExecutionTimeMs float64 `json:"execution_time_ms"`
			LatencyMs       float64 `json:"latency_ms"`
			DurationMs      float64 `json:"duration_ms"`
		} `json:"timing,omitempty"`
	}{
		Alias: (*Alias)(r),
	}

	if err := json.Unmarshal(data, aux); err != nil {
		return err
	}
	if aux.TheoreticalPeak != nil {
		r.TheoreticalPeakBandwidthGBps = *aux.TheoreticalPeak
	}
	if aux.MeasuredPeak != nil {
		r.MeasuredPeakBandwidthGBps = *aux.MeasuredPeak
	}
	if aux.AchievedBandwidth != nil {
		r.AchievedBandwidthGBps = *aux.AchievedBandwidth
	}
	if aux.LogitSimilarity != nil {
		r.LogitCosineSimilarity = *aux.LogitSimilarity
	}

	if aux.Hardware != nil {
		device := aux.Hardware.Platform
		if aux.Hardware.TargetISA != "" {
			device = aux.Hardware.TargetISA
		}
		if r.GetDevice() == "" {
			r.TargetDevice = device
		} else if device != "" && !strings.EqualFold(strings.TrimSpace(device), strings.TrimSpace(r.GetDevice())) {
			return fmt.Errorf("%w: conflicting flat and nested device identities", ErrMismatchedTargetDevice)
		}
	}

	if r.ModelArchitecture == "" && aux.Workload != nil {
		r.ModelArchitecture = aux.Workload.Model
	}

	// Preserve denominator provenance, including invalid values for validation.
	// Never reinterpret a measured roofline as a theoretical hardware peak.
	if aux.Hardware != nil && aux.Hardware.PeakDRAMBandwidthGBps != nil {
		peak := *aux.Hardware.PeakDRAMBandwidthGBps
		if aux.TheoreticalPeak != nil && r.TheoreticalPeakBandwidthGBps != peak {
			return fmt.Errorf("%w: conflicting theoretical ceilings", ErrIncompatibleDenominator)
		}
		r.TheoreticalPeakBandwidthGBps = peak
	}
	if aux.RooflineAttainment != nil && aux.RooflineAttainment.MeasuredRoofline != nil {
		peak := *aux.RooflineAttainment.MeasuredRoofline
		if aux.MeasuredPeak != nil && r.MeasuredPeakBandwidthGBps != peak {
			return fmt.Errorf("%w: conflicting measured ceilings", ErrIncompatibleDenominator)
		}
		r.MeasuredPeakBandwidthGBps = peak
	}

	if aux.RooflineAttainment != nil && aux.RooflineAttainment.UsefulThroughput != nil {
		achieved := *aux.RooflineAttainment.UsefulThroughput
		if aux.AchievedBandwidth != nil && r.AchievedBandwidthGBps != achieved {
			return fmt.Errorf("%w: conflicting achieved bandwidth values", ErrInvalidMetrics)
		}
		r.AchievedBandwidthGBps = achieved
	}

	if aux.NumericalParity != nil && aux.NumericalParity.LogitCosineSimilarity != nil {
		similarity := *aux.NumericalParity.LogitCosineSimilarity
		if aux.LogitSimilarity != nil && r.LogitCosineSimilarity != similarity {
			return fmt.Errorf("%w: conflicting logit similarity values", ErrInvalidMetrics)
		}
		r.LogitCosineSimilarity = similarity
	}

	if r.ExecutionTimeMs == 0 && aux.TimingMetrics != nil {
		if aux.TimingMetrics.ExecutionTimeMs > 0 {
			r.ExecutionTimeMs = aux.TimingMetrics.ExecutionTimeMs
		} else if aux.TimingMetrics.DurationMs > 0 {
			r.ExecutionTimeMs = aux.TimingMetrics.DurationMs
		} else if aux.TimingMetrics.LatencyMs > 0 {
			r.ExecutionTimeMs = aux.TimingMetrics.LatencyMs
		}
	}

	*dst = next
	return nil
}

// VerificationResult contains the detailed audit outcome of ValidateRooflineAttainment.
type VerificationResult struct {
	Policy                       RooflinePolicy       `json:"policy"`
	RooflineDenominator          *RooflineDenominator `json:"roofline_denominator,omitempty"`
	MeasuredPeakBandwidthGBps    float64              `json:"measured_peak_bandwidth_gbps,omitempty"`
	Passed                       bool                 `json:"passed"`
	AttainmentRatio              float64              `json:"attainment_ratio"`
	AchievedBandwidthGBps        float64              `json:"achieved_bandwidth_gbps"`
	TheoreticalPeakBandwidthGBps float64              `json:"theoretical_peak_bandwidth_gbps"`
	TargetDevice                 string               `json:"target_device"`
	ModelArchitecture            string               `json:"model_architecture"`
	LogitCosineSimilarity        float64              `json:"logit_cosine_similarity"`
	Simulated                    bool                 `json:"simulated"`
	ProofToken                   string               `json:"proof_token,omitempty"`
	Signature                    string               `json:"signature,omitempty"`
	VerifiedAt                   time.Time            `json:"verified_at"`
	Violations                   []string             `json:"violations,omitempty"`
}

// Clean reports arithmetic and receipt consistency, not independent execution proof.
func (v *VerificationResult) Clean() bool {
	return v != nil && v.Passed && len(v.Violations) == 0
}

// RooflineValidationError represents a failed verification result wrapping the primary cause.
type RooflineValidationError struct {
	Result     *VerificationResult
	Violations []string
	Err        error
}

func (e *RooflineValidationError) Error() string {
	if len(e.Violations) > 0 {
		return fmt.Sprintf("nativeperf: roofline attainment verification failed: %s", strings.Join(e.Violations, "; "))
	}
	if e.Err != nil {
		return e.Err.Error()
	}
	return "nativeperf: roofline attainment verification failed"
}

func (e *RooflineValidationError) Unwrap() error {
	return e.Err
}

func isInvalidOrMismatchedModel(arch string) bool {
	trimmed := strings.ToLower(strings.TrimSpace(arch))
	if trimmed == "" {
		return true
	}
	switch trimmed {
	case "unknown", "mismatched", "invalid", "none", "null", "undefined":
		return true
	default:
		return false
	}
}

// ValidateRooflineAttainment is the legacy Halo theoretical-ceiling wrapper.
// It preserves the 80% and 0.999900 gates without inventing a missing ceiling.
// Measured ceilings need ValidateRooflineAttainmentWithPolicy: the legacy packet
// consumer has only a theoretical-peak field and cannot preserve measured provenance.
func ValidateRooflineAttainment(receipt *BenchmarkReceipt) (*VerificationResult, error) {
	if receipt == nil {
		return nil, ErrNilReceipt
	}
	legacy := *receipt
	policy := LegacyStrixHaloRooflinePolicy()
	if legacy.BandwidthMetric == "" {
		legacy.BandwidthMetric = policy.Metric
	}
	if legacy.BandwidthUnits == "" {
		legacy.BandwidthUnits = policy.Units
	}
	if legacy.BandwidthScope == "" {
		legacy.BandwidthScope = policy.Scope
	}
	return validateRooflineAttainment(&legacy, policy, ErrSub80PercentAttainment)
}

// ValidateRooflineAttainmentWithPolicy checks compatible full-ceiling arithmetic
// under an explicitly selected policy. Units are decimal GB/s, not Gb/s or GiB/s.
// Receipt signatures and measurement authority must be independently verified;
// neither a synthetic fixture nor a successful ratio constitutes physical proof.
func ValidateRooflineAttainmentWithPolicy(receipt *BenchmarkReceipt, policy RooflinePolicy) (*VerificationResult, error) {
	return validateRooflineAttainment(receipt, policy, ErrSubRooflineAttainment)
}

func validateRooflineAttainment(receipt *BenchmarkReceipt, policy RooflinePolicy, belowFloor error) (*VerificationResult, error) {
	if receipt == nil {
		return nil, ErrNilReceipt
	}
	if err := policy.validate(); err != nil {
		return nil, err
	}
	policy.DeviceAliases = append([]string(nil), policy.DeviceAliases...)
	res := &VerificationResult{
		Policy:                policy,
		TargetDevice:          receipt.GetDevice(),
		ModelArchitecture:     receipt.ModelArchitecture,
		AchievedBandwidthGBps: receipt.AchievedBandwidthGBps,
		LogitCosineSimilarity: receipt.LogitCosineSimilarity,
		Simulated:             receipt.IsSim(),
		ProofToken:            receipt.ProofToken,
		Signature:             receipt.Signature,
		VerifiedAt:            time.Now().UTC(),
		Violations:            make([]string, 0),
	}
	var primaryErr error
	reject := func(cause error, violation string) {
		if primaryErr == nil {
			primaryErr = cause
		}
		res.Violations = append(res.Violations, violation)
	}

	if !finiteRooflineValue(receipt.ExecutionTimeMs) || receipt.ExecutionTimeMs < 0 {
		reject(ErrInvalidMetrics, fmt.Sprintf("invalid execution time: %v ms", receipt.ExecutionTimeMs))
	}
	if !finiteRooflineValue(receipt.LatencyMs) || receipt.LatencyMs < 0 {
		reject(ErrInvalidMetrics, fmt.Sprintf("invalid latency: %v ms", receipt.LatencyMs))
	}
	if !finiteRooflineValue(receipt.TokensPerSecond) || receipt.TokensPerSecond < 0 {
		reject(ErrInvalidMetrics, fmt.Sprintf("invalid tokens per second: %v", receipt.TokensPerSecond))
	}
	if receipt.PromptTokens < 0 || receipt.DecodeTokens < 0 || receipt.TotalTokens < 0 {
		reject(ErrInvalidMetrics, fmt.Sprintf("negative counter metrics (prompt=%d, decode=%d, total=%d)",
			receipt.PromptTokens, receipt.DecodeTokens, receipt.TotalTokens))
	}
	if !policy.matchesDevice(receipt.GetDevice()) ||
		(receipt.TargetDevice != "" && !policy.matchesDevice(receipt.TargetDevice)) {
		reject(ErrMismatchedTargetDevice, fmt.Sprintf("target device %q / %q does not match policy target %q", receipt.Device, receipt.TargetDevice, policy.TargetDevice))
	}
	if isInvalidOrMismatchedModel(receipt.ModelArchitecture) {
		reject(ErrMismatchedModelArch, fmt.Sprintf("missing or mismatched model architecture %q", receipt.ModelArchitecture))
	}
	if receipt.IsSim() && strings.TrimSpace(receipt.ProofToken) == "" && strings.TrimSpace(receipt.Signature) == "" {
		reject(ErrSimulatedWithoutProof, "simulated-only run without proof token or signature")
	}

	denominator, denominatorErr := policy.denominator(receipt)
	if denominatorErr != nil {
		reject(denominatorErr, denominatorErr.Error())
	} else {
		res.RooflineDenominator = &denominator
		if denominator.Kind == RooflineTheoreticalCeiling {
			res.TheoreticalPeakBandwidthGBps = denominator.Value
		} else {
			res.MeasuredPeakBandwidthGBps = denominator.Value
		}
	}
	achievedBW := receipt.AchievedBandwidthGBps
	if !finiteRooflineValue(achievedBW) || achievedBW <= 0 {
		reject(ErrMissingDenominator, fmt.Sprintf("zero bandwidth: achieved bandwidth (%v GB/s) must be finite and positive", achievedBW))
	} else if denominatorErr == nil {
		ratio := achievedBW / denominator.Value
		if !finiteRooflineValue(ratio) {
			reject(ErrInvalidMetrics, "invalid roofline attainment ratio: division must be finite")
		} else {
			res.AttainmentRatio = ratio
			if ratio < policy.MinimumAttainmentRatio {
				reject(belowFloor, fmt.Sprintf("roofline attainment ratio %.4f (%.2f%%) below strict %g%% acceptance gate (achieved: %.2f GB/s, full ceiling: %.2f GB/s)",
					ratio, ratio*100, policy.MinimumAttainmentRatio*100, achievedBW, denominator.Value))
			}
		}
	}
	similarity := receipt.LogitCosineSimilarity
	if !finiteRooflineValue(similarity) || similarity < policy.MinimumLogitCosineSimilarity || similarity > 1.000001 {
		reject(ErrLogitDivergence, fmt.Sprintf("logit cosine similarity %.6f below numerical accuracy gate %.6f or above valid range",
			similarity, policy.MinimumLogitCosineSimilarity))
	}
	res.Passed = len(res.Violations) == 0
	if !res.Passed {
		return res, &RooflineValidationError{Result: res, Violations: res.Violations, Err: primaryErr}
	}
	return res, nil
}

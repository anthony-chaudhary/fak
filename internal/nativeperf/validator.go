package nativeperf

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
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

	// ErrMissingDenominator indicates a full compatible ceiling was not supplied.
	ErrMissingDenominator = errors.New("nativeperf: missing denominator or zero bandwidth in receipt")

	// ErrZeroBandwidth indicates achieved bandwidth was zero or negative.
	ErrZeroBandwidth = errors.New("nativeperf: achieved bandwidth must be positive")

	// ErrSub80PercentAttainment indicates roofline attainment fell below the strict 80% floor.
	ErrSub80PercentAttainment = errors.New("nativeperf: roofline attainment ratio below 80% requirement")

	// ErrLogitDivergence indicates FP16/BF16 logit cosine similarity fell below 0.999900.
	ErrLogitDivergence = errors.New("nativeperf: logit cosine similarity below numerical accuracy gate 0.999900")

	// ErrSimulatedWithoutProof indicates a simulated benchmark run lacked a proof token or signature.
	ErrSimulatedWithoutProof = errors.New("nativeperf: simulated benchmark receipt missing proof token or signature")

	// ErrMismatchedTargetDevice indicates the receipt targets hardware other than AMD Strix Halo (gfx1151).
	ErrMismatchedTargetDevice = errors.New("nativeperf: target device does not match AMD Strix Halo (gfx1151)")

	// ErrMismatchedModelArch indicates the receipt lacks a valid model architecture.
	ErrMismatchedModelArch = errors.New("nativeperf: missing or mismatched model architecture")

	// ErrInvalidMetrics indicates negative timing or invalid counter metrics were present.
	ErrInvalidMetrics = errors.New("nativeperf: negative timing or invalid counter metrics")
)

// BenchmarkReceipt represents a hardware or simulated benchmark execution receipt
// capturing bandwidth throughput, numerical fidelity, and execution provenance.
type BenchmarkReceipt struct {
	RooflineDenominator          *RooflineDenominator `json:"roofline_denominator,omitempty"`
	Schema                       string               `json:"schema,omitempty"`
	ReceiptID                    string               `json:"receipt_id,omitempty"`
	Device                       string               `json:"device,omitempty"`
	TargetDevice                 string               `json:"target_device,omitempty"`
	ModelArchitecture            string               `json:"model_architecture"`
	TheoreticalPeakBandwidthGBps float64              `json:"theoretical_peak_bandwidth_gbps"`
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
func (r *BenchmarkReceipt) UnmarshalJSON(data []byte) error {
	type Alias BenchmarkReceipt
	aux := &struct {
		*Alias
		Hardware *struct {
			TargetISA             string   `json:"target_isa"`
			Platform              string   `json:"platform"`
			PeakDRAMBandwidthGBps *float64 `json:"peak_dram_bandwidth_gbps"`
		} `json:"hardware,omitempty"`
		Workload *struct {
			Model        string `json:"model"`
			WorkloadType string `json:"workload_type"`
		} `json:"workload,omitempty"`
		NumericalParity *struct {
			LogitCosineSimilarity float64 `json:"logit_cosine_similarity"`
		} `json:"numerical_parity,omitempty"`
		RooflineAttainment *struct {
			MeasuredRoofline *float64 `json:"measured_roofline"`
			Units            string   `json:"units"`
			Scope            string   `json:"scope"`
			Metric           string   `json:"metric"`
			UsefulThroughput float64  `json:"useful_throughput"`
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

	if err := json.Unmarshal(data, &aux); err != nil {
		return err
	}

	if r.Device == "" && r.TargetDevice == "" && aux.Hardware != nil {
		if aux.Hardware.TargetISA != "" {
			r.TargetDevice = aux.Hardware.TargetISA
		} else if aux.Hardware.Platform != "" {
			r.TargetDevice = aux.Hardware.Platform
		}
	}

	if r.ModelArchitecture == "" && aux.Workload != nil {
		r.ModelArchitecture = aux.Workload.Model
	}

	if aux.Hardware != nil && aux.Hardware.PeakDRAMBandwidthGBps != nil {
		if r.TheoreticalPeakBandwidthGBps != 0 || r.RooflineDenominator != nil {
			return fmt.Errorf("nativeperf: ambiguous roofline denominator sources")
		}
		r.TheoreticalPeakBandwidthGBps = *aux.Hardware.PeakDRAMBandwidthGBps
	}

	if aux.RooflineAttainment != nil && aux.RooflineAttainment.MeasuredRoofline != nil {
		if r.RooflineDenominator != nil {
			return fmt.Errorf("nativeperf: ambiguous roofline denominator sources")
		}
		r.RooflineDenominator = &RooflineDenominator{*aux.RooflineAttainment.MeasuredRoofline, "measured", r.GetDevice(), aux.RooflineAttainment.Metric, aux.RooflineAttainment.Units, aux.RooflineAttainment.Scope}
	}

	if r.AchievedBandwidthGBps == 0 && aux.RooflineAttainment != nil && aux.RooflineAttainment.UsefulThroughput > 0 {
		r.AchievedBandwidthGBps = aux.RooflineAttainment.UsefulThroughput
	}

	if r.LogitCosineSimilarity == 0 && aux.NumericalParity != nil && aux.NumericalParity.LogitCosineSimilarity > 0 {
		r.LogitCosineSimilarity = aux.NumericalParity.LogitCosineSimilarity
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

	return nil
}

// VerificationResult contains the detailed audit outcome of ValidateRooflineAttainment.
type VerificationResult struct {
	RooflineDenominator          *RooflineDenominator `json:"roofline_denominator,omitempty"`
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

// Clean reports whether the result passed with zero violations.
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

func isGFX1151Device(dev string) bool {
	norm := strings.ToLower(strings.TrimSpace(dev))
	if norm == "" {
		return false
	}
	switch norm {
	case "gfx1151", "strix-halo", "strix_halo", "strixhalo", "strix halo", "amd strix halo", "amd strix halo (gfx1151)", "8060s", "radeon 8060s", "amd radeon 8060s", "ryzen ai max+ 395", "amd ryzen ai max+ 395":
		return true
	default:
		return false
	}
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

// ValidateRooflineAttainment is the legacy Halo-policy wrapper. It accepts the
// named legacy theoretical GB/s field as a full device-DRAM ceiling and rejects
// measured ceilings that the legacy packet schema cannot represent. Arithmetic
// acceptance and proof-token presence do not establish physical execution.
func ValidateRooflineAttainment(receipt *BenchmarkReceipt) (*VerificationResult, error) {
	return validateRooflineAttainment(receipt, HaloRooflinePolicy(), true)
}

// ValidateRooflineAttainmentWithPolicy requires an explicitly described full ceiling.
// It validates arithmetic and compatibility, not independent execution authority.
func ValidateRooflineAttainmentWithPolicy(receipt *BenchmarkReceipt, policy RooflinePolicy) (*VerificationResult, error) {
	return validateRooflineAttainment(receipt, policy, false)
}

func validateRooflineAttainment(receipt *BenchmarkReceipt, policy RooflinePolicy, legacyHalo bool) (*VerificationResult, error) {
	if receipt == nil {
		return nil, ErrNilReceipt
	}

	if err := policy.validate(); err != nil {
		return nil, err
	}
	res := &VerificationResult{
		TargetDevice:                 receipt.GetDevice(),
		ModelArchitecture:            receipt.ModelArchitecture,
		AchievedBandwidthGBps:        receipt.AchievedBandwidthGBps,
		TheoreticalPeakBandwidthGBps: receipt.TheoreticalPeakBandwidthGBps,
		LogitCosineSimilarity:        receipt.LogitCosineSimilarity,
		Simulated:                    receipt.IsSim(),
		ProofToken:                   receipt.ProofToken,
		Signature:                    receipt.Signature,
		VerifiedAt:                   time.Now().UTC(),
		Violations:                   make([]string, 0),
	}

	// 1. Audit timing and counter metrics for non-negative values and validity
	if math.IsNaN(receipt.ExecutionTimeMs) || math.IsInf(receipt.ExecutionTimeMs, 0) || receipt.ExecutionTimeMs < 0 {
		res.Violations = append(res.Violations, fmt.Sprintf("invalid execution time: %v ms", receipt.ExecutionTimeMs))
	}
	if math.IsNaN(receipt.LatencyMs) || math.IsInf(receipt.LatencyMs, 0) || receipt.LatencyMs < 0 {
		res.Violations = append(res.Violations, fmt.Sprintf("invalid latency: %v ms", receipt.LatencyMs))
	}
	if math.IsNaN(receipt.TokensPerSecond) || math.IsInf(receipt.TokensPerSecond, 0) || receipt.TokensPerSecond < 0 {
		res.Violations = append(res.Violations, fmt.Sprintf("invalid tokens per second: %v", receipt.TokensPerSecond))
	}
	if receipt.PromptTokens < 0 || receipt.DecodeTokens < 0 || receipt.TotalTokens < 0 {
		res.Violations = append(res.Violations, fmt.Sprintf("negative counter metrics (prompt=%d, decode=%d, total=%d)",
			receipt.PromptTokens, receipt.DecodeTokens, receipt.TotalTokens))
	}

	// The legacy wrapper retains Halo aliases; explicit policies use exact device IDs.
	matches := policy.matchesDevice(receipt.GetDevice())
	if legacyHalo {
		matches = isGFX1151Device(receipt.GetDevice())
	}
	conflictingDevices := receipt.Device != "" && receipt.TargetDevice != "" && !strings.EqualFold(receipt.Device, receipt.TargetDevice)
	if legacyHalo && isGFX1151Device(receipt.Device) && isGFX1151Device(receipt.TargetDevice) {
		conflictingDevices = false
	}
	if !matches || conflictingDevices {
		res.Violations = append(res.Violations, fmt.Sprintf("target device %q does not match policy %q or conflicts with target_device", receipt.GetDevice(), policy.TargetDevice))
	}

	// 3. Audit model architecture
	if isInvalidOrMismatchedModel(receipt.ModelArchitecture) {
		res.Violations = append(res.Violations, fmt.Sprintf("missing or mismatched model architecture %q", receipt.ModelArchitecture))
	}

	// 4. Audit simulated-only runs (requires proof token or signature)
	if receipt.IsSim() {
		if strings.TrimSpace(receipt.ProofToken) == "" && strings.TrimSpace(receipt.Signature) == "" {
			res.Violations = append(res.Violations, "simulated-only run without proof token or signature")
		}
	}

	// Only a compatible full ceiling may enter the ratio.
	denominator, denominatorErr := resolveRooflineDenominator(receipt, policy, legacyHalo)
	peakBW := 0.0
	if denominatorErr != nil {
		res.Violations = append(res.Violations, denominatorErr.Error())
	} else {
		res.RooflineDenominator = denominator
		peakBW = denominator.Value
		if denominator.Kind == "theoretical" {
			res.TheoreticalPeakBandwidthGBps = peakBW
		}
	}

	achievedBW := receipt.AchievedBandwidthGBps
	if math.IsNaN(achievedBW) || math.IsInf(achievedBW, 0) || achievedBW <= 0 {
		res.Violations = append(res.Violations, fmt.Sprintf("zero bandwidth: achieved bandwidth (%.2f GB/s) must be positive", achievedBW))
	}

	// 6. Apply the selected campaign floor once to the full ceiling.
	if finitePositive(peakBW) && finitePositive(achievedBW) {
		ratio := achievedBW / peakBW
		res.AttainmentRatio = ratio
		if !finitePositive(ratio) || ratio < policy.MinimumAttainmentRatio {
			gate := fmt.Sprintf("policy acceptance gate %.4f", policy.MinimumAttainmentRatio)
			if legacyHalo {
				gate = "strict 80% acceptance gate"
			}
			res.Violations = append(res.Violations, fmt.Sprintf("roofline attainment ratio %.4f (%.2f%%) below %s (achieved: %.2f GB/s, peak: %.2f GB/s)", ratio, ratio*100, gate, achievedBW, peakBW))
		}
	}

	// 7. Apply the selected numerical accuracy floor.
	similarity := receipt.LogitCosineSimilarity
	if math.IsNaN(similarity) || math.IsInf(similarity, 0) || similarity < policy.MinimumLogitCosineSimilarity || similarity > 1.000001 {
		res.Violations = append(res.Violations, fmt.Sprintf("logit cosine similarity %.6f below numerical accuracy gate %.6f",
			similarity, policy.MinimumLogitCosineSimilarity))
	}

	res.Passed = len(res.Violations) == 0
	if !res.Passed {
		var primaryErr error
		for _, v := range res.Violations {
			switch {
			case strings.Contains(v, "missing denominator") || strings.Contains(v, "zero bandwidth"):
				if primaryErr == nil {
					primaryErr = ErrMissingDenominator
				}
			case strings.Contains(v, "simulated-only run"):
				if primaryErr == nil {
					primaryErr = ErrSimulatedWithoutProof
				}
			case strings.Contains(v, "target device"):
				if primaryErr == nil {
					primaryErr = ErrMismatchedTargetDevice
				}
			case strings.Contains(v, "model architecture"):
				if primaryErr == nil {
					primaryErr = ErrMismatchedModelArch
				}
			case strings.Contains(v, "invalid") || strings.Contains(v, "negative"):
				if primaryErr == nil {
					primaryErr = ErrInvalidMetrics
				}
			case strings.Contains(v, "roofline attainment ratio"):
				if primaryErr == nil {
					primaryErr = ErrSub80PercentAttainment
				}
			case strings.Contains(v, "logit cosine similarity"):
				if primaryErr == nil {
					primaryErr = ErrLogitDivergence
				}
			}
		}
		if primaryErr == nil {
			primaryErr = ErrSub80PercentAttainment
		}
		return res, &RooflineValidationError{
			Result:     res,
			Violations: res.Violations,
			Err:        primaryErr,
		}
	}

	return res, nil
}

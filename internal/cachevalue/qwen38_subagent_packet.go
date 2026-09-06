package cachevalue

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strings"
)

const (
	// SubagentWorkloadSchemaV1 is the canonical schema for the frozen subagent workload packet.
	SubagentWorkloadSchemaV1 = "fak.subagent_workload/v1"

	// Strix Halo (Ryzen AI Max+ 395, gfx1151) frozen hardware constants.
	StrixHaloArch                     = "gfx1151"
	StrixHaloComputeUnits             = 40
	StrixHaloWavefrontSize            = 32
	StrixHaloBusWidthBits             = 256
	StrixHaloMemoryClockMHz           = 8533.0
	StrixHaloTheoreticalPeakGBps      = 273.056
	StrixHaloSustainableDRAMBandwidth = 224.0
	StrixHaloTarget80PctBandwidthGBps = 218.44

	// Supported Qwen 3.8 Coder models.
	ModelQwen38Coder35B = "Qwen3.8-Coder-35B"
	ModelQwen38Coder27B = "Qwen3.8-Coder-27B"
	ModelQwen38Coder14B = "Qwen3.8-Coder-14B"

	// Supported quantization formats.
	QuantFormatQ4KM = "Q4_K_M"
	QuantFormatQ3KM = "Q3_K_M"

	// Frozen workload context boundaries.
	MinSharedPrefixTokens = 32768
	MaxSharedPrefixTokens = 65536
	MinSuffixTokens       = 100
	MaxSuffixTokens       = 500

	// Acceptance thresholds.
	MinCosineParity            = 0.999900
	MinRooflineAttainmentRatio = 0.80

	// Numerical tolerance for floating-point roofline comparison.
	rooflineEps = 1e-6
)

const (
	SubagentCoordinator = "coordinator"
	SubagentExplore     = "explore"
	SubagentTester      = "tester"
	SubagentReviewer    = "reviewer"
	SubagentGeneral     = "general"
)

// DefaultSubagents returns the canonical 5-subagent team (1 coordinator + 4 workers).
var DefaultSubagents = []string{
	SubagentCoordinator,
	SubagentExplore,
	SubagentTester,
	SubagentReviewer,
	SubagentGeneral,
}

// HardwareSpec records the physical accelerator architecture and bandwidth ceilings.
type HardwareSpec struct {
	Arch                     string  `json:"arch"`
	ComputeUnits             int     `json:"compute_units,omitempty"`
	WavefrontSize            int     `json:"wavefront_size,omitempty"`
	BusWidth                 int     `json:"bus_width"`
	ClockMHz                 float64 `json:"clock_mhz"`
	PeakBandwidthGBps        float64 `json:"peak_bandwidth_gbps"`
	Target80PctBandwidthGBps float64 `json:"target_80pct_bandwidth_gbps"`
}

// ModelSpec describes the target model, parameter scale, quantization format, and weight bytes.
type ModelSpec struct {
	Name              string `json:"name"`
	ParamCount        int64  `json:"param_count"`
	QuantFormat       string `json:"quant_format"`
	ActiveWeightBytes int64  `json:"active_weight_bytes"`
}

// WorkloadSpec specifies the multi-agent context configuration and turn count.
type WorkloadSpec struct {
	Subagents           []string `json:"subagents"`
	SharedPrefixTokens  int      `json:"shared_prefix_tokens"`
	PrivateSuffixTokens int      `json:"private_suffix_tokens"`
	TotalTurns          int      `json:"total_turns"`
}

// MeasuredRun captures empirical execution telemetry, prefill elision, and numerical parity.
type MeasuredRun struct {
	AttainedBandwidthGBps float64  `json:"attained_bandwidth_gbps"`
	TotalPrefillAvoided   int64    `json:"total_prefill_avoided"`
	ReprefillCount        int64    `json:"reprefill_count"`
	CosineParity          float64  `json:"cosine_parity"`
	AttainmentRatio       float64  `json:"attainment_ratio"`
	Passed80Pct           bool     `json:"passed_80pct"`
	DurationMS            float64  `json:"duration_ms,omitempty"`
	Simulated             bool     `json:"simulated,omitempty"`
	ProofToken            string   `json:"proof_token,omitempty"`
	ProofTokens           []string `json:"proof_tokens,omitempty"`
}

// SubagentWorkloadPacket is the top-level frozen benchmark and acceptance artifact.
type SubagentWorkloadPacket struct {
	SchemaVersion string       `json:"schema_version"`
	HardwareSpec  HardwareSpec `json:"hardware_spec"`
	ModelSpec     ModelSpec    `json:"model_spec"`
	WorkloadSpec  WorkloadSpec `json:"workload_spec"`
	MeasuredRun   MeasuredRun  `json:"measured_run"`
	Digest        string       `json:"digest,omitempty"`
}

// ValidatePacket verifies that the workload packet conforms to schema, non-zero denominator
// constraints, numerical parity gates, and the 80% roofline attainment floor.
func ValidatePacket(packet *SubagentWorkloadPacket) error {
	if packet == nil {
		return errors.New("subagent packet is nil")
	}

	// 1. Check schema version
	if packet.SchemaVersion != SubagentWorkloadSchemaV1 {
		return fmt.Errorf("schema version mismatch: got %q, want %q", packet.SchemaVersion, SubagentWorkloadSchemaV1)
	}

	// 2. Enforce non-zero denominators and valid hardware specification
	if strings.TrimSpace(packet.HardwareSpec.Arch) == "" {
		return errors.New("hardware arch is required")
	}
	if packet.HardwareSpec.BusWidth <= 0 {
		return errors.New("hardware bus width must be greater than zero (non-zero denominator)")
	}
	if packet.HardwareSpec.ClockMHz <= 0 {
		return errors.New("hardware clock MHz must be greater than zero (non-zero denominator)")
	}
	if packet.HardwareSpec.PeakBandwidthGBps <= 0 {
		return errors.New("hardware peak bandwidth must be greater than zero (non-zero denominator)")
	}
	if packet.HardwareSpec.Target80PctBandwidthGBps <= 0 {
		return errors.New("hardware target 80% bandwidth must be greater than zero (non-zero denominator)")
	}

	// 3. Enforce valid model specification
	if strings.TrimSpace(packet.ModelSpec.Name) == "" {
		return errors.New("model name is required")
	}
	if packet.ModelSpec.ParamCount <= 0 {
		return errors.New("model parameter count must be greater than zero (non-zero denominator)")
	}
	if strings.TrimSpace(packet.ModelSpec.QuantFormat) == "" {
		return errors.New("model quant format is required")
	}
	if packet.ModelSpec.ActiveWeightBytes <= 0 {
		return errors.New("model active weight bytes must be greater than zero (non-zero denominator)")
	}

	// 4. Enforce valid workload specification
	if len(packet.WorkloadSpec.Subagents) == 0 {
		return errors.New("workload subagents list must not be empty")
	}
	if packet.WorkloadSpec.SharedPrefixTokens <= 0 {
		return errors.New("workload shared prefix tokens must be greater than zero (non-zero denominator)")
	}
	if packet.WorkloadSpec.PrivateSuffixTokens <= 0 {
		return errors.New("workload private suffix tokens must be greater than zero (non-zero denominator)")
	}
	if packet.WorkloadSpec.TotalTurns <= 0 {
		return errors.New("workload total turns must be greater than zero (non-zero denominator)")
	}

	// 5. Enforce duration and reject unverified simulated runs
	if packet.MeasuredRun.DurationMS < 0 {
		return fmt.Errorf("negative duration %v ms rejected", packet.MeasuredRun.DurationMS)
	}
	if packet.MeasuredRun.Simulated {
		hasProof := strings.TrimSpace(packet.MeasuredRun.ProofToken) != "" || len(packet.MeasuredRun.ProofTokens) > 0
		if !hasProof {
			return errors.New("simulated run rejected: missing verified proof tokens")
		}
	}
	if packet.MeasuredRun.TotalPrefillAvoided < 0 {
		return errors.New("total prefill avoided cannot be negative")
	}
	if packet.MeasuredRun.ReprefillCount < 0 {
		return errors.New("reprefill count cannot be negative")
	}

	// 6. Check cosine similarity >= 0.999900
	if math.IsNaN(packet.MeasuredRun.CosineParity) {
		return errors.New("cosine parity is NaN")
	}
	if packet.MeasuredRun.CosineParity < MinCosineParity {
		return fmt.Errorf("cosine parity %.6f below required threshold %.6f", packet.MeasuredRun.CosineParity, MinCosineParity)
	}

	// 7. Enforce 80% roofline attainment rule: AttainedBandwidthGBps / TheoreticalPeakBandwidthGBps >= 0.80
	ratio := packet.MeasuredRun.AttainedBandwidthGBps / packet.HardwareSpec.PeakBandwidthGBps
	if ratio < MinRooflineAttainmentRatio-rooflineEps || packet.MeasuredRun.AttainedBandwidthGBps+rooflineEps < packet.HardwareSpec.Target80PctBandwidthGBps {
		return fmt.Errorf("roofline attainment ratio %.4f (%.2f GB/s) below 80%% threshold (%.2f GB/s required)",
			ratio, packet.MeasuredRun.AttainedBandwidthGBps, packet.HardwareSpec.Target80PctBandwidthGBps)
	}
	if !packet.MeasuredRun.Passed80Pct {
		return errors.New("measured run Passed80Pct flag must be true for passing packet")
	}
	if packet.MeasuredRun.AttainmentRatio < MinRooflineAttainmentRatio-rooflineEps {
		return fmt.Errorf("attainment ratio %.4f below 80%% threshold", packet.MeasuredRun.AttainmentRatio)
	}

	return nil
}

// ValidateFrozenWorkload verifies that the packet conforms strictly to the frozen AMD Strix Halo
// ideal-cache subagent specifications.
func ValidateFrozenWorkload(packet *SubagentWorkloadPacket) error {
	if err := ValidatePacket(packet); err != nil {
		return err
	}

	// Verify Strix Halo architecture
	if !strings.EqualFold(packet.HardwareSpec.Arch, StrixHaloArch) {
		return fmt.Errorf("frozen hardware arch mismatch: got %q, want %q", packet.HardwareSpec.Arch, StrixHaloArch)
	}
	if packet.HardwareSpec.BusWidth != StrixHaloBusWidthBits {
		return fmt.Errorf("frozen hardware bus width mismatch: got %d, want %d", packet.HardwareSpec.BusWidth, StrixHaloBusWidthBits)
	}

	// Verify supported models and quants
	switch packet.ModelSpec.Name {
	case ModelQwen38Coder35B, ModelQwen38Coder27B, ModelQwen38Coder14B:
		// Known frozen model
	default:
		return fmt.Errorf("unsupported frozen model name %q", packet.ModelSpec.Name)
	}

	switch packet.ModelSpec.QuantFormat {
	case QuantFormatQ4KM, QuantFormatQ3KM:
		// Known frozen quant format
	default:
		return fmt.Errorf("unsupported frozen quant format %q", packet.ModelSpec.QuantFormat)
	}

	// Verify workload context bounds
	if packet.WorkloadSpec.SharedPrefixTokens < MinSharedPrefixTokens || packet.WorkloadSpec.SharedPrefixTokens > MaxSharedPrefixTokens {
		return fmt.Errorf("shared prefix tokens %d outside frozen range [%d, %d]",
			packet.WorkloadSpec.SharedPrefixTokens, MinSharedPrefixTokens, MaxSharedPrefixTokens)
	}
	if packet.WorkloadSpec.PrivateSuffixTokens < MinSuffixTokens || packet.WorkloadSpec.PrivateSuffixTokens > MaxSuffixTokens {
		return fmt.Errorf("private suffix tokens %d outside frozen range [%d, %d]",
			packet.WorkloadSpec.PrivateSuffixTokens, MinSuffixTokens, MaxSuffixTokens)
	}

	return nil
}

// HashPacket serializes the packet deterministically and returns its hex-encoded SHA-256 digest.
// Any existing Digest value is cleared before hashing to guarantee idempotence.
func HashPacket(packet *SubagentWorkloadPacket) string {
	if packet == nil {
		return ""
	}
	cp := *packet
	cp.Digest = ""
	data, err := json.Marshal(cp)
	if err != nil {
		return ""
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

// DefaultStrixHaloHardwareSpec returns standard frozen hardware descriptors for AMD Strix Halo gfx1151.
func DefaultStrixHaloHardwareSpec() HardwareSpec {
	return HardwareSpec{
		Arch:                     StrixHaloArch,
		ComputeUnits:             StrixHaloComputeUnits,
		WavefrontSize:            StrixHaloWavefrontSize,
		BusWidth:                 StrixHaloBusWidthBits,
		ClockMHz:                 StrixHaloMemoryClockMHz,
		PeakBandwidthGBps:        StrixHaloTheoreticalPeakGBps,
		Target80PctBandwidthGBps: StrixHaloTarget80PctBandwidthGBps,
	}
}

// FrozenModelSpec returns the frozen model specification for a given Qwen 3.8 Coder variant.
func FrozenModelSpec(name, quant string) (ModelSpec, error) {
	switch name {
	case ModelQwen38Coder35B:
		var bytes int64
		switch quant {
		case QuantFormatQ4KM:
			bytes = 21_000_000_000
		case QuantFormatQ3KM:
			bytes = 16_000_000_000
		default:
			return ModelSpec{}, fmt.Errorf("unsupported quant format %q for %s", quant, name)
		}
		return ModelSpec{
			Name:              name,
			ParamCount:        35_000_000_000,
			QuantFormat:       quant,
			ActiveWeightBytes: bytes,
		}, nil

	case ModelQwen38Coder27B:
		var bytes int64
		switch quant {
		case QuantFormatQ4KM:
			bytes = 16_200_000_000
		case QuantFormatQ3KM:
			bytes = 12_500_000_000
		default:
			return ModelSpec{}, fmt.Errorf("unsupported quant format %q for %s", quant, name)
		}
		return ModelSpec{
			Name:              name,
			ParamCount:        27_000_000_000,
			QuantFormat:       quant,
			ActiveWeightBytes: bytes,
		}, nil

	case ModelQwen38Coder14B:
		var bytes int64
		switch quant {
		case QuantFormatQ4KM:
			bytes = 8_500_000_000
		case QuantFormatQ3KM:
			bytes = 6_500_000_000
		default:
			return ModelSpec{}, fmt.Errorf("unsupported quant format %q for %s", quant, name)
		}
		return ModelSpec{
			Name:              name,
			ParamCount:        14_000_000_000,
			QuantFormat:       quant,
			ActiveWeightBytes: bytes,
		}, nil

	default:
		return ModelSpec{}, fmt.Errorf("unknown model %q (expected %s, %s, or %s)", name, ModelQwen38Coder35B, ModelQwen38Coder27B, ModelQwen38Coder14B)
	}
}

// DefaultWorkloadSpec returns the frozen 1 lead coordinator + 4 subagents workload spec.
func DefaultWorkloadSpec() WorkloadSpec {
	return WorkloadSpec{
		Subagents:           append([]string(nil), DefaultSubagents...),
		SharedPrefixTokens:  32768,
		PrivateSuffixTokens: 256,
		TotalTurns:          10,
	}
}

// NewValidSubagentWorkloadPacket returns a fully populated, valid acceptance packet fixture.
func NewValidSubagentWorkloadPacket() *SubagentWorkloadPacket {
	model, _ := FrozenModelSpec(ModelQwen38Coder27B, QuantFormatQ4KM)
	hw := DefaultStrixHaloHardwareSpec()
	workload := DefaultWorkloadSpec()

	attained := StrixHaloSustainableDRAMBandwidth // 224.0 GB/s
	ratio := attained / hw.PeakBandwidthGBps      // 224.0 / 273.056 = ~0.820344

	pkt := &SubagentWorkloadPacket{
		SchemaVersion: SubagentWorkloadSchemaV1,
		HardwareSpec:  hw,
		ModelSpec:     model,
		WorkloadSpec:  workload,
		MeasuredRun: MeasuredRun{
			AttainedBandwidthGBps: attained,
			TotalPrefillAvoided:   int64(workload.SharedPrefixTokens) * int64(len(workload.Subagents)-1),
			ReprefillCount:        0,
			CosineParity:          0.999950,
			AttainmentRatio:       ratio,
			Passed80Pct:           true,
			DurationMS:            1250.0,
			Simulated:             false,
		},
	}
	pkt.Digest = HashPacket(pkt)
	return pkt
}

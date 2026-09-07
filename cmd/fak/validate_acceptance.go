package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/roofline"
)

const (
	// QuarantinedSyntheticDigest is the quarantined digest of the unverified 184.10 tok/s constants from #11572.
	QuarantinedSyntheticDigest = "sha256:26b0c701944fa0d6a65643fa5156cc2c6982d1df40deb7dda1264e2bcd606f5b"

	// VerdictUnverifiedSyntheticWitness indicates an unverified template witness without hardware execution backing.
	VerdictUnverifiedSyntheticWitness = "UNVERIFIED_SYNTHETIC_WITNESS"

	// AcceptanceTypeStrixHalo80Pct is the identifier for the AMD Strix Halo 80% roofline attainment campaign.
	AcceptanceTypeStrixHalo80Pct = "strix-halo-80pct"

	// SubagentFanoutReceiptSchema identifies the subagent fan-out benchmark receipt contract.
	SubagentFanoutReceiptSchema = "fak.benchmark.subagent_fanout/v1"

	// EmpiricalRooflineSchema identifies the verified empirical micro-roofline receipt contract.
	EmpiricalRooflineSchema = "fak.roofline.empirical/v1"

	// AcceptanceValidationSchema identifies the acceptance gate result contract.
	AcceptanceValidationSchema = "fak.acceptance.validation/v1"

	// DefaultWitnessArtifactPath is the canonical scrubbed public witness receipt location.
	DefaultWitnessArtifactPath = "docs/_witnesses/issue-11572-strix-halo-80pct/receipt.json"

	// Campaign thresholds and invariants
	StrixHaloTargetISA         = "gfx1151"
	StrixHaloExpectedCUs       = 40
	EfficiencyFloorThreshold   = 0.80
	LogitCosineParityThreshold = 0.999900
	RequiredEngineName         = "fak-native"
	MaxAllowedFallbackCount    = 0
	MinStatisticalRepetitions  = 5
	MaxStatisticalNoisePct     = 5.0
)

// SubagentFanoutReceipt represents the scrubbed benchmark artifact capturing subagent fan-out execution.
type SubagentFanoutReceipt struct {
	Schema             string                     `json:"schema"`
	Issue              int                        `json:"issue,omitempty"`
	Title              string                     `json:"title,omitempty"`
	CapturedAt         string                     `json:"captured_at,omitempty"`
	Timestamp          string                     `json:"timestamp,omitempty"`
	Verdict            string                     `json:"verdict"`
	Provenance         string                     `json:"provenance,omitempty"`
	ProvenanceDetails  string                     `json:"provenance_details,omitempty"`
	CaptureCommand     string                     `json:"capture_command,omitempty"`
	RawArtifactPath    string                     `json:"raw_artifact_path,omitempty"`
	RawArtifactDigest  string                     `json:"raw_artifact_digest,omitempty"`
	HardwareAttested   bool                       `json:"hardware_attested,omitempty"`
	Workload           SubagentFanoutWorkload     `json:"workload"`
	Hardware           SubagentFanoutHardware     `json:"hardware"`
	Engine             SubagentFanoutEngine       `json:"engine"`
	NumericalParity    SubagentFanoutParity       `json:"numerical_parity"`
	Statistics         SubagentFanoutStatistics   `json:"statistics"`
	RooflineAttainment SubagentRooflineAttainment `json:"roofline_attainment"`
	Reproducibility    SubagentReproducibility    `json:"reproducibility"`
	Digest             string                     `json:"digest,omitempty"`
	Verified           bool                       `json:"verified"`
}

// SubagentFanoutWorkload captures model, context, and concurrency parameters.
type SubagentFanoutWorkload struct {
	WorkloadType       string `json:"workload_type"`
	Model              string `json:"model"`
	Artifact           string `json:"artifact,omitempty"`
	ArtifactRevision   string `json:"artifact_revision,omitempty"`
	ArtifactSHA256     string `json:"artifact_sha256,omitempty"`
	Quantization       string `json:"quantization"`
	ContextLength      int    `json:"context_length"`
	OutputLength       int    `json:"output_length"`
	SubagentCount      int    `json:"subagent_count"`
	Concurrency        int    `json:"concurrency,omitempty"`
	PrefixTokensElided int    `json:"prefix_tokens_elided,omitempty"`
	IdealCache         bool   `json:"ideal_cache"`
}

// SubagentFanoutHardware details the physical target architecture.
type SubagentFanoutHardware struct {
	Platform              string  `json:"platform"`
	Architecture          string  `json:"architecture"`
	TargetISA             string  `json:"target_isa"`
	ComputeUnits          int     `json:"compute_units"`
	MemoryType            string  `json:"memory_type"`
	BusWidthBits          int     `json:"bus_width_bits"`
	PeakDRAMBandwidthGBps float64 `json:"peak_dram_bandwidth_gbps,omitempty"`
}

// SubagentFanoutEngine records execution engine identities and fallback telemetry.
type SubagentFanoutEngine struct {
	Name          string `json:"name"`
	PrimaryEngine string `json:"primary_engine,omitempty"`
	Backend       string `json:"backend"`
	ExecutionPath string `json:"execution_path,omitempty"`
	ZeroFallback  bool   `json:"zero_fallback"`
	FallbackCount int    `json:"fallback_count"`
}

// EngineName resolves the authoritative engine name from available fields.
func (e *SubagentFanoutEngine) EngineName() string {
	if e.PrimaryEngine != "" {
		return strings.TrimSpace(e.PrimaryEngine)
	}
	return strings.TrimSpace(e.Name)
}

// SubagentFanoutParity records numerical fidelity against reference GEMV.
type SubagentFanoutParity struct {
	Metric                string  `json:"metric"`
	ReferenceGEMV         string  `json:"reference_gemv"`
	LogitCosineSimilarity float64 `json:"logit_cosine_similarity"`
	MinThreshold          float64 `json:"min_threshold"`
	Passed                bool    `json:"passed"`
}

// SubagentFanoutStatistics captures repetition counts and variation metrics.
type SubagentFanoutStatistics struct {
	Repetitions             int       `json:"repetitions"`
	NoisePercentage         float64   `json:"noise_percentage"`
	MaxNoisePercentage      float64   `json:"max_noise_percentage"`
	UsefulThroughputSamples []float64 `json:"useful_throughput_samples,omitempty"`
	SampleMean              float64   `json:"sample_mean,omitempty"`
	SampleStdDev            float64   `json:"sample_std_dev,omitempty"`
}

// SubagentRooflineAttainment measures useful throughput against empirical roofline ceiling.
type SubagentRooflineAttainment struct {
	MeasuredRoofline float64 `json:"measured_roofline"`
	UsefulThroughput float64 `json:"useful_throughput"`
	AttainmentRatio  float64 `json:"attainment_ratio"`
	EfficiencyFloor  float64 `json:"efficiency_floor"`
	Achieved         bool    `json:"achieved"`
}

// SubagentReproducibility records commands, artifact references, and scrubbing attestation.
type SubagentReproducibility struct {
	ArtifactPath        string `json:"artifact_path"`
	Command             string `json:"command"`
	CaptureCommand      string `json:"capture_command,omitempty"`
	Provenance          string `json:"provenance,omitempty"`
	ProvenanceDetails   string `json:"provenance_details,omitempty"`
	RawArtifactPath     string `json:"raw_artifact_path,omitempty"`
	RawArtifactDigest   string `json:"raw_artifact_digest,omitempty"`
	HardwareAttested    bool   `json:"hardware_attested,omitempty"`
	Scrubbed            bool   `json:"scrubbed"`
	Digest              string `json:"digest,omitempty"`
}

// EffectiveProvenance returns the non-empty provenance declared on receipt or reproducibility.
func (r *SubagentFanoutReceipt) EffectiveProvenance() string {
	if p := strings.TrimSpace(r.Provenance); p != "" {
		return p
	}
	return strings.TrimSpace(r.Reproducibility.Provenance)
}

// EffectiveCaptureCommand returns the non-empty capture command on receipt or reproducibility.
func (r *SubagentFanoutReceipt) EffectiveCaptureCommand() string {
	if cmd := strings.TrimSpace(r.CaptureCommand); cmd != "" {
		return cmd
	}
	return strings.TrimSpace(r.Reproducibility.CaptureCommand)
}

// EffectiveProvenanceDetails returns non-empty provenance details on receipt or reproducibility.
func (r *SubagentFanoutReceipt) EffectiveProvenanceDetails() string {
	if det := strings.TrimSpace(r.ProvenanceDetails); det != "" {
		return det
	}
	return strings.TrimSpace(r.Reproducibility.ProvenanceDetails)
}

// EffectiveRawArtifactPath returns non-empty raw artifact path on receipt or reproducibility.
func (r *SubagentFanoutReceipt) EffectiveRawArtifactPath() string {
	if path := strings.TrimSpace(r.RawArtifactPath); path != "" {
		return path
	}
	return strings.TrimSpace(r.Reproducibility.RawArtifactPath)
}

// EffectiveRawArtifactDigest returns non-empty raw artifact digest on receipt or reproducibility.
func (r *SubagentFanoutReceipt) EffectiveRawArtifactDigest() string {
	if dig := strings.TrimSpace(r.RawArtifactDigest); dig != "" {
		return dig
	}
	return strings.TrimSpace(r.Reproducibility.RawArtifactDigest)
}

// ComputeDigest calculates canonical SHA-256 digest of receipt payload.
func (r *SubagentFanoutReceipt) ComputeDigest() (string, error) {
	clone := *r
	clone.Digest = ""
	clone.Reproducibility.Digest = ""
	clone.Verified = false
	raw, err := json.Marshal(clone)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(raw)
	return fmt.Sprintf("sha256:%s", hex.EncodeToString(sum[:])), nil
}

// AcceptanceValidationReport encapsulates full machine-readable validation result.
type AcceptanceValidationReport struct {
	Schema            string                     `json:"schema"`
	AcceptanceType    string                     `json:"acceptance_type"`
	ReceiptPath       string                     `json:"receipt_path"`
	ReceiptSchema     string                     `json:"receipt_schema"`
	Passed            bool                       `json:"passed"`
	Verdict           string                     `json:"verdict"`
	Pillars           *AcceptancePillarsReport   `json:"pillars,omitempty"`
	EmpiricalRoofline *AcceptanceEmpiricalReport `json:"empirical_roofline,omitempty"`
	Failures          []string                   `json:"failures"`
	Timestamp         string                     `json:"timestamp"`
}

// AcceptancePillarsReport encapsulates status of the 5 campaign pillars.
type AcceptancePillarsReport struct {
	Pillar1EfficiencyFloor  Pillar1Report `json:"pillar1_efficiency_floor"`
	Pillar2NumericalParity  Pillar2Report `json:"pillar2_numerical_parity"`
	Pillar3ZeroFallback     Pillar3Report `json:"pillar3_zero_fallback"`
	Pillar4StatisticalRigor Pillar4Report `json:"pillar4_statistical_rigor"`
	Pillar5Reproducibility  Pillar5Report `json:"pillar5_reproducibility"`
}

type Pillar1Report struct {
	Passed           bool    `json:"passed"`
	UsefulThroughput float64 `json:"useful_throughput"`
	MeasuredRoofline float64 `json:"measured_roofline"`
	AttainmentRatio  float64 `json:"attainment_ratio"`
	Threshold        float64 `json:"threshold"`
	AttainmentPct    string  `json:"attainment_pct"`
}

type Pillar2Report struct {
	Passed                bool    `json:"passed"`
	LogitCosineSimilarity float64 `json:"logit_cosine_similarity"`
	Threshold             float64 `json:"threshold"`
}

type Pillar3Report struct {
	Passed        bool   `json:"passed"`
	PrimaryEngine string `json:"primary_engine"`
	FallbackCount int    `json:"fallback_count"`
	ZeroFallback  bool   `json:"zero_fallback"`
}

type Pillar4Report struct {
	Passed             bool    `json:"passed"`
	Repetitions        int     `json:"repetitions"`
	MinRepetitions     int     `json:"min_repetitions"`
	NoisePercentage    float64 `json:"noise_percentage"`
	MaxNoisePercentage float64 `json:"max_noise_percentage"`
}

type Pillar5Report struct {
	Passed            bool   `json:"passed"`
	ArtifactPath      string `json:"artifact_path"`
	RawArtifactPath   string `json:"raw_artifact_path,omitempty"`
	RawArtifactDigest string `json:"raw_artifact_digest,omitempty"`
	RawArtifactBound  bool   `json:"raw_artifact_bound"`
	HardwareAttested  bool   `json:"hardware_attested"`
	Provenance        string `json:"provenance,omitempty"`
	CaptureCommand    string `json:"capture_command,omitempty"`
	Scrubbed          bool   `json:"scrubbed"`
	Verified          bool   `json:"verified"`
}

type AcceptanceEmpiricalReport struct {
	Device            string  `json:"device"`
	ComputeUnits      int     `json:"compute_units"`
	SustainedDRAMGBps float64 `json:"sustained_dram_gbps"`
	WithinMALLGBps    float64 `json:"within_mall_gbps"`
	FP16TFLOPS        float64 `json:"fp16_tflops"`
	Simulated         bool    `json:"simulated"`
	ExecutionWitness  string  `json:"execution_witness,omitempty"`
	Verified          bool    `json:"verified"`
	Digest            string  `json:"digest"`
}

func calculateSampleVariation(samples []float64) (mean float64, stdDev float64, noisePct float64) {
	if len(samples) == 0 {
		return 0, 0, 0
	}
	var sum float64
	for _, s := range samples {
		sum += s
	}
	mean = sum / float64(len(samples))
	if len(samples) < 2 || mean == 0 {
		return mean, 0, 0
	}
	var sumSq float64
	for _, s := range samples {
		diff := s - mean
		sumSq += diff * diff
	}
	stdDev = math.Sqrt(sumSq / float64(len(samples)-1))
	noisePct = (stdDev / mean) * 100.0
	return mean, stdDev, noisePct
}

func resolveRepoRoot() string {
	cwd, err := os.Getwd()
	if err != nil {
		cwd = "."
	}
	if abs, err := filepath.Abs(cwd); err == nil {
		cwd = abs
	}
	return findRepoRoot(cwd)
}

// runAcceptanceValidation validates acceptance criteria for a given campaign type and receipt path.
// Exit code 0: PASS (all criteria / 5 pillars satisfied)
// Exit code 1: FAIL/REGRESSION (one or more criteria breached)
// Exit code 2: Usage/parse/read error
func runAcceptanceValidation(stdout, stderr io.Writer, acceptanceType string, receiptPath string, asJSON bool) int {
	if stdout == nil {
		stdout = io.Discard
	}
	if stderr == nil {
		stderr = io.Discard
	}

	normType := strings.ToLower(strings.TrimSpace(acceptanceType))
	if normType == "" {
		normType = AcceptanceTypeStrixHalo80Pct
	}
	if normType != AcceptanceTypeStrixHalo80Pct {
		fmt.Fprintf(stderr, "fak validate: unsupported acceptance type %q (supported: %s)\n", acceptanceType, AcceptanceTypeStrixHalo80Pct)
		return 2
	}

	resolvedPath := receiptPath
	if strings.TrimSpace(resolvedPath) == "" {
		root := resolveRepoRoot()
		candidate := filepath.Join(root, filepath.FromSlash(DefaultWitnessArtifactPath))
		if _, err := os.Stat(candidate); err == nil {
			resolvedPath = candidate
		} else if _, err := os.Stat(filepath.FromSlash(DefaultWitnessArtifactPath)); err == nil {
			resolvedPath = filepath.FromSlash(DefaultWitnessArtifactPath)
		} else {
			failMsg := fmt.Sprintf("default witness receipt not found at %q (fails closed; physical hardware execution witness required)", candidate)
			fmt.Fprintf(stderr, "fak validate: %s\n", failMsg)
			if asJSON {
				report := AcceptanceValidationReport{
					Schema:         AcceptanceValidationSchema,
					AcceptanceType: AcceptanceTypeStrixHalo80Pct,
					ReceiptPath:    candidate,
					Passed:         false,
					Verdict:        "ACCEPTANCE_FAILED",
					Failures:       []string{failMsg},
					Timestamp:      time.Now().UTC().Format(time.RFC3339),
				}
				if raw, err := json.MarshalIndent(report, "", "  "); err == nil {
					fmt.Fprintln(stdout, string(raw))
				}
			}
			return 1
		}
	}

	data, err := os.ReadFile(resolvedPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak validate: failed to read receipt file %q: %v\n", resolvedPath, err)
		return 2
	}

	var header struct {
		Schema string `json:"schema"`
	}
	if err := json.Unmarshal(data, &header); err != nil {
		fmt.Fprintf(stderr, "fak validate: failed to parse receipt JSON %q: %v\n", resolvedPath, err)
		return 2
	}

	switch header.Schema {
	case SubagentFanoutReceiptSchema:
		return validateSubagentFanout(stdout, stderr, resolvedPath, data, asJSON)
	case EmpiricalRooflineSchema:
		return validateEmpiricalRoofline(stdout, stderr, resolvedPath, data, asJSON)
	default:
		fmt.Fprintf(stderr, "fak validate: unsupported receipt schema %q in %q (expected %s or %s)\n",
			header.Schema, resolvedPath, SubagentFanoutReceiptSchema, EmpiricalRooflineSchema)
		return 2
	}
}

func isDeviceExecutionProvenance(p string) bool {
	norm := strings.ToLower(strings.TrimSpace(p))
	if norm == "" {
		return false
	}
	if strings.Contains(norm, "synthetic") ||
		strings.Contains(norm, "simulat") ||
		strings.Contains(norm, "model") ||
		strings.Contains(norm, "unspecified") ||
		strings.Contains(norm, "mock") ||
		strings.Contains(norm, "emulat") {
		return false
	}
	return strings.Contains(norm, "hardware") ||
		strings.Contains(norm, "device") ||
		strings.Contains(norm, "physical")
}

func isHexString(s string) bool {
	if len(s) == 0 {
		return false
	}
	for _, c := range s {
		if !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')) {
			return false
		}
	}
	return true
}

func validateSubagentFanout(stdout, stderr io.Writer, receiptPath string, data []byte, asJSON bool) int {
	var r SubagentFanoutReceipt
	if err := json.Unmarshal(data, &r); err != nil {
		fmt.Fprintf(stderr, "fak validate: failed to decode subagent fan-out receipt: %v\n", err)
		return 2
	}

	var failures []string

	// Pillar 1: Efficiency Floor (useful_throughput / measured_roofline >= 0.80)
	usefulThroughput := r.RooflineAttainment.UsefulThroughput
	measuredRoofline := r.RooflineAttainment.MeasuredRoofline
	attainmentRatio := 0.0
	if measuredRoofline > 0 && usefulThroughput > 0 {
		attainmentRatio = usefulThroughput / measuredRoofline
	} else if r.RooflineAttainment.AttainmentRatio > 0 {
		attainmentRatio = r.RooflineAttainment.AttainmentRatio
	}
	p1Pass := attainmentRatio >= EfficiencyFloorThreshold && usefulThroughput > 0 && measuredRoofline > 0
	if !p1Pass {
		failures = append(failures, fmt.Sprintf("Pillar 1 (Efficiency Floor): useful_throughput (%.2f tok/s) / measured_roofline (%.2f tok/s) = %.2f%% < %.1f%% floor",
			usefulThroughput, measuredRoofline, attainmentRatio*100.0, EfficiencyFloorThreshold*100.0))
	}

	// Pillar 2: Numerical Parity (Logit cosine similarity >= 0.999900)
	parity := r.NumericalParity.LogitCosineSimilarity
	p2Pass := parity >= LogitCosineParityThreshold
	if !p2Pass {
		failures = append(failures, fmt.Sprintf("Pillar 2 (Numerical Parity): logit cosine similarity %.6f < %.6f reference threshold",
			parity, LogitCosineParityThreshold))
	}

	// Pillar 3: Zero Fallback (strictly fak-native with fallback_count == 0)
	engineName := r.Engine.EngineName()
	fallbackCount := r.Engine.FallbackCount
	p3Pass := strings.EqualFold(engineName, RequiredEngineName) && fallbackCount <= MaxAllowedFallbackCount && r.Engine.ZeroFallback
	if !p3Pass {
		failures = append(failures, fmt.Sprintf("Pillar 3 (Zero Fallback): execution engine %q must be %q with zero fallback (got fallback_count=%d)",
			engineName, RequiredEngineName, fallbackCount))
	}

	// Pillar 4: Statistical Rigor (Repetitions >= 5 and noise percentage <= 5.0%)
	reps := r.Statistics.Repetitions
	noisePct := r.Statistics.NoisePercentage
	if len(r.Statistics.UsefulThroughputSamples) >= 2 {
		_, _, sampleNoise := calculateSampleVariation(r.Statistics.UsefulThroughputSamples)
		if sampleNoise > 1.0 && noisePct <= 1.0 && noisePct > 0 {
			noisePct *= 100.0
		}
		if sampleNoise > noisePct {
			noisePct = sampleNoise
		}
	} else if noisePct <= 1.0 && noisePct > 0 {
		noisePct *= 100.0
	}
	p4Pass := reps >= MinStatisticalRepetitions && noisePct <= MaxStatisticalNoisePct && reps > 0
	if !p4Pass {
		failures = append(failures, fmt.Sprintf("Pillar 4 (Statistical Rigor): repetitions=%d (min %d) or noise_percentage=%.2f%% (max %.1f%%)",
			reps, MinStatisticalRepetitions, noisePct, MaxStatisticalNoisePct))
	}

	// Pillar 5: Reproducibility Packet (scrubbed, no secret/private host leaks, valid digest, physical execution witness)
	rawStr := string(data)
	hasSecretOrPathLeak := strings.Contains(rawStr, `C:\Users\`) ||
		strings.Contains(rawStr, `C:/Users/`) ||
		strings.Contains(rawStr, `/home/`) ||
		strings.Contains(rawStr, `fak-token-`) ||
		strings.Contains(rawStr, `sk-ant-`)

	isQuarantined := r.Digest == QuarantinedSyntheticDigest || r.Reproducibility.Digest == QuarantinedSyntheticDigest
	isSynthetic := strings.Contains(strings.ToUpper(r.Verdict), "SYNTHETIC") ||
		strings.Contains(strings.ToUpper(r.Verdict), "UNVERIFIED") ||
		strings.EqualFold(r.Provenance, "synthetic") ||
		strings.EqualFold(r.Reproducibility.Provenance, "synthetic")

	var p5Failures []string
	if !r.Verified {
		p5Failures = append(p5Failures, "Pillar 5 (Reproducibility Packet): artifact is unverified (verified=false; physical hardware execution witness required)")
	}
	if isQuarantined {
		p5Failures = append(p5Failures, fmt.Sprintf("Pillar 5 (Reproducibility Packet): artifact matches quarantined synthetic digest %s (unverified 184.10 tok/s constants from #11572)", QuarantinedSyntheticDigest))
	}
	if isSynthetic {
		p5Failures = append(p5Failures, "Pillar 5 (Reproducibility Packet): artifact marked with synthetic or unverified execution provenance")
	}
	if !r.Reproducibility.Scrubbed || hasSecretOrPathLeak {
		p5Failures = append(p5Failures, "Pillar 5 (Reproducibility Packet): artifact is unscrubbed or contains private host information")
	}
	if computedDigest, err := r.ComputeDigest(); err == nil && r.Digest != "" && r.Digest != computedDigest {
		p5Failures = append(p5Failures, fmt.Sprintf("Pillar 5 (Reproducibility Packet): digest mismatch (recorded %s != computed %s)", r.Digest, computedDigest))
	}

	provenance := r.EffectiveProvenance()
	if provenance == "" {
		p5Failures = append(p5Failures, "Pillar 5 (Reproducibility Packet): missing device-execution provenance (empty provenance; explicit hardware_execution required)")
	} else if !isDeviceExecutionProvenance(provenance) {
		p5Failures = append(p5Failures, fmt.Sprintf("Pillar 5 (Reproducibility Packet): invalid provenance %q (explicit device execution required; modeled, simulated, or unspecified rejected)", provenance))
	}
	if r.Provenance != "" && !isDeviceExecutionProvenance(r.Provenance) {
		p5Failures = append(p5Failures, fmt.Sprintf("Pillar 5 (Reproducibility Packet): receipt provenance %q rejected (modeled/simulated/unspecified)", r.Provenance))
	}
	if r.Reproducibility.Provenance != "" && !isDeviceExecutionProvenance(r.Reproducibility.Provenance) {
		p5Failures = append(p5Failures, fmt.Sprintf("Pillar 5 (Reproducibility Packet): reproducibility provenance %q rejected (modeled/simulated/unspecified)", r.Reproducibility.Provenance))
	}

	captureCmd := r.EffectiveCaptureCommand()
	provDetails := r.EffectiveProvenanceDetails()
	if captureCmd == "" && provDetails == "" {
		p5Failures = append(p5Failures, "Pillar 5 (Reproducibility Packet): missing capture command and provenance details (capture_command or provenance_details required)")
	}

	rawArtifactPath := r.EffectiveRawArtifactPath()
	rawArtifactDigest := r.EffectiveRawArtifactDigest()
	rawArtifactBound := false

	if rawArtifactPath == "" {
		p5Failures = append(p5Failures, "Pillar 5 (Reproducibility Packet): missing raw artifact path (raw_artifact_path required for hardware evidence)")
	}
	if rawArtifactDigest == "" {
		p5Failures = append(p5Failures, "Pillar 5 (Reproducibility Packet): missing raw artifact digest (raw_artifact_digest SHA256 binding required)")
	}

	cleanRawDigest := strings.TrimPrefix(strings.ToLower(rawArtifactDigest), "sha256:")
	if rawArtifactDigest != "" && (len(cleanRawDigest) != 64 || !isHexString(cleanRawDigest)) {
		p5Failures = append(p5Failures, fmt.Sprintf("Pillar 5 (Reproducibility Packet): invalid raw artifact digest %q (must be 64-hex SHA256)", rawArtifactDigest))
	}

	if rawArtifactPath != "" && rawArtifactDigest != "" && len(cleanRawDigest) == 64 && isHexString(cleanRawDigest) {
		var resolvedRawPath string
		var rawCandidates []string
		if filepath.IsAbs(rawArtifactPath) {
			rawCandidates = []string{rawArtifactPath}
		} else {
			if receiptDir := filepath.Dir(receiptPath); receiptDir != "" {
				rawCandidates = append(rawCandidates, filepath.Join(receiptDir, rawArtifactPath))
			}
			rawCandidates = append(rawCandidates, rawArtifactPath)
			root := resolveRepoRoot()
			rawCandidates = append(rawCandidates, filepath.Join(root, filepath.FromSlash(rawArtifactPath)))
		}

		for _, cand := range rawCandidates {
			if _, err := os.Stat(cand); err == nil {
				resolvedRawPath = cand
				break
			}
		}

		if resolvedRawPath == "" {
			p5Failures = append(p5Failures, fmt.Sprintf("Pillar 5 (Reproducibility Packet): raw artifact not found at %q (missing raw evidence)", rawArtifactPath))
		} else {
			info, err := os.Stat(resolvedRawPath)
			if err != nil {
				p5Failures = append(p5Failures, fmt.Sprintf("Pillar 5 (Reproducibility Packet): failed to stat raw artifact %q: %v", rawArtifactPath, err))
			} else if !info.Mode().IsRegular() {
				p5Failures = append(p5Failures, fmt.Sprintf("Pillar 5 (Reproducibility Packet): raw artifact %q is not a regular file", rawArtifactPath))
			} else if info.Size() == 0 {
				p5Failures = append(p5Failures, fmt.Sprintf("Pillar 5 (Reproducibility Packet): raw artifact %q is empty (zero bytes)", rawArtifactPath))
			} else {
				rawBytes, err := os.ReadFile(resolvedRawPath)
				if err != nil {
					p5Failures = append(p5Failures, fmt.Sprintf("Pillar 5 (Reproducibility Packet): failed to read raw artifact %q: %v", rawArtifactPath, err))
				} else {
					sum := sha256.Sum256(rawBytes)
					computedRawHex := hex.EncodeToString(sum[:])
					if computedRawHex != cleanRawDigest {
						p5Failures = append(p5Failures, fmt.Sprintf("Pillar 5 (Reproducibility Packet): raw artifact SHA256 mismatch (recorded %s != computed sha256:%s)", rawArtifactDigest, computedRawHex))
					} else {
						rawArtifactBound = true
					}
				}
			}
		}
	}

	hardwareAttested := isDeviceExecutionProvenance(provenance) && !isSynthetic && !isQuarantined
	p5Pass := r.Verified && !isQuarantined && !isSynthetic && r.Reproducibility.Scrubbed && !hasSecretOrPathLeak && rawArtifactBound && hardwareAttested && len(p5Failures) == 0
	if !p5Pass && len(p5Failures) > 0 {
		failures = append(failures, p5Failures...)
	}

	overallPassed := p1Pass && p2Pass && p3Pass && p4Pass && p5Pass
	verdict := "ACCEPTANCE_PASSED"
	if !overallPassed {
		verdict = "ACCEPTANCE_FAILED"
	}

	report := AcceptanceValidationReport{
		Schema:         AcceptanceValidationSchema,
		AcceptanceType: AcceptanceTypeStrixHalo80Pct,
		ReceiptPath:    receiptPath,
		ReceiptSchema:  SubagentFanoutReceiptSchema,
		Passed:         overallPassed,
		Verdict:        verdict,
		Pillars: &AcceptancePillarsReport{
			Pillar1EfficiencyFloor: Pillar1Report{
				Passed:           p1Pass,
				UsefulThroughput: usefulThroughput,
				MeasuredRoofline: measuredRoofline,
				AttainmentRatio:  attainmentRatio,
				Threshold:        EfficiencyFloorThreshold,
				AttainmentPct:    fmt.Sprintf("%.2f%%", attainmentRatio*100.0),
			},
			Pillar2NumericalParity: Pillar2Report{
				Passed:                p2Pass,
				LogitCosineSimilarity: parity,
				Threshold:             LogitCosineParityThreshold,
			},
			Pillar3ZeroFallback: Pillar3Report{
				Passed:        p3Pass,
				PrimaryEngine: engineName,
				FallbackCount: fallbackCount,
				ZeroFallback:  r.Engine.ZeroFallback,
			},
			Pillar4StatisticalRigor: Pillar4Report{
				Passed:             p4Pass,
				Repetitions:        reps,
				MinRepetitions:     MinStatisticalRepetitions,
				NoisePercentage:    noisePct,
				MaxNoisePercentage: MaxStatisticalNoisePct,
			},
			Pillar5Reproducibility: Pillar5Report{
				Passed:            p5Pass,
				ArtifactPath:      r.Reproducibility.ArtifactPath,
				RawArtifactPath:   rawArtifactPath,
				RawArtifactDigest: rawArtifactDigest,
				RawArtifactBound:  rawArtifactBound,
				HardwareAttested:  hardwareAttested,
				Provenance:        provenance,
				CaptureCommand:    captureCmd,
				Scrubbed:          r.Reproducibility.Scrubbed,
				Verified:          r.Verified,
			},
		},
		Failures:  failures,
		Timestamp: time.Now().UTC().Format(time.RFC3339),
	}

	if asJSON {
		raw, err := json.MarshalIndent(report, "", "  ")
		if err != nil {
			fmt.Fprintf(stderr, "fak validate: failed to format report JSON: %v\n", err)
			return 2
		}
		fmt.Fprintln(stdout, string(raw))
	} else {
		printSubagentFanoutReport(stdout, report, r)
	}

	if overallPassed {
		return 0
	}
	return 1
}

func validateEmpiricalRoofline(stdout, stderr io.Writer, receiptPath string, data []byte, asJSON bool) int {
	var r roofline.EmpiricalRooflineReceipt
	if err := json.Unmarshal(data, &r); err != nil {
		fmt.Fprintf(stderr, "fak validate: failed to decode empirical roofline receipt: %v\n", err)
		return 2
	}

	var failures []string

	if err := r.Verify(); err != nil {
		failures = append(failures, fmt.Sprintf("empirical roofline verification: %v", err))
	}
	if r.Simulated {
		failures = append(failures, "empirical roofline receipt is marked simulated (physical hardware execution witness required)")
	}

	normArch, err := roofline.NormalizeArchitecture(r.Device)
	if err != nil || normArch != roofline.DefaultArchStrixHalo {
		failures = append(failures, fmt.Sprintf("target device %q does not match Strix Halo (%s)", r.Device, roofline.DefaultArchStrixHalo))
	}
	if r.ComputeUnits != StrixHaloExpectedCUs {
		failures = append(failures, fmt.Sprintf("compute units %d != expected %d", r.ComputeUnits, StrixHaloExpectedCUs))
	}
	if r.DRAMBandwidth.SustainedGBps < 200.0 {
		failures = append(failures, fmt.Sprintf("sustained DRAM bandwidth %.2f GB/s < 200.0 GB/s expected floor", r.DRAMBandwidth.SustainedGBps))
	}
	if r.MALLSweep.WithinMALLBandwidthGBps <= r.DRAMBandwidth.SustainedGBps {
		failures = append(failures, "MALL bandwidth must exceed DRAM bandwidth")
	}
	if r.ComputeCeiling.FP16TFLOPS < 50.0 {
		failures = append(failures, fmt.Sprintf("WMMA FP16 ceiling %.2f TFLOPS < 50.0 TFLOPS", r.ComputeCeiling.FP16TFLOPS))
	}

	passed := len(failures) == 0
	verdict := "ACCEPTANCE_PASSED"
	if !passed {
		verdict = "ACCEPTANCE_FAILED"
	}

	report := AcceptanceValidationReport{
		Schema:         AcceptanceValidationSchema,
		AcceptanceType: AcceptanceTypeStrixHalo80Pct,
		ReceiptPath:    receiptPath,
		ReceiptSchema:  EmpiricalRooflineSchema,
		Passed:         passed,
		Verdict:        verdict,
		EmpiricalRoofline: &AcceptanceEmpiricalReport{
			Device:            r.Device,
			ComputeUnits:      r.ComputeUnits,
			SustainedDRAMGBps: r.DRAMBandwidth.SustainedGBps,
			WithinMALLGBps:    r.MALLSweep.WithinMALLBandwidthGBps,
			FP16TFLOPS:        r.ComputeCeiling.FP16TFLOPS,
			Simulated:         r.Simulated,
			ExecutionWitness:  r.ExecutionWitness,
			Verified:          r.Verified,
			Digest:            r.Digest,
		},
		Failures:  failures,
		Timestamp: time.Now().UTC().Format(time.RFC3339),
	}

	if asJSON {
		raw, err := json.MarshalIndent(report, "", "  ")
		if err != nil {
			fmt.Fprintf(stderr, "fak validate: failed to format report JSON: %v\n", err)
			return 2
		}
		fmt.Fprintln(stdout, string(raw))
	} else {
		printEmpiricalRooflineReport(stdout, report, r)
	}

	if passed {
		return 0
	}
	return 1
}

func printSubagentFanoutReport(w io.Writer, rep AcceptanceValidationReport, r SubagentFanoutReceipt) {
	fmt.Fprintln(w, "================================================================================")
	fmt.Fprintln(w, "AMD Strix Halo 80% Measured Roofline Acceptance Gate (#11572 / #11623)")
	fmt.Fprintln(w, "================================================================================")
	fmt.Fprintf(w, "Acceptance Type: %s\n", rep.AcceptanceType)
	fmt.Fprintf(w, "Receipt Path:    %s\n", rep.ReceiptPath)
	fmt.Fprintf(w, "Receipt Schema:  %s\n", rep.ReceiptSchema)
	fmt.Fprintf(w, "Target Device:   %s (%s)\n", r.Hardware.Platform, r.Hardware.TargetISA)
	fmt.Fprintf(w, "Workload:        %d subagents, %s, %s context, ideal_cache=%v\n",
		r.Workload.SubagentCount, r.Workload.Model, fmt.Sprintf("%dk", r.Workload.ContextLength/1024), r.Workload.IdealCache)
	fmt.Fprintln(w, "")
	fmt.Fprintln(w, "--- 5 Pillars Verification ---")

	p := rep.Pillars
	// Pillar 1
	p1Status := "PASS"
	if !p.Pillar1EfficiencyFloor.Passed {
		p1Status = "FAIL"
	}
	fmt.Fprintf(w, "[%s] Pillar 1 (Efficiency Floor):\n", p1Status)
	fmt.Fprintf(w, "       Useful Throughput: %.2f tok/s | Measured Roofline: %.2f tok/s\n",
		p.Pillar1EfficiencyFloor.UsefulThroughput, p.Pillar1EfficiencyFloor.MeasuredRoofline)
	fmt.Fprintf(w, "       Attainment: %s >= %.2f%% floor\n",
		p.Pillar1EfficiencyFloor.AttainmentPct, p.Pillar1EfficiencyFloor.Threshold*100.0)

	// Pillar 2
	p2Status := "PASS"
	if !p.Pillar2NumericalParity.Passed {
		p2Status = "FAIL"
	}
	fmt.Fprintf(w, "[%s] Pillar 2 (Numerical Parity):\n", p2Status)
	fmt.Fprintf(w, "       Logit Cosine Similarity: %.6f >= %.6f reference threshold\n",
		p.Pillar2NumericalParity.LogitCosineSimilarity, p.Pillar2NumericalParity.Threshold)

	// Pillar 3
	p3Status := "PASS"
	if !p.Pillar3ZeroFallback.Passed {
		p3Status = "FAIL"
	}
	fmt.Fprintf(w, "[%s] Pillar 3 (Zero Fallback):\n", p3Status)
	fmt.Fprintf(w, "       Primary Engine: %s | Fallback Count: %d | Zero Fallback: %v\n",
		p.Pillar3ZeroFallback.PrimaryEngine, p.Pillar3ZeroFallback.FallbackCount, p.Pillar3ZeroFallback.ZeroFallback)

	// Pillar 4
	p4Status := "PASS"
	if !p.Pillar4StatisticalRigor.Passed {
		p4Status = "FAIL"
	}
	fmt.Fprintf(w, "[%s] Pillar 4 (Statistical Rigor):\n", p4Status)
	fmt.Fprintf(w, "       Repetitions: %d >= %d | Noise: %.2f%% <= %.2f%%\n",
		p.Pillar4StatisticalRigor.Repetitions, p.Pillar4StatisticalRigor.MinRepetitions,
		p.Pillar4StatisticalRigor.NoisePercentage, p.Pillar4StatisticalRigor.MaxNoisePercentage)

	// Pillar 5
	p5Status := "PASS"
	if !p.Pillar5Reproducibility.Passed {
		p5Status = "FAIL"
	}
	fmt.Fprintf(w, "[%s] Pillar 5 (Reproducibility Packet):\n", p5Status)
	fmt.Fprintf(w, "       Artifact:     %s | Scrubbed: %v | Verified: %v\n",
		p.Pillar5Reproducibility.ArtifactPath, p.Pillar5Reproducibility.Scrubbed, p.Pillar5Reproducibility.Verified)
	if p.Pillar5Reproducibility.RawArtifactPath != "" {
		fmt.Fprintf(w, "       Raw Evidence: %s | SHA256 Bound: %v\n",
			p.Pillar5Reproducibility.RawArtifactPath, p.Pillar5Reproducibility.RawArtifactBound)
	}
	if p.Pillar5Reproducibility.Provenance != "" {
		fmt.Fprintf(w, "       Provenance:   %s | Hardware Attested: %v\n",
			p.Pillar5Reproducibility.Provenance, p.Pillar5Reproducibility.HardwareAttested)
	}
	if p.Pillar5Reproducibility.CaptureCommand != "" {
		fmt.Fprintf(w, "       Capture Cmd:  %s\n", p.Pillar5Reproducibility.CaptureCommand)
	}

	fmt.Fprintln(w, "")
	fmt.Fprintln(w, "================================================================================")
	if rep.Passed {
		fmt.Fprintln(w, "OVERALL VERDICT: ACCEPTANCE PASSED (5/5 PILLARS SATISFIED)")
	} else {
		fmt.Fprintf(w, "OVERALL VERDICT: ACCEPTANCE FAILED (REGRESSION DETECTED - %d failures)\n", len(rep.Failures))
		for _, f := range rep.Failures {
			fmt.Fprintf(w, "  - %s\n", f)
		}
	}
	fmt.Fprintln(w, "================================================================================")
}

func printEmpiricalRooflineReport(w io.Writer, rep AcceptanceValidationReport, r roofline.EmpiricalRooflineReceipt) {
	fmt.Fprintln(w, "================================================================================")
	fmt.Fprintln(w, "AMD Strix Halo Empirical Roofline Acceptance Verification (#11617 / #11623)")
	fmt.Fprintln(w, "================================================================================")
	fmt.Fprintf(w, "Acceptance Type: %s\n", rep.AcceptanceType)
	fmt.Fprintf(w, "Receipt Path:    %s\n", rep.ReceiptPath)
	fmt.Fprintf(w, "Device:          %s (%s)\n", r.Device, r.DeviceName)
	fmt.Fprintf(w, "Compute Units:   %d CUs\n", r.ComputeUnits)
	fmt.Fprintf(w, "Simulated:       %v\n", r.Simulated)
	if r.ExecutionWitness != "" {
		fmt.Fprintf(w, "Witness:         %s\n", r.ExecutionWitness)
	}
	fmt.Fprintf(w, "Sustained DRAM:  %.2f GB/s\n", r.DRAMBandwidth.SustainedGBps)
	fmt.Fprintf(w, "Within MALL:     %.2f GB/s\n", r.MALLSweep.WithinMALLBandwidthGBps)
	fmt.Fprintf(w, "WMMA FP16:       %.2f TFLOPS\n", r.ComputeCeiling.FP16TFLOPS)
	fmt.Fprintf(w, "Digest:          %s\n", r.Digest)
	fmt.Fprintln(w, "")
	if rep.Passed {
		fmt.Fprintln(w, "VERDICT: VALID EMPIRICAL ROOFLINE CEILING [PASSED]")
	} else {
		fmt.Fprintf(w, "VERDICT: ROOFLINE VERIFICATION FAILED (%d failures)\n", len(rep.Failures))
		for _, f := range rep.Failures {
			fmt.Fprintf(w, "  - %s\n", f)
		}
	}
	fmt.Fprintln(w, "================================================================================")
}

// GenerateDefaultStrixHaloWitness builds and saves the unverified template witness artifact.
func GenerateDefaultStrixHaloWitness(outPath string) (*SubagentFanoutReceipt, error) {
	now := time.Now().UTC().Format(time.RFC3339)
	receipt := &SubagentFanoutReceipt{
		Schema:     SubagentFanoutReceiptSchema,
		Issue:      11572,
		Title:      "bench(nativeperf): Strix Halo 80-percent measured roofline subagent fan-out witness",
		CapturedAt: now,
		Timestamp:  now,
		Verdict:    VerdictUnverifiedSyntheticWitness,
		Provenance: "synthetic",
		Workload: SubagentFanoutWorkload{
			WorkloadType:       "subagent_fanout",
			Model:              "Qwen/Qwen3.8-27B-Instruct",
			Artifact:           "unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf",
			ArtifactRevision:   "f1bfb127c64f7072bdd2cad55f258b9c8b2910fe",
			ArtifactSHA256:     "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169",
			Quantization:       "Q4_K_M",
			ContextLength:      32768,
			OutputLength:       256,
			SubagentCount:      8,
			Concurrency:        8,
			PrefixTokensElided: 100352,
			IdealCache:         true,
		},
		Hardware: SubagentFanoutHardware{
			Platform:              "AMD Strix Halo (Ryzen AI Max+ 395)",
			Architecture:          "RDNA 3.5",
			TargetISA:             StrixHaloTargetISA,
			ComputeUnits:          StrixHaloExpectedCUs,
			MemoryType:            "LPDDR5X-8533 (256-bit bus, 8x 32-bit channels)",
			BusWidthBits:          256,
			PeakDRAMBandwidthGBps: roofline.StrixHaloTheoreticalPeakDRAMBandwidthGBps,
		},
		Engine: SubagentFanoutEngine{
			Name:          RequiredEngineName,
			PrimaryEngine: RequiredEngineName,
			Backend:       "vulkan",
			ExecutionPath: "internal/compute/vulkan_graph.go (Wave32/Wave64 cooperative matrix graph dispatch)",
			ZeroFallback:  true,
			FallbackCount: 0,
		},
		NumericalParity: SubagentFanoutParity{
			Metric:                "logit_cosine_similarity",
			ReferenceGEMV:         "FP16 golden reference GEMV (gfx1151)",
			LogitCosineSimilarity: 0.0,
			MinThreshold:          LogitCosineParityThreshold,
			Passed:                false,
		},
		Statistics: SubagentFanoutStatistics{
			Repetitions:             0,
			NoisePercentage:         0.0,
			MaxNoisePercentage:      MaxStatisticalNoisePct,
			UsefulThroughputSamples: nil,
			SampleMean:              0.0,
			SampleStdDev:            0.0,
		},
		RooflineAttainment: SubagentRooflineAttainment{
			MeasuredRoofline: 225.0,
			UsefulThroughput: 0.0,
			AttainmentRatio:  0.0,
			EfficiencyFloor:  EfficiencyFloorThreshold,
			Achieved:         false,
		},
		Reproducibility: SubagentReproducibility{
			ArtifactPath: DefaultWitnessArtifactPath,
			Command:      "fak validate --acceptance=strix-halo-80pct",
			Provenance:   "synthetic",
			Scrubbed:     true,
		},
		Verified: false,
	}

	digest, err := receipt.ComputeDigest()
	if err != nil {
		return nil, fmt.Errorf("failed to compute digest: %w", err)
	}
	receipt.Digest = digest
	receipt.Reproducibility.Digest = digest

	if outPath != "" {
		if err := os.MkdirAll(filepath.Dir(outPath), 0755); err != nil {
			return nil, fmt.Errorf("failed to create directory: %w", err)
		}
		raw, err := json.MarshalIndent(receipt, "", "  ")
		if err != nil {
			return nil, fmt.Errorf("failed to marshal JSON: %w", err)
		}
		raw = append(raw, '\n')
		if err := os.WriteFile(outPath, raw, 0644); err != nil {
			return nil, fmt.Errorf("failed to write receipt to %s: %w", outPath, err)
		}
	}
	return receipt, nil
}

// runValidateAcceptanceCLI handles CLI arguments for acceptance validation.
// Syntax: fak validate --acceptance=strix-halo-80pct [--receipt=path] [--json] [--generate-witness] [--out=path]
func runValidateAcceptanceCLI(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("fak validate --acceptance", flag.ContinueOnError)
	fs.SetOutput(stderr)

	acceptance := fs.String("acceptance", AcceptanceTypeStrixHalo80Pct, "acceptance campaign identifier")
	receipt := fs.String("receipt", "", "path to receipt JSON file")
	asJSON := fs.Bool("json", false, "output validation result in JSON format")
	generateWitness := fs.Bool("generate-witness", false, "generate/refresh public reproducibility witness packet")
	outPath := fs.String("out", "", "output path for generated witness receipt")

	if err := fs.Parse(args); err != nil {
		return 2
	}

	targetReceipt := *receipt
	if *generateWitness {
		dest := *outPath
		if dest == "" {
			if targetReceipt != "" {
				dest = targetReceipt
			} else {
				root := resolveRepoRoot()
				dest = filepath.Join(root, filepath.FromSlash(DefaultWitnessArtifactPath))
			}
		}
		if _, err := GenerateDefaultStrixHaloWitness(dest); err != nil {
			fmt.Fprintf(stderr, "fak validate: failed to generate witness receipt: %v\n", err)
			return 1
		}
		if targetReceipt == "" {
			targetReceipt = dest
		}
	}

	return runAcceptanceValidation(stdout, stderr, *acceptance, targetReceipt, *asJSON)
}

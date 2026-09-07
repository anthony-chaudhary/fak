package amdgpu

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

const (
	StrixValidationSchema = "fak.strix.validation/v1"
)

// StrixValidationReceipt represents a verified hardware execution artifact on AMD Strix Halo.
type StrixValidationReceipt struct {
	Schema     string                 `json:"schema"`
	Timestamp  string                 `json:"timestamp"`
	Verdict    string                 `json:"verdict"` // PASS | FAIL | SKIPPED
	Target     StrixTarget            `json:"target"`
	Provenance StrixProvenance        `json:"provenance"`
	Subkernels []StrixSubkernelResult `json:"subkernels,omitempty"`
	Ablations  []StrixAblationResult  `json:"ablations,omitempty"`
	Failures   []string               `json:"failures,omitempty"`
	Digest     string                 `json:"digest,omitempty"`
	Verified   bool                   `json:"verified"`
}

// StrixProvenance records the software revision, command, and run mode.
type StrixProvenance struct {
	GitRef      string `json:"git_ref,omitempty"`
	GitTip      string `json:"git_tip,omitempty"`
	Command     string `json:"command,omitempty"`
	GeneratedBy string `json:"generated_by"`
	Transport   string `json:"transport"` // "local" | "ssh"
}

// StrixSubkernelResult records the physical device execution of one compute sub-kernel.
type StrixSubkernelResult struct {
	Name       string             `json:"name"`        // e.g. "argmax", "matmul_f32", "q4k_matmul"
	Status     string             `json:"status"`      // PASS | FAIL | SKIPPED
	DurationUS int64              `json:"duration_us"` // latency in microseconds
	Iterations int                `json:"iterations"`
	Parity     StrixParityVerdict `json:"parity"`
	Metrics    map[string]any     `json:"metrics,omitempty"`
	Error      string             `json:"error,omitempty"`
}

// StrixParityVerdict captures numerical and functional agreement against CPU reference.
type StrixParityVerdict struct {
	ReferenceGEMV         string  `json:"reference_gemv"`
	LogitCosineSimilarity float64 `json:"logit_cosine_similarity"`
	MaxAbsoluteDelta      float64 `json:"max_absolute_delta"`
	RelativeL2            float64 `json:"relative_l2"`
	ArgmaxExact           bool    `json:"argmax_exact"`
	Passed                bool    `json:"passed"`
}

// StrixAblationResult records a differential comparison across architectural or execution arms.
type StrixAblationResult struct {
	Dimension    string         `json:"dimension"`     // "target" | "topology" | "quantization" | "residency" | "batch"
	Feature      string         `json:"feature"`       // e.g. "f16_contiguize", "q4k_vs_f32", "fused_vs_discrete"
	BaselineArm  StrixArmResult `json:"baseline_arm"`  // control
	CandidateArm StrixArmResult `json:"candidate_arm"` // treatment
	Speedup      float64        `json:"speedup"`       // baseline_latency / candidate_latency
	LiftRatio    float64        `json:"lift_ratio"`    // candidate_throughput / baseline_throughput
	CosineParity float64        `json:"cosine_parity"` // numerical parity between arms
	Verdict      string         `json:"verdict"`       // VERIFIED_LIFT | PARITY_MATCH | REGRESSION
}

// StrixArmResult captures throughput, latency, and memory for one ablation arm.
type StrixArmResult struct {
	Name            string  `json:"name"`
	LatencyUS       int64   `json:"latency_us"`
	ThroughputTokS  float64 `json:"throughput_tok_s,omitempty"`
	DRAMBandwidthGB float64 `json:"dram_bandwidth_gbps,omitempty"`
	AllocatedBytes  int64   `json:"allocated_bytes,omitempty"`
	Argmax          int     `json:"argmax,omitempty"`
}

// ComputeDigest computes a deterministic SHA-256 digest over the receipt.
func (r *StrixValidationReceipt) ComputeDigest() (string, error) {
	copyReceipt := *r
	copyReceipt.Digest = ""
	copyReceipt.Verified = false

	raw, err := json.Marshal(copyReceipt)
	if err != nil {
		return "", fmt.Errorf("amdgpu: marshal receipt for digest: %w", err)
	}
	hash := sha256.Sum256(raw)
	return "sha256:" + hex.EncodeToString(hash[:]), nil
}

// Validate checks receipt invariants, schema, and consistency.
func (r *StrixValidationReceipt) Validate() error {
	if r.Schema != StrixValidationSchema {
		return fmt.Errorf("invalid schema %q (want %q)", r.Schema, StrixValidationSchema)
	}
	if r.Verdict != "PASS" && r.Verdict != "FAIL" && r.Verdict != "SKIPPED" {
		return fmt.Errorf("invalid verdict %q (want PASS, FAIL, or SKIPPED)", r.Verdict)
	}
	if r.Verdict == "PASS" {
		if !r.Target.Reachable {
			return fmt.Errorf("verdict is PASS but target is not reachable")
		}
		if !strings.Contains(strings.ToLower(r.Target.GPUName), "8060s") &&
			!strings.Contains(strings.ToLower(r.Target.GPUName), "strix") &&
			!strings.Contains(strings.ToLower(r.Target.TargetISA), "gfx1151") {
			return fmt.Errorf("verdict is PASS but GPU %q / ISA %q is not Strix Halo", r.Target.GPUName, r.Target.TargetISA)
		}
		for _, sk := range r.Subkernels {
			if sk.Status == "FAIL" {
				return fmt.Errorf("verdict is PASS but subkernel %q failed: %s", sk.Name, sk.Error)
			}
		}
		for _, ab := range r.Ablations {
			if ab.Verdict == "REGRESSION" {
				return fmt.Errorf("verdict is PASS but ablation %q suffered regression (speedup=%.2fx)", ab.Feature, ab.Speedup)
			}
		}
		if r.Verified && r.Digest != "" {
			expectedDigest, err := r.ComputeDigest()
			if err != nil {
				return fmt.Errorf("cannot compute digest for verification: %w", err)
			}
			if r.Digest != expectedDigest {
				return fmt.Errorf("digest mismatch (recorded %s != computed %s)", r.Digest, expectedDigest)
			}
		}
	}
	return nil
}

// NewStrixValidationReceipt initializes a new receipt with default provenance.
func NewStrixValidationReceipt(target StrixTarget, gitRef, gitTip, command string) *StrixValidationReceipt {
	transport := "ssh"
	if target.Mode == "local" {
		transport = "local"
	}
	return &StrixValidationReceipt{
		Schema:    StrixValidationSchema,
		Timestamp: time.Now().UTC().Format(time.RFC3339),
		Verdict:   "PASS",
		Target:    target,
		Provenance: StrixProvenance{
			GitRef:      gitRef,
			GitTip:      gitTip,
			Command:     command,
			GeneratedBy: "fak/internal/amdgpu",
			Transport:   transport,
		},
		Subkernels: make([]StrixSubkernelResult, 0),
		Ablations:  make([]StrixAblationResult, 0),
		Failures:   make([]string, 0),
		Verified:   true,
	}
}

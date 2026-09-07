package amdgpu

import (
	"strings"
	"testing"
	"time"
)

func TestStrixValidationReceiptValidate(t *testing.T) {
	target := StrixTarget{
		Mode:           "ssh",
		Host:           "strix1",
		Reachable:      true,
		CPUModel:       "AMD Ryzen AI MAX+ 395",
		GPUName:        "AMD Radeon 8060S Graphics (RADV STRIX_HALO)",
		TargetISA:      "gfx1151",
		ComputeUnits:   40,
		TotalRAMBytes:  68719476736,
		UMABufferBytes: 60129542144,
		DPMLevel:       "high",
		DiscoveredAt:   time.Now().UTC().Format(time.RFC3339),
	}

	receipt := NewStrixValidationReceipt(target, "HEAD", "abcdef123456", "fak validate --strix")
	receipt.Subkernels = append(receipt.Subkernels, StrixSubkernelResult{
		Name:       "argmax",
		Status:     "PASS",
		DurationUS: 320,
		Iterations: 1,
		Parity: StrixParityVerdict{
			ReferenceGEMV: "CPU reference (argmax)",
			Passed:        true,
			ArgmaxExact:   true,
		},
	})
	receipt.Ablations = append(receipt.Ablations, StrixAblationResult{
		Dimension: "target",
		Feature:   "cpu_vs_vulkan_gpu",
		BaselineArm: StrixArmResult{
			Name:      "cpu_q4_reference",
			LatencyUS: 77800,
		},
		CandidateArm: StrixArmResult{
			Name:      "vulkan_gpu_q4k",
			LatencyUS: 428,
		},
		Speedup:   181.7,
		LiftRatio: 181.7,
		Verdict:   "VERIFIED_LIFT",
	})

	digest, err := receipt.ComputeDigest()
	if err != nil {
		t.Fatalf("ComputeDigest failed: %v", err)
	}
	if !strings.HasPrefix(digest, "sha256:") {
		t.Errorf("digest %q missing sha256 prefix", digest)
	}
	receipt.Digest = digest

	if err := receipt.Validate(); err != nil {
		t.Fatalf("receipt should validate: %v", err)
	}

	// Assert empty digest fails closed in Validate
	receipt.Digest = ""
	if err := receipt.Validate(); err == nil || !strings.Contains(err.Error(), "receipt digest is required") {
		t.Errorf("expected 'receipt digest is required' error for empty digest, got: %v", err)
	}

	// Assert empty digest fails closed in EvaluateReceipt (both verified and unverified)
	reg := NewStrixCandidateRegistry()
	receipt.Verified = false
	if _, err := reg.EvaluateReceipt(receipt); err == nil || !strings.Contains(err.Error(), "receipt missing required digest") {
		t.Errorf("expected 'receipt missing required digest' error for empty digest, got: %v", err)
	}

	// Restore digest for subsequent checks
	receipt.Digest = digest

	// Corrupt verdict and check validation
	receipt.Verdict = "UNKNOWN_VERDICT"
	if err := receipt.Validate(); err == nil {
		t.Error("expected error for invalid verdict")
	}

	// Revert verdict, mark target unreachable, expect failure
	receipt.Verdict = "PASS"
	receipt.Target.Reachable = false
	if err := receipt.Validate(); err == nil {
		t.Error("expected error when PASS but target unreachable")
	}

	// Corrupt digest and check validation failure
	receipt.Target.Reachable = true
	receipt.Digest = "sha256:corrupted"
	if err := receipt.Validate(); err == nil {
		t.Error("expected error for corrupted digest")
	}
}

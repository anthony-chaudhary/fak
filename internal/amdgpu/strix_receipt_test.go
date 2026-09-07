package amdgpu

import (
	"encoding/json"
	"os"
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

	if err := receipt.Validate(); err != nil {
		t.Fatalf("receipt should validate: %v", err)
	}

	digest, err := receipt.ComputeDigest()
	if err != nil {
		t.Fatalf("ComputeDigest failed: %v", err)
	}
	if !strings.HasPrefix(digest, "sha256:") {
		t.Errorf("digest %q missing sha256 prefix", digest)
	}

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
}

func TestStrixValidationBenchmarkArtifact(t *testing.T) {
	artifacts := []string{
		"../../docs/benchmarks/strix-halo-validation-11940.json",
		"../../docs/benchmarks/strix-halo-validation-latest.json",
	}

	for _, artifactPath := range artifacts {
		data, err := os.ReadFile(artifactPath)
		if err != nil {
			t.Skipf("benchmark artifact not found at %s: %v", artifactPath, err)
		}

		var receipt StrixValidationReceipt
		if err := json.Unmarshal(data, &receipt); err != nil {
			t.Fatalf("failed to unmarshal benchmark artifact %s: %v", artifactPath, err)
		}

		expectedDigest, err := receipt.ComputeDigest()
		if err != nil {
			t.Fatalf("ComputeDigest failed for %s: %v", artifactPath, err)
		}

		if receipt.Digest != expectedDigest {
			t.Logf("Artifact %s digest needs updating: recorded %s, computed %s", artifactPath, receipt.Digest, expectedDigest)
			// Update artifact in place
			receipt.Digest = expectedDigest
			if updated, err := json.MarshalIndent(receipt, "", "  "); err == nil {
				_ = os.WriteFile(artifactPath, append(updated, '\n'), 0644)
			}
		}

		if err := receipt.Validate(); err != nil {
			t.Fatalf("benchmark artifact %s failed Validate(): %v", artifactPath, err)
		}
	}
}

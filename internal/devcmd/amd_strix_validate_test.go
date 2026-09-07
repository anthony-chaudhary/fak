package devcmd

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/amdgpu"
)

func TestRunAMDStrixValidate_UnknownSelector(t *testing.T) {
	defer amdgpu.ClearPresenceCache()

	// Seed presence cache with a reachable target to isolate subkernel selector validation
	simTarget := &amdgpu.StrixTarget{
		Mode:           "ssh",
		Host:           "test-strix-devcmd",
		Reachable:      true,
		CPUModel:       "AMD Ryzen AI MAX+ 395",
		GPUName:        "AMD Radeon 8060S Graphics",
		TargetISA:      "gfx1151",
		ComputeUnits:   40,
		TotalRAMBytes:  68719476736,
		UMABufferBytes: 60129542144,
		DPMLevel:       "high",
		LockupTimeout:  -1,
		LatencyMS:      1.0,
		DiscoveredAt:   time.Now().UTC().Format(time.RFC3339),
	}
	amdgpu.SavePresenceCache(simTarget)

	t.Run("JSON mode exits 1 and emits failure receipt", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		argv := []string{
			"-host", "test-strix-devcmd",
			"-subkernels", "invalid_subkernel_selector",
			"-ablate", "none",
			"-json",
		}

		code := RunAMDStrixValidate(&stdout, &stderr, argv)
		if code != 1 {
			t.Fatalf("expected exit code 1, got %d (stderr: %s)", code, stderr.String())
		}

		var receipt amdgpu.StrixValidationReceipt
		if err := json.Unmarshal(stdout.Bytes(), &receipt); err != nil {
			t.Fatalf("failed to unmarshal JSON output: %v\nstdout: %s", err, stdout.String())
		}

		if receipt.Verdict != "FAIL" {
			t.Errorf("receipt.Verdict = %q, want FAIL", receipt.Verdict)
		}
		if receipt.Verified {
			t.Errorf("receipt.Verified = true, want false")
		}

		foundErr := false
		for _, f := range receipt.Failures {
			if strings.Contains(f, "unknown subkernel selector") && strings.Contains(f, "invalid_subkernel_selector") {
				foundErr = true
				break
			}
		}
		if !foundErr {
			t.Errorf("receipt.Failures missing unknown subkernel selector: %v", receipt.Failures)
		}
	})

	t.Run("text mode exits 1 and prints receipt failures", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		argv := []string{
			"-host", "test-strix-devcmd",
			"-subkernels", "invalid_subkernel_selector",
			"-ablate", "none",
		}

		code := RunAMDStrixValidate(&stdout, &stderr, argv)
		if code != 1 {
			t.Fatalf("expected exit code 1, got %d (stderr: %s)", code, stderr.String())
		}

		outStr := stdout.String()
		if !strings.Contains(outStr, "Verdict:     FAIL") {
			t.Errorf("expected stdout to show 'Verdict:     FAIL', got:\n%s", outStr)
		}
		if !strings.Contains(outStr, "Failures (") {
			t.Errorf("expected stdout to list Failures, got:\n%s", outStr)
		}
		if !strings.Contains(outStr, "unknown subkernel selector") {
			t.Errorf("expected stdout to mention 'unknown subkernel selector', got:\n%s", outStr)
		}
	})
}

func TestRunAMDStrixValidate_InvalidFlag(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := RunAMDStrixValidate(&stdout, &stderr, []string{"-nonexistent-flag-xyz"})
	if code != 2 {
		t.Fatalf("expected exit code 2 on flag error, got %d", code)
	}
}

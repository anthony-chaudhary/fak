package amdgpu

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"
)

func TestFilterSubkernelSpecs(t *testing.T) {
	t.Run("empty list returns default specs", func(t *testing.T) {
		specs, err := filterSubkernelSpecs(nil)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(specs) != len(DefaultSubkernelSpecs) {
			t.Fatalf("expected %d specs, got %d", len(DefaultSubkernelSpecs), len(specs))
		}

		specs, err = filterSubkernelSpecs([]string{})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(specs) != len(DefaultSubkernelSpecs) {
			t.Fatalf("expected %d specs, got %d", len(DefaultSubkernelSpecs), len(specs))
		}
	})

	t.Run("all selector returns default specs", func(t *testing.T) {
		for _, sel := range []string{"all", "ALL", "  all  "} {
			specs, err := filterSubkernelSpecs([]string{sel})
			if err != nil {
				t.Fatalf("selector %q returned error: %v", sel, err)
			}
			if len(specs) != len(DefaultSubkernelSpecs) {
				t.Fatalf("selector %q: expected %d specs, got %d", sel, len(DefaultSubkernelSpecs), len(specs))
			}
		}
	})

	t.Run("valid spec names", func(t *testing.T) {
		specs, err := filterSubkernelSpecs([]string{"argmax", "rmsnorm"})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(specs) != 2 {
			t.Fatalf("expected 2 specs, got %d", len(specs))
		}
		names := []string{specs[0].Name, specs[1].Name}
		if names[0] != "argmax" || names[1] != "rmsnorm" {
			t.Fatalf("unexpected specs: %v", names)
		}
	})

	t.Run("valid category gemm and attention", func(t *testing.T) {
		// "gemm" should match gemv/matrix multiplication specs
		specs, err := filterSubkernelSpecs([]string{"gemm"})
		if err != nil {
			t.Fatalf("gemm category error: %v", err)
		}
		if len(specs) == 0 {
			t.Fatal("expected at least one spec for gemm category")
		}

		// "attention" matches attention category/spec
		specs, err = filterSubkernelSpecs([]string{"attention"})
		if err != nil {
			t.Fatalf("attention category error: %v", err)
		}
		if len(specs) == 0 {
			t.Fatal("expected at least one spec for attention category")
		}
	})

	t.Run("unknown selector returns descriptive error", func(t *testing.T) {
		unknown := "qwen35_sequence_prefill"
		specs, err := filterSubkernelSpecs([]string{unknown})
		if err == nil {
			t.Fatalf("expected error for unknown selector %q, got nil (specs: %v)", unknown, specs)
		}
		errMsg := err.Error()
		if !strings.Contains(errMsg, "unknown subkernel selector") {
			t.Errorf("error message does not mention 'unknown subkernel selector': %q", errMsg)
		}
		if !strings.Contains(errMsg, unknown) {
			t.Errorf("error message does not contain unknown selector %q: %q", unknown, errMsg)
		}
		if !strings.Contains(errMsg, "available:") {
			t.Errorf("error message does not list available options: %q", errMsg)
		}
	})

	t.Run("mixed valid and unknown selector returns error", func(t *testing.T) {
		specs, err := filterSubkernelSpecs([]string{"argmax", "unknown_kernel"})
		if err == nil {
			t.Fatalf("expected error for mixed selector, got nil (specs: %v)", specs)
		}
		if !strings.Contains(err.Error(), "unknown_kernel") {
			t.Errorf("expected error to name unknown_kernel, got: %v", err)
		}
	})

	t.Run("empty string selectors return error", func(t *testing.T) {
		_, err := filterSubkernelSpecs([]string{"", "   "})
		if err == nil {
			t.Fatal("expected error for all-empty selectors, got nil")
		}
	})

	t.Run("duplicate selectors do not duplicate specs", func(t *testing.T) {
		specs, err := filterSubkernelSpecs([]string{"argmax", "argmax"})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(specs) != 1 {
			t.Fatalf("expected 1 spec for duplicate selectors, got %d", len(specs))
		}
	})
}

func TestRunSubkernelTests_UnknownSelector(t *testing.T) {
	ctx := context.Background()

	// Unknown selector must return error immediately without running or dereferencing nil target
	results, err := RunSubkernelTests(ctx, nil, []string{"qwen35_sequence_prefill"})
	if err == nil {
		t.Fatalf("expected error for unknown selector, got nil (results: %v)", results)
	}
	if !strings.Contains(err.Error(), "unknown subkernel selector") {
		t.Errorf("expected unknown selector error, got: %v", err)
	}

	// Valid selector with nil target safely returns target is nil error
	_, err = RunSubkernelTests(ctx, nil, []string{"argmax"})
	if err == nil || !strings.Contains(err.Error(), "target is nil") {
		t.Errorf("expected 'target is nil' error, got: %v", err)
	}

	// Valid selector with unreachable target safely returns unreachable error
	target := &StrixTarget{
		Host:      "offline-host",
		Reachable: false,
	}
	_, err = RunSubkernelTests(ctx, target, []string{"argmax"})
	if err == nil || !strings.Contains(err.Error(), "not reachable") {
		t.Errorf("expected 'not reachable' error, got: %v", err)
	}
}

func TestRunStrixValidation_UnknownSelector(t *testing.T) {
	defer func() {
		_ = os.Remove(StrixPresenceFile)
	}()

	// Cache a reachable simulated target so DiscoverStrixTarget succeeds
	target := &StrixTarget{
		Mode:           "ssh",
		Host:           "test-strix-sim",
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
	savePresenceCache(target)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	opts := StrixValidationOpts{
		Host:          "test-strix-sim",
		RunSubkernels: true,
		Subkernels:    []string{"qwen35_sequence_prefill"},
		RunAblations:  false,
		GitRef:        "HEAD",
		Command:       "fak validate --subkernels qwen35_sequence_prefill",
	}

	receipt, err := RunStrixValidation(ctx, opts)
	if receipt == nil {
		t.Fatal("expected non-nil receipt even on subkernel failure")
	}

	if receipt.Verdict != "FAIL" {
		t.Errorf("receipt.Verdict = %q, want FAIL", receipt.Verdict)
	}
	if receipt.Verified {
		t.Errorf("receipt.Verified = true, want false")
	}

	foundFailure := false
	for _, f := range receipt.Failures {
		if strings.Contains(f, "unknown subkernel selector") && strings.Contains(f, "qwen35_sequence_prefill") {
			foundFailure = true
			break
		}
	}
	if !foundFailure {
		t.Errorf("receipt.Failures does not mention unknown subkernel selector: %v", receipt.Failures)
	}
	_ = err
}

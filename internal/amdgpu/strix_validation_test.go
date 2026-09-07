package amdgpu

import (
	"context"
	"fmt"
	"os/exec"
	"strings"
	"testing"
	"time"
)

func TestStrixValidationOrchestrator(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Discover live Strix Halo target (local or via SSH strix1)
	target, err := DiscoverStrixTarget(ctx, "")
	if err != nil || target == nil || !target.Reachable {
		t.Skipf("Strix Halo appliance not reachable for integration test: %v", err)
	}

	gitTip := ""
	if out, err := exec.Command("git", "rev-parse", "HEAD").Output(); err == nil {
		gitTip = strings.TrimSpace(string(out))
	}

	opts := StrixValidationOpts{
		Host:          target.Host,
		RunSubkernels: true,
		Subkernels:    []string{"argmax", "matmul_f32", "q4k_matmul", "rmsnorm", "swiglu"},
		RunAblations:  true,
		Ablations:     []string{"cpu_vs_vulkan_gpu", "fused_vs_discrete_norm_matmul"},
		GitRef:        "HEAD",
		GitTip:        gitTip,
		Command:       "fak validate --strix --subkernels --ablate",
		Timeout:       30 * time.Second,
	}

	receipt, err := RunStrixValidation(ctx, opts)
	if err != nil {
		t.Fatalf("RunStrixValidation failed: %v", err)
	}

	if receipt == nil {
		t.Fatal("expected non-nil receipt")
	}

	if receipt.Schema != StrixValidationSchema {
		t.Errorf("receipt schema = %q, want %q", receipt.Schema, StrixValidationSchema)
	}

	if receipt.Verdict != "PASS" {
		t.Errorf("receipt verdict = %q, want PASS (failures: %v)", receipt.Verdict, receipt.Failures)
	}

	if len(receipt.Subkernels) == 0 {
		t.Error("expected at least one subkernel result")
	}

	if len(receipt.Ablations) == 0 {
		t.Error("expected at least one ablation result")
	}

	if err := receipt.Validate(); err != nil {
		t.Errorf("receipt validation failed: %v", err)
	}

	t.Logf("Live Strix Halo receipt generated: verdict=%s, digest=%s, subkernels=%d, ablations=%d",
		receipt.Verdict, receipt.Digest, len(receipt.Subkernels), len(receipt.Ablations))
}

func TestRunStrixValidation_SourceBindingMissingWhenRequired(t *testing.T) {
	defer ClearPresenceCache()

	target := &StrixTarget{
		Mode:         "ssh",
		Host:         "test-strix-sim-src",
		Reachable:    true,
		CPUModel:     "AMD Ryzen AI MAX+ 395",
		GPUName:      "AMD Radeon 8060S Graphics",
		TargetISA:    "gfx1151",
		ComputeUnits: 40,
		DiscoveredAt: time.Now().UTC().Format(time.RFC3339),
	}
	SavePresenceCache(target)

	ctx := context.Background()
	opts := StrixValidationOpts{
		Host:                 "test-strix-sim-src",
		RequireSourceBinding: true,
		GitTip:               "", // missing when required
		RunSubkernels:        true,
		Subkernels:           []string{"argmax"},
	}

	receipt, err := RunStrixValidation(ctx, opts)
	if err == nil {
		t.Fatal("expected error when GitTip missing under RequireSourceBinding, got nil")
	}
	if receipt == nil {
		t.Fatal("expected non-nil receipt on failure")
	}
	if receipt.Verdict != "FAIL" {
		t.Errorf("receipt.Verdict = %q, want FAIL", receipt.Verdict)
	}
	if receipt.Verified {
		t.Error("receipt.Verified = true, want false")
	}

	foundReason := false
	for _, f := range receipt.Failures {
		if strings.Contains(f, "source binding required but GitTip is missing") {
			foundReason = true
			break
		}
	}
	if !foundReason {
		t.Errorf("expected failure message 'source binding required but GitTip is missing', got: %v", receipt.Failures)
	}
}

func TestRunStrixValidation_SourceBindingMismatch(t *testing.T) {
	defer ClearPresenceCache()

	origVerify := verifySourceBindingFn
	defer func() {
		verifySourceBindingFn = origVerify
	}()

	verifySourceBindingFn = func(ctx context.Context, target *StrixTarget, gitTip, gitRef string) error {
		return fmt.Errorf("source binding mismatch: target HEAD 1111111111111111 does not match GitTip %s", gitTip)
	}

	target := &StrixTarget{
		Mode:         "ssh",
		Host:         "test-strix-sim-mismatch",
		Reachable:    true,
		CPUModel:     "AMD Ryzen AI MAX+ 395",
		GPUName:      "AMD Radeon 8060S Graphics",
		TargetISA:    "gfx1151",
		ComputeUnits: 40,
		DiscoveredAt: time.Now().UTC().Format(time.RFC3339),
	}
	SavePresenceCache(target)

	ctx := context.Background()
	opts := StrixValidationOpts{
		Host:          "test-strix-sim-mismatch",
		GitTip:        "2222222222222222",
		RunSubkernels: true,
		Subkernels:    []string{"argmax"},
	}

	receipt, err := RunStrixValidation(ctx, opts)
	if err == nil {
		t.Fatal("expected error on source binding mismatch, got nil")
	}
	if receipt == nil {
		t.Fatal("expected non-nil receipt on failure")
	}
	if receipt.Verdict != "FAIL" {
		t.Errorf("receipt.Verdict = %q, want FAIL", receipt.Verdict)
	}
	if receipt.Verified {
		t.Error("receipt.Verified = true, want false")
	}

	foundReason := false
	for _, f := range receipt.Failures {
		if strings.Contains(f, "source binding verification failed") {
			foundReason = true
			break
		}
	}
	if !foundReason {
		t.Errorf("expected failure mentioning 'source binding verification failed', got: %v", receipt.Failures)
	}
}

func TestVerifySourceBinding_DirectChecks(t *testing.T) {
	ctx := context.Background()

	t.Run("nil target fails", func(t *testing.T) {
		err := VerifySourceBinding(ctx, nil, "abc", "")
		if err == nil {
			t.Fatal("expected error for nil target")
		}
	})

	t.Run("unreachable target fails", func(t *testing.T) {
		target := &StrixTarget{Host: "offline", Reachable: false}
		err := VerifySourceBinding(ctx, target, "abc", "")
		if err == nil {
			t.Fatal("expected error for unreachable target")
		}
	})

	t.Run("empty gitTip and gitRef returns nil", func(t *testing.T) {
		target := &StrixTarget{Host: "dummy", Reachable: true}
		err := VerifySourceBinding(ctx, target, "", "")
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	})

	t.Run("local git repository check matches or mismatches", func(t *testing.T) {
		headBytes, err := exec.Command("git", "rev-parse", "HEAD").Output()
		if err != nil {
			t.Skip("git not available in environment")
		}
		actualHead := strings.TrimSpace(string(headBytes))
		if len(actualHead) < 8 {
			t.Skip("invalid git HEAD in repo")
		}

		// Point FAK_STRIX_DIR to repo root (current directory)
		t.Setenv("FAK_STRIX_DIR", ".")
		localTarget := &StrixTarget{
			Host:      "localhost",
			Mode:      "local",
			Reachable: true,
		}

		// Exact match should succeed
		if err := VerifySourceBinding(ctx, localTarget, actualHead, ""); err != nil {
			t.Errorf("expected success on exact HEAD match, got: %v", err)
		}

		// Prefix match should succeed
		if err := VerifySourceBinding(ctx, localTarget, actualHead[:8], ""); err != nil {
			t.Errorf("expected success on prefix HEAD match, got: %v", err)
		}

		// Mismatch must fail
		badTip := "0000000000000000000000000000000000000000"
		if err := VerifySourceBinding(ctx, localTarget, badTip, ""); err == nil {
			t.Error("expected failure on mismatched GitTip, got nil")
		} else if !strings.Contains(err.Error(), "mismatch") {
			t.Errorf("error %q should mention 'mismatch'", err.Error())
		}
	})
}

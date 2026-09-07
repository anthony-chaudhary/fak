package amdgpu

import (
	"context"
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

	opts := StrixValidationOpts{
		Host:          target.Host,
		RunSubkernels: true,
		Subkernels:    []string{"argmax", "matmul_f32", "q4k_matmul", "rmsnorm", "swiglu"},
		RunAblations:  true,
		Ablations:     []string{"cpu_vs_vulkan_gpu", "fused_vs_discrete_norm_matmul"},
		GitRef:        "HEAD",
		GitTip:        "test-tip",
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

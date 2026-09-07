package amdgpu

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"
)

func TestFilterSubkernelSpecs_RejectsUnknownSelectors(t *testing.T) {
	// Unknown single selector
	specs, err := FilterSubkernelSpecs([]string{"invalid_name"})
	if err == nil {
		t.Fatalf("expected error for unknown selector, got specs: %v", specs)
	}
	if !strings.Contains(err.Error(), "invalid_name") {
		t.Errorf("error %q should mention invalid_name", err.Error())
	}

	// Unexported filterSubkernelSpecs parity
	specs2, err2 := filterSubkernelSpecs([]string{"invalid_name"})
	if err2 == nil {
		t.Fatalf("expected error from filterSubkernelSpecs, got specs: %v", specs2)
	}

	// Mixed valid and unknown selectors
	_, err = FilterSubkernelSpecs([]string{"argmax", "bogus_kernel"})
	if err == nil {
		t.Fatal("expected error when unknown selector is mixed with valid ones")
	}
	if !strings.Contains(err.Error(), "bogus_kernel") {
		t.Errorf("error %q should mention bogus_kernel", err.Error())
	}

	// Case-insensitive valid spec name
	specs, err = FilterSubkernelSpecs([]string{"ARGMAX"})
	if err != nil {
		t.Fatalf("unexpected error for ARGMAX: %v", err)
	}
	if len(specs) != 1 || specs[0].Name != "argmax" {
		t.Errorf("unexpected specs for ARGMAX: %v", specs)
	}

	// Case-insensitive valid category
	specs, err = FilterSubkernelSpecs([]string{"QUANT"})
	if err != nil {
		t.Fatalf("unexpected error for category QUANT: %v", err)
	}
	if len(specs) == 0 {
		t.Fatal("expected specs for category QUANT, got 0")
	}
	for _, s := range specs {
		if strings.ToLower(s.Category) != "quant" {
			t.Errorf("spec %q category = %q, want quant", s.Name, s.Category)
		}
	}

	// Empty slice returns default specs
	specs, err = FilterSubkernelSpecs(nil)
	if err != nil {
		t.Fatalf("unexpected error for nil: %v", err)
	}
	if len(specs) != len(DefaultSubkernelSpecs) {
		t.Errorf("len(specs) = %d, want %d", len(specs), len(DefaultSubkernelSpecs))
	}
}

func TestRunStrixValidation_UnknownSelector_FailsClosed(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	opts := StrixValidationOpts{
		RunSubkernels: true,
		Subkernels:    []string{"invalid_name"},
		RunAblations:  false,
		GitRef:        "HEAD",
		Command:       "fak-dev amd-strix-validate --subkernels=invalid_name --ablate=none",
	}

	receipt, err := RunStrixValidation(ctx, opts)
	if err == nil {
		t.Fatal("expected RunStrixValidation to return an error for unknown subkernel selector")
	}
	if receipt == nil {
		t.Fatal("expected non-nil receipt even on error")
	}
	if receipt.Verdict != "FAIL" {
		t.Errorf("receipt.Verdict = %q, want FAIL", receipt.Verdict)
	}
	if receipt.Verified {
		t.Errorf("receipt.Verified = true, want false")
	}
	if receipt.SelectedSubkernels != 0 || receipt.SelectedCount != 0 {
		t.Errorf("receipt.SelectedSubkernels = %d, SelectedCount = %d, want 0", receipt.SelectedSubkernels, receipt.SelectedCount)
	}
	if receipt.ExecutedSubkernels != 0 || receipt.ExecutedCount != 0 {
		t.Errorf("receipt.ExecutedSubkernels = %d, ExecutedCount = %d, want 0", receipt.ExecutedSubkernels, receipt.ExecutedCount)
	}
	if len(receipt.Failures) == 0 {
		t.Fatal("expected receipt.Failures to record failure message")
	}

	foundErr := false
	for _, f := range receipt.Failures {
		if strings.Contains(f, "invalid_name") {
			foundErr = true
			break
		}
	}
	if !foundErr {
		t.Errorf("receipt.Failures %v does not mention invalid_name", receipt.Failures)
	}
}

func TestRunStrixValidation_ZeroExecutedSubkernelsCausesFail(t *testing.T) {
	target := StrixTarget{
		Mode:         "local",
		Host:         "local",
		Reachable:    true,
		GPUName:      "AMD Radeon 8060S Graphics (RADV STRIX_HALO)",
		TargetISA:    "gfx1151",
		ComputeUnits: 40,
		DiscoveredAt: time.Now().UTC().Format(time.RFC3339),
	}

	// 1. Receipt validation refuses PASS when SelectedSubkernels > 0 but ExecutedSubkernels == 0
	receipt := NewStrixValidationReceipt(target, "HEAD", "tip", "cmd")
	receipt.Verdict = "PASS"
	receipt.Verified = true
	receipt.SelectedSubkernels = 3
	receipt.ExecutedSubkernels = 0
	if err := receipt.Validate(); err == nil {
		t.Error("expected receipt.Validate() to fail when SelectedSubkernels > 0 and ExecutedSubkernels == 0")
	}

	// 2. Receipt validation refuses PASS when Failures is non-empty
	receipt2 := NewStrixValidationReceipt(target, "HEAD", "tip", "cmd")
	receipt2.Verdict = "PASS"
	receipt2.Verified = true
	receipt2.SelectedSubkernels = 1
	receipt2.ExecutedSubkernels = 1
	receipt2.Subkernels = append(receipt2.Subkernels, StrixSubkernelResult{
		Name:   "argmax",
		Status: "PASS",
	})
	receipt2.Failures = append(receipt2.Failures, "subkernels error: zero subkernels executed")
	if err := receipt2.Validate(); err == nil {
		t.Error("expected receipt.Validate() to fail when Failures is non-empty")
	}

	// 3. RunStrixValidation with unreachable target and RunSubkernels=true fails closed with 0 executed
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	opts := StrixValidationOpts{
		Host:          "non_existent_host_12345",
		RunSubkernels: true,
		Subkernels:    []string{"argmax"},
		RunAblations:  false,
	}
	r, err := RunStrixValidation(ctx, opts)
	if err == nil {
		t.Fatal("expected error for unreachable host")
	}
	if r == nil {
		t.Fatal("expected non-nil receipt")
	}
	if r.Verdict != "FAIL" {
		t.Errorf("verdict = %q, want FAIL", r.Verdict)
	}
	if r.Verified {
		t.Errorf("verified = true, want false")
	}
	if r.SelectedSubkernels != 1 || r.SelectedCount != 1 {
		t.Errorf("selected_subkernels = %d, selected_count = %d, want 1", r.SelectedSubkernels, r.SelectedCount)
	}
	if r.ExecutedSubkernels != 0 || r.ExecutedCount != 0 {
		t.Errorf("executed_subkernels = %d, executed_count = %d, want 0", r.ExecutedSubkernels, r.ExecutedCount)
	}
}

func TestStrixValidationReceipt_SubkernelCountsJSON(t *testing.T) {
	target := StrixTarget{
		Mode:         "ssh",
		Host:         "strix1",
		Reachable:    true,
		GPUName:      "AMD Radeon 8060S Graphics (RADV STRIX_HALO)",
		TargetISA:    "gfx1151",
		ComputeUnits: 40,
		DiscoveredAt: time.Now().UTC().Format(time.RFC3339),
	}

	receipt := NewStrixValidationReceipt(target, "HEAD", "tip", "fak-dev amd-strix-validate")
	receipt.SelectedSubkernels = 4
	receipt.ExecutedSubkernels = 4

	data, err := json.Marshal(receipt)
	if err != nil {
		t.Fatalf("json.Marshal failed: %v", err)
	}

	jsonStr := string(data)
	if !strings.Contains(jsonStr, `"selected_subkernels":4`) && !strings.Contains(jsonStr, `"selected_subkernels": 4`) {
		t.Errorf("JSON output missing selected_subkernels: %s", jsonStr)
	}
	if !strings.Contains(jsonStr, `"executed_subkernels":4`) && !strings.Contains(jsonStr, `"executed_subkernels": 4`) {
		t.Errorf("JSON output missing executed_subkernels: %s", jsonStr)
	}

	var decoded StrixValidationReceipt
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("json.Unmarshal failed: %v", err)
	}
	if decoded.SelectedSubkernels != 4 {
		t.Errorf("decoded.SelectedSubkernels = %d, want 4", decoded.SelectedSubkernels)
	}
	if decoded.ExecutedSubkernels != 4 {
		t.Errorf("decoded.ExecutedSubkernels = %d, want 4", decoded.ExecutedSubkernels)
	}
}

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

package amdgpu

import (
	"archive/tar"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"
	"time"
)

const testTip = "0123456789abcdef0123456789abcdef01234567"
const testHash = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

func validStrixExecutionEvidence() StrixExecutionEvidence {
	exit := 0
	return StrixExecutionEvidence{SourceArchiveSHA256: testHash, BinarySHA256: testHash, ShaderBundleSHA256: testHash, CommandSHA256: testHash, DeviceIdentity: "AMD Radeon 8060S Graphics|gfx1151", EngineIdentity: "fak-native/vulkan", ArtifactRehashed: true, DeviceTimeoutMS: 60000, LeasePathSHA256: testHash, AdmissionWaitMS: 30000, Acquired: true, Released: true, AcquireOrdinal: 1, ReleaseOrdinal: 2, ExitCode: &exit, RawOutputSHA256: testHash, RawOutputBytes: 1}
}

func authorizeStrixReceiptForTest(t *testing.T, r *StrixValidationReceipt) {
	t.Helper()
	binding, err := r.ComputeDigest()
	if err != nil {
		t.Fatalf("ComputeDigest for test authority failed: %v", err)
	}
	r.authority = strixReceiptAuthority{seal: &strixReceiptAuthoritySealValue, binding: binding}
}

func validStrixReceipt(t *testing.T) *StrixValidationReceipt {
	t.Helper()
	r := NewStrixValidationReceipt(StrixTarget{Mode: "ssh", Host: "strix1", Reachable: true, GPUName: "AMD Radeon 8060S Graphics", TargetISA: "gfx1151"}, "HEAD", testTip, "test")
	r.Provenance.SourceArchiveSHA256 = testHash
	r.Provenance.BinarySHA256 = testHash
	r.Provenance.ShaderBundleSHA256 = testHash
	r.Provenance.BuildCommandSHA256 = testHash
	r.Provenance.EngineIdentity = "fak-native/vulkan"
	r.Provenance.CleanupObserved = true
	r.SelectedCount = 1
	r.ExecutedCount = 1
	r.SelectedSubkernels = 1
	r.ExecutedSubkernels = 1
	r.SelectedAblations = 1
	r.ExecutedAblations = 1
	r.Subkernels = []StrixSubkernelResult{{Name: "argmax", Status: "PASS", DurationUS: 1, Iterations: 1, ParityEvents: []StrixParityEvent{NewExactArgmaxParityEvent("fak-native/vulkan", 1, true, true)}, Evidence: validStrixExecutionEvidence()}}
	r.Ablations = []StrixAblationResult{{Dimension: "target", Feature: "cpu_vs_vulkan_gpu", BaselineArm: StrixArmResult{Name: "cpu", LatencyUS: 2, Samples: 1}, CandidateArm: StrixArmResult{Name: "gpu", LatencyUS: 1, Samples: 1}, Speedup: 2, LiftRatio: 2, CosineParity: .9999, Verdict: "VERIFIED_LIFT", Evidence: validStrixExecutionEvidence()}}
	r.Provenance.ExecutionManifestSHA256 = executionManifestDigest(r)
	r.Verified = true
	authorizeStrixReceiptForTest(t, r)
	r.Digest, _ = r.ComputeDigest()
	return r
}

func TestStrixValidationReceiptRejectsSelfAuthoredPhysicalCredit(t *testing.T) {
	authorized := validStrixReceipt(t)
	authorized.SelectedAblations = 0
	authorized.ExecutedAblations = 0
	authorized.Ablations = nil
	authorized.Provenance.ExecutionManifestSHA256 = executionManifestDigest(authorized)
	authorizeStrixReceiptForTest(t, authorized)
	authorized.Digest, _ = authorized.ComputeDigest()
	if err := authorized.Validate(); err != nil {
		t.Fatalf("verifier-authorized control receipt failed validation: %v", err)
	}
	if !authorized.CreditEligible() {
		t.Fatal("verifier-authorized argmax control must earn physical parity credit")
	}
	registry := NewStrixCandidateRegistry()
	before := registry.Scoreboard()
	if comparisons, err := registry.EvaluateReceipt(authorized); err != nil || len(comparisons) != 0 {
		t.Fatalf("argmax-only control must be accepted with no ablation comparisons: comparisons=%d err=%v", len(comparisons), err)
	}
	if after := registry.Scoreboard(); !reflect.DeepEqual(after, before) {
		t.Fatalf("argmax-only control mutated scoreboard: before=%v after=%v", before, after)
	}
	raw, err := json.Marshal(authorized)
	if err != nil {
		t.Fatalf("marshal fabricated receipt: %v", err)
	}
	if bytes.Contains(raw, []byte("authority")) {
		t.Fatalf("opaque authority leaked into serialized receipt: %s", raw)
	}
	assertReadableNonCredit := func(t *testing.T, receipt *StrixValidationReceipt) {
		t.Helper()
		if err := receipt.Validate(); err != nil {
			t.Fatalf("serialized receipt should remain structurally readable: %v", err)
		}
		if receipt.CreditEligible() {
			t.Fatal("serialized receipt earned physical parity credit")
		}
		registry := NewStrixCandidateRegistry()
		before := registry.Scoreboard()
		if _, err := registry.EvaluateReceipt(receipt); err == nil {
			t.Fatal("serialized receipt earned candidate-evaluation credit")
		}
		if after := registry.Scoreboard(); !reflect.DeepEqual(after, before) {
			t.Fatalf("rejected receipt mutated scoreboard: before=%v after=%v", before, after)
		}
	}

	t.Run("genuine_roundtrip", func(t *testing.T) {
		var roundTrip StrixValidationReceipt
		if err := json.Unmarshal(raw, &roundTrip); err != nil {
			t.Fatalf("unmarshal genuine receipt: %v", err)
		}
		assertReadableNonCredit(t, &roundTrip)
	})

	t.Run("reused_authorized_destination", func(t *testing.T) {
		reused := *authorized
		if !reused.CreditEligible() {
			t.Fatal("copied verifier-authorized receipt lost physical parity credit before decoding")
		}
		if err := json.Unmarshal(raw, &reused); err != nil {
			t.Fatalf("unmarshal genuine receipt into authorized destination: %v", err)
		}
		assertReadableNonCredit(t, &reused)
	})

	ablationRaw, err := json.Marshal(validStrixReceipt(t))
	if err != nil {
		t.Fatalf("marshal ablation receipt: %v", err)
	}
	const fabricatedHash = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
	for _, tc := range []struct {
		verdict  string
		verified bool
	}{{verdict: "PASS", verified: true}, {verdict: "FAIL"}, {verdict: "SKIPPED"}} {
		t.Run(tc.verdict, func(t *testing.T) {
			var fabricated StrixValidationReceipt
			if err := json.Unmarshal(ablationRaw, &fabricated); err != nil {
				t.Fatalf("unmarshal fabricated receipt: %v", err)
			}
			fabricated.Verdict = tc.verdict
			fabricated.Verified = tc.verified
			fabricated.Subkernels[0].Evidence.RawOutputSHA256 = fabricatedHash
			fabricated.Ablations[0].Evidence.RawOutputSHA256 = fabricatedHash
			fabricated.Digest, err = fabricated.ComputeDigest()
			if err != nil {
				t.Fatalf("recompute fabricated digest: %v", err)
			}
			assertReadableNonCredit(t, &fabricated)
		})
	}
}

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
	if os.Getenv("FAK_STRIX_LIVE_TEST") != "1" {
		t.Skip("set FAK_STRIX_LIVE_TEST=1 for the explicit physical integration witness")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 180*time.Second)
	defer cancel()

	// Discover live Strix Halo target (local or via SSH strix1)
	target, err := DiscoverStrixTarget(ctx, "")
	if err != nil || target == nil || !target.Reachable {
		t.Skipf("Strix Halo appliance not reachable for integration test: %v", err)
	}

	gitTip := ""
	repoRoot := "."
	if out, err := exec.Command("git", "rev-parse", "HEAD").Output(); err == nil {
		gitTip = strings.TrimSpace(string(out))
	}
	if out, err := exec.Command("git", "rev-parse", "--show-toplevel").Output(); err == nil {
		repoRoot = strings.TrimSpace(string(out))
	}

	archive, err := BuildStrixCandidateArchive(ctx, repoRoot, gitTip, []string{
		"internal/compute/vulkan_test.go",
		"internal/compute/shaders/attention.comp",
		"internal/amdgpu/strix_ablations.go",
		"internal/amdgpu/strix_validation.go",
		"internal/amdgpu/strix_subkernels.go",
	})
	if err != nil {
		t.Fatalf("BuildStrixCandidateArchive failed: %v", err)
	}

	opts := StrixValidationOpts{
		Host:                 target.Host,
		RunSubkernels:        true,
		Subkernels:           []string{"argmax", "matmul_f32", "q4k_matmul", "rmsnorm", "swiglu"},
		RunAblations:         true,
		Ablations:            []string{"cpu_vs_vulkan_gpu", "fused_vs_discrete_norm_matmul"},
		GitRef:               archive.SourceArchiveSHA256,
		GitTip:               gitTip,
		Command:              "fak validate --strix --subkernels --ablate",
		Timeout:              120 * time.Second,
		RequireSourceBinding: true,
		CandidateArchive:     archive.Bytes,
		SourceArchiveSHA256:  archive.SourceArchiveSHA256,
		AdmissionTimeout:     20 * time.Second,
	}

	receipt, err := RunStrixValidation(ctx, opts)
	if receipt != nil {
		t.Logf("Receipt Failures: %v", receipt.Failures)
		for _, sk := range receipt.Subkernels {
			t.Logf("Subkernel %s: status=%s duration=%d us error=%s", sk.Name, sk.Status, sk.DurationUS, sk.Error)
		}
		for _, ab := range receipt.Ablations {
			t.Logf("Ablation %s: verdict=%s speedup=%.2f baseline=%d candidate=%d", ab.Feature, ab.Verdict, ab.Speedup, ab.BaselineArm.LatencyUS, ab.CandidateArm.LatencyUS)
		}
	}
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

func TestRunStrixValidation_LegacyCheckoutBindingCannotEarnV2Credit(t *testing.T) {
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
		t.Fatal("expected legacy checkout-only binding to fail, got nil")
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
		if strings.Contains(f, "v2 validation requires source binding") {
			foundReason = true
			break
		}
	}
	if !foundReason {
		t.Errorf("expected failure mentioning v2 source binding requirement, got: %v", receipt.Failures)
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

func TestBuildStrixCandidateArchiveDeterministicAndTamperClosed(t *testing.T) {
	rootBytes, err := exec.Command("git", "rev-parse", "--show-toplevel").Output()
	if err != nil {
		t.Fatal(err)
	}
	tipBytes, err := exec.Command("git", "rev-parse", "HEAD").Output()
	if err != nil {
		t.Fatal(err)
	}
	root, tip := strings.TrimSpace(string(rootBytes)), strings.TrimSpace(string(tipBytes))
	a, err := BuildStrixCandidateArchive(context.Background(), root, tip, []string{"internal/amdgpu/strix_receipt.go"})
	if err != nil {
		t.Fatalf("first archive: %v", err)
	}
	b, err := BuildStrixCandidateArchive(context.Background(), root, tip, []string{"internal/amdgpu/strix_receipt.go"})
	if err != nil {
		t.Fatalf("second archive: %v", err)
	}
	if a.SourceArchiveSHA256 != b.SourceArchiveSHA256 {
		t.Fatalf("candidate archive is nondeterministic: %+v != %+v", a, b)
	}
	tampered := append([]byte(nil), a.Bytes...)
	tampered[len(tampered)/2] ^= 1
	_, err = stageStrixCandidate(context.Background(), &StrixTarget{Reachable: true, Mode: "local"}, StrixValidationOpts{GitTip: tip, CandidateArchive: tampered, SourceArchiveSHA256: a.SourceArchiveSHA256})
	if err == nil || !strings.Contains(err.Error(), "mismatched exact candidate archive") {
		t.Fatalf("tampered archive did not fail before target access: %v", err)
	}
}

func TestValidateStrixWorkspaceRejectsTraversal(t *testing.T) {
	for _, bad := range []string{"/tmp/fak-strix-validation.ok/../victim", "/tmp/fak-strix-validation.", "/tmp/fak-strix-validation.ok/child", "/var/tmp/fak-strix-validation.ok"} {
		if err := validateStrixWorkspace(bad); err == nil {
			t.Errorf("unsafe workspace accepted: %q", bad)
		}
	}
	if err := validateStrixWorkspace("/tmp/fak-strix-validation.AbC_123-x"); err != nil {
		t.Fatalf("safe workspace rejected: %v", err)
	}
}

func TestBuildStrixCandidateArchiveRejectsOverlaySymlink(t *testing.T) {
	root := t.TempDir()
	run := func(args ...string) string {
		cmd := exec.Command("git", args...)
		cmd.Dir = root
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("git %v: %v: %s", args, err, out)
		}
		return strings.TrimSpace(string(out))
	}
	run("init")
	run("config", "user.email", "test@example.invalid")
	run("config", "user.name", "test")
	if err := os.WriteFile(filepath.Join(root, "base.txt"), []byte("base"), 0600); err != nil {
		t.Fatal(err)
	}
	run("add", "base.txt")
	run("commit", "-m", "base")
	if err := os.Symlink("base.txt", filepath.Join(root, "overlay-link")); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}
	_, err := BuildStrixCandidateArchive(context.Background(), root, run("rev-parse", "HEAD"), []string{"overlay-link"})
	if err == nil || !strings.Contains(err.Error(), "overlay symlink") {
		t.Fatalf("overlay symlink accepted: %v", err)
	}
}

func TestBuildStrixCandidateArchiveRejectsIntermediateSymlink(t *testing.T) {
	root := t.TempDir()
	run := func(args ...string) string {
		cmd := exec.Command("git", args...)
		cmd.Dir = root
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("git %v: %v: %s", args, err, out)
		}
		return strings.TrimSpace(string(out))
	}
	run("init")
	run("config", "user.email", "test@example.invalid")
	run("config", "user.name", "test")
	if err := os.WriteFile(filepath.Join(root, "marker"), []byte("base"), 0600); err != nil {
		t.Fatal(err)
	}
	run("add", "marker")
	run("commit", "-m", "base")
	tip := run("rev-parse", "HEAD")

	inside := filepath.Join(root, "regular")
	outside := t.TempDir()
	for _, dir := range []string{inside, outside} {
		if err := os.MkdirAll(dir, 0700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, "secret"), []byte("not archived"), 0600); err != nil {
			t.Fatal(err)
		}
	}
	for _, tc := range []struct {
		name   string
		target string
		leaf   string
	}{{"outside", outside, "secret"}, {"outside-missing-leaf", outside, "missing"}, {"inside", inside, "secret"}} {
		t.Run(tc.name, func(t *testing.T) {
			link := filepath.Join(root, "alias-"+tc.name)
			if err := os.Symlink(tc.target, link); err != nil {
				t.Skipf("symlinks unavailable: %v", err)
			}
			_, err := BuildStrixCandidateArchive(context.Background(), root, tip, []string{filepath.Join("alias-"+tc.name, tc.leaf)})
			if err == nil || !strings.Contains(err.Error(), "symlink component") {
				t.Fatalf("intermediate %s symlink accepted: %v", tc.name, err)
			}
		})
	}

	t.Run("directory symlink child", func(t *testing.T) {
		dir := filepath.Join(root, "overlay-with-link")
		if err := os.MkdirAll(dir, 0700); err != nil {
			t.Fatal(err)
		}
		if err := os.Symlink(outside, filepath.Join(dir, "child")); err != nil {
			t.Skipf("symlinks unavailable: %v", err)
		}
		if _, err := BuildStrixCandidateArchive(context.Background(), root, tip, []string{"overlay-with-link"}); err == nil || !strings.Contains(err.Error(), "symlink") {
			t.Fatalf("directory symlink child was silently omitted: %v", err)
		}
	})

	t.Run("directory special child", func(t *testing.T) {
		dir := filepath.Join(root, "overlay-with-special")
		if err := os.MkdirAll(dir, 0700); err != nil {
			t.Fatal(err)
		}
		fifo := filepath.Join(dir, "pipe")
		if out, err := exec.Command("mkfifo", fifo).CombinedOutput(); err != nil {
			t.Skipf("mkfifo unavailable: %v: %s", err, out)
		}
		if _, err := BuildStrixCandidateArchive(context.Background(), root, tip, []string{"overlay-with-special"}); err == nil || !strings.Contains(err.Error(), "unsupported mode") {
			t.Fatalf("directory special child was silently omitted: %v", err)
		}
	})
}

func TestBuildStrixCandidateArchiveRejectsWindowsJunction(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("Windows junction/reparse regression")
	}
	root := t.TempDir()
	run := func(args ...string) string {
		cmd := exec.Command("git", args...)
		cmd.Dir = root
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("git %v: %v: %s", args, err, out)
		}
		return strings.TrimSpace(string(out))
	}
	run("init")
	run("config", "user.email", "test@example.invalid")
	run("config", "user.name", "test")
	overlay := filepath.Join(root, "overlay")
	if err := os.MkdirAll(overlay, 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(overlay, "keep"), []byte("base"), 0600); err != nil {
		t.Fatal(err)
	}
	run("add", "overlay/keep")
	run("commit", "-m", "base")

	outside := t.TempDir()
	if err := os.WriteFile(filepath.Join(outside, "secret"), []byte("not archived"), 0600); err != nil {
		t.Fatal(err)
	}
	junction := filepath.Join(overlay, "junction")
	if out, err := exec.Command("cmd.exe", "/c", "mklink", "/J", junction, outside).CombinedOutput(); err != nil {
		t.Skipf("Windows junction unavailable: %v: %s", err, out)
	}
	defer func() { _ = exec.Command("cmd.exe", "/c", "rmdir", junction).Run() }()
	// Owning the parent directory forces the walk callback to validate the
	// junction itself before it can be treated as a traversable directory.
	_, err := BuildStrixCandidateArchive(context.Background(), root, run("rev-parse", "HEAD"), []string{"overlay"})
	if err == nil {
		t.Fatalf("Windows junction escaped overlay validation: %v", err)
	}
}

func TestBuildStrixCandidateArchiveCanceledBeforeTraversal(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	missingRoot := filepath.Join(t.TempDir(), "must-not-be-inspected")
	archive, err := BuildStrixCandidateArchive(ctx, missingRoot, testTip, []string{"anything"})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("already-canceled build returned %v, want context.Canceled", err)
	}
	if len(archive.Bytes) != 0 || archive.SourceArchiveSHA256 != "" {
		t.Fatalf("canceled build emitted archive receipt: %+v", archive)
	}
}

func TestBuildStrixCandidateArchiveWrapsGitCancellation(t *testing.T) {
	original := strixGitArchiveOutput
	t.Cleanup(func() { strixGitArchiveOutput = original })
	ctx, cancel := context.WithCancel(context.Background())
	strixGitArchiveOutput = func(context.Context, string, string) ([]byte, error) {
		cancel()
		return nil, context.Canceled
	}

	archive, err := BuildStrixCandidateArchive(ctx, t.TempDir(), testTip, nil)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("git cancellation returned %v, want context.Canceled", err)
	}
	if len(archive.Bytes) != 0 || archive.SourceArchiveSHA256 != "" {
		t.Fatalf("canceled git archive emitted receipt: %+v", archive)
	}
}

func TestBuildStrixCandidateArchiveAllowsRegularAndDeletionOverlays(t *testing.T) {
	root := t.TempDir()
	run := func(args ...string) string {
		cmd := exec.Command("git", args...)
		cmd.Dir = root
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("git %v: %v: %s", args, err, out)
		}
		return strings.TrimSpace(string(out))
	}
	run("init")
	run("config", "user.email", "test@example.invalid")
	run("config", "user.name", "test")
	if err := os.MkdirAll(filepath.Join(root, "regular"), 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(root, "deleted"), 0700); err != nil {
		t.Fatal(err)
	}
	for name, body := range map[string]string{"single.txt": "base", "regular/nested.txt": "base", "deleted/nested.txt": "remove"} {
		if err := os.WriteFile(filepath.Join(root, filepath.FromSlash(name)), []byte(body), 0600); err != nil {
			t.Fatal(err)
		}
	}
	run("add", ".")
	run("commit", "-m", "base")
	tip := run("rev-parse", "HEAD")
	if err := os.WriteFile(filepath.Join(root, "single.txt"), []byte("single overlay"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "regular", "nested.txt"), []byte("directory overlay"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.RemoveAll(filepath.Join(root, "deleted")); err != nil {
		t.Fatal(err)
	}

	archive, err := BuildStrixCandidateArchive(context.Background(), root, tip, []string{"single.txt", "regular", "deleted", "missing/child"})
	if err != nil {
		t.Fatalf("regular/deletion overlays rejected: %v", err)
	}
	contents := map[string]string{}
	tr := tar.NewReader(bytes.NewReader(archive.Bytes))
	for {
		h, er := tr.Next()
		if er == io.EOF {
			break
		}
		if er != nil {
			t.Fatal(er)
		}
		if h.Typeflag == tar.TypeReg || h.Typeflag == tar.TypeRegA {
			body, er := io.ReadAll(tr)
			if er != nil {
				t.Fatal(er)
			}
			contents[h.Name] = string(body)
		}
	}
	if contents["single.txt"] != "single overlay" || contents["regular/nested.txt"] != "directory overlay" {
		t.Fatalf("regular overlays missing from archive: %+v", contents)
	}
	for name := range contents {
		if strings.HasPrefix(name, "deleted/") || strings.HasPrefix(name, "missing/") {
			t.Fatalf("deletion overlay remained in archive: %q", name)
		}
	}
}

func TestArchiveRejectsCommittedLinksAndBindsOverlayBytes(t *testing.T) {
	if err := rejectArchiveLinkEntry(&tar.Header{Name: "hard", Linkname: "../escape", Typeflag: tar.TypeLink}); err == nil {
		t.Fatal("hardlink entry accepted")
	}
	root := t.TempDir()
	run := func(args ...string) string {
		cmd := exec.Command("git", args...)
		cmd.Dir = root
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("git %v: %v: %s", args, err, out)
		}
		return strings.TrimSpace(string(out))
	}
	run("init")
	run("config", "user.email", "test@example.invalid")
	run("config", "user.name", "test")
	if err := os.WriteFile(filepath.Join(root, "base"), []byte("one"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink("base", filepath.Join(root, "committed-link")); err != nil {
		t.Skipf("symlinks unavailable: %v", err)
	}
	run("add", "base", "committed-link")
	run("commit", "-m", "links")
	tip := run("rev-parse", "HEAD")
	if _, err := BuildStrixCandidateArchive(context.Background(), root, tip, nil); err == nil || !strings.Contains(err.Error(), "archive link") {
		t.Fatalf("committed symlink accepted: %v", err)
	}
	run("rm", "committed-link")
	run("commit", "-m", "unlink")
	tip = run("rev-parse", "HEAD")
	a, err := BuildStrixCandidateArchive(context.Background(), root, tip, []string{"base"})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "base"), []byte("two"), 0600); err != nil {
		t.Fatal(err)
	}
	b, err := BuildStrixCandidateArchive(context.Background(), root, tip, []string{"base"})
	if err != nil {
		t.Fatal(err)
	}
	if a.SourceArchiveSHA256 == b.SourceArchiveSHA256 {
		t.Fatal("transported source digest did not bind changed overlay bytes")
	}
}

func TestStageRejectsDigestMatchingUnsafeCandidateArchive(t *testing.T) {
	makeArchive := func(name, link string, kind byte) []byte {
		var b bytes.Buffer
		w := tar.NewWriter(&b)
		h := &tar.Header{Name: name, Linkname: link, Typeflag: kind, Mode: 0600}
		if kind == tar.TypeReg {
			h.Size = 1
		}
		if err := w.WriteHeader(h); err != nil {
			t.Fatal(err)
		}
		if kind == tar.TypeReg {
			_, _ = w.Write([]byte("x"))
		}
		if err := w.Close(); err != nil {
			t.Fatal(err)
		}
		return b.Bytes()
	}
	for _, tc := range []struct {
		name, link string
		kind       byte
	}{{"../escape", "", tar.TypeReg}, {"/absolute", "", tar.TypeReg}, {"link", "../escape", tar.TypeSymlink}, {"hard", "../escape", tar.TypeLink}} {
		data := makeArchive(tc.name, tc.link, tc.kind)
		_, err := stageStrixCandidate(context.Background(), &StrixTarget{Mode: "local", Reachable: true}, StrixValidationOpts{GitTip: testTip, CandidateArchive: data, SourceArchiveSHA256: digestBytes(data), AdmissionTimeout: time.Second})
		if err == nil {
			t.Fatalf("unsafe digest-matching archive accepted: %+v", tc)
		}
	}
}

func TestRunStrixValidationCleanupUsesFreshContextExactlyOnce(t *testing.T) {
	defer ClearPresenceCache()
	target := &StrixTarget{Mode: "ssh", Host: "cleanup-test", Reachable: true, GPUName: "AMD Radeon 8060S Graphics", TargetISA: "gfx1151"}
	SavePresenceCache(target)
	origStage, origCleanup, origExec := stageStrixCandidateFn, cleanupStrixCandidateFn, executeOneSubkernelFn
	defer func() {
		stageStrixCandidateFn = origStage
		cleanupStrixCandidateFn = origCleanup
		executeOneSubkernelFn = origExec
	}()
	ctx, cancel := context.WithCancel(context.Background())
	stageStrixCandidateFn = func(context.Context, *StrixTarget, StrixValidationOpts) (stagedStrixCandidate, error) {
		return stagedStrixCandidate{SourceBinding: SourceBinding{GitTip: testTip, GitRef: "HEAD", SourceArchiveSHA256: testHash, BinarySHA256: testHash, ShaderBundleSHA256: testHash, BuildCommandSHA256: testHash, WorkDir: "/tmp/fak-strix-validation.safe", AdmissionWait: 30 * time.Second}}, nil
	}
	cleanupCalls := 0
	cleanupStrixCandidateFn = func(cleanCtx context.Context, _ *StrixTarget, work string) bool {
		cleanupCalls++
		if cleanCtx.Err() != nil {
			t.Errorf("cleanup inherited canceled execution context: %v", cleanCtx.Err())
		}
		return true
	}
	executeOneSubkernelFn = func(context.Context, *StrixTarget, SubkernelSpec) StrixSubkernelResult {
		cancel()
		return StrixSubkernelResult{Name: "argmax", Status: "PASS", DurationUS: 1, Iterations: 1, ParityEvents: []StrixParityEvent{NewExactArgmaxParityEvent("fak-native/vulkan", 1, true, true)}, Evidence: validStrixExecutionEvidence()}
	}
	r, err := RunStrixValidation(ctx, StrixValidationOpts{Host: target.Host, RunSubkernels: true, Subkernels: []string{"argmax"}, GitRef: "HEAD", GitTip: testTip, Command: "test", RequireSourceBinding: true, CandidateArchive: []byte("unused by stub"), SourceArchiveSHA256: testHash, AdmissionTimeout: 30 * time.Second})
	if err != nil {
		t.Fatalf("validation failed: %v", err)
	}
	if cleanupCalls != 1 {
		t.Fatalf("cleanup calls=%d want 1", cleanupCalls)
	}
	if !r.Provenance.CleanupObserved {
		t.Fatal("cleanup not recorded")
	}
}

func TestBuildStrixAdmissionCommandStopsOnArtifactMismatch(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("the admission shell contract is exercised under WSL/Linux")
	}
	work := t.TempDir()
	sentinel := filepath.Join(work, "device-ran")
	sb := SourceBinding{
		WorkDir:            work,
		BinarySHA256:       testHash,
		ShaderBundleSHA256: testHash,
		AdmissionWait:      time.Second,
	}
	target := &StrixTarget{Mode: "local", GPUName: "AMD Radeon 8060S Graphics", TargetISA: "gfx1151"}
	command := buildStrixAdmissionCommand(target, sb, "touch "+shellQuote(sentinel))
	out, err := runStrixTargetCommand(context.Background(), target, command, nil)
	if err == nil {
		t.Fatal("artifact mismatch unexpectedly executed successfully")
	}
	if strings.Contains(string(out), "FAK_STRIX_ARTIFACT_REHASH=1") {
		t.Fatalf("artifact mismatch emitted a successful rehash marker: %s", out)
	}
	if _, statErr := os.Stat(sentinel); !os.IsNotExist(statErr) {
		t.Fatalf("device command ran after artifact mismatch: %v", statErr)
	}
}

func TestRunStrixValidationRejectsInvalidAdmissionBeforeStaging(t *testing.T) {
	defer ClearPresenceCache()
	target := &StrixTarget{Mode: "ssh", Host: "admission-test", Reachable: true, GPUName: "AMD Radeon 8060S Graphics", TargetISA: "gfx1151"}
	SavePresenceCache(target)
	orig := stageStrixCandidateFn
	defer func() { stageStrixCandidateFn = orig }()
	called := false
	stageStrixCandidateFn = func(context.Context, *StrixTarget, StrixValidationOpts) (stagedStrixCandidate, error) {
		called = true
		return stagedStrixCandidate{}, nil
	}
	_, err := RunStrixValidation(context.Background(), StrixValidationOpts{Host: target.Host, RunSubkernels: true, Subkernels: []string{"argmax"}, GitTip: testTip, RequireSourceBinding: true, AdmissionTimeout: 61 * time.Second})
	if err == nil || !strings.Contains(err.Error(), "exceeds 60s") {
		t.Fatalf("invalid admission accepted: %v", err)
	}
	if called {
		t.Fatal("workspace staged before invalid admission timeout was rejected")
	}
}

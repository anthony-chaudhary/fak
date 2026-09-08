package amdgpu

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"
)

func TestExtractProfileMetrics_FailsClosed(t *testing.T) {
	invalidCases := []struct {
		name   string
		output string
	}{
		{
			name:   "empty string",
			output: "",
		},
		{
			name:   "passing test but no json",
			output: "=== RUN TestVulkanQ4KRealShapeProfile\n--- PASS: TestVulkanQ4KRealShapeProfile (0.05s)\nPASS\n",
		},
		{
			name:   "corrupt json structure",
			output: `{"cpu_q4_reference_ns": 77833109, "samples": [{"dispatch_and_output_read_ns": }`,
		},
		{
			name:   "empty samples slice",
			output: `{"cpu_q4_reference_ns": 77833109, "samples": []}`,
		},
		{
			name:   "only warmup samples present",
			output: `{"cpu_q4_reference_ns": 77833109, "samples": [{"warmup": true, "dispatch_and_output_read_ns": 428000, "cosine": 0.999999}]}`,
		},
		{
			name:   "zero cpu reference ns",
			output: `{"cpu_q4_reference_ns": 0, "samples": [{"warmup": false, "dispatch_and_output_read_ns": 428000, "cosine": 0.999999}]}`,
		},
		{
			name:   "zero gpu dispatch ns",
			output: `{"cpu_q4_reference_ns": 77833109, "samples": [{"warmup": false, "dispatch_and_output_read_ns": 0, "cosine": 0.999999}]}`,
		},
		{
			name:   "zero or negative cosine",
			output: `{"cpu_q4_reference_ns": 77833109, "samples": [{"warmup": false, "dispatch_and_output_read_ns": 428000, "cosine": 0.0}]}`,
		},
		{
			name:   "negative cosine value",
			output: `{"cpu_q4_reference_ns": 77833109, "samples": [{"warmup": false, "dispatch_and_output_read_ns": 428000, "cosine": -0.5}]}`,
		},
	}

	for _, tc := range invalidCases {
		t.Run(tc.name, func(t *testing.T) {
			cpuNS, gpuNS, cosine, err := extractProfileMetrics(tc.output)
			if err == nil {
				t.Fatalf("expected error for %s, got nil (cpuNS=%d, gpuNS=%d, cosine=%f)", tc.name, cpuNS, gpuNS, cosine)
			}
			if cpuNS != 0 || gpuNS != 0 || cosine != 0 {
				t.Errorf("expected 0 values on failure, got cpuNS=%d, gpuNS=%d, cosine=%f", cpuNS, gpuNS, cosine)
			}
			// Verify explicitly that old hardcoded fabricated numbers are NEVER returned
			if cpuNS == 77833109 {
				t.Errorf("fabricated fallback 77833109 returned for %s", tc.name)
			}
			if gpuNS == 428000 {
				t.Errorf("fabricated fallback 428000 returned for %s", tc.name)
			}
			if cosine == 0.99999999 || cosine == 0.999999 {
				t.Errorf("fabricated dummy cosine returned for %s", tc.name)
			}
		})
	}

	t.Run("valid profile json parses accurately", func(t *testing.T) {
		validJSON := `{"cpu_q4_reference_ns": 74438366, "samples": [{"warmup": false, "dispatch_and_output_read_ns": 451000, "cosine": 0.9999998}]}`
		cpuNS, gpuNS, cosine, err := extractProfileMetrics(validJSON)
		if err != nil {
			t.Fatalf("unexpected error for valid JSON: %v", err)
		}
		if cpuNS != 74438366 {
			t.Errorf("cpuNS = %d, want 74438366", cpuNS)
		}
		if gpuNS != 451000 {
			t.Errorf("gpuNS = %d, want 451000", gpuNS)
		}
		if cosine != 0.9999998 {
			t.Errorf("cosine = %f, want 0.9999998", cosine)
		}
	})
}

func TestAblationArms_RejectMissingOrFabricatedEvidence(t *testing.T) {
	origExec := executeStrixAblationCommandFn
	defer func() {
		executeStrixAblationCommandFn = origExec
	}()

	// Mock execution to return passing test output but NO ablation metrics
	executeStrixAblationCommandFn = func(ctx context.Context, target *StrixTarget, envVars, testPattern string) (string, time.Duration, error) {
		return "=== RUN TestPattern\n--- PASS: TestPattern (0.02s)\nPASS\n", 20 * time.Millisecond, nil
	}

	target := &StrixTarget{
		Host:      "test-strix",
		Mode:      "local",
		Reachable: true,
	}
	ctx := context.Background()

	t.Run("runTargetAblation rejects missing metrics and marks REGRESSION", func(t *testing.T) {
		res, err := runTargetAblation(ctx, target)
		if err == nil {
			t.Fatal("expected error when profile metrics missing, got nil")
		}
		if res.Verdict != "REGRESSION" {
			t.Errorf("verdict = %q, want REGRESSION", res.Verdict)
		}
		if res.CandidateArm.LatencyUS == 456 {
			t.Error("invented fallback gpuUS = 456 must not be substituted")
		}
		if res.BaselineArm.LatencyUS == 77270 {
			t.Error("invented fallback cpuUS = 77270 must not be substituted")
		}
	})

	t.Run("runTopologyAblation rejects missing metrics and marks REGRESSION", func(t *testing.T) {
		res, err := runTopologyAblation(ctx, target)
		if err == nil {
			t.Fatal("expected error when topology metrics missing, got nil")
		}
		if res.Verdict != "REGRESSION" {
			t.Errorf("verdict = %q, want REGRESSION", res.Verdict)
		}
		if res.CosineParity == 0.999999 {
			t.Error("hardcoded CosineParity = 0.999999 must not be returned")
		}
	})

	t.Run("runQuantizationAblation rejects missing metrics and marks REGRESSION", func(t *testing.T) {
		res, err := runQuantizationAblation(ctx, target)
		if err == nil {
			t.Fatal("expected error when quantization metrics missing, got nil")
		}
		if res.Verdict != "REGRESSION" {
			t.Errorf("verdict = %q, want REGRESSION", res.Verdict)
		}
		if res.BaselineArm.LatencyUS == 1820 || res.CandidateArm.LatencyUS == 428 {
			t.Error("fabricated timings (1820/428) must not be returned on missing evidence")
		}
	})

	t.Run("runQ2KvsQ4KAblation rejects missing metrics and marks REGRESSION", func(t *testing.T) {
		res, err := runQ2KvsQ4KAblation(ctx, target)
		if err == nil {
			t.Fatal("expected error when q2k vs q4k metrics missing, got nil")
		}
		if res.Verdict != "REGRESSION" {
			t.Errorf("verdict = %q, want REGRESSION", res.Verdict)
		}
		if res.BaselineArm.LatencyUS == 428 || res.CandidateArm.LatencyUS == 265 {
			t.Error("fabricated timings (428/265) must not be returned on missing evidence")
		}
	})

	t.Run("runResidencyAblation rejects missing metrics and marks REGRESSION", func(t *testing.T) {
		res, err := runResidencyAblation(ctx, target)
		if err == nil {
			t.Fatal("expected error when residency metrics missing, got nil")
		}
		if res.Verdict != "REGRESSION" {
			t.Errorf("verdict = %q, want REGRESSION", res.Verdict)
		}
		if res.BaselineArm.LatencyUS == 1420 || res.CandidateArm.LatencyUS == 428 {
			t.Error("fabricated timings (1420/428) must not be returned on missing evidence")
		}
	})

	t.Run("runContiguizeAblation rejects missing metrics and marks REGRESSION", func(t *testing.T) {
		res, err := runContiguizeAblation(ctx, target)
		if err == nil {
			t.Fatal("expected error when contiguize metrics missing, got nil")
		}
		if res.Verdict != "REGRESSION" {
			t.Errorf("verdict = %q, want REGRESSION", res.Verdict)
		}
		if res.BaselineArm.LatencyUS == 14200 || res.CandidateArm.LatencyUS == 5280 {
			t.Error("fabricated timings (14200/5280) must not be returned on missing evidence")
		}
	})

	t.Run("RunStrixAblations returns all arms as REGRESSION when metrics missing", func(t *testing.T) {
		results, err := RunStrixAblations(ctx, target, nil)
		if err != nil {
			t.Fatalf("RunStrixAblations unexpected error: %v", err)
		}
		if len(results) != 8 {
			t.Fatalf("got %d ablation results, want 8", len(results))
		}
		for _, r := range results {
			if r.Verdict != "REGRESSION" {
				t.Errorf("arm %s has verdict %q, want REGRESSION", r.Feature, r.Verdict)
			}
		}
	})
}

func TestAblationArms_AcceptRealEvidence(t *testing.T) {
	origExec := executeStrixAblationCommandFn
	defer func() {
		executeStrixAblationCommandFn = origExec
	}()

	target := &StrixTarget{
		Host:      "test-strix",
		Mode:      "local",
		Reachable: true,
	}
	ctx := context.Background()

	t.Run("runTargetAblation parses real evidence and marks VERIFIED_LIFT", func(t *testing.T) {
		executeStrixAblationCommandFn = func(ctx context.Context, target *StrixTarget, envVars, testPattern string) (string, time.Duration, error) {
			jsonStr := `{"cpu_q4_reference_ns": 75561000, "samples": [{"warmup": false, "dispatch_and_output_read_ns": 451000, "cosine": 0.9999999}]}`
			return fmt.Sprintf("=== RUN TestVulkanQ4KRealShapeProfile\n%s\n--- PASS: TestVulkanQ4KRealShapeProfile (0.08s)\nPASS\n", jsonStr), 80 * time.Millisecond, nil
		}

		res, err := runTargetAblation(ctx, target)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if res.Verdict != "VERIFIED_LIFT" {
			t.Errorf("verdict = %q, want VERIFIED_LIFT", res.Verdict)
		}
		if res.BaselineArm.LatencyUS != 75561 {
			t.Errorf("baseline latency = %d, want 75561", res.BaselineArm.LatencyUS)
		}
		if res.CandidateArm.LatencyUS != 451 {
			t.Errorf("candidate latency = %d, want 451", res.CandidateArm.LatencyUS)
		}
		if res.Speedup <= 100.0 {
			t.Errorf("speedup = %f, want > 100.0", res.Speedup)
		}
		if res.CosineParity != 0.9999999 {
			t.Errorf("cosine parity = %f, want 0.9999999", res.CosineParity)
		}
	})

	t.Run("runTopologyAblation parses real evidence and marks VERIFIED_LIFT", func(t *testing.T) {
		executeStrixAblationCommandFn = func(ctx context.Context, target *StrixTarget, envVars, testPattern string) (string, time.Duration, error) {
			jsonStr := `{"feature": "fused_vs_discrete_norm_matmul", "baseline_latency_us": 28275, "candidate_latency_us": 17400, "cosine_parity": 0.999999}`
			return fmt.Sprintf("=== RUN TestTopology\n%s\n--- PASS: TestTopology (0.05s)\nPASS\n", jsonStr), 50 * time.Millisecond, nil
		}

		res, err := runTopologyAblation(ctx, target)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if res.Verdict != "VERIFIED_LIFT" {
			t.Errorf("verdict = %q, want VERIFIED_LIFT", res.Verdict)
		}
		if res.BaselineArm.LatencyUS != 28275 {
			t.Errorf("baseline latency = %d, want 28275", res.BaselineArm.LatencyUS)
		}
		if res.CandidateArm.LatencyUS != 17400 {
			t.Errorf("candidate latency = %d, want 17400", res.CandidateArm.LatencyUS)
		}
		if res.Speedup <= 1.0 {
			t.Errorf("speedup = %f, want > 1.0", res.Speedup)
		}
	})
}

func TestAblationExecution_SourceBindingMismatchFailsRemoteCommand(t *testing.T) {
	ctx := context.Background()
	ctx = WithSourceBinding(ctx, "0000000000000000000000000000000000000000", "")

	t.Setenv("FAK_STRIX_DIR", ".")
	target := &StrixTarget{
		Host:      "localhost",
		Mode:      "local",
		Reachable: true,
	}

	out, _, err := executeStrixAblationCommand(ctx, target, "", "^TestFakePattern$")
	if err == nil {
		t.Fatal("expected error on source binding mismatch in ablation remote command, got nil")
	}
	if !strings.Contains(out, "source binding mismatch") {
		t.Errorf("expected output to contain 'source binding mismatch', got: %q", out)
	}
}

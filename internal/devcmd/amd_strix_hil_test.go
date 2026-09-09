package devcmd

import (
	"bytes"
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/amdgpu"
)

func TestRunAMDStrixHIL_HelpAndFlags(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := RunAMDStrixHIL(&stdout, &stderr, []string{"--help"})
	if code != 2 {
		t.Fatalf("help exit = %d, want 2", code)
	}

	code = RunAMDStrixHIL(&stdout, &stderr, []string{"extra_arg"})
	if code != 1 {
		t.Fatalf("positional arg exit = %d, want 1", code)
	}
	if !strings.Contains(stderr.String(), "positional arguments rejected") {
		t.Fatalf("unexpected error message: %s", stderr.String())
	}
}

func TestRunAMDStrixHIL_MockExecution(t *testing.T) {
	origHIL := runStrixHILFn
	defer func() { runStrixHILFn = origHIL }()

	runStrixHILFn = func(ctx context.Context, opts amdgpu.StrixHILOpts) (*amdgpu.StrixHILReceipt, error) {
		r := &amdgpu.StrixHILReceipt{
			Schema:    amdgpu.StrixHILReceiptSchema,
			Timestamp: time.Now().UTC().Format(time.RFC3339),
			Verdict:   "PASS",
			Verified:  true,
			Target: amdgpu.StrixTarget{
				Host:         "strix1",
				Mode:         "ssh",
				Reachable:    true,
				TargetISA:    "gfx1151",
				ComputeUnits: 40,
				GPUName:      "AMD Radeon 8060S Graphics",
			},
			SubkernelCount: 19,
			AblationCount:  2,
			InferenceWitness: &amdgpu.StrixHILInferenceWitness{
				Endpoint:         "http://192.168.1.208:8080/v1/chat/completions",
				Model:            "Qwen3.8-27B-Q4_K_M",
				Prompt:           "2+2",
				Response:         "4",
				PromptTokens:     10,
				CompletionTokens: 1,
				TotalTokens:      11,
				TokensPerSec:     50.0,
				Verified:         true,
			},
		}
		digest, _ := r.ComputeDigest()
		r.Digest = digest
		return r, nil
	}

	// 1. Human output mode
	var stdout, stderr bytes.Buffer
	code := RunAMDStrixHIL(&stdout, &stderr, []string{"--subkernels=none", "--ablate=none"})
	if code != 0 {
		t.Fatalf("exit code = %d, want 0, stderr: %s", code, stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, "Hardware-In-The-Loop (HIL) Validation Receipt") {
		t.Errorf("missing HIL banner in output: %s", out)
	}
	if !strings.Contains(out, "Verdict:      PASS") {
		t.Errorf("missing PASS verdict in output: %s", out)
	}
	if !strings.Contains(out, "Live Model Serving Inference Witness") {
		t.Errorf("missing inference witness in output: %s", out)
	}

	// 2. JSON output mode
	stdout.Reset()
	stderr.Reset()
	code = RunAMDStrixHIL(&stdout, &stderr, []string{"--subkernels=none", "--ablate=none", "--json"})
	if code != 0 {
		t.Fatalf("json exit code = %d, want 0, stderr: %s", code, stderr.String())
	}
	jsonOut := stdout.String()
	if !strings.Contains(jsonOut, `"schema": "fak.strix.hil-receipt/v1"`) {
		t.Errorf("missing schema in json output: %s", jsonOut)
	}
	if !strings.Contains(jsonOut, `"verdict": "PASS"`) {
		t.Errorf("missing verdict in json output: %s", jsonOut)
	}
}

func TestRunAMDStrixHIL_Failures(t *testing.T) {
	origHIL := runStrixHILFn
	defer func() { runStrixHILFn = origHIL }()

	var stdout, stderr bytes.Buffer
	code := RunAMDStrixHIL(&stdout, &stderr, []string{"--timeout=0"})
	if code != 1 {
		t.Fatalf("zero timeout exit = %d, want 1", code)
	}

	runStrixHILFn = func(ctx context.Context, opts amdgpu.StrixHILOpts) (*amdgpu.StrixHILReceipt, error) {
		r := &amdgpu.StrixHILReceipt{
			Schema:    amdgpu.StrixHILReceiptSchema,
			Timestamp: time.Now().UTC().Format(time.RFC3339),
			Verdict:   "FAIL",
			Verified:  false,
			Failures:  []string{"subkernel failure simulated"},
		}
		digest, _ := r.ComputeDigest()
		r.Digest = digest
		return r, fmt.Errorf("verification failed")
	}

	stdout.Reset()
	stderr.Reset()
	code = RunAMDStrixHIL(&stdout, &stderr, []string{"--subkernels=none", "--ablate=none"})
	if code != 1 {
		t.Fatalf("failed receipt exit = %d, want 1", code)
	}
	if !strings.Contains(stdout.String(), "subkernel failure simulated") {
		t.Errorf("expected failure message in output: %s", stdout.String())
	}
}

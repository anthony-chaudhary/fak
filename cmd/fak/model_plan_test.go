package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/modelloadplan"
)

// fak-test:runtime fast est=1s lane=default
func TestValidateModelPlanGPUsClosedContract(t *testing.T) {
	if err := validateModelPlanGPUs(0, false); err == nil || errors.Is(err, ErrModelPlanMultiGPUUnsupported) {
		t.Fatalf("gpus=0 = %v, want a plain below-one refusal", err)
	}
	if err := validateModelPlanGPUs(-2, true); err == nil || errors.Is(err, ErrModelPlanMultiGPUUnsupported) {
		t.Fatalf("gpus=-2 = %v, want a plain below-one refusal", err)
	}
	if err := validateModelPlanGPUs(1, false); err != nil {
		t.Fatalf("qwen38 gpus=1 refused: %v", err)
	}
	if err := validateModelPlanGPUs(1, true); err != nil {
		t.Fatalf("v41 gpus=1 refused: %v", err)
	}
	if err := validateModelPlanGPUs(2, false); !errors.Is(err, ErrModelPlanMultiGPUUnsupported) {
		t.Fatalf("qwen38 gpus=2 = %v, want ErrModelPlanMultiGPUUnsupported", err)
	}
	// V4.1 passes through untouched: its planner owns the GPU-count verdict.
	if err := validateModelPlanGPUs(2, true); err != nil {
		t.Fatalf("v41 gpus=2 = %v, want pass-through (nil)", err)
	}
}

// fak-test:runtime fast est=1s lane=default
func TestRunModelPlanQwen38MultiGPURefusedTyped(t *testing.T) {
	var out, errOut bytes.Buffer
	code := runModelPlan(&out, &errOut, []string{"qwen38:27b", "--gpus", "2"})
	if code != 3 {
		t.Fatalf("code=%d stderr=%s, want 3", code, errOut.String())
	}
	if !strings.Contains(errOut.String(), ErrModelPlanMultiGPUUnsupported.Error()) {
		t.Fatalf("stderr missing the typed multi-GPU refusal:\n%s", errOut.String())
	}
}

// fak-test:runtime fast est=1s lane=default
func TestRunModelPlanGPUsBelowOneRefused(t *testing.T) {
	var out, errOut bytes.Buffer
	code := runModelPlan(&out, &errOut, []string{"qwen38:27b", "--gpus", "0"})
	if code != 2 {
		t.Fatalf("code=%d stderr=%s, want 2", code, errOut.String())
	}
	if !strings.Contains(errOut.String(), "--gpus must be >= 1") {
		t.Fatalf("stderr missing the below-one refusal:\n%s", errOut.String())
	}
}

// fak-test:runtime fast est=1s lane=default
func TestRunModelPlanV41MultiGPUPassesThroughToPlanner(t *testing.T) {
	// With --gpus 2 the V4.1 leaf must NOT take the qwen38 multi-GPU refusal;
	// it proceeds into the V4.1 branch, where the missing placement inputs are
	// the next closed-contract refusal.
	var out, errOut bytes.Buffer
	code := runModelPlan(&out, &errOut, []string{"deepseek-v41-flash", "--gpus", "2", "--memory", "split"})
	if code != 2 {
		t.Fatalf("code=%d stderr=%s, want 2", code, errOut.String())
	}
	if !strings.Contains(errOut.String(), "V4.1 requires") {
		t.Fatalf("stderr took the wrong refusal path:\n%s", errOut.String())
	}
}

func TestRunModelPlanJSONCapturesHarnessGoalAndSelection(t *testing.T) {
	var out, errOut bytes.Buffer
	code := runModelPlan(&out, &errOut, []string{"qwen38:27b", "--setup", "shared", "--goal", "quality", "--local", "require", "--device-gib", "27", "--disk-gib", "27", "--json"})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	var p modelloadplan.Plan
	if err := json.Unmarshal(out.Bytes(), &p); err != nil {
		t.Fatal(err)
	}
	if p.Schema != modelloadplan.Schema || p.Request.Setup != "shared" || p.Selected == nil || p.Selected.Quantization != "Q5_K_M" {
		t.Fatalf("plan = %#v", p)
	}
}

func TestRunModelPlanTextMakesHostedFallbackRunnable(t *testing.T) {
	var out, errOut bytes.Buffer
	code := runModelPlan(&out, &errOut, []string{"--setup", "shared"})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	for _, want := range []string{"selected: hosted/openrouter", "qwen/qwen3.8-27b", "OPENROUTER_API_KEY"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("output missing %q:\n%s", want, out.String())
		}
	}
}

func TestRunModelPlanAutoProbesHostMemory(t *testing.T) {
	total, _, known := compute.HostSystemMemoryInfo()
	if !known || total <= 0 {
		t.Skip("host system memory not probeable on this environment")
	}

	var out, errOut bytes.Buffer
	code := runModelPlan(&out, &errOut, []string{"--disk-gib", "50", "--json"})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	var p modelloadplan.Plan
	if err := json.Unmarshal(out.Bytes(), &p); err != nil {
		t.Fatal(err)
	}

	if p.Request.DeviceBytes != total && p.Request.HostBytes != total {
		t.Fatalf("expected auto-probed memory (%d), got device=%d host=%d", total, p.Request.DeviceBytes, p.Request.HostBytes)
	}

	for _, c := range p.Candidates {
		if c.Kind == "local" {
			for _, r := range c.Reasons {
				if strings.Contains(r, "memory capacity is unknown") {
					t.Errorf("candidate %s has unexpected memory capacity unknown reason: %s", c.ID, r)
				}
			}
		}
	}
}

// fak-test:runtime fast est=1s lane=default
func TestRunModelPlanDeviceSpecNamed(t *testing.T) {
	var out, errOut bytes.Buffer
	code := runModelPlan(&out, &errOut, []string{"qwen38:27b", "--device-spec", "h200", "--json"})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s, want 0", code, errOut.String())
	}
	var plan modelloadplan.Plan
	if err := json.Unmarshal(out.Bytes(), &plan); err != nil {
		t.Fatalf("plan JSON: %v", err)
	}
	if plan.Request.DeviceBytes != 141<<30 {
		t.Fatalf("device bytes = %d, want %d (cited H200 141 GiB)", plan.Request.DeviceBytes, int64(141)<<30)
	}
	found := false
	for _, p := range plan.Provenance {
		if strings.Contains(p, "cited hardware spec h200") {
			found = true
		}
	}
	if !found {
		t.Fatalf("provenance missing the device-spec citation: %v", plan.Provenance)
	}
}

// fak-test:runtime fast est=1s lane=default
func TestRunModelPlanDeviceSpecAutoProbe(t *testing.T) {
	orig := modelPlanProbeGPUName
	modelPlanProbeGPUName = func() (string, bool) { return "NVIDIA H200 NVL 141GB PCIe", true }
	t.Cleanup(func() { modelPlanProbeGPUName = orig })
	var out, errOut bytes.Buffer
	code := runModelPlan(&out, &errOut, []string{"qwen38:27b", "--device-spec", "auto", "--json"})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s, want 0", code, errOut.String())
	}
	var plan modelloadplan.Plan
	if err := json.Unmarshal(out.Bytes(), &plan); err != nil {
		t.Fatalf("plan JSON: %v", err)
	}
	if plan.Request.DeviceBytes != 141<<30 {
		t.Fatalf("device bytes = %d, want %d (auto-probed H200 NVL)", plan.Request.DeviceBytes, int64(141)<<30)
	}
	found := false
	for _, p := range plan.Provenance {
		if strings.Contains(p, "cited hardware spec h200-nvl") {
			found = true
		}
	}
	if !found {
		t.Fatalf("provenance missing the h200-nvl citation: %v", plan.Provenance)
	}
}

// fak-test:runtime fast est=1s lane=default
func TestRunModelPlanDeviceSpecAutoNoGPURefused(t *testing.T) {
	orig := modelPlanProbeGPUName
	modelPlanProbeGPUName = func() (string, bool) { return "", false }
	t.Cleanup(func() { modelPlanProbeGPUName = orig })
	var out, errOut bytes.Buffer
	code := runModelPlan(&out, &errOut, []string{"qwen38:27b", "--device-spec", "auto"})
	if code != 2 {
		t.Fatalf("code=%d, want 2", code)
	}
	if !strings.Contains(errOut.String(), "no GPU surface") {
		t.Fatalf("stderr missing the no-GPU refusal:\n%s", errOut.String())
	}
}

// fak-test:runtime fast est=1s lane=default
func TestRunModelPlanDeviceSpecUnknownRefused(t *testing.T) {
	var out, errOut bytes.Buffer
	code := runModelPlan(&out, &errOut, []string{"qwen38:27b", "--device-spec", "totally-unknown-card"})
	if code != 2 {
		t.Fatalf("code=%d, want 2", code)
	}
	if !strings.Contains(errOut.String(), "no cited hardware spec") {
		t.Fatalf("stderr missing the unknown-card refusal:\n%s", errOut.String())
	}
}

// fak-test:runtime fast est=1s lane=default
func TestRunModelPlanDeviceSpecConflictsRefused(t *testing.T) {
	var out, errOut bytes.Buffer
	code := runModelPlan(&out, &errOut, []string{"qwen38:27b", "--device-spec", "h200", "--device-gib", "27"})
	if code != 2 {
		t.Fatalf("code=%d, want 2", code)
	}
	if !strings.Contains(errOut.String(), "mutually exclusive") {
		t.Fatalf("stderr missing the mutual-exclusion refusal:\n%s", errOut.String())
	}
	var out2, errOut2 bytes.Buffer
	code = runModelPlan(&out2, &errOut2, []string{"deepseek-v41-flash", "--device-spec", "h200"})
	if code != 2 {
		t.Fatalf("v41 code=%d, want 2", code)
	}
	if !strings.Contains(errOut2.String(), "not applicable to deepseek-v41-flash") {
		t.Fatalf("stderr missing the v41 refusal:\n%s", errOut2.String())
	}
}

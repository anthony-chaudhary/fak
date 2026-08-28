package main

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/runtimecap"
)

func TestRunRuntimeCapabilitiesJSONWitness(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runRuntimeCapabilities(&stdout, &stderr, nil); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	var report runtimecap.Report
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("decode: %v\n%s", err, stdout.String())
	}
	if report.Schema != runtimecap.Schema || report.GOOS == "" || report.GOARCH == "" {
		t.Fatalf("identity fields = %+v", report)
	}
	if !report.BinaryRunnable || !report.ControlPlaneRunnable || report.ModelExecution.Engine != "fak-native" || report.ModelExecution.Backend != "cpu-ref" {
		t.Fatalf("runtime report = %+v", report)
	}
	if report.ModelExecution.PayloadLoaded {
		t.Fatal("runtime diagnostics loaded a model payload")
	}
}

func TestRunRuntimeCapabilitiesRequestedBackendRefusesSilentFallback(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runRuntimeCapabilities(&stdout, &stderr, []string{"--backend", "definitely-not-a-backend"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	var report runtimecap.Report
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	if report.RequestedBackend == nil || report.RequestedBackend.ExactMatch || report.ModelExecution.Runnable || report.ModelExecution.Backend != "" {
		t.Fatalf("silent fallback: %+v", report)
	}
}

func TestRunRuntimeCapabilitiesPreferredBackendCanEmitLocalCPUDegradedReceipt(t *testing.T) {
	var stdout, stderr bytes.Buffer
	args := []string{
		"--prefer-backend", "vulkan",
		"--fallback-policy", runtimecap.FallbackPolicyLocalCPUDegrade,
		"--cpu-envelope", "qwen25-1p5b-q8-windows-amd64",
		"--goos", "windows",
		"--goarch", "amd64",
		"--host-total-ram-bytes", "17179869184",
		"--host-free-ram-bytes", "12884901888",
	}
	if code := runRuntimeCapabilities(&stdout, &stderr, args); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	var report runtimecap.Report
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("decode: %v\n%s", err, stdout.String())
	}
	if report.PreferredBackend == nil || report.PreferredBackend.Selected != "cpu-ref" {
		t.Fatalf("preferred backend = %+v", report.PreferredBackend)
	}
	if !report.ModelExecution.Runnable || report.ModelExecution.Mode != runtimecap.FallbackPolicyLocalCPUDegrade {
		t.Fatalf("execution = %+v", report.ModelExecution)
	}
	if report.ModelExecution.LocalCPUDegraded == nil || report.ModelExecution.LocalCPUDegraded.RequestedBackend != "vulkan" {
		t.Fatalf("receipt = %+v", report.ModelExecution.LocalCPUDegraded)
	}
}

func TestRunRuntimeCapabilitiesRejectsConflictingBackendFlags(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runRuntimeCapabilities(&stdout, &stderr, []string{"--backend", "cpu-ref", "--prefer-backend", "vulkan"})
	if code != 2 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
}

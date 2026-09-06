package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/nativeperf"
)

func TestRunBenchTraceSubagentJSON(t *testing.T) {
	var stdout, stderr bytes.Buffer
	args := []string{"--subagent", "--json", "--subagent-id", "test-agent", "--turn", "2"}

	code := runBenchTrace(&stdout, &stderr, args)
	if code != 0 {
		t.Fatalf("runBenchTrace failed with code %d, stderr: %s", code, stderr.String())
	}

	var receipt nativeperf.SubagentTraceReceipt
	if err := json.Unmarshal(stdout.Bytes(), &receipt); err != nil {
		t.Fatalf("failed to unmarshal JSON output: %v\nOutput: %s", err, stdout.String())
	}

	if err := receipt.Validate(); err != nil {
		t.Fatalf("receipt failed validation: %v", err)
	}

	if receipt.Schema != nativeperf.SubagentTraceSchema {
		t.Errorf("schema = %q, want %q", receipt.Schema, nativeperf.SubagentTraceSchema)
	}
	if receipt.SubagentID != "test-agent" {
		t.Errorf("subagent_id = %q, want %q", receipt.SubagentID, "test-agent")
	}
	if receipt.Turn != 2 {
		t.Errorf("turn = %d, want 2", receipt.Turn)
	}
	if receipt.TotalWallUS <= 0 {
		t.Errorf("TotalWallUS = %f, want > 0", receipt.TotalWallUS)
	}
	if receipt.GPUKernelWallUS <= 0 {
		t.Errorf("GPUKernelWallUS = %f, want > 0", receipt.GPUKernelWallUS)
	}
	if receipt.HostCPUOverheadUS <= 0 {
		t.Errorf("HostCPUOverheadUS = %f, want > 0", receipt.HostCPUOverheadUS)
	}

	for _, p := range nativeperf.SubagentPhases() {
		if _, ok := receipt.PhasesUS[p]; !ok {
			t.Errorf("missing phase %q in receipt", p)
		}
	}
}

func TestRunBenchTraceSubagentText(t *testing.T) {
	var stdout, stderr bytes.Buffer
	args := []string{"--subagent", "--subagent-id", "reporter-agent"}

	code := runBenchTrace(&stdout, &stderr, args)
	if code != 0 {
		t.Fatalf("runBenchTrace failed with code %d, stderr: %s", code, stderr.String())
	}

	out := stdout.String()
	expectedSubstrings := []string{
		"Subagent Trace Receipt (fak.subagent.trace/v1)",
		"Subagent ID:            reporter-agent",
		"Phase Decomposition:",
		"host_dispatch:",
		"prefix_tree_lookup:",
		"kv_allocation:",
		"gpu_kernel:",
		"token_sampling:",
		"Overhead Isolation:",
		"Host CPU Overhead:",
		"GPU Kernel Wall:",
		"Status: VALIDATED",
	}

	for _, s := range expectedSubstrings {
		if !strings.Contains(out, s) {
			t.Errorf("stdout missing expected substring %q\nOutput:\n%s", s, out)
		}
	}
}

func TestRunBenchTraceSynthetic(t *testing.T) {
	var stdout, stderr bytes.Buffer
	args := []string{"--subagent", "--synthetic", "--json", "--subagent-id", "synthetic-agent", "--turn", "7"}

	code := runBenchTrace(&stdout, &stderr, args)
	if code != 0 {
		t.Fatalf("runBenchTrace failed with code %d, stderr: %s", code, stderr.String())
	}

	var receipt nativeperf.SubagentTraceReceipt
	if err := json.Unmarshal(stdout.Bytes(), &receipt); err != nil {
		t.Fatalf("failed to unmarshal JSON: %v", err)
	}

	if err := receipt.Validate(); err != nil {
		t.Fatalf("receipt validation failed: %v", err)
	}

	// Calibrated synthetic timings:
	// host_dispatch: 145, prefix_tree_lookup: 48, kv_allocation: 32, gpu_kernel: 950, token_sampling: 45
	// Total = 1220 µs. Host = 270 µs (22.13%), GPU = 950 µs (77.87%)
	const wantTotal = 1220.0
	if receipt.TotalWallUS != wantTotal {
		t.Errorf("TotalWallUS = %f, want %f", receipt.TotalWallUS, wantTotal)
	}
	if receipt.GPUKernelWallUS != 950.0 {
		t.Errorf("GPUKernelWallUS = %f, want 950.0", receipt.GPUKernelWallUS)
	}
	if receipt.HostCPUOverheadUS != 270.0 {
		t.Errorf("HostCPUOverheadUS = %f, want 270.0", receipt.HostCPUOverheadUS)
	}
}

func TestRunBenchTraceFileInspection(t *testing.T) {
	tempDir := t.TempDir()
	traceFile := filepath.Join(tempDir, "trace.json")

	phases := map[string]float64{
		nativeperf.SubagentPhaseHostDispatch:     100.0,
		nativeperf.SubagentPhasePrefixTreeLookup: 50.0,
		nativeperf.SubagentPhaseKVAllocation:     50.0,
		nativeperf.SubagentPhaseGPUKernel:        800.0,
		nativeperf.SubagentPhaseTokenSampling:    0.0,
	}
	inputReceipt, err := nativeperf.NewSubagentTraceReceipt(1, "file-agent", phases, 1000.0)
	if err != nil {
		t.Fatalf("failed to create receipt: %v", err)
	}

	data, err := inputReceipt.JSON()
	if err != nil {
		t.Fatalf("failed to marshal receipt: %v", err)
	}
	if err := os.WriteFile(traceFile, data, 0644); err != nil {
		t.Fatalf("failed to write trace file: %v", err)
	}

	var stdout, stderr bytes.Buffer
	args := []string{"--trace", traceFile, "--json"}

	code := runBenchTrace(&stdout, &stderr, args)
	if code != 0 {
		t.Fatalf("runBenchTrace with --trace failed with code %d, stderr: %s", code, stderr.String())
	}

	var parsed nativeperf.SubagentTraceReceipt
	if err := json.Unmarshal(stdout.Bytes(), &parsed); err != nil {
		t.Fatalf("failed to unmarshal JSON: %v", err)
	}

	if parsed.SubagentID != "file-agent" || parsed.TotalWallUS != 1000.0 {
		t.Errorf("parsed receipt mismatch: ID=%s Total=%f", parsed.SubagentID, parsed.TotalWallUS)
	}
}

func TestRunBenchTraceFlagsAndHelp(t *testing.T) {
	// 1. No arguments prints help and returns 0
	var stdout, stderr bytes.Buffer
	code := runBenchTrace(&stdout, &stderr, []string{})
	if code != 0 {
		t.Errorf("empty args code = %d, want 0", code)
	}
	if !strings.Contains(stdout.String(), "Usage: fak bench trace") {
		t.Errorf("empty args did not print help: %s", stdout.String())
	}

	// 2. Help flag returns 0
	stdout.Reset()
	stderr.Reset()
	code = runBenchTrace(&stdout, &stderr, []string{"--help"})
	if code != 0 {
		t.Errorf("--help code = %d, want 0", code)
	}

	// 3. Unknown flag returns 1
	stdout.Reset()
	stderr.Reset()
	code = runBenchTrace(&stdout, &stderr, []string{"--unknown-flag"})
	if code != 1 {
		t.Errorf("--unknown-flag code = %d, want 1", code)
	}

	// 4. Missing target flag returns 2
	stdout.Reset()
	stderr.Reset()
	code = runBenchTrace(&stdout, &stderr, []string{"--turn", "3"})
	if code != 2 {
		t.Errorf("missing target flag code = %d, want 2", code)
	}
	if !strings.Contains(stderr.String(), "target flag required") {
		t.Errorf("stderr missing expected error message: %s", stderr.String())
	}
}

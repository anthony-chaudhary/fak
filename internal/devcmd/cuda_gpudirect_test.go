package devcmd

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func TestRunCUDAGPUDirect_Inspect(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := RunCUDAGPUDirect(&stdout, &stderr, []string{"inspect"})
	if code != 0 {
		t.Fatalf("RunCUDAGPUDirect inspect returned %d: %s", code, stderr.String())
	}
	out := stdout.String()
	for _, expected := range []string{
		"CUDA GPU Direct Topology",
		"RTX 5090",
		"Samsung SSD 990 PRO",
		"DIRECT_CPU_ROOT_COMPLEX",
		"M2A_CPU",
		"ACS Status",
	} {
		if !strings.Contains(out, expected) {
			t.Errorf("inspect output missing expected %q:\n%s", expected, out)
		}
	}

	// Test JSON mode
	stdout.Reset()
	stderr.Reset()
	code = RunCUDAGPUDirect(&stdout, &stderr, []string{"inspect", "--json"})
	if code != 0 {
		t.Fatalf("RunCUDAGPUDirect inspect --json returned %d: %s", code, stderr.String())
	}
	var res map[string]any
	if err := json.Unmarshal(stdout.Bytes(), &res); err != nil {
		t.Fatalf("invalid JSON output: %v\nOutput: %s", err, stdout.String())
	}
	if _, ok := res["gpus"]; !ok {
		t.Errorf("missing 'gpus' in JSON output: %+v", res)
	}
	if _, ok := res["nvme_devices"]; !ok {
		t.Errorf("missing 'nvme_devices' in JSON output: %+v", res)
	}
	if _, ok := res["routes"]; !ok {
		t.Errorf("missing 'routes' in JSON output: %+v", res)
	}
	if _, ok := res["host"]; !ok {
		t.Errorf("missing 'host' in JSON output: %+v", res)
	}
}

func TestRunCUDAGPUDirect_Audit(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := RunCUDAGPUDirect(&stdout, &stderr, []string{"audit"})
	if code != 0 {
		t.Fatalf("RunCUDAGPUDirect audit returned %d: %s", code, stderr.String())
	}
	out := stdout.String()
	for _, expected := range []string{
		"CUDA GPU Direct & BaM P2PDMA Hardware Audit",
		"Above 4G Decoding",
		"Resizable BAR (ReBAR)",
		"PCIe Ten Bit Tag",
		"IOMMU / ACS",
		"NVreg_EnableResizableBar=1",
		"Healthy:             true (Verdict: PASS)",
	} {
		if !strings.Contains(out, expected) {
			t.Errorf("audit output missing expected %q:\n%s", expected, out)
		}
	}

	// Test JSON mode
	stdout.Reset()
	stderr.Reset()
	code = RunCUDAGPUDirect(&stdout, &stderr, []string{"audit", "--json"})
	if code != 0 {
		t.Fatalf("RunCUDAGPUDirect audit --json returned %d: %s", code, stderr.String())
	}
	var res map[string]any
	if err := json.Unmarshal(stdout.Bytes(), &res); err != nil {
		t.Fatalf("invalid JSON output: %v\nOutput: %s", err, stdout.String())
	}
	if res["healthy"] != true {
		t.Errorf("expected healthy=true, got %+v", res)
	}
	if res["verdict"] != "PASS" {
		t.Errorf("expected verdict=PASS, got %+v", res)
	}
	checks, ok := res["checks"].([]any)
	if !ok || len(checks) < 5 {
		t.Errorf("expected at least 5 audit checks, got %+v", res["checks"])
	}
}

func TestRunCUDAGPUDirect_Bench(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := RunCUDAGPUDirect(&stdout, &stderr, []string{"bench"})
	if code != 0 {
		t.Fatalf("RunCUDAGPUDirect bench returned %d: %s", code, stderr.String())
	}
	out := stdout.String()
	for _, expected := range []string{
		"CUDA BaM NVMe P2PDMA Zero-Copy Microbenchmark Results",
		"Staging Copies:       0",
		"Zero-Copy Invariant:  VERIFIED",
		"Throughput:",
		"IOPS:",
	} {
		if !strings.Contains(out, expected) {
			t.Errorf("bench output missing expected %q:\n%s", expected, out)
		}
	}

	// Test JSON mode
	stdout.Reset()
	stderr.Reset()
	code = RunCUDAGPUDirect(&stdout, &stderr, []string{"bench", "--json"})
	if code != 0 {
		t.Fatalf("RunCUDAGPUDirect bench --json returned %d: %s", code, stderr.String())
	}
	var res map[string]any
	if err := json.Unmarshal(stdout.Bytes(), &res); err != nil {
		t.Fatalf("invalid JSON output: %v\nOutput: %s", err, stdout.String())
	}
	if res["staging_copy_count"] != float64(0) {
		t.Errorf("expected staging_copy_count=0, got %+v", res["staging_copy_count"])
	}
	if res["zero_copy_verified"] != true {
		t.Errorf("expected zero_copy_verified=true, got %+v", res["zero_copy_verified"])
	}
	if res["throughput_gbps"].(float64) <= 0 {
		t.Errorf("expected positive throughput_gbps, got %+v", res["throughput_gbps"])
	}
	if res["iops"].(float64) <= 0 {
		t.Errorf("expected positive iops, got %+v", res["iops"])
	}
}

func TestRunCUDAGPUDirect_Qwen38(t *testing.T) {
	// 1. Verify human table output
	var stdout, stderr bytes.Buffer
	code := RunCUDAGPUDirect(&stdout, &stderr, []string{"qwen38"})
	if code != 0 {
		t.Fatalf("RunCUDAGPUDirect qwen38 failed with code %d: %s", code, stderr.String())
	}
	out := stdout.String()
	for _, expected := range []string{
		"Qwen3.8 CUDA Direct Storage & Cache Swap Architecture",
		"Baseline (CPU-staged)",
		"Reference (llama.cpp)",
		"fak-native (CUDA BaM P2PDMA)",
		"Speedup vs Baseline",
		"Speedup vs Reference",
		"Zero-copy NVMe P2PDMA validated",
	} {
		if !strings.Contains(out, expected) {
			t.Errorf("qwen38 output missing expected %q:\n%s", expected, out)
		}
	}

	// 2. Verify machine-readable JSON output
	stdout.Reset()
	stderr.Reset()
	code = RunCUDAGPUDirect(&stdout, &stderr, []string{"qwen38", "--json"})
	if code != 0 {
		t.Fatalf("RunCUDAGPUDirect qwen38 --json returned %d: %s", code, stderr.String())
	}
	var receipt Qwen38CUDADirectSwapReceipt
	if err := json.Unmarshal(stdout.Bytes(), &receipt); err != nil {
		t.Fatalf("invalid JSON output: %v\nOutput: %s", err, stdout.String())
	}

	if receipt.Schema != Qwen38CUDADirectSwapReceiptSchema {
		t.Errorf("expected schema %q, got %q", Qwen38CUDADirectSwapReceiptSchema, receipt.Schema)
	}
	if receipt.Verdict != "PASS" {
		t.Errorf("expected verdict 'PASS', got %q", receipt.Verdict)
	}
	if receipt.StagingCopyCount != 0 {
		t.Errorf("expected staging copy count 0, got %d", receipt.StagingCopyCount)
	}
	if receipt.DirectDMABandwidthGBps <= 0 {
		t.Errorf("expected positive direct DMA bandwidth, got %f", receipt.DirectDMABandwidthGBps)
	}
	if receipt.FakNative.StagingCopyCount != 0 {
		t.Errorf("expected fak_native staging copy count 0, got %d", receipt.FakNative.StagingCopyCount)
	}
	if receipt.Baseline.StagingCopyCount != 3 {
		t.Errorf("expected baseline staging copy count 3, got %d", receipt.Baseline.StagingCopyCount)
	}
	if receipt.SpeedupVsBaseline.TTFTSpeedup <= 1.0 {
		t.Errorf("expected TTFT speedup > 1.0, got %f", receipt.SpeedupVsBaseline.TTFTSpeedup)
	}

	// 3. Test alias subcommands
	for _, alias := range []string{"qwen38-bench", "qwen38-sim"} {
		stdout.Reset()
		stderr.Reset()
		code = RunCUDAGPUDirect(&stdout, &stderr, []string{alias})
		if code != 0 {
			t.Errorf("RunCUDAGPUDirect(%q) failed with code %d: %s", alias, code, stderr.String())
		}
	}
}

func TestRunCUDAGPUDirect_InvalidMode(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := RunCUDAGPUDirect(&stdout, &stderr, []string{"nonexistent_mode"})
	if code != 2 {
		t.Fatalf("expected exit code 2 for invalid mode, got %d", code)
	}
	if !strings.Contains(stderr.String(), "unknown mode") {
		t.Errorf("expected 'unknown mode' in stderr, got: %s", stderr.String())
	}
}

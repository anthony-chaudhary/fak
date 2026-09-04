package devcmd

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func TestQwen38OverflowBench(t *testing.T) {
	// 1. Verify formatted human table output
	var stdout, stderr bytes.Buffer
	code := runQwen38OverflowBench(&stdout, &stderr, nil, false)
	if code != 0 {
		t.Fatalf("runQwen38OverflowBench human table failed with code %d: %s", code, stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, "Qwen3.8 GPU Direct") {
		t.Errorf("expected header 'Qwen3.8 GPU Direct', got:\n%s", out)
	}
	if !strings.Contains(out, "Baseline (CPU-staged)") {
		t.Errorf("expected 'Baseline (CPU-staged)' in output, got:\n%s", out)
	}
	if !strings.Contains(out, "fak-native (GPU Direct)") {
		t.Errorf("expected 'fak-native (GPU Direct)' in output, got:\n%s", out)
	}
	if !strings.Contains(out, "Reference (llama.cpp)") {
		t.Errorf("expected 'Reference (llama.cpp)' in output, got:\n%s", out)
	}
	if !strings.Contains(out, "Speedup vs Baseline") {
		t.Errorf("expected 'Speedup vs Baseline' in output, got:\n%s", out)
	}

	// 2. Verify machine-readable JSON output
	stdout.Reset()
	stderr.Reset()
	code = runQwen38OverflowBench(&stdout, &stderr, nil, true)
	if code != 0 {
		t.Fatalf("runQwen38OverflowBench JSON failed with code %d: %s", code, stderr.String())
	}

	var res map[string]any
	if err := json.Unmarshal(stdout.Bytes(), &res); err != nil {
		t.Fatalf("invalid JSON output: %v\nOutput: %s", err, stdout.String())
	}

	if res["schema"] != "fak.modelengine.qwen38-gpudirect-swap/1" {
		t.Errorf("expected schema 'fak.modelengine.qwen38-gpudirect-swap/1', got %v", res["schema"])
	}

	stagingCopyRaw, ok := res["staging_copy_count"]
	if !ok {
		t.Fatalf("missing top-level 'staging_copy_count' in JSON: %+v", res)
	}
	if stagingCopyRaw != float64(0) {
		t.Errorf("expected top-level staging_copy_count == 0, got %v", stagingCopyRaw)
	}

	arms, ok := res["arms"].(map[string]any)
	if !ok {
		t.Fatalf("missing 'arms' object in JSON: %+v", res)
	}

	baseline, ok := arms["baseline"].(map[string]any)
	if !ok {
		t.Fatalf("missing baseline arm in JSON: %+v", arms)
	}
	if baseline["staging_copy_count"] != float64(3) {
		t.Errorf("expected baseline staging_copy_count == 3, got %v", baseline["staging_copy_count"])
	}
	if baseline["ttft_ms"].(float64) <= 0 {
		t.Errorf("expected positive baseline TTFT, got %v", baseline["ttft_ms"])
	}

	native, ok := arms["fak_native"].(map[string]any)
	if !ok {
		t.Fatalf("missing fak_native arm in JSON: %+v", arms)
	}
	if native["staging_copy_count"] != float64(0) {
		t.Errorf("expected fak_native staging_copy_count == 0, got %v", native["staging_copy_count"])
	}
	if native["ttft_ms"].(float64) <= 0 {
		t.Errorf("expected positive fak_native TTFT, got %v", native["ttft_ms"])
	}
	if native["bandwidth_gbps"].(float64) <= 0 {
		t.Errorf("expected positive direct DMA bandwidth, got %v", native["bandwidth_gbps"])
	}

	reference, ok := arms["reference"].(map[string]any)
	if !ok {
		t.Fatalf("missing reference arm in JSON: %+v", arms)
	}
	if reference["ttft_ms"].(float64) <= 0 {
		t.Errorf("expected positive reference TTFT, got %v", reference["ttft_ms"])
	}

	// 3. Verify subcommands via RunAMDGPUDirect
	for _, sub := range []string{"qwen38", "qwen38-bench", "qwen38-overflow"} {
		stdout.Reset()
		stderr.Reset()
		code = RunAMDGPUDirect(&stdout, &stderr, []string{sub})
		if code != 0 {
			t.Errorf("RunAMDGPUDirect(%q) failed with code %d: %s", sub, code, stderr.String())
		}
	}
}

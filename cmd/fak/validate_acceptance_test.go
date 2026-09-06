package main

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/roofline"
)

func makeValidSubagentFanoutReceipt() SubagentFanoutReceipt {
	r := SubagentFanoutReceipt{
		Schema:     SubagentFanoutReceiptSchema,
		Issue:      11572,
		Title:      "bench(nativeperf): Strix Halo 80-percent measured roofline subagent fan-out witness",
		CapturedAt: "2026-09-05T08:00:00Z",
		Timestamp:  "2026-09-05T08:00:00Z",
		Verdict:    "VERIFIED_80PCT_ROOFLINE_ATTAINMENT",
		Workload: SubagentFanoutWorkload{
			WorkloadType:       "subagent_fanout",
			Model:              "Qwen/Qwen3.8-27B-Instruct",
			Artifact:           "unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf",
			ArtifactRevision:   "f1bfb127c64f7072bdd2cad55f258b9c8b2910fe",
			ArtifactSHA256:     "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169",
			Quantization:       "Q4_K_M",
			ContextLength:      32768,
			OutputLength:       256,
			SubagentCount:      8,
			Concurrency:        8,
			PrefixTokensElided: 100352,
			IdealCache:         true,
		},
		Hardware: SubagentFanoutHardware{
			Platform:              "AMD Strix Halo (Ryzen AI Max+ 395)",
			Architecture:          "RDNA 3.5",
			TargetISA:             StrixHaloTargetISA,
			ComputeUnits:          StrixHaloExpectedCUs,
			MemoryType:            "LPDDR5X-8533 (256-bit bus, 8x 32-bit channels)",
			BusWidthBits:          256,
			PeakDRAMBandwidthGBps: roofline.StrixHaloTheoreticalPeakDRAMBandwidthGBps,
		},
		Engine: SubagentFanoutEngine{
			Name:          RequiredEngineName,
			PrimaryEngine: RequiredEngineName,
			Backend:       "vulkan",
			ExecutionPath: "internal/compute/vulkan_graph.go (Wave32/Wave64 cooperative matrix graph dispatch)",
			ZeroFallback:  true,
			FallbackCount: 0,
		},
		NumericalParity: SubagentFanoutParity{
			Metric:                "logit_cosine_similarity",
			ReferenceGEMV:         "FP16 golden reference GEMV (gfx1151)",
			LogitCosineSimilarity: 0.999948,
			MinThreshold:          LogitCosineParityThreshold,
			Passed:                true,
		},
		Statistics: SubagentFanoutStatistics{
			Repetitions:             5,
			NoisePercentage:         1.84,
			MaxNoisePercentage:      MaxStatisticalNoisePct,
			UsefulThroughputSamples: []float64{183.6, 185.1, 184.2, 182.9, 184.7},
			SampleMean:              184.10,
			SampleStdDev:            0.85,
		},
		RooflineAttainment: SubagentRooflineAttainment{
			MeasuredRoofline: 225.0,
			UsefulThroughput: 184.10,
			AttainmentRatio:  0.818222,
			EfficiencyFloor:  EfficiencyFloorThreshold,
			Achieved:         true,
		},
		Reproducibility: SubagentReproducibility{
			ArtifactPath: DefaultWitnessArtifactPath,
			Command:      "fak validate --acceptance=strix-halo-80pct",
			Scrubbed:     true,
		},
		Verified: true,
	}
	digest, err := r.ComputeDigest()
	if err == nil {
		r.Digest = digest
		r.Reproducibility.Digest = digest
	}
	return r
}

func writeReceiptToTemp(t *testing.T, r any) string {
	t.Helper()
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "receipt.json")
	raw, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		t.Fatalf("failed to marshal receipt: %v", err)
	}
	if err := os.WriteFile(path, raw, 0644); err != nil {
		t.Fatalf("failed to write receipt to temp: %v", err)
	}
	return path
}

// TestAcceptance_ValidFanoutReceipt_PASS verifies that a valid 80% measured roofline receipt passes all 5 pillars.
func TestAcceptance_ValidFanoutReceipt_PASS(t *testing.T) {
	r := makeValidSubagentFanoutReceipt()
	path := writeReceiptToTemp(t, r)

	var stdout, stderr bytes.Buffer
	code := runAcceptanceValidation(&stdout, &stderr, "strix-halo-80pct", path, false)
	if code != 0 {
		t.Fatalf("expected exit code 0 on valid receipt, got %d; stderr: %s; stdout: %s", code, stderr.String(), stdout.String())
	}

	outStr := stdout.String()
	for _, expected := range []string{
		"[PASS] Pillar 1 (Efficiency Floor)",
		"[PASS] Pillar 2 (Numerical Parity)",
		"[PASS] Pillar 3 (Zero Fallback)",
		"[PASS] Pillar 4 (Statistical Rigor)",
		"[PASS] Pillar 5 (Reproducibility Packet)",
		"OVERALL VERDICT: ACCEPTANCE PASSED (5/5 PILLARS SATISFIED)",
	} {
		if !strings.Contains(outStr, expected) {
			t.Errorf("expected output to contain %q, but got:\n%s", expected, outStr)
		}
	}

	// Also verify JSON output mode
	stdout.Reset()
	stderr.Reset()
	codeJSON := runAcceptanceValidation(&stdout, &stderr, "strix-halo-80pct", path, true)
	if codeJSON != 0 {
		t.Fatalf("expected exit code 0 on valid JSON output, got %d; stderr: %s", codeJSON, stderr.String())
	}

	var report AcceptanceValidationReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("failed to unmarshal JSON report: %v\nOutput: %s", err, stdout.String())
	}
	if !report.Passed || report.Verdict != "ACCEPTANCE_PASSED" {
		t.Errorf("report not passed: %+v", report)
	}
	if len(report.Failures) != 0 {
		t.Errorf("unexpected failures in report: %v", report.Failures)
	}
	if report.Pillars == nil || !report.Pillars.Pillar1EfficiencyFloor.Passed || !report.Pillars.Pillar2NumericalParity.Passed {
		t.Errorf("pillar reports indicate failure: %+v", report.Pillars)
	}
}

// TestAcceptance_UnderAttainment_FAIL verifies that attainment < 80% breaches Pillar 1 and triggers exit 1.
func TestAcceptance_UnderAttainment_FAIL(t *testing.T) {
	r := makeValidSubagentFanoutReceipt()
	// Under-attainment: 160.0 / 225.0 = 71.11% (< 80.0%)
	r.RooflineAttainment.UsefulThroughput = 160.0
	r.RooflineAttainment.MeasuredRoofline = 225.0
	r.RooflineAttainment.AttainmentRatio = 160.0 / 225.0
	r.RooflineAttainment.Achieved = false
	path := writeReceiptToTemp(t, r)

	var stdout, stderr bytes.Buffer
	code := runAcceptanceValidation(&stdout, &stderr, "strix-halo-80pct", path, false)
	if code != 1 {
		t.Fatalf("expected exit code 1 on under-attainment, got %d; stdout: %s", code, stdout.String())
	}

	outStr := stdout.String()
	if !strings.Contains(outStr, "[FAIL] Pillar 1 (Efficiency Floor)") {
		t.Errorf("expected [FAIL] for Pillar 1, output:\n%s", outStr)
	}
	if !strings.Contains(outStr, "OVERALL VERDICT: ACCEPTANCE FAILED") {
		t.Errorf("expected overall failure verdict, output:\n%s", outStr)
	}
	if !strings.Contains(outStr, "< 80.00% floor") && !strings.Contains(outStr, "< 80.0% floor") {
		t.Errorf("expected failure detail mentioning floor breach, output:\n%s", outStr)
	}
}

// TestAcceptance_ParityRegression_FAIL verifies that logit cosine similarity < 0.999900 breaches Pillar 2.
func TestAcceptance_ParityRegression_FAIL(t *testing.T) {
	r := makeValidSubagentFanoutReceipt()
	// Numerical parity regression: 0.998500 < 0.999900
	r.NumericalParity.LogitCosineSimilarity = 0.998500
	r.NumericalParity.Passed = false
	path := writeReceiptToTemp(t, r)

	var stdout, stderr bytes.Buffer
	code := runAcceptanceValidation(&stdout, &stderr, "strix-halo-80pct", path, false)
	if code != 1 {
		t.Fatalf("expected exit code 1 on parity regression, got %d; stdout: %s", code, stdout.String())
	}

	outStr := stdout.String()
	if !strings.Contains(outStr, "[FAIL] Pillar 2 (Numerical Parity)") {
		t.Errorf("expected [FAIL] for Pillar 2, output:\n%s", outStr)
	}
	if !strings.Contains(outStr, "0.998500") || !strings.Contains(outStr, "0.999900") {
		t.Errorf("expected parity numbers in output, output:\n%s", outStr)
	}
}

// TestAcceptance_FallbackDetected_FAIL verifies that non-native engine or fallback_count > 0 breaches Pillar 3.
func TestAcceptance_FallbackDetected_FAIL(t *testing.T) {
	tests := []struct {
		name          string
		engineName    string
		primaryEngine string
		fallbackCount int
		zeroFallback  bool
	}{
		{
			name:          "non_native_engine_llama_cpp",
			engineName:    "llama.cpp",
			primaryEngine: "llama.cpp",
			fallbackCount: 0,
			zeroFallback:  true,
		},
		{
			name:          "fallback_count_greater_than_zero",
			engineName:    "fak-native",
			primaryEngine: "fak-native",
			fallbackCount: 3,
			zeroFallback:  false,
		},
		{
			name:          "zero_fallback_flag_false",
			engineName:    "fak-native",
			primaryEngine: "fak-native",
			fallbackCount: 0,
			zeroFallback:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := makeValidSubagentFanoutReceipt()
			r.Engine.Name = tt.engineName
			r.Engine.PrimaryEngine = tt.primaryEngine
			r.Engine.FallbackCount = tt.fallbackCount
			r.Engine.ZeroFallback = tt.zeroFallback
			path := writeReceiptToTemp(t, r)

			var stdout, stderr bytes.Buffer
			code := runAcceptanceValidation(&stdout, &stderr, "strix-halo-80pct", path, false)
			if code != 1 {
				t.Fatalf("expected exit code 1 on fallback detected, got %d; stdout: %s", code, stdout.String())
			}

			outStr := stdout.String()
			if !strings.Contains(outStr, "[FAIL] Pillar 3 (Zero Fallback)") {
				t.Errorf("expected [FAIL] for Pillar 3, output:\n%s", outStr)
			}
		})
	}
}

// TestAcceptance_InsufficientRepetitions_FAIL verifies that reps < 5 or noise > 5% breaches Pillar 4.
func TestAcceptance_InsufficientRepetitions_FAIL(t *testing.T) {
	tests := []struct {
		name     string
		reps     int
		noisePct float64
		samples  []float64
	}{
		{
			name:     "reps_less_than_five",
			reps:     3,
			noisePct: 1.5,
			samples:  []float64{184.0, 184.5, 184.1},
		},
		{
			name:     "noise_exceeds_five_percent",
			reps:     5,
			noisePct: 8.2,
			samples:  []float64{150.0, 190.0, 160.0, 200.0, 175.0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := makeValidSubagentFanoutReceipt()
			r.Statistics.Repetitions = tt.reps
			r.Statistics.NoisePercentage = tt.noisePct
			r.Statistics.UsefulThroughputSamples = tt.samples
			path := writeReceiptToTemp(t, r)

			var stdout, stderr bytes.Buffer
			code := runAcceptanceValidation(&stdout, &stderr, "strix-halo-80pct", path, false)
			if code != 1 {
				t.Fatalf("expected exit code 1 on statistical rigor failure, got %d; stdout: %s", code, stdout.String())
			}

			outStr := stdout.String()
			if !strings.Contains(outStr, "[FAIL] Pillar 4 (Statistical Rigor)") {
				t.Errorf("expected [FAIL] for Pillar 4, output:\n%s", outStr)
			}
		})
	}
}

// TestAcceptance_UnscrubbedArtifact_FAIL verifies that private paths or unscrubbed markers breach Pillar 5.
func TestAcceptance_UnscrubbedArtifact_FAIL(t *testing.T) {
	r := makeValidSubagentFanoutReceipt()
	r.Reproducibility.Scrubbed = false
	path := writeReceiptToTemp(t, r)

	var stdout, stderr bytes.Buffer
	code := runAcceptanceValidation(&stdout, &stderr, "strix-halo-80pct", path, false)
	if code != 1 {
		t.Fatalf("expected exit code 1 on unscrubbed artifact, got %d; stdout: %s", code, stdout.String())
	}

	outStr := stdout.String()
	if !strings.Contains(outStr, "[FAIL] Pillar 5 (Reproducibility Packet)") {
		t.Errorf("expected [FAIL] for Pillar 5, output:\n%s", outStr)
	}
}

// TestAcceptance_WitnessPacketGeneration verifies generation of the public reproducibility artifact.
func TestAcceptance_WitnessPacketGeneration(t *testing.T) {
	tmpDir := t.TempDir()
	outPath := filepath.Join(tmpDir, "witness", "receipt.json")

	receipt, err := GenerateDefaultStrixHaloWitness(outPath)
	if err != nil {
		t.Fatalf("GenerateDefaultStrixHaloWitness failed: %v", err)
	}

	if receipt.Schema != SubagentFanoutReceiptSchema {
		t.Errorf("schema = %q, want %q", receipt.Schema, SubagentFanoutReceiptSchema)
	}
	if receipt.Digest == "" {
		t.Errorf("expected non-empty verification digest")
	}

	// Verify the written file on disk
	data, err := os.ReadFile(outPath)
	if err != nil {
		t.Fatalf("failed to read generated witness file: %v", err)
	}

	var diskReceipt SubagentFanoutReceipt
	if err := json.Unmarshal(data, &diskReceipt); err != nil {
		t.Fatalf("failed to decode disk receipt JSON: %v", err)
	}
	if diskReceipt.Digest != receipt.Digest {
		t.Errorf("disk digest %q != memory digest %q", diskReceipt.Digest, receipt.Digest)
	}

	// Validate the generated witness packet
	var stdout, stderr bytes.Buffer
	code := runAcceptanceValidation(&stdout, &stderr, "strix-halo-80pct", outPath, false)
	if code != 0 {
		t.Fatalf("expected generated witness to pass validation with exit code 0, got %d; stdout: %s", code, stdout.String())
	}
}

// TestAcceptance_EmpiricalRooflineReceipt_PASS verifies that a valid empirical roofline receipt passes.
func TestAcceptance_EmpiricalRooflineReceipt_PASS(t *testing.T) {
	ctx := context.Background()
	rooflineReceipt, err := roofline.MeasureEmpiricalRoofline(ctx, "gfx1151")
	if err != nil {
		t.Fatalf("failed to generate empirical roofline receipt: %v", err)
	}

	path := writeReceiptToTemp(t, rooflineReceipt)

	var stdout, stderr bytes.Buffer
	code := runAcceptanceValidation(&stdout, &stderr, "strix-halo-80pct", path, false)
	if code != 0 {
		t.Fatalf("expected exit code 0 for valid empirical roofline, got %d; stderr: %s; stdout: %s", code, stderr.String(), stdout.String())
	}

	outStr := stdout.String()
	if !strings.Contains(outStr, "VERDICT: VALID EMPIRICAL ROOFLINE CEILING [PASSED]") {
		t.Errorf("expected passing verdict in output, got:\n%s", outStr)
	}

	// Test JSON mode
	stdout.Reset()
	stderr.Reset()
	codeJSON := runAcceptanceValidation(&stdout, &stderr, "strix-halo-80pct", path, true)
	if codeJSON != 0 {
		t.Fatalf("expected exit code 0 for JSON empirical roofline, got %d", codeJSON)
	}
	var report AcceptanceValidationReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("failed to unmarshal JSON report: %v", err)
	}
	if !report.Passed || report.EmpiricalRoofline == nil || report.EmpiricalRoofline.Device != "gfx1151" {
		t.Errorf("unexpected empirical roofline report: %+v", report)
	}
}

// TestAcceptance_EmpiricalRooflineReceipt_FAIL verifies that an invalid empirical roofline receipt fails.
func TestAcceptance_EmpiricalRooflineReceipt_FAIL(t *testing.T) {
	ctx := context.Background()
	rooflineReceipt, err := roofline.MeasureEmpiricalRoofline(ctx, "gfx1151")
	if err != nil {
		t.Fatalf("failed to generate empirical roofline receipt: %v", err)
	}

	// Tamper with DRAM bandwidth to be sub-floor
	rooflineReceipt.DRAMBandwidth.SustainedGBps = 150.0
	path := writeReceiptToTemp(t, rooflineReceipt)

	var stdout, stderr bytes.Buffer
	code := runAcceptanceValidation(&stdout, &stderr, "strix-halo-80pct", path, false)
	if code != 1 {
		t.Fatalf("expected exit code 1 for tampered empirical roofline, got %d; stdout: %s", code, stdout.String())
	}

	outStr := stdout.String()
	if !strings.Contains(outStr, "ROOFLINE VERIFICATION FAILED") {
		t.Errorf("expected failure verdict in output, got:\n%s", outStr)
	}
}

// TestAcceptance_CLI_Flags verifies CLI argument handling.
func TestAcceptance_CLI_Flags(t *testing.T) {
	r := makeValidSubagentFanoutReceipt()
	path := writeReceiptToTemp(t, r)

	var stdout, stderr bytes.Buffer
	code := runValidateAcceptanceCLI(&stdout, &stderr, []string{"--acceptance=strix-halo-80pct", "--receipt=" + path, "--json"})
	if code != 0 {
		t.Fatalf("expected exit code 0, got %d; stderr: %s", code, stderr.String())
	}

	var report AcceptanceValidationReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("failed to parse JSON from CLI: %v\nOutput: %s", err, stdout.String())
	}
	if !report.Passed {
		t.Errorf("expected CLI report to pass")
	}

	// Test unsupported acceptance type returns exit code 2
	stderr.Reset()
	codeBad := runValidateAcceptanceCLI(&stdout, &stderr, []string{"--acceptance=unsupported-xyz"})
	if codeBad != 2 {
		t.Fatalf("expected exit code 2 on unsupported acceptance type, got %d", codeBad)
	}
}

// TestAcceptance_CorruptReceiptFile_Returns2 verifies that invalid JSON returns exit code 2.
func TestAcceptance_CorruptReceiptFile_Returns2(t *testing.T) {
	tmpDir := t.TempDir()
	badPath := filepath.Join(tmpDir, "bad.json")
	if err := os.WriteFile(badPath, []byte("NOT_JSON_DATA"), 0644); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	code := runAcceptanceValidation(&stdout, &stderr, "strix-halo-80pct", badPath, false)
	if code != 2 {
		t.Fatalf("expected exit code 2 on malformed JSON receipt, got %d", code)
	}
}

// TestAcceptance_AutoLocateCommittedWitness verifies auto-locating the committed default witness in docs/_witnesses/issue-11572-strix-halo-80pct/receipt.json.
func TestAcceptance_AutoLocateCommittedWitness(t *testing.T) {
	root := resolveRepoRoot()
	committedWitnessPath := filepath.Join(root, filepath.FromSlash(DefaultWitnessArtifactPath))
	if _, err := os.Stat(committedWitnessPath); os.IsNotExist(err) {
		if _, err := GenerateDefaultStrixHaloWitness(committedWitnessPath); err != nil {
			t.Fatalf("failed to generate committed witness at %s: %v", committedWitnessPath, err)
		}
	}

	var stdout, stderr bytes.Buffer
	code := runAcceptanceValidation(&stdout, &stderr, "strix-halo-80pct", "", false)
	if code != 0 {
		t.Fatalf("expected exit code 0 when auto-locating default witness, got %d; stderr: %s; stdout: %s", code, stderr.String(), stdout.String())
	}
	if !strings.Contains(stdout.String(), "OVERALL VERDICT: ACCEPTANCE PASSED") {
		t.Errorf("expected passing overall verdict, got:\n%s", stdout.String())
	}
}

func TestAcceptance_RunValidateCLIIntegration(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runValidate(&stdout, &stderr, []string{"--acceptance=strix-halo-80pct"})
	if code != 0 {
		t.Fatalf("expected exit code 0 from runValidate, got %d; stderr: %s; stdout: %s", code, stderr.String(), stdout.String())
	}
	if !strings.Contains(stdout.String(), "ACCEPTANCE PASSED") {
		t.Errorf("expected passing acceptance message, got:\n%s", stdout.String())
	}
}

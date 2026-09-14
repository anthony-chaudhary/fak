package qwen38quantrun

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func TestCaptureStrixModelbenchC1ObservationIsRunnerBound(t *testing.T) {
	manifest, digest, executable := validC1AdapterManifest(t)
	receipt := c1AdapterReceipt(t, manifest)
	report := c1AdapterReportJSON(t, receipt)

	var capturedArgv, capturedEnv []string
	runner := func(_ context.Context, binary string, argv, env []string) ([]byte, error) {
		if binary == "" {
			t.Fatal("runner received an empty binary")
		}
		capturedArgv = slices.Clone(argv)
		capturedEnv = slices.Clone(env)
		return report, nil
	}
	observation, err := captureStrixModelbenchC1(context.Background(), StrixModelbenchC1Request{
		Manifest: manifest, ManifestDigest: digest,
		ExecutablePath: executable, ArtifactPath: filepath.Join(t.TempDir(), "model.gguf"),
		ExpectedExecutable: manifest.Candidate.ExecutableSHA256,
		Environment:        []string{"PATH=/usr/bin", "SECRET=do-not-leak", "HOME=/root"},
	}, runner)
	if err != nil {
		t.Fatalf("capture failed: %v", err)
	}
	if !observation.Valid() {
		t.Fatal("observation is not valid after a successful capture")
	}
	if concurrency, ok := observation.ChallengeConcurrency(); !ok || concurrency != 1 {
		t.Fatalf("concurrency = %d, ok=%v; want c=1", concurrency, ok)
	}
	if got, ok := observation.ApprovedCellDigest(); !ok || got != digest {
		t.Fatalf("approved digest = %q, ok=%v; want %q", got, ok, digest)
	}
	if credit, ok := observation.PerformanceCredit(); credit || !ok {
		t.Fatalf("performance credit = %v, ok=%v; want false/true", credit, ok)
	}
	if reason, ok := observation.PerformanceCreditReason(); !ok || reason != StrixModelbenchC1PerformanceCreditReason {
		t.Fatalf("credit reason = %q, ok=%v; want %q", reason, ok, StrixModelbenchC1PerformanceCreditReason)
	}
	bound, ok := observation.Receipt()
	if !ok {
		t.Fatal("bound receipt unavailable")
	}
	if bound.Schema != compute.Qwen38VulkanDecodeReceiptV3Schema || len(bound.Runs) != 1 {
		t.Fatalf("bound receipt schema/runs = %q/%d", bound.Schema, len(bound.Runs))
	}
	if capturedArgv[0] != executable {
		t.Fatalf("argv[0] = %q, want the pinned executable", capturedArgv[0])
	}
	for _, flag := range []string{"-raw-decode", "-raw-ignore-eos", "-require-non-reference"} {
		if !containsArg(capturedArgv, flag) {
			t.Fatalf("argv is missing %q: %v", flag, capturedArgv)
		}
	}
	if !pairArg(capturedArgv, "-decode-reps", "1") {
		t.Fatalf("argv must pin c=1 decode reps: %v", capturedArgv)
	}
	if !pairArg(capturedArgv, "-decode-steps", "128") {
		t.Fatalf("argv must use the manifest output boundary: %v", capturedArgv)
	}
	for _, entry := range capturedEnv {
		if strings.Contains(entry, "SECRET") {
			t.Fatalf("scrubbed environment leaked a non-allowlisted variable: %q", entry)
		}
	}
	if !containsEntry(capturedEnv, "PATH=/usr/bin") || !containsEntry(capturedEnv, "HOME=/root") {
		t.Fatalf("scrubbed environment dropped an allowlisted variable: %v", capturedEnv)
	}

	mutated, _ := observation.Receipt()
	mutated.Runs[0].OutputTokenIDs = nil
	again, _ := observation.Receipt()
	if len(again.Runs) == 0 || len(again.Runs[0].OutputTokenIDs) == 0 {
		t.Fatal("receipt accessor did not return an isolated copy")
	}
}

func TestCaptureStrixModelbenchC1RefusesNonC1Cell(t *testing.T) {
	manifest, _, executable := validC1AdapterManifest(t)
	nonC1 := manifest
	nonC1.Challenge.Concurrency = 4
	nonC1.Challenge.AcceptedOutputTPS = 24.13
	resealed, err := SealStrixComparisonCellManifest(nonC1)
	if err != nil {
		t.Fatal(err)
	}
	runner := func(context.Context, string, []string, []string) ([]byte, error) {
		t.Fatal("runner must not be invoked for a non-c1 cell")
		return nil, nil
	}
	_, err = captureStrixModelbenchC1(context.Background(), StrixModelbenchC1Request{
		Manifest: resealed, ManifestDigest: resealed.Digest,
		ExecutablePath: executable, ArtifactPath: filepath.Join(t.TempDir(), "m.gguf"),
		ExpectedExecutable: manifest.Candidate.ExecutableSHA256,
	}, runner)
	if err == nil || !strings.Contains(err.Error(), "not c=1") {
		t.Fatalf("non-c1 cell was not refused: %v", err)
	}
}

func TestCaptureStrixModelbenchC1RefusesMismatchedExecutable(t *testing.T) {
	manifest, digest, executable := validC1AdapterManifest(t)
	runner := func(context.Context, string, []string, []string) ([]byte, error) {
		t.Fatal("runner must not be invoked when the executable hash mismatches")
		return nil, nil
	}
	_, err := captureStrixModelbenchC1(context.Background(), StrixModelbenchC1Request{
		Manifest: manifest, ManifestDigest: digest,
		ExecutablePath: executable, ArtifactPath: filepath.Join(t.TempDir(), "m.gguf"),
		ExpectedExecutable: strings.Repeat("9", 64),
	}, runner)
	if err == nil || !strings.Contains(err.Error(), "executable SHA-256 mismatch") {
		t.Fatalf("mismatched executable was not refused: %v", err)
	}
}

func TestCaptureStrixModelbenchC1RefusesCallerShapedReport(t *testing.T) {
	manifest, digest, executable := validC1AdapterManifest(t)
	valid := c1AdapterReportJSON(t, c1AdapterReceipt(t, manifest))

	cases := []struct {
		name   string
		report []byte
		want   string
	}{
		{"unavailable", []byte(`{"schema":"x","canonical_physical_receipt":{"status":"UNAVAILABLE","credit_eligible":false}}`), "not AVAILABLE"},
		{"absent-receipt", []byte(`{"schema":"x","canonical_physical_receipt":{"status":"AVAILABLE","credit_eligible":true}}`), "absent"},
		{"unknown-field", []byte(`{"schema":"x","canonical_physical_receipt":{"status":"AVAILABLE","credit_eligible":true},"extra":1}`), "unknown field"},
		{"duplicate-field", append(valid[:len(valid)-1:len(valid)-1], []byte(`,"schema":"y"}`)...), "duplicate JSON field"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			runner := func(context.Context, string, []string, []string) ([]byte, error) { return tc.report, nil }
			_, err := captureStrixModelbenchC1(context.Background(), StrixModelbenchC1Request{
				Manifest: manifest, ManifestDigest: digest,
				ExecutablePath: executable, ArtifactPath: filepath.Join(t.TempDir(), "m.gguf"),
				ExpectedExecutable: manifest.Candidate.ExecutableSHA256,
			}, runner)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("report was not refused with %q: %v", tc.want, err)
			}
		})
	}

	t.Run("tampered-receipt-field", func(t *testing.T) {
		var decoded map[string]any
		if err := json.Unmarshal(valid, &decoded); err != nil {
			t.Fatal(err)
		}
		envelope := decoded["canonical_physical_receipt"].(map[string]any)
		receipt := envelope["receipt"].(map[string]any)
		receipt["model"].(map[string]any)["quantization"] = "Q8_0"
		tampered, err := json.Marshal(decoded)
		if err != nil {
			t.Fatal(err)
		}
		runner := func(context.Context, string, []string, []string) ([]byte, error) { return tampered, nil }
		_, err = captureStrixModelbenchC1(context.Background(), StrixModelbenchC1Request{
			Manifest: manifest, ManifestDigest: digest,
			ExecutablePath: executable, ArtifactPath: filepath.Join(t.TempDir(), "m.gguf"),
			ExpectedExecutable: manifest.Candidate.ExecutableSHA256,
		}, runner)
		if err == nil || !strings.Contains(err.Error(), "quantization") {
			t.Fatalf("quantization mutation was not refused: %v", err)
		}
	})
}

func TestStrixModelbenchC1ObservationZeroValueIsUnavailable(t *testing.T) {
	var zero StrixModelbenchC1Observation
	if zero.Valid() {
		t.Fatal("zero observation reported valid")
	}
	if _, ok := zero.ChallengeConcurrency(); ok {
		t.Fatal("zero observation reported a concurrency")
	}
	if _, ok := zero.Receipt(); ok {
		t.Fatal("zero observation reported a receipt")
	}
	if credit, ok := zero.PerformanceCredit(); credit || ok {
		t.Fatalf("zero observation credit = %v, ok=%v; want false/false", credit, ok)
	}
}

func validC1AdapterManifest(t *testing.T) (StrixComparisonCellManifest, string, string) {
	t.Helper()
	dir := t.TempDir()
	executable := filepath.Join(dir, "modelbench")
	if err := os.WriteFile(executable, []byte("#!/bin/sh\nexit 0\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	digest, err := hashRegularFile(executable)
	if err != nil {
		t.Fatal(err)
	}
	manifest := validStrixComparisonCellManifest(t)
	manifest.Candidate.ExecutableSHA256 = digest
	sealed, err := SealStrixComparisonCellManifest(manifest)
	if err != nil {
		t.Fatal(err)
	}
	return sealed, sealed.Digest, executable
}

func c1AdapterReceipt(t *testing.T, manifest StrixComparisonCellManifest) compute.Qwen38VulkanDecodeReceipt {
	t.Helper()
	packet, err := ImportPromptPacket(manifest.Workload.PromptPacketBytes)
	if err != nil {
		t.Fatal(err)
	}
	outTokens := make([]int32, manifest.Workload.AcceptedOutputTokens)
	for i := range outTokens {
		outTokens[i] = int32(1000 + i)
	}
	logprobs := make([]float64, len(outTokens))
	for i := range logprobs {
		logprobs[i] = -0.5 - float64(i)
	}
	clean := false
	zeroFallbacks := uint64(0)
	finiteLogits, cpuParity := true, true
	ignoreEOS, eosStopped := true, false
	complete := true
	run := compute.Qwen38VulkanDecodeRun{
		Repetition: 1, Kind: compute.Qwen38VulkanRunKindMeasured,
		ContextLimit:                manifest.Workload.ContextTokens,
		ContextTokens:               len(packet.PromptTokenIDs) + len(outTokens),
		GeneratedTokenLimit:         manifest.Workload.AcceptedOutputTokens,
		ActualGeneratedTokens:       len(outTokens),
		Sampler:                     "greedy",
		SeedPolicy:                  "not_applicable_greedy",
		IgnoreEOS:                   &ignoreEOS,
		EOSStopped:                  &eosStopped,
		OutputTokenIDs:              slices.Clone(outTokens),
		SelectedTokenLogprobs:       slices.Clone(logprobs),
		SessionSetupNanoseconds:     5_000,
		PrefillNanoseconds:          10_000,
		FirstSampleNanoseconds:      5_000,
		DecodeNanoseconds:           25_000,
		TeardownNanoseconds:         5_000,
		CandidateElapsedNanoseconds: 50_000,
		CPUVerificationNanoseconds:  10_000,
		Resources: &compute.Qwen38VulkanRunResources{
			PeakProcessMemoryBytes: 8 << 30,
			PeakDeviceMemoryBytes:  6 << 30,
			TransfersComplete:      &complete,
			Counters: compute.Qwen38VulkanDecodeCounters{
				ComputeDispatches:      40,
				Q4KMatmulDispatches:    30,
				OtherComputeDispatches: 10,
				DispatchSubmits:        8,
				H2D:                    compute.Qwen38VulkanTransferCounters{Count: 3, Bytes: 3072},
				D2H:                    compute.Qwen38VulkanTransferCounters{Count: 1, Bytes: 4},
				D2D:                    compute.Qwen38VulkanTransferCounters{Count: 2, Bytes: 2048},
				Q4KStageCalls:          4,
				Q4KStageBytes:          4096,
				TensorHome: compute.Qwen38VulkanTensorHomeCounters{
					Hits: 8, Admissions: 2, Bypasses: 1, ResidentBytes: 2048, CopiedBytes: 2048,
				},
			},
		},
	}
	promptIDs := make([]int32, len(packet.PromptTokenIDs))
	for i, id := range packet.PromptTokenIDs {
		promptIDs[i] = int32(id)
	}
	raw := compute.Qwen38VulkanRawDecodeResult{
		Source: compute.Qwen38VulkanSourceIdentity{
			GitCommit: strings.Repeat("a", 40), SourceArchiveSHA256: strings.Repeat("b", 64),
			BinarySHA256: strings.Repeat("c", 64), Dirty: &clean,
		},
		Model: compute.Qwen38VulkanModelIdentity{
			Name: manifest.Workload.Model, ArtifactPath: "/models/qwen3.8-27b-q4_k_m.gguf",
			ArtifactSHA256: manifest.Workload.ArtifactSHA256, TensorInventorySHA256: strings.Repeat("d", 64),
			TokenizerSHA256: manifest.Workload.TokenizerSHA256, TemplateSHA256: manifest.Workload.TemplateSHA256,
			Quantization: manifest.Workload.Quantization,
		},
		Device: compute.Qwen38VulkanDeviceIdentity{
			OS: "linux", Arch: "amd64", Kernel: "6.14", Name: "AMD Radeon 8060S Graphics|gfx1151",
			MesaVersion: "25.2", VulkanVersion: "1.4", Firmware: "amdgpu-test",
		},
		Engine: compute.Qwen38VulkanEngineIdentity{
			Name: "fak-native", Backend: compute.Qwen38VulkanDecodeBackend, Runtime: compute.Qwen38VulkanDecodeRuntime,
			ExecutedPath: "modelbench/raw-decode/vulkan", FallbackCount: &zeroFallbacks,
		},
		CaptureCommand:              "modelbench -raw-decode",
		PromptTokenIDs:              promptIDs,
		GeneratedTokenLimit:         manifest.Workload.AcceptedOutputTokens,
		OutputTokenIDs:              slices.Clone(outTokens),
		OutputText:                  "c1 adapter output",
		FiniteLogits:                &finiteLogits,
		CPUModelParity:              &cpuParity,
		Runs:                        []compute.Qwen38VulkanDecodeRun{run},
		CandidateElapsedNanoseconds: 50_000,
		CPUVerificationNanoseconds:  10_000,
		ResourceScope:               compute.Qwen38VulkanResourceScopeReportedRuns,
		ReportedRuns:                1,
	}
	receipt, err := compute.BuildQwen38VulkanDecodeReceiptV3(raw)
	if err != nil {
		t.Fatalf("build v3 receipt: %v", err)
	}
	return receipt
}

func c1AdapterReportJSON(t *testing.T, receipt compute.Qwen38VulkanDecodeReceipt) []byte {
	t.Helper()
	report := map[string]any{
		"schema": "fak.modelbench.report/v1",
		"canonical_physical_receipt": map[string]any{
			"status":          "AVAILABLE",
			"credit_eligible": true,
			"receipt":         receipt,
		},
	}
	raw, err := json.Marshal(report)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

func containsArg(argv []string, want string) bool {
	return slices.Contains(argv, want)
}

func pairArg(argv []string, key, value string) bool {
	for i := 0; i+1 < len(argv); i++ {
		if argv[i] == key && argv[i+1] == value {
			return true
		}
	}
	return false
}

func containsEntry(env []string, want string) bool {
	return slices.Contains(env, want)
}

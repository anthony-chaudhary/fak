package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func validNativeCapacityReceipt(t *testing.T) nativeCapacityReceipt {
	t.Helper()
	receipt := nativeCapacityReceipt{
		Schema: nativeCapacityReceiptSchema,
		Issue:  nativeCapacityIssue,
		Artifact: nativeCapacityArtifactIdentity{
			nativeCapacityFileIdentity: nativeCapacityFileIdentity{Bytes: nativeCapacityArtifactBytes, SHA256: nativeCapacityArtifactSHA256},
			Model:                      "qwen38:27b", ModelRevision: "unsloth/Qwen3.8-27B-GGUF@f1bfb127c64f7072bdd2cad55f258b9c8b2910fe",
		},
		Source: nativeCapacitySourceIdentity{Revision: strings.Repeat("a", 40)},
		Binary: nativeCapacityFileIdentity{Bytes: 123456, SHA256: strings.Repeat("b", 64)},
		Host: nativeCapacityHostIdentity{
			ScrubbedID: "m3pro-18gpu-36g", SOC: "Apple M3 Pro", GPUCores: 18,
			MemoryBytes: nativeCapacityHostBytes, OSKernel: "Darwin 25.6.0", Architecture: "darwin/arm64",
		},
		Environment: nativeCapacityEnvironment{
			Present: append([]string(nil), nativeCapacityPresentEnvironment...),
			Absent:  append([]string(nil), nativeCapacityAbsentEnvironment...),
		},
		CacheDisplacement: nativeCapacityCacheDisplacement{
			TotalBytes: 64_915_847_712,
			Sources: []nativeCapacityDisplacementSource{
				{ScrubbedID: "cache-model-a", Bytes: 20_000_000_000, SHA256: strings.Repeat("c", 64)},
				{ScrubbedID: "cache-model-b", Bytes: 20_000_000_000, SHA256: strings.Repeat("d", 64)},
				{ScrubbedID: "cache-model-c", Bytes: 24_915_847_712, SHA256: strings.Repeat("e", 64)},
			},
		},
		Readiness: nativeCapacityReadiness{Endpoint: "http://127.0.0.1:18971/v1/models", HTTPStatus: 200, ElapsedMillis: 42_000, DeadlineSeconds: 420},
		Execution: nativeCapacityExecution{Engine: "inkernel", Backend: "metal", ForwardPath: "metal/qwen35-hybrid-session-v1", IdentitySource: "startup+models+metrics"},
		Memory: nativeCapacityMemory{
			RSSSamplesBytes:    []uint64{18 << 30, 20 << 30},
			MaxSampledRSSBytes: 20 << 30, TimeMaximumResidentSetBytes: 21 << 30,
			OSFootprintSamplesBytes:    []uint64{25 << 30, 30 << 30},
			OSPeakMemoryFootprintBytes: 30 << 30, SwapUsedBeforeBytes: 3 << 30,
			SwapUsedPeakBytes: 10 << 30, SwapUsedAfterBytes: 9 << 30,
			SwapPeakDeltaBytes: 7 << 30, SwapDeltaBytes: 6 << 30,
		},
		Outcome: nativeCapacityOutcome{Admission: "refused", Reason: "positive-swap-delta", RequiredBytes: 43 << 30, RequiredMethod: "ceil-gib(host-memory-bytes+peak-swap-delta-bytes)"},
		Watcher: nativeCapacityWatcher{
			Port: 8090, OwnerCommandSHA256: strings.Repeat("f", 64), Signal: "TERM", MatchedTERMs: 2,
			WatcherStopped: true, HardDeadlineSeconds: 600,
			Restoration: nativeCapacityRestoration{
				Restored: true, OwnerCommandSHA256: strings.Repeat("f", 64), HealthHTTPStatus: 200,
				ModelsHTTPStatus: 200, ElapsedMillis: 10_000, DeadlineSeconds: 120,
			},
		},
	}
	var err error
	receipt.BindingSHA256, err = nativeCapacityBinding(receipt)
	if err != nil {
		t.Fatal(err)
	}
	return receipt
}

func TestNativeCapacityReceiptReadbackBindsCompleteEvidence(t *testing.T) {
	receipt := validNativeCapacityReceipt(t)
	if err := validateNativeCapacityReceipt(receipt); err != nil {
		t.Fatalf("valid receipt rejected: %v", err)
	}

	tests := []struct {
		name string
		edit func(*nativeCapacityReceipt)
		want string
	}{
		{name: "artifact", edit: func(r *nativeCapacityReceipt) { r.Artifact.SHA256 = strings.Repeat("0", 64) }, want: "artifact identity"},
		{name: "source", edit: func(r *nativeCapacityReceipt) { r.Source.Revision = "" }, want: "source/binary"},
		{name: "dirty source", edit: func(r *nativeCapacityReceipt) { r.Source.Dirty = true }, want: "source/binary"},
		{name: "binary", edit: func(r *nativeCapacityReceipt) { r.Binary.SHA256 = "" }, want: "source/binary"},
		{name: "host", edit: func(r *nativeCapacityReceipt) { r.Host.MemoryBytes = 64 << 30 }, want: "host"},
		{name: "free cpu present", edit: func(r *nativeCapacityReceipt) {
			r.Environment.Present = append(r.Environment.Present, "FAK_Q4K_FREE_CPU=1")
		}, want: "no-FAK_Q4K_FREE_CPU"},
		{name: "free cpu absence missing", edit: func(r *nativeCapacityReceipt) {
			r.Environment.Absent = removeString(r.Environment.Absent, "FAK_Q4K_FREE_CPU")
		}, want: "no-FAK_Q4K_FREE_CPU"},
		{name: "cache displacement", edit: func(r *nativeCapacityReceipt) { r.CacheDisplacement.TotalBytes = 36 << 30 }, want: "cache displacement"},
		{name: "readiness", edit: func(r *nativeCapacityReceipt) { r.Readiness.HTTPStatus = 503 }, want: "readiness"},
		{name: "engine", edit: func(r *nativeCapacityReceipt) { r.Execution.Engine = "llama.cpp" }, want: "fak-native"},
		{name: "fallback", edit: func(r *nativeCapacityReceipt) { r.Execution.FallbackCount = 1 }, want: "zero fallback"},
		{name: "rss", edit: func(r *nativeCapacityReceipt) { r.Memory.MaxSampledRSSBytes = 0 }, want: "RSS"},
		{name: "time l", edit: func(r *nativeCapacityReceipt) { r.Memory.TimeMaximumResidentSetBytes = 0 }, want: "/usr/bin/time -l"},
		{name: "footprint", edit: func(r *nativeCapacityReceipt) { r.Memory.OSPeakMemoryFootprintBytes = 0 }, want: "OS footprint"},
		{name: "swap", edit: func(r *nativeCapacityReceipt) { r.Memory.SwapPeakDeltaBytes++ }, want: "swap"},
		{name: "outcome", edit: func(r *nativeCapacityReceipt) { r.Outcome.Admission = "admitted" }, want: "fail closed"},
		{name: "required bound", edit: func(r *nativeCapacityReceipt) { r.Outcome.RequiredBytes++ }, want: "required bound"},
		{name: "term only", edit: func(r *nativeCapacityReceipt) { r.Watcher.Signal = "KILL" }, want: "TERM-only"},
		{name: "unmatched signal", edit: func(r *nativeCapacityReceipt) { r.Watcher.UnmatchedSignals = 1 }, want: "exact-owner"},
		{name: "watcher stopped", edit: func(r *nativeCapacityReceipt) { r.Watcher.WatcherStopped = false }, want: "watcher"},
		{name: "restoration", edit: func(r *nativeCapacityReceipt) { r.Watcher.Restoration.HealthHTTPStatus = 503 }, want: "restoration"},
		{name: "private path", edit: func(r *nativeCapacityReceipt) { r.Host.ScrubbedID = "/Users/private" }, want: "private marker"},
		{name: "binding", edit: func(r *nativeCapacityReceipt) { r.BindingSHA256 = strings.Repeat("0", 64) }, want: "binding digest"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := receipt
			got.Environment.Present = append([]string(nil), receipt.Environment.Present...)
			got.Environment.Absent = append([]string(nil), receipt.Environment.Absent...)
			got.CacheDisplacement.Sources = append([]nativeCapacityDisplacementSource(nil), receipt.CacheDisplacement.Sources...)
			got.Memory.RSSSamplesBytes = append([]uint64(nil), receipt.Memory.RSSSamplesBytes...)
			got.Memory.OSFootprintSamplesBytes = append([]uint64(nil), receipt.Memory.OSFootprintSamplesBytes...)
			test.edit(&got)
			if err := validateNativeCapacityReceipt(got); err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("err=%v, want %q", err, test.want)
			}
		})
	}
}

func TestNativeCapacityCapturePlanIsBoundedAndNoFREECPU(t *testing.T) {
	plan := nativeCapacityCapturePlan()
	if plan.Schema != nativeCapacityPlanSchema || plan.Issue != nativeCapacityIssue || plan.HardwareGate == "" {
		t.Fatalf("plan identity=%+v", plan)
	}
	if plan.WatcherPort != 8090 || plan.WatcherSignal != "TERM" || plan.WatcherHardDeadline <= plan.ReadinessDeadlineSeconds || plan.RestorationDeadline <= 0 {
		t.Fatalf("watch bounds=%+v", plan)
	}
	if plan.CacheDisplacementMinimum <= 36<<30 || !nativeCapacityContainsString(plan.Environment.Absent, "FAK_Q4K_FREE_CPU") || nativeCapacityContainsString(plan.Environment.Present, "FAK_Q4K_FREE_CPU=1") {
		t.Fatalf("control envelope=%+v", plan)
	}
	joined := strings.Join(plan.ServeCommand, " ")
	for _, want := range []string{"/usr/bin/time -l", "127.0.0.1:18971", "qwen38:27b", "--metal", "--context-budget-tokens 4096"} {
		if !strings.Contains(joined, want) {
			t.Errorf("serve command missing %q: %s", want, joined)
		}
	}
}

func TestNativeCapacityRejectsIssue8964FREECPUReceipt(t *testing.T) {
	path := filepath.Join("..", "..", "docs", "_witnesses", "issue-CHILD-qwen38-startup-bisect", "bisect.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var old struct {
		DocumentedRecipe struct {
			Environment []string `json:"environment"`
		} `json:"documented_recipe"`
	}
	if err := json.Unmarshal(data, &old); err != nil {
		t.Fatal(err)
	}
	if !nativeCapacityContainsString(old.DocumentedRecipe.Environment, "FAK_Q4K_FREE_CPU=1") {
		t.Fatalf("#8964 fixture unexpectedly lacks FREE_CPU: %v", old.DocumentedRecipe.Environment)
	}
	if err := validateNativeCapacityEnvironment(nativeCapacityEnvironment{Present: old.DocumentedRecipe.Environment, Absent: nativeCapacityAbsentEnvironment}); err == nil {
		t.Fatal("the refuted #8964 FREE_CPU receipt was accepted")
	}
}

func TestNativeCapacityCanonicalNoFREECPUWitness(t *testing.T) {
	path := filepath.Join("..", "..", "docs", "_witnesses", "issue-8971-streamed-q4k-capacity", "canonical-no-free-cpu.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	receipt, err := decodeNativeCapacityReceipt(data)
	if err != nil {
		t.Fatalf("canonical receipt rejected: %v", err)
	}
	if receipt.Outcome.Admission != "refused" || receipt.Outcome.RequiredBytes != streamedQ4KMetalCapacityBytes || receipt.Memory.SwapPeakDeltaBytes <= 0 {
		t.Fatalf("canonical refusal=%+v swap_peak_delta=%d", receipt.Outcome, receipt.Memory.SwapPeakDeltaBytes)
	}
}

func TestNativeCapacityReceiptCLIReadback(t *testing.T) {
	receipt := validNativeCapacityReceipt(t)
	data, err := json.MarshalIndent(receipt, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "receipt.json")
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	if code := runNativePerformance(&stdout, &stderr, []string{"--capacity-receipt", path}); code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, stderr.String())
	}
	var readback nativeCapacityReadback
	if err := json.Unmarshal(stdout.Bytes(), &readback); err != nil {
		t.Fatal(err)
	}
	if !readback.Valid || !readback.Restored8090 || readback.Admission != "refused" || readback.RequiredBytes == 0 || readback.SwapPeakDeltaBytes <= 0 || readback.ArtifactSHA256 != nativeCapacityArtifactSHA256 || readback.MaxSampledRSSBytes == 0 || readback.TimeMaximumResidentBytes == 0 || readback.OSPeakMemoryFootprintBytes == 0 {
		t.Fatalf("readback=%+v", readback)
	}
}

func removeString(values []string, remove string) []string {
	out := make([]string, 0, len(values))
	for _, value := range values {
		if value != remove {
			out = append(out, value)
		}
	}
	return out
}

func nativeCapacityContainsString(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}

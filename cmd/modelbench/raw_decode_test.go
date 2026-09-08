package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/mathx"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/rawdecode"
)

func syntheticTestConfig() model.Config {
	return model.Config{
		HiddenSize:       32,
		NumLayers:        2,
		NumHeads:         4,
		NumKVHeads:       2,
		HeadDim:          8,
		IntermediateSize: 64,
		VocabSize:        97,
		RMSNormEps:       1e-5,
		RopeTheta:        10000,
		EOSTokenID:       -1,
	}
}

func testRawDecodeFlags(steps, reps int) *benchFlags {
	f := testCompleteBenchFlags()
	*f.decodeSteps = steps
	*f.decodeReps = reps
	return f
}

func setRawDecodeTestFlags(rawDecode bool, promptIDs string, contextLimit int, ignoreEOS, verifyCPU bool) func() {
	prevRawDecode := *rawDecodeFlag
	prevPromptIDs := *rawPromptIDsFlag
	prevContext := *rawContextFlag
	prevIgnoreEOS := *rawIgnoreEOSFlag
	prevVerifyCPU := *rawVerifyCPUFlag

	*rawDecodeFlag = rawDecode
	*rawPromptIDsFlag = promptIDs
	*rawContextFlag = contextLimit
	*rawIgnoreEOSFlag = ignoreEOS
	*rawVerifyCPUFlag = verifyCPU

	return func() {
		*rawDecodeFlag = prevRawDecode
		*rawPromptIDsFlag = prevPromptIDs
		*rawContextFlag = prevContext
		*rawIgnoreEOSFlag = prevIgnoreEOS
		*rawVerifyCPUFlag = prevVerifyCPU
	}
}

func TestRawDecodeSynthetic(t *testing.T) {
	defer setRawDecodeTestFlags(true, "1,2,3", 256, false, false)()

	m := model.NewSynthetic(syntheticTestConfig())
	f := testRawDecodeFlags(5, 2)

	report, err := executeRawDecode(f, m, "synthetic-test", 12.5, 0, nil, nil)
	if err != nil {
		t.Fatalf("executeRawDecode failed: %v", err)
	}

	if report["raw_decode"] != true {
		t.Errorf("expected raw_decode: true, got %v", report["raw_decode"])
	}
	promptIDs, ok := report["prompt_ids"].([]int)
	if !ok || !reflect.DeepEqual(promptIDs, []int{1, 2, 3}) {
		t.Errorf("expected prompt_ids [1, 2, 3], got %v", report["prompt_ids"])
	}
	if report["actual_tokens_generated"] != 5 {
		t.Errorf("expected 5 actual_tokens_generated, got %v", report["actual_tokens_generated"])
	}
	if report["actual_step_calls"] != 4 {
		t.Errorf("expected 4 actual_step_calls, got %v", report["actual_step_calls"])
	}

	stepTokens, ok := report["step_tokens"].([]int)
	if !ok || len(stepTokens) != 4 {
		t.Fatalf("expected 4 step_tokens, got %v", report["step_tokens"])
	}

	prefillOutputID, ok := report["prefill_output_id"].(int)
	if !ok {
		t.Fatalf("expected int prefill_output_id, got %T", report["prefill_output_id"])
	}

	effectiveIDs, ok := report["effective_ids"].([]int)
	if !ok || len(effectiveIDs) != 8 {
		t.Fatalf("expected 8 effective_ids, got %v", report["effective_ids"])
	}
	wantEffective := append([]int{1, 2, 3, prefillOutputID}, stepTokens...)
	if !reflect.DeepEqual(effectiveIDs, wantEffective) {
		t.Errorf("effective_ids mismatch: got %v, want %v", effectiveIDs, wantEffective)
	}

	if report["finite_logits"] != true {
		t.Errorf("expected finite_logits: true")
	}
	if report["eos_stopped"] != false {
		t.Errorf("expected eos_stopped: false")
	}

	timings, ok := report["timings"].(map[string]any)
	if !ok {
		t.Fatalf("missing timings in report")
	}
	for _, key := range []string{"load_ms", "session_setup_ms", "prefill_ms", "first_sample_ms", "decode_ms", "teardown_ms", "total_ms"} {
		if _, present := timings[key]; !present {
			t.Errorf("missing timing key %q", key)
		}
	}

	hostStages, ok := report["host_stages"].([]map[string]any)
	if !ok || len(hostStages) != 5 {
		t.Errorf("expected 5 host stages, got %v", report["host_stages"])
	}

	steps, ok := report["steps"].([]rawStepInfo)
	if !ok || len(steps) != 5 {
		t.Errorf("expected 5 steps recorded, got %v", report["steps"])
	}
	if _, ok := report["margin_summary"].(map[string]any); !ok {
		t.Errorf("missing margin_summary")
	}

	attempt, ok := report["canonical_physical_receipt"].(rawDecodePhysicalReceiptAttempt)
	if !ok {
		t.Fatalf("missing canonical physical receipt attempt: %T", report["canonical_physical_receipt"])
	}
	if attempt.Status != "UNAVAILABLE" || attempt.CreditEligible || attempt.Receipt != nil {
		t.Fatalf("synthetic raw decode became creditable: %+v", attempt)
	}
	if attempt.Observed.Model.Name != "" || attempt.Observed.Model.Quantization != "" {
		t.Fatalf("caller-supplied model identity escaped into physical evidence: %+v", attempt.Observed.Model)
	}
	if !reflect.DeepEqual(attempt.Observed.PromptTokenIDs, []int32{1, 2, 3}) || len(attempt.Observed.OutputTokenIDs) != 5 {
		t.Fatalf("observed token identity was not preserved: prompt=%v output=%v", attempt.Observed.PromptTokenIDs, attempt.Observed.OutputTokenIDs)
	}
	if attempt.Observed.Engine.Name != "" || attempt.Observed.Engine.Backend != "" || attempt.Observed.Engine.FallbackCount != nil {
		t.Fatalf("synthetic execution was relabeled as a physical engine: %+v", attempt.Observed.Engine)
	}
	if attempt.Observed.Source.GitCommit != "" || attempt.Observed.Source.SourceArchiveSHA256 != "" || attempt.Observed.Source.Dirty != nil || attempt.Observed.Source.DiffSHA256 != "" || attempt.Observed.Model.ArtifactSHA256 != "" || attempt.Observed.Device.Name != "" || attempt.Observed.OutputText != "" || attempt.Observed.PeakProcessMemoryBytes != nil || attempt.Observed.PeakDeviceMemoryBytes != nil || attempt.Observed.Counters != nil {
		t.Fatalf("unobserved physical identity or telemetry was invented: %+v", attempt.Observed)
	}
	if attempt.Observed.Source.BinarySHA256 != "" {
		t.Fatalf("incomplete executable tuple leaked binary-only identity: %+v", attempt.Observed.Source)
	}
	if attempt.Observed.FiniteLogits == nil || !*attempt.Observed.FiniteLogits || attempt.Observed.CPUModelParity != nil {
		t.Fatalf("quality evidence presence mismatch: finite=%v parity=%v", attempt.Observed.FiniteLogits, attempt.Observed.CPUModelParity)
	}
	runs, ok := report["runs"].([]map[string]any)
	if !ok || len(runs) != 2 {
		t.Fatalf("expected two raw runs, got %T %v", report["runs"], report["runs"])
	}
	rep0Timing := runs[0]["timings"].(map[string]any)
	wantElapsedNS := uint64(math.Round(rep0Timing["total_ms"].(float64) * 1e6))
	if attempt.Observed.CandidateElapsedNanoseconds != wantElapsedNS || attempt.Observed.Runs[0].CandidateElapsedNanoseconds != wantElapsedNS {
		t.Fatalf("physical attempt candidate_elapsed_ns=%d run0=%d, want repetition-zero elapsed_ns=%d", attempt.Observed.CandidateElapsedNanoseconds, attempt.Observed.Runs[0].CandidateElapsedNanoseconds, wantElapsedNS)
	}
}

func TestRawDecodePhysicalReceiptLeavesIncompleteExecutableTupleUnavailable(t *testing.T) {
	execution := rawdecode.Execution{
		PromptTokenIDs: []int{1}, ContextLimit: 8, GeneratedLimit: 1, FiniteLogits: true,
	}
	attempt := rawDecodePhysicalReceipt(execution, []rawRepOutput{{generatedTokens: []int{2}, prefillDur: time.Nanosecond}})
	if attempt.Status != "UNAVAILABLE" || attempt.CreditEligible || attempt.Receipt != nil {
		t.Fatalf("binary-only observation became creditable: %+v", attempt)
	}
	if attempt.Observed.Source.GitCommit != "" || attempt.Observed.Source.BinarySHA256 != "" || attempt.Observed.Source.Dirty != nil || attempt.Observed.Source.SourceArchiveSHA256 != "" || attempt.Observed.Source.DiffSHA256 != "" {
		t.Fatalf("incomplete executable tuple escaped into source identity: %+v", attempt.Observed.Source)
	}
}

func TestRawDecodePhysicalReceiptCarriesOnlyCompleteGGUFObservation(t *testing.T) {
	execution := rawdecode.Execution{
		ArtifactPath:          "fixtures/qwen3.8-q4_k_m.gguf",
		ArtifactSHA256:        strings.Repeat("a", 64),
		TensorInventorySHA256: "sha256:" + strings.Repeat("b", 64),
		TokenizerSHA256:       strings.Repeat("c", 64),
		TemplateSHA256:        strings.Repeat("d", 64),
		Quantization:          "Q4_K_M",
		ModelName:             "qwen3.8-q4_k_m.gguf [gguf-q4k]",
		PromptTokenIDs:        []int{1},
		ContextLimit:          8,
		GeneratedLimit:        1,
		FiniteLogits:          true,
	}
	attempt := rawDecodePhysicalReceipt(execution, []rawRepOutput{{generatedTokens: []int{2}, prefillDur: time.Nanosecond}})
	if attempt.Status != "UNAVAILABLE" || attempt.CreditEligible || attempt.Receipt != nil {
		t.Fatalf("software-only model provenance became creditable: %+v", attempt)
	}
	modelIdentity := attempt.Observed.Model
	if modelIdentity.Name != execution.ModelName || modelIdentity.ArtifactPath != execution.ArtifactPath ||
		modelIdentity.ArtifactSHA256 != execution.ArtifactSHA256 || modelIdentity.TensorInventorySHA256 != execution.TensorInventorySHA256 ||
		modelIdentity.TokenizerSHA256 != execution.TokenizerSHA256 || modelIdentity.TemplateSHA256 != execution.TemplateSHA256 ||
		modelIdentity.Quantization != execution.Quantization {
		t.Fatalf("canonical model identity mismatch: %+v", modelIdentity)
	}
	if attempt.Observed.Source != (compute.Qwen38VulkanSourceIdentity{}) {
		t.Fatalf("unobserved source identity was invented: model=%+v source=%+v", modelIdentity, attempt.Observed.Source)
	}

	for _, missing := range []struct {
		name  string
		clear func(*rawdecode.Execution)
	}{
		{name: "tensor inventory", clear: func(e *rawdecode.Execution) { e.TensorInventorySHA256 = "" }},
		{name: "tokenizer", clear: func(e *rawdecode.Execution) { e.TokenizerSHA256 = "" }},
		{name: "template", clear: func(e *rawdecode.Execution) { e.TemplateSHA256 = "" }},
	} {
		t.Run("missing "+missing.name, func(t *testing.T) {
			incomplete := execution
			missing.clear(&incomplete)
			incompleteAttempt := rawDecodePhysicalReceipt(incomplete, []rawRepOutput{{generatedTokens: []int{2}, prefillDur: time.Nanosecond}})
			if incompleteAttempt.Status != "UNAVAILABLE" || incompleteAttempt.Observed.Model != (compute.Qwen38VulkanModelIdentity{}) || incompleteAttempt.CreditEligible || incompleteAttempt.Receipt != nil {
				t.Fatalf("partial GGUF provenance escaped as canonical model identity: %+v", incompleteAttempt)
			}
		})
	}
}

type vulkanNamedRawDecodeTestBackend struct{ compute.Backend }

func (vulkanNamedRawDecodeTestBackend) Name() string { return compute.Qwen38VulkanDecodeBackend }

func TestRawDecodeVulkanNameDoesNotClaimPhysicalEngineIdentity(t *testing.T) {
	execution := rawdecode.Execution{
		Backend:        rawdecode.BackendObservation{Selected: compute.Qwen38VulkanDecodeBackend},
		PromptTokenIDs: []int{1}, ContextLimit: 8, GeneratedLimit: 1, FiniteLogits: true,
	}
	attempt := rawDecodePhysicalReceipt(execution, []rawRepOutput{{generatedTokens: []int{2}, prefillDur: time.Nanosecond}})
	if attempt.Status != "UNAVAILABLE" || attempt.CreditEligible || attempt.Receipt != nil {
		t.Fatalf("named test backend became creditable: %+v", attempt)
	}
	if attempt.Observed.Engine.Backend != "" || attempt.Observed.Engine.ExecutedPath != "" || attempt.Observed.Engine.Name != "" || attempt.Observed.Engine.Runtime != "" || attempt.Observed.Engine.FallbackCount != nil {
		t.Fatalf("backend registry name was relabeled as physical execution identity: %+v", attempt.Observed.Engine)
	}
}

func TestRawDecodePhysicalReceiptCarriesBackendObservationWithoutPromoting(t *testing.T) {
	backendExecution := compute.BackendExecutionObservation{
		Identity: compute.BackendRuntimeIdentity{Backend: "vulkan", Device: "fixture-device", Driver: "fixture-driver", Runtime: "Vulkan 1.3"},
		Counters: compute.BackendCounterSnapshot{
			ComputeDispatches: 9, Q4KMatmulDispatches: 7, OtherDispatches: 2, DispatchSubmits: 3,
			H2DBytes: 64, D2HBytes: 32, D2DCopies: 1, Q4KStageCalls: 2, Q4KStageBytes: 128,
			TensorHomeHits: 4, TensorHomeAdmissions: 1,
		},
		TensorHomeEntries: 2, TensorHomeResidentBytes: 256,
		DeviceMemoryObserved: true, DeviceMemoryTotalBytes: 64 << 30, DeviceMemoryFreeBytes: 48 << 30,
	}
	execution := rawdecode.Execution{
		ArtifactSHA256: strings.Repeat("a", 64), Engine: "fak-in-kernel via compute HAL backend \"vulkan\"",
		Backend: rawdecode.BackendObservation{Selected: "vulkan"}, PromptTokenIDs: []int{1},
		ContextLimit: 8, GeneratedLimit: 1, FiniteLogits: true,
		Runs: []rawdecode.Run{{BackendExecution: &backendExecution}},
	}
	attempt := rawDecodePhysicalReceipt(execution, []rawRepOutput{{generatedTokens: []int{2}, prefillDur: time.Nanosecond}})
	if attempt.Status != "UNAVAILABLE" || attempt.CreditEligible || attempt.Receipt != nil {
		t.Fatalf("partial backend observation became creditable: %+v", attempt)
	}
	if len(attempt.BackendExecutions) != 1 || attempt.BackendExecutions[0].Counters.ComputeDispatches != 9 || attempt.BackendExecutions[0].Counters.H2DBytes != 64 {
		t.Fatalf("backend execution delta was not preserved: %+v", attempt.BackendExecutions)
	}
	if attempt.Observed.Device.Name != "fixture-device" || attempt.Observed.Device.VulkanVersion != "Vulkan 1.3" || attempt.Observed.Engine.Backend != "vulkan" || attempt.Observed.Engine.Runtime != compute.Qwen38VulkanDecodeRuntime || attempt.Observed.Engine.FallbackCount == nil || *attempt.Observed.Engine.FallbackCount != 0 {
		t.Fatalf("backend-owned canonical identity mapping mismatch: device=%+v engine=%+v", attempt.Observed.Device, attempt.Observed.Engine)
	}
	if attempt.Observed.Device.MesaVersion != "" || attempt.Observed.PeakProcessMemoryBytes != nil || attempt.Observed.PeakDeviceMemoryBytes != nil || attempt.Observed.Counters != nil {
		t.Fatalf("unavailable driver, peak memory, or incomplete counters were synthesized: %+v", attempt.Observed)
	}

	for name, mutate := range map[string]func(*rawdecode.Execution){
		"unsupported":       func(candidate *rawdecode.Execution) { candidate.Runs[0].BackendExecution = nil },
		"identity mismatch": func(candidate *rawdecode.Execution) { candidate.Runs[0].BackendExecution.Identity.Backend = "cpu" },
		"incomplete memory": func(candidate *rawdecode.Execution) { candidate.Runs[0].BackendExecution.DeviceMemoryObserved = false },
	} {
		t.Run(name, func(t *testing.T) {
			candidate := execution
			candidate.Runs = append([]rawdecode.Run(nil), execution.Runs...)
			observationCopy := backendExecution
			candidate.Runs[0].BackendExecution = &observationCopy
			mutate(&candidate)
			got := rawDecodePhysicalReceipt(candidate, []rawRepOutput{{generatedTokens: []int{2}, prefillDur: time.Nanosecond}})
			if got.Status != "UNAVAILABLE" || got.CreditEligible || got.Receipt != nil || len(got.BackendExecutions) != 0 || got.Observed.Engine.FallbackCount != nil {
				t.Fatalf("invalid backend observation was exposed or promoted: %+v", got)
			}
		})
	}
}

func TestRawDecodePhysicalReceiptSeparatesNativeRuntimeFromVulkanAPI(t *testing.T) {
	backendExecution := compute.BackendExecutionObservation{
		Identity: compute.BackendRuntimeIdentity{
			Backend: "vulkan", Device: "fixture-device", Driver: "fixture-driver", Runtime: "vulkan-1.3.0",
		},
		Counters: compute.BackendCounterSnapshot{
			ComputeDispatches: 1, Q4KMatmulDispatches: 1, DispatchSubmits: 1,
		},
		DeviceMemoryObserved: true, DeviceMemoryTotalBytes: 64 << 30, DeviceMemoryFreeBytes: 48 << 30,
	}
	execution := rawdecode.Execution{
		ArtifactSHA256: strings.Repeat("a", 64), Engine: "fak-in-kernel via compute HAL backend \"vulkan\"",
		Backend: rawdecode.BackendObservation{Selected: "vulkan"}, PromptTokenIDs: []int{1},
		ContextLimit: 8, GeneratedLimit: 1, FiniteLogits: true,
		Runs: []rawdecode.Run{{BackendExecution: &backendExecution}},
	}

	attempt := rawDecodePhysicalReceipt(execution, []rawRepOutput{{generatedTokens: []int{2}, prefillDur: time.Nanosecond}})
	if attempt.Status != "UNAVAILABLE" || attempt.CreditEligible || attempt.Receipt != nil {
		t.Fatalf("incomplete physical identity became creditable: %+v", attempt)
	}
	if attempt.Observed.Device.VulkanVersion != backendExecution.Identity.Runtime {
		t.Fatalf("device Vulkan version=%q, want observed API identity %q", attempt.Observed.Device.VulkanVersion, backendExecution.Identity.Runtime)
	}
	if attempt.Observed.Engine.Runtime != compute.Qwen38VulkanDecodeRuntime {
		t.Fatalf("engine runtime=%q, want canonical %q", attempt.Observed.Engine.Runtime, compute.Qwen38VulkanDecodeRuntime)
	}
	if attempt.Observed.Engine.Runtime == backendExecution.Identity.Runtime {
		t.Fatalf("Vulkan API identity escaped into engine runtime: %+v", attempt.Observed.Engine)
	}
}

func TestRawDecodeExecutorProductionAdapterCallsSeamOnceAndFailsReceiptClosed(t *testing.T) {
	defer setRawDecodeTestFlags(true, "1", 8, false, false)()
	previousDigest := *rawArtifactSHA256Flag
	*rawArtifactSHA256Flag = strings.Repeat("a", 64)
	defer func() { *rawArtifactSHA256Flag = previousDigest }()

	f := testRawDecodeFlags(1, 1)
	*f.gguf = "selected.gguf"
	*f.name = "caller-model-label"
	*f.out = filepath.Join(t.TempDir(), "raw.json")
	calls := 0
	execute := func(_ context.Context, req rawdecode.Request) (rawdecode.Execution, error) {
		calls++
		if req.ArtifactPath != "selected.gguf" || req.ExpectedArtifactSHA256 != strings.Repeat("a", 64) {
			t.Fatalf("adapter request mismatch: %+v", req)
		}
		return rawdecode.Execution{
			ArtifactSHA256: strings.Repeat("a", 64), ModelName: "observed-model",
			Backend:        rawdecode.BackendObservation{Selected: "legacy"},
			PromptTokenIDs: []int{1}, ContextLimit: 8, GeneratedLimit: 1,
			FiniteLogits: true,
			Runs:         []rawdecode.Run{{GeneratedTokens: []int{2}, PrefillOutputID: 2, PrefillDuration: time.Nanosecond}},
		}, nil
	}
	if err := runRawDecodeArtifactWith(f, nil, execute); err != nil {
		t.Fatalf("runRawDecodeArtifactWith: %v", err)
	}
	if calls != 1 {
		t.Fatalf("executor seam calls=%d, want 1", calls)
	}
	b, err := os.ReadFile(*f.out)
	if err != nil {
		t.Fatal(err)
	}
	var report map[string]any
	if err := json.Unmarshal(b, &report); err != nil {
		t.Fatal(err)
	}
	attempt := report["canonical_physical_receipt"].(map[string]any)
	if attempt["status"] != "UNAVAILABLE" || attempt["credit_eligible"] != false {
		t.Fatalf("incomplete observation became physical receipt: %v", attempt)
	}
	observed, ok := attempt["observed"].(map[string]any)
	if !ok {
		t.Fatalf("missing observed physical result: %v", attempt)
	}
	source, ok := observed["source"].(map[string]any)
	if !ok || source["git_commit"] != "" || source["binary_sha256"] != "" || source["dirty"] != nil {
		t.Fatalf("production report emitted partial executable identity: %v", source)
	}
}

func TestRawDecodeGreedyFeedbackDiffersFromSyntheticSchedule(t *testing.T) {
	defer setRawDecodeTestFlags(true, "10,20,30", 256, false, false)()

	cfg := syntheticTestConfig()
	m := model.NewSynthetic(cfg)
	f := testRawDecodeFlags(5, 1)

	report, err := executeRawDecode(f, m, "synthetic-feedback", 5.0, 0, nil, nil)
	if err != nil {
		t.Fatalf("executeRawDecode failed: %v", err)
	}

	prefillOutputID := report["prefill_output_id"].(int)
	stepTokens := report["step_tokens"].([]int)
	greedyTokens := append([]int{prefillOutputID}, stepTokens...)

	// Synthetic schedule as defined in stepDecode / benchmark harness
	id := 30
	syntheticTokens := make([]int, 5)
	for i := 0; i < 5; i++ {
		id = (id*48271 + 1) % cfg.VocabSize
		syntheticTokens[i] = id
	}

	if reflect.DeepEqual(greedyTokens, syntheticTokens) {
		t.Fatalf("greedy feedback unexpectedly matched synthetic LCG schedule %v", greedyTokens)
	}
}

func TestRawDecodeFirstTokenSampledFromPrefill(t *testing.T) {
	defer setRawDecodeTestFlags(true, "4,5,6", 256, false, false)()

	m := model.NewSynthetic(syntheticTestConfig())

	// Independent reference: prefill once to compute expected first token
	sRef := m.NewSession()
	refLogits := sRef.Prefill([]int{4, 5, 6})
	wantFirstToken := mathx.ArgmaxF32(refLogits)
	sRef.Close()

	// N = 1 requested output
	f1 := testRawDecodeFlags(1, 1)
	report1, err := executeRawDecode(f1, m, "first-token-n1", 1.0, 0, nil, nil)
	if err != nil {
		t.Fatalf("executeRawDecode N=1 failed: %v", err)
	}
	if report1["actual_tokens_generated"] != 1 {
		t.Errorf("N=1: expected 1 actual_tokens_generated, got %v", report1["actual_tokens_generated"])
	}
	if report1["actual_step_calls"] != 0 {
		t.Errorf("N=1: expected 0 actual_step_calls, got %v", report1["actual_step_calls"])
	}
	if report1["prefill_output_id"] != wantFirstToken {
		t.Errorf("N=1: prefill_output_id=%v, want %v", report1["prefill_output_id"], wantFirstToken)
	}
	if len(report1["step_tokens"].([]int)) != 0 {
		t.Errorf("N=1: expected empty step_tokens, got %v", report1["step_tokens"])
	}

	// N = 4 requested outputs (requires at most 3 step calls)
	f4 := testRawDecodeFlags(4, 1)
	report4, err := executeRawDecode(f4, m, "first-token-n4", 1.0, 0, nil, nil)
	if err != nil {
		t.Fatalf("executeRawDecode N=4 failed: %v", err)
	}
	if report4["actual_tokens_generated"] != 4 {
		t.Errorf("N=4: expected 4 actual_tokens_generated, got %v", report4["actual_tokens_generated"])
	}
	if report4["actual_step_calls"] != 3 {
		t.Errorf("N=4: expected 3 actual_step_calls, got %v", report4["actual_step_calls"])
	}
	if report4["prefill_output_id"] != wantFirstToken {
		t.Errorf("N=4: prefill_output_id=%v, want %v", report4["prefill_output_id"], wantFirstToken)
	}
}

func TestRawDecodeEOSHandlingAndIgnoreEOS(t *testing.T) {
	promptIDs := "7,8,9"
	cfg := syntheticTestConfig()
	m := model.NewSynthetic(cfg)

	// Step 1: probe natural continuation to discover generated token
	probeCleanup := setRawDecodeTestFlags(true, promptIDs, 256, true, false)
	fProbe := testRawDecodeFlags(5, 1)
	probeReport, err := executeRawDecode(fProbe, m, "eos-probe", 1.0, 0, nil, nil)
	probeCleanup()
	if err != nil {
		t.Fatalf("probe execution failed: %v", err)
	}

	firstTok := probeReport["prefill_output_id"].(int)

	// Case 1: scalar EOSTokenID matches generated token
	m.Cfg.EOSTokenID = firstTok

	// 1a: raw-ignore-eos = false -> stops immediately on EOS
	cleanup1a := setRawDecodeTestFlags(true, promptIDs, 256, false, false)
	f1a := testRawDecodeFlags(5, 1)
	rep1a, err := executeRawDecode(f1a, m, "eos-prefill-stop", 1.0, 0, nil, nil)
	cleanup1a()
	if err != nil {
		t.Fatalf("Case 1a failed: %v", err)
	}
	if rep1a["eos_stopped"] != true {
		t.Errorf("Case 1a: expected eos_stopped: true")
	}
	if rep1a["actual_tokens_generated"] != 1 {
		t.Errorf("Case 1a: expected 1 actual_tokens_generated, got %v", rep1a["actual_tokens_generated"])
	}
	if rep1a["actual_step_calls"] != 0 {
		t.Errorf("Case 1a: expected 0 actual_step_calls, got %v", rep1a["actual_step_calls"])
	}

	// 1b: raw-ignore-eos = true -> continues past EOS for all requested steps
	cleanup1b := setRawDecodeTestFlags(true, promptIDs, 256, true, false)
	f1b := testRawDecodeFlags(5, 1)
	rep1b, err := executeRawDecode(f1b, m, "eos-prefill-ignore", 1.0, 0, nil, nil)
	cleanup1b()
	if err != nil {
		t.Fatalf("Case 1b failed: %v", err)
	}
	if rep1b["eos_stopped"] != false {
		t.Errorf("Case 1b: expected eos_stopped: false")
	}
	if rep1b["actual_tokens_generated"] != 5 {
		t.Errorf("Case 1b: expected 5 actual_tokens_generated, got %v", rep1b["actual_tokens_generated"])
	}
	if rep1b["actual_step_calls"] != 4 {
		t.Errorf("Case 1b: expected 4 actual_step_calls, got %v", rep1b["actual_step_calls"])
	}

	// Case 2: no EOS set (EOSTokenID = -1)
	m.Cfg.EOSTokenID = -1
	m.Cfg.EOSTokenIDs = nil

	cleanup2 := setRawDecodeTestFlags(true, promptIDs, 256, false, false)
	f2 := testRawDecodeFlags(5, 1)
	rep2, err := executeRawDecode(f2, m, "no-eos", 1.0, 0, nil, nil)
	cleanup2()
	if err != nil {
		t.Fatalf("Case 2 failed: %v", err)
	}
	if rep2["eos_stopped"] != false {
		t.Errorf("Case 2: expected eos_stopped: false")
	}
	if rep2["actual_tokens_generated"] != 5 {
		t.Errorf("Case 2: expected 5 actual_tokens_generated, got %v", rep2["actual_tokens_generated"])
	}
	if rep2["actual_step_calls"] != 4 {
		t.Errorf("Case 2: expected 4 actual_step_calls, got %v", rep2["actual_step_calls"])
	}

	// Case 3: EOSTokenIDs list contains the token
	m.Cfg.EOSTokenID = -1
	m.Cfg.EOSTokenIDs = []int{firstTok}

	// 3a: raw-ignore-eos = false -> stops immediately on list EOS
	cleanup3a := setRawDecodeTestFlags(true, promptIDs, 256, false, false)
	f3a := testRawDecodeFlags(5, 1)
	rep3a, err := executeRawDecode(f3a, m, "eos-list-stop", 1.0, 0, nil, nil)
	cleanup3a()
	if err != nil {
		t.Fatalf("Case 3a failed: %v", err)
	}
	if rep3a["eos_stopped"] != true {
		t.Errorf("Case 3a: expected eos_stopped: true")
	}
	if rep3a["actual_tokens_generated"] != 1 {
		t.Errorf("Case 3a: expected 1 actual_tokens_generated, got %v", rep3a["actual_tokens_generated"])
	}

	// 3b: raw-ignore-eos = true -> continues past list EOS
	cleanup3b := setRawDecodeTestFlags(true, promptIDs, 256, true, false)
	f3b := testRawDecodeFlags(5, 1)
	rep3b, err := executeRawDecode(f3b, m, "eos-list-ignore", 1.0, 0, nil, nil)
	cleanup3b()
	if err != nil {
		t.Fatalf("Case 3b failed: %v", err)
	}
	if rep3b["eos_stopped"] != false {
		t.Errorf("Case 3b: expected eos_stopped: false")
	}
	if rep3b["actual_tokens_generated"] != 5 {
		t.Errorf("Case 3b: expected 5 actual_tokens_generated, got %v", rep3b["actual_tokens_generated"])
	}
	if rep3b["actual_step_calls"] != 4 {
		t.Errorf("Case 3b: expected 4 actual_step_calls, got %v", rep3b["actual_step_calls"])
	}
}

func TestRawDecodeContextLimitEnforcement(t *testing.T) {
	// Prompt length = 3 ("1,2,3"), requested decode steps = 5 -> required = 8
	f := testRawDecodeFlags(5, 1)

	// Context too small
	cleanup1 := setRawDecodeTestFlags(true, "1,2,3", 7, false, false)
	err1 := validateRawDecodeFlags(f)
	cleanup1()
	if err1 == nil {
		t.Errorf("expected error when rawContext (7) < prompt (3) + steps (5) = 8")
	}

	// Context non-positive
	cleanup2 := setRawDecodeTestFlags(true, "1,2,3", 0, false, false)
	err2 := validateRawDecodeFlags(f)
	cleanup2()
	if err2 == nil {
		t.Errorf("expected error when rawContext <= 0")
	}

	// Context exactly equal to minimum required
	cleanup3 := setRawDecodeTestFlags(true, "1,2,3", 8, false, false)
	err3 := validateRawDecodeFlags(f)
	cleanup3()
	if err3 != nil {
		t.Errorf("unexpected error when rawContext == minRequired: %v", err3)
	}
}

func TestRawDecodeVerifyCPUAgreement(t *testing.T) {
	defer setRawDecodeTestFlags(true, "2,4,6", 256, false, true)()

	m := model.NewSynthetic(syntheticTestConfig())
	f := testRawDecodeFlags(4, 1)

	report, err := executeRawDecode(f, m, "verify-cpu-test", 1.0, 0, nil, nil)
	if err != nil {
		t.Fatalf("executeRawDecode with verify-cpu failed: %v", err)
	}

	vRaw, ok := report["verify_cpu"]
	if !ok {
		t.Fatalf("report missing verify_cpu section")
	}
	v, ok := vRaw.(*cpuVerifyResult)
	if !ok {
		t.Fatalf("expected *cpuVerifyResult, got %T", vRaw)
	}

	if !v.Passed {
		t.Errorf("expected verify_cpu.Passed: true")
	}
	if !v.AllAgree {
		t.Errorf("expected verify_cpu.AllAgree: true")
	}
	if v.MinCosine < 0.9999 {
		t.Errorf("expected MinCosine near 1.0, got %f", v.MinCosine)
	}
	if v.MaxDelta > 1e-4 {
		t.Errorf("expected MaxDelta near 0, got %f", v.MaxDelta)
	}
	if !v.Prefill.Agree {
		t.Errorf("expected prefill verification to agree")
	}
	for i, step := range v.Steps {
		if !step.Agree {
			t.Errorf("step %d did not agree", i+1)
		}
	}
	attempt := report["canonical_physical_receipt"].(rawDecodePhysicalReceiptAttempt)
	if attempt.Observed.CPUModelParity == nil || !*attempt.Observed.CPUModelParity {
		t.Fatalf("passing CPU replay was not preserved as observed parity: %+v", attempt.Observed)
	}
	run := attempt.Observed.Runs[0]
	wantCandidate := run.SessionSetupNanoseconds + run.PrefillNanoseconds + run.FirstSampleNanoseconds + run.DecodeNanoseconds + run.TeardownNanoseconds
	if run.CandidateElapsedNanoseconds != wantCandidate || attempt.Observed.CandidateElapsedNanoseconds != wantCandidate {
		t.Fatalf("candidate timing includes non-candidate work: run=%+v top_level=%d", run, attempt.Observed.CandidateElapsedNanoseconds)
	}
	if attempt.Observed.CPUVerificationNanoseconds != run.CPUVerificationNanoseconds {
		t.Fatalf("CPU verification timing was not preserved separately: run=%+v top_level=%d", run, attempt.Observed.CPUVerificationNanoseconds)
	}
}

func TestRawDecodeFlagValidationConflictingModes(t *testing.T) {
	tests := []struct {
		name      string
		modify    func(f *benchFlags)
		promptIDs string
		steps     int
		reps      int
		wantErr   string
	}{
		{
			name:      "conflicts with loadOnly",
			modify:    func(f *benchFlags) { *f.loadOnly = true },
			promptIDs: "1,2",
			steps:     4,
			reps:      1,
			wantErr:   "-load-only",
		},
		{
			name:      "conflicts with verify",
			modify:    func(f *benchFlags) { *f.verify = true },
			promptIDs: "1,2",
			steps:     4,
			reps:      1,
			wantErr:   "-verify",
		},
		{
			name:      "conflicts with smoke",
			modify:    func(f *benchFlags) { *f.smoke = true },
			promptIDs: "1,2",
			steps:     4,
			reps:      1,
			wantErr:   "-smoke",
		},
		{
			name:      "conflicts with preflight",
			modify:    func(f *benchFlags) { *f.preflight = true },
			promptIDs: "1,2",
			steps:     4,
			reps:      1,
			wantErr:   "-preflight",
		},
		{
			name:      "conflicts with checkpoint",
			modify:    func(f *benchFlags) { *f.checkpoint = "ckpt.json" },
			promptIDs: "1,2",
			steps:     4,
			reps:      1,
			wantErr:   "-checkpoint",
		},
		{
			name:      "conflicts with resume",
			modify:    func(f *benchFlags) { *f.resume = "resume.json" },
			promptIDs: "1,2",
			steps:     4,
			reps:      1,
			wantErr:   "-resume",
		},
		{
			name:      "conflicts with workload",
			modify:    func(f *benchFlags) { *f.workloadPath = "workload.json" },
			promptIDs: "1,2",
			steps:     4,
			reps:      1,
			wantErr:   "-workload",
		},
		{
			name:      "conflicts with phaseProfile",
			modify:    func(f *benchFlags) { *f.phaseProfile = true },
			promptIDs: "1,2",
			steps:     4,
			reps:      1,
			wantErr:   "-phase-profile",
		},
		{
			name:      "empty prompt ids",
			modify:    func(f *benchFlags) {},
			promptIDs: "",
			steps:     4,
			reps:      1,
			wantErr:   "-raw-prompt-ids must be non-empty",
		},
		{
			name:      "invalid non-numeric prompt id",
			modify:    func(f *benchFlags) {},
			promptIDs: "1,abc,3",
			steps:     4,
			reps:      1,
			wantErr:   "invalid token ID",
		},
		{
			name:      "negative prompt id",
			modify:    func(f *benchFlags) {},
			promptIDs: "1,-2,3",
			steps:     4,
			reps:      1,
			wantErr:   "invalid token ID",
		},
		{
			name:      "empty part in prompt ids",
			modify:    func(f *benchFlags) {},
			promptIDs: "1,,3",
			steps:     4,
			reps:      1,
			wantErr:   "empty token ID",
		},
		{
			name:      "zero decode steps",
			modify:    func(f *benchFlags) {},
			promptIDs: "1,2,3",
			steps:     0,
			reps:      1,
			wantErr:   "-decode-steps must be >= 1",
		},
		{
			name:      "zero decode reps",
			modify:    func(f *benchFlags) {},
			promptIDs: "1,2,3",
			steps:     4,
			reps:      0,
			wantErr:   "-decode-reps must be >= 1",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cleanup := setRawDecodeTestFlags(true, tc.promptIDs, 256, false, false)
			defer cleanup()

			f := testRawDecodeFlags(tc.steps, tc.reps)
			tc.modify(f)

			err := validateRawDecodeFlags(f)
			if err == nil {
				t.Fatalf("expected error containing %q, got nil", tc.wantErr)
			}
			if !reflect.ValueOf(err.Error()).IsValid() {
				t.Fatalf("empty error string")
			}
		})
	}
}

func TestRawDecodeAllowsZeroPromptID(t *testing.T) {
	ids, err := parsePromptIDs("0, 1, 2")
	if err != nil {
		t.Fatalf("unexpected error with token 0: %v", err)
	}
	want := []int{0, 1, 2}
	if !reflect.DeepEqual(ids, want) {
		t.Errorf("got %v, want %v", ids, want)
	}
}

func TestRunRawDecodeWriteReportFile(t *testing.T) {
	tmpDir := t.TempDir()
	outPath := filepath.Join(tmpDir, "raw_decode_receipt.json")

	defer setRawDecodeTestFlags(true, "1,2,3", 256, false, false)()

	m := model.NewSynthetic(syntheticTestConfig())
	f := testRawDecodeFlags(3, 2)
	*f.out = outPath

	if err := runRawDecode(f, m, "synthetic-write-test", 10.0, 7.25, nil, nil); err != nil {
		t.Fatalf("runRawDecode failed: %v", err)
	}

	data, err := os.ReadFile(outPath)
	if err != nil {
		t.Fatalf("failed to read written report %s: %v", outPath, err)
	}

	var parsed map[string]any
	if err := json.Unmarshal(data, &parsed); err != nil {
		t.Fatalf("failed to unmarshal written report JSON: %v", err)
	}

	timings := parsed["timings"].(map[string]any)
	if timings["quant_ms"] != 7.25 || timings["verify_cpu_ms"] != float64(0) {
		t.Fatalf("written phase timings = %v", timings)
	}
	if parsed["raw_decode"] != true {
		t.Errorf("expected raw_decode: true in written file")
	}
	if parsed["reps"] != float64(2) {
		t.Errorf("expected reps: 2, got %v", parsed["reps"])
	}
	runs, ok := parsed["runs"].([]any)
	if !ok || len(runs) != 2 {
		t.Errorf("expected 2 runs in written file, got %v", parsed["runs"])
	}
}

func TestRawDecodePhaseTimings(t *testing.T) {
	for _, tc := range []struct {
		name   string
		verify bool
	}{{"disabled", false}, {"enabled", true}} {
		t.Run(tc.name, func(t *testing.T) {
			defer setRawDecodeTestFlags(true, "2,4,6", 256, true, tc.verify)()
			cfg := syntheticTestConfig()
			cfg.HiddenSize, cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim = 128, 8, 4, 16
			cfg.IntermediateSize, cfg.VocabSize = 256, 257
			m := model.NewSynthetic(cfg)
			f := testRawDecodeFlags(4, 2)
			*f.quant = true
			quantMS := quantizeIfNeeded(f, m)
			if quantMS <= 0 {
				t.Fatal("actual quantization was not timed")
			}
			report, err := executeRawDecode(f, m, "phase-timings", 3.0, quantMS, nil, nil)
			if err != nil {
				t.Fatal(err)
			}
			timings := report["timings"].(map[string]any)
			if timings["quant_ms"] != quantMS {
				t.Fatalf("quantization time = %v, want actual %v", timings["quant_ms"], quantMS)
			}
			verifyMS, ok := timings["verify_cpu_ms"].(float64)
			// Very fast synthetic replay can fall below the host clock's resolution.
			// Preserve the observed zero rather than manufacturing a timing floor;
			// the result and host-stage presence below prove that replay ran.
			if !ok || verifyMS < 0 || (!tc.verify && verifyMS != 0) {
				t.Fatalf("verification time = %v, enabled=%v", timings["verify_cpu_ms"], tc.verify)
			}
			if _, present := report["verify_cpu"]; present != tc.verify {
				t.Fatalf("verification result presence = %v, enabled=%v", present, tc.verify)
			}
			for _, run := range report["runs"].([]map[string]any) {
				timing := run["timings"].(map[string]any)
				v, ok := timing["verify_cpu_ms"].(float64)
				if !ok || v < 0 || (!tc.verify && v != 0) {
					t.Fatalf("per-run verification time = %v, enabled=%v", timing["verify_cpu_ms"], tc.verify)
				}
				var sum float64
				seenVerification := false
				for _, stage := range run["host_stages"].([]map[string]any) {
					sum += stage["duration_ms"].(float64)
					seenVerification = seenVerification || stage["stage"] == "verify_cpu"
				}
				if seenVerification != tc.verify || math.Abs(timing["total_ms"].(float64)-sum) > 1e-9 {
					t.Fatalf("per-run total %v differs from measured phases %v (verification=%v)", timing["total_ms"], sum, seenVerification)
				}
			}
		})
	}
}

// TestRawDecodePackedCPUReplaySnapshots covers real packed Q4_K/Q8 logits whose
// backing buffer is reused by a legacy CPU session. The loader still follows the
// published non-Vulkan dense-non-Q4_K conversion policy; no GPU backend is constructed.
func TestRawDecodePackedCPUReplaySnapshots(t *testing.T) {
	t.Setenv("FAK_GGUF_LOAD_WORKERS", "1")
	t.Setenv("FAK_PAGED_KV", "0")
	defer setRawDecodeTestFlags(true, "0,1", 16, true, true)()
	f := testRawDecodeFlags(4, 1)
	*f.gguf, *f.q4k, *f.backendName = benchMixedQuantGGUF(t), true, "cpu-ref"
	*f.quant, *f.metal = false, false
	m, _, err := loadModel(f, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer m.CloseWeights()
	if !m.HasQ4K("model.layers.0.mlp.up_proj.weight") || m.HasQ8("model.layers.0.mlp.up_proj.weight") {
		t.Fatal("fixture must retain its packed Q4_K projection")
	}
	for _, name := range []string{"lm_head.weight", "model.layers.0.self_attn.v_proj.weight", "model.layers.0.self_attn.o_proj.weight", "model.layers.0.mlp.gate_proj.weight", "model.layers.0.mlp.down_proj.weight"} {
		if !m.HasQ8(name) || m.HasKQuant(name) {
			t.Fatalf("fixture must load %s into the converted Q8 store", name)
		}
	}

	// Prove this fixture exercises the public borrow contract and has changing
	// logits, so a constant-output fixture cannot conceal missing snapshots.
	probe := newBenchSession(m, f, nil)
	defer probe.Close()
	borrowed := probe.Prefill([]int{0, 1})
	if len(borrowed) == 0 || !allFinite(borrowed) {
		t.Fatal("fixture prefill logits must be nonempty and finite")
	}
	prefillCopy := append([]float32(nil), borrowed...)
	stepOne := probe.Step(mathx.ArgmaxF32(prefillCopy))
	if len(stepOne) != len(borrowed) || &stepOne[0] != &borrowed[0] {
		t.Fatal("fixture must exercise the reused quantized logits buffer")
	}
	stepOneCopy := append([]float32(nil), stepOne...)
	stepTwo := probe.Step(mathx.ArgmaxF32(stepOneCopy))
	if len(stepTwo) != len(stepOne) || &stepTwo[0] != &stepOne[0] || !allFinite(stepTwo) {
		t.Fatal("fixture must reuse finite logits across consecutive Steps")
	}
	stepTwoCopy := append([]float32(nil), stepTwo...)
	prefillDelta := maxAbsDelta(prefillCopy, stepOneCopy)
	stepDelta := maxAbsDelta(stepOneCopy, stepTwoCopy)
	t.Logf("packed fixture: reused=true prefill_to_step_delta=%g step_to_step_delta=%g", prefillDelta, stepDelta)
	if !(prefillDelta > 0) || !(stepDelta > 0) {
		t.Fatal("fixture logits must change across Prefill and successive Steps")
	}

	// Both candidate and reference now execute the same loaded model on native
	// CPU with identical resident flags and token inputs. Any nonzero replay
	// delta is a bookkeeping error, not a CPU/GPU numerical tolerance question.
	report, err := executeRawDecode(f, m, "packed-cpu-replay", 1.0, 0, nil, nil)
	if err != nil {
		t.Fatalf("same-CPU packed replay failed: %v", err)
	}
	verify, ok := report["verify_cpu"].(*cpuVerifyResult)
	if !ok || verify == nil {
		t.Fatal("missing packed CPU replay evidence")
	}
	if !verify.Passed || !verify.AllAgree || verify.MaxDelta != 0 || verify.MinCosine < 0.999999 {
		t.Fatalf("same-CPU replay must compare each preserved output: passed=%v agree=%v max_delta=%g min_cosine=%g", verify.Passed, verify.AllAgree, verify.MaxDelta, verify.MinCosine)
	}
	if len(verify.Steps) != 3 {
		t.Fatalf("want three actual replay Steps, got %d", len(verify.Steps))
	}
	for _, step := range append([]stepVerify{verify.Prefill}, verify.Steps...) {
		if !step.Agree || step.MaxDelta != 0 || step.Cosine < 0.999999 {
			t.Fatalf("output %d was not compared with its own preserved logits: %+v", step.Step, step)
		}
	}
}

type divergingTestBackend struct {
	compute.Backend
}

func (b *divergingTestBackend) Read(t compute.Tensor) []float32 {
	raw := b.Backend.Read(t)
	out := append([]float32(nil), raw...)
	if len(out) > 1 {
		curArgmax := mathx.ArgmaxF32(out)
		other := (curArgmax + 1) % len(out)
		out[other] = out[curArgmax] + 100.0
	}
	return out
}

func TestRawDecodeCPUDivergencePreservesReceipt(t *testing.T) {
	defer setRawDecodeTestFlags(true, "2,4,6", 256, false, true)()

	m := model.NewSynthetic(syntheticTestConfig())
	f := testRawDecodeFlags(4, 1)

	be := &divergingTestBackend{Backend: compute.Default()}

	// 1. executeRawDecode must return both report and error describing divergence.
	report, err := executeRawDecode(f, m, "divergence-test", 1.0, 0, be, nil)
	if err == nil {
		t.Fatal("expected executeRawDecode to return error upon divergence, got nil")
	}
	if !strings.Contains(err.Error(), "divergence") {
		t.Fatalf("expected divergence error, got %v", err)
	}
	if report == nil {
		t.Fatal("expected executeRawDecode to preserve report receipt on divergence, got nil")
	}
	vRaw, ok := report["verify_cpu"]
	if !ok || vRaw == nil {
		t.Fatalf("report missing verify_cpu section on divergence: %v", report)
	}
	v, ok := vRaw.(*cpuVerifyResult)
	if !ok {
		t.Fatalf("expected *cpuVerifyResult, got %T", vRaw)
	}
	if v.Passed {
		t.Errorf("expected verify_cpu.Passed: false on divergence")
	}
	if v.AllAgree {
		t.Errorf("expected verify_cpu.AllAgree: false on divergence")
	}
	if v.AllArgmaxAgree {
		t.Errorf("expected verify_cpu.AllArgmaxAgree: false on divergence")
	}
	attempt := report["canonical_physical_receipt"].(rawDecodePhysicalReceiptAttempt)
	if attempt.Status != "UNAVAILABLE" || attempt.Observed.CPUModelParity == nil || *attempt.Observed.CPUModelParity {
		t.Fatalf("divergent CPU replay was not preserved as non-creditable parity=false: %+v", attempt)
	}

	// 2. runRawDecode with output file must write the report JSON despite error.
	tmpDir := t.TempDir()
	outPath := filepath.Join(tmpDir, "divergence_receipt.json")
	fOut := testRawDecodeFlags(4, 1)
	*fOut.out = outPath

	errRun := runRawDecode(fOut, m, "divergence-run-test", 1.0, 0, be, nil)
	if errRun == nil {
		t.Fatal("expected runRawDecode to return divergence error, got nil")
	}
	data, readErr := os.ReadFile(outPath)
	if readErr != nil {
		t.Fatalf("runRawDecode must write report file even on verification failure: %v", readErr)
	}
	var parsed map[string]any
	if unmarshalErr := json.Unmarshal(data, &parsed); unmarshalErr != nil {
		t.Fatalf("failed to unmarshal divergence report JSON: %v", unmarshalErr)
	}
	vMap, ok := parsed["verify_cpu"].(map[string]any)
	if !ok {
		t.Fatalf("missing verify_cpu in written divergence JSON: %v", parsed)
	}
	if vMap["all_argmax_agree"] != false {
		t.Errorf("expected verify_cpu.all_argmax_agree=false, got %v", vMap["all_argmax_agree"])
	}
	if vMap["all_agree"] != false {
		t.Errorf("expected verify_cpu.all_agree=false, got %v", vMap["all_agree"])
	}
	if vMap["passed"] != false {
		t.Errorf("expected verify_cpu.passed=false, got %v", vMap["passed"])
	}

	// 3. runRawDecode to stdout must emit the receipt JSON when -out is empty.
	fStdout := testRawDecodeFlags(4, 1)
	rPipe, wPipe, pipeErr := os.Pipe()
	if pipeErr != nil {
		t.Fatalf("failed to create pipe: %v", pipeErr)
	}
	oldStdout := os.Stdout
	os.Stdout = wPipe

	outCh := make(chan []byte)
	go func() {
		var buf bytes.Buffer
		_, _ = io.Copy(&buf, rPipe)
		_ = rPipe.Close()
		outCh <- buf.Bytes()
	}()

	errStdout := runRawDecode(fStdout, m, "divergence-stdout-test", 1.0, 0, be, nil)
	_ = wPipe.Close()
	os.Stdout = oldStdout

	if errStdout == nil {
		t.Fatal("expected runRawDecode to return divergence error for stdout run, got nil")
	}
	capturedStdout := <-outCh

	var parsedStdout map[string]any
	if unmarshalErr := json.Unmarshal(capturedStdout, &parsedStdout); unmarshalErr != nil {
		t.Fatalf("failed to unmarshal stdout divergence JSON: %v, raw output: %q", unmarshalErr, string(capturedStdout))
	}
	stdoutVMap, ok := parsedStdout["verify_cpu"].(map[string]any)
	if !ok {
		t.Fatalf("missing verify_cpu in stdout divergence JSON: %v", parsedStdout)
	}
	if stdoutVMap["all_argmax_agree"] != false {
		t.Errorf("stdout expected verify_cpu.all_argmax_agree=false, got %v", stdoutVMap["all_argmax_agree"])
	}
	if stdoutVMap["all_agree"] != false {
		t.Errorf("stdout expected verify_cpu.all_agree=false, got %v", stdoutVMap["all_agree"])
	}
	if stdoutVMap["passed"] != false {
		t.Errorf("stdout expected verify_cpu.passed=false, got %v", stdoutVMap["passed"])
	}
}

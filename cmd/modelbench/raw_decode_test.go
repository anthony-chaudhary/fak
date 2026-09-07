package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/mathx"
	"github.com/anthony-chaudhary/fak/internal/model"
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
	f := testRawDecodeFlags(5, 1)

	report, err := executeRawDecode(f, m, "synthetic-test", 12.5, nil, nil)
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
}

func TestRawDecodeGreedyFeedbackDiffersFromSyntheticSchedule(t *testing.T) {
	defer setRawDecodeTestFlags(true, "10,20,30", 256, false, false)()

	cfg := syntheticTestConfig()
	m := model.NewSynthetic(cfg)
	f := testRawDecodeFlags(5, 1)

	report, err := executeRawDecode(f, m, "synthetic-feedback", 5.0, nil, nil)
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
	report1, err := executeRawDecode(f1, m, "first-token-n1", 1.0, nil, nil)
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
	report4, err := executeRawDecode(f4, m, "first-token-n4", 1.0, nil, nil)
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
	probeReport, err := executeRawDecode(fProbe, m, "eos-probe", 1.0, nil, nil)
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
	rep1a, err := executeRawDecode(f1a, m, "eos-prefill-stop", 1.0, nil, nil)
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
	rep1b, err := executeRawDecode(f1b, m, "eos-prefill-ignore", 1.0, nil, nil)
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
	rep2, err := executeRawDecode(f2, m, "no-eos", 1.0, nil, nil)
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
	rep3a, err := executeRawDecode(f3a, m, "eos-list-stop", 1.0, nil, nil)
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
	rep3b, err := executeRawDecode(f3b, m, "eos-list-ignore", 1.0, nil, nil)
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

	report, err := executeRawDecode(f, m, "verify-cpu-test", 1.0, nil, nil)
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

	if err := runRawDecode(f, m, "synthetic-write-test", 10.0, nil, nil); err != nil {
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

package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"math"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/appversion"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/rawdecode"
)

var (
	rawDecodeFlag         = flag.Bool("raw-decode", false, "capture raw greedy decode with exact token accounting")
	rawPromptIDsFlag      = flag.String("raw-prompt-ids", "", "comma-separated prompt token IDs for raw decode (e.g. 1,2,3)")
	rawContextFlag        = flag.Int("raw-context", 256, "positive logical token context limit for raw decode")
	rawIgnoreEOSFlag      = flag.Bool("raw-ignore-eos", false, "ignore EOS during raw decode and continue until decode-steps")
	rawVerifyCPUFlag      = flag.Bool("raw-verify-cpu", false, "replay raw decode against fresh CPU session to verify logits and argmax")
	rawArtifactSHA256Flag = flag.String("raw-artifact-sha256", "", "required expected SHA-256 of the -gguf artifact for production raw decode")
)

func rawDecodeEnabled() bool {
	return rawDecodeFlag != nil && *rawDecodeFlag
}

// parsePromptIDs parses a comma-separated list of non-negative integer token IDs.
func parsePromptIDs(csv string) ([]int, error) {
	trimmed := strings.TrimSpace(csv)
	if trimmed == "" {
		return nil, errors.New("-raw-prompt-ids must be non-empty")
	}
	parts := strings.Split(trimmed, ",")
	ids := make([]int, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			return nil, errors.New("empty token ID in -raw-prompt-ids")
		}
		id, err := strconv.Atoi(p)
		if err != nil || id < 0 {
			return nil, fmt.Errorf("invalid token ID %q: must be non-negative integer", p)
		}
		ids = append(ids, id)
	}
	if len(ids) == 0 {
		return nil, errors.New("-raw-prompt-ids must contain at least one token ID")
	}
	return ids, nil
}

// validateRawDecodeFlags checks conflicting modes and validates parameters when -raw-decode is set.
func validateRawDecodeFlags(f *benchFlags) error {
	if !rawDecodeEnabled() {
		return nil
	}
	if f == nil {
		return errors.New("nil benchFlags")
	}

	// Mode exclusions: reject conflicting benchmark, readback, and profile modes.
	if f.loadOnly != nil && *f.loadOnly {
		return errors.New("-raw-decode cannot be combined with -load-only")
	}
	if f.verify != nil && *f.verify {
		return errors.New("-raw-decode cannot be combined with -verify")
	}
	if f.smoke != nil && *f.smoke {
		return errors.New("-raw-decode cannot be combined with -smoke")
	}
	if f.preflight != nil && *f.preflight {
		return errors.New("-raw-decode cannot be combined with -preflight")
	}
	if f.checkpoint != nil && *f.checkpoint != "" {
		return errors.New("-raw-decode cannot be combined with -checkpoint")
	}
	if f.resume != nil && *f.resume != "" {
		return errors.New("-raw-decode cannot be combined with -resume")
	}
	if f.workloadPath != nil && *f.workloadPath != "" {
		return errors.New("-raw-decode cannot be combined with -workload")
	}
	if f.phaseProfile != nil && *f.phaseProfile {
		return errors.New("-raw-decode cannot be combined with -phase-profile")
	}
	if f.loadProfile != nil && *f.loadProfile {
		return errors.New("-raw-decode cannot be combined with -load-profile")
	}
	if f.loadProfileTrace != nil && *f.loadProfileTrace {
		return errors.New("-raw-decode cannot be combined with -load-profile-trace")
	}
	if f.nativeProfileOut != nil && *f.nativeProfileOut != "" {
		return errors.New("-raw-decode cannot be combined with -native-performance-profile")
	}
	if f.nativeProfileReadback != nil && *f.nativeProfileReadback != "" {
		return errors.New("-raw-decode cannot be combined with -native-performance-readback")
	}
	if f.nativeProfileCompare != nil && *f.nativeProfileCompare != "" {
		return errors.New("-raw-decode cannot be combined with -native-performance-compare")
	}
	if f.qwenSwapOut != nil && *f.qwenSwapOut != "" {
		return errors.New("-raw-decode cannot be combined with -qwen38-paged-swap")
	}
	if f.qwenSwapReadback != nil && *f.qwenSwapReadback != "" {
		return errors.New("-raw-decode cannot be combined with -qwen38-paged-swap-readback")
	}

	promptIDsStr := ""
	if rawPromptIDsFlag != nil {
		promptIDsStr = *rawPromptIDsFlag
	}
	promptIDs, err := parsePromptIDs(promptIDsStr)
	if err != nil {
		return err
	}

	steps := 32
	if f.decodeSteps != nil {
		steps = *f.decodeSteps
	}
	if steps < 1 {
		return fmt.Errorf("-decode-steps must be >= 1 (got %d)", steps)
	}

	reps := 5
	if f.decodeReps != nil {
		reps = *f.decodeReps
	}
	if reps < 1 {
		return fmt.Errorf("-decode-reps must be >= 1 (got %d)", reps)
	}

	rawContext := 256
	if rawContextFlag != nil {
		rawContext = *rawContextFlag
	}
	if rawContext <= 0 {
		return fmt.Errorf("-raw-context must be > 0 (got %d)", rawContext)
	}
	minRequired := len(promptIDs) + steps
	if rawContext < minRequired {
		return fmt.Errorf("-raw-context (%d) must be >= prompt length (%d) + requested decode steps (%d) = %d",
			rawContext, len(promptIDs), steps, minRequired)
	}

	return nil
}

func maxAbsDelta(a, b []float32) float64 {
	var maxAbs float64
	for i := range a {
		d := math.Abs(float64(a[i] - b[i]))
		if d > maxAbs {
			maxAbs = d
		}
	}
	return maxAbs
}

type rawStepInfo struct {
	Step    int     `json:"step"`
	TokenID int     `json:"token_id"`
	Top1    float32 `json:"top1"`
	Top2    float32 `json:"top2"`
	Margin  float32 `json:"margin"`
}

type cpuVerifyResult struct {
	Passed         bool         `json:"passed"`
	AllAgree       bool         `json:"all_agree"`
	AllArgmaxAgree bool         `json:"all_argmax_agree"`
	MinCosine      float64      `json:"min_cosine"`
	MaxDelta       float64      `json:"max_delta"`
	Prefill        stepVerify   `json:"prefill"`
	Steps          []stepVerify `json:"steps,omitempty"`
}

type stepVerify struct {
	Step        int     `json:"step"`
	DeviceToken int     `json:"device_token"`
	CPUToken    int     `json:"cpu_token"`
	Agree       bool    `json:"agree"`
	Cosine      float64 `json:"cosine"`
	MaxDelta    float64 `json:"max_delta"`
}

type rawRepOutput struct {
	sessionSetupDur time.Duration
	prefillDur      time.Duration
	firstSampleDur  time.Duration
	decodeDur       time.Duration
	teardownDur     time.Duration
	cpuVerifyDur    time.Duration
	prefillOutputID int
	stepTokens      []int
	generatedTokens []int
	eosStopped      bool
	stepsInfo       []rawStepInfo
	hostStages      []map[string]any
	cpuVerify       *cpuVerifyResult
}

type rawDecodePhysicalReceiptAttempt struct {
	Status         string                              `json:"status"`
	CreditEligible bool                                `json:"credit_eligible"`
	Reason         string                              `json:"reason,omitempty"`
	Observed       compute.Qwen38VulkanRawDecodeResult `json:"observed"`
	Receipt        *compute.Qwen38VulkanDecodeReceipt  `json:"receipt,omitempty"`
}

func rawDecodeInt32IDs(ids []int) ([]int32, error) {
	out := make([]int32, len(ids))
	for i, id := range ids {
		if id < 0 || int64(id) > math.MaxInt32 {
			return nil, fmt.Errorf("token ID %d cannot be represented by canonical int32 token identity", id)
		}
		out[i] = int32(id)
	}
	return out, nil
}

// rawDecodePhysicalReceipt maps only executor observations. It never accepts
// caller-supplied source, binary, device, memory, or counter identity.
func rawDecodePhysicalReceipt(execution rawdecode.Execution, repOutputs []rawRepOutput) rawDecodePhysicalReceiptAttempt {
	finiteLogits := execution.FiniteLogits
	observed := compute.Qwen38VulkanRawDecodeResult{
		GeneratedTokenLimit: execution.GeneratedLimit,
		FiniteLogits:        &finiteLogits,
	}
	if execution.ArtifactSHA256 != "" {
		observed.Model.ArtifactSHA256 = execution.ArtifactSHA256
	}
	if execution.Backend.Selected != "" && execution.Backend.Selected != "legacy" {
		observed.Engine.Backend = execution.Backend.Selected
	}

	if len(repOutputs) == 0 {
		return rawDecodePhysicalReceiptAttempt{Status: "UNAVAILABLE", Reason: "raw decode produced no repetitions", Observed: observed}
	}
	var err error
	if observed.PromptTokenIDs, err = rawDecodeInt32IDs(execution.PromptTokenIDs); err != nil {
		return rawDecodePhysicalReceiptAttempt{Status: "UNAVAILABLE", Reason: err.Error(), Observed: observed}
	}
	allParityObserved, allParityPassed := true, true
	observed.Runs = make([]compute.Qwen38VulkanDecodeRun, len(repOutputs))
	for i, rep := range repOutputs {
		outputTokenIDs, convertErr := rawDecodeInt32IDs(rep.generatedTokens)
		if convertErr != nil {
			return rawDecodePhysicalReceiptAttempt{Status: "UNAVAILABLE", Reason: convertErr.Error(), Observed: observed}
		}
		ignoreEOSObserved, eosStoppedObserved := execution.IgnoreEOS, rep.eosStopped
		candidateElapsed := rep.sessionSetupDur + rep.prefillDur + rep.firstSampleDur + rep.decodeDur + rep.teardownDur
		observed.Runs[i] = compute.Qwen38VulkanDecodeRun{
			Repetition:                  i + 1,
			ContextLimit:                execution.ContextLimit,
			ContextTokens:               len(execution.PromptTokenIDs) + len(rep.generatedTokens),
			GeneratedTokenLimit:         execution.GeneratedLimit,
			ActualGeneratedTokens:       len(rep.generatedTokens),
			Sampler:                     "greedy",
			SeedPolicy:                  "not_applicable_greedy",
			IgnoreEOS:                   &ignoreEOSObserved,
			EOSStopped:                  &eosStoppedObserved,
			OutputTokenIDs:              outputTokenIDs,
			SessionSetupNanoseconds:     uint64(max(rep.sessionSetupDur.Nanoseconds(), 0)),
			PrefillNanoseconds:          uint64(max(rep.prefillDur.Nanoseconds(), 0)),
			FirstSampleNanoseconds:      uint64(max(rep.firstSampleDur.Nanoseconds(), 0)),
			DecodeNanoseconds:           uint64(max(rep.decodeDur.Nanoseconds(), 0)),
			TeardownNanoseconds:         uint64(max(rep.teardownDur.Nanoseconds(), 0)),
			CandidateElapsedNanoseconds: uint64(max(candidateElapsed.Nanoseconds(), 0)),
			CPUVerificationNanoseconds:  uint64(max(rep.cpuVerifyDur.Nanoseconds(), 0)),
		}
		if rep.cpuVerify == nil {
			allParityObserved = false
		} else if !rep.cpuVerify.Passed {
			allParityPassed = false
		}
	}
	observed.OutputTokenIDs = slices.Clone(observed.Runs[0].OutputTokenIDs)
	observed.CandidateElapsedNanoseconds = observed.Runs[0].CandidateElapsedNanoseconds
	observed.CPUVerificationNanoseconds = observed.Runs[0].CPUVerificationNanoseconds
	if allParityObserved {
		observed.CPUModelParity = &allParityPassed
	}
	receipt, err := compute.BuildQwen38VulkanDecodeReceipt(observed)
	if err != nil {
		return rawDecodePhysicalReceiptAttempt{Status: "UNAVAILABLE", Reason: err.Error(), Observed: observed}
	}
	return rawDecodePhysicalReceiptAttempt{Status: "AVAILABLE", CreditEligible: true, Observed: observed, Receipt: &receipt}
}

// executeRawDecode is the loaded-model adapter retained for existing modelbench
// sources and tests. Token generation lives only in internal/rawdecode.
func executeRawDecode(f *benchFlags, m *model.Model, modelName string, loadMS, quantMS float64, be compute.Backend, registeredBackends []string) (map[string]any, error) {
	if err := validateRawDecodeFlags(f); err != nil {
		return nil, err
	}
	promptIDs, err := parsePromptIDs(*rawPromptIDsFlag)
	if err != nil {
		return nil, err
	}
	req := rawDecodeRequest(f, promptIDs)
	req.ModelName = modelName
	execution, runErr := rawdecode.ExecuteModel(req, m, be)
	execution.LoadDuration = time.Duration(math.Round(loadMS * float64(time.Millisecond)))
	execution.QuantDuration = time.Duration(math.Round(quantMS * float64(time.Millisecond)))
	execution.Backend.RegisteredBackends = slices.Clone(registeredBackends)
	return rawDecodeReport(f, execution, runErr, modelName)
}

func rawDecodeRequest(f *benchFlags, promptIDs []int) rawdecode.Request {
	return rawdecode.Request{
		PromptTokenIDs: slices.Clone(promptIDs), ContextLimit: *rawContextFlag,
		GeneratedTokenLimit: *f.decodeSteps, Repetitions: *f.decodeReps,
		IgnoreEOS: *rawIgnoreEOSFlag, VerifyCPU: *rawVerifyCPUFlag,
		BackendName: *f.backendName, Quant: *f.quant, Lean: *f.lean, Q4K: *f.q4k,
		StreamQ4K: streamQ4KEnabled(f), Metal: *f.metal,
		Q4KGateUpOutputSlab: *f.q4kGateUpSlab,
		VulkanQ4KProfile:    *f.vulkanQ4KProfile, VulkanStageQ4K: *f.vulkanStageQ4K,
		RequireNonReference: *f.requireNonReference,
	}
}

func rawDecodeRepOutputs(execution rawdecode.Execution) []rawRepOutput {
	outputs := make([]rawRepOutput, len(execution.Runs))
	for i, run := range execution.Runs {
		var verify *cpuVerifyResult
		if run.CPUVerification != nil {
			converted := cpuVerifyResult{
				Passed: run.CPUVerification.Passed, AllAgree: run.CPUVerification.AllAgree,
				AllArgmaxAgree: run.CPUVerification.AllArgmaxAgree, MinCosine: run.CPUVerification.MinCosine,
				MaxDelta: run.CPUVerification.MaxDelta, Prefill: stepVerify(run.CPUVerification.Prefill),
			}
			for _, step := range run.CPUVerification.Steps {
				converted.Steps = append(converted.Steps, stepVerify(step))
			}
			verify = &converted
		}
		steps := make([]rawStepInfo, len(run.Steps))
		for j, step := range run.Steps {
			steps[j] = rawStepInfo(step)
		}
		hostStages := []map[string]any{
			{"stage": "session_setup", "duration_ms": float64(run.SessionSetupDuration.Nanoseconds()) / 1e6},
			{"stage": "prefill", "duration_ms": float64(run.PrefillDuration.Nanoseconds()) / 1e6},
			{"stage": "first_sample", "duration_ms": float64(run.FirstSampleDuration.Nanoseconds()) / 1e6},
			{"stage": "decode", "duration_ms": float64(run.DecodeDuration.Nanoseconds()) / 1e6},
			{"stage": "teardown", "duration_ms": float64(run.TeardownDuration.Nanoseconds()) / 1e6},
		}
		if run.CPUVerification != nil {
			hostStages = append(hostStages, map[string]any{"stage": "verify_cpu", "duration_ms": float64(run.CPUVerifyDuration.Nanoseconds()) / 1e6})
		}
		outputs[i] = rawRepOutput{
			sessionSetupDur: run.SessionSetupDuration, prefillDur: run.PrefillDuration,
			firstSampleDur: run.FirstSampleDuration, decodeDur: run.DecodeDuration,
			teardownDur: run.TeardownDuration, cpuVerifyDur: run.CPUVerifyDuration,
			prefillOutputID: run.PrefillOutputID, stepTokens: slices.Clone(run.StepTokens),
			generatedTokens: slices.Clone(run.GeneratedTokens), eosStopped: run.EOSStopped,
			stepsInfo: steps, hostStages: hostStages, cpuVerify: verify,
		}
	}
	return outputs
}

func rawDecodeReport(f *benchFlags, execution rawdecode.Execution, runErr error, displayModelName string) (map[string]any, error) {
	promptIDs := slices.Clone(execution.PromptTokenIDs)
	rawContext := execution.ContextLimit
	reps, ignoreEOS := len(execution.Runs), execution.IgnoreEOS
	modelName := displayModelName
	if modelName == "" {
		modelName = execution.ModelName
	}
	loadMS := float64(execution.LoadDuration.Nanoseconds()) / 1e6
	quantMS := float64(execution.QuantDuration.Nanoseconds()) / 1e6
	repOutputs := rawDecodeRepOutputs(execution)
	if len(repOutputs) == 0 {
		if runErr != nil {
			return nil, runErr
		}
		return nil, errors.New("raw decode produced no repetitions")
	}

	rep0 := repOutputs[0]
	effectiveIDs := make([]int, 0, len(promptIDs)+len(rep0.generatedTokens))
	effectiveIDs = append(effectiveIDs, promptIDs...)
	effectiveIDs = append(effectiveIDs, rep0.generatedTokens...)

	nRuns := len(repOutputs)
	setupDurs := make([]time.Duration, nRuns)
	prefillDurs := make([]time.Duration, nRuns)
	firstSampleDurs := make([]time.Duration, nRuns)
	decodeDurs := make([]time.Duration, nRuns)
	teardownDurs := make([]time.Duration, nRuns)
	cpuVerifyDurs := make([]time.Duration, nRuns)
	totalDurs := make([]time.Duration, nRuns)

	for i, ro := range repOutputs {
		setupDurs[i] = ro.sessionSetupDur
		prefillDurs[i] = ro.prefillDur
		firstSampleDurs[i] = ro.firstSampleDur
		decodeDurs[i] = ro.decodeDur
		teardownDurs[i] = ro.teardownDur
		cpuVerifyDurs[i] = ro.cpuVerifyDur
		totalDurs[i] = ro.sessionSetupDur + ro.prefillDur + ro.firstSampleDur + ro.decodeDur + ro.teardownDur + ro.cpuVerifyDur
	}

	timings := map[string]any{
		"load_ms":          loadMS,
		"quant_ms":         quantMS,
		"session_setup_ms": medianMS(setupDurs),
		"prefill_ms":       medianMS(prefillDurs),
		"first_sample_ms":  medianMS(firstSampleDurs),
		"decode_ms":        medianMS(decodeDurs),
		"teardown_ms":      medianMS(teardownDurs),
		"verify_cpu_ms":    medianMS(cpuVerifyDurs),
		"total_ms":         medianMS(totalDurs),
	}

	runs := make([]map[string]any, nRuns)
	for i, ro := range repOutputs {
		runs[i] = map[string]any{
			"rep":                     i + 1,
			"prefill_output_id":       ro.prefillOutputID,
			"step_tokens":             ro.stepTokens,
			"actual_tokens_generated": len(ro.generatedTokens),
			"actual_step_calls":       len(ro.stepTokens),
			"eos_stopped":             ro.eosStopped,
			"timings": map[string]any{
				"session_setup_ms": float64(ro.sessionSetupDur.Nanoseconds()) / 1e6,
				"prefill_ms":       float64(ro.prefillDur.Nanoseconds()) / 1e6,
				"first_sample_ms":  float64(ro.firstSampleDur.Nanoseconds()) / 1e6,
				"decode_ms":        float64(ro.decodeDur.Nanoseconds()) / 1e6,
				"teardown_ms":      float64(ro.teardownDur.Nanoseconds()) / 1e6,
				"verify_cpu_ms":    float64(ro.cpuVerifyDur.Nanoseconds()) / 1e6,
				"total_ms":         float64((ro.sessionSetupDur + ro.prefillDur + ro.firstSampleDur + ro.decodeDur + ro.teardownDur + ro.cpuVerifyDur).Nanoseconds()) / 1e6,
			},
			"steps":       ro.stepsInfo,
			"host_stages": ro.hostStages,
		}
		if ro.cpuVerify != nil {
			runs[i]["verify_cpu"] = ro.cpuVerify
		}
	}

	var sumMargin, minMargin, maxMargin float32
	minMargin = float32(math.MaxFloat32)
	maxMargin = -float32(math.MaxFloat32)
	for _, s := range rep0.stepsInfo {
		sumMargin += s.Margin
		if s.Margin < minMargin {
			minMargin = s.Margin
		}
		if s.Margin > maxMargin {
			maxMargin = s.Margin
		}
	}
	avgMargin := float32(0)
	if len(rep0.stepsInfo) > 0 {
		avgMargin = sumMargin / float32(len(rep0.stepsInfo))
	}
	marginSummary := map[string]any{
		"min_margin": minMargin,
		"max_margin": maxMargin,
		"avg_margin": avgMargin,
	}

	engine, precision := execution.Engine, execution.Precision
	backendReport := map[string]any{"selected": execution.Backend.Selected, "registered_backends": execution.Backend.RegisteredBackends}
	if execution.Backend.Selected != "" && execution.Backend.Selected != "legacy" {
		backendReport["native_host_stages"] = "permitted"
		backendReport["device_coverage"] = "unmeasured"
		backendReport["tier"], backendReport["class"], backendReport["caps"] = execution.Backend.Tier, execution.Backend.Class, execution.Backend.Caps
	}
	physicalReceipt := rawDecodePhysicalReceipt(execution, repOutputs)

	report := map[string]any{
		"app_version":                appversion.Current(),
		"engine":                     engine,
		"model":                      modelName,
		"precision":                  precision,
		"backend":                    backendReport,
		"raw_decode":                 true,
		"raw_ignore_eos":             ignoreEOS,
		"raw_context":                rawContext,
		"prompt_ids":                 promptIDs,
		"effective_ids":              effectiveIDs,
		"prefill_output_id":          rep0.prefillOutputID,
		"step_tokens":                rep0.stepTokens,
		"actual_tokens_generated":    len(rep0.generatedTokens),
		"actual_step_calls":          len(rep0.stepTokens),
		"eos_stopped":                rep0.eosStopped,
		"finite_logits":              true,
		"timings":                    timings,
		"timing_scope":               "load_ms and quant_ms occur once; per-repetition total_ms covers candidate setup through teardown plus optional CPU verification, excluding report assembly",
		"steps":                      rep0.stepsInfo,
		"margin_summary":             marginSummary,
		"host_stages":                rep0.hostStages,
		"canonical_physical_receipt": physicalReceipt,
		"reps":                       reps,
		"host": map[string]any{
			"os":         runtime.GOOS,
			"arch":       runtime.GOARCH,
			"num_cpu":    runtime.NumCPU(),
			"gomaxprocs": runtime.GOMAXPROCS(0),
		},
	}
	report["model_config"] = modelConfigReport(execution.ModelConfig)
	report["eos_id"] = execution.ModelConfig.EOSTokenID
	if reps > 1 {
		report["runs"] = runs
	}
	for _, ro := range repOutputs {
		if ro.cpuVerify != nil {
			report["verify_cpu"] = ro.cpuVerify
			if !ro.cpuVerify.Passed {
				break
			}
		}
	}

	return report, runErr
}

type rawDecodeArtifactExecutor func(context.Context, rawdecode.Request) (rawdecode.Execution, error)

func runRawDecodeArtifactWith(f *benchFlags, profiler *ggufload.LoadProfiler, execute rawDecodeArtifactExecutor) error {
	if err := validateRawDecodeFlags(f); err != nil {
		return err
	}
	if f.gguf == nil || strings.TrimSpace(*f.gguf) == "" {
		return errors.New("-raw-decode production execution requires an explicit -gguf artifact")
	}
	if rawArtifactSHA256Flag == nil || strings.TrimSpace(*rawArtifactSHA256Flag) == "" {
		return errors.New("-raw-decode production execution requires -raw-artifact-sha256")
	}
	promptIDs, err := parsePromptIDs(*rawPromptIDsFlag)
	if err != nil {
		return err
	}
	req := rawDecodeRequest(f, promptIDs)
	req.ArtifactPath = *f.gguf
	req.ExpectedArtifactSHA256 = strings.TrimSpace(*rawArtifactSHA256Flag)
	req.ModelName = strings.TrimSpace(*f.name)
	req.LoadProfiler = profiler
	execution, runErr := execute(context.Background(), req)
	report, reportErr := rawDecodeReport(f, execution, runErr, execution.ModelName)
	if report != nil {
		writeReport(f, report)
	}
	return reportErr
}

func runRawDecodeArtifact(f *benchFlags, profiler *ggufload.LoadProfiler) error {
	return runRawDecodeArtifactWith(f, profiler, rawdecode.Execute)
}

// runRawDecode performs raw greedy decode and writes the resulting JSON report.
func runRawDecode(f *benchFlags, m *model.Model, modelName string, loadMS, quantMS float64, be compute.Backend, registeredBackends []string) error {
	report, err := executeRawDecode(f, m, modelName, loadMS, quantMS, be, registeredBackends)
	if report != nil {
		writeReport(f, report)
	}
	if err != nil {
		return err
	}
	return nil
}

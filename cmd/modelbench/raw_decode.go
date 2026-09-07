package main

import (
	"errors"
	"flag"
	"fmt"
	"math"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/appversion"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/mathx"
	"github.com/anthony-chaudhary/fak/internal/model"
)

var (
	rawDecodeFlag    = flag.Bool("raw-decode", false, "capture raw greedy decode with exact token accounting")
	rawPromptIDsFlag = flag.String("raw-prompt-ids", "", "comma-separated prompt token IDs for raw decode (e.g. 1,2,3)")
	rawContextFlag   = flag.Int("raw-context", 256, "positive logical token context limit for raw decode")
	rawIgnoreEOSFlag = flag.Bool("raw-ignore-eos", false, "ignore EOS during raw decode and continue until decode-steps")
	rawVerifyCPUFlag = flag.Bool("raw-verify-cpu", false, "replay raw decode against fresh CPU session to verify logits and argmax")
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
	Passed    bool         `json:"passed"`
	AllAgree  bool         `json:"all_agree"`
	MinCosine float64      `json:"min_cosine"`
	MaxDelta  float64      `json:"max_delta"`
	Prefill   stepVerify   `json:"prefill"`
	Steps     []stepVerify `json:"steps,omitempty"`
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

// executeRawDecode runs raw greedy decode across reps, returning the structured report map.
func executeRawDecode(f *benchFlags, m *model.Model, modelName string, loadMS, quantMS float64, be compute.Backend, registeredBackends []string) (map[string]any, error) {
	if err := validateRawDecodeFlags(f); err != nil {
		return nil, err
	}
	promptIDs, err := parsePromptIDs(*rawPromptIDsFlag)
	if err != nil {
		return nil, err
	}
	if m != nil && m.Cfg.VocabSize > 0 {
		for _, id := range promptIDs {
			if id >= m.Cfg.VocabSize {
				return nil, fmt.Errorf("prompt token ID %d exceeds model vocabulary size %d", id, m.Cfg.VocabSize)
			}
		}
	}

	requestedSteps := 32
	if f != nil && f.decodeSteps != nil {
		requestedSteps = *f.decodeSteps
	}
	reps := 5
	if f != nil && f.decodeReps != nil {
		reps = *f.decodeReps
	}
	ignoreEOS := false
	if rawIgnoreEOSFlag != nil {
		ignoreEOS = *rawIgnoreEOSFlag
	}
	rawContext := 256
	if rawContextFlag != nil {
		rawContext = *rawContextFlag
	}
	verifyCPU := false
	if rawVerifyCPUFlag != nil {
		verifyCPU = *rawVerifyCPUFlag
	}

	isEOS := func(id int) bool {
		if m == nil {
			return false
		}
		return m.Cfg.IsEOS(id)
	}

	repOutputs := make([]rawRepOutput, 0, reps)

	for r := 0; r < reps; r++ {
		t0 := time.Now()
		s := newBenchSession(m, f, be)
		t1 := time.Now()
		sessionSetupDur := t1.Sub(t0)
		prefillLogits := s.Prefill(promptIDs)
		t2 := time.Now()
		prefillDur := t2.Sub(t1)

		if !allFinite(prefillLogits) {
			s.Close()
			return nil, errors.New("raw decode: prefill produced non-finite logits")
		}

		top1Idx, top1, top2 := logitTop2(prefillLogits)

		prefillOutputID := top1Idx
		stepsInfo := []rawStepInfo{
			{
				Step:    0,
				TokenID: prefillOutputID,
				Top1:    top1,
				Top2:    top2,
				Margin:  top1 - top2,
			},
		}
		generatedTokens := []int{prefillOutputID}
		stepTokens := make([]int, 0, requestedSteps-1)
		// Quantized sessions reuse their output buffers. Keep snapshots only when
		// CPU replay needs them after later calls and candidate teardown.
		var verifyPrefillLogits []float32
		var deviceStepLogits [][]float32
		if verifyCPU {
			verifyPrefillLogits = append([]float32(nil), prefillLogits...)
			deviceStepLogits = make([][]float32, 0, requestedSteps-1)
		}

		eosStopped := false
		if isEOS(prefillOutputID) && !ignoreEOS {
			eosStopped = true
		}

		t3 := time.Now()
		// This boundary is shared with prefill and decode: finite checks, first
		// token evidence, optional snapshot and bookkeeping stay in the interval.
		firstSampleDur := t3.Sub(t2)
		if !eosStopped && requestedSteps > 1 {
			prevTok := prefillOutputID
			for step := 1; step < requestedSteps; step++ {
				if len(promptIDs)+len(generatedTokens) >= rawContext {
					break
				}
				stepLogits := s.Step(prevTok)
				if !allFinite(stepLogits) {
					s.Close()
					return nil, fmt.Errorf("raw decode: step %d produced non-finite logits", step)
				}
				tokIdx, sTop1, sTop2 := logitTop2(stepLogits)
				stepTokens = append(stepTokens, tokIdx)
				generatedTokens = append(generatedTokens, tokIdx)
				if verifyCPU {
					deviceStepLogits = append(deviceStepLogits, append([]float32(nil), stepLogits...))
				}
				stepsInfo = append(stepsInfo, rawStepInfo{
					Step:    step,
					TokenID: tokIdx,
					Top1:    sTop1,
					Top2:    sTop2,
					Margin:  sTop1 - sTop2,
				})
				prevTok = tokIdx
				if isEOS(prevTok) && !ignoreEOS {
					eosStopped = true
					break
				}
			}
		}
		t4 := time.Now()
		decodeDur := t4.Sub(t3)
		s.Close()
		t5 := time.Now()
		teardownDur := t5.Sub(t4)

		var cpuVerify *cpuVerifyResult
		var cpuVerifyDur time.Duration
		if verifyCPU {
			cpuSession := m.NewSession()
			if f != nil {
				if f.quant != nil {
					cpuSession.Quant = *f.quant
				}
				if f.q4k != nil {
					cpuSession.Q4K = *f.q4k
				}
				if f.q4kGateUpSlab != nil {
					cpuSession.Q4KGateUpOutputSlab = *f.q4kGateUpSlab
				}
			}
			cpuPrefillLogits := cpuSession.Prefill(promptIDs)
			if !allFinite(cpuPrefillLogits) {
				cpuSession.Close()
				return nil, errors.New("raw verify cpu: CPU prefill produced non-finite logits")
			}
			cpuArgmax := mathx.ArgmaxF32(cpuPrefillLogits)
			cos0 := cosineF32(verifyPrefillLogits, cpuPrefillLogits)
			maxDelta0 := maxAbsDelta(verifyPrefillLogits, cpuPrefillLogits)
			agree0 := (prefillOutputID == cpuArgmax)

			minCos := cos0
			maxD := maxDelta0
			allAgree := agree0

			pVerify := stepVerify{
				Step:        0,
				DeviceToken: prefillOutputID,
				CPUToken:    cpuArgmax,
				Agree:       agree0,
				Cosine:      cos0,
				MaxDelta:    maxDelta0,
			}

			var stepVerifies []stepVerify
			prevReplayTok := prefillOutputID
			for i, stepTok := range stepTokens {
				cpuStepLogits := cpuSession.Step(prevReplayTok)
				if !allFinite(cpuStepLogits) {
					cpuSession.Close()
					return nil, fmt.Errorf("raw verify cpu: CPU step %d produced non-finite logits", i+1)
				}
				cArgmax := mathx.ArgmaxF32(cpuStepLogits)
				cCos := cosineF32(deviceStepLogits[i], cpuStepLogits)
				cDelta := maxAbsDelta(deviceStepLogits[i], cpuStepLogits)
				cAgree := (stepTok == cArgmax)
				if cCos < minCos {
					minCos = cCos
				}
				if cDelta > maxD {
					maxD = cDelta
				}
				if !cAgree {
					allAgree = false
				}
				stepVerifies = append(stepVerifies, stepVerify{
					Step:        i + 1,
					DeviceToken: stepTok,
					CPUToken:    cArgmax,
					Agree:       cAgree,
					Cosine:      cCos,
					MaxDelta:    cDelta,
				})
				prevReplayTok = stepTok
			}
			cpuSession.Close()

			cpuVerify = &cpuVerifyResult{
				Passed:    allAgree,
				AllAgree:  allAgree,
				MinCosine: minCos,
				MaxDelta:  maxD,
				Prefill:   pVerify,
				Steps:     stepVerifies,
			}
			if !allAgree {
				return nil, fmt.Errorf("raw decode CPU verification failed: argmax divergence between device and CPU reference")
			}
			// Includes fresh CPU session setup, replay, comparisons and Close.
			cpuVerifyDur = time.Since(t5)
		}

		hostStages := []map[string]any{
			{"stage": "session_setup", "duration_ms": float64(sessionSetupDur.Nanoseconds()) / 1e6},
			{"stage": "prefill", "duration_ms": float64(prefillDur.Nanoseconds()) / 1e6},
			{"stage": "first_sample", "duration_ms": float64(firstSampleDur.Nanoseconds()) / 1e6},
			{"stage": "decode", "duration_ms": float64(decodeDur.Nanoseconds()) / 1e6},
			{"stage": "teardown", "duration_ms": float64(teardownDur.Nanoseconds()) / 1e6},
		}
		if verifyCPU {
			hostStages = append(hostStages, map[string]any{"stage": "verify_cpu", "duration_ms": float64(cpuVerifyDur.Nanoseconds()) / 1e6})
		}

		repOutputs = append(repOutputs, rawRepOutput{
			sessionSetupDur: sessionSetupDur,
			prefillDur:      prefillDur,
			firstSampleDur:  firstSampleDur,
			decodeDur:       decodeDur,
			teardownDur:     teardownDur,
			cpuVerifyDur:    cpuVerifyDur,
			prefillOutputID: prefillOutputID,
			stepTokens:      stepTokens,
			generatedTokens: generatedTokens,
			eosStopped:      eosStopped,
			stepsInfo:       stepsInfo,
			hostStages:      hostStages,
			cpuVerify:       cpuVerify,
		})
	}

	rep0 := repOutputs[0]
	effectiveIDs := make([]int, 0, len(promptIDs)+len(rep0.generatedTokens))
	effectiveIDs = append(effectiveIDs, promptIDs...)
	effectiveIDs = append(effectiveIDs, rep0.generatedTokens...)

	setupDurs := make([]time.Duration, reps)
	prefillDurs := make([]time.Duration, reps)
	firstSampleDurs := make([]time.Duration, reps)
	decodeDurs := make([]time.Duration, reps)
	teardownDurs := make([]time.Duration, reps)
	cpuVerifyDurs := make([]time.Duration, reps)
	totalDurs := make([]time.Duration, reps)

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

	runs := make([]map[string]any, reps)
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

	engine, precision, backendReport := describeEngine(f, be, registeredBackends)

	report := map[string]any{
		"app_version":             appversion.Current(),
		"engine":                  engine,
		"model":                   modelName,
		"precision":               precision,
		"backend":                 backendReport,
		"raw_decode":              true,
		"raw_ignore_eos":          ignoreEOS,
		"raw_context":             rawContext,
		"prompt_ids":              promptIDs,
		"effective_ids":           effectiveIDs,
		"prefill_output_id":       rep0.prefillOutputID,
		"step_tokens":             rep0.stepTokens,
		"actual_tokens_generated": len(rep0.generatedTokens),
		"actual_step_calls":       len(rep0.stepTokens),
		"eos_stopped":             rep0.eosStopped,
		"finite_logits":           true,
		"timings":                 timings,
		"timing_scope":            "load_ms and quant_ms occur once; per-repetition total_ms covers candidate setup through teardown plus optional CPU verification, excluding report assembly",
		"steps":                   rep0.stepsInfo,
		"margin_summary":          marginSummary,
		"host_stages":             rep0.hostStages,
		"reps":                    reps,
		"host": map[string]any{
			"os":         runtime.GOOS,
			"arch":       runtime.GOARCH,
			"num_cpu":    runtime.NumCPU(),
			"gomaxprocs": runtime.GOMAXPROCS(0),
		},
	}
	if m != nil {
		report["model_config"] = modelConfigReport(m.Cfg)
		report["eos_id"] = m.Cfg.EOSTokenID
	}
	if reps > 1 {
		report["runs"] = runs
	}
	if rep0.cpuVerify != nil {
		report["verify_cpu"] = rep0.cpuVerify
	}

	return report, nil
}

// runRawDecode performs raw greedy decode and writes the resulting JSON report.
func runRawDecode(f *benchFlags, m *model.Model, modelName string, loadMS, quantMS float64, be compute.Backend, registeredBackends []string) error {
	report, err := executeRawDecode(f, m, modelName, loadMS, quantMS, be, registeredBackends)
	if err != nil {
		return err
	}
	writeReport(f, report)
	return nil
}

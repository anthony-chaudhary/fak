package rawdecode

import (
	"context"
	"errors"
	"fmt"
	"math"
	"slices"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// preserveExecutionCleanup leaves an incomplete body unwind in flight while
// attempting cleanup once. Cleanup failures join an ordinary returned error;
// without an earlier failure, an incomplete cleanup retains panic semantics.
func preserveExecutionCleanup(err *error, bodyComplete *bool, label string, cleanup func() error) {
	failed := !*bodyComplete
	hadError := *err != nil
	cleanupComplete := false
	var cleanupPanic any
	cleanupErr := func() (err error) {
		defer func() { cleanupPanic = recover() }()
		err = cleanup()
		cleanupComplete = true
		return err
	}()
	if failed {
		return
	}
	if !cleanupComplete {
		if !hadError {
			panic(cleanupPanic)
		}
		cleanupErr = fmt.Errorf("panic: %v", cleanupPanic)
	}
	if cleanupErr != nil {
		*err = errors.Join(*err, fmt.Errorf("%s: %w", label, cleanupErr))
	}
}

func executeObservedRun(ctx context.Context, req Request, m loadedModel, be compute.Backend, executionSeal *generationExecutionSeal, repetition int, now func() time.Time, since func(time.Time) time.Duration) (run Run, logprobs []float64, err error) {
	window, available, err := compute.BeginBackendExecutionObservation(be)
	if err != nil {
		return Run{}, nil, fmt.Errorf("raw decode: begin backend observation for run %d: %w", repetition+1, err)
	}
	bodyComplete := false
	if available {
		defer preserveExecutionCleanup(&err, &bodyComplete, fmt.Sprintf("raw decode: end backend observation for run %d", repetition+1), func() error {
			observed, endErr := window.End()
			if endErr != nil {
				return endErr
			}
			if err != nil {
				return nil
			}
			if !completeBackendExecutionObservation(observed) || observed.Identity.Backend != be.Name() {
				return errors.New("backend observation is incomplete or malformed")
			}
			run.BackendExecution = &observed
			run.timing.backendExecution = observed
			run.timing.observed = true
			return nil
		})
	}
	run, logprobs, err = executeRun(ctx, req, m, be, executionSeal, repetition, now, since)
	bodyComplete = true
	return run, logprobs, err
}

func executeRun(ctx context.Context, req Request, m loadedModel, be compute.Backend, executionSeal *generationExecutionSeal, repetition int, now func() time.Time, since func(time.Time) time.Duration) (run Run, selectedTokenLogprobs []float64, err error) {
	if err := ctx.Err(); err != nil {
		return Run{}, nil, err
	}
	start := now()
	s, err := m.NewCandidateSession(be, req)
	if err != nil {
		return Run{}, nil, fmt.Errorf("raw decode: create candidate session: %w", err)
	}
	sessionClosed := false
	closeSession := func() error {
		if !sessionClosed {
			sessionClosed = true
			s.Close()
		}
		return nil
	}
	bodyComplete := false
	defer preserveExecutionCleanup(&err, &bodyComplete, "raw decode: close candidate session", closeSession)
	run, selectedTokenLogprobs, err = func() (Run, []float64, error) {
		setupDone := now()
		if err := ctx.Err(); err != nil {
			return Run{}, nil, err
		}
		logits := s.Prefill(req.PromptTokenIDs)
		prefillDone := now()
		if err := ctx.Err(); err != nil {
			return Run{}, nil, err
		}
		token, top1, top2, selectedTokenLogprob, err := greedySelection(logits)
		if err != nil {
			return Run{}, nil, errors.New("raw decode: prefill produced non-finite logits")
		}
		run = Run{PrefillOutputID: token, GeneratedTokens: []int{token}, Steps: []Step{{Step: 0, TokenID: token, Top1: top1, Top2: top2, Margin: top1 - top2}}}
		selectedTokenLogprobs = []float64{selectedTokenLogprob}
		var candidateLogits [][]float32
		if req.VerifyCPU {
			candidateLogits = append(candidateLogits, slices.Clone(logits))
		}
		sampled := now()
		ignoreEOS := req.IgnoreEOS
		shouldStopOnEOS := func(token int) bool { return m.IsEOS(token) && !ignoreEOS }
		run.EOSStopped = shouldStopOnEOS(token)
		previous := token
		for step := 1; !run.EOSStopped && step < req.GeneratedTokenLimit && len(req.PromptTokenIDs)+len(run.GeneratedTokens) < req.ContextLimit; step++ {
			if err := ctx.Err(); err != nil {
				return Run{}, nil, err
			}
			logits = s.Step(previous)
			if err := ctx.Err(); err != nil {
				return Run{}, nil, err
			}
			token, top1, top2, selectedTokenLogprob, err = greedySelection(logits)
			if err != nil {
				return Run{}, nil, fmt.Errorf("raw decode: step %d produced non-finite logits", step)
			}
			run.StepTokens = append(run.StepTokens, token)
			run.GeneratedTokens = append(run.GeneratedTokens, token)
			run.Steps = append(run.Steps, Step{Step: step, TokenID: token, Top1: top1, Top2: top2, Margin: top1 - top2})
			selectedTokenLogprobs = append(selectedTokenLogprobs, selectedTokenLogprob)
			if req.VerifyCPU {
				candidateLogits = append(candidateLogits, slices.Clone(logits))
			}
			previous = token
			run.EOSStopped = shouldStopOnEOS(token)
		}
		if err := ctx.Err(); err != nil {
			return Run{}, nil, err
		}
		run.generation = GenerationObservation{
			observed:       true,
			executionSeal:  executionSeal,
			repetition:     repetition,
			ignoreEOS:      ignoreEOS,
			eosStopped:     run.EOSStopped,
			outputTokenIDs: slices.Clone(run.GeneratedTokens),
		}
		decoded := now()
		_ = closeSession()
		closed := now()
		boundaries := [6]time.Time{start, setupDone, prefillDone, sampled, decoded, closed}
		if !monotonicTimingBoundaries(boundaries) {
			return Run{}, nil, errors.New("raw decode: timing clock regressed during repetition")
		}
		run.timing = TimingObservation{
			executionSeal: executionSeal,
			repetition:    repetition,
			boundaries:    boundaries,
		}
		run.SessionSetupDuration = run.timing.SessionSetupDuration()
		run.PrefillDuration = run.timing.PrefillDuration()
		run.FirstSampleDuration = run.timing.FirstSampleDuration()
		run.DecodeDuration = run.timing.DecodeDuration()
		run.TeardownDuration = run.timing.TeardownDuration()
		if req.VerifyCPU {
			verifyStart := now()
			if err := ctx.Err(); err != nil {
				return Run{}, nil, err
			}
			verification, err := verifyCPU(ctx, req, m.NewCPUSession(req), run, candidateLogits)
			run.CPUVerifyDuration = since(verifyStart)
			if err != nil {
				return Run{}, nil, err
			}
			run.CPUVerification = &verification
		}
		return run, selectedTokenLogprobs, nil
	}()
	bodyComplete = true
	return run, selectedTokenLogprobs, err
}

func verifyCPU(ctx context.Context, req Request, s session, run Run, candidate [][]float32) (result CPUVerification, err error) {
	bodyComplete := false
	defer preserveExecutionCleanup(&err, &bodyComplete, "raw verify cpu: close session", func() error { s.Close(); return nil })
	result, err = func() (CPUVerification, error) {
		if err := ctx.Err(); err != nil {
			return CPUVerification{}, err
		}
		logits := s.Prefill(req.PromptTokenIDs)
		if err := ctx.Err(); err != nil {
			return CPUVerification{}, err
		}
		if !allFinite(logits) {
			return CPUVerification{}, errors.New("raw verify cpu: CPU prefill produced non-finite logits")
		}
		verification := compareStep(0, run.PrefillOutputID, candidate[0], logits)
		result = CPUVerification{Passed: verification.Agree, AllAgree: verification.Agree, AllArgmaxAgree: verification.Agree, MinCosine: verification.Cosine, MaxDelta: verification.MaxDelta, Prefill: verification}
		previous := run.PrefillOutputID
		for i, token := range run.StepTokens {
			if err := ctx.Err(); err != nil {
				return CPUVerification{}, err
			}
			logits = s.Step(previous)
			if err := ctx.Err(); err != nil {
				return CPUVerification{}, err
			}
			if !allFinite(logits) {
				return CPUVerification{}, fmt.Errorf("raw verify cpu: CPU step %d produced non-finite logits", i+1)
			}
			step := compareStep(i+1, token, candidate[i+1], logits)
			result.Steps = append(result.Steps, step)
			result.Passed = result.Passed && step.Agree
			result.AllAgree, result.AllArgmaxAgree = result.Passed, result.Passed
			result.MinCosine = math.Min(result.MinCosine, step.Cosine)
			result.MaxDelta = math.Max(result.MaxDelta, step.MaxDelta)
			previous = token
		}
		return result, nil
	}()
	bodyComplete = true
	return result, err
}

package main

import (
	"context"
	"errors"
	"fmt"
	"time"
)

type modelCanaryMonitorState struct {
	ctx            context.Context
	cfg            modelCanaryRunConfig
	durations      modelCanaryDurations
	deps           modelCanaryRunDeps
	receipt        *modelCanaryRunReceipt
	candidate      modelCanaryProcess
	request        modelCanaryProcess
	preflight      modelCanaryPreflight
	now            func() time.Time
	setFailure     func(modelCanaryPhase, string, error)
	requestRunning *bool
	terminalSet    *bool
}

func monitorModelCanary(state modelCanaryMonitorState) {
	ctx, cfg, durations, deps := state.ctx, state.cfg, state.durations, state.deps
	receipt, candidate, request := state.receipt, state.candidate, state.request
	preflight, now, setFailure := state.preflight, state.now, state.setFailure
	requestRunning, terminalSet := state.requestRunning, state.terminalSet
	if !*terminalSet {
		addModelCanaryEvent(receipt, now(), modelCanaryPhaseMonitoring, "monitor", "started", "")
		deadline := now().Add(durations.RequestDeadline)
		guard := modelCanaryGuardState{}
		var previousSampleAt time.Time
		for sequence := 1; ; sequence++ {
			if err := ctx.Err(); err != nil {
				setFailure(modelCanaryPhaseTerminalDecision, modelCanaryReasonCanceled, err)
				break
			}
			done, exitCode, pollErr := deps.PollRequest(request)
			if done {
				*requestRunning = false
				receipt.RequestExitCode = new(int)
				*receipt.RequestExitCode = exitCode
				var evidenceErr error
				lateRequest := false
				if deps.RequestEvidence == nil {
					evidenceErr = errors.New("request terminal evidence source is unavailable")
				} else {
					evidence, readErr := deps.RequestEvidence(request)
					if readErr == nil {
						readErr = validateModelCanaryRequestEvidence(evidence)
					}
					if readErr == nil {
						completedAt, _ := time.Parse(time.RFC3339Nano, evidence.CompletedAt)
						if completedAt.After(deadline) {
							lateRequest = true
							readErr = fmt.Errorf("request completed at %s after deadline %s", completedAt.UTC().Format(time.RFC3339Nano), deadline.UTC().Format(time.RFC3339Nano))
						}
					}
					if readErr != nil {
						evidenceErr = readErr
					} else {
						receipt.RequestEvidence = &evidence
					}
				}
				if evidenceErr != nil {
					if lateRequest {
						setFailure(modelCanaryPhaseTerminalDecision, modelCanaryReasonRequestDeadline, evidenceErr)
					} else {
						setFailure(modelCanaryPhaseTerminalDecision, modelCanaryReasonRequestEvidenceUnavailable, evidenceErr)
					}
				} else if pollErr != nil || exitCode != 0 {
					if pollErr == nil {
						pollErr = fmt.Errorf("request exited with status %d", exitCode)
					}
					setFailure(modelCanaryPhaseTerminalDecision, modelCanaryReasonRequestStartFailed, pollErr)
				} else {
					receipt.Outcome = "complete"
					receipt.TerminalPhase = modelCanaryPhaseTerminalDecision
				}
				terminalErr := evidenceErr
				if terminalErr == nil {
					terminalErr = pollErr
				}
				terminalOK := terminalErr == nil && exitCode == 0
				addModelCanaryEvent(receipt, now(), modelCanaryPhaseTerminalDecision, "request_terminal", map[bool]string{true: "ok", false: "error"}[terminalOK], errorDetail(terminalErr))
				*terminalSet = true
				break
			}
			if !now().Before(deadline) {
				setFailure(modelCanaryPhaseTerminalDecision, modelCanaryReasonRequestDeadline, errors.New("request deadline elapsed"))
				break
			}
			sample, sampleErr := deps.Sample(ctx, candidate, preflight.BaselineSwapBytes)
			if sampleErr != nil {
				setFailure(modelCanaryPhaseMonitoring, modelCanaryReasonObservationUnavailable, sampleErr)
				addModelCanaryEvent(receipt, now(), modelCanaryPhaseMonitoring, "sample", "refused", sampleErr.Error())
				break
			}
			sample.Sequence = sequence
			if sample.ObservedAt == "" {
				sample.ObservedAt = now().UTC().Format(time.RFC3339Nano)
			}
			sampleAt, err := validateModelCanarySampleAt(sample, candidate.Identity, preflight.BaselineSwapBytes, previousSampleAt, now(), durations.SampleInterval)
			if err != nil {
				setFailure(modelCanaryPhaseMonitoring, modelCanaryReasonObservationUnavailable, err)
				addModelCanaryEvent(receipt, now(), modelCanaryPhaseMonitoring, "sample", "refused", err.Error())
				break
			}
			previousSampleAt = sampleAt
			receipt.Samples = append(receipt.Samples, sample)
			guard = foldModelCanaryGuard(guard, cfg.Watcher, sample)
			receipt.Guard = guard
			addModelCanaryEvent(receipt, now(), modelCanaryPhaseMonitoring, "sample", "ok", fmt.Sprintf("sample=%d", sequence))
			if guard.TrippedMetric != "" {
				setFailure(modelCanaryPhaseTerminalDecision, modelCanaryReasonGuardTripped, fmt.Errorf("%s reached %d consecutive crossings", guard.TrippedMetric, cfg.Watcher.ConsecutiveCrossings))
				addModelCanaryEvent(receipt, now(), modelCanaryPhaseTerminalDecision, "safety_stop", "tripped", guard.TrippedMetric)
				break
			}
			if err := deps.Sleep(ctx, durations.SampleInterval); err != nil {
				setFailure(modelCanaryPhaseTerminalDecision, modelCanaryReasonCanceled, err)
				break
			}
		}
	}

}

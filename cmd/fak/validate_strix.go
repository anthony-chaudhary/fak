package main

import (
	"context"
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/amdgpu"
)

// isGPURelatedValidation reports whether any changed paths affect GPU/compute subsystems.
func isGPURelatedValidation(mine []string) bool {
	gpuRoots := []string{
		"internal/amdgpu",
		"internal/compute",
		"internal/roofline",
		"internal/model",
		"cmd/fak/validate_acceptance",
	}
	for _, p := range mine {
		norm := strings.ReplaceAll(p, "\\", "/")
		for _, root := range gpuRoots {
			if strings.HasPrefix(norm, root) || strings.Contains(norm, "strix") || strings.Contains(norm, "vulkan") {
				return true
			}
		}
	}
	return false
}

// shouldRunStrixValidation determines whether Strix Halo validation should run.
func shouldRunStrixValidation(explicitStrix bool, mine []string) bool {
	return explicitStrix || isGPURelatedValidation(mine)
}

// executeStrixValidationPhase runs physical validation on Strix Halo during fak validate.
func executeStrixValidationPhase(
	ctx context.Context,
	stdout, stderr io.Writer,
	res *validateResult,
	recorder *validateRecorder,
	explicitStrix bool,
	hostOverride string,
	subkernelsArg string,
	ablateArg string,
	mine []string,
) error {
	// If not explicit and changes do not touch GPU packages, skip immediately (0.0 ms)
	if !explicitStrix && !isGPURelatedValidation(mine) {
		res.SkippedPhases = append(res.SkippedPhases, "strix_validation")
		return nil
	}

	phase := recorder.start("strix_validation")

	// In auto mode, probe with a fast timeout (1.5s)
	probeTimeout := 1500 * time.Millisecond
	if explicitStrix {
		probeTimeout = 5 * time.Second
	}

	probeCtx, cancel := context.WithTimeout(ctx, probeTimeout)
	target, err := amdgpu.DiscoverStrixTarget(probeCtx, hostOverride)
	cancel()

	if err != nil || target == nil || !target.Reachable {
		if explicitStrix {
			phase.finish(fmt.Errorf("strix validation required but appliance unreachable: %v", err))
			res.Failures = append(res.Failures, ciPreflightFailure{
				Step:   "strix-validation",
				Detail: fmt.Sprintf("explicit --strix demanded but appliance unreachable: %v", err),
				Files:  []string{hostOverride},
			})
			res.OK = false
			return fmt.Errorf("strix unreachable: %w", err)
		}
		// In auto mode, unreachable Strix Halo fails open as advisory skip
		phase.finish(nil)
		res.SkippedPhases = append(res.SkippedPhases, "strix_validation")
		return nil
	}

	// Prepare validation options
	skList := []string{"argmax", "matmul_f32", "q4k_matmul", "rmsnorm", "swiglu"}
	if subkernelsArg != "" && subkernelsArg != "all" {
		skList = strings.Split(subkernelsArg, ",")
	}

	abList := []string{"cpu_vs_vulkan_gpu", "fused_vs_discrete_norm_matmul"}
	if ablateArg != "" && ablateArg != "all" {
		abList = strings.Split(ablateArg, ",")
	}

	opts := amdgpu.StrixValidationOpts{
		Host:          target.Host,
		RunSubkernels: true,
		Subkernels:    skList,
		RunAblations:  true,
		Ablations:     abList,
		GitRef:        res.Ref,
		GitTip:        res.Tip,
		Command:       "fak validate --strix",
		Timeout:       25 * time.Second,
	}

	receipt, valErr := amdgpu.RunStrixValidation(ctx, opts)
	phase.finish(valErr)

	if receipt != nil {
		res.StrixValidation = receipt
		if err := receipt.Validate(); err != nil {
			receipt.Verdict = "FAIL"
			receipt.Verified = false
			res.OK = false
			res.Failures = append(res.Failures, ciPreflightFailure{
				Step:   "strix-validation",
				Detail: fmt.Sprintf("receipt invariant validation failed: %v", err),
				Files:  []string{target.Host},
			})
		} else if receipt.Verdict != "PASS" {
			res.OK = false
			for _, f := range receipt.Failures {
				res.Failures = append(res.Failures, ciPreflightFailure{
					Step:   "strix-validation",
					Detail: f,
					Files:  []string{target.Host},
				})
			}
		}
	}

	return valErr
}

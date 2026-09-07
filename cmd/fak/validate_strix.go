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
		"cmd/fak/validate_acceptance.go",
		"cmd/fak/validate_acceptance_test.go",
	}
	gpuKeywords := []string{
		"strix",
		"halo",
		"vulkan",
		"gfx115",
	}
	for _, p := range mine {
		norm := strings.ReplaceAll(p, "\\", "/")
		lower := strings.ToLower(norm)
		for _, kw := range gpuKeywords {
			if strings.Contains(lower, kw) {
				return true
			}
		}
		for _, root := range gpuRoots {
			if norm == root || strings.HasPrefix(norm, root+"/") {
				return true
			}
		}
	}
	return false
}

var (
	discoverStrixTargetFn = amdgpu.DiscoverStrixTarget
	runStrixValidationFn  = amdgpu.RunStrixValidation
)

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
	target, err := discoverStrixTargetFn(probeCtx, hostOverride)
	cancel()

	if err != nil || target == nil || !target.Reachable {
		unreachErr := err
		if unreachErr == nil {
			if target != nil && target.Error != "" {
				unreachErr = fmt.Errorf("%s", target.Error)
			} else {
				unreachErr = fmt.Errorf("appliance unreachable or target not discovered")
			}
		}
		detail := fmt.Sprintf("strix hardware validation required for relevant changes but appliance is unreachable (pending hardware evidence): %v", unreachErr)
		if explicitStrix {
			detail = fmt.Sprintf("strix hardware validation required for relevant changes but appliance is unreachable (pending hardware evidence): explicit --strix demanded: %v", unreachErr)
		}
		phase.finish(fmt.Errorf("strix validation required but appliance unreachable: %w", unreachErr))
		var failFiles []string
		if hostOverride != "" {
			failFiles = []string{hostOverride}
		} else if target != nil && target.Host != "" {
			failFiles = []string{target.Host}
		}
		res.Failures = append(res.Failures, ciPreflightFailure{
			Step:   "strix-validation",
			Detail: detail,
			Files:  failFiles,
		})
		res.OK = false
		return fmt.Errorf("strix unreachable: %w", unreachErr)
	}

	// Prepare validation options
	skList := []string{"argmax", "matmul_f32", "q4k_matmul", "rmsnorm", "swiglu"}
	if subkernelsArg != "" && subkernelsArg != "all" {
		skList = strings.Split(subkernelsArg, ",")
	}

	runAblations := false
	var abList []string
	if ablateArg != "" {
		runAblations = true
		if ablateArg == "all" {
			abList = []string{"cpu_vs_vulkan_gpu", "fused_vs_discrete_norm_matmul"}
		} else {
			abList = strings.Split(ablateArg, ",")
		}
	}

	opts := amdgpu.StrixValidationOpts{
		Host:          target.Host,
		RunSubkernels: true,
		Subkernels:    skList,
		RunAblations:  runAblations,
		Ablations:     abList,
		GitRef:        res.Ref,
		GitTip:        res.Tip,
		Command:       "fak validate --strix",
		Timeout:       25 * time.Second,
	}

	receipt, valErr := runStrixValidationFn(ctx, opts)
	phase.finish(valErr)

	if receipt == nil {
		res.OK = false
		detail := "strix validation returned nil receipt"
		if valErr != nil {
			detail = fmt.Sprintf("strix validation failed: %v", valErr)
		}
		var files []string
		if target != nil && target.Host != "" {
			files = []string{target.Host}
		}
		res.Failures = append(res.Failures, ciPreflightFailure{
			Step:   "strix-validation",
			Detail: detail,
			Files:  files,
		})
		if valErr == nil {
			valErr = fmt.Errorf("strix validation returned nil receipt")
		}
		return valErr
	}

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
		return fmt.Errorf("receipt invariant validation failed: %w", err)
	}

	if receipt.Verdict != "PASS" {
		res.OK = false
		if len(receipt.Failures) == 0 {
			res.Failures = append(res.Failures, ciPreflightFailure{
				Step:   "strix-validation",
				Detail: fmt.Sprintf("strix validation verdict: %s", receipt.Verdict),
				Files:  []string{target.Host},
			})
		} else {
			for _, f := range receipt.Failures {
				res.Failures = append(res.Failures, ciPreflightFailure{
					Step:   "strix-validation",
					Detail: f,
					Files:  []string{target.Host},
				})
			}
		}
		return fmt.Errorf("strix validation failed with verdict: %s", receipt.Verdict)
	}

	if valErr != nil {
		res.OK = false
		res.Failures = append(res.Failures, ciPreflightFailure{
			Step:   "strix-validation",
			Detail: fmt.Sprintf("strix validation execution error: %v", valErr),
			Files:  []string{target.Host},
		})
		return valErr
	}

	return nil
}

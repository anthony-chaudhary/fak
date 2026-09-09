package main

import (
	"context"
	"errors"
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
	discoverStrixTargetFn           = amdgpu.DiscoverStrixTarget
	newStrixControllerAuthorityFn   = amdgpu.NewStrixControllerAuthority
	strixControllerAuthorityValidFn = func(authority amdgpu.StrixControllerAuthority) bool {
		return authority.Epoch() != "" &&
			authority.ObservedRevision() != "" &&
			authority.ExecutableSHA256() != "" &&
			authority.ExecutableBytes() > 0 &&
			!authority.AdmittedAt().IsZero()
	}
	runStrixValidationFn         = amdgpu.RunStrixValidation
	buildStrixCandidateArchiveFn = amdgpu.BuildStrixCandidateArchive
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
	root string,
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

	authority, err := newStrixControllerAuthorityFn(ctx, root)
	if err == nil && !strixControllerAuthorityValidFn(authority) {
		err = errors.New("controller authority constructor returned unusable authority")
	}
	if err != nil {
		code := "CONTROLLER_AUTHORITY_UNAVAILABLE"
		recovery := ""
		var refusal interface {
			Code() string
			Recovery() string
		}
		if errors.As(err, &refusal) {
			code = refusal.Code()
			recovery = refusal.Recovery()
		}
		detail := fmt.Sprintf("strix hardware validation required for relevant changes but controller authority refused: %s (observed_revision=%s executable_sha256=%s)",
			code, authority.ObservedRevision(), authority.ExecutableSHA256())
		var cleanupOwner interface{ RetryCleanup() error }
		if errors.As(err, &cleanupOwner) {
			if cleanupErr := cleanupOwner.RetryCleanup(); cleanupErr != nil {
				detail += "; cleanup_retry=pending"
			} else {
				detail += "; cleanup_retry=complete"
			}
		}
		if recovery != "" {
			detail += fmt.Sprintf("; recovery: %s", recovery)
		}
		phase.finish(fmt.Errorf("strix controller authority refused: %s", code))
		res.Failures = append(res.Failures, ciPreflightFailure{
			Step:   "strix-controller-authority",
			Detail: detail,
			Files:  mine,
		})
		res.OK = false
		return fmt.Errorf("strix controller authority refused: %s: %w", code, err)
	}
	// Keep the opaque authority alive for the full phase. A later receipt-binding
	// leaf can consume it without reconstructing or serializing private evidence.
	_ = authority

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
	skList := append([]string(nil), amdgpu.DefaultCreditableSubkernelSelectors...)
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

	candidate, archiveErr := buildStrixCandidateArchiveFn(ctx, root, res.Tip, mine)
	if archiveErr != nil {
		phase.finish(archiveErr)
		res.OK = false
		res.Failures = append(res.Failures, ciPreflightFailure{Step: "strix-validation", Detail: fmt.Sprintf("exact candidate archive failed: %v", archiveErr), Files: mine})
		return fmt.Errorf("strix candidate archive: %w", archiveErr)
	}
	admissionTimeout := 30 * time.Second
	runCount := len(skList) + len(abList)
	// Reserve a fresh-build budget plus the complete worst-case admission and
	// TERM/kill-after envelope for every sequential device execution.
	totalTimeout := 3*time.Minute + time.Duration(runCount)*(admissionTimeout+65*time.Second)

	opts := amdgpu.StrixValidationOpts{
		Host:                 target.Host,
		RunSubkernels:        true,
		Subkernels:           skList,
		RunAblations:         runAblations,
		Ablations:            abList,
		GitRef:               res.Ref,
		GitTip:               res.Tip,
		Command:              "fak validate --strix",
		Timeout:              totalTimeout,
		RequireSourceBinding: true,
		CandidateArchive:     candidate.Bytes,
		SourceArchiveSHA256:  candidate.SourceArchiveSHA256,
		AdmissionTimeout:     admissionTimeout,
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
	if !receipt.CreditEligible() {
		res.OK = false
		res.Failures = append(res.Failures, ciPreflightFailure{
			Step:   "strix-validation",
			Detail: "receipt is integrity-readable but not eligible for current physical Strix credit",
			Files:  []string{target.Host},
		})
		return fmt.Errorf("strix validation receipt is not credit eligible")
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

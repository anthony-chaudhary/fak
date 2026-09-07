package amdgpu

import (
	"context"
	"fmt"
	"time"
)

// StrixValidationOpts configures an execution of the Strix Halo validation suite.
type StrixValidationOpts struct {
	Host          string        `json:"host,omitempty"`
	Subkernels    []string      `json:"subkernels,omitempty"`
	Ablations     []string      `json:"ablations,omitempty"`
	RunSubkernels bool          `json:"run_subkernels"`
	RunAblations  bool          `json:"run_ablations"`
	GitRef        string        `json:"git_ref,omitempty"`
	GitTip        string        `json:"git_tip,omitempty"`
	Command       string        `json:"command,omitempty"`
	Timeout       time.Duration `json:"timeout,omitempty"`
}

// RunStrixValidation orchestrates sub-kernel tests and ablation arms on the Strix Halo machine.
func RunStrixValidation(ctx context.Context, opts StrixValidationOpts) (*StrixValidationReceipt, error) {
	if opts.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, opts.Timeout)
		defer cancel()
	}

	target, err := DiscoverStrixTarget(ctx, opts.Host)
	if err != nil || target == nil || !target.Reachable {
		receipt := NewStrixValidationReceipt(
			StrixTarget{
				Mode:         "ssh",
				Host:         opts.Host,
				Reachable:    false,
				TargetISA:    "gfx1151",
				ComputeUnits: 40,
				DiscoveredAt: time.Now().UTC().Format(time.RFC3339),
			},
			opts.GitRef,
			opts.GitTip,
			opts.Command,
		)
		receipt.Verdict = "FAIL"
		errMsg := "Strix Halo appliance unreachable"
		if err != nil {
			errMsg = fmt.Sprintf("Strix Halo appliance unreachable: %v", err)
		}
		receipt.Failures = append(receipt.Failures, errMsg)
		receipt.Verified = false
		digest, _ := receipt.ComputeDigest()
		receipt.Digest = digest
		return receipt, err
	}

	receipt := NewStrixValidationReceipt(*target, opts.GitRef, opts.GitTip, opts.Command)

	// 1. Run Subkernels if enabled
	if opts.RunSubkernels {
		skResults, skErr := RunSubkernelTests(ctx, target, opts.Subkernels)
		if skErr != nil {
			receipt.Failures = append(receipt.Failures, fmt.Sprintf("subkernels error: %v", skErr))
			receipt.Verdict = "FAIL"
		}
		receipt.Subkernels = skResults
		for _, sk := range skResults {
			if sk.Status == "FAIL" {
				receipt.Verdict = "FAIL"
				receipt.Failures = append(receipt.Failures, fmt.Sprintf("subkernel %q failed: %s", sk.Name, sk.Error))
			}
		}
	}

	// 2. Run Ablations if enabled
	if opts.RunAblations {
		abResults, abErr := RunStrixAblations(ctx, target, opts.Ablations)
		if abErr != nil {
			receipt.Failures = append(receipt.Failures, fmt.Sprintf("ablations error: %v", abErr))
			receipt.Verdict = "FAIL"
		}
		receipt.Ablations = abResults
		for _, ab := range abResults {
			if ab.Verdict == "REGRESSION" {
				receipt.Verdict = "FAIL"
				receipt.Failures = append(receipt.Failures, fmt.Sprintf("ablation %q suffered regression (speedup=%.2fx)", ab.Feature, ab.Speedup))
			}
		}
	}

	// 3. Seal and digest receipt
	receipt.Verified = (receipt.Verdict == "PASS")
	digest, err := receipt.ComputeDigest()
	if err == nil {
		receipt.Digest = digest
	}

	return receipt, nil
}

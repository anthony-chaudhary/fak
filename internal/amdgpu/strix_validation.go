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

	var selectedSpecs []SubkernelSpec
	if opts.RunSubkernels {
		var skErr error
		selectedSpecs, skErr = FilterSubkernelSpecs(opts.Subkernels)
		if skErr != nil {
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
			receipt.Verified = false
			receipt.SelectedCount = 0
			receipt.SelectedSubkernels = 0
			receipt.ExecutedCount = 0
			receipt.ExecutedSubkernels = 0
			receipt.Failures = append(receipt.Failures, fmt.Sprintf("subkernel selection error: %v", skErr))
			digest, _ := receipt.ComputeDigest()
			receipt.Digest = digest
			return receipt, skErr
		}
		if len(selectedSpecs) == 0 {
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
			receipt.Verified = false
			receipt.SelectedCount = 0
			receipt.SelectedSubkernels = 0
			receipt.ExecutedCount = 0
			receipt.ExecutedSubkernels = 0
			receipt.Failures = append(receipt.Failures, "subkernels enabled but zero subkernels selected")
			digest, _ := receipt.ComputeDigest()
			receipt.Digest = digest
			return receipt, fmt.Errorf("amdgpu: zero subkernels selected")
		}
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
		if opts.RunSubkernels {
			receipt.SelectedCount = len(selectedSpecs)
			receipt.SelectedSubkernels = len(selectedSpecs)
			receipt.ExecutedCount = 0
			receipt.ExecutedSubkernels = 0
		}
		digest, _ := receipt.ComputeDigest()
		receipt.Digest = digest
		return receipt, err
	}

	receipt := NewStrixValidationReceipt(*target, opts.GitRef, opts.GitTip, opts.Command)
	if opts.RunSubkernels {
		receipt.SelectedCount = len(selectedSpecs)
		receipt.SelectedSubkernels = len(selectedSpecs)
	}

	var validationErr error

	// 1. Run Subkernels if enabled
	if opts.RunSubkernels {
		skResults, skErr := RunSubkernelTests(ctx, target, opts.Subkernels)
		receipt.ExecutedCount = len(skResults)
		receipt.ExecutedSubkernels = len(skResults)
		if skErr != nil {
			receipt.Failures = append(receipt.Failures, fmt.Sprintf("subkernels error: %v", skErr))
			receipt.Verdict = "FAIL"
			receipt.Verified = false
			if validationErr == nil {
				validationErr = skErr
			}
		}
		receipt.Subkernels = skResults
		for _, sk := range skResults {
			if sk.Status == "FAIL" {
				receipt.Verdict = "FAIL"
				receipt.Verified = false
				receipt.Failures = append(receipt.Failures, fmt.Sprintf("subkernel %q failed: %s", sk.Name, sk.Error))
			}
		}
		if len(skResults) == 0 {
			receipt.Verdict = "FAIL"
			receipt.Verified = false
			receipt.Failures = append(receipt.Failures, "subkernels error: zero subkernels executed")
			if validationErr == nil {
				validationErr = fmt.Errorf("amdgpu: zero subkernels executed")
			}
		}
	}

	// 2. Run Ablations if enabled
	if opts.RunAblations {
		abResults, abErr := RunStrixAblations(ctx, target, opts.Ablations)
		if abErr != nil {
			receipt.Failures = append(receipt.Failures, fmt.Sprintf("ablations error: %v", abErr))
			receipt.Verdict = "FAIL"
			receipt.Verified = false
			if validationErr == nil {
				validationErr = abErr
			}
		}
		receipt.Ablations = abResults
		for _, ab := range abResults {
			if ab.Verdict == "REGRESSION" {
				receipt.Verdict = "FAIL"
				receipt.Verified = false
				receipt.Failures = append(receipt.Failures, fmt.Sprintf("ablation %q suffered regression (speedup=%.2fx)", ab.Feature, ab.Speedup))
			}
		}
	}

	// 3. Seal and digest receipt
	receipt.Verified = (receipt.Verdict == "PASS" && len(receipt.Failures) == 0)
	digest, err := receipt.ComputeDigest()
	if err == nil {
		receipt.Digest = digest
	}

	return receipt, validationErr
}

package amdgpu

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

type sourceBindingKey struct{}

// SourceBinding contains committed source provenance tokens bound to validation.
type SourceBinding struct {
	GitTip string
	GitRef string
}

// WithSourceBinding returns a new context carrying source binding tokens.
func WithSourceBinding(ctx context.Context, gitTip, gitRef string) context.Context {
	return context.WithValue(ctx, sourceBindingKey{}, SourceBinding{
		GitTip: strings.TrimSpace(gitTip),
		GitRef: strings.TrimSpace(gitRef),
	})
}

// SourceBindingFromContext retrieves source binding tokens from the context if present.
func SourceBindingFromContext(ctx context.Context) (SourceBinding, bool) {
	sb, ok := ctx.Value(sourceBindingKey{}).(SourceBinding)
	return sb, ok
}

// StrixValidationOpts configures an execution of the Strix Halo validation suite.
type StrixValidationOpts struct {
	Host                 string        `json:"host,omitempty"`
	Subkernels           []string      `json:"subkernels,omitempty"`
	Ablations            []string      `json:"ablations,omitempty"`
	RunSubkernels        bool          `json:"run_subkernels"`
	RunAblations         bool          `json:"run_ablations"`
	GitRef               string        `json:"git_ref,omitempty"`
	GitTip               string        `json:"git_tip,omitempty"`
	Command              string        `json:"command,omitempty"`
	Timeout              time.Duration `json:"timeout,omitempty"`
	RequireSourceBinding bool          `json:"require_source_binding,omitempty"`
}

var verifySourceBindingFn = VerifySourceBinding

// VerifySourceBinding verifies that the target appliance's source tree matches GitTip/GitRef.
func VerifySourceBinding(ctx context.Context, target *StrixTarget, gitTip, gitRef string) error {
	if target == nil || !target.Reachable {
		return fmt.Errorf("target is nil or unreachable")
	}
	cleanTip := strings.TrimSpace(gitTip)
	cleanRef := strings.TrimSpace(gitRef)
	if cleanTip == "" && cleanRef == "" {
		return nil
	}

	remoteDir := os.Getenv("FAK_STRIX_DIR")
	if remoteDir == "" {
		remoteDir = "/var/lib/fak/repo"
	}

	checkCmd := fmt.Sprintf("cd %s && git rev-parse HEAD", remoteDir)
	var cmd *exec.Cmd
	if target.Mode == "local" {
		cmd = exec.CommandContext(ctx, "bash", "-c", checkCmd)
	} else {
		cmd = exec.CommandContext(ctx, "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", target.Host, checkCmd)
	}
	windowgate.ConfigureBackgroundCommand(cmd)

	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed to resolve git commit on target: %v (output: %s)", err, truncateOutput(string(out), 150))
	}

	actualHead := strings.TrimSpace(string(out))
	if actualHead == "" {
		return fmt.Errorf("target returned empty git commit")
	}

	if cleanTip != "" {
		if !strings.HasPrefix(actualHead, cleanTip) && !strings.HasPrefix(cleanTip, actualHead) {
			return fmt.Errorf("source binding mismatch: target HEAD %s does not match GitTip %s", actualHead, cleanTip)
		}
	}

	return nil
}

// RunStrixValidation orchestrates sub-kernel tests and ablation arms on the Strix Halo machine.
func RunStrixValidation(ctx context.Context, opts StrixValidationOpts) (*StrixValidationReceipt, error) {
	if opts.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, opts.Timeout)
		defer cancel()
	}

	// Propagate source binding to context for downstream commands
	if opts.GitTip != "" || opts.GitRef != "" {
		ctx = WithSourceBinding(ctx, opts.GitTip, opts.GitRef)
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
		var errMsg string
		if err != nil {
			errMsg = fmt.Sprintf("strix halo target unreachable: %v", err)
		} else {
			errMsg = "strix halo target unreachable: target is nil or unreachable"
		}
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

	// Machine admission: verify source binding against GitTip / GitRef
	if opts.RequireSourceBinding && strings.TrimSpace(opts.GitTip) == "" {
		receipt.Verdict = "FAIL"
		receipt.Verified = false
		receipt.Failures = append(receipt.Failures, "source binding required but GitTip is missing")
		digest, _ := receipt.ComputeDigest()
		receipt.Digest = digest
		return receipt, fmt.Errorf("amdgpu: source binding required but GitTip is missing")
	}

	if opts.GitTip != "" || opts.RequireSourceBinding {
		if bindErr := verifySourceBindingFn(ctx, target, opts.GitTip, opts.GitRef); bindErr != nil {
			receipt.Verdict = "FAIL"
			receipt.Verified = false
			receipt.Failures = append(receipt.Failures, fmt.Sprintf("source binding verification failed: %v", bindErr))
			digest, _ := receipt.ComputeDigest()
			receipt.Digest = digest
			return receipt, fmt.Errorf("amdgpu: source binding verification failed: %w", bindErr)
		}
	}

	if !opts.RunSubkernels && !opts.RunAblations {
		receipt.Verdict = "FAIL"
		receipt.Failures = append(receipt.Failures, "missing validation evidence: neither subkernels nor ablations requested")
	}

	var validationErr error

	// 1. Run Subkernels if enabled
	if opts.RunSubkernels {
		skResults, skErr := RunSubkernelTests(ctx, target, opts.Subkernels)
		receipt.ExecutedCount = len(skResults)
		receipt.ExecutedSubkernels = len(skResults)
		if skErr != nil {
			receipt.Verdict = "FAIL"
			receipt.Verified = false
			receipt.Failures = append(receipt.Failures, fmt.Sprintf("subkernels error: %v", skErr))
			if validationErr == nil {
				validationErr = skErr
			}
		}
		if len(opts.Subkernels) > 0 && len(skResults) == 0 {
			receipt.Verdict = "FAIL"
			receipt.Verified = false
			if skErr == nil {
				receipt.Failures = append(receipt.Failures, "subkernels requested but none were executed")
				if validationErr == nil {
					validationErr = fmt.Errorf("amdgpu: zero subkernels executed")
				}
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
	}

	// 2. Run Ablations if enabled
	if opts.RunAblations {
		abResults, abErr := RunAblationTests(ctx, target, opts.Ablations)
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

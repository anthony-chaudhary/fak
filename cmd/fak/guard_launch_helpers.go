package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"
)

// resolveGuardMetal mirrors resolveRunMetal: it resolves the Apple-Silicon Metal
// session-forward decision through the shared serve resolver, rewriting the verb in the
// error so a guard failure reads "fak guard:" rather than "fak serve:".
func resolveGuardMetal(flag, env bool, backendName string) (bool, error) {
	use, err := resolveServeMetal(flag, env, backendName)
	if err != nil {
		return false, errors.New(strings.ReplaceAll(err.Error(), "fak serve:", "fak guard:"))
	}
	return use, nil
}

// admitGuardLaunchLease runs launch lease admission and routes its diagnostics through the
// startup progress line so they never interleave with the phase output.
func admitGuardLaunchLease(progress *guardStartupProgress, cfg guardArbitrateConfig) (*guardArbitrateLease, error) {
	progress.Phase("lease admission")
	var stderr strings.Builder
	lease, err := guardArbitrateAcquire(context.Background(), &stderr, guardArbitrateConfig{
		Mode: cfg.Mode, Lane: cfg.Lane, Tree: cfg.Tree, Force: cfg.Force,
	})
	if stderr.Len() > 0 {
		progress.EndLine()
		fmt.Fprint(os.Stderr, stderr.String())
	}
	if err != nil {
		progress.Abort()
		fmt.Fprintf(os.Stderr, "fak guard: %v\n", err)
	}
	return lease, err
}

// configureGuardCodexApproveForMe ensures a Codex command uses automated approval reviewer
// integration (approvals_reviewer = "auto_review") without stripping the sandbox.
func configureGuardCodexApproveForMe(command []string) []string {
	if len(command) == 0 {
		return command
	}
	filtered := make([]string, 0, len(command))
	for _, arg := range command {
		if arg != "--dangerously-bypass-approvals-and-sandbox" {
			filtered = append(filtered, arg)
		}
	}
	command = filtered
	for _, arg := range command {
		if strings.Contains(arg, "approvals_reviewer") {
			return command
		}
	}
	out := make([]string, 0, len(command)+2)
	out = append(out, command[0], "-c", `approvals_reviewer="auto_review"`)
	out = append(out, command[1:]...)
	return out
}

package main

import (
	"fmt"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// serve_ncpumoe.go — the operator-facing door to the GRADED expert spill (#5628, epic #5606).
//
// The sizing math landed with #5612: Model.ResolveExpertSpillPlacement builds a budget from
// measured resident bytes and real MoE layer ordinals, Session.ApplyExpertSpillPlacement installs
// it, and InKernelPlanner.SetExpertSpill resolves it once per serve against the measured device
// budget. What did not land is the way an operator ASKS for it. Until this flag, the only door was
// the environment variable agent.ExpertSpillEnv — reachable from a Go caller or a shell export, and
// listed in no --help output.
//
// The flag is spelled the way llama.cpp spells it, matching agent.ParseExpertSpillGrade's own
// vocabulary, so an operator carrying a working --n-cpu-moe number types the same thing here.
//
// It is also the STRICT door, which the env deliberately is not. setExpertSpillFromEnv logs and
// continues on a bad grade, because taking down an already-constructed planner over a mistyped
// optional knob trades a degraded placement for no serve at all. Its own comment names the other
// posture: "a caller that parses an operator FLAG gets the error back and can refuse the launch
// outright, which is the right posture when the operator is still at the terminal." That is this.
// Refusing at flag time also means a typo costs nothing — the refusal lands before the multi-minute
// GGUF load, not after it.
//
// The value reaches the planner through agent.ExpertSpillEnv, the seam NewInKernelPlanner already
// reads, so no new plumbing is threaded through the gateway config. Promoting it to a
// gateway.Config field read at the NewInKernelPlanner call site is a mechanical follow-up that
// changes no behavior; this keeps the flag and the env knob resolving to one placement either way.

// serveNCPUMoEFlag is the flag name, kept in one place so the registration and the refusals agree.
const serveNCPUMoEFlag = "n-cpu-moe"

// applyServeNCPUMoE validates an operator's --n-cpu-moe grade and carries it to the in-kernel
// planner, returning a refusal the caller can fail the launch on.
//
// An EMPTY value means the flag was not passed. Any ambient agent.ExpertSpillEnv is then left
// exactly as it was, so an un-passed flag is byte-for-byte the previous path — including the case
// where a host profile already exports the env var.
//
// A value that IS passed wins over the ambient environment, and that deliberately includes the
// explicit "off": an operator who types --n-cpu-moe off on a host whose profile exports
// FAK_N_CPU_MOE=auto means off, and the more explicit of the two should decide.
func applyServeNCPUMoE(v string) error {
	if strings.TrimSpace(v) == "" {
		return nil
	}
	if _, _, err := agent.ParseExpertSpillGrade(v); err != nil {
		return fmt.Errorf("fak serve: --%s: %w", serveNCPUMoEFlag, err)
	}
	// Normalized, so the planner's own parse of the env var sees the same token the operator's
	// grade parsed to and the placement log line reads back what was actually applied.
	return os.Setenv(agent.ExpertSpillEnv, strings.ToLower(strings.TrimSpace(v)))
}

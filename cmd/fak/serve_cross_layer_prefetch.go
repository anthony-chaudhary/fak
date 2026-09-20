package main

import (
	"fmt"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// serve_cross_layer_prefetch.go — the operator-facing door to per-session next-layer gate prefetch
// (#1297/#1401, R3.5 of the activated-expert offload ladder #5614, epic #5606).
//
// The predictor landed in internal/model (expert_readahead.go): while layer L computes, layer L+1's
// router gate is applied to L's hidden state and the predicted top-k is staged into the routed-expert
// ring as HINTS — never a demand, so a mispredict cannot change logits. The knob is a Session field
// (a sibling of ExpertPrefetch) precisely so it can be enabled for ONE serve without turning it on
// fleet-wide; the planner installs it per session (agent.SetCrossLayerGatePrefetch), and the
// precision/recall ledger rides out through the expert-residency report (MoEResidencyReport.CrossLayer).
//
// What this flag adds is the way an operator ASKS for it. Until it existed the only door was the
// environment variable agent.CrossLayerGatePrefetchEnv — reachable from a Go caller or a shell export,
// and listed in no --help output. Like --n-cpu-moe, the flag is the STRICT door: a value that is not a
// boolean REFUSES the launch here, before the multi-minute GGUF load, rather than silently serving
// with the knob off. The value reaches the planner through agent.CrossLayerGatePrefetchEnv, the seam
// NewInKernelPlanner already reads, so the flag and the env agree.

// serveCrossLayerGatePrefetchFlag is the flag name, kept in one place so the registration and the
// refusals agree.
const serveCrossLayerGatePrefetchFlag = "cross-layer-gate-prefetch"

// applyServeCrossLayerGatePrefetch validates an operator's --cross-layer-gate-prefetch value and
// carries it to the in-kernel planner, returning a refusal the caller can fail the launch on.
//
// An EMPTY value means the flag was not passed. Any ambient agent.CrossLayerGatePrefetchEnv is then
// left exactly as it was, so an un-passed flag is byte-for-byte the previous path — including the case
// where a host profile already exports the env var.
//
// A value that IS passed wins over the ambient environment, including an explicit "false": an operator
// who types --cross-layer-gate-prefetch=false on a host whose profile exports
// FAK_CROSS_LAYER_GATE_PREFETCH=1 means off, and the more explicit of the two should decide.
func applyServeCrossLayerGatePrefetch(v string) error {
	if strings.TrimSpace(v) == "" {
		return nil
	}
	on, _, err := agent.ParseCrossLayerGatePrefetch(v)
	if err != nil {
		return fmt.Errorf("fak serve: --%s: %w", serveCrossLayerGatePrefetchFlag, err)
	}
	// Normalized, so the planner's own parse of the env var sees the same token the operator typed and
	// the install log line reads back what was actually applied.
	return os.Setenv(agent.CrossLayerGatePrefetchEnv, fmt.Sprintf("%t", on))
}

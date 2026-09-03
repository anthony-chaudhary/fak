package main

import (
	"flag"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// registerGuardAdmissionFlag declares the guard/manage --native-admission-token-budget
// flag: the same name, default, and semantics as serve's (serve.go), extracted here so
// the surface can be parsed and asserted in tests. The default is the gateway's shipping
// policy budget, never a literal, so the two doors cannot drift apart on it.
func registerGuardAdmissionFlag(fs *flag.FlagSet) *int {
	return fs.Int("native-admission-token-budget", gateway.DefaultAdmissionPolicy().TokenBudget,
		"with --gguf (a fak-native in-kernel model): cap the total token footprint admitted by the request scheduler; must be positive (default 8192). Same flag and semantics as `fak serve`. Raise it when the wrapped harness's FIRST prompt alone exceeds the default — opencode 1.18's system+tools floor prompt is ~45k tokens, so the managed seat sheds every turn (429 \"request tokens exceed scheduler token budget\") until this is raised (#10597).")
}

// newGuardNativeAdmissionController is the guard/manage twin of
// newServeNativeAdmissionController: same admission gate, same policy seam, sourced
// from the guard launch flag set instead of serveFlags. The startup note names the
// launching verb (manage or guard) so the bound readback stays honest about which
// front door installed it.
func newGuardNativeAdmissionController(commandName string, budget int) (*gateway.AdmissionController, gateway.StartupMessage, error) {
	policy, err := nativeAdmissionPolicyForBudget(budget)
	if err != nil {
		return nil, gateway.StartupMessage{}, err
	}
	return gateway.NewAdmissionController(policy), newServeStartupMessage(commandName, "native-admission-token-budget", "info",
		fmt.Sprintf("native scheduler admission token budget=%d", policy.TokenBudget)), nil
}

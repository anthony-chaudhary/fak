package main

import (
	"fmt"
	"strconv"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

const inkling2SpecEnv = "FAK_INKLING2_SPEC"

// inkling2ExplicitSpecDiagnostic reports the mismatch between an operator's
// positive speculative-depth request and the current serve engines, which all
// execute plain Step decode. Parsing mirrors the positivity gate only; the
// runtime's existing parser remains the sole owner of depth aliases and clamps.
func inkling2ExplicitSpecDiagnostic(lookup func(string) (string, bool)) gateway.StartupMessage {
	if lookup == nil {
		return gateway.StartupMessage{}
	}
	raw, set := lookup(inkling2SpecEnv)
	requested, err := strconv.Atoi(raw)
	if !set || err != nil || requested <= 0 {
		return gateway.StartupMessage{}
	}

	return newServeStartupMessage("serve", "inkling2-spec-config", "warn",
		fmt.Sprintf("explicit %s=%q requested speculative decode, but fak serve currently selects plain decode; the setting is not active", inkling2SpecEnv, raw))
}

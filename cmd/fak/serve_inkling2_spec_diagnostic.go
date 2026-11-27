package main

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

const inkling2SpecEnv = "FAK_INKLING2_SPEC"

// inkling2ExplicitSpecDiagnostic reports when an operator explicitly configures
// FAK_INKLING2_SPEC in fak serve. Callers with the resolved decode banner use
// inkling2ExplicitSpecDiagnosticFor to report active spec vs plain fallback.
func inkling2ExplicitSpecDiagnostic(lookup func(string) (string, bool)) gateway.StartupMessage {
	return inkling2ExplicitSpecDiagnosticFor("", lookup)
}

func inkling2ExplicitSpecDiagnosticFor(decode string, lookup func(string) (string, bool)) gateway.StartupMessage {
	if lookup == nil {
		return gateway.StartupMessage{}
	}
	raw, set := lookup(inkling2SpecEnv)
	requested, err := strconv.Atoi(strings.TrimSpace(raw))
	if !set || err != nil || requested <= 0 {
		return gateway.StartupMessage{}
	}
	if decode != "" && (!strings.Contains(decode, "spec=") || strings.Contains(decode, "spec=off")) {
		return newServeStartupMessage("serve", "inkling2-spec-config", "warn",
			fmt.Sprintf("explicit %s=%q requested speculative decode, but fak serve selected plain decode (%s)", inkling2SpecEnv, raw, decode))
	}
	detail := ""
	if idx := strings.Index(decode, "spec="); idx >= 0 {
		specPart := decode[idx:]
		if semi := strings.IndexByte(specPart, ';'); semi >= 0 {
			specPart = strings.TrimSpace(specPart[:semi])
		}
		detail = fmt.Sprintf(" (%s)", specPart)
	}
	return newServeStartupMessage("serve", "inkling2-spec-config", "info",
		fmt.Sprintf("explicit %s=%q active for greedy decode in fak serve%s", inkling2SpecEnv, raw, detail))
}

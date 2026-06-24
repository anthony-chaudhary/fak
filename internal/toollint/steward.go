package toollint

// steward.go — the tool linter's contribution to the DEFCONFIG: an always-on
// single-invariant steward asserting the booted tool SURFACE carries no
// error-severity finding. This is how the loudest lint rules (TL003 — a static
// answer that swallows a write; TL008 — a fast-path serve that bypasses a provable
// policy Deny) become a kernel-resident invariant gated by the same steward sweep
// that guards every other invariant, not merely an out-of-band `fak lint`.
//
// It honors the steward contract (ABI: never block on your own opinion — abstain
// unless an INDEPENDENTLY-AUTHORED witness is available). A lint finding is
// DETERMINISTICALLY re-derivable from the registries by anyone, so the finding
// itself (its Code, the offending tool, and the kernel mechanism it predicts) is the
// witness: a third party re-runs the same rules over the same registries and
// confirms. On a clean surface — the default — it abstains.

import (
	"context"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
)

// surfaceSteward implements abi.Steward over the live tool surface.
type surfaceSteward struct{}

// Name returns "tool-surface-sound", this steward's stable id.
func (surfaceSteward) Name() string { return "tool-surface-sound" }

func (surfaceSteward) Check(context.Context) (violated bool, witness string) {
	return surfaceViolation(kernelSurfaceFacts())
}

// surfaceViolation is the PURE decision the steward reports: a violation iff the
// surface has any error-severity finding, witnessed by the first such finding
// rendered as "<code> <tool>: <mechanism>". Factored out so the steward's verdict is
// unit-testable on synthetic facts without touching the process-global registries.
func surfaceViolation(facts []ToolFacts) (bool, string) {
	for _, f := range Lint(facts).Findings {
		if f.Severity == SevError {
			return true, string(f.Code) + " " + f.Tool + ": " + f.Mechanism
		}
	}
	return false, ""
}

// kernelSurfaceFacts is the steward's view of the booted surface: the registries-only
// facts (FromKernel) folded with the adjudicator's Deny set, so the policy-bypass
// rule (TL008) — which needs the policy the kernel-only collector cannot see — is
// covered alongside the name/schema rules. Only a denied tool that is ALSO on the
// fast path is marked (the TL008 hazard); a denied tool on no fast path never enters
// the facts, so it cannot pad the surface.
func kernelSurfaceFacts() []ToolFacts {
	facts := FromKernel()
	denied := map[string]bool{}
	for _, t := range adjudicator.Default.DeniedTools() {
		denied[t] = true
	}
	if len(denied) > 0 {
		for i := range facts {
			if denied[facts[i].Name] {
				facts[i].PolicyDenied = true
			}
		}
	}
	return facts
}

func init() {
	// Enroll the invariant in the global registry. It is dormant until a steward
	// sweep drives it (the same way the agentdojo ASR steward is enrolled), and
	// abstains on the clean default surface, so registering it is side-effect-free
	// for a healthy kernel.
	abi.RegisterSteward(surfaceSteward{})
	abi.RegisterCapability("toollint.v1")
}

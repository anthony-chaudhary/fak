package ifc

import (
	"context"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// scopeCeilingBy is the forensics id this gate stamps on every verdict it emits
// (the result-side dual of the call-side residency gate's "engine-residency").
const scopeCeilingBy = "ifc-scope-ceiling"

// ScopeCeilingGate is the result-side ShareScope ceiling — the dual of the
// engine residencyGate. abi/types.go documents a load-bearing invariant for the
// cross-agent shared-result pool: "a result is never shared more widely than its
// scope". That invariant had only a DOWNWARD realization — StampGate clamps a
// TAINTED result's scope down to ScopeAgent (ifc.go). Nothing checked the UPWARD
// bound: a result TAGGED ScopeFleet/ScopeTenant was admitted as-is into whatever
// target received it, with no rung confirming the share target actually sits
// inside that boundary. This gate closes that gap on the kernel's result-admit
// path. It never touches a ScopeAgent (default, private) result, so the common
// case costs a single enum compare and never perturbs the fold.
//
// The share target scope is read from the call Meta tag "share_target" (one of
// agent|fleet|tenant) — the same tag surface the call-side residencyGate reads
// ("sensitivity"/"data_sensitivity"). The gate proves "declared result scope ≤
// declared target scope"; it does NOT verify the target tag is truthful, the
// same trust assumption the residency gate already makes (a caller that mislabels
// its own target tag is outside this gate's reach).
//
// It is registered at rank 21, ABOVE the rank-20 StampGate, so the ceiling folds
// AFTER taint is stamped and the tainted-data down-clamp has already run.
type ScopeCeilingGate struct{}

// NewScopeCeilingGate returns the result-side scope-ceiling gate.
func NewScopeCeilingGate() ScopeCeilingGate { return ScopeCeilingGate{} }

// Caps reports no negotiated capabilities (the gate runs unconditionally on the
// result chain, like StampGate).
func (ScopeCeilingGate) Caps() []abi.Capability { return nil }

// Admit confines a result to its declared ShareScope on the share path.
//
//   - Rung 0 (DEFAULT, no work): a ScopeAgent result (the fail-closed zero value)
//     is private — never shared — so there is nothing to confine. A nil result
//     admits. Returns VerdictAllow (admit-as-is); rank-0 Allow is the fold
//     identity, so this rung never perturbs the most-restrictive outcome.
//   - Rung 1 (boundary confinement): engages ONLY for ScopeFleet/ScopeTenant
//     results. The share is confined iff result scope ≤ declared target scope.
//     A wider result into a narrower target is a boundary crossing →
//     VerdictQuarantine citing ReasonTrustViolation (chosen over VerdictDeny so
//     the rung is enforceable today with zero kernel edits — admitResult already
//     pages out a quarantined result).
//   - Escalation (fail-closed): a wider-than-Agent result with NO readable
//     share_target tag is INDETERMINATE — neither in- nor out-of-bounds can be
//     proved from local values — so it is confined with share_target=unknown.
//
// The witness discloses only the two scopes, never the payload (mirrors the
// residencyGate's bounded disclosure).
func (ScopeCeilingGate) Admit(_ context.Context, c *abi.ToolCall, r *abi.Result) abi.Verdict {
	if !enabled || r == nil {
		return abi.Verdict{Kind: abi.VerdictAllow, By: scopeCeilingBy}
	}
	scope := r.Payload.Scope
	// Rung 0 — the private default. ScopeAgent is never shared; admit as-is.
	if scope == abi.ScopeAgent {
		return abi.Verdict{Kind: abi.VerdictAllow, By: scopeCeilingBy}
	}
	// Rung 1 — only ScopeFleet/ScopeTenant (wider-than-Agent) results engage.
	target, ok := readShareTarget(c)
	if !ok {
		// Escalation: a wider-than-Agent result with an unknowable target is
		// confined (fail-closed), not admitted.
		return abi.Verdict{
			Kind:    abi.VerdictQuarantine,
			Reason:  abi.ReasonTrustViolation,
			By:      scopeCeilingBy,
			Payload: abi.WitnessPayload{Claim: scopeName(scope) + " result shared to an unknown target boundary"},
			Meta:    map[string]string{"share_target": "unknown", "result_scope": scopeName(scope)},
		}
	}
	if scope > target {
		// result scope is WIDER than the share target's declared boundary.
		return abi.Verdict{
			Kind:    abi.VerdictQuarantine,
			Reason:  abi.ReasonTrustViolation,
			By:      scopeCeilingBy,
			Payload: abi.WitnessPayload{Claim: scopeName(scope) + " result shared into a " + scopeName(target) + " target"},
			Meta:    map[string]string{"share_target": scopeName(target), "result_scope": scopeName(scope)},
		}
	}
	// In-bounds: declared result scope ≤ declared target scope. Admit.
	return abi.Verdict{Kind: abi.VerdictAllow, By: scopeCeilingBy}
}

// readShareTarget resolves the call's declared share-target scope from the
// "share_target" Meta tag. The bool is false when no readable tag is present
// (the INDETERMINATE / fail-closed arm).
func readShareTarget(c *abi.ToolCall) (abi.ShareScope, bool) {
	if c == nil || c.Meta == nil {
		return 0, false
	}
	raw, ok := c.Meta["share_target"]
	if !ok {
		return 0, false
	}
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "":
		return 0, false
	case "agent":
		return abi.ScopeAgent, true
	case "fleet":
		return abi.ScopeFleet, true
	case "tenant":
		return abi.ScopeTenant, true
	}
	return 0, false
}

// scopeName renders a ShareScope as its lowercase token (the value space of the
// "share_target" tag and the verdict Meta).
func scopeName(s abi.ShareScope) string {
	switch s {
	case abi.ScopeFleet:
		return "fleet"
	case abi.ScopeTenant:
		return "tenant"
	}
	return "agent"
}

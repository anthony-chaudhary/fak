// adjudicate.go puts the ship gate ON the kernel's decision path: a registered
// Adjudicator that holds a ship-shaped tool call behind the require-witness rung.
//
// The rest of this package (shipgate.go) is the RSI keep-or-revert measurement, a
// driver-blind utility. This file is the in-band complement: when an agent invokes
// a ship/release action, the gate does NOT take the agent's word that the ship is
// safe — it returns VerdictRequireWitness carrying the agent's claimed effect, and
// the kernel corroborates that claim against git evidence (the witness resolver,
// internal/witness) before the call dispatches. An unwitnessed ship — no claim, or
// a claim git refutes — is fail-closed by the kernel to UNWITNESSED / refuted.
//
// This adjudicator is PURE (a tool-name + meta read); it never shells out. The
// git work happens later, in the witness resolver, off the fast decide path —
// exactly the split internal/witness already uses, which is why this package
// (which imports os/exec for the RSI worktree helpers) is NOT on architest's
// hot-path exec-free list.
package shipgate

import (
	"context"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// shipTools are the high-level ship/release actions the gate witness-guards. The
// low-level git mutations a ship is built from (push/merge/tag) are DENIED outright
// by the dev-agent policy floor; THESE are the allowed-but-corroborated actions.
var shipTools = map[string]bool{
	"ship":         true,
	"release":      true,
	"publish":      true,
	"deploy":       true,
	"ship_release": true,
}

// ShipShaped reports whether a tool name is a ship/release action the gate guards.
func ShipShaped(tool string) bool { return shipTools[tool] }

// Gate (the adjudicator) is the registered ship rung. Stateless: the corroboration
// is the witness resolver's job; the gate only routes a ship call to it.
type ShipAdjudicator struct{}

func (ShipAdjudicator) Caps() []abi.Capability { return nil }

// Adjudicate routes a ship-shaped call to the require-witness rung. The claim is
// the effect the caller asserts the ship landed (a git-checkable claim string in
// Meta["witness"], e.g. "ancestor:<ref>" or "clean:."); the kernel hands it to the
// witness resolver. A non-ship call Defers (the gate has no opinion). With no claim
// attached, RequireWitness still fires and the kernel's witness fold abstains, so a
// claim-less ship is fail-closed to UNWITNESSED — exactly the intended posture.
func (ShipAdjudicator) Adjudicate(ctx context.Context, c *abi.ToolCall) abi.Verdict {
	if c == nil || !ShipShaped(c.Tool) {
		return abi.Verdict{Kind: abi.VerdictDefer, By: "shipgate"}
	}
	claim := ""
	if c.Meta != nil {
		claim = c.Meta["witness"]
	}
	return abi.Verdict{
		Kind:    abi.VerdictRequireWitness,
		By:      "shipgate",
		Payload: abi.WitnessPayload{Claim: claim},
		Meta:    map[string]string{"shipgate": "ship", "tool": c.Tool},
	}
}

// DefaultAdjudicator is the registered instance.
var DefaultAdjudicator = ShipAdjudicator{}

func init() {
	// Rank 40: after plancfi (25) / ifc-sink (30), before the rank-100 monitor. The
	// fold takes the most-restrictive verdict, so rank only orders the work — a
	// ship call the monitor would ALLOW is lifted to RequireWitness here.
	abi.RegisterAdjudicator(40, DefaultAdjudicator)
	abi.RegisterCapability("shipgate.v1")
}

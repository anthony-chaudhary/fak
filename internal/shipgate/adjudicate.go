package shipgate

import (
	"context"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

var shipTools = map[string]bool{
	"ship":         true,
	"release":      true,
	"publish":      true,
	"deploy":       true,
	"ship_release": true,
}

// ShipShaped reports whether tool is a guarded release action.
func ShipShaped(tool string) bool { return shipTools[tool] }

// ShipAdjudicator intercepts ship-shaped tool calls to require witness evidence.
type ShipAdjudicator struct{}

// Caps returns the capabilities required by the ship adjudicator.
func (ShipAdjudicator) Caps() []abi.Capability { return nil }

// Adjudicate evaluates tool calls and requires witness corroboration for release actions.
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

// DefaultAdjudicator is the registered singleton instance.
var DefaultAdjudicator = ShipAdjudicator{}

func init() {
	abi.RegisterAdjudicator(40, DefaultAdjudicator)
	abi.RegisterCapability("shipgate.v1")
}

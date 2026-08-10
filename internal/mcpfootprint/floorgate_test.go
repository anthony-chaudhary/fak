package mcpfootprint

import (
	"errors"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// fatTool builds a tool whose schema is deliberately heavy, used to push the measured
// floor over the budget so the growth refusal can be witnessed firing.
func fatTool(name string, bytes int) agent.ToolDef {
	return td(name, strings.Repeat("x", bytes), `{"type":"object"}`)
}

// TestFloorGateRefusesGrowth is the load-bearing witness for #2924: registering one
// more tool schema pushes the always-sent floor past the committed budget and the
// gate REFUSES it. This is the "you can't gate what you don't count" gap closed —
// the refusal is what makes "keep the core narrow" a measurement rather than a taste.
func TestFloorGateRefusesGrowth(t *testing.T) {
	defs := gateway.MCPFloorToolDefs()
	if len(defs) == 0 {
		t.Fatal("fak MCP registry priced as 0 tools — export is not seeing the registry")
	}

	// The real registry, as committed, must pass its own budget.
	if err := CheckFloor(Price(defs)); err != nil {
		t.Fatalf("committed registry fails its own floor gate: %v", err)
	}

	// Add a tool fat enough to clear the budget, and the gate must refuse.
	grown := append(append([]agent.ToolDef(nil), defs...), fatTool("fak_bloat", 4096))
	err := CheckFloor(Price(grown))
	if err == nil {
		t.Fatal("floor gate admitted a registry grown past its budget — growth is ungated")
	}
	var fge *FloorGateError
	if !errors.As(err, &fge) {
		t.Fatalf("want *FloorGateError, got %T: %v", err, err)
	}
	if fge.Reason != ReasonFloorBudgetExceeded {
		t.Fatalf("Reason=%q, want %q", fge.Reason, ReasonFloorBudgetExceeded)
	}
	if fge.Measured <= fge.Budget {
		t.Fatalf("Measured=%d must exceed Budget=%d on a growth refusal", fge.Measured, fge.Budget)
	}
	// The refusal has to tell the author how to justify it, or it just blocks.
	if msg := fge.Error(); !strings.Contains(msg, "FloorBudgetTokens") {
		t.Errorf("growth refusal does not name the constant to raise: %s", msg)
	}
}

// TestFloorGateDemandsBankedWin proves the ratchet only ever tightens: a real
// reduction (deferring cold schemas, #3231/#3232) that is not banked into the
// constant reds the gate, so the recovered headroom cannot be silently refilled.
func TestFloorGateDemandsBankedWin(t *testing.T) {
	// Price a deliberately tiny registry against the committed budget.
	tiny := Price([]agent.ToolDef{td("a", "short", `{"type":"object"}`)})
	err := CheckFloor(tiny)
	if err == nil {
		t.Fatal("floor gate admitted an unbanked reduction — the ratchet does not tighten")
	}
	var fge *FloorGateError
	if !errors.As(err, &fge) {
		t.Fatalf("want *FloorGateError, got %T: %v", err, err)
	}
	if fge.Reason != ReasonFloorBudgetStale {
		t.Fatalf("Reason=%q, want %q", fge.Reason, ReasonFloorBudgetStale)
	}
	if !strings.Contains(fge.Error(), "Re-pin") {
		t.Errorf("stale refusal does not tell the author to re-pin: %s", fge.Error())
	}
}

// TestFloorGateBandBoundaries pins the exact admit/refuse edges, so a future edit to
// the comparison operators cannot quietly widen the band. The band is (budget-slack,
// budget]: the ceiling itself is admissible, one token over is not, and sitting exactly
// slack-below is still admissible.
func TestFloorGateBandBoundaries(t *testing.T) {
	// bytes -> tokens uses the shared 4.5-char JSON-schema divisor, so drive the band with a synthetic footprint.
	fpAt := func(tokens int) agent.Footprint {
		return Price([]agent.ToolDef{td("t", strings.Repeat("x", tokens*9/2), ``)})
	}
	const budget, slack = 100, 10

	for _, tc := range []struct {
		name   string
		tokens int
		want   string // "" == admit
	}{
		{"at the ceiling admits", budget, ""},
		{"one over the ceiling refuses", budget + 1, ReasonFloorBudgetExceeded},
		{"exactly slack below admits", budget - slack, ""},
		{"one past the slack refuses", budget - slack - 1, ReasonFloorBudgetStale},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fp := fpAt(tc.tokens)
			if got := ToolSchemaFootprint(fp); got != tc.tokens {
				t.Fatalf("synthetic footprint priced %d tokens, want %d", got, tc.tokens)
			}
			err := checkFloorAgainst(fp, budget, slack)
			if tc.want == "" {
				if err != nil {
					t.Fatalf("tokens=%d: want admit, got %v", tc.tokens, err)
				}
				return
			}
			var fge *FloorGateError
			if !errors.As(err, &fge) {
				t.Fatalf("tokens=%d: want *FloorGateError %s, got %v", tc.tokens, tc.want, err)
			}
			if fge.Reason != tc.want {
				t.Fatalf("tokens=%d: Reason=%q, want %q", tc.tokens, fge.Reason, tc.want)
			}
		})
	}
}

// TestFloorGateRefusesEmptyRegistry proves the gate fails closed. A registry that
// prices as zero tools means the export stopped seeing the tools; greening on "I
// measured nothing" would be worse than having no gate.
func TestFloorGateRefusesEmptyRegistry(t *testing.T) {
	err := CheckFloor(Price(nil))
	if err == nil {
		t.Fatal("floor gate admitted an empty registry — it greens on measuring nothing")
	}
}

// TestToolSchemaFootprintIsTheToolsBucket binds the summed `tool_schema_footprint`
// metric to the per-tool report a reader sees: the exact per-tool BYTES reconstruct
// the bucket the gate is denominated in. (Per-tool Tokens are each floored, so they
// are only guaranteed to sum to <= the metric — asserted as an inequality, not an
// equality, so a rounding change surfaces here rather than in a false budget.)
func TestToolSchemaFootprintIsTheToolsBucket(t *testing.T) {
	fp := Price(gateway.MCPFloorToolDefs())

	if got := ToolSchemaFootprint(fp); got != fp.Tools.Tokens {
		t.Fatalf("ToolSchemaFootprint=%d, Tools.Tokens=%d", got, fp.Tools.Tokens)
	}

	sumBytes, sumTokens := 0, 0
	for _, pt := range fp.PerTool {
		sumBytes += pt.Bytes
		sumTokens += pt.Tokens
	}
	if sumBytes != fp.Tools.Bytes {
		t.Fatalf("per-tool bytes sum=%d != Tools.Bytes=%d — the metric double-counts", sumBytes, fp.Tools.Bytes)
	}
	if sumTokens > ToolSchemaFootprint(fp) {
		t.Fatalf("sum of floored per-tool tokens (%d) exceeds the metric (%d)", sumTokens, ToolSchemaFootprint(fp))
	}
}

// TestCommittedBudgetMatchesMeasuredFloor proves FloorBudgetTokens is a measurement,
// not a hand-typed number that drifted from the registry. It is the same assertion
// the docs/context-budget/mcp-tool-floor.md baseline table makes, enforced in code.
func TestCommittedBudgetMatchesMeasuredFloor(t *testing.T) {
	fp := Price(gateway.MCPFloorToolDefs())
	measured := ToolSchemaFootprint(fp)
	if measured > FloorBudgetTokens || measured < FloorBudgetTokens-FloorRatchetSlackTokens {
		t.Fatalf("committed FloorBudgetTokens=%d is outside the ratchet band for the measured floor of %d est. tokens (%d tools). %v",
			FloorBudgetTokens, measured, fp.ToolCount, CheckFloor(fp))
	}
	t.Logf("gated floor: %d tools, %d est. tokens (budget %d, slack %d, %s)",
		fp.ToolCount, measured, FloorBudgetTokens, FloorRatchetSlackTokens, fp.Provenance)
}

package mcpfootprint

import (
	"errors"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// fatDescTool builds a tool whose DESCRIPTION alone is deliberately heavy, used to push
// the measured description sum over the budget so the growth refusal can be witnessed.
func fatDescTool(name string, bytes int) agent.ToolDef {
	return td(name, strings.Repeat("x", bytes), `{"type":"object"}`)
}

// TestDescriptionBudgetPassesAtHEAD is acceptance criterion 1: a fak footprint-backed
// test asserts the total always-shipped fak_* description tokens are under the declared
// budget. It prices the ACTUAL MCP registry (the same export fak footprint reads) and
// runs the committed gate against it — the registry as shipped must pass its own budget.
func TestDescriptionBudgetPassesAtHEAD(t *testing.T) {
	defs := gateway.MCPFloorToolDefs()
	if len(defs) == 0 {
		t.Fatal("fak MCP registry priced as 0 tools — export is not seeing the registry")
	}
	measured := DescriptionTokens(defs)
	t.Logf("always-sent fak_* description floor: %d est. tokens across %d tools (budget %d, slack %d)",
		measured, len(defs), DescriptionBudgetTokens, DescriptionRatchetSlackTokens)
	top := PerToolDescription(defs)
	for i := 0; i < min(6, len(top)); i++ {
		t.Logf("  top %d: %4d tok  %s", i+1, top[i].Tokens, top[i].Name)
	}
	if err := CheckDescriptions(defs); err != nil {
		t.Fatalf("committed registry fails its own description budget: %v", err)
	}
}

// TestDescriptionBudgetRefusesGrowth is acceptance criterion 2 (the fail half): a
// synthetic over-budget description edit trips the gate. Adding one tool whose
// description alone clears the budget must refuse with DESC_BUDGET_EXCEEDED.
func TestDescriptionBudgetRefusesGrowth(t *testing.T) {
	defs := gateway.MCPFloorToolDefs()
	if len(defs) == 0 {
		t.Fatal("fak MCP registry priced as 0 tools")
	}
	grown := append(append([]agent.ToolDef(nil), defs...), fatDescTool("fak_verbose", 4096))
	err := CheckDescriptions(grown)
	if err == nil {
		t.Fatal("description budget admitted a registry grown past its ceiling — verbosity is ungated")
	}
	var dbe *DescBudgetError
	if !errors.As(err, &dbe) {
		t.Fatalf("want *DescBudgetError, got %T: %v", err, err)
	}
	if dbe.Reason != ReasonDescBudgetExceeded {
		t.Fatalf("Reason=%q, want %q", dbe.Reason, ReasonDescBudgetExceeded)
	}
	if dbe.Measured <= dbe.Budget {
		t.Fatalf("Measured=%d must exceed Budget=%d on a growth refusal", dbe.Measured, dbe.Budget)
	}
	// The refusal must tell the author how to justify it, or it just blocks.
	if msg := dbe.Error(); !strings.Contains(msg, "DescriptionBudgetTokens") {
		t.Errorf("growth refusal does not name the constant to raise: %s", msg)
	}
}

// TestDescriptionBudgetDemandsBankedWin proves the ratchet only ever tightens: a real
// trim that is not banked into the constant reds the gate as STALE, so the recovered
// headroom cannot be silently refilled by future verbosity.
func TestDescriptionBudgetDemandsBankedWin(t *testing.T) {
	tiny := []agent.ToolDef{td("fak_a", "short", `{"type":"object"}`)}
	err := CheckDescriptions(tiny)
	if err == nil {
		t.Fatal("description budget admitted an unbanked trim — the ratchet does not tighten")
	}
	var dbe *DescBudgetError
	if !errors.As(err, &dbe) {
		t.Fatalf("want *DescBudgetError, got %T: %v", err, err)
	}
	if dbe.Reason != ReasonDescBudgetStale {
		t.Fatalf("Reason=%q, want %q", dbe.Reason, ReasonDescBudgetStale)
	}
	if !strings.Contains(dbe.Error(), "Re-pin") {
		t.Errorf("stale refusal does not tell the author to re-pin: %s", dbe.Error())
	}
}

// TestDescriptionBudgetBandBoundaries pins the exact admit/refuse edges so a future edit
// to the comparison operators cannot quietly widen the band. The band is (budget-slack,
// budget]: the ceiling itself admits, one token over refuses, sitting exactly slack-below
// still admits, one past the slack refuses as stale.
func TestDescriptionBudgetBandBoundaries(t *testing.T) {
	// bytes -> tokens uses the shared 4.5-char JSON-schema divisor, so drive the band with a single description of
	// known length: a description of tokens*4.5 bytes prices to exactly tokens.
	defsAt := func(tokens int) []agent.ToolDef {
		return []agent.ToolDef{td("t", strings.Repeat("x", tokens*9/2), ``)}
	}
	const budget, slack = 100, 10
	for _, tc := range []struct {
		name   string
		tokens int
		want   string // "" == admit
	}{
		{"at the ceiling admits", budget, ""},
		{"one over the ceiling refuses", budget + 1, ReasonDescBudgetExceeded},
		{"exactly slack below admits", budget - slack, ""},
		{"one past the slack refuses", budget - slack - 1, ReasonDescBudgetStale},
	} {
		t.Run(tc.name, func(t *testing.T) {
			defs := defsAt(tc.tokens)
			if got := DescriptionTokens(defs); got != tc.tokens {
				t.Fatalf("synthetic description priced %d tokens, want %d", got, tc.tokens)
			}
			err := checkDescriptionsAgainst(defs, budget, slack)
			if tc.want == "" {
				if err != nil {
					t.Fatalf("tokens=%d: want admit, got %v", tc.tokens, err)
				}
				return
			}
			var dbe *DescBudgetError
			if !errors.As(err, &dbe) {
				t.Fatalf("tokens=%d: want *DescBudgetError %s, got %v", tc.tokens, tc.want, err)
			}
			if dbe.Reason != tc.want {
				t.Fatalf("tokens=%d: Reason=%q, want %q", tc.tokens, dbe.Reason, tc.want)
			}
		})
	}
}

// TestDescriptionBudgetFailsClosed proves the gate refuses an empty registry rather than
// greening on "I measured nothing" — the same fail-closed posture as the floor gate.
func TestDescriptionBudgetFailsClosed(t *testing.T) {
	if err := CheckDescriptions(nil); err == nil {
		t.Fatal("description budget admitted an empty registry — it greens on measuring nothing")
	}
}

// TestDescriptionIsDescriptionOnly proves the priced number is the DESCRIPTION slice,
// not the whole schema: it must equal the summed description bytes floored, and it must
// be strictly less than the full schema floor (which also carries names + parameters).
func TestDescriptionIsDescriptionOnly(t *testing.T) {
	defs := gateway.MCPFloorToolDefs()
	descBytes := 0
	for _, d := range defs {
		descBytes += len(d.Function.Description)
	}
	if got := DescriptionFootprint(defs).Tools.Bytes; got != descBytes {
		t.Fatalf("description bytes = %d, want summed len(Description) = %d", got, descBytes)
	}
	schemaTokens := ToolSchemaFootprint(Price(defs))
	descTokens := DescriptionTokens(defs)
	if descTokens >= schemaTokens {
		t.Fatalf("description tokens (%d) must be < full schema tokens (%d) — names+params are unpriced here",
			descTokens, schemaTokens)
	}
}

// TestCommittedDescriptionBudgetMatchesMeasured proves DescriptionBudgetTokens is a
// measurement, not a hand-typed number that drifted from the registry: the committed
// ceiling must sit inside the ratchet band of the real measured description floor.
func TestCommittedDescriptionBudgetMatchesMeasured(t *testing.T) {
	measured := DescriptionTokens(gateway.MCPFloorToolDefs())
	if measured > DescriptionBudgetTokens || measured < DescriptionBudgetTokens-DescriptionRatchetSlackTokens {
		t.Fatalf("committed DescriptionBudgetTokens=%d is outside the ratchet band for the measured floor of %d est. tokens. %v",
			DescriptionBudgetTokens, measured, CheckDescriptions(gateway.MCPFloorToolDefs()))
	}
}

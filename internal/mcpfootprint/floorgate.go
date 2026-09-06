package mcpfootprint

// floorgate.go — the tool-schema footprint BUDGET and floor-growth gate (#2924).
//
// #3230 landed the MEASUREMENT: Price() reports what each registered tool's schema
// costs on every API call, and PerToolSorted() ranks them. Measurement alone does not
// keep the core narrow — "every tool ships on every API call" is still enforced by
// taste unless a number can REFUSE a change. This file is the refusal.
//
// The rule is a one-way ratchet on the always-sent floor:
//
//   - GROWTH is refused (FLOOR_BUDGET_EXCEEDED). Adding a tool, or fattening an
//     existing tool's description/JSON-Schema, pushes the measured floor past the
//     committed FloorBudgetTokens ceiling and reds the gate. The ONLY way through is
//     to raise the constant in the SAME commit — which is exactly the justification
//     the issue asks for: a reviewer sees the new tax as a diff line, bound to the
//     change that caused it, instead of discovering it a quarter later.
//
//   - A BANKED WIN is required (FLOOR_BUDGET_STALE). When a deferral lever (#3231
//     cold-schema defer, #3232 the gateway lever) drives the floor well BELOW the
//     ceiling, the gate also reds until the ceiling is re-pinned down. Otherwise the
//     slack silently becomes headroom for future bloat and the ratchet stops
//     ratcheting. This is the same discipline internal/pythongate applies to the
//     tools/*.py baseline: it only ever tightens.
//
// The budget is denominated in ESTIMATED tokens (agent.FootprintProvenance) because
// the pricing is the deterministic, offline ~4-char/token house estimator — the same
// walk as EstimateAnthropicTokens, so the gated number can never drift from the one
// `fak footprint` prints. It is NOT a provider-billed count and must never be
// compared against one (Law A2: every value carries its provenance).

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// FloorBudgetTokens is the committed ceiling on fak's always-sent MCP tool-schema
// floor, in ESTIMATED tokens. It is the measured baseline pinned in
// docs/context-budget/mcp-tool-floor.md (21 tools · 4930 est. tokens · 22187 bytes).
//
// Changing this number is the whole point: it is the reviewable justification for a
// change to the per-call floor. Raise it only alongside the tool that grew it; lower
// it whenever a deferral banks a win. Re-pinned 19→21 tools (#11644/#11648) after
// context restore and chat tools were added.
const FloorBudgetTokens = 4930

// FloorRatchetSlackTokens is how far the measured floor may sit BELOW the budget
// before the gate demands the ceiling be re-pinned. It absorbs incidental churn (a
// reworded description) without nagging, while still forcing a real reduction to be
// banked into the constant. ~5.5% of the current floor.
const FloorRatchetSlackTokens = 250

// Gate refusal reasons. These are the closed-vocabulary tokens the gate names, in the
// same spirit as the dos.toml [reasons.*] blocks a guard refusal cites.
const (
	ReasonFloorBudgetExceeded = "FLOOR_BUDGET_EXCEEDED"
	ReasonFloorBudgetStale    = "FLOOR_BUDGET_STALE"
)

// ToolSchemaFootprint is the summed per-call token cost of every registered tool
// schema — the `tool_schema_footprint` metric #2924 asks for. It is the Tools bucket
// of a priced footprint: the floor of (sum of per-tool BYTES)/4.
//
// Note it is derived from the exact byte sum, NOT by adding the per-tool Tokens
// fields. Those are each floored independently, so they can sum to slightly under
// this value (floor-of-sum >= sum-of-floors). Bytes are the additive quantity; the
// gate is denominated in the single floored token total so the budget is one number
// with no rounding slop.
func ToolSchemaFootprint(fp agent.Footprint) int { return fp.Tools.Tokens }

// FloorGateError is a structured floor-gate refusal: which direction the floor moved,
// what was measured, and what the committed ceiling says. Callers (the gate test, and
// any future CI shell) can branch on Reason rather than string-matching a message.
type FloorGateError struct {
	Reason    string // ReasonFloorBudgetExceeded | ReasonFloorBudgetStale
	Measured  int    // the priced floor, in ESTIMATED tokens
	Budget    int    // the committed ceiling (FloorBudgetTokens)
	ToolCount int    // how many tool schemas were priced
}

func (e *FloorGateError) Error() string {
	switch e.Reason {
	case ReasonFloorBudgetExceeded:
		return fmt.Sprintf(
			"%s: the always-sent tool-schema floor grew to %d est. tokens across %d tools, over the committed budget of %d "+
				"(+%d). Every registered tool's schema ships on EVERY API call, so this is a per-turn tax paid forever. "+
				"Justify it by raising mcpfootprint.FloorBudgetTokens to %d in the same commit (and re-pin the baseline table "+
				"in docs/context-budget/mcp-tool-floor.md), or shrink the schema — prefer deferring a cold tool over paying the floor.",
			e.Reason, e.Measured, e.ToolCount, e.Budget, e.Measured-e.Budget, e.Measured)
	case ReasonFloorBudgetStale:
		return fmt.Sprintf(
			"%s: the tool-schema floor fell to %d est. tokens across %d tools, %d below the committed budget of %d — "+
				"a reduction was won but never banked. Re-pin mcpfootprint.FloorBudgetTokens to %d so the ratchet tightens "+
				"and the slack cannot be silently refilled by future bloat.",
			e.Reason, e.Measured, e.ToolCount, e.Budget-e.Measured, e.Budget, e.Measured)
	default:
		return fmt.Sprintf("%s: floor %d est. tokens vs budget %d", e.Reason, e.Measured, e.Budget)
	}
}

// CheckFloor gates a priced footprint against the committed budget. It returns nil
// when the measured floor sits inside the band [Budget-Slack, Budget], and a
// *FloorGateError naming the direction otherwise.
//
// The gate fails CLOSED on a broken measurement: an empty registry prices as 0
// tokens, which lands far under the ratchet floor and refuses as STALE rather than
// passing vacuously. A gate that greens on "I measured nothing" is worse than no gate.
func CheckFloor(fp agent.Footprint) error {
	return checkFloorAgainst(fp, FloorBudgetTokens, FloorRatchetSlackTokens)
}

// checkFloorAgainst is CheckFloor with the budget and slack injected, so a test can
// drive both refusal directions without editing the committed constant.
func checkFloorAgainst(fp agent.Footprint, budget, slack int) error {
	measured := ToolSchemaFootprint(fp)
	if measured > budget {
		return &FloorGateError{
			Reason:    ReasonFloorBudgetExceeded,
			Measured:  measured,
			Budget:    budget,
			ToolCount: fp.ToolCount,
		}
	}
	if measured < budget-slack {
		return &FloorGateError{
			Reason:    ReasonFloorBudgetStale,
			Measured:  measured,
			Budget:    budget,
			ToolCount: fp.ToolCount,
		}
	}
	return nil
}

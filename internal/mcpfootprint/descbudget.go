package mcpfootprint

// descbudget.go — the always-shipped fak_* tool DESCRIPTION token budget and gate (#3608).
//
// floorgate.go gates the WHOLE tool-schema floor (name + description + JSON-Schema
// parameters). This file gates the narrower, purely-prose slice #3608 names: the
// multi-paragraph tool DESCRIPTIONS. A description is the one part of a tool schema
// that has no machine consumer — the JSON-Schema parameters are validated, the name is
// dispatched on, but the description is prompt-prefix prose the model reads, so it is
// the slice most prone to silently fattening into a per-turn tax. #3231 defers the COLD
// schemas; this keeps the HOT (always-sent) descriptions lean, and the two compose.
//
// The gate is the same one-way ratchet floorgate.go applies to the whole schema:
//
//   - GROWTH is refused (DESC_BUDGET_EXCEEDED). Fattening a description (or adding a
//     tool whose description pushes the sum over the ceiling) reds the gate. The only
//     way through is to raise DescriptionBudgetTokens in the SAME commit — the reviewable
//     diff line that binds the new prose tax to the change that caused it.
//
//   - A BANKED WIN is required (DESC_BUDGET_STALE). Trimming the hot descriptions well
//     below the ceiling reds the gate until the ceiling is re-pinned down, so the
//     recovered headroom cannot be silently refilled by future verbosity. Same
//     discipline as internal/pythongate: it only ever tightens.
//
// The number is denominated in ESTIMATED tokens because it is priced through the SAME
// agent.RequestFootprint estimator mcpfootprint.Price uses (a description-only ToolDef
// carries no name or parameter bytes), so the gated number can never drift from the
// `fak footprint` floor or from EstimateAnthropicTokens. It is NOT a provider-billed
// count and must never be compared against one (Law A2: every value carries provenance).

import (
	"fmt"
	"sort"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// DescriptionBudgetTokens is the committed ceiling on the SUM of always-shipped fak_*
// MCP tool DESCRIPTION tokens, in ESTIMATED tokens. It is the measured baseline pinned
// in docs/context-budget/mcp-tool-floor.md.
//
// Changing this number is the whole point: it is the reviewable justification for a
// change to the per-call prose tax. Raise it only alongside the description that grew
// it; lower it whenever a trim banks a win. Pinned at the measured HEAD floor: 19 tools
// · 1552 est. description tokens (see docs/context-budget/mcp-tool-floor.md). Re-pinned
// 26→19 tools (#6011) after #6022 retired the repository index MCP tools.
const DescriptionBudgetTokens = 1552

// DescriptionRatchetSlackTokens is how far the measured description sum may sit BELOW
// the budget before the gate demands the ceiling be re-pinned. It absorbs incidental
// churn (a reworded sentence) without nagging, while still forcing a real reduction to
// be banked into the constant.
const DescriptionRatchetSlackTokens = 200

// Gate refusal reasons — closed-vocabulary tokens, in the same spirit as the
// floorgate.go FLOOR_BUDGET_* reasons and the dos.toml [reasons.*] blocks.
const (
	ReasonDescBudgetExceeded = "DESC_BUDGET_EXCEEDED"
	ReasonDescBudgetStale    = "DESC_BUDGET_STALE"
)

// descOnlyDefs strips each tool to its description alone (blank name, no parameters) so
// the SHARED estimator prices only the prose body — the same char-walk and ~4-char/token
// divisor as EstimateAnthropicTokens, never a second estimator.
func descOnlyDefs(tools []agent.ToolDef) []agent.ToolDef {
	out := make([]agent.ToolDef, 0, len(tools))
	for _, t := range tools {
		out = append(out, agent.ToolDef{
			Type:     "function",
			Function: agent.ToolDefFunction{Description: t.Function.Description},
		})
	}
	return out
}

// DescriptionFootprint prices the description-only floor: an agent.Footprint whose Tools
// bucket is the summed always-sent description cost, priced through mcpfootprint.Price.
func DescriptionFootprint(tools []agent.ToolDef) agent.Footprint {
	return Price(descOnlyDefs(tools))
}

// DescriptionTokens is the summed always-sent fak_* description cost in ESTIMATED tokens
// — the single number the budget gates. Derived from the exact byte sum (floor of the
// total), not by adding per-tool floors, so there is no rounding slop in the gated value.
func DescriptionTokens(tools []agent.ToolDef) int {
	return DescriptionFootprint(tools).Tools.Tokens
}

// PerToolDescription ranks each tool's description-only cost largest-first (ties broken
// by name), with names preserved — the "where the prose went" view a trimmer reads
// first before cutting. It is the VIEW half of record -> view -> gate.
func PerToolDescription(tools []agent.ToolDef) []agent.ToolFootprint {
	out := make([]agent.ToolFootprint, 0, len(tools))
	for _, t := range tools {
		tf := agent.ToolFootprint{Name: t.Function.Name}
		if fp := Price(descOnlyDefs([]agent.ToolDef{t})); len(fp.PerTool) == 1 {
			tf.Bytes, tf.Tokens = fp.PerTool[0].Bytes, fp.PerTool[0].Tokens
		}
		out = append(out, tf)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Bytes != out[j].Bytes {
			return out[i].Bytes > out[j].Bytes
		}
		return out[i].Name < out[j].Name
	})
	return out
}

// DescBudgetError is a structured description-budget refusal: which direction the sum
// moved, what was measured, and what the committed ceiling says. Callers branch on
// Reason rather than string-matching the message.
type DescBudgetError struct {
	Reason    string // ReasonDescBudgetExceeded | ReasonDescBudgetStale
	Measured  int    // the priced description sum, in ESTIMATED tokens
	Budget    int    // the committed ceiling (DescriptionBudgetTokens)
	ToolCount int    // how many tool descriptions were priced
}

func (e *DescBudgetError) Error() string {
	switch e.Reason {
	case ReasonDescBudgetExceeded:
		return fmt.Sprintf(
			"%s: the always-sent fak_* tool descriptions grew to %d est. tokens across %d tools, over the committed budget of %d "+
				"(+%d). Every registered tool's description ships on EVERY API call, so this prose is a per-turn tax paid forever. "+
				"Justify it by raising mcpfootprint.DescriptionBudgetTokens to %d in the same commit (and re-pin the baseline in "+
				"docs/context-budget/mcp-tool-floor.md), or trim the description — the parameters are validated and the name is "+
				"dispatched on, but the description is prose no machine reads.",
			e.Reason, e.Measured, e.ToolCount, e.Budget, e.Measured-e.Budget, e.Measured)
	case ReasonDescBudgetStale:
		return fmt.Sprintf(
			"%s: the fak_* tool descriptions fell to %d est. tokens across %d tools, %d below the committed budget of %d — "+
				"a trim was won but never banked. Re-pin mcpfootprint.DescriptionBudgetTokens to %d so the ratchet tightens "+
				"and the slack cannot be silently refilled by future verbosity.",
			e.Reason, e.Measured, e.ToolCount, e.Budget-e.Measured, e.Budget, e.Measured)
	default:
		return fmt.Sprintf("%s: descriptions %d est. tokens vs budget %d", e.Reason, e.Measured, e.Budget)
	}
}

// CheckDescriptions gates the always-sent tool descriptions against the committed
// budget. It returns nil when the measured sum sits inside the band
// [Budget-Slack, Budget], and a *DescBudgetError naming the direction otherwise.
//
// The gate fails CLOSED on a broken measurement: an empty registry prices as 0 tokens,
// which lands far under the ratchet floor and refuses as STALE rather than passing
// vacuously. A gate that greens on "I measured nothing" is worse than no gate.
func CheckDescriptions(tools []agent.ToolDef) error {
	return checkDescriptionsAgainst(tools, DescriptionBudgetTokens, DescriptionRatchetSlackTokens)
}

// checkDescriptionsAgainst is CheckDescriptions with the budget and slack injected, so a
// test can drive both refusal directions without editing the committed constant.
func checkDescriptionsAgainst(tools []agent.ToolDef, budget, slack int) error {
	measured := DescriptionTokens(tools)
	if measured > budget {
		return &DescBudgetError{Reason: ReasonDescBudgetExceeded, Measured: measured, Budget: budget, ToolCount: len(tools)}
	}
	if measured < budget-slack {
		return &DescBudgetError{Reason: ReasonDescBudgetStale, Measured: measured, Budget: budget, ToolCount: len(tools)}
	}
	return nil
}

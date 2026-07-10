// budget.go — the per-task budget readout: what the current task has spent
// against a soft target, broken down by category, from real usage records
// (#2091). The ask is a running "you've spent N, budget M, here is where it
// went" signal an agent can query mid-task, so it can deliberately trade
// thoroughness against cost instead of discovering it over-explored (or
// under-verified) only in retrospect.
//
// This file owns the PURE shape + fold, the same split spend.go uses: the CLI
// (cmd/fak/budget.go) reads the real gateway-usage ledger rows and hands this
// package a plain BudgetSpend snapshot, so internal/metrics stays a leaf that
// imports no ledger package.
//
// Provenance discipline (same as spend.go's conflation fence): every category
// names WHO AUTHORED its number. Model token totals are provider-relayed, so
// they are OBSERVED; fak's own kernel/harness counts (served turns, adjudicated
// tool calls) are fak-authored, so they are WITNESSED. Arithmetic never upgrades
// authorship, and a category missing its label fails the gate.
package metrics

import "fmt"

// BudgetReadoutSchema stamps the JSON envelope so a reader can bind the shape.
const BudgetReadoutSchema = "fak-task-budget-readout/1"

// BudgetSpend is the plain per-task spend snapshot the CLI extracts from the
// current task's latest gateway-usage ledger row (gatewayusageledger.Counters).
// Kept as plain scalars so this package imports no ledger type — the CLI owns
// the disk read and the mapping.
type BudgetSpend struct {
	// InputTokens / OutputTokens are the provider-relayed model prompt/completion
	// token totals — the two axes that sum to task token spend.
	InputTokens  uint64
	OutputTokens uint64
	// CachedTokens is prompt tokens served from cache: a SAVING, reported beside
	// spend but never added to it (a cached read is cheaper, not extra spend).
	CachedTokens uint64
	// Turns is the served-turn count fak's harness coordinator observed for the task.
	Turns uint64
	// ToolCalls is the count of tool calls fak's kernel adjudicated for the task.
	ToolCalls uint64
}

// BudgetTarget is the operator's SOFT target for the task. A zero field means
// "no target on this axis" — the readout still reports spend, it just cannot
// report a remaining/percent for that axis. Nothing enforces the target; it is a
// self-pacing signal, not a cap.
type BudgetTarget struct {
	Tokens uint64 // soft target for total spend tokens (input+output)
	Turns  uint64 // soft target for served turns
}

// BudgetCategory is one labeled line of the "here is where it went" breakdown.
// It reuses SpendProvenance so a budget category and a spend figure carry the
// same closed WITNESSED/OBSERVED authorship vocabulary.
type BudgetCategory struct {
	Name       string          `json:"name"`
	Spent      uint64          `json:"spent"`
	Unit       string          `json:"unit"`
	Provenance SpendProvenance `json:"provenance"`
	Note       string          `json:"note,omitempty"`
}

// BudgetAxis is spend-vs-target for one axis (tokens or turns). Remaining is
// signed so an over-budget task reads negative rather than silently clamping.
type BudgetAxis struct {
	Spent       uint64  `json:"spent"`
	Target      uint64  `json:"target,omitempty"` // 0 = no target set on this axis
	HasTarget   bool    `json:"has_target"`
	Remaining   int64   `json:"remaining"`    // target - spent; negative once over
	PercentUsed float64 `json:"percent_used"` // spent/target*100; 0 when no target
	Over        bool    `json:"over"`         // spent exceeds a set target
}

// BudgetReadout is the full per-task readout `fak budget` emits.
type BudgetReadout struct {
	Schema     string           `json:"schema"`
	Session    string           `json:"session,omitempty"`
	Tokens     BudgetAxis       `json:"tokens"`
	Turns      BudgetAxis       `json:"turns"`
	Categories []BudgetCategory `json:"categories"`
	// CachedTokens is reported alongside as a saving, not counted as spend.
	CachedTokens uint64 `json:"cached_tokens"`
	Note         string `json:"note,omitempty"`
}

// foldAxis derives the spend-vs-target view for one axis. A zero target leaves
// HasTarget false and every derived field zero, so the readout reports the raw
// spend without inventing a percentage against a target that was never set.
func foldAxis(spent, target uint64) BudgetAxis {
	a := BudgetAxis{Spent: spent}
	if target > 0 {
		a.HasTarget = true
		a.Target = target
		a.Remaining = int64(target) - int64(spent)
		a.PercentUsed = float64(spent) / float64(target) * 100
		a.Over = spent > target
	}
	return a
}

// FoldBudget is the PURE fold: a per-task spend snapshot + a soft target in, the
// labeled readout out. Deterministic — no I/O, no clock, no wall-time read.
// Token spend is input+output (cached prompt tokens are reported separately as a
// saving, never added to spend). session labels which task the numbers belong to.
func FoldBudget(session string, spend BudgetSpend, target BudgetTarget) BudgetReadout {
	totalTokens := spend.InputTokens + spend.OutputTokens
	return BudgetReadout{
		Schema:       BudgetReadoutSchema,
		Session:      session,
		Tokens:       foldAxis(totalTokens, target.Tokens),
		Turns:        foldAxis(spend.Turns, target.Turns),
		CachedTokens: spend.CachedTokens,
		Categories: []BudgetCategory{
			{Name: "model_input_tokens", Spent: spend.InputTokens, Unit: "tokens", Provenance: SpendObserved,
				Note: "provider-relayed prompt tokens"},
			{Name: "model_output_tokens", Spent: spend.OutputTokens, Unit: "tokens", Provenance: SpendObserved,
				Note: "provider-relayed completion tokens"},
			{Name: "cached_prompt_tokens", Spent: spend.CachedTokens, Unit: "tokens", Provenance: SpendObserved,
				Note: "prompt tokens served from cache — a saving, not counted in token spend"},
			{Name: "served_turns", Spent: spend.Turns, Unit: "turns", Provenance: SpendWitnessed,
				Note: "turns fak's harness coordinator served this task"},
			{Name: "tool_calls", Spent: spend.ToolCalls, Unit: "calls", Provenance: SpendWitnessed,
				Note: "tool calls fak's kernel adjudicated this task"},
		},
		Note: "categories are the coarse model/turn/tool axes the gateway-usage ledger carries; a finer per-tool reads/edits split needs the decision-journal by-kind sink (follow-on)",
	}
}

// GateBudgetLabeled is the unlabeled-category gate, mirroring GateSpendLabeled:
// it returns one defect per budget category missing its unit or carrying a
// provenance outside the closed WITNESSED/OBSERVED set. An empty result is the
// only pass; the CLI treats any defect as a hard failure so an unlabeled
// category can never reach an operator as if it were witnessed.
func GateBudgetLabeled(r BudgetReadout) []string {
	var defects []string
	for i, c := range r.Categories {
		id := fmt.Sprintf("category %d (%s)", i, c.Name)
		if c.Unit == "" {
			defects = append(defects, id+": budget category carries no unit")
		}
		switch c.Provenance {
		case SpendWitnessed, SpendObserved:
		default:
			defects = append(defects, id+": budget category is unlabeled — provenance "+
				fmt.Sprintf("%q", string(c.Provenance))+" is not WITNESSED or OBSERVED")
		}
	}
	return defects
}

package agent

import (
	"sync"

	"github.com/anthony-chaudhary/fak/internal/ctxplan"
)

// inkernel_turntax.go — issue #1538: make the in-kernel turn path RECORD the per-turn cache
// decision it was already taking implicitly.
//
// Before this seam, which of the three per-turn strategies a turn took was decided by whichever
// branch of generateReusedContextWithBias happened to run — the radix prefix match served the
// turn (REUSE) or it did not and the whole prompt was prefilled (COLD PREFILL) — and nothing
// anywhere recorded WHY. The strategy was real but implicit, so no operator surface could
// answer "what did this turn do about the cache, and what did that choice cost?".
//
// ctxplan.PlanTurnTax (internal/ctxplan/turntax.go) is the deterministic chooser; this file is
// the DEFAULT wiring of it onto the live decode path. Every turn that reaches the decode seam
// now appends exactly one closed-vocabulary decision + reason to a planner-scoped ledger.
//
// WHERE IT RUNS, AND WHY THAT MAKES IT A PLANNER. recordTurnTax is called at the point where the
// lookup and every servability/trust gate have settled but BEFORE any prefill or decode compute
// happens. That ordering is load-bearing: the decision is made from signals known ahead of the
// work, so this is a planner, not a post-hoc attribution of work already done. (Attribution of
// the realized turn stays where it was — the cacheobs taps and the cachemeta provider record in
// inkernel_planner.go — and is deliberately NOT merged with this.)
//
// PROVENANCE. This ledger is the KERNEL-local context decision: fak's own radix KV prefix tree
// on this box. It is not provider prefix-cache attribution (cachemeta / the provider_prefix
// axis), not the context-plan budget model, and not forecast provenance. A surface reporting
// several of those keeps them on separate axes.

// inKernelTurnTaxState is the planner-scoped per-turn cache-decision ledger. It is embedded in
// InKernelPlanner (matching the moeResidencyState idiom) because the decision is made inside a
// session this planner builds and closes PER REQUEST: without planner-scoped storage every
// turn's decision would be destroyed at that teardown and no serve surface could ever see it.
//
// The zero value is a usable empty ledger, so a planner built by any constructor records from
// its first turn with no initialization step to forget.
type inKernelTurnTaxState struct {
	turnTaxMu  sync.Mutex
	turnTaxLog ctxplan.TurnTaxLog
}

// recordTurnTax decides and records this turn's cache strategy, returning the decision.
//
// Signal mapping from the decode seam:
//
//   - promptTok    → PromptTokens: the whole prompt, the cold-prefill tax.
//   - cacheable    → MatchedPrefix: the LOOKUP-side radix match (the #3390 cacheability half),
//     recorded pre-gate so a prefix that matched but could not be served stays visible instead
//     of being folded into a plain miss.
//   - served > 0   → PrefixTrusted: whether the servability/trust gates actually let this turn
//     decode from that match. cacheable > 0 with served == 0 is exactly the refused-prefix case,
//     and the planner books it as COLD PREFILL with the refusal named in the reason — the
//     cold-path correctness this issue requires to stay explicit.
//
// QueryTokens is deliberately a structural 0 here, mirroring the honest structural zero the
// external-KV-transfer bucket keeps in inkernel_planner.go: the O(1) session query
// (internal/contextq, ctxplan's Index.ProbePlan) is NOT reachable from this decode seam, so on
// this path the planner genuinely chooses between reuse and cold prefill. Claiming a query
// candidate the path cannot execute would make the ledger a wish rather than a record. When a
// query tier is wired to this seam it passes its estimate here and the query branch — already
// implemented and tested in ctxplan — starts winning turns with no change to this call.
//
// One known approximation, recorded here rather than hidden: on an exact-duplicate transcript
// the seam refeeds the final token, so the realized serve is one token shorter than `cacheable`.
// The ledger books the lookup-side number; a single token cannot change which strategy wins, and
// the pre-gate value is the one that keeps a refused match visible.
func (p *InKernelPlanner) recordTurnTax(promptTok, cacheable, served int) ctxplan.TurnTaxDecision {
	p.turnTaxMu.Lock()
	defer p.turnTaxMu.Unlock()
	return p.turnTaxLog.Append(ctxplan.TurnTaxSignals{
		PromptTokens:  promptTok,
		MatchedPrefix: cacheable,
		PrefixTrusted: served > 0,
	})
}

// TurnTaxDecisions returns this planner's recorded per-turn cache decisions, newest last, as a
// defensive copy. Each entry carries both the signals and the decision, so a caller can replay
// the ledger and re-derive every choice without any other state.
func (p *InKernelPlanner) TurnTaxDecisions() []ctxplan.TurnTaxLogEntry {
	p.turnTaxMu.Lock()
	defer p.turnTaxMu.Unlock()
	return p.turnTaxLog.Entries()
}

// TurnTaxSummary folds the retained window into per-strategy counts plus the token taxes behind
// them — the O(1) readout a serve surface prints instead of walking every turn.
func (p *InKernelPlanner) TurnTaxSummary() ctxplan.TurnTaxSummary {
	p.turnTaxMu.Lock()
	defer p.turnTaxMu.Unlock()
	return p.turnTaxLog.Summary()
}

// ExplainTurnTax renders the recorded decisions as an operator-readable report: one line per
// turn naming the strategy, its tax, and the reason it won, plus the strategy-count footer.
func (p *InKernelPlanner) ExplainTurnTax() string {
	p.turnTaxMu.Lock()
	defer p.turnTaxMu.Unlock()
	return p.turnTaxLog.Explain()
}

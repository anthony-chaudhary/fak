package ctxplan

import (
	"strconv"
	"strings"
)

// turntax.go — issue #1538: the per-turn turn-tax planner. The three per-turn cache
// strategies already exist in the tree and each is reached by whichever code path happens to
// run: prefix REUSE (internal/agent's in-kernel radix prefix match), an O(1) session QUERY
// (internal/contextq, ctxplan's own Index.ProbePlan), and a COLD PREFILL (the reuse path's
// miss branch). What was missing is the DECISION: nothing picked among the three per turn,
// and nothing recorded WHY the turn ended up on the path it took. This file supplies exactly
// that — one closed-vocabulary decision per turn, with its reason and the token taxes that
// produced it.
//
// TURN TAX. The planner compares the three strategies on the only per-turn cost that is
// knowable before the turn runs: how many prompt tokens this turn must COMPUTE (or retrieve)
// to get the context it needs. That number is the turn's tax.
//
//   - reuse        tax = PromptTokens - ReusableTokens (recompute only the unmatched suffix)
//   - query        tax = QueryTokens                   (retrieve the facts instead of
//     materializing an over-budget context)
//   - cold prefill tax = PromptTokens                  (compute the whole prompt)
//
// Cheapest AVAILABLE strategy wins; ties break reuse > query > cold prefill (a tie kept on
// the warm path keeps the session's KV hot for the next turn). This is a token-count model,
// not a wall-clock one: it is deterministic and auditable, which is what a recorded decision
// needs. It deliberately does NOT model device time, batch effects, or provider pricing.
//
// COLD-PATH CORRECTNESS IS EXPLICIT. Cold prefill is ALWAYS available — it is the fail-open
// floor, never gated on anything, so there is no signal state in which the planner has no
// answer. Reuse is available only when a matched prefix is ALSO trusted: an untrusted match
// (the materialization verdict refused it — wrong model, wrong tokenizer, stale epoch,
// cross-tenant) contributes ZERO reusable tokens and the turn falls to the cold path with
// that refusal named in the reason. The planner can never trade correctness for tax.
//
// PROVENANCE. This is the KERNEL-local decision only: fak's own KV prefix tree and session
// index on this box. It is not provider prefix-cache attribution (that stays in cachemeta /
// the provider_prefix telemetry axis), not the context-plan budget model (adaptive.go sizes
// W; this file only READS the resulting budget as a signal), and not forecast provenance
// (Forecast/Outcome). A caller that reports all four keeps them on separate axes.
//
// DETERMINISM. PlanTurnTax is a pure function of its signals: no clock, no randomness, no
// I/O. The same signals always produce the same decision, so a recorded decision REPLAYS —
// which is what TurnTaxLog.Replay checks.

// TurnTaxStrategy is the CLOSED vocabulary of per-turn cache strategies. Exactly one is
// produced per turn; there is no "undecided" state once PlanTurnTax runs, which is what
// keeps a turn from silently taking a path nothing recorded.
type TurnTaxStrategy string

const (
	// TurnTaxReuse: serve the turn from the cached KV prefix and compute only the
	// unmatched suffix (the in-kernel radix prefix match). Chosen when a matched prefix is
	// trusted and its recompute tail is the cheapest available option.
	TurnTaxReuse TurnTaxStrategy = "reuse"
	// TurnTaxQuery: answer the turn from the O(1) session index instead of materializing
	// the full history resident. Chosen when the prompt does not fit the resident budget and
	// retrieving the needed spans costs fewer tokens than recomputing them.
	TurnTaxQuery TurnTaxStrategy = "query"
	// TurnTaxColdPrefill: compute the whole prompt fresh. The fail-open floor — always
	// available, chosen whenever nothing cheaper is BOTH available and trusted. This is the
	// correct answer for a first turn, for a prefix the verdict refused, and for any signal
	// state the planner does not recognize.
	TurnTaxColdPrefill TurnTaxStrategy = "cold_prefill"
)

// validTurnTaxStrategies is the membership set PlanTurnTax always produces one member of
// — used by tests and by any caller (de)serializing a persisted decision, so a corrupt or
// foreign value fails closed rather than being silently accepted.
var validTurnTaxStrategies = map[TurnTaxStrategy]bool{
	TurnTaxReuse:       true,
	TurnTaxQuery:       true,
	TurnTaxColdPrefill: true,
}

// ValidTurnTaxStrategy reports whether s is a member of the closed vocabulary.
func ValidTurnTaxStrategy(s TurnTaxStrategy) bool { return validTurnTaxStrategies[s] }

func (s TurnTaxStrategy) String() string {
	if ValidTurnTaxStrategy(s) {
		return string(s)
	}
	if s == "" {
		return "(unset)"
	}
	return "unknown(" + string(s) + ")"
}

// TurnTaxSignals is the per-turn input the planner decides from. Every field is a fact the
// caller already holds BEFORE the turn runs (a lookup-side prefix match, a verdict, an index
// probe estimate, the sized resident budget) — nothing here is measured after the fact, which
// is what makes this a planner rather than an attribution.
type TurnTaxSignals struct {
	// PromptTokens is this turn's full prompt length in tokens — the cold-prefill tax and
	// the ceiling every other tax is compared against. A non-positive value normalizes to 0
	// (an empty turn), which decides cold prefill at zero tax.
	PromptTokens int `json:"prompt_tokens"`
	// MatchedPrefix is the LOOKUP-SIDE match: how many leading prompt tokens the KV index
	// says it holds for this prompt, before any trust gate runs. It is clamped to
	// [0, PromptTokens] — an index cannot match more of the prompt than exists.
	MatchedPrefix int `json:"matched_prefix_tokens"`
	// PrefixTrusted is the materialization verdict for that match: may these cached tokens
	// actually be served for THIS turn (same model, tokenizer, epoch, tenant, quality)?
	// False turns the whole match into 0 reusable tokens — the planner never serves an
	// unverified prefix to save tax. A caller with no verdict to offer must pass false.
	PrefixTrusted bool `json:"prefix_trusted"`
	// QueryTokens is the tax of answering this turn from the O(1) session index instead:
	// the tokens the retrieved spans would occupy. 0 (or negative) means NO queryable index
	// is available this turn, so the query strategy is not a candidate at all — it is the
	// caller's single switch for "this turn has no index to ask".
	QueryTokens int `json:"query_tokens"`
	// ResidentBudget is the working-set budget W the context planner sized for this turn
	// (adaptive.go's RecommendBudget). It gates the QUERY strategy only: a query substitutes
	// for materializing an OVER-BUDGET context, so when the whole prompt already fits within
	// W there is nothing for a query to displace and the turn just sends the prompt. 0 (or
	// negative) means "budget unknown" and does NOT veto a query — an unknown budget is not
	// evidence the prompt fits.
	ResidentBudget int `json:"resident_budget,omitempty"`
}

// TurnTaxDecision is the recorded per-turn cache decision: the chosen strategy, the
// operator-readable reason it was chosen (in the PageFaultDecision / Plan.Explain style), and
// the three taxes plus availability flags the choice was made from. It is self-describing —
// a persisted decision needs no side join to explain itself.
type TurnTaxDecision struct {
	Strategy TurnTaxStrategy `json:"strategy"`
	Reason   string          `json:"reason"`
	// PromptTokens is the normalized prompt length the taxes are denominated in.
	PromptTokens int `json:"prompt_tokens"`
	// ReusableTokens is the prefix match AFTER the trust gate: MatchedPrefix when the verdict
	// trusted it, 0 when it did not. The gap between this and MatchedPrefix is exactly what
	// correctness cost the turn, and it is recorded rather than folded into a miss.
	ReusableTokens int `json:"reusable_tokens"`
	// MatchedPrefix echoes the pre-gate lookup match, so a reader can see a refused prefix.
	MatchedPrefix int `json:"matched_prefix_tokens"`
	// ReuseTax / QueryTax / ColdTax are the per-strategy token taxes. A tax is only
	// meaningful when its Available flag is set; ColdTax is always meaningful because cold
	// prefill is always available.
	ReuseTax       int  `json:"reuse_tax"`
	ReuseAvailable bool `json:"reuse_available"`
	QueryTax       int  `json:"query_tax"`
	QueryAvailable bool `json:"query_available"`
	ColdTax        int  `json:"cold_tax"`
}

// PlanTurnTax is the deterministic per-turn decision: given the turn's signals it returns
// EXACTLY ONE closed-vocabulary strategy with a non-empty reason — never a silent
// no-decision. It is pure (no clock, no randomness, no I/O), so the same signals always
// reproduce the same decision, which is what makes a recorded TurnTaxLog replayable.
//
// Availability, evaluated first (a strategy that is not available can never be chosen, no
// matter how cheap):
//
//   - cold prefill is ALWAYS available (the fail-open floor).
//   - reuse is available iff the matched prefix is non-empty AND the verdict trusted it.
//   - query is available iff QueryTokens > 0 AND the prompt does not already fit the sized
//     resident budget (a known budget the prompt fits vetoes the query; an unknown budget
//     does not).
//
// Selection, over the available strategies only: lowest tax wins, and ties break
// reuse > query > cold prefill so a tie stays on the warm path.
func PlanTurnTax(sig TurnTaxSignals) TurnTaxDecision {
	prompt := sig.PromptTokens
	if prompt < 0 {
		prompt = 0
	}
	cached := sig.MatchedPrefix
	if cached < 0 {
		cached = 0
	}
	if cached > prompt {
		// An index cannot hold more of the prompt than the prompt has; clamp rather than
		// trust a caller's over-report, which would otherwise mint a negative reuse tax.
		cached = prompt
	}
	reusable := 0
	if sig.PrefixTrusted {
		reusable = cached
	}
	queryTokens := sig.QueryTokens
	if queryTokens < 0 {
		queryTokens = 0
	}

	d := TurnTaxDecision{
		PromptTokens:   prompt,
		MatchedPrefix:  cached,
		ReusableTokens: reusable,
		ColdTax:        prompt,
		ReuseTax:       prompt - reusable,
		ReuseAvailable: reusable > 0,
		QueryTax:       queryTokens,
		QueryAvailable: queryTokens > 0 && !promptFitsBudget(prompt, sig.ResidentBudget),
	}

	// Lowest available tax wins. Cold prefill seeds the comparison because it is the only
	// strategy that is always available; query improves on it strictly; reuse takes ties so
	// an equal-tax turn stays on the warm path.
	d.Strategy = TurnTaxColdPrefill
	best := d.ColdTax
	if d.QueryAvailable && d.QueryTax < best {
		d.Strategy, best = TurnTaxQuery, d.QueryTax
	}
	if d.ReuseAvailable && d.ReuseTax <= best {
		d.Strategy = TurnTaxReuse
	}
	d.Reason = turnTaxReason(d, sig)
	return d
}

// promptFitsBudget reports whether the whole prompt already fits the sized resident working
// set. A non-positive budget is UNKNOWN, not "fits": an absent budget must not silently veto
// the query strategy (that would be a fits-claim with no evidence behind it).
func promptFitsBudget(prompt, budget int) bool {
	return budget > 0 && prompt <= budget
}

// turnTaxReason renders the operator-readable WHY for a decision — the load-bearing half of
// this issue's target ("records why"). Every branch names the numbers that produced the
// choice, so a reader can re-derive the decision from the sentence alone.
func turnTaxReason(d TurnTaxDecision, sig TurnTaxSignals) string {
	switch d.Strategy {
	case TurnTaxReuse:
		r := "trusted prefix covers " + strconv.Itoa(d.ReusableTokens) + " of " +
			strconv.Itoa(d.PromptTokens) + " prompt tokens: reusing costs " +
			strconv.Itoa(d.ReuseTax) + " vs " + strconv.Itoa(d.ColdTax) + " cold"
		if d.QueryAvailable {
			r += " and " + strconv.Itoa(d.QueryTax) + " to query"
		}
		return r
	case TurnTaxQuery:
		r := "prompt of " + strconv.Itoa(d.PromptTokens) + " tokens exceeds the " +
			strconv.Itoa(sig.ResidentBudget) + "-token resident budget"
		if sig.ResidentBudget <= 0 {
			r = "resident budget unknown for a " + strconv.Itoa(d.PromptTokens) + "-token prompt"
		}
		r += ": querying the session index costs " + strconv.Itoa(d.QueryTax) + " vs " +
			strconv.Itoa(d.ColdTax) + " to prefill cold"
		if d.MatchedPrefix > 0 && !d.ReuseAvailable {
			r += " (the " + strconv.Itoa(d.MatchedPrefix) + "-token prefix match was not trusted)"
		} else if d.ReuseAvailable {
			r += " and " + strconv.Itoa(d.ReuseTax) + " to reuse"
		}
		return r
	default:
		switch {
		case d.PromptTokens == 0:
			return "empty prompt: nothing to reuse or query, the cold path is the only correct one"
		case d.MatchedPrefix > 0 && !d.ReuseAvailable:
			return "prefix matched " + strconv.Itoa(d.MatchedPrefix) + " tokens but its materialization " +
				"verdict refused them: prefilling all " + strconv.Itoa(d.ColdTax) +
				" tokens cold rather than serving an unverified prefix"
		case sig.QueryTokens > 0 && !d.QueryAvailable:
			return "no trusted cached prefix, and the " + strconv.Itoa(sig.QueryTokens) +
				"-token query is moot because the " + strconv.Itoa(d.PromptTokens) +
				"-token prompt already fits the " + strconv.Itoa(sig.ResidentBudget) +
				"-token resident budget: prefilling cold"
		case d.QueryAvailable:
			return "no trusted cached prefix, and querying costs " + strconv.Itoa(d.QueryTax) +
				" against " + strconv.Itoa(d.ColdTax) + " to prefill cold: taking the cold path"
		default:
			return "no cached prefix and no queryable session index: prefilling all " +
				strconv.Itoa(d.ColdTax) + " tokens cold"
		}
	}
}

// Explain renders one decision as a single operator-readable line, in the PageFaultLog /
// Plan.Explain style.
func (d TurnTaxDecision) Explain() string {
	return "turn-tax: " + d.Strategy.String() + " (tax " + strconv.Itoa(d.Tax()) + " of " +
		strconv.Itoa(d.PromptTokens) + " prompt tokens) — " + d.Reason
}

// Tax is the token tax of the CHOSEN strategy — the number a turn-cost report adds up.
func (d TurnTaxDecision) Tax() int {
	switch d.Strategy {
	case TurnTaxReuse:
		return d.ReuseTax
	case TurnTaxQuery:
		return d.QueryTax
	default:
		return d.ColdTax
	}
}

// Saved is the tax this decision avoided versus prefilling the whole prompt cold — the
// honest per-turn saving, 0 on the cold path by construction (never negative: a strategy is
// only chosen when its tax is at or below the cold tax).
func (d TurnTaxDecision) Saved() int {
	if s := d.ColdTax - d.Tax(); s > 0 {
		return s
	}
	return 0
}

// TurnTaxSignalsForBudget builds the planner's signals with the resident budget sized from
// the context planner's own difficulty model (adaptive.go's RecommendBudget) rather than a
// caller-invented number. It READS the budget model — it does not change how RecommendBudget
// sizes W — so the same difficulty signals that pick the working set also gate the query
// strategy, with no second budget model to drift.
func TurnTaxSignalsForBudget(promptTokens, matchedPrefix int, prefixTrusted bool, queryTokens int, d Difficulty, bounds BudgetBounds) TurnTaxSignals {
	return TurnTaxSignals{
		PromptTokens:   promptTokens,
		MatchedPrefix:  matchedPrefix,
		PrefixTrusted:  prefixTrusted,
		QueryTokens:    queryTokens,
		ResidentBudget: RecommendBudget(d, bounds).Tokens,
	}
}

// TurnTaxLog is the append-only, replayable ledger of per-turn cache decisions — the
// RECORD half of "makes a cache decision by default and records why". Append is the only way
// to decide-and-record in one call, so a caller cannot act on a decision it never recorded.
// Entries are kept in occurrence order and bounded to the shared recent window
// (defaultMaxLedgerEntries, as PageFaultLog and ObjectiveLog are), so a long-lived session
// cannot grow it without limit. The zero value is a usable empty log.
type TurnTaxLog struct {
	entries []TurnTaxLogEntry
	// maxEntries bounds the retained window: 0 = the default cap; <0 = unbounded;
	// >0 = a custom cap. See effectiveLedgerCap.
	maxEntries int
}

// TurnTaxLogEntry pairs one turn's signals with the decision computed for them, so the log
// is self-contained: replaying it needs no external state.
type TurnTaxLogEntry struct {
	Signals  TurnTaxSignals  `json:"signals"`
	Decision TurnTaxDecision `json:"decision"`
}

// Append plans sig, records the (signals, decision) pair as the next entry, and returns the
// decision — the one call a turn path needs to both record the transition and act on it. It
// never mutates a prior entry: the log is append-only, matching ctxplan's audit posture.
func (l *TurnTaxLog) Append(sig TurnTaxSignals) TurnTaxDecision {
	d := PlanTurnTax(sig)
	l.entries = append(l.entries, TurnTaxLogEntry{Signals: sig, Decision: d})
	if max, bounded := effectiveLedgerCap(l.maxEntries); bounded && len(l.entries) > max {
		// Drop the oldest decisions beyond the window (see PageFaultLog.Append).
		l.entries = l.entries[len(l.entries)-max:]
	}
	return d
}

// Entries returns a defensive copy of the logged entries in occurrence order.
func (l *TurnTaxLog) Entries() []TurnTaxLogEntry {
	return append([]TurnTaxLogEntry(nil), l.entries...)
}

// Replay recomputes PlanTurnTax for every logged entry from its own stored signals and
// reports any entry whose strategy no longer matches. A log built from pure signals always
// replays with zero diverged entries — the determinism witness. A non-empty diverged set
// means either the log was tampered with or the planner's logic changed incompatibly since
// the entry was recorded.
func (l *TurnTaxLog) Replay() (diverged []int, allMatch bool) {
	allMatch = true
	for i, e := range l.entries {
		if got := PlanTurnTax(e.Signals); got.Strategy != e.Decision.Strategy {
			diverged = append(diverged, i)
			allMatch = false
		}
	}
	return diverged, allMatch
}

// TurnTaxSummary folds the log into per-strategy counts plus the token taxes behind them —
// the O(1) readout an operator surface prints instead of walking every entry.
type TurnTaxSummary struct {
	Reuse       int `json:"reuse"`
	Query       int `json:"query"`
	ColdPrefill int `json:"cold_prefill"`
	// PromptTokens is the total prompt tokens across the retained window; Tax is what the
	// chosen strategies actually cost; Saved is PromptTokens-Tax, the honest window saving.
	PromptTokens int `json:"prompt_tokens"`
	Tax          int `json:"tax"`
	Saved        int `json:"saved"`
}

// Summary computes the strategy counts and token taxes over every retained entry.
func (l *TurnTaxLog) Summary() TurnTaxSummary {
	var s TurnTaxSummary
	for _, e := range l.entries {
		switch e.Decision.Strategy {
		case TurnTaxReuse:
			s.Reuse++
		case TurnTaxQuery:
			s.Query++
		case TurnTaxColdPrefill:
			s.ColdPrefill++
		}
		s.PromptTokens += e.Decision.PromptTokens
		s.Tax += e.Decision.Tax()
		s.Saved += e.Decision.Saved()
	}
	return s
}

// Explain renders the log as an operator-readable report, in the PageFaultLog.Explain style:
// one line per decision plus the strategy-count and tax footer.
func (l *TurnTaxLog) Explain() string {
	var b strings.Builder
	s := l.Summary()
	b.WriteString("ctxplan turn-tax log: " + strconv.Itoa(len(l.entries)) + " decision(s)\n")
	for i, e := range l.entries {
		b.WriteString("  [turn ")
		b.WriteString(strconv.Itoa(i))
		b.WriteString("] ")
		b.WriteString(e.Decision.Explain())
		b.WriteString("\n")
	}
	b.WriteString("  totals: reuse=" + strconv.Itoa(s.Reuse) +
		" query=" + strconv.Itoa(s.Query) +
		" cold_prefill=" + strconv.Itoa(s.ColdPrefill) +
		" prompt_tokens=" + strconv.Itoa(s.PromptTokens) +
		" tax=" + strconv.Itoa(s.Tax) +
		" saved=" + strconv.Itoa(s.Saved) + "\n")
	return b.String()
}

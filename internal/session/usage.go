package session

import (
	"crypto/sha256"
	"encoding/hex"
	"strconv"
)

// Usage is the per-turn token accounting the model boundary reports after a
// successful turn. OutputTokens debits the historical output-token budget.
// ContextTokens debits the long-context guardrail: the prompt/context window the
// model had to read for this turn, normalized by the caller from provider usage.
// CostMicroCents debits the spend ceiling (Budget.SpendMicroCentsLeft): the
// PRICED cost of this turn in micro-cents (1e-8 USD), computed by the caller —
// the table stays price-blind so the per-MTok price table lives in exactly one
// place (the host, which knows the provider). 0 = unpriced turn, no spend debit;
// a dollar-blind host therefore leaves a configured spend budget honestly
// untouched rather than debiting a guessed cost.
type Usage struct {
	OutputTokens   int
	ContextTokens  int
	CostMicroCents int64
}

// DebitUsage records a completed turn's token usage against the session's live
// budgets. Output-token exhaustion is still observed by the next Decide, matching
// Debit's original boundary discipline. Context-token exhaustion is the long-window
// reset trigger: the session is moved to Draining immediately and a continuation id
// is minted so the next boundary can tell a supervisor which fresh window to start.
func (t *Table) DebitUsage(trace string, u Usage) State {
	if t == nil || (u.OutputTokens <= 0 && u.ContextTokens <= 0 && u.CostMicroCents <= 0) {
		if t != nil {
			return t.Get(trace)
		}
		return DefaultState(trace)
	}
	t.mu.Lock()
	cur := t.getLocked(trace)
	if cur.Run.terminal() || cur.Run == Paused {
		t.mu.Unlock()
		return cur
	}
	changed := false
	// Record this turn's cost in the bounded per-session ring (#756) BEFORE the budget
	// arms — the ring tracks the cost a turn actually incurred, so it fills even for a
	// session with unbounded budgets (the runaway `fak ps` most needs to see is one with
	// no cap to drain). The push is O(1) on a fixed array; it never grows.
	cur.Cost = cur.Cost.push(TurnCost{OutputTokens: u.OutputTokens, ContextTokens: u.ContextTokens})
	changed = true
	if u.OutputTokens > 0 && !cur.Budget.tokensUnbounded() {
		cur.Budget.TokensLeft -= u.OutputTokens
		changed = true
	}
	// Spend ceiling FIRST: money is the hardest ceiling, so when one turn crosses
	// both the spend and the context budget the spend drain wins — no continuation
	// id is minted and no fresh-window reset fires past a spent dollar cap (the
	// reset path continues context drains, never spend drains). The exhaustion is
	// taken immediately, like the context axis: the drain must be observable to a
	// supervisor the instant the ceiling is crossed, not one boundary later.
	spendDrained := false
	if u.CostMicroCents > 0 && cur.Budget.spendBounded() {
		cur.Budget.SpendMicroCentsLeft -= u.CostMicroCents
		changed = true
		if cur.Budget.SpendMicroCentsLeft <= 0 {
			cur.Budget.SpendMicroCentsLeft = 0
			cur.Run = Draining
			cur.Reason = ReasonBudgetSpend
			spendDrained = true
		}
	}
	// fireKind/fire capture which budget event (if any) this debit triggers; the observer
	// itself runs AFTER the lock is released so a slow webhook never stalls other sessions.
	fireKind, fire := BudgetWarn, false
	relayFire := false
	if u.ContextTokens > 0 && cur.Budget.contextBounded() {
		prevLeft := cur.Budget.ContextTokensLeft
		cur.Budget.ContextTokensLeft -= u.ContextTokens
		changed = true
		switch {
		case cur.Budget.ContextTokensLeft <= 0:
			cur.Budget.ContextTokensLeft = 0
			if !spendDrained {
				cur.Run = Draining
				cur.Reason = ReasonBudgetContext
				if cur.ContinuationID == "" {
					cur.ContinuationID = continuationID(trace, cur.Rev+1)
				}
				cur.CacheAffinity = cacheAffinityForContinuation(cur, cur.ContinuationID, ReasonBudgetContext)
				fireKind, fire = BudgetExhausted, t.obs != nil
			}
		case !spendDrained && t.crossedWarnLocked(cur.Budget, prevLeft):
			fireKind, fire = BudgetWarn, true
		}
		if t.crossedRelaySoftLocked(trace, cur.Budget, u.ContextTokens) {
			relayFire = true
		}
	}
	if !changed {
		t.mu.Unlock()
		return cur
	}
	out := t.putLocked(cur)
	obs := t.obs
	relayObs := t.relayObs
	var ev BudgetEvent
	if fire && obs != nil {
		ev = budgetEvent(out, fireKind, u.ContextTokens)
	}
	var relayEv RelayShadowEvent
	if relayFire && relayObs != nil {
		relayEv = relayShadowEvent(out, u.ContextTokens, t.relaySoftMark)
	}
	t.mu.Unlock()
	if fire && obs != nil {
		obs(ev)
	}
	if relayFire && relayObs != nil {
		relayObs(relayEv)
	}
	return out
}

func continuationID(trace string, rev uint64) string {
	h := sha256.Sum256([]byte(trace + "\x00" + strconv.FormatUint(rev, 10)))
	return "win-" + hex.EncodeToString(h[:])[:16]
}

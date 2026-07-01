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
type Usage struct {
	OutputTokens  int
	ContextTokens int
}

// DebitUsage records a completed turn's token usage against the session's live
// budgets. Output-token exhaustion is still observed by the next Decide, matching
// Debit's original boundary discipline. Context-token exhaustion is the long-window
// reset trigger: the session is moved to Draining immediately and a continuation id
// is minted so the next boundary can tell a supervisor which fresh window to start.
func (t *Table) DebitUsage(trace string, u Usage) State {
	if t == nil || (u.OutputTokens <= 0 && u.ContextTokens <= 0) {
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
	// fireKind/fire capture which budget event (if any) this debit triggers; the observer
	// itself runs AFTER the lock is released so a slow webhook never stalls other sessions.
	fireKind, fire := BudgetWarn, false
	if u.ContextTokens > 0 && cur.Budget.contextBounded() {
		prevLeft := cur.Budget.ContextTokensLeft
		cur.Budget.ContextTokensLeft -= u.ContextTokens
		changed = true
		switch {
		case cur.Budget.ContextTokensLeft <= 0:
			cur.Budget.ContextTokensLeft = 0
			cur.Run = Draining
			cur.Reason = ReasonBudgetContext
			if cur.ContinuationID == "" {
				cur.ContinuationID = continuationID(trace, cur.Rev+1)
			}
			cur.CacheAffinity = cacheAffinityForContinuation(cur, cur.ContinuationID, ReasonBudgetContext)
			fireKind, fire = BudgetExhausted, t.obs != nil
		case t.crossedWarnLocked(cur.Budget, prevLeft):
			fireKind, fire = BudgetWarn, true
		}
	}
	if !changed {
		t.mu.Unlock()
		return cur
	}
	out := t.putLocked(cur)
	obs := t.obs
	var ev BudgetEvent
	if fire && obs != nil {
		ev = budgetEvent(out, fireKind)
	}
	t.mu.Unlock()
	if fire && obs != nil {
		obs(ev)
	}
	return out
}

func continuationID(trace string, rev uint64) string {
	h := sha256.Sum256([]byte(trace + "\x00" + strconv.FormatUint(rev, 10)))
	return "win-" + hex.EncodeToString(h[:])[:16]
}

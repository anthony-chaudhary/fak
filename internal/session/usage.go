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
	defer t.mu.Unlock()
	cur := t.getLocked(trace)
	if cur.Run.terminal() || cur.Run == Paused {
		return cur
	}
	changed := false
	if u.OutputTokens > 0 && !cur.Budget.tokensUnbounded() {
		cur.Budget.TokensLeft -= u.OutputTokens
		changed = true
	}
	if u.ContextTokens > 0 && cur.Budget.contextBounded() {
		cur.Budget.ContextTokensLeft -= u.ContextTokens
		changed = true
		if cur.Budget.ContextTokensLeft <= 0 {
			cur.Budget.ContextTokensLeft = 0
			cur.Run = Draining
			cur.Reason = ReasonBudgetContext
			if cur.ContinuationID == "" {
				cur.ContinuationID = continuationID(trace, cur.Rev+1)
			}
		}
	}
	if !changed {
		return cur
	}
	return t.putLocked(cur)
}

func continuationID(trace string, rev uint64) string {
	h := sha256.Sum256([]byte(trace + "\x00" + strconv.FormatUint(rev, 10)))
	return "win-" + hex.EncodeToString(h[:])[:16]
}

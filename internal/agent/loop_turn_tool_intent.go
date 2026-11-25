package agent

import (
	"context"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/session"
)

// toolIntent carries the tool-lifecycle facts one dispatch knows into the session's
// advisory TurnIntent while its calls are outstanding (fak#13383): IsSpeculative for an
// unconfirmed streaming speculation, WillDiscard once cancellation decides the results
// go unconsumed. Timing, prefix and settle fields stay zero (fak#13387 adds them). A flag
// is owned only when this dispatch turned it on, and every write is a Rev-guarded
// CompareAndSet rather than a whole-intent SetTurnIntent, so a concurrent write and the
// other intent fields survive, a Stopped record is never revived and no record is minted.
type toolIntent struct {
	table  *session.Table
	trace  string
	mu     sync.Mutex
	owned  session.TurnIntent
	closed bool
}

// toolIntentAttempts bounds the re-read/retry loop when a concurrent write moves Rev.
const toolIntentAttempts = 4

func (r *armRunner) toolIntent() *toolIntent {
	if r.cfg == nil || r.cfg.table == nil {
		return nil
	}
	return &toolIntent{table: r.cfg.table, trace: r.cfg.trace}
}

// publish records the facts known at dispatch, arms the cancellation fact, and returns
// the cleanup that withdraws every owned flag on any dispatch exit.
func (ti *toolIntent) publish(ctx context.Context, speculative bool) func() {
	ti.mark(session.TurnIntent{IsSpeculative: speculative})
	stop := context.AfterFunc(ctx, func() { ti.mark(session.TurnIntent{WillDiscard: true}) })
	return func() {
		stop()
		ti.withdraw()
	}
}

func (ti *toolIntent) mark(facts session.TurnIntent) {
	ti.mu.Lock()
	defer ti.mu.Unlock()
	if ti.closed || facts.IsZero() {
		return
	}
	var gained session.TurnIntent
	if ti.swap(func(in *session.TurnIntent) bool {
		gained = session.TurnIntent{
			IsSpeculative: facts.IsSpeculative && !in.IsSpeculative,
			WillDiscard:   facts.WillDiscard && !in.WillDiscard,
		}
		in.IsSpeculative = in.IsSpeculative || gained.IsSpeculative
		in.WillDiscard = in.WillDiscard || gained.WillDiscard
		return !gained.IsZero()
	}) {
		ti.owned.IsSpeculative = ti.owned.IsSpeculative || gained.IsSpeculative
		ti.owned.WillDiscard = ti.owned.WillDiscard || gained.WillDiscard
	}
}

func (ti *toolIntent) withdraw() {
	ti.mu.Lock()
	defer ti.mu.Unlock()
	ti.closed = true
	owned := ti.owned
	if owned.IsZero() {
		return
	}
	ti.swap(func(in *session.TurnIntent) bool {
		changed := (owned.IsSpeculative && in.IsSpeculative) || (owned.WillDiscard && in.WillDiscard)
		in.IsSpeculative = in.IsSpeculative && !owned.IsSpeculative
		in.WillDiscard = in.WillDiscard && !owned.WillDiscard
		return changed
	})
}

// swap applies edit to the live record's Intent under the table's optimistic Rev guard.
// An unseen (Rev 0) or Stopped record is never written; a lost race re-reads and retries.
func (ti *toolIntent) swap(edit func(*session.TurnIntent) bool) bool {
	for attempt := 0; attempt < toolIntentAttempts; attempt++ {
		cur := ti.table.Get(ti.trace)
		if cur.Rev == 0 || cur.Run == session.Stopped {
			return false
		}
		next := cur
		if !edit(&next.Intent) {
			return false
		}
		if _, ok := ti.table.CompareAndSet(ti.trace, cur.Rev, next); ok {
			return true
		}
	}
	return false
}

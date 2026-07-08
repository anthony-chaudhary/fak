package fakclient

import "context"

// changes_consumer.go — a reference consumer for the fak change feed
// (GET /v1/fak/changes), the worked example behind the consumer contract (#3173).
// It is intentionally small: the value is the CONTRACT it demonstrates, not the code.
//
// The feed is a cursor-drained, bounded-window, at-least-once change stream — the
// same shape a database CDC consumer expects (see the concept page
// docs/explainers/change-data-capture-for-agents.md and the how-to in
// docs/integrations/consuming-the-fak-changes-feed.md). A correct consumer needs four
// habits, and this type encodes all four so an application does not reinvent them:
//
//   - Resume from a cursor. Persist Cursor() and rebuild with NewConsumerAt so a
//     restart pages forward from where it left off instead of re-reading the window.
//   - Make delivery idempotent. Delivery is at-least-once: the same Seq can be
//     re-presented (e.g. after a handler error retries). Drain dedupes by Seq, so the
//     handler only ever sees each Seq advance once.
//   - Survive a retention gap. The server's window is bounded; a consumer that lapses
//     past it loses the events that fell off. Drain reports that with Gapped() so the
//     application can reconcile against an authority instead of assuming continuity.
//   - Fail without losing your place. A handler error stops the drain with the cursor
//     left BEFORE the failing event, so the next Drain retries from there.

// ChangeHandler processes one drained change event. Returning an error stops the
// current Drain at that event WITHOUT advancing the cursor past it, so the next Drain
// re-presents it — the at-least-once contract that lets a handler fail safely.
type ChangeHandler func(ctx context.Context, ev ChangeEvent) error

// Consumer is a resumable, at-least-once reader over one Client's change feed. It
// owns only a cursor and a sticky gap flag; it holds no goroutine and no network
// state, so it is cheap to build and safe to persist across restarts via Cursor().
// It is NOT safe for concurrent Drain calls — drive it from a single loop.
type Consumer struct {
	client *Client
	cursor uint64
	gapped bool
}

// NewConsumer starts a consumer at the beginning of the retained window (cursor 0 —
// the first Drain reads everything the server still holds).
func NewConsumer(c *Client) *Consumer { return &Consumer{client: c} }

// NewConsumerAt resumes a consumer after a previously persisted cursor. Pass the
// Cursor() value saved by an earlier run so the first Drain pages forward from it.
func NewConsumerAt(c *Client, cursor uint64) *Consumer {
	return &Consumer{client: c, cursor: cursor}
}

// Cursor is the Seq the consumer will resume AFTER on the next Drain. Persist it to
// make the consumer durable across process restarts.
func (cs *Consumer) Cursor() uint64 { return cs.cursor }

// Drain fetches every change after the consumer's cursor and calls handle for each,
// in Seq order, advancing the cursor as it goes. It returns the number of events
// delivered to handle. A handler error stops the drain with the cursor left just
// before the failing event (the next Drain retries it). Even when handle is called
// zero times, the cursor advances to the server's head so a lapsed consumer re-syncs
// forward rather than re-scanning an elapsed window.
func (cs *Consumer) Drain(ctx context.Context, handle ChangeHandler) (int, error) {
	since := cs.cursor
	resp, err := cs.client.Changes(ctx, since)
	if err != nil {
		return 0, err
	}

	// Conservative retention-gap signal: the first event we got back sits ABOVE the
	// next Seq we expected. On the coherence feed Seq is monotone but NOT contiguous —
	// one counter is shared across the mutation and revocation buses and the drain is
	// scoped per principal, so intervening Seqs may simply have never been ours to see.
	// So a raised flag means "you MIGHT have missed events; reconcile against the
	// durable journal (GET /v1/fak/events) if completeness matters" — never a proof of
	// loss. It over-reports rather than under-reports, because a needless reconcile is
	// safe and a missed one is not.
	if since > 0 && len(resp.Events) > 0 && resp.Events[0].Seq > since+1 {
		cs.gapped = true
	}

	delivered := 0
	for _, ev := range resp.Events {
		if ev.Seq <= cs.cursor {
			continue // dedupe: at-least-once delivery can re-present an already-seen Seq
		}
		if err := handle(ctx, ev); err != nil {
			return delivered, err // leave the cursor before the failing event; next Drain retries it
		}
		cs.cursor = ev.Seq
		delivered++
	}

	// Re-sync to head. Adopt the server's head cursor even if it is beyond the last
	// event we saw, so a consumer that lapsed past the window stops re-requesting the
	// elapsed span. Only ever advance forward.
	if resp.Cursor > cs.cursor {
		cs.cursor = resp.Cursor
	}
	return delivered, nil
}

// Gapped reports whether any Drain since the last call to Gapped observed a possible
// retention gap, and clears the flag. See Drain for the honest, conservative meaning:
// a true result is a prompt to reconcile against an authority, not a proof that events
// were lost.
func (cs *Consumer) Gapped() bool {
	g := cs.gapped
	cs.gapped = false
	return g
}

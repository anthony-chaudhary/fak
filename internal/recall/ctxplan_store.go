package recall

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/ctxplan"
)

// CtxStore adapts a reloaded *Session into a ctxplan.Store — the first REAL-store
// backing for the context planner (#545). Until now ctxplan.Store was backed only by
// the in-memory MemStore reference implementation; this adapter lets the planner view
// and page in against a durable recall core image, so the same O(1)-view planner that
// runs in the demo now runs over a finished session's real history — with the recall
// trust gate enforcing every page-in.
//
// The adapter is a thin lowering, exactly the "higher-tier follow-on" the ctxplan
// store.go comment names: it imports both packages, so it lives here in recall (tier 3,
// above ctxplan's stdlib-only tier-1 leaf), never inside ctxplan. Each recall.Page
// becomes a ctxplan.Span carrying SAFE metadata only (never the sealed bytes); a
// quarantined page maps to a Sealed span, a tombstoned page to a Tombstoned span, and
// Store.Materialize routes through Session.Resolve — so poison and suppression surface
// as ctxplan.ErrSealed / ctxplan.ErrTombstoned and the planner reports them as Refused,
// never rendered. The lossless property is inherited from recall: an elided span keeps
// its content-address Digest, so it stays one demand-page away.
type CtxStore struct {
	session *Session
}

// Compile-time assertion that *CtxStore satisfies ctxplan.Store.
var _ ctxplan.Store = (*CtxStore)(nil)

// NewCtxStore wraps a reloaded Session as a ctxplan.Store. The session is the source of
// truth: Spans reads its page table, Materialize pages in through its trust gate, and
// tombstone state is read live so a context-control change between plans is honored.
func NewCtxStore(s *Session) *CtxStore { return &CtxStore{session: s} }

// Session returns the underlying reloaded session the adapter backs the planner with.
func (c *CtxStore) Session() *Session { return c.session }

// pageSpanID is the stable span id the adapter assigns the page at `step`. It encodes
// the step so Materialize can reverse it; the planner treats ids as opaque keys, so any
// reversible scheme is correct. These are recall PAGES, so the prefix is "page:" (not
// the MemStore's "span:"), keeping the two stores' id spaces self-documenting.
func pageSpanID(step int) string { return "page:" + strconv.Itoa(step) }

// pageStep reverses pageSpanID, reporting whether id names a page in this store.
func pageStep(id string) (int, bool) {
	const prefix = "page:"
	if !strings.HasPrefix(id, prefix) {
		return 0, false
	}
	n, err := strconv.Atoi(strings.TrimPrefix(id, prefix))
	if err != nil || n < 0 {
		return 0, false
	}
	return n, true
}

// Spans lowers the session's page table into ctxplan spans — SAFE metadata only. A
// quarantined page is Sealed (the planner scores it 0 and never prefers it for the
// resident view); a tombstoned page is Tombstoned. Page-in still routes through
// Session.Resolve, so clearance is honored on the demand-page path regardless of the
// Sealed planning hint, and a cleared page the re-screen clears can still be paged in.
// A page carrying a learned outcome-utility (#540) forwards it through
// Attrs["utility"], the field the planner's utility signal reads.
func (c *CtxStore) Spans(_ context.Context) ([]ctxplan.Span, error) {
	pages := c.session.Pages()
	out := make([]ctxplan.Span, 0, len(pages))
	for _, p := range pages {
		s := ctxplan.Span{
			ID:         pageSpanID(p.Step),
			Step:       p.Step,
			Role:       p.Role,
			Descriptor: p.Descriptor,
			Digest:     p.Digest,
			Bytes:      p.Len,
			Durability: ctxplan.NormDurability(p.Durability),
			Sealed:     p.Quarantined,
			Tombstoned: c.session.Tombstoned(p.Step),
		}
		if p.Utility > 0 {
			if s.Attrs == nil {
				s.Attrs = make(map[string]string, 1)
			}
			s.Attrs["utility"] = strconv.FormatFloat(p.Utility, 'f', -1, 64)
		}
		out = append(out, s)
	}
	return out, nil
}

// Materialize pages a span in through the recall trust gate. It reverses the span id to
// a page step and calls Session.Resolve, so a quarantined page surfaces as
// ctxplan.ErrSealed and a tombstoned page as ctxplan.ErrTombstoned — the planner then
// reports the span as Refused and never renders it (poison / suppression never enters
// context, even via the recovery handle). The bytes returned are BYTE-IDENTICAL to what
// Resolve hands back, and Resolve guarantees len(body) == Page.Len, so the planner's
// Span.Bytes cost contract holds without divergence.
func (c *CtxStore) Materialize(ctx context.Context, id string) ([]byte, error) {
	step, ok := pageStep(id)
	if !ok {
		return nil, fmt.Errorf("recall: ctxplan store: unknown span id %q", id)
	}
	body, err := c.session.Resolve(ctx, step)
	if err != nil {
		return nil, mapCtxPlanErr(err)
	}
	return body, nil
}

// mapCtxPlanErr re-wraps a recall page-in error so it speaks ctxplan's sealed /
// tombstoned vocabulary: the planner branches on errors.Is(err, ctxplan.ErrSealed) and
// errors.Is(err, ctxplan.ErrTombstoned). Any other error (a missing page, absent CAS
// bytes) is returned unwrapped — the planner treats it as a generic page-in refusal.
func mapCtxPlanErr(err error) error {
	switch {
	case errors.Is(err, ErrSealed):
		return fmt.Errorf("%w: %v", ctxplan.ErrSealed, err)
	case errors.Is(err, ErrTombstoned):
		return fmt.Errorf("%w: %v", ctxplan.ErrTombstoned, err)
	default:
		return err
	}
}

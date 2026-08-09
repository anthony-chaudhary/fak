package main

// The waiter plane behind `fak loop region` (#5505): what a REFUSAL does instead of evaporating.
//
// Before this, a refused region admission returned a verdict and no object — there was no line to
// stand in, so every retry re-raced from scratch and whoever polled first after a release won. A
// caller that had waited four hours had exactly the same chance as one that arrived 200ms ago.
//
// Now a refusal MINTS a ticket (internal/leasequeue.Store) whose enqueue clock survives every
// retry, and the report it prints is that waiter's place in line, the blocker holding it there,
// its bounded poll schedule and — only when the blocking lease's expiry is actually declared — an
// ETA. An ADMIT drops the ticket, so a caller that got in stops reserving a place.
//
// It is BEST-EFFORT by construction: every failure here is swallowed into a nil report. The
// admission verdict and the exit code are computed before any of this runs and are never touched
// by it, so a broken or unwritable queue degrades the REPORT and never the DECISION.

import (
	"fmt"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/leasequeue"
	"github.com/anthony-chaudhary/fak/internal/leaseref"
	"github.com/anthony-chaudhary/fak/internal/regionadmit"
)

// loopRegionQueueReport is one waiter's standing, ready to print or serialize.
type loopRegionQueueReport struct {
	Entry  leasequeue.Entry
	Depth  int
	Oldest int64
	Dir    string
}

// loopRegionHolders projects the live lease records into the queue's holder shape, carrying each
// lease's EXPIRY so a waiter can be told how long its blocker has left. A lease with no TTL has no
// declared expiry, which yields no ETA rather than a guessed one.
func loopRegionHolders(live []leaseref.Record) []leasequeue.Holder {
	out := make([]leasequeue.Holder, 0, len(live))
	for _, r := range live {
		h := leasequeue.Holder{Lease: regionadmit.Lease{ID: r.ID, Holder: r.Holder, Tree: r.TreeGlobs}}
		if r.TTLSeconds > 0 {
			from := r.AcquiredAt
			if r.RenewedAt > from {
				from = r.RenewedAt
			}
			if from > 0 {
				h.ExpiresUnix = from + r.TTLSeconds
			}
		}
		out = append(out, h)
	}
	return out
}

// loopRegionTicket is the waiter's stable identity for this exact question. The tree is RESOLVED
// through the taxonomy first, so asking by `--lane gateway` and asking by that lane's canonical
// globs mint the same ticket rather than two places in one line.
func loopRegionTicket(req regionadmit.Request, tax regionadmit.Taxonomy, class string) leasequeue.Ticket {
	tree := regionadmit.ResolveTree(req, tax)
	// The enqueue/renew/park clocks are Mint's to set — it is the half that knows whether this
	// waiter is arriving or returning, and only it may preserve a place already earned.
	return leasequeue.Ticket{
		ID:    leasequeue.TicketID(req.Actor, req.Lane, tree),
		Actor: req.Actor,
		Lane:  req.Lane,
		Tree:  tree,
		Class: class,
	}
}

// loopRegionEnqueue mints (or refreshes) this waiter's ticket and folds the whole live queue into
// its standing. A nil report means the queue was unavailable — never that the waiter was admitted.
func loopRegionEnqueue(root string, req regionadmit.Request, tax regionadmit.Taxonomy, live []leaseref.Record, class string, now time.Time) *loopRegionQueueReport {
	store, err := leasequeue.OpenStore(root)
	if err != nil {
		return nil
	}
	minted, err := store.Mint(loopRegionTicket(req, tax, class), now)
	if err != nil {
		return nil
	}
	tickets, err := store.Live(now)
	if err != nil {
		return nil
	}
	res := leasequeue.Plan(tickets, loopRegionHolders(live), tax, leasequeue.Params{NowUnix: now.Unix()})
	entry, ok := res.Find(minted.ID)
	if !ok {
		return nil
	}
	return &loopRegionQueueReport{Entry: entry, Depth: res.Depth, Oldest: res.OldestWaitSeconds, Dir: store.Dir()}
}

// loopRegionDequeue drops this waiter's ticket once it has been admitted, so a caller that got in
// stops holding a place it no longer needs. Best-effort and silent.
func loopRegionDequeue(root string, req regionadmit.Request, tax regionadmit.Taxonomy) {
	store, err := leasequeue.OpenStore(root)
	if err != nil {
		return
	}
	_ = store.Drop(leasequeue.TicketID(req.Actor, req.Lane, regionadmit.ResolveTree(req, tax)))
}

// payload is the machine-readable half, nested under "queue" in the verb's JSON.
func (r *loopRegionQueueReport) payload() map[string]any {
	if r == nil {
		return nil
	}
	e := r.Entry
	out := map[string]any{
		"schema":              leasequeue.Schema,
		"ticket":              e.ID,
		"place":               e.Place,
		"depth":               r.Depth,
		"oldest_wait_seconds": r.Oldest,
		"wait_seconds":        e.WaitSeconds,
		"standing":            string(e.Standing),
		"class":               e.Class,
		"parks":               e.Parks,
		"poll_status":         string(e.Poll.Status),
		"backoff_seconds":     e.Poll.BackoffSeconds,
		"next_retry_unix":     e.Poll.NextRetryUnix,
		"eta_known":           e.ETAKnown,
		"dir":                 r.Dir,
	}
	if e.ETAKnown {
		out["eta_seconds"] = e.ETASeconds
	}
	if e.Blocker != nil {
		out["blocker"] = map[string]any{
			"id":     e.Blocker.ID,
			"holder": e.Blocker.Holder,
			"kind":   e.BlockerKind,
			"tree":   append([]string(nil), e.Blocker.Tree...),
		}
	}
	return out
}

// line is the human half: the waiter's place, who holds it there, and when to ask again.
func (r *loopRegionQueueReport) line() string {
	if r == nil {
		return ""
	}
	e := r.Entry
	var b strings.Builder
	fmt.Fprintf(&b, "QUEUED ticket %s place %d of %d (waited %s", e.ID, e.Place, r.Depth,
		time.Duration(e.WaitSeconds)*time.Second)
	if e.Standing != "" {
		fmt.Fprintf(&b, ", %s", e.Standing)
	}
	b.WriteString(")")
	if e.Blocker != nil {
		fmt.Fprintf(&b, "; blocked by %s %s", e.BlockerKind, e.Blocker.ID)
		if e.Blocker.Holder != "" {
			fmt.Fprintf(&b, " (holder %s)", e.Blocker.Holder)
		}
		if e.ETAKnown {
			fmt.Fprintf(&b, ", expires in %s", time.Duration(e.ETASeconds)*time.Second)
		}
	}
	if e.Poll.NextRetryUnix > 0 {
		fmt.Fprintf(&b, "; retry after %s (park %d of %d)",
			time.Unix(e.Poll.NextRetryUnix, 0).UTC().Format(time.RFC3339), e.Parks, e.Poll.MaxParks)
	}
	return b.String()
}

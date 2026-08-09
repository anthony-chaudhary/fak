// Package leasequeue is the WAITER PLANE a region-admission refusal never had.
//
// # The gap it closes
//
// internal/regionadmit answers "may this actor act on this region right now?" and, when the
// answer is no, returns a verdict and NO OBJECT. Every refusal site in the shipped binary drops
// it on the floor:
//
//   - cmd/fak/loop_region.go prints REFUSE and exits 3;
//   - cmd/fak/loop_drive_region.go returns a refusal struct and honest-stops;
//   - cmd/fak/dispatch_tick.go returns a map with refused:true.
//
// None of them records that a waiter EXISTS. So there is no line to stand in: every retry
// re-races from scratch and whoever polls first after a release wins. A caller that has waited
// four hours has exactly the same chance as one that arrived 200ms ago — the claim plane is
// scheduled by lottery.
//
// This package is the missing object. Given the set of waiting tickets, the live holders and the
// lane taxonomy, Plan returns each waiter's PLACE IN LINE, the blocker that holds it there, its
// poll schedule and — when the blocker's expiry is actually known — an ETA.
//
// # One predicate, one aging law (no second scheduler)
//
// It computes NOTHING itself. Every conflict test is regionadmit.Decide, the same predicate the
// dispatch tick and `fak loop drive` already run. The order is dispatchaging.Fold, the same
// anti-starvation law the issue-dispatch order uses — applied over each waiter's ENQUEUE clock,
// so the "no ready unit waits forever" guarantee survives the lease hop instead of stopping at
// the pick. The poll schedule is seatpark.Decide, the same bounded backoff the no-seat transient
// already uses. A private twin of any of the three would be the failure mode, not the feature.
//
// # Conservative backfill
//
// Waiters are walked in rank order. Each is tested against the live holders PLUS this pass's
// grants PLUS every higher-ranked still-blocked waiter's RESERVATION. A lower-ranked waiter may
// therefore be granted only when it conflicts with nobody ahead of it — HPC conservative
// backfill. It is cheap here because the conflict test is static tree geometry: no runtime
// duration estimate is needed, so no estimate has to be invented.
//
// One waiter list serves both contention shapes, because they fall out of the same predicate: a
// named lane is a mutex (one line) while a tree is a conflict graph (overlap is not transitive).
// A waiter's Place counts only the waiters ahead of it that ACTUALLY conflict with it, so a
// waiter on a disjoint tree is never told it is fifth in a line it is not standing in.
//
// # Pure, and honest about what it is not
//
// Plan reads no clock (Params.NowUnix is data) and does no I/O — same input, same Result. It
// HOLDS nothing, ACQUIRES nothing, REAPS nothing, and never infers a holder's liveness: a
// stalled holder that stopped heartbeating is still a blocker here. Ticket minting/reading is the
// one I/O half and lives in store.go, off the pure path, the same leaf/shell split regionadmit
// uses for LoadTaxonomy.
//
// A zero-value Params leaves the order at ARRIVAL semantics (dispatchaging with aging disabled
// orders by base weight, then wait, then ID), so wiring this in changes no existing decision
// until the knobs are turned on.
//
// Tier: mechanism (2) — see internal/architest.
package leasequeue

import (
	"sort"

	"github.com/anthony-chaudhary/fak/internal/dispatchaging"
	"github.com/anthony-chaudhary/fak/internal/regionadmit"
	"github.com/anthony-chaudhary/fak/internal/seatpark"
)

// Schema tags the machine-readable Result.
const Schema = "fak.leasequeue.v1"

// The priority CLASS of a waiter. The class already exists in the admission refusal text ("you
// are an interactive operator (not a dispatch loop)") but not in the ORDER, so an operator queues
// behind every background loop. This is the closed vocabulary that puts it in the order.
const (
	// ClassInteractive is an operator at a terminal, waiting on the answer.
	ClassInteractive = "interactive"
	// ClassLoop is a background dispatch/drive loop, which can poll indefinitely.
	ClassLoop = "loop"
	// ClassUnknown is an unclassified waiter; it ranks as a loop (the conservative direction —
	// an unclassified caller never queue-jumps an operator).
	ClassUnknown = ""
)

// ClassWeight maps a priority class onto the SAME priority-weight taxonomy dispatchaging already
// declares for the dispatch order (P0=1000, P1=400, P2=150, unlabeled=60 — see the constant
// block in internal/dispatchaging). Nothing new is invented: an interactive operator is given the
// P1 tier and a background loop the unlabeled tier, which is the whole of the class policy.
//
// The hard starvation deadline still applies on top, so a hot class can NEVER starve a cold one:
// a loop that waits past dispatchaging.Params.StarvationSeconds is force-promoted ahead of every
// non-starved interactive waiter. Callers that want a different policy set Ticket.BaseWeight
// directly and never call this.
func ClassWeight(class string) int {
	switch class {
	case ClassInteractive:
		return 400
	default:
		return 60
	}
}

// Ticket is one waiter's durable claim on its place in line — the object a refusal mints instead
// of evaporating. Its ID is stable across retries (see TicketID), which is the entire mechanism:
// a re-attempt REFRESHES one ticket rather than minting a new one, so the enqueue clock is
// preserved and repeated attempts are ordered by arrival instead of re-raced from scratch.
type Ticket struct {
	// ID is the waiter's stable identity across retries.
	ID string `json:"id"`
	// Actor names the would-be holder (a session id, "loop:nightly", host:pid).
	Actor string `json:"actor,omitempty"`
	// Lane and Tree are the region the waiter is asking for — the same pair regionadmit.Request
	// carries, so the queue asks the admission predicate exactly the caller's own question.
	Lane string   `json:"lane,omitempty"`
	Tree []string `json:"tree,omitempty"`
	// Class is the priority class (see ClassWeight). Advisory: BaseWeight wins when set.
	Class string `json:"class,omitempty"`
	// BaseWeight is the priority weight the aging law boosts. <= 0 falls back to ClassWeight.
	BaseWeight int `json:"base_weight,omitempty"`
	// EnqueuedUnix is when the waiter FIRST joined the line — the wait clock the aging law reads.
	// It is preserved across every refresh; that preservation is what makes the order fair.
	EnqueuedUnix int64 `json:"enqueued_unix"`
	// RenewedUnix is the waiter's most recent re-attempt: its liveness heartbeat. A waiter that
	// stops polling stops renewing and its ticket lapses, so an abandoned waiter cannot reserve a
	// region forever.
	RenewedUnix int64 `json:"renewed_unix,omitempty"`
	// TTLSeconds is how long a ticket survives without a renewal. <= 0 never lapses.
	TTLSeconds int64 `json:"ttl_seconds,omitempty"`
	// Parks counts consecutive refused polls; it drives the seatpark backoff window.
	Parks int `json:"parks,omitempty"`
	// LastParkUnix is when the waiter was most recently refused (the backoff anchor).
	LastParkUnix int64 `json:"last_park_unix,omitempty"`
}

// lastSeenUnix is the later of the enqueue and renew stamps — the instant the lapse window is
// measured from.
func (t Ticket) lastSeenUnix() int64 {
	if t.RenewedUnix > t.EnqueuedUnix {
		return t.RenewedUnix
	}
	return t.EnqueuedUnix
}

// Lapsed reports whether the waiter stopped polling long enough that its ticket is abandoned. A
// ticket with no TTL, or with no clock at all, never lapses: this predicate may drop a GHOST, it
// may never drop a waiter that is still standing in line.
func (t Ticket) Lapsed(nowUnix int64) bool {
	if t.TTLSeconds <= 0 || t.lastSeenUnix() <= 0 {
		return false
	}
	return nowUnix-t.lastSeenUnix() > t.TTLSeconds
}

// weight is the ticket's base priority weight: an explicit BaseWeight, else its class weight.
func (t Ticket) weight() int {
	if t.BaseWeight > 0 {
		return t.BaseWeight
	}
	return ClassWeight(t.Class)
}

// request is the admission question this waiter is asking. SelfID is the ticket id so a waiter's
// own reservation is never counted as its own blocker.
func (t Ticket) request() regionadmit.Request {
	return regionadmit.Request{Actor: t.Actor, Lane: t.Lane, Tree: t.Tree, SelfID: t.ID}
}

// reservation projects the waiter into the lease shape the predicate tests against, so a
// higher-ranked waiter's claim blocks a lower-ranked one exactly as a live holder would. The tree
// is resolved through the taxonomy for the same reason the dispatch tick resolves it before
// recording: a named lane with no explicit tree means the lane's canonical tree, not "unknown".
func (t Ticket) reservation(tax regionadmit.Taxonomy) regionadmit.Lease {
	return regionadmit.Lease{
		ID:     t.ID,
		Holder: t.Actor,
		Lane:   t.Lane,
		Tree:   regionadmit.ResolveTree(regionadmit.Request{Lane: t.Lane, Tree: t.Tree}, tax),
	}
}

// Holder is one live lease as the queue sees it: the admission projection plus the expiry the ETA
// is read from. ExpiresUnix is separate from the Lease because regionadmit deliberately carries
// no clock — admission is geometry, not time.
type Holder struct {
	// Lease is the live holder in the shape the admission predicate consumes.
	Lease regionadmit.Lease `json:"lease"`
	// ExpiresUnix is when this holder's lease lapses (unix seconds). 0 == UNKNOWN, and an unknown
	// expiry yields no ETA at all rather than a guessed one.
	ExpiresUnix int64 `json:"expires_unix,omitempty"`
}

// Params are the knobs. The ZERO value is arrival semantics with the documented seatpark backoff:
// aging disabled, so the order is base weight then wait then ID.
type Params struct {
	// NowUnix is the clock, supplied as data — this package never reads one.
	NowUnix int64 `json:"now_unix"`
	// Aging is the anti-starvation law applied over each waiter's enqueue clock. Its own NowUnix
	// is overwritten with Params.NowUnix so the two clocks can never disagree.
	Aging dispatchaging.Params `json:"aging"`
	// Poll is the bounded backoff schedule handed to each queued waiter.
	Poll seatpark.Policy `json:"poll"`
}

// Entry is one waiter's full standing: where it sits, what holds it there, and when to poll.
type Entry struct {
	Ticket
	// Rank is the 0-based position in the aged order over ALL waiters (grants included).
	Rank int `json:"rank"`
	// Grant reports that this pass would admit the waiter — it is at the head of its own line.
	Grant bool `json:"grant"`
	// Place is the waiter's 1-based place in the line it is ACTUALLY standing in: 1 + the number
	// of higher-ranked blocked waiters that genuinely conflict with it. 0 on a grant. This is the
	// number the ticket's "3rd in line" is; it is per-conflict-graph, never a global count, so a
	// waiter on a disjoint tree is not told it is behind a line it does not share.
	Place int `json:"place,omitempty"`
	// Blocker names what holds this waiter, as evidence. Nil on a grant.
	Blocker *regionadmit.Lease `json:"blocker,omitempty"`
	// BlockerKind is "lease" (a live holder) or "waiter" (a higher-ranked reservation) — the
	// difference between "someone is working" and "someone is ahead of you in line".
	BlockerKind string `json:"blocker_kind,omitempty"`
	// Reason and Rung are the admission refusal's closed tokens, passed through unchanged.
	Reason string `json:"reason,omitempty"`
	Rung   string `json:"rung,omitempty"`
	Detail string `json:"detail,omitempty"`
	// WaitSeconds is how long the waiter has stood in line (the evidence the order rests on).
	WaitSeconds int64 `json:"wait_seconds"`
	// Standing and EffectiveWeight are the aging verdict: fresh | aging | starved.
	Standing        dispatchaging.Standing `json:"standing"`
	EffectiveWeight int                    `json:"effective_weight"`
	// Poll is the bounded retry schedule for this waiter (when to ask again, and when to stop).
	Poll seatpark.Decision `json:"poll"`
	// ETASeconds is how long until the blocking HOLDER's lease expires. It is meaningful only
	// when ETAKnown is true: a blocker with no declared expiry, and a blocker that is another
	// waiter, produce no ETA rather than an invented one.
	ETASeconds int64 `json:"eta_seconds,omitempty"`
	ETAKnown   bool  `json:"eta_known"`
}

// BlockedByWaiter and BlockedByLease are the two BlockerKind values.
const (
	BlockedByLease  = "lease"
	BlockedByWaiter = "waiter"
)

// Result is the whole waiter plane for one pass.
type Result struct {
	Schema string `json:"schema"`
	// Entries is every live waiter in aged rank order.
	Entries []Entry `json:"entries"`
	// Granted is the IDs this pass would admit, in rank order.
	Granted []string `json:"granted,omitempty"`
	// Depth is how many waiters remain blocked — the demand nothing could previously compute.
	Depth int `json:"depth"`
	// OldestWaitSeconds is the longest any live waiter has stood in line. This is the answer to
	// "what is the oldest wait on this lane?" that no surface could previously give.
	OldestWaitSeconds int64 `json:"oldest_wait_seconds"`
	// Lapsed is the IDs dropped as abandoned before ranking (evidence, so a vanished waiter is
	// explained rather than silently missing).
	Lapsed []string `json:"lapsed,omitempty"`
}

// Find returns the entry for a waiter id.
func (r Result) Find(id string) (Entry, bool) {
	for _, e := range r.Entries {
		if e.ID == id {
			return e, true
		}
	}
	return Entry{}, false
}

// Plan is THE waiter-plane decision: same tickets + same holders + same clock in, same Result
// out. No clock read, no I/O, total over any input.
//
// The policy, in order:
//
//  1. abandon lapsed tickets (a waiter that stopped polling is not standing in line);
//  2. rank the survivors with dispatchaging.Fold over their ENQUEUE clock, so the dispatch
//     order's anti-starvation guarantee crosses the lease hop;
//  3. walk in rank order, testing each waiter with regionadmit.Decide against the live holders
//     plus this pass's grants plus every higher-ranked blocked waiter's reservation — conservative
//     backfill, so a late arrival on a free region still cannot overtake a waiter ahead of it on
//     a region they share;
//  4. hand every blocked waiter its place, its blocker, its seatpark poll schedule, and an ETA
//     only when the blocking holder's expiry is actually declared.
func Plan(tickets []Ticket, holders []Holder, tax regionadmit.Taxonomy, p Params) Result {
	out := Result{Schema: Schema}

	live := make([]regionadmit.Lease, 0, len(holders))
	expiry := make(map[string]int64, len(holders))
	for _, h := range holders {
		live = append(live, h.Lease)
		if h.ExpiresUnix > 0 {
			expiry[h.Lease.ID] = h.ExpiresUnix
		}
	}

	byID := make(map[string]Ticket, len(tickets))
	cands := make([]dispatchaging.Candidate, 0, len(tickets))
	for _, t := range tickets {
		if t.Lapsed(p.NowUnix) {
			out.Lapsed = append(out.Lapsed, t.ID)
			continue
		}
		byID[t.ID] = t
		cands = append(cands, dispatchaging.Candidate{
			ID:         t.ID,
			BaseWeight: t.weight(),
			ReadySince: t.EnqueuedUnix,
		})
	}
	sort.Strings(out.Lapsed)

	aging := p.Aging
	aging.NowUnix = p.NowUnix
	order := dispatchaging.Fold(cands, aging)
	out.OldestWaitSeconds = order.OldestWaitSeconds

	// grants accumulates this pass's admissions; reserved accumulates the still-blocked waiters
	// ahead of the current one. Both are fed to the predicate as if they were live leases, which
	// is what makes the backfill conservative.
	granted := append([]regionadmit.Lease(nil), live...)
	var reserved []Entry

	for _, r := range order.Order {
		t, ok := byID[r.ID]
		if !ok {
			continue
		}
		e := Entry{
			Ticket:          t,
			Rank:            r.Rank,
			WaitSeconds:     r.WaitSeconds,
			Standing:        r.Standing,
			EffectiveWeight: r.EffectiveWeight,
		}

		// A fresh slice each pass: appending into granted's spare capacity would let one
		// waiter's reservation leak into the next waiter's admission set.
		against := make([]regionadmit.Lease, 0, len(granted)+len(reserved))
		against = append(against, granted...)
		for i := range reserved {
			against = append(against, reserved[i].reservation(tax))
		}
		dec := regionadmit.Decide(t.request(), against, tax)
		if dec.Admit {
			e.Grant = true
			e.Poll = seatpark.Decide(seatpark.Input{TaskID: t.ID, NowUnix: p.NowUnix, Policy: p.Poll})
			granted = append(granted, t.reservation(tax))
			out.Granted = append(out.Granted, t.ID)
			out.Entries = append(out.Entries, e)
			continue
		}

		e.Reason, e.Rung, e.Detail = dec.Reason, dec.Rung, dec.Detail
		if dec.Conflict != nil {
			c := *dec.Conflict
			e.Blocker = &c
			e.BlockerKind = BlockedByLease
			if _, isWaiter := byID[c.ID]; isWaiter {
				e.BlockerKind = BlockedByWaiter
			}
			if exp, known := expiry[c.ID]; known {
				e.ETAKnown = true
				if eta := exp - p.NowUnix; eta > 0 {
					e.ETASeconds = eta
				}
			}
		}
		e.Place = 1 + aheadOf(t, reserved, tax)
		e.Poll = seatpark.Decide(seatpark.Input{
			TaskID:       t.ID,
			Parks:        t.Parks,
			LastParkUnix: t.LastParkUnix,
			NowUnix:      p.NowUnix,
			Policy:       p.Poll,
		})
		out.Depth++
		reserved = append(reserved, e)
		out.Entries = append(out.Entries, e)
	}
	return out
}

// aheadOf counts how many already-blocked waiters genuinely conflict with t — the waiters that
// must clear before t can be admitted, and therefore the ones t is actually standing behind. A
// higher-ranked waiter on a disjoint region is ahead in the ORDER but not in t's LINE, so it is
// not counted: the place a waiter is told is the place it is really in.
func aheadOf(t Ticket, reserved []Entry, tax regionadmit.Taxonomy) int {
	n := 0
	for i := range reserved {
		if !regionadmit.Decide(t.request(), []regionadmit.Lease{reserved[i].reservation(tax)}, tax).Admit {
			n++
		}
	}
	return n
}

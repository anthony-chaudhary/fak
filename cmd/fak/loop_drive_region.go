package main

// The loop-drive region hold: `fak loop drive` participating in the SAME lease
// fabric the dispatch tick already writes (internal/leaseref refs/fak/locks/*),
// admitted by the SAME decision (internal/regionadmit). A GOAL.md that declares
// `lane:` and/or `region:` (or the --lane/--tree flags) makes the drive:
//
//   - refuse to start over a live overlapping lease (COLLISION_RISK, with the
//     conflicting lease named as evidence) instead of silently racing it;
//   - hold a fenced lease on its region while it runs, renewed each turn, so
//     dispatch workers, other loop drives, and manual sessions that consult
//     the fabric see this loop and stay off its tree;
//   - honest-stop when its lease is reaped and taken over mid-drive (the
//     fence's STALE_LEASE — the paused-then-resumed-holder hazard).
//
// A GOAL.md with no lane and no region keeps the historical uncoordinated
// drive byte-for-byte. Infra errors (unreadable lease store) fail OPEN with a
// stderr warning — the same posture as the dispatch tick's acquire — while a
// WITNESSED collision always refuses.

import (
	"context"
	"fmt"
	"io"
	"os"
	"strings"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/leaseref"
	"github.com/anthony-chaudhary/fak/internal/loopdrive"
	"github.com/anthony-chaudhary/fak/internal/loopmgr"
	"github.com/anthony-chaudhary/fak/internal/regionadmit"
)

// loopDriveRegionTTLS is the region lease TTL when the drive has no deadline.
// Renewed every turn; sized so one long agent turn cannot silently expire the
// lease mid-drive, while a crashed drive's ghost stays bounded (garden reaps).
const loopDriveRegionTTLS int64 = 3600

// loopDriveHoldSeq disambiguates holds created inside one process — the pid
// alone cannot (see the holder comment in newLoopDriveRegionHold).
var loopDriveHoldSeq uint64

// loopDriveRegionRefusal is a structured region refusal: Reason is from the
// closed vocabulary (COLLISION_RISK, or a leaseref fence token such as
// STALE_LEASE / LEASE_CONTENDED), Detail names the conflicting evidence.
type loopDriveRegionRefusal struct {
	Reason string
	Detail string
}

type loopDriveRegionHold struct {
	store  *leaseref.Store
	tax    regionadmit.Taxonomy
	id     string
	holder string
	lane   string
	tree   []string
	ttl    int64
	held   bool
}

// newLoopDriveRegionHold resolves the drive's region config: flag overrides
// win over the GOAL.md spec; both empty returns nil (region admission off).
func newLoopDriveRegionHold(opt loopDriveOptions, spec loopdrive.Spec) *loopDriveRegionHold {
	lane := strings.TrimSpace(opt.Lane)
	if lane == "" {
		lane = strings.TrimSpace(spec.Lane)
	}
	tree := opt.Region
	if len(tree) == 0 {
		tree = spec.Region
	}
	if lane == "" && len(tree) == 0 {
		return nil
	}
	ttl := loopDriveRegionTTLS
	if !opt.Deadline.IsZero() {
		if remaining := int64(time.Until(opt.Deadline).Seconds()) + 300; remaining > ttl {
			ttl = remaining
		}
	}
	tax, err := regionadmit.LoadTaxonomy(".")
	if err != nil {
		// No taxonomy is not fatal: the decision still enforces tree geometry;
		// lane serialization/exclusivity just has no lane data to act on.
		tax = regionadmit.Taxonomy{}
	}
	return &loopDriveRegionHold{
		store: leaseref.NewInDir(""),
		tax:   tax,
		id:    "loop-" + cleanDispatchLeaseToken(spec.Loop),
		// The holder must be PROCESS-unique: dispatchLeaseHolder alone can be
		// pinned fleet-wide (FAK_LEASE_OWNER / CLAUDE_CODE_SESSION_ID), and a
		// same-holder AcquireFenced silently RENEWs — two concurrent drives of
		// one loop would then both believe they hold the region. The pid
		// suffix makes the second drive refuse (LEASE_HELD); the tradeoff is
		// that a crash-restarted drive waits out its predecessor's TTL. The
		// per-process sequence keeps two holds unique even inside one process.
		holder: fmt.Sprintf("loop:%s@%s#%d-%d", spec.Loop, dispatchLeaseHolder(), os.Getpid(), atomic.AddUint64(&loopDriveHoldSeq, 1)),
		lane:   lane,
		tree:   tree,
		ttl:    ttl,
	}
}

// ensure makes the region hold true for the coming turn: the first call
// decides admission against the live lease set and acquires the fenced lease;
// later calls renew it. A structured refusal means the drive must honest-stop;
// an error is an infra failure the caller warns about and fails open on.
//
// Renew outcomes are split by what they witness: STALE_LEASE means a peer
// holds the region NOW (terminal — honest-stop); NO_LEASE means the lease
// lapsed with no taker (e.g. one turn outran the TTL) — that is not a peer
// conflict, so the hold falls through to a fresh admission + reacquire, and a
// peer who took the region meanwhile refuses THERE. LEASE_CONTENDED is a lost
// CAS against a racing reaper — transient by contract, retried once.
func (h *loopDriveRegionHold) ensure(now time.Time) (*loopDriveRegionRefusal, error) {
	if h == nil {
		return nil, nil
	}
	ctx := context.Background()
	if h.held {
		verdict, err := h.renewOnce(ctx, now)
		if err != nil {
			return nil, err
		}
		switch {
		case verdict.OK:
			return nil, nil
		case string(verdict.Reason) == leaseref.ReasonNoLease:
			h.held = false // lapsed, untaken: fall through to reacquire
		default:
			h.held = false
			return &loopDriveRegionRefusal{
				Reason: string(verdict.Reason),
				Detail: fmt.Sprintf("region lease %s lost mid-drive: %s", h.id, verdict.Detail),
			}, nil
		}
	}
	live, _, err := h.store.Live(ctx, now)
	if err != nil {
		return nil, fmt.Errorf("read live leases: %w", err)
	}
	dec := regionadmit.Decide(regionadmit.Request{
		Actor:  h.holder,
		Lane:   h.lane,
		Tree:   h.tree,
		SelfID: h.id,
	}, regionLeases(live), h.tax)
	if !dec.Admit {
		return &loopDriveRegionRefusal{Reason: dec.Reason, Detail: dec.Detail}, nil
	}
	rec := leaseref.Record{
		ID:         h.id,
		TreeGlobs:  regionadmit.ResolveTree(regionadmit.Request{Lane: h.lane, Tree: h.tree}, h.tax),
		Holder:     h.holder,
		TTLSeconds: h.ttl,
	}
	verdict, err := h.acquireOnce(ctx, rec, now)
	if err != nil {
		return nil, err
	}
	if !verdict.OK {
		return &loopDriveRegionRefusal{
			Reason: string(verdict.Reason),
			Detail: fmt.Sprintf("region lease %s: %s", h.id, verdict.Detail),
		}, nil
	}
	h.held = true
	return nil, nil
}

// renewOnce renews the held lease, retrying a single LEASE_CONTENDED (a lost
// CAS is transient — "re-read and retry" is the fence's own contract).
func (h *loopDriveRegionHold) renewOnce(ctx context.Context, now time.Time) (leaseref.FenceVerdict, error) {
	_, verdict, err := h.store.Renew(ctx, h.id, h.holder, h.ttl, now)
	if err != nil {
		return leaseref.FenceVerdict{}, fmt.Errorf("renew region lease %s: %w", h.id, err)
	}
	if !verdict.OK && string(verdict.Reason) == leaseref.ReasonLeaseContended {
		_, verdict, err = h.store.Renew(ctx, h.id, h.holder, h.ttl, now)
		if err != nil {
			return leaseref.FenceVerdict{}, fmt.Errorf("renew region lease %s: %w", h.id, err)
		}
	}
	return verdict, nil
}

// acquireOnce acquires the lease with the same single LEASE_CONTENDED retry.
func (h *loopDriveRegionHold) acquireOnce(ctx context.Context, rec leaseref.Record, now time.Time) (leaseref.FenceVerdict, error) {
	_, verdict, err := h.store.AcquireFenced(ctx, rec, now)
	if err != nil {
		return leaseref.FenceVerdict{}, fmt.Errorf("acquire region lease %s: %w", h.id, err)
	}
	if !verdict.OK && string(verdict.Reason) == leaseref.ReasonLeaseContended {
		_, verdict, err = h.store.AcquireFenced(ctx, rec, now)
		if err != nil {
			return leaseref.FenceVerdict{}, fmt.Errorf("acquire region lease %s: %w", h.id, err)
		}
	}
	return verdict, nil
}

// release drops the held lease. Nil-safe and idempotent; an unheld or already
// released hold is a no-op.
func (h *loopDriveRegionHold) release() {
	if h == nil || !h.held {
		return
	}
	h.held = false
	_ = h.store.Release(context.Background(), h.id)
}

// evidence is the ledger evidence ref for a held region lease, nil otherwise.
func (h *loopDriveRegionHold) evidence() []loopmgr.EvidenceRef {
	if h == nil || !h.held {
		return nil
	}
	return []loopmgr.EvidenceRef{{Kind: "region_lease", Ref: h.id}}
}

// refuseLoopDriveRegion records the structured region refusal on the loop
// ledger and reports it, mirroring the governor-refusal exit shape (exit 3).
func refuseLoopDriveRegion(stderr io.Writer, opt loopDriveOptions, goalPath string, spec loopdrive.Spec, hold *loopDriveRegionHold, refuse *loopDriveRegionRefusal, iterations int, tokensUsed int64) int {
	ev := []loopmgr.EvidenceRef{
		{Kind: "goal", Ref: goalPath},
	}
	if hold != nil {
		ev = append(ev, loopmgr.EvidenceRef{Kind: "region_lease", Ref: hold.id})
	}
	if err := appendLoopRunEvent(opt.LedgerPath, loopmgr.Event{
		LoopID:       spec.Loop,
		Kind:         loopmgr.EventAdmit,
		Source:       opt.Source,
		Principal:    opt.Principal,
		Status:       loopmgr.StatusRefused,
		Reason:       refuse.Reason,
		Summary:      refuse.Detail,
		EvidenceRefs: ev,
		Metrics:      map[string]int64{"iterations": int64(iterations), "tokens_used": tokensUsed},
	}); err != nil {
		fmt.Fprintf(stderr, "fak loop drive: %v\n", err)
		return 1
	}
	fmt.Fprintf(stderr, "fak loop drive: refused by region admission: %s %s\n", refuse.Reason, refuse.Detail)
	return 3
}

// regionLeases projects live leaseref records into the region-admission lease
// shape; the decision infers each lease's lane from its tree.
func regionLeases(recs []leaseref.Record) []regionadmit.Lease {
	out := make([]regionadmit.Lease, 0, len(recs))
	for _, r := range recs {
		out = append(out, regionadmit.Lease{ID: r.ID, Holder: r.Holder, Tree: r.TreeGlobs})
	}
	return out
}

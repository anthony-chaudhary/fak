package main

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchorder"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/leaseref"
)

// leasewrite_endpoint.go wires the multi-node dev-server WRITE plane (#2299, epic #2254
// plane 1 — the atomicity closure): POST /v1/leases/{acquire,renew,release} over the
// SAME leaseref store the read plane (leaseplane_endpoint.go) and the `fak leaseref` CLI
// verbs use. The gateway is leaseref-blind; this file is the one seam that translates the
// injected gateway request into leaseref.AcquireFenced / Renew / ReleaseFenced and folds
// the leaseref.FenceVerdict back into the gateway's wire shape.
//
// The gateway serializes calls into this function (leaseWriteMu), so it is already a
// single arbiter; this closure adds no lock of its own. On an ACCEPTED write it best-effort
// publishes the accepted lease to origin (plane 0) so offline nodes converge too — a push
// failure never fails the arbitrated verdict (the CAS already committed locally; the
// coordinator's store is authoritative for reachable nodes, and the next `fak leaseref
// sync` reconverges the laggards), so a publish error is dropped, not returned.
//
// That publish is HANDED OFF, never performed inline (#5422). The gateway holds
// leaseWriteMu across this whole call, so a git push executed here would put a network
// round trip inside the single-arbiter critical section and queue every concurrent lease
// write behind the slowest push. leasePublishQueue takes the request and returns at once;
// the push runs on the queue's own goroutine after this function — and therefore the
// mutex — has been released.
func init() {
	gateway.SetLeaseWriteFunc(serveLeaseWrite)
}

// serveLeaseWrite is the single-arbiter fenced-write body: it dispatches on the verb,
// performs the fenced write against the coordinator clone, and returns the deny-as-value
// verdict. A non-nil error is reserved for INFRASTRUCTURE failure (git not executable, an
// unreadable record); every policy refusal is an ok:false result with a nil error.
func serveLeaseWrite(ctx context.Context, op string, req gateway.LeaseWriteRequest) (gateway.LeaseWriteResult, error) {
	now := time.Now()
	store := leaseref.NewInDir(leasePlaneDir())

	// settle states the post-write policy every verb shares exactly once. An
	// INFRASTRUCTURE error propagates untouched. An ACCEPTED write REQUESTS a publish to
	// origin (plane 0) so offline nodes converge — a release included, which converges as
	// a deletion. The request is a hand-off, not a push: it latches work onto
	// leasePublishes and returns immediately, so the arbiter's critical section never
	// contains a network round trip. That publish is best-effort exactly as before: a push
	// failure never un-accepts the arbitrated verdict, because the local CAS already
	// committed and is authoritative here. Every verdict, accept or refusal, then folds to
	// the gateway's wire shape.
	settle := func(rec leaseref.Record, v leaseref.FenceVerdict, err error) (gateway.LeaseWriteResult, error) {
		if err != nil {
			return gateway.LeaseWriteResult{}, err
		}
		if v.OK {
			leasePublishes.request(store)
		}
		return leaseVerdictToResult(op, req, rec, v), nil
	}

	switch op {
	case "acquire":
		live, _, err := store.StrictLiveSnapshot(ctx, now)
		if err != nil {
			return gateway.LeaseWriteResult{
				OK:     false,
				Reason: leaseref.ReasonLeaseHeld,
				Op:     op,
				ID:     req.ID,
				Detail: fmt.Sprintf("coordinator authority check failed: %v", err),
			}, nil
		}
		for _, active := range live {
			if active.ID == req.ID {
				continue
			}
			if leaseTreesOverlap(req.TreeGlobs, active.TreeGlobs) {
				return gateway.LeaseWriteResult{
					OK:                false,
					Reason:            leaseref.ReasonLeaseHeld,
					Op:                op,
					ID:                req.ID,
					Generation:        req.Generation,
					CurrentGeneration: active.Generation,
					Holder:            active.Holder,
					Detail:            fmt.Sprintf("tree overlap with live lease %s held by %q (generation %d)", active.ID, active.Holder, active.Generation),
				}, nil
			}
		}

		return settle(store.AcquireFenced(ctx, leaseref.Record{
			ID:          req.ID,
			TreeGlobs:   req.TreeGlobs,
			Holder:      req.Holder,
			TTLSeconds:  req.TTLSeconds,
			Description: req.Description,
		}, now))

	case "renew":
		return settle(store.Renew(ctx, req.ID, req.Holder, req.TTLSeconds, now))

	case "release":
		// A release deletes the record rather than writing one, so there is nothing to
		// fold back beyond the verdict itself.
		v, err := store.ReleaseFenced(ctx, req.ID, req.Holder, req.Generation, now)
		return settle(leaseref.Record{}, v, err)
	}

	// The gateway already rejects an unknown verb with 404 before it reaches here; this is
	// the defensive fail-closed for a future caller that bypasses that check.
	return gateway.LeaseWriteResult{
		OK:     false,
		Reason: "UNKNOWN_OP",
		Op:     op,
		ID:     req.ID,
		Detail: "unknown lease write verb " + op,
	}, nil
}

// leaseVerdictToResult folds a leaseref.FenceVerdict (and, on an accepted acquire/renew,
// the written Record) into the gateway's wire verdict. The closed reason vocabulary
// (LEASE_HELD / STALE_LEASE / LEASE_CONTENDED / NO_LEASE) crosses UNCHANGED — the
// gateway's --json contract is exactly leaseref's, so the HTTP plane and the CLI can never
// drift apart. On OK the Generation carries the assigned fencing token from the written
// record; on a refusal it carries the verdict's presented/current generations so the
// caller learns who actually owns the lease.
func leaseVerdictToResult(op string, req gateway.LeaseWriteRequest, rec leaseref.Record, v leaseref.FenceVerdict) gateway.LeaseWriteResult {
	res := gateway.LeaseWriteResult{
		OK:                v.OK,
		Reason:            v.Reason,
		Op:                op,
		ID:                req.ID,
		CurrentGeneration: v.Current,
		Detail:            v.Detail,
	}
	if v.OK {
		res.Generation = rec.Generation
		res.Holder = rec.Holder
		res.TreeGlobs = rec.TreeGlobs
		if res.Holder == "" {
			res.Holder = req.Holder
		}
	} else {
		// A refusal names who actually holds the live lease and at what token, so the
		// caller can halt-and-reacquire rather than guess.
		res.Generation = v.Presented
		res.Holder = v.Holder
	}
	return res
}

// publishAcceptedLease pushes the local refs/fak/locks/* namespace to origin (plane 0)
// so nodes that cannot reach this coordinator still converge on the accepted lease via a
// git fetch. Best-effort: a push failure is dropped here — the arbitrated verdict stands
// on the coordinator's authoritative local store, and `fak leaseref sync` reconverges the
// laggards on the next tick.
func publishAcceptedLease(ctx context.Context, store *leaseref.Store) {
	_, _ = store.Sync(ctx, "origin", true, false)
}

// leasePublish is the plane-0 publish BOUNDARY, held as a package var so a test can
// observe the hand-off without a real network push. Production is publishAcceptedLease.
var leasePublish = publishAcceptedLease

// leasePublishTimeout bounds ONE asynchronous publish. The push no longer rides the
// request's context — that one is cancelled the moment the arbiter's response is written —
// so it carries its own deadline: a wedged `git push` must expire rather than pin the
// publisher goroutine forever. It is deliberately wider than the gateway's 15s write
// timeout that used to bound it inline, because it no longer holds a caller waiting.
const leasePublishTimeout = 30 * time.Second

// leasePublishes is the process-wide publisher. One queue, because the thing being
// published is one namespace (refs/fak/locks/*) on one coordinator clone.
var leasePublishes leasePublishQueue

// leasePublishQueue moves the plane-0 publish OFF the gateway's leaseWriteMu critical
// section (#5422) without reordering a publish ahead of the accepted write it represents.
//
// THE ORDERING GUARANTEE IT KEEPS — publish-after-write, serialized, coalescing:
//
//   - SERIALIZED. At most one publish is ever in flight. Concurrent pushes of the same
//     refspec to the same remote could otherwise land out of order and leave origin
//     holding an OLDER namespace snapshot than one already published.
//   - PUBLISH-AFTER-WRITE. request() is called only after the local CAS has committed, so
//     the publish it causes always begins strictly after that write is readable in the
//     store. A push therefore never carries a namespace older than the write that asked
//     for it.
//   - COALESCING, NOT DROPPING. If a publish is already in flight when request() lands,
//     that push may have read the ref store BEFORE this write committed, so it cannot be
//     assumed to cover it: the request is LATCHED and the runner starts a fresh push as
//     soon as the current one returns. No accepted write is left unpublished; several
//     accepted writes may be covered by one push, which is sound because Sync publishes
//     the whole namespace snapshot rather than a single record.
//
// What deliberately changed: the publish is no longer complete when the 200 is written. It
// was already best-effort and eventually-convergent (a push failure was, and still is,
// dropped), so no caller could have relied on origin being current at response time.
type leasePublishQueue struct {
	mu       sync.Mutex
	inFlight bool            // a publish goroutine is running
	pending  bool            // an accepted write landed while one was running
	store    *leaseref.Store // the store the next publish reads
	idle     chan struct{}   // closed when the queue drains; nil while idle
}

// request latches a plane-0 publish for store and returns IMMEDIATELY. It is the only
// call the arbiter makes, and it performs no I/O, so nothing here can extend the
// leaseWriteMu critical section beyond a mutex hand-off.
func (q *leasePublishQueue) request(store *leaseref.Store) {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.store = store
	if q.inFlight {
		q.pending = true
		return
	}
	q.inFlight = true
	q.idle = make(chan struct{})
	go q.run()
}

// run drains the queue: publish, then publish again if a write latched one while the
// previous push was out on the network, until nothing is pending.
func (q *leasePublishQueue) run() {
	for {
		q.mu.Lock()
		store := q.store
		q.mu.Unlock()

		if store != nil {
			ctx, cancel := context.WithTimeout(context.Background(), leasePublishTimeout)
			leasePublish(ctx, store)
			cancel()
		}

		q.mu.Lock()
		if !q.pending {
			q.inFlight = false
			close(q.idle)
			q.idle = nil
			q.mu.Unlock()
			return
		}
		q.pending = false
		q.mu.Unlock()
	}
}

// idleC returns a channel closed once every latched publish has run. An already-drained
// queue answers with a closed channel, so a caller never blocks on a queue with no work.
func (q *leasePublishQueue) idleC() <-chan struct{} {
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.idle == nil {
		drained := make(chan struct{})
		close(drained)
		return drained
	}
	return q.idle
}

// leaseTreesOverlap reports whether two lease tree glob sets overlap geometrically.
// If either set is empty, there is no spatial tree collision.
func leaseTreesOverlap(a, b []string) bool {
	if len(a) == 0 || len(b) == 0 {
		return false
	}
	return dispatchorder.TreesOverlap(a, b)
}

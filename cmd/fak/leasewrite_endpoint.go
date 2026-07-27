package main

import (
	"context"
	"time"

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
// sync` reconverges the laggards), so a publish error is logged, not returned.
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

	switch op {
	case "acquire":
		rec, v, err := store.AcquireFenced(ctx, leaseref.Record{
			ID:          req.ID,
			TreeGlobs:   req.TreeGlobs,
			Holder:      req.Holder,
			TTLSeconds:  req.TTLSeconds,
			Description: req.Description,
		}, now)
		if err != nil {
			return gateway.LeaseWriteResult{}, err
		}
		if v.OK {
			// Plane-0 convergence: publish the accepted lease to origin so offline nodes
			// see it too. Best-effort — a push failure never un-accepts the arbitrated
			// verdict (the local CAS already committed and is authoritative here).
			publishAcceptedLease(ctx, store)
		}
		return leaseVerdictToResult(op, req, rec, v), nil

	case "renew":
		rec, v, err := store.Renew(ctx, req.ID, req.Holder, req.TTLSeconds, now)
		if err != nil {
			return gateway.LeaseWriteResult{}, err
		}
		if v.OK {
			publishAcceptedLease(ctx, store)
		}
		return leaseVerdictToResult(op, req, rec, v), nil

	case "release":
		v, err := store.ReleaseFenced(ctx, req.ID, req.Holder, req.Generation, now)
		if err != nil {
			return gateway.LeaseWriteResult{}, err
		}
		if v.OK {
			// A release is a delete; publish so the deletion converges to origin too.
			publishAcceptedLease(ctx, store)
		}
		return leaseVerdictToResult(op, req, leaseref.Record{}, v), nil
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
// git fetch. Best-effort: a push failure is swallowed here — the arbitrated verdict stands
// on the coordinator's authoritative local store, and `fak leaseref sync` reconverges the
// laggards on the next tick.
func publishAcceptedLease(ctx context.Context, store *leaseref.Store) {
	_, _ = store.Sync(ctx, "origin", true, false)
}

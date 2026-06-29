package leaseref

// reap.go is the CLEANUP side of the bounded-lease / bounded-session contract: a crashed
// holder's lapsed lease — and a crashed node's lapsed session descriptor — is REAPABLE,
// and left unreaped it accretes as a DEAD REF in the refs/fak/locks/* namespace that
// every List/Live scan, and every cross-machine arbiter that folds LiveLeases, must read
// past. Live and LiveSessions already PARTITION the namespace into live-vs-expired and
// hand back the expired ids; these two helpers package the obvious next step (delete each
// expired id) that the `fak leaseref reap` CLI used to open-code for leases — and add the
// SESSION-side reaper that previously did not exist at all, so a crashed node's descriptor
// no longer lingers under refs/fak/locks/session-* indefinitely.
//
// Reaping is an ordinary converging ref delete (update-ref -d), so it is BEST-EFFORT and
// IDEMPOTENT: two peers racing the same reap each just delete an already-absent ref, which
// Release / RemoveSession treat as the already-reaped success state. A per-id delete
// failure is collected (errors.Join) and never aborts the sweep, so one unreapable ref
// does not strand the rest. The namespace split is preserved by construction — Reap reads
// Live (lock leases only; session refs are filtered out) and ReapSessions reads
// LiveSessions (session refs only) — so neither reaper ever cross-deletes the other kind.

import (
	"context"
	"errors"
	"fmt"
	"time"
)

// Reap deletes every LOCK-LEASE record expired at time now and returns the ids actually
// reaped. It reads Live (the live-vs-expired partition over refs/fak/locks/<id>, session
// refs excluded) then Releases each expired id. A read failure from Live is returned with
// no ids reaped; a per-id Release failure is joined into err and never aborts the sweep.
// Safe to call concurrently with a peer's reap: each delete is idempotent.
func (s *Store) Reap(ctx context.Context, now time.Time) (reaped []string, err error) {
	_, expired, lerr := s.Live(ctx, now)
	if lerr != nil {
		return nil, lerr
	}
	var errs []error
	for _, id := range expired {
		if rerr := s.Release(ctx, id); rerr != nil {
			errs = append(errs, fmt.Errorf("reap %s: %w", id, rerr))
			continue
		}
		reaped = append(reaped, id)
	}
	return reaped, errors.Join(errs...)
}

// ReapSessions deletes every SESSION descriptor expired at time now and returns the ids
// reaped — the symmetric session-side cleanup LiveSessions anticipated ("a caller can
// remove them, each via RemoveSession") but no caller provided. It reads LiveSessions (the
// live-vs-expired partition over refs/fak/locks/session-<id>, lock leases excluded) then
// RemoveSessions each expired id, joining per-id failures into err without aborting the
// sweep. Idempotent and converging, exactly like Reap.
func (s *Store) ReapSessions(ctx context.Context, now time.Time) (reaped []string, err error) {
	_, expired, lerr := s.LiveSessions(ctx, now)
	if lerr != nil {
		return nil, lerr
	}
	var errs []error
	for _, id := range expired {
		if rerr := s.RemoveSession(ctx, id); rerr != nil {
			errs = append(errs, fmt.Errorf("reap session %s: %w", id, rerr))
			continue
		}
		reaped = append(reaped, id)
	}
	return reaped, errors.Join(errs...)
}

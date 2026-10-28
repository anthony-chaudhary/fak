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
// Lock-lease reaping is guarded by the immutable OID whose expiry was checked:
// a concurrently renewed lease survives both batch deletion and its CAS fallback.
// Session reaping retains its existing best-effort, idempotent delete behavior.
// The namespace split is preserved: Live scans lock leases, LiveSessions sessions.

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"strings"
	"time"
)

// Reap deletes expired LOCK-LEASE records using the immutable blob OIDs whose
// expiry was checked. The normal path uses one ref listing, one batch blob read,
// and one guarded batch delete, regardless of the expired backlog size.
func (s *Store) Reap(ctx context.Context, now time.Time) (reaped []string, err error) {
	var observed []observedRecord
	var ok bool
	if s.runStdin != nil {
		observed, ok = s.batchReadObservedRecords(ctx)
	}
	if !ok {
		// Preserve the existing missing/corrupt-entry skip behavior. Pin each
		// fallback read to its resolved OID, never to the mutable ref name.
		observed, err = listRefs(ctx, s, isLeaseRef, func(ctx context.Context, ref string) (observedRecord, error) {
			oid, exists, rerr := s.currentOID(ctx, ref)
			if rerr != nil {
				return observedRecord{}, rerr
			}
			if !exists {
				return observedRecord{}, fmt.Errorf("lease %s vanished", ref)
			}
			rec, rerr := s.readRef(ctx, oid)
			return observedRecord{ref: ref, oid: oid, record: rec}, rerr
		})
		if err != nil {
			return nil, err
		}
	}
	sort.Slice(observed, func(i, j int) bool { return observed[i].ref < observed[j].ref })
	var ids, refs, oids []string
	var errs []error
	var batch strings.Builder
	for _, row := range observed {
		if !row.record.Expired(now) {
			continue
		}
		// The source ref, not the payload's possibly inconsistent ID, owns
		// this observation. Keep it paired with the exact batch header OID.
		ids = append(ids, leaseIDFromRef(row.ref))
		refs, oids = append(refs, row.ref), append(oids, row.oid)
		fmt.Fprintf(&batch, "delete %s %s\n", row.ref, row.oid)
	}
	if len(ids) == 0 {
		return nil, errors.Join(errs...)
	}
	if s.runStdin != nil {
		_, code, berr := s.runStdin(ctx, s.dir, batch.String(), "update-ref", "--stdin")
		if berr == nil && code == 0 {
			return ids, errors.Join(errs...)
		}
	}
	// Preserve the same expected OIDs after any batch failure. Never degrade
	// to Release or an unguarded delete, even when a peer has renewed a lease.
	for i, id := range ids {
		deleted, derr := s.casDelete(ctx, refs[i], oids[i])
		if derr != nil {
			errs = append(errs, fmt.Errorf("reap %s: %w", id, derr))
		} else if !deleted {
			errs = append(errs, fmt.Errorf("reap %s: expected-OID delete refused; re-read before retry", id))
		} else {
			reaped = append(reaped, id)
		}
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
	return s.reapRefs(ctx, expired, "reap session",
		func(id string) string { return refPrefix + sessionPrefix + id }, s.RemoveSession)
}

// reapRefs deletes every expired id/key, preferring ONE batched `git update-ref --stdin`
// transaction (the delete-side mirror of the read-side `--batch` fix on #4987 / root
// cause #2 on #4990): at ~8k refs that is a SINGLE process spawn, not ~8k serial
// `git update-ref -d` spawns — the per-ref cost that made a full drain unbounded in
// practice on this host. refOf maps each id/key to its full ref for the batch payload;
// del is the per-id idempotent deleter (Release / RemoveSession / a keyed deleteRef) the
// batch degrades to. It falls back to the per-ref loop whenever no stdin runner is wired
// (the injected-Runner test path) OR the batched transaction returns non-zero (lock
// contention, a git that could not run) — the fallback treats an already-absent ref as
// success, so the reap stays idempotent and best-effort exactly as before, just unbatched.
// Because a no-<oldvalue> `delete` is itself idempotent (a ref a peer already reaped is a
// no-op that never aborts the transaction — verified against real git), the batch and the
// fallback converge on the same post-state.
func (s *Store) reapRefs(ctx context.Context, expired []string, label string, refOf func(string) string, del func(context.Context, string) error) (reaped []string, err error) {
	if len(expired) == 0 {
		return nil, nil
	}
	if s.runStdin != nil {
		refs := make([]string, 0, len(expired))
		for _, id := range expired {
			refs = append(refs, refOf(id))
		}
		if code, berr := s.batchDeleteRefs(ctx, refs); berr == nil && code == 0 {
			return append([]string(nil), expired...), nil
		}
		// A non-executable git, a lock-contended transaction (exit != 0), or any other
		// batch failure degrades to the proven per-ref idempotent loop below.
	}
	return reapEach(ctx, expired, label, del)
}

// batchDeleteRefs deletes every ref in ONE `git update-ref --stdin` transaction. The
// payload is one `delete <ref>` line per ref with NO <oldvalue>, so a ref a peer already
// reaped is an idempotent no-op that does not abort the transaction (real-git behavior
// this package relies on). Every ref is a validated refs/fak/locks/* segment — validID /
// validSessionID / IntentKey all forbid whitespace and ref-special bytes — so the
// space-delimited line format is unambiguous and needs no -z quoting. Returns git's exit
// code (0 = the whole batch was deleted in one process); a non-zero code routes the caller
// to the per-ref fallback.
func (s *Store) batchDeleteRefs(ctx context.Context, refs []string) (int, error) {
	var b strings.Builder
	for _, ref := range refs {
		b.WriteString("delete ")
		b.WriteString(ref)
		b.WriteByte('\n')
	}
	_, code, err := s.runStdin(ctx, s.dir, b.String(), "update-ref", "--stdin")
	if err != nil {
		return -1, fmt.Errorf("leaseref: git not executable: %w", err)
	}
	return code, nil
}

// reapEach deletes each expired id via del (Release / RemoveSession / a keyed deleteRef),
// collecting a per-id failure into the joined error without aborting the sweep — the
// per-ref FALLBACK reapRefs uses when the batched delete is unavailable or fails. label
// prefixes the wrapped error ("reap" / "reap session" / "reap intent") so a caller can
// tell which namespace a failed id came from.
func reapEach(ctx context.Context, expired []string, label string, del func(context.Context, string) error) (reaped []string, err error) {
	var errs []error
	for _, id := range expired {
		if rerr := del(ctx, id); rerr != nil {
			errs = append(errs, fmt.Errorf("%s %s: %w", label, id, rerr))
			continue
		}
		reaped = append(reaped, id)
	}
	return reaped, errors.Join(errs...)
}

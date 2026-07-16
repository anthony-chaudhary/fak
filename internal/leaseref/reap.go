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
	"strings"
	"time"
)

// Reap deletes every LOCK-LEASE record expired at time now and returns the ids actually
// reaped. It reads Live (the live-vs-expired partition over refs/fak/locks/<id>, session
// refs excluded) then deletes each expired id. A read failure from Live is returned with
// no ids reaped; a per-id delete failure is joined into err and never aborts the sweep.
// Safe to call concurrently with a peer's reap: each delete is idempotent.
func (s *Store) Reap(ctx context.Context, now time.Time) (reaped []string, err error) {
	_, expired, lerr := s.Live(ctx, now)
	if lerr != nil {
		return nil, lerr
	}
	return s.reapRefs(ctx, expired, "reap", func(id string) string { return refPrefix + id }, s.Release)
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

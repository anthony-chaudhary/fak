package leaseref

// release.go is the RELEASE twin of the fenced acquire (fence.go) — the named
// follow-on of docs/region-admission.md: "when the work is done" an operator or
// loop should not have to wait out the TTL (or leak an exclusive-lane lease that
// stalls the whole fleet) when it can hand the region back explicitly.
//
// THE HAZARD IT AVOIDS (releasing a lease that is no longer yours). A blind
// `update-ref -d` by id would let a paused-then-resumed holder — or a typo'd
// operator — delete a lease a NEWER holder now legitimately owns, silently
// un-fencing the peer's region. So ReleaseFenced applies the same discipline as
// Renew: it re-reads the live lease, admits the delete only when the caller's
// holder (and, when presented, generation) still matches, and performs the delete
// under an `update-ref -d <ref> <old>` OLD-VALUE compare-and-swap so a same-host
// racer that advanced the ref wins and the stale delete loses (LEASE_CONTENDED).
//
// Two deliberate asymmetries against the write side:
//   - An ABSENT lease is an OK release, not NO_LEASE: the desired post-state (no
//     lease) already holds, the same idempotence Store.Release promises.
//   - An EXPIRED lease is releasable by ANYONE: deleting a lapsed record is
//     single-id reap semantics (reap.go does exactly this in bulk), not a write
//     under the lease, so the holder check applies only while the lease is LIVE.
//
// DENY-AS-VALUE, as everywhere in this package: every policy outcome is a
// FenceVerdict; the returned error is reserved for infrastructure failure.

import (
	"context"
	"fmt"
	"time"
)

// ReleaseFenced deletes the lease at refs/fak/locks/<id> iff the caller may: the
// ref is absent (idempotent OK), the record is expired (single-id reap), or the
// caller IS the live holder — presenting a non-zero generation additionally
// requires it to match the live lease's. A live lease held by a different (or
// anonymous) holder refuses STALE_LEASE; a ref that advanced between the read and
// the CAS delete refuses LEASE_CONTENDED (re-read and retry).
func (s *Store) ReleaseFenced(ctx context.Context, id, holder string, generation int64, now time.Time) (FenceVerdict, error) {
	if !validID(id) {
		return FenceVerdict{}, fmt.Errorf("leaseref: invalid lease id %q", id)
	}
	ref := refPrefix + id
	oldOID, hasRef, err := s.currentOID(ctx, ref)
	if err != nil {
		return FenceVerdict{}, err
	}
	v := FenceVerdict{Presented: generation}
	if !hasRef {
		// Idempotent: the desired post-state (no lease) already holds.
		v.OK = true
		v.Detail = "lease " + id + " already absent (released or reaped)"
		return v, nil
	}
	cur, err := s.readRef(ctx, ref)
	if err != nil {
		return FenceVerdict{}, err
	}
	v.Current = cur.Generation
	v.Holder = cur.Holder

	if !cur.Expired(now) {
		// A LIVE lease may be released only by its proven holder. An anonymous live
		// lease (empty cur.Holder) cannot be proven to be the caller's — the same
		// conservative posture AcquireFenced takes — so it must expire or be reaped.
		if holder == "" || cur.Holder != holder {
			v.Reason = ReasonStaleLease
			v.Detail = fmt.Sprintf("lease %s is held live by %q, not %q — not released; let it expire or have its holder release it", id, cur.Holder, holder)
			return v, nil
		}
		if generation != 0 && cur.Generation != 0 && generation != cur.Generation {
			v.Reason = ReasonStaleLease
			v.Detail = fmt.Sprintf("lease %s is live at generation %d; presented generation %d is stale — halt and reacquire before releasing", id, cur.Generation, generation)
			return v, nil
		}
	}

	deleted, err := s.casDelete(ctx, ref, oldOID)
	if err != nil {
		return FenceVerdict{}, err
	}
	if !deleted {
		v.Reason = ReasonLeaseContended
		v.Detail = fmt.Sprintf("lease %s changed under the release (CAS lost); re-read and retry", id)
		return v, nil
	}
	v.OK = true
	v.Detail = "lease " + id + " released"
	return v, nil
}

// casDelete removes ref under an `update-ref -d <ref> <old>` OLD-VALUE
// compare-and-swap: a ref that advanced (or vanished) since oldOID was read makes
// git exit non-zero and the delete is reported lost (a value, not an error) — the
// delete-side mirror of casWrite. A retrying caller re-reads; a vanished ref then
// resolves to the idempotent absent-OK.
func (s *Store) casDelete(ctx context.Context, ref, oldOID string) (bool, error) {
	_, code, err := s.run(ctx, s.dir, "update-ref", "-d", ref, oldOID)
	if err != nil {
		return false, fmt.Errorf("leaseref: git not executable: %w", err)
	}
	return code == 0, nil
}

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
	"errors"
	"fmt"
	"strings"
	"time"
)

// ReleaseFenced deletes the lease at refs/fak/locks/<id> iff the caller may: the
// ref is absent (idempotent OK), the record is expired (single-id reap), or the
// caller IS the live holder — presenting a non-zero generation additionally
// requires it to match the live lease's. A live lease held by a different (or
// anonymous) holder refuses STALE_LEASE; a ref that advanced between the read and
// the CAS delete refuses LEASE_CONTENDED (re-read and retry).
func (s *Store) ReleaseFenced(ctx context.Context, id, holder string, generation int64, now time.Time) (FenceVerdict, error) {
	return s.releaseFenced(ctx, "", id, holder, generation, now)
}

func (s *Store) releaseFenced(ctx context.Context, remote, id, holder string, generation int64, now time.Time) (FenceVerdict, error) {
	ref, oldOID, hasRef, err := s.resolveLeaseRef(ctx, id)
	if err != nil {
		return FenceVerdict{}, err
	}
	v := FenceVerdict{Presented: generation}
	if !hasRef {
		if remote != "" {
			if err := s.releaseRemoteRef(ctx, remote, ref, ""); err != nil {
				if errors.Is(err, errRemoteReleaseContended) {
					v.Reason = ReasonLeaseContended
					v.Detail = "remote lease changed; release refused and local lease retained"
					return v, nil
				}
				return v, err
			}
		}
		// Idempotent: the desired post-state (no lease) already holds.
		v.OK = true
		v.Detail = "lease " + id + " already absent (released or reaped)"
		return v, nil
	}
	readTarget := ref
	if remote != "" {
		readTarget = oldOID
	}
	cur, err := s.readRef(ctx, readTarget)
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

	if remote != "" {
		if err := s.releaseRemoteRef(ctx, remote, ref, oldOID); err != nil {
			if errors.Is(err, errRemoteReleaseContended) {
				v.Reason = ReasonLeaseContended
				v.Detail = "remote lease changed; release refused and local lease retained"
				return v, nil
			}
			return v, err
		}
	}

	genFloor := cur.Generation
	if hist, ok, err := s.resolveHistoryRef(ctx, id); err == nil && ok && hist.Generation > genFloor {
		genFloor = hist.Generation
	}
	if err := s.writeHistoryRecord(ctx, id, genFloor, now); err != nil {
		return FenceVerdict{}, err
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

// writeHistoryRecord writes a historical generation floor record under refs/fak/history/<id>.
func (s *Store) writeHistoryRecord(ctx context.Context, id string, generation int64, now time.Time) error {
	ref := historyRefPrefix + id
	hr := HistoryRecord{
		ID:         id,
		Generation: generation,
		ReleasedAt: now.Unix(),
	}
	_, err := s.putBlobRef(ctx, ref, hr)
	return err
}

// ReleaseFencedRemote deletes only the holder-validated remote object before the
// local CAS deletion. A configured but unavailable remote fails without releasing
// locally; an unconfigured remote preserves standalone local-only behavior.
// If history persistence or local CAS fails after remote deletion, the remote
// remains released and local state is retained; retry rechecks both sides.
// This is deletion fencing, not distributed acquisition arbitration.
func (s *Store) ReleaseFencedRemote(ctx context.Context, remote, id, holder string, generation int64, now time.Time) (FenceVerdict, error) {
	if !validRemote(remote) {
		return FenceVerdict{}, fmt.Errorf("leaseref: invalid remote %q", remote)
	}
	_, code, err := s.run(ctx, s.dir, "remote", "get-url", remote)
	if err != nil {
		return FenceVerdict{}, err
	}
	if code == 2 {
		return s.ReleaseFenced(ctx, id, holder, generation, now)
	}
	if code != 0 {
		return FenceVerdict{}, fmt.Errorf("leaseref: remote lookup exited %d", code)
	}
	return s.releaseFenced(ctx, remote, id, holder, generation, now)
}

var errRemoteReleaseContended = errors.New("leaseref: remote lease changed")

func (s *Store) releaseRemoteRef(ctx context.Context, remote, ref, oldOID string) error {
	out, code, err := s.run(ctx, s.dir, "ls-remote", "--refs", remote, ref)
	if err != nil {
		return err
	}
	if code != 0 {
		return fmt.Errorf("leaseref: remote release lookup exited %d; local lease retained", code)
	}
	fields := strings.Fields(out)
	if len(fields) == 0 {
		return nil
	}
	if len(fields) != 2 || fields[0] != oldOID || fields[1] != ref {
		return errRemoteReleaseContended
	}
	// The expected object fences a replacement arriving after ls-remote. Never
	// use a wildcard, prune, or unconditional force push for a release.
	_, code, err = s.run(ctx, s.dir, "push", "--force-with-lease="+ref+":"+oldOID, remote, ":"+ref)
	if err != nil {
		return err
	}
	if code != 0 {
		// Distinguish a proven replacement from transport/server failure.
		current, probeCode, probeErr := s.run(ctx, s.dir, "ls-remote", "--refs", remote, ref)
		if probeErr == nil && probeCode == 0 {
			fields := strings.Fields(current)
			if len(fields) == 2 && fields[1] == ref && fields[0] != oldOID {
				return errRemoteReleaseContended
			}
		}
		return fmt.Errorf("leaseref: fenced remote release exited %d; local lease retained", code)
	}
	return nil
}

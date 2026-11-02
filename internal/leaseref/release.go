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
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

// ReleaseFenced deletes the lease at refs/fak/locks/<id> iff the caller may: the
// ref is absent (idempotent OK), the record is expired (single-id reap), or the
// caller IS the live holder and presents the live generation. Generation zero
// is accepted only for a legacy generation-zero record; every positive live
// generation requires the exact positive token. A live lease held by a different (or
// anonymous) holder refuses STALE_LEASE; a ref that advanced between the read and
// the CAS delete refuses LEASE_CONTENDED (re-read and retry).
func (s *Store) ReleaseFenced(ctx context.Context, id, holder string, generation int64, now time.Time) (FenceVerdict, error) {
	ref, oldOID, hasRef, err := s.resolveLeaseRef(ctx, id)
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
	var admitted bool
	v, admitted = releaseAdmission(id, holder, generation, now, cur)
	if !admitted {
		return v, nil
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

func releaseAdmission(id, holder string, generation int64, now time.Time, cur Record) (FenceVerdict, bool) {
	v := FenceVerdict{Presented: generation}
	v.Current = cur.Generation
	v.Holder = cur.Holder
	// A caller that presents a fencing token is claiming a specific lease epoch.
	// Validate that claim even when the record is expired: expiry permits an
	// uncredentialed (generation-zero) maintenance reap, but it must not turn a
	// stale positive token into authority to delete a different epoch.
	if generation > 0 && generation != cur.Generation {
		v.Reason = ReasonStaleLease
		v.Detail = fmt.Sprintf("lease %s is at generation %d; presented generation %d is stale — not released", id, cur.Generation, generation)
		return v, false
	}

	if !cur.Expired(now) {
		// A LIVE lease may be released only by its proven holder. An anonymous live
		// lease (empty cur.Holder) cannot be proven to be the caller's — the same
		// conservative posture AcquireFenced takes — so it must expire or be reaped.
		if holder == "" || cur.Holder != holder {
			v.Reason = ReasonStaleLease
			v.Detail = fmt.Sprintf("lease %s is held live by %q, not %q — not released; let it expire or have its holder release it", id, cur.Holder, holder)
			return v, false
		}
		if generation != cur.Generation {
			v.Reason = ReasonStaleLease
			if generation == 0 && cur.Generation > 0 {
				v.Detail = fmt.Sprintf("lease %s requires positive generation %d; generation 0 is legacy-only — halt and reacquire before releasing", id, cur.Generation)
			} else {
				v.Detail = fmt.Sprintf("lease %s is live at generation %d; presented generation %d is stale — halt and reacquire before releasing", id, cur.Generation, generation)
			}
			return v, false
		}
	}

	return v, true
}

// ReleaseFencedRemote releases one generic lease from remote and then locally.
// The immutable local blob supplies both the admission facts and the expected
// OID. Changed direct-blob OIDs are preserved: remote uses an explicit expected-
// OID lease, and local deletion keeps that same OID after network I/O.
// Remote safety assumes the existing native protocol: direct blob lease refs
// at one configured destination. This client cannot prove arbitrary server-side
// same-OID symbolic-alias mutation safety; --symref only rejects advertised aliases.
// This is release convergence, not cross-machine acquisition arbitration.
// A failed/uncertain transport never authorizes local deletion or a retry.
func (s *Store) ReleaseFencedRemote(ctx context.Context, id, holder string, generation int64, now time.Time, remote string) (FenceVerdict, error) {
	v := FenceVerdict{Presented: generation}
	ref := refPrefix + id
	if !syncStatusRefValid(ref) || !isLeaseRef(ref) {
		return v, fmt.Errorf("leaseref: remote release requires one generic lease id")
	}
	if !validRemote(remote) {
		return v, fmt.Errorf("leaseref: remote release requires one safe remote token")
	}
	oldOID, localPresent, err := s.releaseLocalOID(ctx, ref)
	if err != nil {
		return v, err
	}
	if !localPresent {
		endpoint, err := s.releaseDestination(ctx, remote)
		if err != nil {
			return v, err
		}
		_, remotePresent, err := s.releaseRemoteOID(ctx, endpoint, ref)
		if err != nil {
			return v, err
		}
		if remotePresent {
			v.Reason = ReasonStaleLease
			v.Detail = "remote lease exists without a matching local snapshot; not released"
			return v, nil
		}
		v.OK = true
		v.Detail = "lease " + id + " already absent locally and at remote"
		return v, nil
	}
	var cur Record
	if err := s.releaseBlob(ctx, oldOID, &cur); err != nil {
		return v, err
	}
	if (cur.ID != "" && cur.ID != id) || cur.Generation < 0 {
		return v, fmt.Errorf("leaseref: frozen lease identity does not match its ref")
	}
	var admitted bool
	v, admitted = releaseAdmission(id, holder, generation, now, cur)
	if !admitted {
		return v, nil
	}
	endpoint, err := s.releaseDestination(ctx, remote)
	if err != nil {
		return v, err
	}
	remoteOID, remotePresent, err := s.releaseRemoteOID(ctx, endpoint, ref)
	if err != nil {
		return v, err
	}
	if remotePresent && remoteOID != oldOID {
		v.Reason = ReasonLeaseContended
		v.Detail = "remote lease differs from the frozen local snapshot; neither lease released"
		return v, nil
	}
	// Retain a nondecreasing LOCAL generation floor before remote side effects.
	// Distributed generation-history publication remains the separate protocol.
	if err := s.retainRemoteReleaseFloor(ctx, id, cur.Generation, now); err != nil {
		return v, err
	}
	if remotePresent {
		_, code, err := s.run(ctx, s.dir, "push", "--porcelain",
			"--force-with-lease="+ref+":"+oldOID, endpoint, ":"+ref)
		if err != nil || code != 0 {
			return v, fmt.Errorf("leaseref: fenced remote release did not complete (exit %d); remote outcome may be uncertain, local lease retained; reconcile before retry", code)
		}
	}
	deleted, err := s.releaseDeleteNoDeref(ctx, ref, oldOID)
	if err != nil {
		return v, fmt.Errorf("leaseref: remote absence observed but local delete failed; reconcile partial release: %w", err)
	}
	if !deleted {
		v.Reason = ReasonLeaseContended
		v.Detail = "remote absence observed but local lease changed; replacement retained, reconcile partial release"
		return v, nil
	}
	// These are bounded observations, not a promise that another node can never
	// acquire later. Never use a fresh observation as authority for another delete.
	_, localPresent, err = s.releaseLocalOID(ctx, ref)
	if err != nil {
		return v, fmt.Errorf("leaseref: local delete completed but its absence is unconfirmed; reconcile partial release: %w", err)
	}
	_, remotePresent, err = s.releaseRemoteOID(ctx, endpoint, ref)
	if err != nil {
		return v, fmt.Errorf("leaseref: local delete completed but remote absence is unconfirmed; reconcile partial release: %w", err)
	}
	if localPresent || remotePresent {
		v.Reason = ReasonLeaseContended
		v.Detail = "lease appeared after the fenced release; replacement retained, reconcile partial release"
		return v, nil
	}
	v.OK = true
	v.Detail = "lease " + id + " released locally and at remote"
	return v, nil
}

// Unlike the older tolerant currentOID reader, absence here requires the
// documented quiet-missing exit and empty output, not an arbitrary Git failure.
func (s *Store) releaseLocalOID(ctx context.Context, ref string) (string, bool, error) {
	_, symCode, symErr := s.run(ctx, s.dir, "symbolic-ref", "-q", ref)
	if symErr != nil || symCode != 1 {
		return "", false, fmt.Errorf("leaseref: release requires a positively non-symbolic local ref")
	}
	out, code, err := s.run(ctx, s.dir, "rev-parse", "--verify", "--quiet", ref)
	if err != nil {
		return "", false, fmt.Errorf("leaseref: local release observation failed: %w", err)
	}
	oid := strings.TrimSpace(out)
	if code == 1 && oid == "" {
		return "", false, nil
	}
	if code != 0 || !validObjectID(oid) {
		return "", false, fmt.Errorf("leaseref: local release observation is unknown (exit %d)", code)
	}
	return oid, true, nil
}

func (s *Store) releaseRemoteOID(ctx context.Context, remote, ref string) (string, bool, error) {
	out, code, err := s.run(ctx, s.dir, "ls-remote", "--symref", "--refs", "--exit-code", remote, ref)
	if err != nil {
		return "", false, fmt.Errorf("leaseref: remote release observation failed")
	}
	if code == 2 && strings.TrimSpace(out) == "" {
		return "", false, nil
	}
	fields := strings.Fields(out)
	if code != 0 || len(fields) != 2 || fields[1] != ref || !validObjectID(fields[0]) {
		return "", false, fmt.Errorf("leaseref: remote release observation is unknown (exit %d)", code)
	}
	return fields[0], true, nil
}

// Read the admitted object, never the mutable ref or a replacement-object view.
func (s *Store) releaseBlob(ctx context.Context, oid string, dst any) error {
	out, code, err := s.run(ctx, s.dir, "--no-replace-objects", "cat-file", "blob", oid)
	if err != nil || code != 0 {
		return fmt.Errorf("leaseref: frozen release object unreadable (exit %d)", code)
	}
	if err := json.Unmarshal([]byte(out), dst); err != nil {
		return fmt.Errorf("leaseref: frozen release object is not a valid record")
	}
	return nil
}

func (s *Store) retainRemoteReleaseFloor(ctx context.Context, id string, generation int64, now time.Time) error {
	ref := historyRefPrefix + id
	oid, present, err := s.releaseLocalOID(ctx, ref)
	if err != nil {
		return err
	}
	if present {
		var history HistoryRecord
		if err := s.releaseBlob(ctx, oid, &history); err != nil {
			return err
		}
		if (history.ID != "" && history.ID != id) || history.Generation < 0 {
			return fmt.Errorf("leaseref: invalid retained release generation record")
		}
		if history.Generation >= generation {
			return nil
		}
	}
	written, err := s.releaseHistoryNoDeref(ctx, ref, HistoryRecord{
		ID: id, Generation: generation, ReleasedAt: now.Unix(),
	}, oid, present)
	if err != nil {
		return err
	}
	if !written {
		return fmt.Errorf("leaseref: generation history changed before remote release; neither lease released")
	}
	return nil
}

// releaseDestination freezes one configured direct-blob lease endpoint.
// Fetch/push URL divergence and fan-out are unsupported here, not successful
// release. Pass the same resolved token to every observation and mutation.
func (s *Store) releaseDestination(ctx context.Context, remote string) (string, error) {
	read := func(push bool) (string, error) {
		args := []string{"remote", "get-url"}
		if push {
			args = append(args, "--push")
		}
		args = append(args, "--all", remote)
		out, code, err := s.run(ctx, s.dir, args...)
		if err != nil || code != 0 {
			return "", fmt.Errorf("leaseref: release destination is unavailable")
		}
		lines := strings.Split(strings.TrimSuffix(out, "\n"), "\n")
		if len(lines) != 1 || !validRemote(lines[0]) {
			return "", fmt.Errorf("leaseref: release requires exactly one configured destination")
		}
		return lines[0], nil
	}
	fetch, err := read(false)
	if err != nil {
		return "", err
	}
	push, err := read(true)
	if err != nil {
		return "", err
	}
	if fetch != push {
		return "", fmt.Errorf("leaseref: differing fetch/push release destinations are unsupported")
	}
	return push, nil
}

// These mutations never dereference a concurrently installed local symbolic ref:
// its peer target is protected. Git's OID CAS can still replace the addressed
// alias itself when its referent has the expected OID; this is not a ref-type CAS.
func (s *Store) releaseDeleteNoDeref(ctx context.Context, ref, oldOID string) (bool, error) {
	_, code, err := s.run(ctx, s.dir, "update-ref", "--no-deref", "-d", ref, oldOID)
	if err != nil {
		return false, fmt.Errorf("leaseref: local release mutation could not execute")
	}
	return code == 0, nil
}

func (s *Store) releaseHistoryNoDeref(ctx context.Context, ref string, history HistoryRecord, oldOID string, present bool) (bool, error) {
	b, err := json.Marshal(history)
	if err != nil {
		return false, err
	}
	oid, err := s.writeBlob(ctx, b)
	if err != nil {
		return false, err
	}
	if !present {
		oldOID, err = s.zeroOID(ctx)
		if err != nil {
			return false, err
		}
	}
	_, code, err := s.run(ctx, s.dir, "update-ref", "--no-deref", ref, oid, oldOID)
	if err != nil {
		return false, fmt.Errorf("leaseref: history mutation could not execute")
	}
	return code == 0, nil
}

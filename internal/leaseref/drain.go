package leaseref

// drain.go is the CONVERGENCE half of the session-descriptor reap. ReapSessions deletes
// expired refs/fak/locks/session-* refs LOCALLY (git update-ref -d), but the shared sync
// refspec is deliberately NO-PRUNE — sync.go states it outright: "DELETIONS DO NOT RIDE A
// REFSPEC", because a blanket push --prune would race-delete a peer's not-yet-fetched LIVE
// lock lease. So a local reap's deletes never reach origin, and the next force-fetch
// re-creates every locally-reaped descriptor from origin's still-present copy: a treadmill
// that leaves origin's expired-descriptor backlog structurally undrainable. The #5344
// detector half (audit trips verdict ACTION on expired descriptors) already landed; this is
// the drainer half (#5358) that the detector's ~0 target depends on.
//
// The SAFE convergence is NOT a blanket prune (which sync.go rightly forbids) but a TARGETED
// per-id delete-push scoped to the session- sub-namespace: for each id the reaper proved
// expired, push the one-sided delete refspec :refs/fak/locks/session-<id> and nothing else,
// so only proven-expired descriptors are removed on origin and a live lock lease is never
// touched.
//
// THIS FILE IS THE REPORT-FIRST SLICE, mirroring the census-first discipline of the other
// retention rows: ReportDescriptorDrain computes, READ-ONLY, exactly which expired ids WOULD be
// delete-pushed and the precise delete refspecs a drain would use — deleting and pushing
// NOTHING. The target set is proven and testable before the origin-mutation seam is wired.

import (
	"context"
	"fmt"
	"time"
)

// deleteRefspec is the one-sided push refspec that DELETES a ref on the remote: a leading
// colon with an empty local (source) side (":<dst>") is git push's delete form. The
// destination is built through SessionDescriptor.Ref() — the SAME canonical
// refs/fak/locks/session-<id> builder PublishSession and ReapSessions use — so the drain
// targets exactly the descriptor ref and never a lock lease, a branch, or any other ref:
// the targeted alternative to the forbidden blanket prune, with one source of truth for the
// ref path.
func deleteRefspec(id string) string {
	return ":" + SessionDescriptor{ID: id}.Ref()
}

// DescriptorDrainReport is the READ-ONLY projection of what a session-descriptor drain WOULD
// converge to a remote: the expired ids the reaper would delete locally, the exact per-id
// delete refspecs (:refs/fak/locks/session-<id>) a targeted delete-push would use, and how
// many LIVE descriptors were excluded from the target set. Producing the plan mutates
// nothing — no ref is deleted and no push runs — so it is the witness that proves the drain
// target before the push seam exists.
type DescriptorDrainReport struct {
	Remote         string   `json:"remote"`
	ExpiredIDs     []string `json:"expired_ids"`
	DeleteRefspecs []string `json:"delete_refspecs"`
	LiveExcluded   int      `json:"live_excluded"`
}

// ReportDescriptorDrain reports, WITHOUT mutating anything, which expired session descriptors a
// drain would delete-push to remote. It reads LiveSessions (the live-vs-expired partition
// over refs/fak/locks/session-*, lock leases excluded by the namespace split), takes the
// expired ids as the drain target, and builds one delete refspec per id. A LIVE descriptor is
// never in the target set — LiveSessions excludes it — so its heartbeat-fresh ref is
// preserved and the count of live descriptors is reported separately as the excluded
// population. The remote is validated for argv hygiene up front (the same rule Sync applies)
// so the reported plan is one a real push could execute verbatim. Strictly read-only: no ref
// is deleted, no push runs.
func (s *Store) ReportDescriptorDrain(ctx context.Context, remote string, now time.Time) (DescriptorDrainReport, error) {
	if !validRemote(remote) {
		return DescriptorDrainReport{}, fmt.Errorf("leaseref: invalid remote %q (must be one safe git argv token)", remote)
	}
	live, expired, err := s.LiveSessions(ctx, now)
	if err != nil {
		return DescriptorDrainReport{}, err
	}
	refspecs := make([]string, 0, len(expired))
	for _, id := range expired {
		refspecs = append(refspecs, deleteRefspec(id))
	}
	return DescriptorDrainReport{
		Remote:         remote,
		ExpiredIDs:     append([]string(nil), expired...),
		DeleteRefspecs: refspecs,
		LiveExcluded:   len(live),
	}, nil
}

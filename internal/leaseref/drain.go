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
// TWO SLICES, report-first then converge, mirroring the census-first discipline of the other
// retention rows. ReportDescriptorDrain computes, READ-ONLY, exactly which expired ids WOULD be
// delete-pushed and the precise delete refspecs a drain would use — deleting and pushing
// NOTHING — so the target set is proven and testable before any origin mutation.
// ConvergeDescriptorDrain then wires the origin-mutation seam ON TOP of that same plan: it stays
// DRY-RUN by default (report only, exactly ReportDescriptorDrain) and delete-pushes the proven
// target set only when a caller opts in with apply=true, reaping each drained id locally too so
// this clone's own later glob sync push cannot resurrect it. The live bulk drain of a real
// origin remains a deliberate operator opt-in, never an automatic sweep.

import (
	"context"
	"errors"
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

// DescriptorDrainResult is what a ConvergeDescriptorDrain call DID (in dry-run it is what a
// drain WOULD do): the READ-ONLY plan (DescriptorDrainReport) plus the convergence outcome.
// Applied is true only when the drain actually delete-pushed; Pushed counts the delete
// refspecs whose origin removal succeeded, and ReapedLocal counts the local descriptor refs
// this clone then removed so its own later `fak leaseref sync` cannot resurrect them.
type DescriptorDrainResult struct {
	DescriptorDrainReport
	Applied     bool `json:"applied"`
	Pushed      int  `json:"pushed"`
	ReapedLocal int  `json:"reaped_local"`
}

// drainDeleteChunk bounds how many session ids ride ONE push+reap chunk. Each delete refspec
// is ~55 bytes (":refs/fak/locks/session-" + id); at 256 per chunk that is ~14KB of argv,
// safely under the Windows ~32KB command-line limit while keeping the process-spawn count
// low for a multi-thousand-ref backlog (the ~5882 origin descriptors #5358 targets).
const drainDeleteChunk = 256

// ConvergeDescriptorDrain is the ORIGIN-mutating half of the session-descriptor reap that
// ReportDescriptorDrain only planned: for each proven-expired descriptor it delete-pushes the
// one-sided refspec :refs/fak/locks/session-<id> to remote (the targeted alternative to the
// forbidden blanket prune sync.go documents), removing ONLY expired descriptors and never a
// live lock lease. When apply is false it is byte-for-byte ReportDescriptorDrain wrapped in a
// result — it computes the plan and pushes NOTHING, the dry-run default every retention
// collector shares. A drained id's LOCAL ref is then reaped too, so this clone's own next
// glob `fak leaseref sync` push cannot re-materialize the descriptor on origin. Best-effort
// and converging: a chunk whose push fails leaves that chunk's local refs intact for a retry
// (origin untouched), and re-running after a clean drain is a no-op (LiveSessions no longer
// reports the reaped ids). The push rides the injectable Runner seam, so the sanctioned
// FLEET_ALLOW_REF_PRUNE opt-in for the pre-push deletion gate is the CALLER's env concern, not
// this library's — keeping the drain transport env-agnostic and testable.
func (s *Store) ConvergeDescriptorDrain(ctx context.Context, remote string, now time.Time, apply bool) (DescriptorDrainResult, error) {
	plan, err := s.ReportDescriptorDrain(ctx, remote, now)
	if err != nil {
		return DescriptorDrainResult{}, err
	}
	res := DescriptorDrainResult{DescriptorDrainReport: plan}
	if !apply || len(plan.ExpiredIDs) == 0 {
		return res, nil // dry-run, or nothing expired: the plan IS the answer, mutate nothing
	}
	res.Applied = true
	res.Pushed, res.ReapedLocal, err = s.drainExpired(ctx, remote, plan.ExpiredIDs)
	return res, err
}

// drainExpired delete-pushes the expired session ids to remote in bounded chunks and, for each
// chunk whose push SUCCEEDED, reaps the same ids locally. A chunk push that could not run or
// exited non-zero is collected into the joined error and its local refs are left untouched, so
// the failed ids stay in LiveSessions for the next run to retry rather than being stranded on
// origin. Each chunk is ONE `git push` process (not one per id) and the local reap rides the
// batched update-ref path, keeping the spawn count low across a multi-thousand-ref drain.
func (s *Store) drainExpired(ctx context.Context, remote string, ids []string) (pushed, reaped int, err error) {
	var errs []error
	for start := 0; start < len(ids); start += drainDeleteChunk {
		end := start + drainDeleteChunk
		if end > len(ids) {
			end = len(ids)
		}
		chunk := ids[start:end]
		args := make([]string, 0, len(chunk)+2)
		args = append(args, "push", remote)
		for _, id := range chunk {
			args = append(args, deleteRefspec(id))
		}
		_, code, rerr := s.run(ctx, s.dir, args...)
		if rerr != nil {
			errs = append(errs, fmt.Errorf("leaseref: drain push git not executable: %w", rerr))
			continue // leave this chunk's local refs for a retry; origin was not mutated
		}
		if code != 0 {
			errs = append(errs, fmt.Errorf("leaseref: drain push [%d:%d] exited %d", start, end, code))
			continue
		}
		pushed += len(chunk)
		// Origin's copies are gone: reap this clone's local copies of the SAME ids so a later
		// glob `fak leaseref sync` push from here can never resurrect the drained descriptors.
		// reapRefs is idempotent + best-effort, exactly like ReapSessions.
		r, kerr := s.reapRefs(ctx, chunk, "drain reap session",
			func(id string) string { return refPrefix + sessionPrefix + id }, s.RemoveSession)
		reaped += len(r)
		if kerr != nil {
			errs = append(errs, kerr)
		}
	}
	return pushed, reaped, errors.Join(errs...)
}

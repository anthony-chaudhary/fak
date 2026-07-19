// reclaim.go adds the DECISION half of stale-git-index.lock recovery on top of the
// read-only observer in status.go. commitlane still never removes a file — this file
// only OBSERVES the evidence Status already gathered and ADVISES whether a stale
// .git/index.lock is safe to reclaim. The actual os.Remove is performed by a trusted
// binary path in the CLI layer (which may import commitlane) once this decision says
// Reap; keeping the removal out of the observer preserves the package invariant while
// giving a worker a reachable, evidence-gated path off the fleet-wide wedge (#5294).
//
// WHY A DECISION, NOT A DELETE, LIVES HERE. gitgate — the natural home for a guarded
// forward path — cannot import commitlane: safecommit imports gitgate and commitlane
// imports safecommit, so commitlane→…→gitgate is a cycle. Rather than duplicate the
// lock/process detection, the reclaim REUSES this package's Report and exposes a pure
// verdict; the tiny actuator lives one layer up in cmd/fak where the os.Remove is
// admissible and commitlane is importable.
//
// THE EVIDENCE BAR (mirrors #5294's forensic test). A git .git/index.lock carries no
// holder pid (unlike fak-commit.lock), so "holder dead" is proven indirectly: the lock
// is present, its mtime is frozen past the grace window (IndexLock.StaleHint, default
// DefaultStaleIndexAge), and NO matching live index-writing git/fak process exists
// (len(LiveWriters) == 0). Critically, that no-writer proof is only trustworthy when
// the process inventory actually RAN: a failed probe leaves LiveWriters empty for the
// wrong reason, so DecideIndexLockReclaim requires ProcessProbe == "ok" and otherwise
// fails CLOSED (keep). The decision reads raw facts, not Report.Verdict, because a
// dead-pid fak-commit.lock (VerdictStale) can mask an independently reapable orphaned
// index.lock in the verdict precedence.
package commitlane

// IndexLockReclaimReason is the closed vocabulary explaining a reclaim decision, so a
// loop (and the tests) can match on the reason without parsing prose.
type IndexLockReclaimReason string

const (
	// ReclaimReapStale: index.lock is present, the process probe succeeded, no matching
	// live writer exists, and the lock is stale past the grace window — the orphaned-lock
	// signature. Safe to remove through the trusted binary path.
	ReclaimReapStale IndexLockReclaimReason = "reap_stale"
	// ReclaimKeepAbsent: no .git/index.lock — nothing to reclaim.
	ReclaimKeepAbsent IndexLockReclaimReason = "keep_absent"
	// ReclaimKeepProbeFailed: the process inventory did not run cleanly (error or
	// not_run), so "no live writer" cannot be proven. Fail CLOSED — never remove a lock
	// while a writer might be alive and simply unlisted.
	ReclaimKeepProbeFailed IndexLockReclaimReason = "keep_probe_failed"
	// ReclaimKeepLiveWriter: a matching live git/fak index writer is running, so the
	// lock is (or may be) genuinely held. Keep it and let the writer finish.
	ReclaimKeepLiveWriter IndexLockReclaimReason = "keep_live_writer"
	// ReclaimKeepFresh: the lock is present with no live writer but is younger than the
	// grace window — a writer could be mid-write between process samples. Keep until it
	// ages past the window.
	ReclaimKeepFresh IndexLockReclaimReason = "keep_fresh"
)

// IndexLockReclaimDecision is the read-only verdict on whether a stale git index.lock
// may be reclaimed, with the closed-vocabulary reason and the lock path the actuator
// would remove (empty when absent).
type IndexLockReclaimDecision struct {
	Reap   bool                   `json:"reap"`
	Reason IndexLockReclaimReason `json:"reason"`
	Path   string                 `json:"path,omitempty"`
}

// DecideIndexLockReclaim maps a lane Report to a reclaim decision WITHOUT touching any
// file. It reaps ONLY under the full evidence Status already gathered: the lock is
// present, the process probe SUCCEEDED (a failed probe fails closed to keep), no
// matching live writer exists, and the lock is stale past the grace window. Every
// other lane state keeps the lock. The checks read raw facts in priority order —
// absence, then probe trust, then a live writer, then freshness — so the reapable case
// is the sole fall-through and cannot be reached without every guard passing.
func DecideIndexLockReclaim(rep Report) IndexLockReclaimDecision {
	d := IndexLockReclaimDecision{Path: rep.IndexLock.Path}
	switch {
	case !rep.IndexLock.Present:
		d.Reason = ReclaimKeepAbsent
	case rep.ProcessProbe != "ok":
		d.Reason = ReclaimKeepProbeFailed
	case len(rep.LiveWriters) > 0:
		d.Reason = ReclaimKeepLiveWriter
	case !rep.IndexLock.StaleHint:
		d.Reason = ReclaimKeepFresh
	default:
		// Present, probe trustworthy, no live writer, stale past the grace window:
		// the orphaned-lock signature. This is the only branch that reaps.
		d.Reap = true
		d.Reason = ReclaimReapStale
	}
	return d
}

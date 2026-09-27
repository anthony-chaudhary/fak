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
// holder pid (unlike fak-commit.lock), so "holder dead" is proven by STALENESS ALONE:
// once the lock's mtime is frozen past the grace window (IndexLock.StaleHint, default
// DefaultStaleIndexAge), no live git writer holds it — a real writer touches index.lock
// far more often than the grace window. That single fact reaps, matching the age-only
// reap safecommit.recoverStaleIndexLock ALREADY performs inside every fak commit: the
// standalone verb must not be STRICTER than the commit path it assists, or it false-
// closes forever in the exact high-contention tree it exists for (#5335 item 3). A
// by-NAME LiveWriter is an unrelated concurrent git/fak process in a shared multi-peer
// tree, never proof THIS lock is held, so it does NOT veto a stale reap; it (and a
// failed process probe) only refine the keep REASON for a FRESH lock, where a writer may
// genuinely be mid-write between process samples. The per-file next-index decision still
// gates on the named OwnerAlive pid, because THAT lock's filename embeds its writer's
// pid. The decision reads raw facts, not Report.Verdict, because a dead-pid
// fak-commit.lock (VerdictStale) can mask an independently reapable orphaned index.lock
// in the verdict precedence.
//
// NAMED-OWNER PROOFS. A bare index.lock names nobody, but the orphans fak manufactures
// most often DO have a named creator beside them, and a dead named creator is stronger
// evidence than fifteen frozen minutes, so either proof reaps once the lock is frozen past
// the SHORT owner-dead window (IndexLock.FrozenHint) and is not advancing:
//
//   - a sibling .git/fak-commit.lock whose stamped pid is dead (fak writes it before it
//     lets git touch the index), and
//   - a paired .git/next-index-<pid>.lock whose pid is dead (IndexLock.OwnerPID /
//     OwnerDead): `git commit -- <paths>` writes index.lock, then creates its temporary
//     index, and holds both through its hooks, so a next-index file not older than the lock
//     names the lock's creator (safecommit.PartialCommitOwner). A commit killed mid-hook —
//     the 2026-09-26 incident — leaves exactly this pair.
//
// Neither proof is vetoed by an unrelated by-name live writer or a failed process probe:
// the owner is named and proven dead by pid, not inferred from an inventory. When the
// pairing proof holds, the dead writer's next-index partner is reaped with its lock, since
// it is the same writer's rollback residue.
package commitlane

// IndexLockReclaimReason is the closed vocabulary explaining a reclaim decision, so a
// loop (and the tests) can match on the reason without parsing prose.
type IndexLockReclaimReason string

const (
	// ReclaimReapStale: index.lock is present and stale past the grace window — the
	// orphaned-lock signature. index.lock carries no owner pid, so staleness alone refutes
	// an active holder (a live writer touches it far more often than the grace window); an
	// unrelated by-name writer does NOT veto this, matching safecommit's age-only reap
	// inside every fak commit. Safe to remove through the trusted binary path. (For a
	// next-index temp file, whose filename names its writer, a live OwnerAlive pid still
	// vetoes — see DecideNextIndexReclaim.)
	ReclaimReapStale IndexLockReclaimReason = "reap_stale"
	// ReclaimKeepAbsent: no .git/index.lock — nothing to reclaim.
	ReclaimKeepAbsent IndexLockReclaimReason = "keep_absent"
	// ReclaimKeepProbeFailed: a FRESH lock whose process inventory did not run cleanly
	// (error or not_run), so "no live writer" cannot be proven. Fail CLOSED — never remove
	// a young lock while a writer might be alive and simply unlisted. (A STALE index.lock
	// reaps regardless: staleness alone refutes a holder, so the probe is not consulted.)
	ReclaimKeepProbeFailed IndexLockReclaimReason = "keep_probe_failed"
	// ReclaimKeepLiveWriter: a FRESH lock with a matching live git/fak index writer, which
	// may genuinely be mid-write. Keep it and let the writer finish. (Only consulted for a
	// fresh lock — a stale index.lock reaps regardless, since a by-name writer in a shared
	// tree is not proof THIS lock is held.)
	ReclaimKeepLiveWriter IndexLockReclaimReason = "keep_live_writer"
	// ReclaimKeepFresh: the lock is present with no live writer but is younger than the
	// grace window — a writer could be mid-write between process samples. Keep until it
	// ages past the window.
	ReclaimKeepFresh IndexLockReclaimReason = "keep_fresh"
	// ReclaimReapOwnerDead: the lock is frozen past the SHORT owner-dead window, is not
	// advancing, AND its creator is NAMED and provably dead by pid. Two named-owner proofs
	// qualify:
	//   - a sibling .git/fak-commit.lock holding a pid that is no longer running (fak writes
	//     it before it touches the index) — #5335's dominant case, a `fak commit` killed at
	//     its tool timeout orphaning its OWN index.lock;
	//   - a paired .git/next-index-<pid>.lock (not older than the lock) whose pid is no longer
	//     running: a partial `git commit -- <paths>` killed mid-hook leaves its index.lock and
	//     its temporary index together, and the temp file's name is the lock's owner
	//     (IndexLock.OwnerPID/OwnerDead) — the 2026-09-26 incident.
	// Either is the same direct dead-owner refutation the next-index path already trusts, so
	// it does not need the fifteen-minute stand-in for "the holder is gone", and neither an
	// unrelated by-name live writer nor a failed process probe vetoes it. For the pairing
	// proof the dead partner next-index file is reaped with the same reason (see
	// DecideNextIndexReclaim).
	ReclaimReapOwnerDead IndexLockReclaimReason = "reap_owner_dead"
	// ReclaimKeepAdvancing: a second sample a settle window later saw the lock's mtime move
	// or its size change. Some process is writing THIS lock right now, so it is not an
	// orphan whatever its mtime reads — keep it, and let no age gate override that. This is
	// the guard that makes the age-only reaps above safe: age can misread a live-but-slow
	// writer as abandoned, an advancing mtime cannot.
	ReclaimKeepAdvancing IndexLockReclaimReason = "keep_advancing"
	// ReclaimKeepLiveOwner: a next-index-<pid>.lock whose named pid is STILL RUNNING. Keep:
	// that process may still rename the temp file over .git/index, and deleting it would
	// corrupt an in-flight index write. For index.lock it is reachable only when a paired
	// next-index file names the lock's creator (IndexLock.OwnerPID) and that pid is live —
	// a bare index.lock has no owner to name — and only for a lock no reap branch took.
	ReclaimKeepLiveOwner IndexLockReclaimReason = "keep_live_owner"
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
// file. It reaps a STALE index.lock (present + frozen past the grace window) on
// staleness ALONE — index.lock has no owner pid, so a frozen mtime already refutes an
// active holder, matching the age-only reap safecommit.recoverStaleIndexLock performs
// inside every fak commit (the verb must not be stricter than the commit path it
// assists, or it false-closes forever under contention — #5335 item 3). A failed
// process probe and an unrelated by-name live writer do NOT veto a stale reap; they only
// refine the keep REASON for a FRESH lock, where a writer could be mid-write between
// process samples. The checks read raw facts in priority order — absence, then an
// advancing lock, then staleness (reap), then a dead named owner (reap), then, for a fresh
// lock, a live named owner, probe trust and a live writer.
//
// Two witnesses refine that (#5335 item 3). An ADVANCING lock — one whose mtime moved across
// the observer's bounded settle window — is kept unconditionally, ahead of every age gate:
// age can misread a live-but-slow writer as abandoned, a moving mtime cannot, so this is
// what keeps the age-only reaps safe. And a lock frozen past the SHORT owner-dead window
// whose creator is NAMED and dead reaps without serving out the long grace window, because
// that window only ever stood in for the owner proof this case actually has. The creator is
// named either by a sibling fak-commit.lock holding a dead pid, or by a paired
// next-index-<pid>.lock whose pid is dead (a partial commit killed mid-hook; see
// IndexLock.OwnerPID). Neither named-owner branch consults the process inventory.
func DecideIndexLockReclaim(rep Report) IndexLockReclaimDecision {
	d := IndexLockReclaimDecision{Path: rep.IndexLock.Path}
	switch {
	case !rep.IndexLock.Present:
		d.Reason = ReclaimKeepAbsent
	case rep.IndexLock.Advancing:
		// Someone is writing THIS lock right now — the one fact that outranks every age
		// gate below, including a stale mtime, because it is observed on the lock itself
		// rather than inferred from the clock or from an unrelated process listing.
		d.Reason = ReclaimKeepAdvancing
	case rep.IndexLock.StaleHint:
		// Present and stale past the grace window: the orphaned-lock signature. A frozen
		// mtime refutes any active holder — no git writer holds index.lock for the grace
		// window without touching it — and index.lock carries no owner pid, so a by-name
		// LiveWriter is an unrelated concurrent process, never proof THIS lock is held.
		// Reap, matching the age-only reap safecommit.recoverStaleIndexLock already
		// performs inside every fak commit. The probe/live-writer checks below gate only
		// the FRESH branch; the next-index decision keeps its OwnerAlive veto because that
		// lock names its writer's pid.
		d.Reap = true
		d.Reason = ReclaimReapStale
	case indexLockOwnerDead(rep):
		// Frozen past the short window AND the sibling fak-commit.lock names a dead creator.
		// The creator is proven dead by pid, not inferred from the absence of a by-name
		// writer, so the long grace window buys nothing here — and waiting it out is what
		// left the lane wedged while the swarm manufactured the next orphan. Unrelated live
		// writers do not veto: they are not this lock's named owner, and the advancing check
		// above has already ruled out anyone writing this lock.
		d.Reap = true
		d.Reason = ReclaimReapOwnerDead
	case partialCommitOwnerDead(rep):
		// Frozen past the short window AND a paired next-index-<pid>.lock names this lock's
		// creator, whose pid is gone: a partial commit killed mid-hook (2026-09-26). Same
		// standing as the sibling-lock proof above — named and dead by pid — so an unrelated
		// by-name writer or a failed inventory does not veto it.
		d.Reap = true
		d.Reason = ReclaimReapOwnerDead
	case rep.IndexLock.OwnerPID > 0 && !rep.IndexLock.OwnerDead:
		// Fresh lock whose paired next-index file names a LIVE partial commit: that process
		// holds this lock (it may be sitting in a hook). Keep, and say who holds it.
		d.Reason = ReclaimKeepLiveOwner
	case rep.ProcessProbe != "ok":
		// Fresh lock + untrusted inventory: cannot rule out a live mid-write holder. Keep.
		d.Reason = ReclaimKeepProbeFailed
	case len(rep.LiveWriters) > 0:
		// Fresh lock + a live writer: it may genuinely be mid-write. Keep until it ages out.
		d.Reason = ReclaimKeepLiveWriter
	default:
		d.Reason = ReclaimKeepFresh
	}
	return d
}

// indexLockOwnerDead reports whether a present index.lock has a NAMED creator that is
// provably dead: fak writes .git/fak-commit.lock, stamped with its own pid, before it lets
// git touch the index, so a fak-created index.lock always has a sibling that names its
// owner. safecommit's probe reports Stale only for a lock that exists, parsed a pid, and
// found that pid gone — a garbage or foreign lock body yields Stale=false, so an
// unattributable sibling can never manufacture this evidence. The frozen-age half is
// required alongside it so a *new* lock created by a live peer while a dead fak lock happens
// to be lying around is never mistaken for that dead fak's residue.
func indexLockOwnerDead(rep Report) bool {
	return rep.IndexLock.FrozenHint && rep.CommitLock.Stale && rep.CommitLock.HolderPID > 0
}

// partialCommitOwnerDead reports the second named-owner proof: a present, NON-advancing
// index.lock frozen past the short owner-dead window whose paired next-index-<pid>.lock
// (IndexLock.OwnerPID, attributed by safecommit.PartialCommitOwner on raw mtimes) names a
// creator that is no longer running — and no live pairing writer exists (OwnerDead). The
// freeze is required alongside the pid so a lock a live writer is still filling can never
// be mistaken for the dead writer's residue; the pairing's age ordering already refuses
// residue that predates a newer lock. It is also the gate for reaping that partner file in
// DecideNextIndexReclaim, so the two decisions cannot disagree.
func partialCommitOwnerDead(rep Report) bool {
	lock := rep.IndexLock
	return lock.Present && !lock.Advancing && lock.FrozenHint && lock.OwnerDead && lock.OwnerPID > 0
}

// NextIndexReclaim is the per-file reclaim verdict for one observed
// .git/next-index-<pid>.lock temp file, in the same closed reason vocabulary as the
// index.lock decision so a loop can match reasons without parsing prose.
type NextIndexReclaim struct {
	Path   string                 `json:"path"`
	PID    int                    `json:"pid,omitempty"`
	Reap   bool                   `json:"reap"`
	Reason IndexLockReclaimReason `json:"reason"`
}

// DecideNextIndexReclaim maps a lane Report to one verdict per observed next-index temp
// file WITHOUT touching any file. It carries the SAME evidence bar as the index.lock
// decision — the probe must have run cleanly (a failed probe fails CLOSED) and no
// matching live git/fak writer may exist — and then adds the two per-file gates:
//
//   - the named owner pid must be DEAD. This is evidence index.lock cannot offer: a
//     next-index filename embeds the pid of the process writing it, so a live owner is a
//     direct refutation of "orphaned", not an inference from a process-name match.
//   - the file must be stale past the grace window, so a writer that is merely slow
//     between process samples is never raced.
//
// Reap is again the sole fall-through for every row but one: when the index.lock decision
// holds the partial-commit named-owner proof (partialCommitOwnerDead), the row that proof
// named — PID == IndexLock.OwnerPID, owner not alive — is that dead writer's rollback
// partner and is reaped with reason ReclaimReapOwnerDead alongside its lock, without the
// probe, live-writer, or staleness gates (its owner is named and dead by pid, exactly as
// for the lock). A live named owner still vetoes it, and every other row keeps the gates.
func DecideNextIndexReclaim(rep Report) []NextIndexReclaim {
	if len(rep.NextIndexLocks) == 0 {
		return nil
	}
	pairedDead := partialCommitOwnerDead(rep)
	out := make([]NextIndexReclaim, 0, len(rep.NextIndexLocks))
	for _, lock := range rep.NextIndexLocks {
		d := NextIndexReclaim{Path: lock.Path, PID: lock.PID}
		switch {
		case pairedDead && lock.PID == rep.IndexLock.OwnerPID && !lock.OwnerAlive:
			d.Reap = true
			d.Reason = ReclaimReapOwnerDead
		case rep.ProcessProbe != "ok":
			d.Reason = ReclaimKeepProbeFailed
		case len(rep.LiveWriters) > 0:
			d.Reason = ReclaimKeepLiveWriter
		case lock.OwnerAlive:
			d.Reason = ReclaimKeepLiveOwner
		case !lock.StaleHint:
			d.Reason = ReclaimKeepFresh
		default:
			d.Reap = true
			d.Reason = ReclaimReapStale
		}
		out = append(out, d)
	}
	return out
}

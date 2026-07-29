package treedoctor

// reflocks.go — diagnosing the git lock files that block REF MAINTENANCE, and the loose-ref
// pressure that builds up behind them (#5335).
//
// The doctor probed exactly one lock, `.git/fak-commit.lock`, and matched neither
// `.git/packed-refs.lock` nor `.git/AUTO_MERGE.lock`. In the tree this was written against
// both were 0 bytes and frozen at the same instant two days earlier, so every
// `git pack-refs --all` had been failing with
//
//	fatal: Unable to create '.../packed-refs.lock': File exists.   (rc=128)
//
// for two days. Ref packing being blocked is not a cosmetic wart: loose refs had grown to
// 7,463 (6,120 under refs/fak/locks, 1,332 under refs/fak/wip) against a 2,337-byte
// packed-refs. Every ref enumeration git performs then walks thousands of files instead of
// reading one — which manufactures precisely the O(refs) pressure this ticket originally
// blamed for the hook hang. So the two facts belong in one diagnosis: the lock that blocks
// packing, and the pile that grows while it is blocked.
//
// WHY THIS IS NOT AGE-ONLY, AND MUST NOT BECOME AGE-ONLY.
// safecommit/staleindexlock.go DELIBERATELY excluded ref locks from its age-only reap:
//
//	"Only the index lock is auto-reaped. A ref lock (refs/heads/*.lock, packed-refs) can be
//	 held by a concurrent push whose window is legitimately long, and it carries no
//	 comparable age guarantee, so ref-lock contention keeps the plain ride-then-LOCK_BUSY
//	 behavior from lockcontention.go."
//
// That precedent stands. git holds index.lock for the millisecond an index write takes, so
// age alone is a sound orphan proof there; a ref lock has no such guarantee, and a
// long-running push legitimately holds one. This diagnosis therefore requires BOTH a
// conservative freeze (DefaultRefLockStaleAge, an hour — four times the index lock's
// fifteen minutes) AND the two-sample ADVANCING witness, mirroring the one
// commitlane.probeIndexLock added for #5335: sample the lock, wait a bounded settle window,
// sample again, and if the mtime moved or the size changed then some process is writing
// THIS lock right now and no age gate may fire against it. Age can misread a live-but-slow
// writer as abandoned; an advancing mtime cannot. Report-only by default; a reap happens
// only under --apply.

import (
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
)

// RefLockNames are the non-index git locks whose orphaning silently disables ref
// maintenance. Both are top-level files in the git common dir, like the locks the doctor
// already probes.
//
//   - packed-refs.lock gates `git pack-refs`, the ONLY thing that folds loose refs away.
//   - AUTO_MERGE.lock gates updates to the AUTO_MERGE ref a merge writes. In the observed
//     tree it carried the identical frozen mtime, i.e. the same crashed writer left both.
var RefLockNames = []string{"packed-refs.lock", "AUTO_MERGE.lock"}

// refLockBlocks names, per lock, the operation an orphan of it blocks — so the report says
// what is BROKEN, not merely what file exists.
var refLockBlocks = map[string]string{
	"packed-refs.lock": "git pack-refs --all (loose refs cannot be packed away)",
	"AUTO_MERGE.lock":  "git ref updates through AUTO_MERGE (merge/rebase bookkeeping)",
}

// Closed reason vocabulary for a ref-lock verdict, so a caller can match without parsing
// prose. It mirrors commitlane's reclaim reasons on purpose: the same evidence bar, named
// the same way.
const (
	RefLockKeepAbsent    = "keep_absent"    // no such lock — nothing to diagnose
	RefLockKeepAdvancing = "keep_advancing" // mtime/size moved across the settle window: a LIVE writer
	RefLockKeepFresh     = "keep_fresh"     // frozen, but younger than the conservative stale age
	RefLockReapFrozen    = "reap_frozen"    // frozen past the stale age AND not advancing — an orphan
)

// DefaultRefLockStaleAge is how long a ref lock's mtime must be frozen before it is judged
// abandoned. It is FOUR TIMES safecommit.DefaultStaleIndexLockAge on purpose: index.lock is
// held for milliseconds so fifteen minutes is already overwhelming there, whereas a ref lock
// can legitimately be held for the length of a push. An hour of a completely frozen mtime,
// combined with the advancing witness below, is the conservative bar the ref-lock exclusion
// in safecommit/staleindexlock.go demands.
const DefaultRefLockStaleAge = 60 * time.Minute

// DefaultRefLockSettleWindow bounds the pause between the two samples that witness whether
// a ref lock is ADVANCING. It is spent ONLY when the lock is present, so a healthy tree pays
// nothing.
const DefaultRefLockSettleWindow = 300 * time.Millisecond

// DefaultLooseRefPressure is the loose-ref count above which the report flags pressure. Well
// under the 7,463 observed, and well above what a healthy packed repo carries.
const DefaultLooseRefPressure = 1000

// maxLooseRefWalk bounds the refs walk so a pathological ref store cannot stall the doctor.
// Hitting the bound is itself pressure, and the report says the count was truncated.
const maxLooseRefWalk = 50000

// RefLockOptions configures the ref-lock diagnosis. Its zero value is the production
// configuration; every field exists so the settle/age evidence can be exercised in a test
// without a real repo or a wall-clock wait.
type RefLockOptions struct {
	// StaleAge is the freeze a lock must reach to be judged abandoned. Zero =>
	// DefaultRefLockStaleAge.
	StaleAge time.Duration
	// SettleWindow bounds the pause between the two advancing samples. Zero =>
	// DefaultRefLockSettleWindow; negative => skip the second sample entirely (Advancing
	// then stays false, which never widens a reap on its own — the age gate must still pass).
	SettleWindow time.Duration
	// Sleep is the seam behind that pause (nil => time.Sleep). A test injects a function
	// that mutates the fixture, which is how a LIVE writer is simulated with no goroutine
	// and no real wait.
	Sleep func(time.Duration)
	// LoosePressure is the loose-ref count above which pressure is flagged. Zero =>
	// DefaultLooseRefPressure.
	LoosePressure int
}

// RefLockState is the verdict on ONE ref lock.
type RefLockState struct {
	Name         string `json:"name"`
	Path         string `json:"path"`
	Present      bool   `json:"present"`
	SizeBytes    int64  `json:"size_bytes"`
	AgeSeconds   int64  `json:"age_seconds,omitempty"`
	SettleMillis int64  `json:"settle_millis,omitempty"`
	// Advancing is the live-writer witness: a second sample a settle window later saw the
	// mtime move or the size change. An advancing lock is being written right now, so it is
	// never an orphan however old its mtime reads — this outranks every age gate.
	Advancing bool `json:"advancing,omitempty"`
	// Stale is the reap verdict: present, frozen past StaleAge, and NOT advancing.
	Stale  bool   `json:"stale"`
	Reason string `json:"reason,omitempty"` // RefLock* token
	Blocks string `json:"blocks,omitempty"` // what an orphan of this lock disables
}

// LooseRefState reports loose-ref pressure — the pile that grows while ref packing is
// blocked. Pure observation: nothing here ever deletes a ref.
type LooseRefState struct {
	Total int `json:"total"`
	// ByNamespace counts loose refs per namespace under refs/ (e.g. "fak/locks", "heads",
	// "remotes/origin"), so the report names WHERE the growth is rather than only how much.
	ByNamespace map[string]int `json:"by_namespace,omitempty"`
	// PackedRefsBytes is the size of .git/packed-refs, the file pack-refs would fold them
	// into. A tiny packed-refs beside thousands of loose refs is the packing-is-blocked
	// signature.
	PackedRefsBytes int64 `json:"packed_refs_bytes"`
	Truncated       bool  `json:"truncated,omitempty"` // the walk hit maxLooseRefWalk
	Pressure        bool  `json:"pressure"`
}

// RefLockReport bundles the ref-lock verdicts with the loose-ref pressure they cause.
type RefLockReport struct {
	Locks     []RefLockState `json:"locks,omitempty"`
	LooseRefs LooseRefState  `json:"loose_refs"`
}

// Stale returns the ref locks judged abandoned — the ones Sweep would reap under --apply.
func (r RefLockReport) Stale() []RefLockState {
	var out []RefLockState
	for _, l := range r.Locks {
		if l.Stale {
			out = append(out, l)
		}
	}
	return out
}

// StaleRefLocks is the Report-level accessor, matching PrunableWorktrees /
// SweepableLockResidue.
func (r Report) StaleRefLocks() []RefLockState { return r.RefLocks.Stale() }

// RefMaintenanceBlocked reports whether ref packing is currently disabled by an orphaned
// lock — the state that turns a bounded ref store into an unbounded one.
func (r Report) RefMaintenanceBlocked() bool {
	for _, l := range r.RefLocks.Locks {
		if l.Stale && l.Name == "packed-refs.lock" {
			return true
		}
	}
	return false
}

// diagnoseRefLocks probes every RefLockNames entry and counts loose-ref pressure. It is
// read-only: it never removes a file, and it spends the settle window only for locks that
// are actually present.
func diagnoseRefLocks(repoRoot string, opts RefLockOptions, now time.Time) RefLockReport {
	gitDir := filepath.Join(repoRoot, ".git")
	rep := RefLockReport{LooseRefs: countLooseRefs(gitDir, opts.LoosePressure)}
	for _, name := range RefLockNames {
		rep.Locks = append(rep.Locks, probeRefLock(filepath.Join(gitDir, name), name, opts, now))
	}
	return rep
}

// probeRefLock samples one ref lock TWICE, a bounded settle window apart, and derives the
// verdict. The second sample is the whole point: a single stat can only say how old an
// mtime looks, which conflates an orphan left by a killed writer (frozen forever) with a
// live push that is merely slow (still moving). See the file header for why a ref lock may
// never be reaped on age alone.
func probeRefLock(path, name string, opts RefLockOptions, now time.Time) RefLockState {
	out := RefLockState{Name: name, Path: path, Blocks: refLockBlocks[name], Reason: RefLockKeepAbsent}

	first, err := os.Stat(path)
	if err != nil || first.IsDir() {
		return out
	}
	out.Present = true
	out.SizeBytes = first.Size()

	settle := opts.SettleWindow
	if settle == 0 {
		settle = DefaultRefLockSettleWindow
	}
	second := first
	if settle > 0 {
		sleep := opts.Sleep
		if sleep == nil {
			sleep = time.Sleep
		}
		sleep(settle)
		out.SettleMillis = int64(settle / time.Millisecond)
		if again, aerr := os.Stat(path); aerr == nil && !again.IsDir() {
			second = again
		} else if aerr != nil && os.IsNotExist(aerr) {
			// The lock cleared itself inside the settle window. There is nothing to reclaim,
			// and reporting a vanished file as present would invite a later --apply to delete
			// whatever a live writer creates next.
			out.Present = false
			out.SizeBytes = 0
			out.Reason = RefLockKeepAbsent
			return out
		}
		out.Advancing = second.ModTime().After(first.ModTime()) || second.Size() != first.Size()
		out.SizeBytes = second.Size()
	}

	age := now.Sub(second.ModTime())
	if age < 0 {
		age = 0
	}
	out.AgeSeconds = int64(age / time.Second)

	staleAge := opts.StaleAge
	if staleAge == 0 {
		staleAge = DefaultRefLockStaleAge
	}
	switch {
	case out.Advancing:
		// Someone is writing THIS lock right now — the one fact that outranks every age gate,
		// because it is observed on the lock itself rather than inferred from the clock.
		out.Reason = RefLockKeepAdvancing
	case staleAge > 0 && age >= staleAge:
		out.Stale = true
		out.Reason = RefLockReapFrozen
	default:
		out.Reason = RefLockKeepFresh
	}
	return out
}

// countLooseRefs walks <gitDir>/refs and tallies loose refs per namespace, alongside the
// size of packed-refs. It is pure observation, bounded by maxLooseRefWalk so a pathological
// ref store cannot stall the doctor; a truncated walk is reported as such rather than
// silently under-counting.
func countLooseRefs(gitDir string, pressureAt int) LooseRefState {
	if pressureAt <= 0 {
		pressureAt = DefaultLooseRefPressure
	}
	out := LooseRefState{ByNamespace: map[string]int{}}
	if fi, err := os.Stat(filepath.Join(gitDir, "packed-refs")); err == nil {
		out.PackedRefsBytes = fi.Size()
	}

	refsDir := filepath.Join(gitDir, "refs")
	_ = filepath.WalkDir(refsDir, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() {
			return nil
		}
		if out.Total >= maxLooseRefWalk {
			out.Truncated = true
			return filepath.SkipAll
		}
		out.Total++
		out.ByNamespace[refNamespace(refsDir, path)]++
		return nil
	})
	if len(out.ByNamespace) == 0 {
		out.ByNamespace = nil
	}
	out.Pressure = out.Total >= pressureAt
	return out
}

// refNamespace derives the reporting namespace for one loose ref: the first two path
// segments below refs/ when the ref is nested that deep (so `refs/fak/locks/<id>` groups as
// "fak/locks" and `refs/remotes/origin/main` as "remotes/origin"), otherwise the first.
// Grouping at that depth is what turns "7,463 loose refs" into the actionable form —
// 6,120 of them are lock records, not branches.
func refNamespace(refsDir, path string) string {
	rel, err := filepath.Rel(refsDir, path)
	if err != nil {
		return "?"
	}
	parts := strings.Split(filepath.ToSlash(rel), "/")
	if len(parts) >= 3 {
		return parts[0] + "/" + parts[1]
	}
	return parts[0]
}

// sweepRefLocks turns the ref-lock diagnosis into actions: report-only lines by default, an
// actual os.Remove under apply. Loose-ref pressure is always advisory — treedoctor never
// deletes a ref, and packing is git's job once the lock is gone.
func sweepRefLocks(rep RefLockReport, apply bool) []string {
	var actions []string
	for _, l := range rep.Locks {
		if !l.Stale {
			continue
		}
		desc := l.Name + " frozen " + compactSeconds(l.AgeSeconds) + " and not advancing — blocks " + l.Blocks
		if !apply {
			actions = append(actions, "would reap orphaned "+desc)
			continue
		}
		if err := os.Remove(l.Path); err == nil {
			actions = append(actions, "reaped orphaned "+desc)
		} else {
			// Never silent: a refused remove on a lock we judged abandoned is exactly the
			// state that leaves ref maintenance blocked with no explanation (#5335).
			actions = append(actions, "FAILED to reap orphaned "+l.Path+": "+err.Error())
		}
	}
	if rep.LooseRefs.Pressure {
		actions = append(actions, "advisory: loose-ref pressure — "+describeLoosePressure(rep.LooseRefs)+
			"; `git pack-refs --all` folds them away once no packed-refs.lock is held")
	}
	return actions
}

// describeLoosePressure renders the loose-ref tally with its top namespaces, so the operator
// sees where the growth is without opening the JSON.
func describeLoosePressure(s LooseRefState) string {
	var b strings.Builder
	b.WriteString(strconv.Itoa(s.Total))
	b.WriteString(" loose refs vs a ")
	b.WriteString(strconv.FormatInt(s.PackedRefsBytes, 10))
	b.WriteString("-byte packed-refs")
	if s.Truncated {
		b.WriteString(" (count truncated)")
	}
	names := make([]string, 0, len(s.ByNamespace))
	for ns := range s.ByNamespace {
		names = append(names, ns)
	}
	sort.Slice(names, func(i, j int) bool {
		if s.ByNamespace[names[i]] != s.ByNamespace[names[j]] {
			return s.ByNamespace[names[i]] > s.ByNamespace[names[j]]
		}
		return names[i] < names[j]
	})
	if len(names) > 3 {
		names = names[:3]
	}
	for i, ns := range names {
		if i == 0 {
			b.WriteString(" (")
		} else {
			b.WriteString(", ")
		}
		b.WriteString(ns)
		b.WriteString(" ")
		b.WriteString(strconv.Itoa(s.ByNamespace[ns]))
	}
	if len(names) > 0 {
		b.WriteString(")")
	}
	return b.String()
}

// compactSeconds renders an age as a compact h/m/s string for an action line.
func compactSeconds(sec int64) string {
	f := func(n int64) string { return strconv.FormatInt(n, 10) }
	switch {
	case sec >= 3600:
		return f(sec/3600) + "h" + f((sec%3600)/60) + "m"
	case sec >= 60:
		return f(sec/60) + "m"
	default:
		return f(sec) + "s"
	}
}

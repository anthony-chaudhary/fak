package workerworktree

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// THE COLD-WORKTREE LEAK (#5351, census #5345, retention/GC epic #3873)
// Prepare CREATES a ~450MB detached worktree per worker; Reap collects exactly ONE
// (its --worktree is required) and only the dispatch witness sweep calls it on a
// clean exit. A worker whose session died mid-land — or whose lane finished without
// the sweep firing — leaves its worktree behind forever, so the namespace grows
// unbounded (5.8GB across ~13 worktrees observed). This file adds the BULK cold
// side: enumerate every worker worktree, decide which are COLD (their lane lease is
// no longer live AND they are past an age grace floor), and hand the caller the plan
// so it can Reap the cold ones. It DELETES NOTHING itself — planning is pure.
//
// THE SAFETY INVARIANT (same as #5343 for sessions): a worktree whose worker lane
// lease is still LIVE is NEVER cold. A live worker may be mid-build; reaping its
// worktree corrupts an in-flight land. Liveness keys on the lease's own TTL/liveness
// (read by the caller), NOT on a pid — a dead pid on a lease is EXPECTED and does not
// alone prove staleness. On top of that a worktree younger than the age floor is kept
// even when its lease looks dead, to tolerate a lease-record lag during a land. Both
// gates err toward KEEPING: a false keep only wastes disk; a false reap loses work.
//
// THE THIRD GATE — UNLANDED WORK. Lease-death plus age prove nobody is WATCHING a
// worktree; they say nothing about whether it still HOLDS work. A worker that died
// after editing but before landing leaves a dead lease, ages past the floor, and reads
// as textbook cold while carrying the only copy of its diff — so the first two gates
// alone turn "a false reap loses work" into exactly that. Measured 2026-07-26: of 20
// cold-eligible worktrees, 17 carried uncommitted entries and one held 15 modified
// TRACKED files after 253h. This gate closes that hole by asking git what the worktree
// still contains, and keeps any worktree that answers with anything at all.
//
// Kept-because-dirty is reported as HeldByWork rather than folded into the generic
// keep, because the two demand different operator actions: a live or young worktree
// needs only WAITING, while one holding unlanded work needs a decision — land it or
// abandon it. The sweep cannot make that call, so it surfaces it instead of guessing.

// DefaultColdAgeFloor is the grace window a worker worktree must exceed before it is
// eligible for the cold sweep even with a dead lease. It absorbs a brief lease-record
// lag during a land (the lease can read dead for a moment while the worker finishes),
// so a freshly-prepared worktree is never swept out from under a starting worker.
const DefaultColdAgeFloor = 30 * time.Minute

// LeaseLiveFn reports whether the worker bound to wtPath still holds a LIVE lane
// lease. Injected so the enumeration is testable without real leases; the production
// wiring reads leaseref live records and matches by lane. It MUST fail toward "live"
// (return true) when liveness cannot be determined, so an unreadable lease store can
// never cause a false reap.
type LeaseLiveFn func(wtPath string) bool

// ColdWorktree is one enumerated worker worktree with its cold-reap decision. The
// caller reaps only those with Eligible == true, and only under an explicit apply
// opt-in; Reason carries the human why-eligible / why-kept for the dry-run ledger.
type ColdWorktree struct {
	Path      string `json:"path"`
	AgeSec    int64  `json:"age_sec"`
	LeaseLive bool   `json:"lease_live"`
	Eligible  bool   `json:"eligible"`
	Reason    string `json:"reason"`
	// Unlanded is the number of uncommitted `git status --porcelain` entries the
	// worktree still carries, or -1 when git could not be asked. It is 0 for a
	// worktree the earlier gates already kept, which is NOT a cleanliness claim —
	// the probe is skipped there because its answer cannot change the verdict.
	Unlanded int `json:"unlanded"`
	// HeldByWork marks a worktree that passed BOTH the lease and age gates and was
	// kept only because it still holds unlanded work. This is the set an operator
	// must triage — land it or abandon it — and the set an --even-if-unlanded
	// override promotes back to reapable.
	HeldByWork bool `json:"held_by_work,omitempty"`
}

// WorktreeAge returns how long ago the worktree directory was last modified — the age
// proxy the grace floor compares against. A stat error (or a future mtime) yields 0,
// which reads as "brand new" and is therefore KEPT, so a probe failure never over-reaps.
func WorktreeAge(wtPath string, now time.Time) time.Duration {
	fi, err := os.Stat(wtPath)
	if err != nil {
		return 0
	}
	d := now.Sub(fi.ModTime())
	if d < 0 {
		return 0
	}
	return d
}

// LaneOf recovers the lane segment from a worker worktree path — the inverse of
// DirName's <marker>-<lane>-<hashedKey> composition. Returns "" when the basename is
// not a lane-bearing worker worktree (the bare marker, or a name too short to carry a
// hashed key), so a caller that binds by lane treats it as unclassifiable and keeps it.
func LaneOf(wtPath string) string {
	name := filepath.Base(filepath.Clean(wtPath))
	rest := strings.TrimPrefix(name, WorktreeMarker+"-")
	if rest == name || rest == "" {
		return "" // not "<marker>-..." — no lane to recover
	}
	// rest is "<lane>-<hashedKey>"; the hashed key is the trailing keyHashLen chars
	// after the final '-'. Strip exactly that suffix to leave the (possibly dashed) lane.
	cut := len(rest) - keyHashLen - 1
	if cut <= 0 || rest[cut] != '-' {
		return ""
	}
	return rest[:cut]
}

// UnlandedCount reports how many uncommitted entries a worker worktree still carries,
// counting BOTH tracked modifications and untracked files: a worker that died before
// landing may hold its only new source file as untracked, so ignoring `??` would reap
// exactly the work this gate exists to protect. Git's own ignore rules already exclude
// build output, so an ignored artifact tree does not read as work.
//
// Returns -1 when git could not be asked (a missing directory, a broken worktree
// registration, git absent). That is deliberately NOT 0: the sweep treats "cannot tell"
// as "carries work" and keeps the worktree, so a probe failure never over-reaps.
func UnlandedCount(wtPath string, git GitRunner) int {
	rc, out := run(git, wtPath, []string{"status", "--porcelain"})
	if rc != 0 {
		return -1
	}
	n := 0
	for _, line := range strings.Split(out, "\n") {
		if strings.TrimSpace(line) != "" {
			n++
		}
	}
	return n
}

// coldEligible is the PURE reap decision for ONE worktree: cold iff its lane lease is
// NOT live, it is at least ageFloor old, AND it carries no unlanded work. Returns the
// verdict, whether it was kept SOLELY for unlanded work, and a human reason for the
// ledger. Every keep-branch documents WHY the worktree was spared.
//
// unlanded is the UnlandedCount reading: 0 means nothing uncommitted, a positive count
// means real content, and -1 means the probe could not answer — which keeps the
// worktree, since an unreadable tree is the case where a wrong reap is least recoverable.
func coldEligible(leaseLive bool, age, ageFloor time.Duration, unlanded int) (eligible, heldByWork bool, reason string) {
	if leaseLive {
		return false, false, "kept: worker lane lease still live"
	}
	if age < ageFloor {
		return false, false, fmt.Sprintf("kept: lease dead but age %s under grace floor %s",
			age.Round(time.Second), ageFloor.Round(time.Second))
	}
	if unlanded < 0 {
		return false, true, "kept: lease dead and past floor, but its working tree could not be read — refusing to reap what cannot be inspected"
	}
	if unlanded > 0 {
		return false, true, fmt.Sprintf("kept: lease dead and age %s past floor, but %d uncommitted entr%s remain — land or abandon before reaping",
			age.Round(time.Second), unlanded, plural(unlanded, "y", "ies"))
	}
	return true, false, fmt.Sprintf("cold: lease dead, age %s past grace floor %s, working tree clean",
		age.Round(time.Second), ageFloor.Round(time.Second))
}

// plural picks the singular or plural suffix for n, so a reason line reads as prose
// ("1 uncommitted entry" / "9 uncommitted entries") in an operator-facing ledger.
func plural(n int, one, many string) string {
	if n == 1 {
		return one
	}
	return many
}

// ColdReapList enumerates the live worker worktrees (via Count — the same read `list`
// uses) and decides which are COLD, DELETING NOTHING. A worktree is cold iff leaseLive
// reports its lane lease dead, WorktreeAge puts it past ageFloor, AND UnlandedCount
// finds nothing uncommitted still in it. The caller applies
// Reap to the Eligible entries, and only under an explicit apply opt-in — this function
// is the dry-run plan. A nil leaseLive is treated as "cannot prove liveness", so every
// worktree is kept (fail toward keeping).
func ColdReapList(root string, git GitRunner, now time.Time, ageFloor time.Duration, leaseLive LeaseLiveFn) []ColdWorktree {
	if ageFloor <= 0 {
		ageFloor = DefaultColdAgeFloor
	}
	_, paths := Count(root, git)
	out := make([]ColdWorktree, 0, len(paths))
	for _, p := range paths {
		live := true // fail toward keeping when there is no oracle
		if leaseLive != nil {
			live = leaseLive(p)
		}
		age := WorktreeAge(p, now)
		// Probe the working tree ONLY for a worktree the first two gates would
		// otherwise release. A live or young worktree is kept whatever its status
		// says, so asking git there would spend a subprocess per worktree to reach
		// the same verdict — and this sweep runs over every worktree in the namespace.
		unlanded := 0
		if !live && age >= ageFloor {
			unlanded = UnlandedCount(p, git)
		}
		eligible, heldByWork, reason := coldEligible(live, age, ageFloor, unlanded)
		out = append(out, ColdWorktree{
			Path:       p,
			AgeSec:     int64(age / time.Second),
			LeaseLive:  live,
			Eligible:   eligible,
			Reason:     reason,
			Unlanded:   unlanded,
			HeldByWork: heldByWork,
		})
	}
	return out
}

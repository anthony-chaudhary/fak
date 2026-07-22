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

// coldEligible is the PURE reap decision for ONE worktree: cold iff its lane lease is
// NOT live AND it is at least ageFloor old. Returns the verdict plus a human reason for
// the ledger. Both keep-branches document WHY the worktree was spared.
func coldEligible(leaseLive bool, age, ageFloor time.Duration) (bool, string) {
	if leaseLive {
		return false, "kept: worker lane lease still live"
	}
	if age < ageFloor {
		return false, fmt.Sprintf("kept: lease dead but age %s under grace floor %s",
			age.Round(time.Second), ageFloor.Round(time.Second))
	}
	return true, fmt.Sprintf("cold: lease dead and age %s past grace floor %s",
		age.Round(time.Second), ageFloor.Round(time.Second))
}

// ColdReapList enumerates the live worker worktrees (via Count — the same read `list`
// uses) and decides which are COLD, DELETING NOTHING. A worktree is cold iff leaseLive
// reports its lane lease dead AND WorktreeAge puts it past ageFloor. The caller applies
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
		eligible, reason := coldEligible(live, age, ageFloor)
		out = append(out, ColdWorktree{
			Path:      p,
			AgeSec:    int64(age / time.Second),
			LeaseLive: live,
			Eligible:  eligible,
			Reason:    reason,
		})
	}
	return out
}

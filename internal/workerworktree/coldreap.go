package workerworktree

import (
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"sync"
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

// DefaultColdReapConcurrency bounds filesystem and git pressure during an all-cold
// census. Callers may select a smaller ceiling through ColdReapOptions.
const DefaultColdReapConcurrency = 8

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
	// ReclaimBytes is the logical file-byte estimate for this worktree when it is
	// eligible and the complete tree was readable. ReclaimBytesKnown distinguishes
	// a measured empty tree from an estimate that could not be completed.
	ReclaimBytes      int64 `json:"reclaim_bytes"`
	ReclaimBytesKnown bool  `json:"reclaim_bytes_known"`
}

// ColdReapProgress is emitted after each worktree census completes. Byte totals
// include eligible worktrees only; EligibleBytesUnknown counts eligible trees whose
// complete logical size could not be measured, so zero is never presented as known.
type ColdReapProgress struct {
	Completed            int   `json:"completed"`
	Total                int   `json:"total"`
	Eligible             int   `json:"eligible"`
	EligibleBytes        int64 `json:"eligible_bytes"`
	EligibleBytesKnown   int   `json:"eligible_bytes_known"`
	EligibleBytesUnknown int   `json:"eligible_bytes_unknown"`
}

// ColdReapOptions controls the bounded all-cold census. Size may be supplied by
// tests or callers that have a more appropriate accounting source. Progress is
// called serially as results arrive; it must return promptly.
type ColdReapOptions struct {
	Concurrency int
	Size        func(path string) (int64, error)
	Progress    func(ColdReapProgress)
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

// worktreeLogicalBytes returns the complete logical file size beneath path. Any
// unreadable entry makes the estimate unknown; partial totals are never reported as
// complete. Symlinks are counted by their link metadata and never followed.
func worktreeLogicalBytes(path string) (int64, error) {
	var total int64
	err := filepath.WalkDir(path, func(_ string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		info, err := entry.Info()
		if err != nil {
			return err
		}
		if !info.IsDir() {
			total += info.Size()
		}
		return nil
	})
	if err != nil {
		return 0, err
	}
	return total, nil
}

// ColdReapList enumerates and plans with the bounded default census settings.
// It remains the compatibility entry point for callers that do not need progress.
func ColdReapList(root string, git GitRunner, now time.Time, ageFloor time.Duration, leaseLive LeaseLiveFn) []ColdWorktree {
	return ColdReapListWithOptions(root, git, now, ageFloor, leaseLive, ColdReapOptions{})
}

// ColdReapListWithOptions enumerates worker worktrees and decides which are cold,
// deleting nothing. Expensive per-tree status and size probes run under one explicit
// worker-pool ceiling. Results retain Count's order even though progress reflects
// actual completion order. A nil lease oracle fails toward live; status or size probe
// failures fail toward preserving work or reporting bytes unknown, respectively.
func ColdReapListWithOptions(root string, git GitRunner, now time.Time, ageFloor time.Duration, leaseLive LeaseLiveFn, opts ColdReapOptions) []ColdWorktree {
	if ageFloor <= 0 {
		ageFloor = DefaultColdAgeFloor
	}
	concurrency := opts.Concurrency
	if concurrency <= 0 {
		concurrency = DefaultColdReapConcurrency
	}
	size := opts.Size
	if size == nil {
		size = worktreeLogicalBytes
	}

	_, paths := Count(root, git)
	out := make([]ColdWorktree, len(paths))
	if len(paths) == 0 {
		return out
	}
	if concurrency > len(paths) {
		concurrency = len(paths)
	}

	type job struct {
		index int
		path  string
	}
	type result struct {
		index int
		item  ColdWorktree
	}
	jobs := make(chan job)
	results := make(chan result)
	var workers sync.WaitGroup
	workers.Add(concurrency)
	for range concurrency {
		go func() {
			defer workers.Done()
			for work := range jobs {
				p := work.path
				live := true
				if leaseLive != nil {
					live = leaseLive(p)
				}
				age := WorktreeAge(p, now)
				unlanded := 0
				if !live && age >= ageFloor {
					unlanded = UnlandedCount(p, git)
				}
				eligible, heldByWork, reason := coldEligible(live, age, ageFloor, unlanded)
				item := ColdWorktree{
					Path: p, AgeSec: int64(age / time.Second), LeaseLive: live,
					Eligible: eligible, Reason: reason, Unlanded: unlanded,
					HeldByWork: heldByWork,
				}
				if eligible {
					if bytes, err := size(p); err == nil {
						item.ReclaimBytes = bytes
						item.ReclaimBytesKnown = true
					}
				}
				results <- result{index: work.index, item: item}
			}
		}()
	}
	go func() {
		for i, p := range paths {
			jobs <- job{index: i, path: p}
		}
		close(jobs)
		workers.Wait()
		close(results)
	}()

	progress := ColdReapProgress{Total: len(paths)}
	for result := range results {
		out[result.index] = result.item
		progress.Completed++
		if result.item.Eligible {
			progress.Eligible++
			if result.item.ReclaimBytesKnown {
				progress.EligibleBytes += result.item.ReclaimBytes
				progress.EligibleBytesKnown++
			} else {
				progress.EligibleBytesUnknown++
			}
		}
		if opts.Progress != nil {
			opts.Progress(progress)
		}
	}
	return out
}

package workerworktree

import (
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// THE PER-WORKER WORKTREE TAX (#3572, child of #3165)
// Prepare materializes a FULL working tree per worker (`git worktree add --detach`)
// and Reap force-removes it. At 100x that create/destroy is paid per dispatch: the
// checkout cost N times over, plus `.git/worktrees` admin churn under contention.
// Prepare's existing Reused bit only fires when the SAME lane+key re-prepares, so a
// NEW worker always paid full price even with an identical worktree sitting idle.
//
// THE POOL. Keep at most K IDLE worktrees per lane. Prepare LEASES one — a fast
// `reset --hard <newBase>` + `clean -fd` re-points the already-materialized tree at
// the new base instead of checking the whole tree out again — and reports it with the
// existing Reused bit. Reap RETURNS the worktree (reset clean, mark idle) up to the
// cap and force-removes past it, exactly as today.
//
// IDLE STATE LIVES OUTSIDE THE MEMBERS. One marker file per idle member sits in a
// sidecar dir under the worktree ROOT, not in the worktree: leasing runs `git clean`
// inside a member, which would delete an in-tree marker, and an in-tree marker would
// also show up in the worker's own `git status`. Keeping it on disk (rather than in a
// process-local map) is what lets a `fak worktree worker reap` in one process hand a
// warm tree to a `prepare` in the next — the actual dispatch shape.
//
// FALLBACK IS ALWAYS THE OLD PATH. A miss (pool off, empty, all members claimed by a
// racing peer, or a member that will not reset clean) falls straight through to
// `worktree add` / `worktree remove --force`, so the pool can only ever remove work,
// never add a new failure mode.
//
// LEAKED MEMBERS STAY RECLAIMABLE (the ordering #3572's Notes ask for). An idle member
// is an ordinary marker-named worker worktree with a clean tree, so ColdReapList sees
// it, and once its lane lease is dead and it is past the age floor it grades cold and
// the cold sweep reclaims it. Returning a member resets it clean precisely so that
// stays true — a member left dirty would grade HeldByWork and pin disk forever.
const (
	// PoolCapEnv bounds the warm pool: the maximum number of IDLE worktrees kept per
	// LANE. Unset, unparsable, or negative means defaultPoolCap; 0 disables the pool
	// and reproduces the pre-#3572 create/reap byte for byte.
	//
	// CONFIG-SURFACE DEBT (#2863/#2862): this is behavioral configuration in the
	// environment. It is spelled as a const identifier, which is the idiom this
	// package's four existing knobs (WorktreeRootEnv, LandReadbackEnv, IsolatedLandEnv,
	// IsolatedLandRetryEnv) already use and which envconfiglint's literal-only scanner
	// cannot see — recorded here rather than left implicit. Relocates to: a `--pool`
	// flag on `fak worktree worker prepare|reap` plus a field the dispatch spawn passes
	// in, so the cap arrives as an argument instead of a process-env re-read.
	PoolCapEnv = "FLEET_WORKER_WORKTREE_POOL"
	// defaultPoolCap is 0 — OFF. gen/next: the pool changes what Reap DOES (a reaped
	// worktree stops disappearing), so it stays gated behind an explicit operator
	// opt-in until dogfood evidence promotes it. Flipping this to a small K is the
	// promotion step, not part of shipping the mechanism.
	defaultPoolCap = 0
	// poolStateDir is the sidecar dir under the worktree root holding the idle markers.
	poolStateDir = ".fak-wt-pool"
	// poolIdleExt suffixes an idle marker; the rest of the name is the member's dir name.
	poolIdleExt = ".idle"
)

// PoolCap reports how many idle worktrees the warm pool keeps per lane. Fail-open to
// the default on anything unreadable: a typo'd knob must never mean "unbounded".
func PoolCap() int {
	v := strings.TrimSpace(os.Getenv(PoolCapEnv))
	if v == "" {
		return defaultPoolCap
	}
	n, err := strconv.Atoi(v)
	if err != nil || n < 0 {
		return defaultPoolCap
	}
	return n
}

// resolveWorktreeRoot is Path's root resolution, reused so the pool's sidecar dir and
// the worktrees it indexes can never disagree about where the root is.
func resolveWorktreeRoot(wtRoot string) string {
	if wtRoot == "" {
		return DefaultRoot()
	}
	return wtRoot
}

func poolStatePath(wtRoot string) string {
	return filepath.Join(resolveWorktreeRoot(wtRoot), poolStateDir)
}

func poolMarker(wtRoot, dirName string) string {
	return filepath.Join(poolStatePath(wtRoot), dirName+poolIdleExt)
}

// poolIdleMembers lists the idle members of lane under wtRoot, sorted for a
// deterministic lease order. A marker whose worktree directory is GONE (cold-swept,
// hand-removed) is dropped and its marker cleared: a member that no longer exists must
// not hold a pool slot, or the cap silently starves after the first sweep. An empty
// lane matches every member, which is what the cap accounting for a bare-marker path
// would need; callers pass a real lane.
func poolIdleMembers(wtRoot, lane string) []string {
	stateDir := poolStatePath(wtRoot)
	entries, err := os.ReadDir(stateDir)
	if err != nil {
		return nil
	}
	root := resolveWorktreeRoot(wtRoot)
	var out []string
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), poolIdleExt) {
			continue
		}
		dirName := strings.TrimSuffix(e.Name(), poolIdleExt)
		if lane != "" && LaneOf(dirName) != lane {
			continue
		}
		wt := filepath.Join(root, dirName)
		if _, err := os.Stat(wt); err != nil {
			os.Remove(filepath.Join(stateDir, e.Name()))
			continue
		}
		out = append(out, wt)
	}
	sort.Strings(out)
	return out
}

// resetPooled re-points an already-materialized worktree at base and drops everything
// uncommitted, so the next lessee sees the same tree `worktree add --detach <base>`
// would have given it. The build caches are EXCLUDED from the clean on purpose: they
// are the whole point of pairing with #3242's warm GOCACHE — cleaning them would hand
// back a worktree that is warm on checkout and cold on build. Returns false on any git
// error, which the callers read as "this member is no longer trustworthy".
func resetPooled(wt, base string, git GitRunner) bool {
	if rc, _ := run(git, wt, []string{"reset", "--hard", base}); rc != 0 {
		return false
	}
	rc, _ := run(git, wt, []string{"clean", "-fd", "-e", ".gocache", "-e", ".gotmp"})
	return rc == 0
}

// leasePooled hands a NEW worker an idle member of its lane, re-pointed at base. The
// claim is the marker REMOVAL: exactly one of N racing Prepares can remove a given
// file, so a loser sees the error and moves to the next candidate rather than two
// workers editing one tree. Returns handled=false when nothing could be leased — the
// caller then creates a worktree exactly as before.
func leasePooled(root, lane, base, wtRoot string, git GitRunner) (Result, bool) {
	for _, wt := range poolIdleMembers(wtRoot, lane) {
		if err := os.Remove(poolMarker(wtRoot, filepath.Base(wt))); err != nil {
			continue // a peer leased it first
		}
		if resetPooled(wt, base, git) {
			return Result{OK: true, Path: wt, BaseSHA: base, Reused: true,
				Reason: "warm worktree pool hit (#3572)"}, true
		}
		// It would not come back clean (pruned admin record, broken registration, git
		// absent). It is not editing space any more: destroy it so it can never be
		// leased again, then try the next member.
		run(git, root, []string{"worktree", "remove", "--force", wt})
		run(git, root, []string{"worktree", "prune"})
	}
	return Result{}, false
}

// returnPooled parks a finished worker's worktree as an idle member instead of
// destroying it, when its lane is under the cap. Returns handled=false for overflow (or
// any hiccup), which the caller reads as "force-remove exactly as today".
//
// The member being returned is EXCLUDED from its own cap count, so re-reaping an
// already-idle worktree is idempotent rather than tipping the lane over the cap and
// destroying the very member it just parked. That matters because Reap is best-effort
// and the dispatch witness sweep can fire twice on one worktree.
//
// WHAT THIS DOES NOT CHANGE: uncommitted work in the worktree is discarded here. So did
// the force-remove it replaces — the reset is not a new way to lose a diff, and the
// cold sweep's unlanded-work gate (coldreap.go) is still the arm that protects a worker
// that died before landing.
func returnPooled(root, wtPath string, capacity int, git GitRunner) (Result, bool) {
	lane := LaneOf(wtPath)
	if lane == "" {
		return Result{}, false // bare marker / unclassifiable: not a poolable member
	}
	wtRoot := filepath.Dir(filepath.Clean(wtPath))
	idle := 0
	for _, m := range poolIdleMembers(wtRoot, lane) {
		if !samePath(m, wtPath) {
			idle++
		}
	}
	if idle >= capacity {
		return Result{}, false // overflow: reap it for real
	}
	// Reset to its OWN tip: the member keeps whatever base it was pinned at, and the
	// lease re-points it at the next worker's base anyway. All that matters here is
	// that it is parked CLEAN.
	if !resetPooled(wtPath, "HEAD", git) {
		return Result{}, false
	}
	if err := os.MkdirAll(poolStatePath(wtRoot), 0o755); err != nil {
		return Result{}, false
	}
	marker := poolMarker(wtRoot, filepath.Base(filepath.Clean(wtPath)))
	if err := os.WriteFile(marker, []byte("fak-worker-worktree-pool/1 lane="+lane+"\n"), 0o644); err != nil {
		return Result{}, false
	}
	return Result{OK: true, Path: wtPath, Removed: false,
		Reason: "returned to warm worktree pool (#3572)"}, true
}

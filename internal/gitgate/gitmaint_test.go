package gitgate

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// fakeMaint answers the git invocations RunMaint issues from a small keyed model and
// records every argv. posture maps a config key to its value (a missing key models an
// UNSET config: git exits non-zero). The loose/in-pack/pack counters model the object
// DB; a grace `loose-objects` run folds loose objects into packs, so `count-objects`
// after the grace tier reports the drop with nothing pruned. onGrace fires when a
// maintenance step runs — a test uses it to fabricate a lock mid-run (the TOCTOU case).
type fakeMaint struct {
	posture map[string]string
	loose   int
	inPack  int
	packs   int
	// unreachable models the loose objects with NO reachable referent — the ~86%
	// backlog only a prune can reclaim (#5079). A `git prune --expire=…` call drops
	// them from `loose` without adding to `in-pack` (removed, not folded).
	unreachable int
	// prunePackable models loose copies already present in a pack. Git's
	// loose-objects task prunes the previous set before packing today's loose objects;
	// the explicit following prune-packed step clears the new set in the same run.
	prunePackable int
	calls         [][]string
	onGrace       func(args []string)
	// fsmon is the text `git fsmonitor--daemon status` returns (classified by readPosture);
	// fsmonErr models git being unable to run the probe at all (→ unknown).
	fsmon    string
	fsmonErr bool
	// fsmonStartFails models the #5068 clone shape where `git fsmonitor--daemon start`
	// cannot bring a daemon up (git #75781, #26154): start exits non-zero and the
	// status probe keeps reporting not-watching.
	fsmonStartFails bool
}

func (f *fakeMaint) run(_ context.Context, dir string, args ...string) (string, int, error) {
	f.calls = append(f.calls, append([]string{dir}, args...))
	switch {
	case len(args) >= 3 && args[0] == "config" && args[1] == "--get":
		if v, ok := f.posture[args[2]]; ok {
			return v + "\n", 0, nil
		}
		return "", 1, nil // unset key → git exits non-zero
	case len(args) >= 1 && args[0] == "count-objects":
		return f.countText(), 0, nil
	case len(args) >= 2 && args[0] == "multi-pack-index" && args[1] == "write":
		return "", 0, nil
	case len(args) >= 3 && args[0] == "commit-graph" && args[1] == "write":
		return "", 0, nil
	case len(args) >= 2 && args[0] == "maintenance" && args[1] == "run":
		if f.onGrace != nil {
			f.onGrace(args)
		}
		if reachable := f.loose - f.unreachable; hasArg(args, "--task=loose-objects") && reachable > 10 {
			packed := reachable - 10
			f.inPack += packed // packed first; the loose duplicate still exists until prune-packed
			f.prunePackable += packed
		}
		return "", 0, nil
	case len(args) >= 1 && args[0] == "prune-packed":
		if f.onGrace != nil {
			f.onGrace(args)
		}
		f.loose -= f.prunePackable
		f.prunePackable = 0
		return "", 0, nil
	case len(args) >= 1 && args[0] == "prune":
		if f.onGrace != nil {
			f.onGrace(args)
		}
		drop := f.unreachable
		if drop > f.loose {
			drop = f.loose
		}
		f.loose -= drop // removed outright — in-pack does NOT grow
		f.unreachable = 0
		return "", 0, nil
	case len(args) >= 3 && args[0] == "config" && args[1] == "--unset":
		if _, ok := f.posture[args[2]]; ok {
			delete(f.posture, args[2])
			return "", 0, nil
		}
		return "", 5, nil // git exits 5 when the key to unset does not exist
	case len(args) >= 2 && args[0] == "fsmonitor--daemon" && args[1] == "start":
		if f.fsmonStartFails {
			return "error: fsmonitor--daemon failed to start", 128, nil
		}
		f.fsmon = "fsmonitor-daemon is watching 'C:/work/clone'"
		f.fsmonErr = false
		return "", 0, nil
	case len(args) >= 2 && args[0] == "fsmonitor--daemon" && args[1] == "status":
		if f.fsmonErr {
			return "", 1, fmt.Errorf("fsmonitor--daemon status: could not run")
		}
		// A not-watching status exits non-zero on git; readPosture classifies by text.
		code := 0
		if strings.Contains(strings.ToLower(f.fsmon), "not watching") {
			code = 1
		}
		return f.fsmon, code, nil
	}
	return "", 0, nil
}

func (f *fakeMaint) countText() string {
	return fmt.Sprintf("count: %d\nsize: 1.00 MiB\nin-pack: %d\npacks: %d\nprune-packable: %d\ngarbage: 0\nsize-garbage: 0 bytes\n",
		f.loose, f.inPack, f.packs, f.prunePackable)
}

func hasArg(args []string, want string) bool {
	for _, a := range args {
		if a == want {
			return true
		}
	}
	return false
}

// safePosture is the shared no-auto-gc posture the hot clone holds (incl. the
// core.untrackedCache=true cold-status speedup asserted since #5069).
func safePosture() map[string]string {
	return map[string]string{"gc.auto": "0", "maintenance.auto": "false", "core.untrackedCache": "true"}
}

// scratchGit builds a scratch common-dir with the object-DB subtree present and no
// locks. Returns the gitDir path; the caller seeds locks by writing *.lock files.
func scratchGit(t *testing.T) string {
	t.Helper()
	gitDir := filepath.Join(t.TempDir(), ".git")
	for _, sub := range []string{
		filepath.Join("objects", "info"),
		filepath.Join("objects", "pack"),
		"refs",
		"worktrees",
	} {
		if err := os.MkdirAll(filepath.Join(gitDir, sub), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	return gitDir
}

func writeLock(t *testing.T, gitDir, rel string) {
	t.Helper()
	p := filepath.Join(gitDir, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte("held\n"), 0o644); err != nil {
		t.Fatal(err)
	}
}

// ranTiers returns the set of tiers whose steps actually executed a mutating git verb,
// derived from the recorded calls (not the result), so a test proves what hit git.
func mutatingCalls(calls [][]string) []string {
	var out []string
	for _, c := range calls {
		if len(c) < 2 {
			continue
		}
		switch c[1] { // c[0] is the dir
		case "multi-pack-index", "commit-graph", "maintenance", "prune-packed", "prune":
			out = append(out, strings.Join(c[1:], " "))
		}
	}
	return out
}

// TestRunMaintCleanSafeRunsBothTiers: unlocked + safe posture ⇒ the always-safe tier
// AND the safe-with-grace tier both run, the loose backlog folds into a pack with
// nothing pruned, and no refusal is recorded.
func TestRunMaintCleanSafeRunsBothTiers(t *testing.T) {
	gitDir := scratchGit(t)
	f := &fakeMaint{posture: safePosture(), loose: 100, inPack: 500, packs: 11}
	res := RunMaint(context.Background(), f.run, MaintOptions{RepoRoot: filepath.Dir(gitDir), GitCommonDir: gitDir, Apply: true})

	if res.GraceRefused != "" {
		t.Fatalf("clean+safe should not refuse the grace tier: %q", res.GraceRefused)
	}
	if res.Incident {
		t.Fatalf("clean+safe must not be an incident")
	}
	muts := mutatingCalls(f.calls)
	// 2 always-safe + 3 grace verbs must have hit git.
	for _, want := range []string{
		"multi-pack-index write", "commit-graph write --reachable",
		"maintenance run --task=loose-objects", "prune-packed", "maintenance run --task=incremental-repack",
	} {
		if !containsStr(muts, want) {
			t.Fatalf("expected git step %q to run; got %v", want, muts)
		}
	}
	if res.LooseDelta() <= 0 {
		t.Fatalf("loose backlog should drop: before=%d after=%d", res.Before.Count, res.After.Count)
	}
	if res.After.InPack <= res.Before.InPack {
		t.Fatalf("folded objects should land in a pack (moved, not deleted): before in-pack=%d after=%d",
			res.Before.InPack, res.After.InPack)
	}
}

// TestRunMaintAlwaysSafeRunsWhenLocked: a live lock defers the grace tier with a
// structured LOCKED reason, but the always-safe tier still runs (safe mid-commit) and
// it is NOT an incident.
func TestRunMaintAlwaysSafeRunsWhenLocked(t *testing.T) {
	gitDir := scratchGit(t)
	writeLock(t, gitDir, "index.lock")
	f := &fakeMaint{posture: safePosture(), loose: 100, inPack: 500, packs: 11}
	res := RunMaint(context.Background(), f.run, MaintOptions{RepoRoot: filepath.Dir(gitDir), GitCommonDir: gitDir, Apply: true})

	if res.GraceRefused != MaintReasonLocked {
		t.Fatalf("grace tier should be LOCKED, got %q", res.GraceRefused)
	}
	if res.Incident {
		t.Fatalf("a lock is a deferral, not an incident")
	}
	muts := mutatingCalls(f.calls)
	if !containsStr(muts, "multi-pack-index write") || !containsStr(muts, "commit-graph write --reachable") {
		t.Fatalf("always-safe tier must still run when locked; got %v", muts)
	}
	for _, mustNot := range []string{"maintenance run --task=loose-objects", "prune-packed", "maintenance run --task=incremental-repack"} {
		if containsStr(muts, mustNot) {
			t.Fatalf("grace step %q must NOT run under a lock; got %v", mustNot, muts)
		}
	}
	if res.LooseDelta() != 0 {
		t.Fatalf("nothing should be folded under a lock: delta=%d", res.LooseDelta())
	}
}

// TestRunMaintWorktreeLockDefers: a per-worktree fak-commit.lock (a peer session
// committing in a linked worktree) is seen by the preflight and defers the grace tier.
func TestRunMaintWorktreeLockDefers(t *testing.T) {
	gitDir := scratchGit(t)
	writeLock(t, gitDir, "worktrees/wt-verify/fak-commit.lock")
	f := &fakeMaint{posture: safePosture(), loose: 100, inPack: 500, packs: 11}
	res := RunMaint(context.Background(), f.run, MaintOptions{RepoRoot: filepath.Dir(gitDir), GitCommonDir: gitDir, Apply: true})

	if res.GraceRefused != MaintReasonLocked {
		t.Fatalf("worktree lock should defer grace with LOCKED, got %q (locks=%v)", res.GraceRefused, res.Locks)
	}
	if !containsStr(res.Locks, "worktrees/wt-verify/fak-commit.lock") {
		t.Fatalf("preflight must report the worktree lock; got %v", res.Locks)
	}
}

// TestRunMaintPostureDriftRefusesGrace: gc.auto>0 / maintenance.auto unset ⇒ the grace
// tier REFUSES with POSTURE_DRIFT and is surfaced as an incident, while the always-safe
// tier still runs.
func TestRunMaintPostureDriftRefusesGrace(t *testing.T) {
	cases := []struct {
		name    string
		posture map[string]string
	}{
		{"gc.auto nonzero", map[string]string{"gc.auto": "6700", "maintenance.auto": "false", "core.untrackedCache": "true"}},
		{"gc.auto unset", map[string]string{"maintenance.auto": "false", "core.untrackedCache": "true"}},
		{"maintenance.auto true", map[string]string{"gc.auto": "0", "maintenance.auto": "true", "core.untrackedCache": "true"}},
		{"maintenance.auto unset", map[string]string{"gc.auto": "0", "core.untrackedCache": "true"}},
		{"untrackedCache unset", map[string]string{"gc.auto": "0", "maintenance.auto": "false"}},
		{"untrackedCache false", map[string]string{"gc.auto": "0", "maintenance.auto": "false", "core.untrackedCache": "false"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			gitDir := scratchGit(t)
			f := &fakeMaint{posture: tc.posture, loose: 100, inPack: 500, packs: 11}
			res := RunMaint(context.Background(), f.run, MaintOptions{RepoRoot: filepath.Dir(gitDir), GitCommonDir: gitDir, Apply: true})

			if res.GraceRefused != MaintReasonPostureDrift {
				t.Fatalf("posture drift should refuse grace with POSTURE_DRIFT, got %q", res.GraceRefused)
			}
			if !res.Incident {
				t.Fatalf("posture drift must be surfaced as an incident")
			}
			if res.Posture.Safe {
				t.Fatalf("posture should read unsafe: %+v", res.Posture)
			}
			muts := mutatingCalls(f.calls)
			if !containsStr(muts, "multi-pack-index write") {
				t.Fatalf("always-safe tier must run even under posture drift; got %v", muts)
			}
			if containsStr(muts, "maintenance run --task=loose-objects") {
				t.Fatalf("grace step must NOT run under posture drift; got %v", muts)
			}
		})
	}
}

// TestRunMaintDryRunMutatesNothing: with Apply=false, RunMaint probes locks + posture
// and plans the steps, but issues NO mutating git verb — only the read probes.
func TestRunMaintDryRunMutatesNothing(t *testing.T) {
	gitDir := scratchGit(t)
	f := &fakeMaint{posture: safePosture(), loose: 100, inPack: 500, packs: 11}
	res := RunMaint(context.Background(), f.run, MaintOptions{RepoRoot: filepath.Dir(gitDir), GitCommonDir: gitDir, Apply: false})

	if muts := mutatingCalls(f.calls); len(muts) != 0 {
		t.Fatalf("dry-run must mutate nothing; got git steps %v", muts)
	}
	for _, s := range res.Steps {
		if s.Ran {
			t.Fatalf("dry-run step should be planned-only, not ran: %+v", s)
		}
	}
	if res.LooseDelta() != 0 {
		t.Fatalf("dry-run must not fold objects: delta=%d", res.LooseDelta())
	}
}

// TestRunMaintTOCTOURechecksLocks: no lock at preflight, but a peer takes a lock after
// the first grace step; the per-step re-probe catches it and defers the remaining grace
// step with LOCKED — the always-safe tier and the first grace step still ran.
func TestRunMaintTOCTOURechecksLocks(t *testing.T) {
	gitDir := scratchGit(t)
	f := &fakeMaint{posture: safePosture(), loose: 100, inPack: 500, packs: 11}
	f.onGrace = func(args []string) {
		// A peer starts committing right after the loose-objects fold: fabricate its
		// lock so the pre-step re-probe for prune-packed and incremental-repack sees it.
		if hasArg(args, "--task=loose-objects") {
			writeLock(t, gitDir, "index.lock")
		}
	}
	res := RunMaint(context.Background(), f.run, MaintOptions{RepoRoot: filepath.Dir(gitDir), GitCommonDir: gitDir, Apply: true})

	if res.GraceRefused != MaintReasonLocked {
		t.Fatalf("a lock taken mid-run should defer the rest with LOCKED, got %q", res.GraceRefused)
	}
	muts := mutatingCalls(f.calls)
	if !containsStr(muts, "maintenance run --task=loose-objects") {
		t.Fatalf("the first grace step should have run before the lock appeared; got %v", muts)
	}
	if containsStr(muts, "prune-packed") {
		t.Fatalf("prune-packed must be deferred after the mid-run lock; got %v", muts)
	}
	if containsStr(muts, "maintenance run --task=incremental-repack") {
		t.Fatalf("the second grace step must be deferred after the mid-run lock; got %v", muts)
	}
}

// TestProbeLocksCoversLockSet pins the lock-name coverage the preflight relies on.
func TestProbeLocksCoversLockSet(t *testing.T) {
	for _, rel := range []string{
		"index.lock",
		"gc.pid",
		"packed-refs.lock",
		"fak-commit.lock",
		"maintenance.lock",
		"objects/info/commit-graph.lock",
		"objects/pack/multi-pack-index.lock",
		"refs/heads/main.lock",
		"worktrees/wt1/fak-commit.lock",
	} {
		t.Run(rel, func(t *testing.T) {
			gitDir := scratchGit(t)
			writeLock(t, gitDir, rel)
			locks := probeLocks(gitDir)
			if !containsStr(locks, rel) {
				t.Fatalf("probeLocks did not report %q; got %v", rel, locks)
			}
		})
	}
	// A clean scratch git reports no locks.
	if locks := probeLocks(scratchGit(t)); len(locks) != 0 {
		t.Fatalf("clean git should have no locks; got %v", locks)
	}
}

// TestProbeLocksExcludesLeaseNamespace is the #4602 GAP-2 unit witness: fak's own lease
// heartbeat locks under refs/fak/locks/ (session-*, intent-*, resolve-*) are object-DB
// orthogonal and thus NOT counted as transaction locks — but a genuine ref-transaction
// lock (refs/heads/main.lock) and a real refs/tags lock ARE still counted, even when held
// alongside the leases. The exclusion is narrowed to fak's namespace; it does not blind
// the preflight to real ref transactions.
func TestProbeLocksExcludesLeaseNamespace(t *testing.T) {
	leases := []string{
		"refs/fak/locks/session-win-abc123.lock",
		"refs/fak/locks/session-guard.lock",
		"refs/fak/locks/intent-issue-4602.lock",
		"refs/fak/locks/resolve-cmd.lock",
	}
	gitDir := scratchGit(t)
	for _, rel := range leases {
		writeLock(t, gitDir, rel)
	}
	if locks := probeLocks(gitDir); len(locks) != 0 {
		t.Fatalf("lease-namespace locks must not count as transactions; got %v", locks)
	}

	// Real ref-transaction locks held alongside the leases are still reported; the leases
	// stay excluded.
	writeLock(t, gitDir, "refs/heads/main.lock")
	writeLock(t, gitDir, "refs/tags/v1.lock")
	locks := probeLocks(gitDir)
	for _, want := range []string{"refs/heads/main.lock", "refs/tags/v1.lock"} {
		if !containsStr(locks, want) {
			t.Fatalf("a real ref-transaction lock %q must still be reported; got %v", want, locks)
		}
	}
	for _, lease := range leases {
		if containsStr(locks, lease) {
			t.Fatalf("lease lock %q must stay excluded even with a real lock present; got %v", lease, locks)
		}
	}
}

// TestRunMaintSessionLeaseLocksDoNotDeferGrace is the #4602 GAP-2 end-to-end proof, and
// the failure-class witness that FAILS before this fix: on an always-live box the only
// locks present are fak session/intent lease heartbeats under refs/fak/locks/. The old
// probeLocks counted them, so len(res.Locks)>0 pinned the grace tier at MaintReasonLocked
// forever and the loose backlog was unbounded. Now the grace tier RUNS — the leases are
// excluded, nothing is refused, and the loose objects fold into a pack (moved, not pruned).
func TestRunMaintSessionLeaseLocksDoNotDeferGrace(t *testing.T) {
	gitDir := scratchGit(t)
	for _, rel := range []string{
		"refs/fak/locks/session-win-1.lock",
		"refs/fak/locks/session-win-2.lock",
		"refs/fak/locks/session-guard.lock",
		"refs/fak/locks/intent-issue-4602.lock",
	} {
		writeLock(t, gitDir, rel)
	}
	f := &fakeMaint{posture: safePosture(), loose: 100, inPack: 500, packs: 11}
	res := RunMaint(context.Background(), f.run, MaintOptions{RepoRoot: filepath.Dir(gitDir), GitCommonDir: gitDir, Apply: true})

	if res.GraceRefused != "" {
		t.Fatalf("session-lease heartbeats must not defer the grace tier: GraceRefused=%q locks=%v", res.GraceRefused, res.Locks)
	}
	if len(res.Locks) != 0 {
		t.Fatalf("session-lease heartbeats must not be reported as transaction locks; got %v", res.Locks)
	}
	if !containsStr(mutatingCalls(f.calls), "maintenance run --task=loose-objects") {
		t.Fatalf("the grace fold must run when only session leases are held; got %v", mutatingCalls(f.calls))
	}
	if res.LooseDelta() <= 0 {
		t.Fatalf("the loose backlog should fold once lease locks no longer block grace: before=%d after=%d", res.Before.Count, res.After.Count)
	}
}

// TestRunMaintRealRefLockStillDefersDespiteLeases: a genuine refs/heads/main.lock held
// alongside the session leases still defers the grace tier with LOCKED, and the real lock
// (not the leases) is what the preflight reports — the fix narrows the exclusion, it does
// not disable transaction-lock detection.
func TestRunMaintRealRefLockStillDefersDespiteLeases(t *testing.T) {
	gitDir := scratchGit(t)
	writeLock(t, gitDir, "refs/fak/locks/session-win-1.lock")
	writeLock(t, gitDir, "refs/heads/main.lock")
	f := &fakeMaint{posture: safePosture(), loose: 100, inPack: 500, packs: 11}
	res := RunMaint(context.Background(), f.run, MaintOptions{RepoRoot: filepath.Dir(gitDir), GitCommonDir: gitDir, Apply: true})

	if res.GraceRefused != MaintReasonLocked {
		t.Fatalf("a real ref-transaction lock must still defer grace with LOCKED, got %q", res.GraceRefused)
	}
	if !containsStr(res.Locks, "refs/heads/main.lock") {
		t.Fatalf("the real ref lock must be reported; got %v", res.Locks)
	}
	if containsStr(res.Locks, "refs/fak/locks/session-win-1.lock") {
		t.Fatalf("the session lease must stay excluded even when a real lock defers; got %v", res.Locks)
	}
	if res.LooseDelta() != 0 {
		t.Fatalf("nothing should fold while a real ref lock is held: delta=%d", res.LooseDelta())
	}
}

func containsStr(xs []string, want string) bool {
	for _, x := range xs {
		if x == want {
			return true
		}
	}
	return false
}

// TestLooseBacklogHighPredicate: the pure #4602 high-water predicate — fail-closed on an
// unavailable count, and the threshold is an inclusive floor.
func TestLooseBacklogHighPredicate(t *testing.T) {
	cases := []struct {
		name string
		co   CountObjects
		want bool
	}{
		{"unavailable-never-high", CountObjects{Available: false, Count: 1_000_000}, false},
		{"below-threshold", CountObjects{Available: true, Count: LooseBacklogThreshold - 1}, false},
		{"at-threshold", CountObjects{Available: true, Count: LooseBacklogThreshold}, true},
		{"above-threshold", CountObjects{Available: true, Count: LooseBacklogThreshold + 5}, true},
	}
	for _, tc := range cases {
		if got := LooseBacklogHigh(tc.co); got != tc.want {
			t.Errorf("%s: LooseBacklogHigh(count=%d,avail=%v)=%v want %v", tc.name, tc.co.Count, tc.co.Available, got, tc.want)
		}
	}
}

// TestRunMaintLooseBacklogHighFromPreRunCount: RunMaint sets LooseBacklogHigh from the
// PRE-run count (the invisible backlog the operator needs to see), even though the grace
// tier then folds those loose objects away. A small backlog leaves the flag false.
func TestRunMaintLooseBacklogHighFromPreRunCount(t *testing.T) {
	gitDir := scratchGit(t)

	high := &fakeMaint{posture: safePosture(), loose: LooseBacklogThreshold + 2_000, inPack: 500, packs: 11}
	res := RunMaint(context.Background(), high.run, MaintOptions{RepoRoot: filepath.Dir(gitDir), GitCommonDir: gitDir, Apply: true})
	if !res.LooseBacklogHigh {
		t.Fatalf("a %d-loose pre-run backlog should set LooseBacklogHigh (before.Count=%d)", LooseBacklogThreshold+2_000, res.Before.Count)
	}
	if res.After.Count >= res.Before.Count {
		t.Fatalf("sanity: grace tier should still fold loose objects: before=%d after=%d", res.Before.Count, res.After.Count)
	}

	low := &fakeMaint{posture: safePosture(), loose: 100, inPack: 500, packs: 11}
	res2 := RunMaint(context.Background(), low.run, MaintOptions{RepoRoot: filepath.Dir(gitDir), GitCommonDir: gitDir, Apply: true})
	if res2.LooseBacklogHigh {
		t.Fatalf("a 100-loose backlog must not set LooseBacklogHigh (before.Count=%d)", res2.Before.Count)
	}
}

// TestReadPostureFsmonitorHealth is the #4603 failure-class witness: core.fsmonitor=true
// with a DEAD builtin daemon ("not watching") must read as posture DRIFT — the cold-git-op
// stall the safe-posture check previously ignored — while fsmonitor OFF or an affirmatively
// watching daemon stays safe. It also pins that the daemon is probed ONLY when a git-true
// value selects the builtin (a hook-program PATH and the off/unset cases run no probe).
func TestReadPostureFsmonitorHealth(t *testing.T) {
	base := func() map[string]string { return safePosture() }
	cases := []struct {
		name        string
		fsmonitor   string // "" = key unset
		fsmon       string // `git fsmonitor--daemon status` text
		fsmonErr    bool   // git could not run the probe at all
		wantSafe    bool
		wantDaemon  string
		wantProbed  bool
		driftNeedle string // substring required in Drift when !wantSafe
	}{
		{"unset-is-off-safe", "", "", false, true, "", false, ""},
		{"explicit-false-safe", "false", "", false, true, "", false, ""},
		{"true-watching-safe", "true", "fsmonitor-daemon is watching '/x'", false, true, fsmonitorWatching, true, ""},
		{"true-dead-daemon-drift", "true", "fsmonitor-daemon is not watching '/x'", false, false, fsmonitorNotWatching, true, "not-watching"},
		{"true-unprobeable-unknown-drift", "true", "", true, false, fsmonitorUnknown, true, "unknown"},
		{"hook-program-path-safe", "/usr/local/bin/my-fsmonitor-hook", "", false, true, "", false, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			posture := base()
			if tc.fsmonitor != "" {
				posture["core.fsmonitor"] = tc.fsmonitor
			}
			f := &fakeMaint{posture: posture, fsmon: tc.fsmon, fsmonErr: tc.fsmonErr}
			p := readPosture(context.Background(), f.run, t.TempDir())
			if p.Safe != tc.wantSafe {
				t.Fatalf("Safe=%v want %v (drift=%q, daemon=%q)", p.Safe, tc.wantSafe, p.Drift, p.FsmonitorDaemon)
			}
			if p.FsmonitorDaemon != tc.wantDaemon {
				t.Fatalf("FsmonitorDaemon=%q want %q", p.FsmonitorDaemon, tc.wantDaemon)
			}
			probed := false
			for _, c := range f.calls {
				if len(c) >= 3 && c[1] == "fsmonitor--daemon" && c[2] == "status" {
					probed = true
				}
			}
			if probed != tc.wantProbed {
				t.Fatalf("daemon probed=%v want %v (fsmonitor=%q)", probed, tc.wantProbed, tc.fsmonitor)
			}
			if !tc.wantSafe && tc.driftNeedle != "" && !strings.Contains(p.Drift, tc.driftNeedle) {
				t.Fatalf("drift %q should mention %q", p.Drift, tc.driftNeedle)
			}
		})
	}
}

// TestReadPostureUntrackedCache is the #5069 witness: core.untrackedCache is MANAGED by
// the posture assert — unset or git-false reads as drift (every cold `git status` would
// silently full-scan the ~10k-file tree), while any git-true spelling is safe. This is
// what keeps the daemon-independent cold-status speedup from regressing by manual drift.
func TestReadPostureUntrackedCache(t *testing.T) {
	cases := []struct {
		name     string
		value    string // "" = key unset
		wantSafe bool
	}{
		{"unset-drift", "", false},
		{"false-drift", "false", false},
		{"true-safe", "true", true},
		{"on-safe", "on", true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			posture := safePosture()
			delete(posture, "core.untrackedCache")
			if tc.value != "" {
				posture["core.untrackedCache"] = tc.value
			}
			f := &fakeMaint{posture: posture}
			p := readPosture(context.Background(), f.run, t.TempDir())
			if p.Safe != tc.wantSafe {
				t.Fatalf("Safe=%v want %v (drift=%q)", p.Safe, tc.wantSafe, p.Drift)
			}
			if p.UntrackedCache != tc.value {
				t.Fatalf("UntrackedCache=%q want %q", p.UntrackedCache, tc.value)
			}
			if !tc.wantSafe && !strings.Contains(p.Drift, "core.untrackedCache") {
				t.Fatalf("drift %q should mention core.untrackedCache", p.Drift)
			}
		})
	}
}

// TestRunMaintFsmonitorDeadDaemonRefusesGrace is the end-to-end #4603 proof: a hot clone
// with core.fsmonitor=true but a DEAD daemon refuses the grace tier with POSTURE_DRIFT and
// raises an incident (the operator must start the daemon or unset the key), while the
// add-only always-safe tier still runs and nothing is folded.
func TestRunMaintFsmonitorDeadDaemonRefusesGrace(t *testing.T) {
	gitDir := scratchGit(t)
	posture := safePosture()
	posture["core.fsmonitor"] = "true"
	f := &fakeMaint{posture: posture, loose: 100, inPack: 500, packs: 11,
		fsmon: "fsmonitor-daemon is not watching 'C:/work/fak'"}
	res := RunMaint(context.Background(), f.run, MaintOptions{RepoRoot: filepath.Dir(gitDir), GitCommonDir: gitDir, Apply: true})

	if res.Posture.Safe {
		t.Fatalf("fsmonitor=true with a dead daemon must read unsafe: %+v", res.Posture)
	}
	if res.Posture.FsmonitorDaemon != fsmonitorNotWatching {
		t.Fatalf("daemon health=%q want %q", res.Posture.FsmonitorDaemon, fsmonitorNotWatching)
	}
	if res.GraceRefused != MaintReasonPostureDrift {
		t.Fatalf("dead-daemon drift should refuse grace with POSTURE_DRIFT, got %q", res.GraceRefused)
	}
	if !res.Incident {
		t.Fatalf("dead-daemon drift must be surfaced as an incident")
	}
	muts := mutatingCalls(f.calls)
	if !containsStr(muts, "multi-pack-index write") {
		t.Fatalf("always-safe tier must still run under fsmonitor drift; got %v", muts)
	}
	if containsStr(muts, "maintenance run --task=loose-objects") {
		t.Fatalf("grace step must NOT run under fsmonitor drift; got %v", muts)
	}
	if res.LooseDelta() != 0 {
		t.Fatalf("nothing should be folded under posture drift: delta=%d", res.LooseDelta())
	}
}

// pruneCalls returns every recorded `git prune …` argv (after the dir word), so a
// grace-prune test can assert exactly what — if anything — was asked of git.
func pruneCalls(calls [][]string) [][]string {
	var out [][]string
	for _, c := range calls {
		if len(c) >= 2 && c[1] == "prune" {
			out = append(out, c[1:])
		}
	}
	return out
}

// TestGracePruneDefaultOff is the #5079 default-off witness: with MaintOptions.GracePrune
// unset, an otherwise clean+quiet+safe apply run issues NO prune argv at all — the tier is
// recorded as skipped with the structured PRUNE_OFF reason, it is not an incident, and the
// fold tiers run exactly as before.
func TestGracePruneDefaultOff(t *testing.T) {
	gitDir := scratchGit(t)
	f := &fakeMaint{posture: safePosture(), loose: 100, unreachable: 80, inPack: 500, packs: 11}
	res := RunMaint(context.Background(), f.run, MaintOptions{RepoRoot: filepath.Dir(gitDir), GitCommonDir: gitDir, Apply: true})

	if got := pruneCalls(f.calls); len(got) != 0 {
		t.Fatalf("default-off must never issue a prune argv; got %v", got)
	}
	if res.GracePruneRefused != MaintReasonPruneOff {
		t.Fatalf("grace-prune should be held back with PRUNE_OFF, got %q", res.GracePruneRefused)
	}
	if res.Incident {
		t.Fatalf("default-off prune is not an incident")
	}
	if !containsStr(mutatingCalls(f.calls), "maintenance run --task=loose-objects") {
		t.Fatalf("the fold tiers must be unaffected by the prune gate; got %v", mutatingCalls(f.calls))
	}
	var found bool
	for _, s := range res.Steps {
		if s.Tier == gracePruneTier {
			found = true
			if s.Ran || s.Skipped != MaintReasonPruneOff {
				t.Fatalf("grace-prune step should be skipped PRUNE_OFF, got %+v", s)
			}
		}
	}
	if !found {
		t.Fatalf("the skipped grace-prune step must still be recorded; steps=%+v", res.Steps)
	}
}

// TestGracePruneRunsInQuietWindow is the positive-path witness: opted in, safe posture, no
// transaction lock, no session lease ⇒ exactly ONE `git prune` runs, its argv carries the
// 2-week floor expire, and the unreachable backlog drops WITHOUT growing in-pack (removed,
// not folded — the reclaim the fold tiers structurally cannot deliver).
func TestGracePruneRunsInQuietWindow(t *testing.T) {
	gitDir := scratchGit(t)
	f := &fakeMaint{posture: safePosture(), loose: 100, unreachable: 80, inPack: 500, packs: 11}
	res := RunMaint(context.Background(), f.run, MaintOptions{
		RepoRoot: filepath.Dir(gitDir), GitCommonDir: gitDir, Apply: true, GracePrune: true})

	prunes := pruneCalls(f.calls)
	if len(prunes) != 1 {
		t.Fatalf("exactly one prune per run; got %v", prunes)
	}
	if want := []string{"prune", "--expire=" + defaultPruneExpire}; !equalArgs(prunes[0], want) {
		t.Fatalf("prune argv = %v, want %v", prunes[0], want)
	}
	if res.GracePruneRefused != "" {
		t.Fatalf("quiet+safe+opted-in must not refuse grace-prune: %q", res.GracePruneRefused)
	}
	// The fold tier grows in-pack by what it MOVED; anything the run dropped beyond
	// that vanished outright — the prune's reclaim. It must equal the unreachable set.
	removedOutright := res.LooseDelta() - (res.After.InPack - res.Before.InPack)
	if removedOutright != 80 {
		t.Fatalf("prune should reclaim the 80 unreachable objects outright (removed, not folded): got %d (before=%+v after=%+v)",
			removedOutright, res.Before, res.After)
	}
}

// TestGracePruneRefusesUnderSessionLease is the load-bearing quiet-window witness (#5079):
// the SAME session-lease heartbeats that must NOT defer the fold tier (#4602 GAP 2) MUST
// refuse the grace-prune tier with SESSION_LIVE — a fold can run beside a live session, a
// prune must not. No prune argv is issued; the leases are reported as the evidence.
func TestGracePruneRefusesUnderSessionLease(t *testing.T) {
	gitDir := scratchGit(t)
	for _, rel := range []string{
		"refs/fak/locks/session-win-1.lock",
		"refs/fak/locks/intent-issue-5079", // a loose lease REF (no .lock suffix) also blocks
	} {
		writeLock(t, gitDir, rel)
	}
	f := &fakeMaint{posture: safePosture(), loose: 100, unreachable: 80, inPack: 500, packs: 11}
	res := RunMaint(context.Background(), f.run, MaintOptions{
		RepoRoot: filepath.Dir(gitDir), GitCommonDir: gitDir, Apply: true, GracePrune: true})

	if res.GraceRefused != "" {
		t.Fatalf("session leases must still not defer the FOLD tier: %q", res.GraceRefused)
	}
	if !containsStr(mutatingCalls(f.calls), "maintenance run --task=loose-objects") {
		t.Fatalf("the fold must run beside live sessions; got %v", mutatingCalls(f.calls))
	}
	if res.GracePruneRefused != MaintReasonSessionLive {
		t.Fatalf("grace-prune must refuse with SESSION_LIVE under a live lease, got %q", res.GracePruneRefused)
	}
	if got := pruneCalls(f.calls); len(got) != 0 {
		t.Fatalf("no prune argv may be issued while a session is live; got %v", got)
	}
	for _, want := range []string{"refs/fak/locks/session-win-1.lock", "refs/fak/locks/intent-issue-5079"} {
		if !containsStr(res.SessionLeases, want) {
			t.Fatalf("the refusing leases must be reported; got %v", res.SessionLeases)
		}
	}
}

// TestGracePruneRefusesUnderTransactionLockAndPostureDrift: the prune tier honors the SAME
// gates as the fold tier — a live transaction lock refuses LOCKED, posture drift refuses
// POSTURE_DRIFT — before the quiet-window question is even asked. No prune argv either way.
func TestGracePruneRefusesUnderTransactionLockAndPostureDrift(t *testing.T) {
	t.Run("transaction lock", func(t *testing.T) {
		gitDir := scratchGit(t)
		writeLock(t, gitDir, "index.lock")
		f := &fakeMaint{posture: safePosture(), loose: 100, unreachable: 80, inPack: 500, packs: 11}
		res := RunMaint(context.Background(), f.run, MaintOptions{
			RepoRoot: filepath.Dir(gitDir), GitCommonDir: gitDir, Apply: true, GracePrune: true})
		if res.GracePruneRefused != MaintReasonLocked {
			t.Fatalf("grace-prune under index.lock should refuse LOCKED, got %q", res.GracePruneRefused)
		}
		if got := pruneCalls(f.calls); len(got) != 0 {
			t.Fatalf("no prune argv under a transaction lock; got %v", got)
		}
	})
	t.Run("posture drift", func(t *testing.T) {
		gitDir := scratchGit(t)
		f := &fakeMaint{posture: map[string]string{"maintenance.auto": "false", "core.untrackedCache": "true"}, // gc.auto unset
			loose: 100, unreachable: 80, inPack: 500, packs: 11}
		res := RunMaint(context.Background(), f.run, MaintOptions{
			RepoRoot: filepath.Dir(gitDir), GitCommonDir: gitDir, Apply: true, GracePrune: true})
		if res.GracePruneRefused != MaintReasonPostureDrift {
			t.Fatalf("grace-prune under posture drift should refuse POSTURE_DRIFT, got %q", res.GracePruneRefused)
		}
		if got := pruneCalls(f.calls); len(got) != 0 {
			t.Fatalf("no prune argv under posture drift; got %v", got)
		}
	})
}

// TestGracePruneExpireFloorEnforced is the DoD expire-floor witness: a sub-floor or
// unparseable expire (`now` above all — the Mata-class forbidden form) REFUSES with
// PRUNE_EXPIRE_UNSAFE and no prune argv is ever built, while floor-or-wider windows run
// with exactly the requested expire. Every argv that DOES reach git carries an expire from
// the validated ≥2w set — `--expire=now` can never be emitted.
func TestGracePruneExpireFloorEnforced(t *testing.T) {
	cases := []struct {
		expire   string
		wantRun  bool
		wantArgv string // the --expire value expected when wantRun
	}{
		{"", true, defaultPruneExpire},
		{"2.weeks.ago", true, "2.weeks.ago"},
		{"6.weeks.ago", true, "6.weeks.ago"},
		{"1.months.ago", true, "1.months.ago"},
		{"1.years.ago", true, "1.years.ago"},
		{"now", false, ""},
		{"all", false, ""},
		{"1.weeks.ago", false, ""},
		{"0.weeks.ago", false, ""},
		{"13.days.ago", false, ""},  // day units are not provably ≥2w — fail-closed
		{"2.weeks", false, ""},      // not the closed <n>.<unit>.ago form
		{"2026-01-01", false, ""},   // free-text dates refused
		{"-3.weeks.ago", false, ""}, // negative refused
	}
	for _, tc := range cases {
		name := tc.expire
		if name == "" {
			name = "<default>"
		}
		t.Run(name, func(t *testing.T) {
			gitDir := scratchGit(t)
			f := &fakeMaint{posture: safePosture(), loose: 100, unreachable: 80, inPack: 500, packs: 11}
			res := RunMaint(context.Background(), f.run, MaintOptions{
				RepoRoot: filepath.Dir(gitDir), GitCommonDir: gitDir, Apply: true, GracePrune: true, PruneExpire: tc.expire})

			prunes := pruneCalls(f.calls)
			if tc.wantRun {
				if len(prunes) != 1 || !equalArgs(prunes[0], []string{"prune", "--expire=" + tc.wantArgv}) {
					t.Fatalf("expire %q should run as --expire=%s; got %v", tc.expire, tc.wantArgv, prunes)
				}
				if res.GracePruneRefused != "" {
					t.Fatalf("valid expire %q must not refuse: %q", tc.expire, res.GracePruneRefused)
				}
			} else {
				if len(prunes) != 0 {
					t.Fatalf("sub-floor expire %q must never reach git; got %v", tc.expire, prunes)
				}
				if res.GracePruneRefused != MaintReasonPruneExpireUnsafe {
					t.Fatalf("sub-floor expire %q should refuse PRUNE_EXPIRE_UNSAFE, got %q", tc.expire, res.GracePruneRefused)
				}
				// The recorded skipped step must not carry the refused value either.
				for _, s := range res.Steps {
					if s.Tier == gracePruneTier && !equalArgs(s.Args, []string{"prune", "--expire=" + defaultPruneExpire}) {
						t.Fatalf("a refused expire must not appear in the step record: %+v", s)
					}
				}
			}
		})
	}
}

// TestGracePruneDryRunPlansOnly: dry-run + opted-in + quiet plans the prune step (Ran=false,
// no refusal) and issues no mutating argv at all.
func TestGracePruneDryRunPlansOnly(t *testing.T) {
	gitDir := scratchGit(t)
	f := &fakeMaint{posture: safePosture(), loose: 100, unreachable: 80, inPack: 500, packs: 11}
	res := RunMaint(context.Background(), f.run, MaintOptions{
		RepoRoot: filepath.Dir(gitDir), GitCommonDir: gitDir, Apply: false, GracePrune: true})

	if got := pruneCalls(f.calls); len(got) != 0 {
		t.Fatalf("dry-run must issue no prune argv; got %v", got)
	}
	if res.GracePruneRefused != "" {
		t.Fatalf("a quiet dry-run plans the prune, it does not refuse it: %q", res.GracePruneRefused)
	}
	for _, s := range res.Steps {
		if s.Tier == gracePruneTier && (s.Ran || s.Skipped != "") {
			t.Fatalf("dry-run prune step should be planned-only: %+v", s)
		}
	}
}

func equalArgs(got, want []string) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if got[i] != want[i] {
			return false
		}
	}
	return true
}

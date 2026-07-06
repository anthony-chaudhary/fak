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
	calls   [][]string
	onGrace func(args []string)
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
		if hasArg(args, "--task=loose-objects") && f.loose > 10 {
			f.inPack += f.loose - 10 // loose folded into a pack — moved, not deleted
			f.loose = 10
		}
		return "", 0, nil
	}
	return "", 0, nil
}

func (f *fakeMaint) countText() string {
	return fmt.Sprintf("count: %d\nsize: 1.00 MiB\nin-pack: %d\npacks: %d\nprune-packable: 0\ngarbage: 0\nsize-garbage: 0 bytes\n",
		f.loose, f.inPack, f.packs)
}

func hasArg(args []string, want string) bool {
	for _, a := range args {
		if a == want {
			return true
		}
	}
	return false
}

// safePosture is the shared no-auto-gc posture the hot clone holds.
func safePosture() map[string]string {
	return map[string]string{"gc.auto": "0", "maintenance.auto": "false"}
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
		case "multi-pack-index", "commit-graph", "maintenance":
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
	// 2 always-safe + 2 grace verbs must have hit git.
	for _, want := range []string{
		"multi-pack-index write", "commit-graph write --reachable",
		"maintenance run --task=loose-objects", "maintenance run --task=incremental-repack",
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
	for _, mustNot := range []string{"maintenance run --task=loose-objects", "maintenance run --task=incremental-repack"} {
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
		{"gc.auto nonzero", map[string]string{"gc.auto": "6700", "maintenance.auto": "false"}},
		{"gc.auto unset", map[string]string{"maintenance.auto": "false"}},
		{"maintenance.auto true", map[string]string{"gc.auto": "0", "maintenance.auto": "true"}},
		{"maintenance.auto unset", map[string]string{"gc.auto": "0"}},
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
		// lock so the pre-step re-probe for incremental-repack sees it.
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

func containsStr(xs []string, want string) bool {
	for _, x := range xs {
		if x == want {
			return true
		}
	}
	return false
}

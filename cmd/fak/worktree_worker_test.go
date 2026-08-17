package main

import (
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/leaseref"
	"github.com/anthony-chaudhary/fak/internal/workerworktree"
)

// mustKeys asserts the marshaled JSON has exactly the expected top-level keys —
// the CLI contract a caller (tools/worker_worktree.py's consumers, the dispatcher)
// parses. Extra or missing keys are a contract break.
func mustKeys(t *testing.T, v any, want ...string) map[string]any {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var got map[string]any
	if err := json.Unmarshal(b, &got); err != nil {
		t.Fatalf("unmarshal %s: %v", b, err)
	}
	for _, k := range want {
		if _, ok := got[k]; !ok {
			t.Fatalf("json %s missing key %q", b, k)
		}
	}
	return got
}

// TestPrepareJSONShape proves `fak worktree worker prepare` emits one object with
// the primitive's fields flattened plus the child env, and that the env isolates
// GOCACHE into the worktree — the shape a spawn site reads.
func TestPrepareJSONShape(t *testing.T) {
	out := worktreePrepareOut{
		Result: workerworktree.Result{OK: true, Path: "/wt/fak-worker-wt-cmd-abc", BaseSHA: "feedface", Reused: false},
		Env:    workerworktree.WorktreeEnv(nil, "/wt/fak-worker-wt-cmd-abc"),
	}
	got := mustKeys(t, out, "ok", "path", "base_sha", "env")
	if got["ok"] != true {
		t.Fatalf("ok = %v, want true", got["ok"])
	}
	env, ok := got["env"].(map[string]any)
	if !ok {
		t.Fatalf("env not an object: %T", got["env"])
	}
	if _, ok := env["GOCACHE"]; !ok {
		t.Fatal("prepare env must carry GOCACHE to isolate the build")
	}
}

// TestPrepareFailOpenJSONShape proves a failed prepare is still a well-formed
// object (ok=false, reason set) — never a crash — and omits env.
func TestPrepareFailOpenJSONShape(t *testing.T) {
	out := worktreePrepareOut{Result: workerworktree.Result{OK: false, Reason: "could not resolve trunk HEAD — fail open"}}
	got := mustKeys(t, out, "ok", "reason")
	if got["ok"] != false {
		t.Fatalf("ok = %v, want false", got["ok"])
	}
	if _, hasEnv := got["env"]; hasEnv {
		t.Fatal("a failed prepare must not carry env")
	}
}

// TestLandJSONShape proves `fak worktree worker land` emits the applied/committed
// verdict object.
func TestLandJSONShape(t *testing.T) {
	res := workerworktree.Result{OK: true, Applied: true, Committed: true}
	got := mustKeys(t, res, "ok", "applied", "committed")
	if got["committed"] != true {
		t.Fatalf("committed = %v, want true", got["committed"])
	}
}

// TestReapJSONShape proves `fak worktree worker reap` emits the removed verdict.
func TestReapJSONShape(t *testing.T) {
	res := workerworktree.Result{OK: true, Path: "/wt/fak-worker-wt-cmd-abc", Removed: true}
	mustKeys(t, res, "ok", "path", "removed")
}

// TestGCJSONShape pins the owner-stamped GC CLI's stable top-level report contract.
func TestGCJSONShape(t *testing.T) {
	out := workerworktree.GCReport{
		Mode:      "dry-run",
		MaxAgeSec: 1800,
		Worktrees: []workerworktree.GCWorktree{},
		WouldReap: 0,
		Reaped:    0,
	}
	got := mustKeys(t, out, "mode", "max_age_sec", "worktrees", "would_reap", "reaped")
	if got["mode"] != "dry-run" {
		t.Fatalf("mode = %v, want dry-run", got["mode"])
	}
}

// TestListJSONShape proves `fak worktree worker list` preserves count/paths and
// includes the lifecycle inventory added by #5994. Empty collections render as []
// (never null), so JSON consumers can always range both fields.
func TestListJSONShape(t *testing.T) {
	empty := worktreeWorkerListOut{Count: 0, Paths: []string{}, Inventory: []workerworktree.InventoryRow{}}
	b, err := json.Marshal(empty)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if string(b) != `{"count":0,"paths":[],"inventory":[]}` {
		t.Fatalf("empty list json = %s, want empty paths and inventory arrays", b)
	}
	got := mustKeys(t, worktreeWorkerListOut{Count: 2, Paths: []string{"/a", "/b"}, Inventory: []workerworktree.InventoryRow{}}, "count", "paths", "inventory")
	if got["count"].(float64) != 2 {
		t.Fatalf("count = %v, want 2", got["count"])
	}
}

// TestDispatchLeaseLanes proves the lease-id -> lane parser the cold sweep's liveness
// gate binds on: a lane lease yields its lane; an issue lease yields both the full tail
// and the issue-stripped lane; a non-resolve id yields nothing.
func TestDispatchLeaseLanes(t *testing.T) {
	cases := []struct {
		id   string
		want []string
	}{
		{"resolve-cmd", []string{"cmd"}},
		{"resolve-gateway-5351", []string{"gateway-5351", "gateway"}},
		{"resolve-some-lane-42", []string{"some-lane-42", "some-lane"}},
		{"session-abc", nil},
		{"resolve-", nil},
	}
	for _, c := range cases {
		if got := dispatchLeaseLanes(c.id); !reflect.DeepEqual(got, c.want) {
			t.Fatalf("dispatchLeaseLanes(%q) = %v, want %v", c.id, got, c.want)
		}
	}
}

// TestWorktreeColdReapPlanEndToEnd drives the bulk cold sweep against a REAL temp git
// repo with three worker worktrees, covering every #5351 DoD case in one pass:
//   - a dead-lease worktree past the age floor is COLD (eligible),
//   - a dead-lease worktree under the floor is KEPT,
//   - a worktree whose lane holds a LIVE lease is NEVER reaped,
//   - the default DRY-RUN reaps nothing but reports the eligible set,
//   - --apply then removes exactly the cold one via workerworktree.Reap.
func TestWorktreeColdReapPlanEndToEnd(t *testing.T) {
	if testing.Short() {
		t.Skip("integration test; skipped under -short")
	}
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	repo := t.TempDir()
	wtRoot := t.TempDir()
	git := func(args ...string) (string, error) {
		c := exec.Command("git", append([]string{"-C", repo}, args...)...)
		c.Env = append(os.Environ(),
			"GIT_AUTHOR_NAME=t", "GIT_AUTHOR_EMAIL=t@t",
			"GIT_COMMITTER_NAME=t", "GIT_COMMITTER_EMAIL=t@t")
		out, err := c.CombinedOutput()
		return string(out), err
	}
	if _, err := git("init", "-q", "-b", "main"); err != nil {
		if _, e2 := git("init", "-q"); e2 != nil {
			t.Skipf("git init failed: %v", e2)
		}
		_, _ = git("symbolic-ref", "HEAD", "refs/heads/main")
	}
	_, _ = git("config", "user.email", "t@t")
	_, _ = git("config", "user.name", "t")
	if err := os.WriteFile(filepath.Join(repo, "seed.txt"), []byte("seed\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := git("add", "seed.txt"); err != nil {
		t.Fatal(err)
	}
	if out, err := git("commit", "-qm", "seed"); err != nil {
		t.Skipf("seed commit failed: %s", out)
	}

	// Three real worker worktrees: docs/old (dead lease), docs/young (dead lease),
	// cmd/held (its lane holds a live lease below).
	mk := func(lane, key string) string {
		res := workerworktree.Prepare(repo, lane, key, "", wtRoot, nil)
		if !res.OK {
			t.Fatalf("prepare %s/%s: %+v", lane, key, res)
		}
		return res.Path
	}
	oldWT := mk("docs", "111")
	youngWT := mk("docs", "222")
	heldWT := mk("cmd", "333")

	// A live lane lease on "cmd" protects heldWT (dispatchLaneLeaseID("cmd")).
	if _, err := leaseref.NewInDir(repo).Acquire(context.Background(),
		leaseref.Record{ID: "resolve-cmd", TTLSeconds: 3600, AcquiredAt: time.Now().Unix()}); err != nil {
		t.Fatalf("acquire live lease: %v", err)
	}

	// Age oldWT and heldWT past the floor; leave youngWT fresh.
	now := time.Now()
	floor := 30 * time.Minute
	back := now.Add(-2 * time.Hour)
	for _, p := range []string{oldWT, heldWT} {
		if err := os.Chtimes(p, back, back); err != nil {
			t.Fatalf("chtimes %s: %v", p, err)
		}
	}

	// DRY-RUN: exactly oldWT is eligible; nothing is deleted.
	dry := worktreeColdReapReport(repo, false, floor, now, false)
	if dry.Mode != "dry-run" || dry.Reaped != 0 {
		t.Fatalf("dry-run must delete nothing: mode=%s reaped=%d", dry.Mode, dry.Reaped)
	}
	if dry.WouldReap != 1 {
		t.Fatalf("want exactly 1 eligible (oldWT), got would_reap=%d: %+v", dry.WouldReap, dry.Worktrees)
	}
	// Count returns git's forward-slash paths; Prepare returns native paths — compare
	// on a normalized form so the Windows slash direction does not spuriously mismatch.
	norm := func(p string) string { return filepath.ToSlash(filepath.Clean(p)) }
	elig := map[string]bool{}
	for _, w := range dry.Worktrees {
		if w.Eligible {
			elig[norm(w.Path)] = true
		}
	}
	if !elig[norm(oldWT)] {
		t.Fatalf("oldWT (dead lease, past floor) must be eligible; ledger=%+v", dry.Worktrees)
	}
	if elig[norm(heldWT)] {
		t.Fatal("heldWT (live lane lease) must NEVER be eligible")
	}
	if elig[norm(youngWT)] {
		t.Fatal("youngWT (under age floor) must be kept")
	}
	if _, err := os.Stat(oldWT); err != nil {
		t.Fatalf("dry-run must not delete oldWT: %v", err)
	}

	// APPLY: oldWT is reaped, the protected/young ones survive.
	got := worktreeColdReapReport(repo, true, floor, now, false)
	if got.Mode != "apply" || got.Reaped != 1 || got.WouldReap != 1 {
		t.Fatalf("apply want reaped=1 would_reap=1 mode=apply, got %+v", got)
	}
	if _, err := os.Stat(oldWT); !os.IsNotExist(err) {
		t.Fatalf("apply must remove oldWT, stat err=%v", err)
	}
	if _, err := os.Stat(heldWT); err != nil {
		t.Fatalf("apply must keep the live-lease worktree heldWT: %v", err)
	}
	if _, err := os.Stat(youngWT); err != nil {
		t.Fatalf("apply must keep the young worktree youngWT: %v", err)
	}
}

func newColdReapProbeFixture(t *testing.T) (repo, wt string, now time.Time, floor time.Duration) {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	repo = t.TempDir()
	wtRoot := t.TempDir()
	git := func(args ...string) (string, error) {
		c := exec.Command("git", append([]string{"-C", repo}, args...)...)
		c.Env = append(os.Environ(),
			"GIT_AUTHOR_NAME=t", "GIT_AUTHOR_EMAIL=t@t",
			"GIT_COMMITTER_NAME=t", "GIT_COMMITTER_EMAIL=t@t")
		out, err := c.CombinedOutput()
		return string(out), err
	}
	if _, err := git("init", "-q", "-b", "main"); err != nil {
		if _, e2 := git("init", "-q"); e2 != nil {
			t.Skipf("git init failed: %v", e2)
		}
		_, _ = git("symbolic-ref", "HEAD", "refs/heads/main")
	}
	_, _ = git("config", "user.email", "t@t")
	_, _ = git("config", "user.name", "t")
	if err := os.WriteFile(filepath.Join(repo, "seed.txt"), []byte("seed\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := git("add", "seed.txt"); err != nil {
		t.Fatal(err)
	}
	if out, err := git("commit", "-qm", "seed"); err != nil {
		t.Skipf("seed commit failed: %s", out)
	}
	res := workerworktree.Prepare(repo, "docs", "probe", "", wtRoot, nil)
	if !res.OK {
		t.Fatalf("prepare probe worktree: %+v", res)
	}
	now = time.Now()
	floor = 30 * time.Minute
	back := now.Add(-2 * time.Hour)
	if err := os.Chtimes(res.Path, back, back); err != nil {
		t.Fatalf("age probe worktree: %v", err)
	}
	return repo, res.Path, now, floor
}

func TestWorktreeColdReapProcessActiveAtPlanningIsExcluded(t *testing.T) {
	repo, wt, now, floor := newColdReapProbeFixture(t)
	var reapCalls []string

	got := worktreeColdReapReportWithProbes(
		repo,
		true,
		floor,
		now,
		false,
		func(string) (bool, error) { return true, nil },
		func(_, path string) workerworktree.Result {
			reapCalls = append(reapCalls, path)
			return workerworktree.Result{OK: true, Path: path, Removed: true}
		},
	)

	if got.WouldReap != 0 || got.Reaped != 0 {
		t.Fatalf("active process must exclude worktree from the reap plan: %+v", got)
	}
	if len(reapCalls) != 0 {
		t.Fatalf("active process must not reach reap/unregister, calls=%v", reapCalls)
	}
	if len(got.Worktrees) != 1 || got.Worktrees[0].Eligible {
		t.Fatalf("active process worktree must be kept: %+v", got.Worktrees)
	}
	if !strings.Contains(got.Worktrees[0].Reason, "active process") {
		t.Fatalf("active process keep must carry its reason, got %q for %s", got.Worktrees[0].Reason, wt)
	}
}

func TestWorktreeColdReapProcessBecomesLiveAtApply(t *testing.T) {
	repo, _, now, floor := newColdReapProbeFixture(t)
	probeCalls := 0
	var reapCalls []string

	got := worktreeColdReapReportWithProbes(
		repo,
		true,
		floor,
		now,
		false,
		func(string) (bool, error) {
			probeCalls++
			return probeCalls >= 2, nil
		},
		func(_, path string) workerworktree.Result {
			reapCalls = append(reapCalls, path)
			return workerworktree.Result{OK: true, Path: path, Removed: true}
		},
	)

	if got.WouldReap != 1 || got.Reaped != 0 {
		t.Fatalf("apply-time liveness flip must refuse the planned reap: %+v", got)
	}
	if len(reapCalls) != 0 {
		t.Fatalf("apply-time live process must not reach reap/unregister, calls=%v", reapCalls)
	}
	if len(got.Failures) != 1 || got.Failures[0].Reason != "process_live" {
		t.Fatalf("want one process_live failure, got %+v", got.Failures)
	}
}

func TestWorktreeColdReapRemovedClaimRequiresAbsentDirectory(t *testing.T) {
	repo, wt, now, floor := newColdReapProbeFixture(t)
	reapCalls := 0

	got := worktreeColdReapReportWithProbes(
		repo,
		true,
		floor,
		now,
		false,
		func(string) (bool, error) { return false, nil },
		func(_, path string) workerworktree.Result {
			reapCalls++
			return workerworktree.Result{OK: true, Path: path, Removed: true}
		},
	)

	if got.Reaped != 0 || reapCalls != 1 {
		t.Fatalf("a removed claim with a remaining directory must not count as reaped: calls=%d out=%+v", reapCalls, got)
	}
	if len(got.Failures) != 1 || got.Failures[0].Reason != "directory_remains" {
		t.Fatalf("want one directory_remains failure, got %+v", got.Failures)
	}
	if _, err := os.Stat(wt); err != nil {
		t.Fatalf("fake reap must leave directory in place: %v", err)
	}
}

func TestWorktreeColdReapCountsVerifiedRemoval(t *testing.T) {
	repo, wt, now, floor := newColdReapProbeFixture(t)

	got := worktreeColdReapReportWithProbes(
		repo,
		true,
		floor,
		now,
		false,
		func(string) (bool, error) { return false, nil },
		func(root, path string) workerworktree.Result {
			return workerworktree.ForceReap(root, path, nil)
		},
	)

	if got.WouldReap != 1 || got.Reaped != 1 || len(got.Failures) != 0 {
		t.Fatalf("verified removal must count exactly one reap: %+v", got)
	}
	if _, err := os.Stat(wt); !os.IsNotExist(err) {
		t.Fatalf("verified reap must remove directory, stat err=%v", err)
	}
}

// TestGoBuildVerifyFailOpenWithoutToolchain documents the fail-open contract of
// the --verify go-build witness: it never blocks a land merely because a probe
// could not run. (When `go` IS present, as in CI, it actually builds; this asserts
// the no-toolchain branch is a clean pass, not a crash.)
func TestLandVerifyFlagParsesGoBuild(t *testing.T) {
	// The hook selector is exercised via the internal Land test with a fake hook;
	// here we only assert the CLI's go-build hook is a valid VerifyHook value.
	var _ workerworktree.VerifyHook = worktreeWorkerGoBuildVerify
}

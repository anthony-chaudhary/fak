package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"sort"
	"strings"
	"sync/atomic"
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
		Result:   workerworktree.Result{OK: true, Path: "/wt/fak-worker-wt-cmd-abc", BaseSHA: "feedface", Reused: false},
		Env:      workerworktree.WorktreeEnv(nil, "/wt/fak-worker-wt-cmd-abc"),
		Capacity: workerworktree.AssessCapacity(50, 51, true, "issue #10459 burst", nil),
	}
	got := mustKeys(t, out, "ok", "path", "base_sha", "env", "capacity")
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
	capacity := got["capacity"].(map[string]any)
	if capacity["status"] != string(workerworktree.CapacityGrowthJustified) || capacity["reason"] != "issue #10459 burst" || capacity["allowed"] != true {
		t.Fatalf("prepare capacity evidence = %v", capacity)
	}
}

// TestPrepareFailOpenJSONShape proves a failed prepare is still a well-formed
// object (ok=false, reason set) — never a crash — and omits env.
func TestPrepareFailOpenJSONShape(t *testing.T) {
	out := worktreePrepareOut{
		Result:   workerworktree.Result{OK: false, Reason: "could not resolve trunk HEAD — fail open"},
		Capacity: workerworktree.AssessCapacity(0, 0, false, "", nil),
	}
	got := mustKeys(t, out, "ok", "reason", "capacity")
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
	changedPaths := []string{"cmd/fak/worktree_worker.go", "internal/workerworktree/land.go"}
	patch := []byte("diff --git a/a b/a\n+patch bytes\n")
	res := workerworktree.Result{OK: true, Applied: true, Committed: true, Cost: &workerworktree.LandCostReceipt{
		Schema:          "fak-worker-land-cost/1",
		PatchScopeFiles: workerworktree.PatchScopeFiles(len(changedPaths)),
		PatchScopeBytes: workerworktree.PatchScopeBytes(len(patch)),
		CacheState:      "fresh-isolated-index",
		Phases:          []workerworktree.LandPhaseCost{},
	}}
	got := mustKeys(t, res, "ok", "applied", "committed", "cost")
	if got["committed"] != true {
		t.Fatalf("committed = %v, want true", got["committed"])
	}
	cost := got["cost"].(map[string]any)
	if cost["patch_scope_files"] != float64(len(changedPaths)) || cost["patch_scope_bytes"] != float64(len(patch)) {
		t.Fatalf("patch scope provenance = %v, want files=%d bytes=%d", cost, len(changedPaths), len(patch))
	}
	for _, ambiguous := range []string{"scanned_files", "scanned_bytes"} {
		if _, ok := cost[ambiguous]; ok {
			t.Fatalf("current cost receipt must not emit ambiguous key %q: %v", ambiguous, cost)
		}
	}
}

func TestLandProgressJSONShape(t *testing.T) {
	var out bytes.Buffer
	emit := worktreeWorkerProgressEmitter(&out)
	emit(workerworktree.LandProgressEvent{
		Schema: "fak-worker-land-progress/1", Phase: "admission", Status: "started", LandElapsedMS: 3,
	})
	var got map[string]any
	if err := json.Unmarshal(bytes.TrimSpace(out.Bytes()), &got); err != nil {
		t.Fatalf("decode progress: %v; output=%q", err, out.String())
	}
	if got["schema"] != "fak-worker-land-progress/1" || got["phase"] != "admission" || got["status"] != "started" {
		t.Fatalf("progress JSON = %v", got)
	}
}

func TestWorkerLandDisambiguationTimeoutFlagUsesResolverReceiptAndNoRetry(t *testing.T) {
	previous, hadPrevious := os.LookupEnv(workerworktree.DisambiguationTimeoutEnv)
	if err := os.Unsetenv(workerworktree.DisambiguationTimeoutEnv); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if hadPrevious {
			_ = os.Setenv(workerworktree.DisambiguationTimeoutEnv, previous)
		} else {
			_ = os.Unsetenv(workerworktree.DisambiguationTimeoutEnv)
		}
	})
	repo, worktree, base := newDisambiguationLandFixture(t)
	var events []workerworktree.LandProgressEvent
	res, err := withWorkerLandDisambiguationTimeout("1", true, func() workerworktree.Result {
		return workerworktree.Land(
			repo, worktree, base, "", []string{"cmd/fak/sample.go"}, nil, nil,
			workerworktree.WithLandProgress(func(event workerworktree.LandProgressEvent) {
				events = append(events, event)
			}),
		)
	})
	if err != nil {
		t.Fatalf("bridge timeout flag: %v", err)
	}
	if res.OK || res.Applied || res.Committed || res.Disambiguation == nil {
		t.Fatalf("1ms oracle timeout must be a typed pre-CAS refusal: %+v", res)
	}
	timeout := res.Disambiguation.Timeout
	if timeout.DefaultTimeoutMS != 120_000 || timeout.RequestedTimeoutMS == nil ||
		*timeout.RequestedTimeoutMS != 1 || timeout.EffectiveTimeoutMS != 1 ||
		timeout.RecoveryMode != "explicit_bounded_override" {
		t.Fatalf("CLI flag and resolver receipt diverged: %+v", timeout)
	}
	if diagnostic := res.Disambiguation.Before.Diagnostic; diagnostic == nil ||
		diagnostic.Code != workerworktree.DisambiguationTimeoutCode || diagnostic.TimeoutMS != 1 {
		t.Fatalf("failure-matched timeout diagnostic missing: %+v", res.Disambiguation.Before)
	}
	starts := 0
	for _, event := range events {
		if event.Phase == "whole-tree-disambiguation" && event.Status == "started" {
			starts++
			if event.Attempt != 1 {
				t.Fatalf("disambiguation ran on retry attempt %d: %+v", event.Attempt, events)
			}
		}
		if event.Phase == "cas-rebase" {
			t.Fatalf("timeout triggered a retry: %+v", events)
		}
	}
	if starts != 1 {
		t.Fatalf("whole-tree witness attempts=%d, want exactly one: %+v", starts, events)
	}
	if _, present := os.LookupEnv(workerworktree.DisambiguationTimeoutEnv); present {
		t.Fatalf("%s leaked after land", workerworktree.DisambiguationTimeoutEnv)
	}
}

func TestWorkerLandDisambiguationTimeoutFlagLeavesBoundsToSameResolver(t *testing.T) {
	repo, worktree, base := newDisambiguationLandFixture(t)
	res, err := withWorkerLandDisambiguationTimeout("900001", true, func() workerworktree.Result {
		return workerworktree.Land(repo, worktree, base, "", []string{"cmd/fak/sample.go"}, nil, nil)
	})
	if err != nil {
		t.Fatalf("bridge timeout flag: %v", err)
	}
	if res.OK || res.Disambiguation == nil || res.Disambiguation.Before.Diagnostic == nil {
		t.Fatalf("out-of-range timeout must be refused by the oracle resolver: %+v", res)
	}
	timeout := res.Disambiguation.Timeout
	if timeout.DefaultTimeoutMS != 120_000 || timeout.RequestedTimeoutMS == nil ||
		*timeout.RequestedTimeoutMS != 900001 || timeout.EffectiveTimeoutMS != 0 ||
		timeout.RecoveryMode != "invalid_explicit_override" {
		t.Fatalf("invalid CLI flag and resolver receipt diverged: %+v", timeout)
	}
	diagnostic := res.Disambiguation.Before.Diagnostic
	if diagnostic.Code != workerworktree.DisambiguationTimeoutCode ||
		diagnostic.Witness != "configuration" || diagnostic.Subphase != "timeout-config" {
		t.Fatalf("invalid timeout diagnostic = %+v", diagnostic)
	}
}

func newDisambiguationLandFixture(t *testing.T) (repo, worktree, base string) {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git unavailable")
	}
	root := t.TempDir()
	repo = filepath.Join(root, "repo")
	worktree = filepath.Join(root, "worker")
	if err := os.MkdirAll(filepath.Join(repo, "cmd", "fak"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(repo, "tools"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(repo, "cmd", "fak", "sample.go"), []byte("package sample\n\nconst value = 1\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	// A valid executable target that intentionally outlives the 1 ms accepted
	// timeout. CommandContext must cancel it; the 900001 ms case refuses before it.
	if err := os.WriteFile(filepath.Join(repo, "tools", "concept_disambiguation_scorecard.py"), []byte("import time\ntime.sleep(30)\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	git := func(dir string, args ...string) string {
		t.Helper()
		cmd := exec.Command("git", append([]string{"-C", dir}, args...)...)
		cmd.Env = append(os.Environ(),
			"GIT_AUTHOR_NAME=t", "GIT_AUTHOR_EMAIL=t@t",
			"GIT_COMMITTER_NAME=t", "GIT_COMMITTER_EMAIL=t@t")
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("git -C %s %s: %v: %s", dir, strings.Join(args, " "), err, out)
		}
		return strings.TrimSpace(string(out))
	}
	git(repo, "init", "-q", "-b", "main")
	git(repo, "config", "user.email", "t@t")
	git(repo, "config", "user.name", "t")
	// Keep the synthetic repository deterministic after the test returns. Git can
	// otherwise start background maintenance that races t.TempDir cleanup and leaves
	// .git non-empty after the resolver has already produced its typed receipt.
	git(repo, "config", "maintenance.auto", "false")
	git(repo, "config", "gc.auto", "0")
	git(repo, "config", "commit.gpgsign", "false")
	git(repo, "config", "core.hooksPath", "")
	git(repo, "add", ".")
	git(repo, "commit", "-qm", "base")
	base = git(repo, "rev-parse", "HEAD")
	git(repo, "worktree", "add", "--detach", worktree, base)
	if err := os.WriteFile(filepath.Join(worktree, "cmd", "fak", "sample.go"), []byte("package sample\n\nconst value = 2\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_ = exec.Command("git", "-C", repo, "worktree", "remove", "--force", worktree).Run()
	})
	return repo, worktree, base
}

// TestReapJSONShape proves `fak worktree worker reap` emits the removed verdict.
func TestReapJSONShape(t *testing.T) {
	res := workerworktree.Result{OK: true, Path: "/wt/fak-worker-wt-cmd-abc", Removed: true}
	mustKeys(t, res, "ok", "path", "removed")
}

func TestWorktreeWorkerReapCommandHelper(t *testing.T) {
	if os.Getenv("FAK_REAP_COMMAND_HELPER") != "1" {
		return
	}
	for i, arg := range os.Args {
		if arg == "--" {
			worktreeWorkerReap(os.Args[i+1:])
			return
		}
	}
	os.Exit(2)
}

type reapCommandResult struct {
	code    int
	stdout  string
	stderr  string
	elapsed time.Duration
}

func runReapCommand(t *testing.T, deadline time.Duration, env []string, args ...string) reapCommandResult {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), deadline)
	defer cancel()
	cmdArgs := append([]string{"-test.run=^TestWorktreeWorkerReapCommandHelper$", "--"}, args...)
	cmd := exec.CommandContext(ctx, os.Args[0], cmdArgs...)
	cmd.Env = append(os.Environ(), append([]string{
		"FAK_REAP_COMMAND_HELPER=1",
		workerworktree.PoolCapEnv + "=0",
	}, env...)...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	started := time.Now()
	err := cmd.Run()
	res := reapCommandResult{stdout: stdout.String(), stderr: stderr.String(), elapsed: time.Since(started)}
	if err == nil {
		return res
	}
	if ctx.Err() != nil {
		res.code = -1
		return res
	}
	if exitErr, ok := err.(*exec.ExitError); ok {
		res.code = exitErr.ExitCode()
		return res
	}
	t.Fatalf("run reap helper: %v", err)
	return reapCommandResult{}
}

func reapReceipt(t *testing.T, stdout string) map[string]any {
	t.Helper()
	for _, line := range strings.Split(stdout, "\n") {
		if !strings.HasPrefix(strings.TrimSpace(line), "{") {
			continue
		}
		var got map[string]any
		if err := json.Unmarshal([]byte(line), &got); err != nil {
			t.Fatalf("decode reap receipt %q: %v", line, err)
		}
		return got
	}
	t.Fatalf("missing reap JSON receipt; stdout=%q", stdout)
	return nil
}

func newSingleReapFixture(t *testing.T) (repo, worktree, base string) {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git unavailable")
	}
	t.Setenv(workerworktree.PoolCapEnv, "0")
	root := t.TempDir()
	repo = filepath.Join(root, "repo")
	if err := os.MkdirAll(repo, 0o755); err != nil {
		t.Fatal(err)
	}
	git := func(args ...string) string {
		t.Helper()
		cmd := exec.Command("git", append([]string{"-C", repo}, args...)...)
		cmd.Env = append(os.Environ(),
			"GIT_AUTHOR_NAME=t", "GIT_AUTHOR_EMAIL=t@t",
			"GIT_COMMITTER_NAME=t", "GIT_COMMITTER_EMAIL=t@t")
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("git %s: %v: %s", strings.Join(args, " "), err, out)
		}
		return strings.TrimSpace(string(out))
	}
	git("init", "-q", "-b", "main")
	git("config", "user.email", "t@t")
	git("config", "user.name", "t")
	if err := os.WriteFile(filepath.Join(repo, "owned.txt"), []byte("base\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	git("add", "owned.txt")
	git("commit", "-qm", "base")
	base = git("rev-parse", "HEAD")
	prepared := workerworktree.Prepare(repo, "cmd", "8503", base, filepath.Join(root, "workers"), nil)
	if !prepared.OK {
		t.Fatalf("prepare: %+v", prepared)
	}
	worktree = prepared.Path
	t.Cleanup(func() { _ = workerworktree.ForceReap(repo, worktree, nil) })
	return repo, worktree, base
}

func TestSingleReapDirtyWorktreeReturnsTypedRefusalAndPreservesWork(t *testing.T) {
	repo, worktree, _ := newSingleReapFixture(t)
	if err := os.WriteFile(filepath.Join(worktree, "owned.txt"), []byte("unlanded\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	res := runReapCommand(t, 3*time.Second, nil, "--root", repo, "--worktree", worktree, "--max-wait", "500ms")
	if res.code != 1 {
		t.Fatalf("dirty reap exit=%d stdout=%q stderr=%q", res.code, res.stdout, res.stderr)
	}
	got := reapReceipt(t, res.stdout)
	if got["code"] != "DIRTY_WORKTREE_REFUSED" || got["ok"] != false || got["preserved"] != true || got["removed"] == true {
		t.Fatalf("dirty reap receipt=%v", got)
	}
	if b, err := os.ReadFile(filepath.Join(worktree, "owned.txt")); err != nil || string(b) != "unlanded\n" {
		t.Fatalf("dirty work was not preserved: body=%q err=%v", b, err)
	}
}

func TestSingleReapExplicitSupersessionRemovesRegisteredDirtyWorktree(t *testing.T) {
	repo, worktree, _ := newSingleReapFixture(t)
	want := []byte("already landed\n")
	if err := os.WriteFile(filepath.Join(worktree, "owned.txt"), want, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(repo, "owned.txt"), want, 0o644); err != nil {
		t.Fatal(err)
	}
	git := func(args ...string) string {
		t.Helper()
		cmd := exec.Command("git", append([]string{"-C", repo}, args...)...)
		cmd.Env = append(os.Environ(),
			"GIT_AUTHOR_NAME=t", "GIT_AUTHOR_EMAIL=t@t",
			"GIT_COMMITTER_NAME=t", "GIT_COMMITTER_EMAIL=t@t")
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("git %s: %v: %s", strings.Join(args, " "), err, out)
		}
		return strings.TrimSpace(string(out))
	}
	git("add", "owned.txt")
	git("commit", "-qm", "landed independently")
	supersededBy := git("rev-parse", "HEAD")
	res := runReapCommand(t, 8*time.Second, nil, "--root", repo, "--worktree", worktree, "--superseded-by", supersededBy, "--max-wait", "5s")
	if res.code != 0 {
		t.Fatalf("authorized reap exit=%d stdout=%q stderr=%q", res.code, res.stdout, res.stderr)
	}
	got := reapReceipt(t, res.stdout)
	if got["code"] != "VERIFIED_WORKTREE_REAPED" || got["removed"] != true {
		t.Fatalf("authorized reap receipt=%v", got)
	}
	if _, err := os.Stat(worktree); !os.IsNotExist(err) {
		t.Fatalf("authorized reap left directory: %v", err)
	}
	if listed := git("worktree", "list", "--porcelain"); strings.Contains(filepath.ToSlash(listed), filepath.ToSlash(worktree)) {
		t.Fatalf("authorized reap left git registration: %s", listed)
	}
}

func TestSingleReapStatusStallReturnsWithinBound(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("the fake git shim is a POSIX script; the acceptance gate runs under WSL")
	}
	repo, worktree, _ := newSingleReapFixture(t)
	if err := os.WriteFile(filepath.Join(worktree, "owned.txt"), []byte("unlanded\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	realGit, err := exec.LookPath("git")
	if err != nil {
		t.Fatal(err)
	}
	shimDir := t.TempDir()
	shim := filepath.Join(shimDir, "git")
	body := "#!/bin/sh\nif [ \"$PWD\" = \"$FAK_REAP_STALL_DIR\" ] && [ \"$1\" = status ]; then sleep 30; fi\nexec \"$FAK_REAP_REAL_GIT\" \"$@\"\n"
	if err := os.WriteFile(shim, []byte(body), 0o755); err != nil {
		t.Fatal(err)
	}
	env := []string{
		"PATH=" + shimDir + string(os.PathListSeparator) + os.Getenv("PATH"),
		"FAK_REAP_STALL_DIR=" + worktree,
		"FAK_REAP_REAL_GIT=" + realGit,
	}
	res := runReapCommand(t, 3*time.Second, env, "--root", repo, "--worktree", worktree, "--max-wait", "200ms")
	if res.code != 1 || res.elapsed > 2*time.Second {
		t.Fatalf("stalled reap was not bounded: exit=%d elapsed=%s stdout=%q stderr=%q", res.code, res.elapsed, res.stdout, res.stderr)
	}
	if !strings.Contains(res.stderr, "REAP_PROGRESS code=REAP_STARTED") {
		t.Fatalf("stalled reap emitted no progress before refusal: %q", res.stderr)
	}
	got := reapReceipt(t, res.stdout)
	if got["code"] != "REAP_TIMEOUT" || got["ok"] != false || got["preserved"] != true {
		t.Fatalf("stalled reap receipt=%v", got)
	}
	if _, err := os.Stat(worktree); err != nil {
		t.Fatalf("timed-out reap removed worktree: %v", err)
	}
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

// TestListJSONShape pins both list contracts: each includes the same deterministic
// capacity evidence, while --json adds the versioned typed lifecycle inventory.
// Empty collections are [] (never null) in both forms.
func TestListJSONShape(t *testing.T) {
	capacity := workerworktree.AssessCapacity(0, 0, true, "", nil)
	legacy := worktreeWorkerListOut{Count: 0, Paths: []string{}, Inventory: []workerworktree.InventoryRow{}, Capacity: capacity}
	b, err := json.Marshal(legacy)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	wantLegacy := `{"count":0,"paths":[],"inventory":[],"capacity":{"schema":"fak-worker-worktree-capacity/1","setpoint":50,"current_count":0,"prospective_count":0,"inventory_known":true,"above_setpoint":false,"allowed":true,"status":"WITHIN_SETPOINT","reason":"","message":"worker worktree capacity 0 is at or below advisory setpoint 50","contraction_recommendations":[]}}`
	if string(b) != wantLegacy {
		t.Fatalf("legacy list json = %s", b)
	}
	got := mustKeys(t, worktreeWorkerListOut{Count: 2, Paths: []string{"/a", "/b"}, Inventory: []workerworktree.InventoryRow{}, Capacity: capacity}, "count", "paths", "inventory", "capacity")
	if got["count"].(float64) != 2 {
		t.Fatalf("count = %v, want 2", got["count"])
	}

	typed := worktreeWorkerLifecycleOut{
		Schema:    worktreeWorkerLifecycleSchema,
		Count:     0,
		Paths:     []string{},
		Inventory: []worktreeWorkerLifecycleRow{},
		Capacity:  capacity,
	}
	b, err = json.Marshal(typed)
	if err != nil {
		t.Fatalf("marshal typed: %v", err)
	}
	wantTyped := `{"schema":"fak-worker-worktree-lifecycle/1","count":0,"paths":[],"inventory":[],"capacity":{"schema":"fak-worker-worktree-capacity/1","setpoint":50,"current_count":0,"prospective_count":0,"inventory_known":true,"above_setpoint":false,"allowed":true,"status":"WITHIN_SETPOINT","reason":"","message":"worker worktree capacity 0 is at or below advisory setpoint 50","contraction_recommendations":[]}}`
	if string(b) != wantTyped {
		t.Fatalf("typed list json = %s", b)
	}
}

func TestWorktreeWorkerListJSONCommandUsesRegisteredEvidence(t *testing.T) {
	repo, worktree, base := newSingleReapFixture(t)
	read, write, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	oldStdout := os.Stdout
	os.Stdout = write
	worktreeWorkerList([]string{"--root", repo, "--json"})
	os.Stdout = oldStdout
	if err := write.Close(); err != nil {
		t.Fatal(err)
	}
	raw, err := io.ReadAll(read)
	if err != nil {
		t.Fatal(err)
	}
	if err := read.Close(); err != nil {
		t.Fatal(err)
	}

	var got worktreeWorkerLifecycleOut
	if err := json.Unmarshal(bytes.TrimSpace(raw), &got); err != nil {
		t.Fatalf("decode list --json: %v; output=%q", err, raw)
	}
	if got.Schema != worktreeWorkerLifecycleSchema || got.Count != 1 ||
		len(got.Paths) != 1 || !sameResolvedPath(got.Paths[0], worktree) || len(got.Inventory) != 1 {
		t.Fatalf("list --json header = %+v", got)
	}
	if !got.Capacity.Allowed || !got.Capacity.InventoryKnown || got.Capacity.CurrentCount != 1 || got.Capacity.Status != workerworktree.CapacityWithinSetpoint {
		t.Fatalf("list --json capacity = %+v", got.Capacity)
	}
	row := got.Inventory[0]
	if !sameResolvedPath(row.Path, worktree) || row.HeadSHA != base || row.BaseSHA != base {
		t.Fatalf("revision association = %+v, want path=%q head/base=%q", row, worktree, base)
	}
	if row.Association.State != worktreeEvidenceAssociated ||
		row.Association.OwnerPID != os.Getpid() ||
		row.Association.LeaseID != "resolve-cmd" ||
		row.Association.Lane != "cmd" {
		t.Fatalf("owner association = %+v", row.Association)
	}
	if row.Liveness.Owner != worktreeEvidenceLive ||
		row.Liveness.Lease != worktreeEvidenceReleased ||
		row.Cleanliness.State != worktreeEvidenceClean ||
		row.Lifecycle != worktreeLifecycleReady ||
		row.ReapReadiness.Reapable ||
		row.ReapReadiness.Reason != "OWNER_LIVE" {
		t.Fatalf("live clean worktree lifecycle = %+v", row)
	}
}

func TestWorktreeWorkerCapacityEvidenceOnlyRecommendsCleanSafeTrees(t *testing.T) {
	rows := []worktreeWorkerLifecycleRow{
		{
			Path: "/wt/cold", Liveness: worktreeWorkerLiveness{Owner: worktreeEvidenceDead, Lease: worktreeEvidenceReleased},
			Cleanliness: worktreeWorkerCleanliness{State: worktreeEvidenceClean, DirtyPaths: []string{}},
			Lifecycle:   worktreeLifecycleCold, ReapReadiness: worktreeWorkerReapReadiness{Reapable: true, Verdict: worktreeReapable, Reason: "COLD_CLEAN"},
		},
		{
			Path: "/wt/owner-dead-clean", Liveness: worktreeWorkerLiveness{Owner: worktreeEvidenceDead, Lease: worktreeEvidenceLive},
			Cleanliness: worktreeWorkerCleanliness{State: worktreeEvidenceClean, DirtyPaths: []string{}},
			Lifecycle:   worktreeLifecycleRetained, ReapReadiness: worktreeWorkerReapReadiness{Verdict: worktreeKeep, Reason: "LEASE_LIVE"},
		},
		{
			Path: "/wt/dirty", Liveness: worktreeWorkerLiveness{Owner: worktreeEvidenceDead, Lease: worktreeEvidenceReleased},
			Cleanliness: worktreeWorkerCleanliness{State: worktreeEvidenceDirty, DirtyPaths: []string{"do-not-delete.go"}},
			Lifecycle:   worktreeLifecycleDirty, ReapReadiness: worktreeWorkerReapReadiness{Verdict: worktreeKeep, Reason: "WORKTREE_DIRTY"},
		},
	}
	paths := make([]string, 51)
	advisory := worktreeWorkerCapacityAdvisory("/repo", workerworktree.CapacityCensus{Known: true, Paths: paths}, 51, "", rows)
	if advisory.Status != workerworktree.CapacityJustificationNeeded || len(advisory.ContractionRecommendations) != 2 {
		t.Fatalf("capacity advisory = %+v", advisory)
	}
	for _, recommendation := range advisory.ContractionRecommendations {
		if recommendation.Path == "/wt/dirty" || strings.Contains(recommendation.Action, "--apply") || strings.Contains(recommendation.Action, "--even-if-unlanded") {
			t.Fatalf("dirty/destructive contraction recommendation = %+v", recommendation)
		}
	}

	var human bytes.Buffer
	worktreeWorkerWriteCapacityHuman(&human, advisory)
	want := "capacity advisory: worker worktree capacity 51 exceeds advisory setpoint 50; provide --capacity-reason to justify growth; the operation remains allowed\n" +
		"capacity contraction: candidate=/wt/cold basis=COLD_REAPABLE; inspect with: fak worktree worker reap --all-cold\n" +
		"capacity contraction: candidate=/wt/owner-dead-clean basis=OWNER_DEAD_CLEAN; inspect with: fak worktree worker gc --dry-run\n"
	if human.String() != want {
		t.Fatalf("human capacity output = %q, want %q", human.String(), want)
	}
}

func TestWorktreeWorkerCapacityUnknownInventoryWarnsAndFailsOpen(t *testing.T) {
	advisory := worktreeWorkerCapacityAdvisory("/repo", workerworktree.CapacityCensus{Known: false, Paths: []string{}}, 0, "", nil)
	var human bytes.Buffer
	worktreeWorkerWriteCapacityHuman(&human, advisory)
	if !advisory.Allowed || advisory.Status != workerworktree.CapacityInventoryUnknown || !strings.Contains(human.String(), "inventory is unknown") || !strings.Contains(human.String(), "failed open") {
		t.Fatalf("unknown inventory advisory=%+v human=%q", advisory, human.String())
	}
}

func TestWorktreeWorkerLifecycleRowJoinsPresentMissingAndInvalidIntent(t *testing.T) {
	root := t.TempDir()
	present := filepath.Join(root, "present")
	missing := filepath.Join(root, "missing")
	invalid := filepath.Join(root, "invalid")
	for _, path := range []string{present, missing, invalid} {
		if err := os.MkdirAll(path, 0o755); err != nil {
			t.Fatal(err)
		}
	}

	message := "feat(worktree): expose intent (#10528) (fak workerworktree)"
	if err := workerworktree.SaveIntent(present, "base-present", message, []string{"z.txt", "a.txt"}); err != nil {
		t.Fatal(err)
	}
	intentDir := filepath.Join(filepath.Dir(invalid), ".fak-worker-intents")
	if err := os.MkdirAll(intentDir, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(intentDir, filepath.Base(invalid)+".json"), []byte("{not-json\n"), 0o600); err != nil {
		t.Fatal(err)
	}

	probes := worktreeWorkerLifecycleProbes{
		ReadOwner:    func(string) (workerworktree.OwnerStamp, error) { return workerworktree.OwnerStamp{}, os.ErrNotExist },
		ProcessAlive: func(int) (bool, error) { return false, nil },
		LeaseLive:    func(string) (bool, error) { return false, nil },
		Inspect: func(_ string, path string) (worktreeWorkerRevisionEvidence, error) {
			return worktreeWorkerRevisionEvidence{HeadSHA: "head", BaseSHA: "base", Cleanliness: worktreeEvidenceClean}, nil
		},
	}

	presentRow := worktreeWorkerLifecycleRowForPath(root, present, probes)
	if presentRow.Intent.Status != worktreeWorkerIntentPresent || presentRow.Intent.IssueNumber != 10528 || presentRow.Intent.Message != message || !reflect.DeepEqual(presentRow.Intent.Paths, []string{"a.txt", "z.txt"}) {
		t.Fatalf("present intent = %+v", presentRow.Intent)
	}
	if presentRow.Lifecycle == "" || presentRow.ReapReadiness.Reason == "" || presentRow.Association.State == "" || presentRow.Cleanliness.State == "" {
		t.Fatalf("present row lost lifecycle evidence: %+v", presentRow)
	}

	missingRow := worktreeWorkerLifecycleRowForPath(root, missing, probes)
	if missingRow.Intent.Status != worktreeWorkerIntentMissing || missingRow.Intent.Diagnostic == "" {
		t.Fatalf("missing intent = %+v", missingRow.Intent)
	}
	if missingRow.Lifecycle == "" || missingRow.ReapReadiness.Reason == "" {
		t.Fatalf("missing intent suppressed lifecycle evidence: %+v", missingRow)
	}

	invalidRow := worktreeWorkerLifecycleRowForPath(root, invalid, probes)
	if invalidRow.Intent.Status != worktreeWorkerIntentInvalid || !strings.Contains(invalidRow.Intent.Diagnostic, "decode worker intent") {
		t.Fatalf("invalid intent = %+v", invalidRow.Intent)
	}
	if invalidRow.Lifecycle == "" || invalidRow.ReapReadiness.Reason == "" {
		t.Fatalf("invalid intent suppressed lifecycle evidence: %+v", invalidRow)
	}
}

func TestWorktreeWorkerSnapshotRowsDoNotExportLocalIntent(t *testing.T) {
	row := worktreeWorkerLifecycleRow{
		Path: "/private/local/worktree",
		Intent: worktreeWorkerIntent{
			Status:      worktreeWorkerIntentPresent,
			IssueNumber: 10528,
			Message:     "secret local subject",
			Paths:       []string{"private/owned.txt"},
		},
		Association: worktreeWorkerAssociation{State: worktreeEvidenceAssociated, Lane: "lane-a"},
		Liveness:    worktreeWorkerLiveness{Owner: worktreeEvidenceLive, Lease: worktreeEvidenceLive},
		Cleanliness: worktreeWorkerCleanliness{State: worktreeEvidenceDirty, DirtyPaths: []string{"private/dirty.txt"}},
		Lifecycle:   worktreeLifecycleRetained,
	}
	b, err := json.Marshal(worktreeWorkerSnapshotRows([]worktreeWorkerLifecycleRow{row}))
	if err != nil {
		t.Fatal(err)
	}
	text := string(b)
	for _, secret := range []string{row.Path, row.Intent.Message, row.Intent.Paths[0], row.Cleanliness.DirtyPaths[0], "issue_number", "intent"} {
		if strings.Contains(text, secret) {
			t.Fatalf("remote snapshot leaked %q: %s", secret, text)
		}
	}
}

func TestWorktreeWorkerLifecycleInventoryDistinguishesSafeStates(t *testing.T) {
	now := time.Date(2026, 8, 27, 12, 0, 0, 0, time.UTC)
	root := t.TempDir()
	path := func(lane, suffix string) string {
		return filepath.Join(root, "fak-worker-wt-"+lane+"-"+suffix)
	}
	ready := path("ready", "aaaaaaaaaaaa")
	retained := path("retained", "bbbbbbbbbbbb")
	cold := path("cold", "cccccccccccc")
	dirty := path("dirty", "dddddddddddd")
	malformed := path("malformed", "eeeeeeeeeeee")

	stamps := map[string]workerworktree.OwnerStamp{
		ready:    {Schema: worktreeOwnerStampSchema, PID: 101, LeaseID: "resolve-ready", CreatedAt: now},
		retained: {Schema: worktreeOwnerStampSchema, PID: 202, LeaseID: "resolve-retained", CreatedAt: now},
		cold:     {Schema: worktreeOwnerStampSchema, PID: 303, LeaseID: "resolve-cold", CreatedAt: now},
		dirty:    {Schema: worktreeOwnerStampSchema, PID: 404, LeaseID: "resolve-dirty", CreatedAt: now},
	}
	ownerLive := map[int]bool{101: true}
	leaseLive := map[string]bool{"resolve-ready": true, "resolve-retained": true}
	inspections := map[string]worktreeWorkerRevisionEvidence{
		ready:    {HeadSHA: "head-ready", BaseSHA: "base-ready", Cleanliness: worktreeEvidenceClean, DirtyPaths: []string{}},
		retained: {HeadSHA: "head-retained", BaseSHA: "base-retained", Cleanliness: worktreeEvidenceClean, DirtyPaths: []string{}},
		cold:     {HeadSHA: "head-cold", BaseSHA: "base-cold", Cleanliness: worktreeEvidenceClean, DirtyPaths: []string{}},
		dirty:    {HeadSHA: "head-dirty", BaseSHA: "base-dirty", Cleanliness: worktreeEvidenceDirty, DirtyPaths: []string{"z.go", "a.go"}},
		malformed: {
			HeadSHA: "head-malformed", BaseSHA: "base-malformed",
			Cleanliness: worktreeEvidenceClean, DirtyPaths: []string{},
		},
	}
	paths := []string{malformed, dirty, cold, retained, ready}
	rows := worktreeWorkerLifecycleInventory(root,
		paths,
		worktreeWorkerLifecycleProbes{
			ReadOwner: func(path string) (workerworktree.OwnerStamp, error) {
				if path == malformed {
					return workerworktree.OwnerStamp{}, os.ErrInvalid
				}
				return stamps[path], nil
			},
			ProcessAlive: func(pid int) (bool, error) { return ownerLive[pid], nil },
			LeaseLive:    func(id string) (bool, error) { return leaseLive[id], nil },
			Inspect: func(_, path string) (worktreeWorkerRevisionEvidence, error) {
				return inspections[path], nil
			},
		})
	if len(rows) != len(paths) {
		t.Fatalf("rows = %d, want one per path %d: %+v", len(rows), len(paths), rows)
	}
	byPath := map[string]worktreeWorkerLifecycleRow{}
	for _, row := range rows {
		byPath[row.Path] = row
	}
	assert := func(path string, lifecycle worktreeWorkerLifecycleState, reapable bool, reason string) {
		t.Helper()
		row := byPath[path]
		if row.Lifecycle != lifecycle || row.ReapReadiness.Reapable != reapable ||
			row.ReapReadiness.Reason != reason {
			t.Fatalf("%s row = %+v, want lifecycle=%s reapable=%v reason=%s",
				filepath.Base(path), row, lifecycle, reapable, reason)
		}
	}
	assert(ready, worktreeLifecycleReady, false, "OWNER_LIVE")
	assert(retained, worktreeLifecycleRetained, false, "LEASE_LIVE")
	assert(cold, worktreeLifecycleCold, true, "COLD_CLEAN")
	assert(dirty, worktreeLifecycleDirty, false, "WORKTREE_DIRTY")
	assert(malformed, worktreeLifecycleUnknown, false, "ASSOCIATION_UNKNOWN")

	if got := byPath[ready].Association; got.State != worktreeEvidenceAssociated ||
		got.OwnerPID != 101 || got.LeaseID != "resolve-ready" || got.Lane != "ready" {
		t.Fatalf("ready association = %+v", got)
	}
	if got := byPath[ready].Liveness; got.Owner != worktreeEvidenceLive || got.Lease != worktreeEvidenceLive {
		t.Fatalf("ready liveness = %+v", got)
	}
	if got := byPath[retained].Liveness; got.Owner != worktreeEvidenceDead || got.Lease != worktreeEvidenceLive {
		t.Fatalf("retained liveness = %+v", got)
	}
	if got := byPath[cold].Liveness; got.Owner != worktreeEvidenceDead || got.Lease != worktreeEvidenceReleased {
		t.Fatalf("cold liveness = %+v", got)
	}
	if got := byPath[dirty].Cleanliness; got.State != worktreeEvidenceDirty ||
		!reflect.DeepEqual(got.DirtyPaths, []string{"a.go", "z.go"}) {
		t.Fatalf("dirty cleanliness = %+v", got)
	}
	if got := byPath[malformed]; got.Association.State != worktreeEvidenceUnknown ||
		got.ReapReadiness.Verdict != worktreeKeep {
		t.Fatalf("malformed evidence overclaimed: %+v", got)
	}
	for i := 1; i < len(rows); i++ {
		if strings.ToLower(filepath.ToSlash(rows[i-1].Path)) > strings.ToLower(filepath.ToSlash(rows[i].Path)) {
			t.Fatalf("rows are not deterministically sorted: %q before %q", rows[i-1].Path, rows[i].Path)
		}
	}
}

func TestWorktreeWorkerLifecycleUnknownProbeIsNeverReapable(t *testing.T) {
	path := filepath.Join(t.TempDir(), "fak-worker-wt-cmd-ffffffffffff")
	rows := worktreeWorkerLifecycleInventory("repo", []string{path}, worktreeWorkerLifecycleProbes{
		ReadOwner: func(string) (workerworktree.OwnerStamp, error) {
			return workerworktree.OwnerStamp{
				Schema: worktreeOwnerStampSchema, PID: 42, LeaseID: "resolve-cmd",
				CreatedAt: time.Date(2026, 8, 27, 12, 0, 0, 0, time.UTC),
			}, nil
		},
		ProcessAlive: func(int) (bool, error) { return false, nil },
		LeaseLive:    func(string) (bool, error) { return false, os.ErrPermission },
		Inspect: func(_, _ string) (worktreeWorkerRevisionEvidence, error) {
			return worktreeWorkerRevisionEvidence{
				HeadSHA: "head", BaseSHA: "base", Cleanliness: worktreeEvidenceClean, DirtyPaths: []string{},
			}, nil
		},
	})
	if len(rows) != 1 {
		t.Fatalf("rows = %+v", rows)
	}
	row := rows[0]
	if row.Lifecycle != worktreeLifecycleUnknown || row.ReapReadiness.Reapable ||
		row.ReapReadiness.Reason != "LEASE_LIVENESS_UNKNOWN" {
		t.Fatalf("unknown lease evidence must be kept: %+v", row)
	}
}

func TestWorktreeWorkerLifecycleInventoryIsBoundedParallelAndDeterministic(t *testing.T) {
	const pathCount = 205
	const probeDelay = 10 * time.Millisecond

	root := t.TempDir()
	paths := make([]string, pathCount)
	for i := range paths {
		// Reverse input order so stable output cannot be an artifact of dispatch order.
		pathIndex := pathCount - 1 - i
		paths[i] = filepath.Join(root, fmt.Sprintf("fak-worker-wt-lane-%03d-%012x", pathIndex, pathIndex))
	}
	errorPath := paths[pathCount/2]

	var active atomic.Int32
	var maxActive atomic.Int32
	probes := worktreeWorkerLifecycleProbes{
		ReadOwner: func(path string) (workerworktree.OwnerStamp, error) {
			return workerworktree.OwnerStamp{
				Schema: worktreeOwnerStampSchema, PID: 42, LeaseID: "resolve-" + workerworktree.LaneOf(path),
				CreatedAt: time.Date(2026, 8, 30, 12, 0, 0, 0, time.UTC),
			}, nil
		},
		ProcessAlive: func(int) (bool, error) { return false, nil },
		LeaseLive:    func(string) (bool, error) { return false, nil },
		Inspect: func(_, path string) (worktreeWorkerRevisionEvidence, error) {
			now := active.Add(1)
			for old := maxActive.Load(); now > old && !maxActive.CompareAndSwap(old, now); old = maxActive.Load() {
			}
			time.Sleep(probeDelay)
			active.Add(-1)
			if path == errorPath {
				return worktreeWorkerRevisionEvidence{}, os.ErrPermission
			}
			return worktreeWorkerRevisionEvidence{
				HeadSHA: "head-" + filepath.Base(path), BaseSHA: "base",
				Cleanliness: worktreeEvidenceClean, DirtyPaths: []string{},
			}, nil
		},
	}

	started := time.Now()
	rows := worktreeWorkerLifecycleInventory(root, paths, probes)
	elapsed := time.Since(started)
	serialDuration := pathCount * probeDelay
	if elapsed >= serialDuration/2 {
		t.Fatalf("inventory elapsed=%s, want actual parallel speedup below %s (serial=%s)", elapsed, serialDuration/2, serialDuration)
	}
	if got := maxActive.Load(); got != worktreeWorkerLifecycleConcurrency {
		t.Fatalf("max concurrent probes=%d, want fixed ceiling %d", got, worktreeWorkerLifecycleConcurrency)
	}
	if len(rows) != pathCount {
		t.Fatalf("rows=%d, want %d", len(rows), pathCount)
	}

	wantPaths := append([]string(nil), paths...)
	sort.Slice(wantPaths, func(i, j int) bool {
		left := strings.ToLower(filepath.ToSlash(wantPaths[i]))
		right := strings.ToLower(filepath.ToSlash(wantPaths[j]))
		if left == right {
			return wantPaths[i] < wantPaths[j]
		}
		return left < right
	})
	for i, row := range rows {
		if row.Path != wantPaths[i] {
			t.Fatalf("row[%d].path=%q, want deterministic order %q", i, row.Path, wantPaths[i])
		}
		if row.Path == errorPath {
			if row.Lifecycle != worktreeLifecycleUnknown || row.ReapReadiness.Reapable ||
				row.ReapReadiness.Reason != "REVISION_UNKNOWN" {
				t.Fatalf("failed probe must stay attributed and fail closed: %+v", row)
			}
			continue
		}
		if row.Lifecycle != worktreeLifecycleCold || !row.ReapReadiness.Reapable ||
			row.ReapReadiness.Reason != "COLD_CLEAN" {
			t.Fatalf("healthy row %q inherited another path's error: %+v", row.Path, row)
		}
	}
}

func TestReadWorktreeWorkerOwnerRejectsMalformedEvidence(t *testing.T) {
	path := filepath.Join(t.TempDir(), "fak-worker-wt-cmd-123456789abc")
	stampPath := workerworktree.OwnerStampPath(path)
	if err := os.MkdirAll(filepath.Dir(stampPath), 0o755); err != nil {
		t.Fatal(err)
	}
	for _, raw := range []string{
		`{`,
		`{"schema":"unknown","pid":1,"lease_id":"resolve-cmd","created_at":"2026-08-27T12:00:00Z"}`,
		`{"schema":"fak-worker-worktree-owner/1","pid":1,"lease_id":"","created_at":"2026-08-27T12:00:00Z"}`,
	} {
		if err := os.WriteFile(stampPath, []byte(raw), 0o600); err != nil {
			t.Fatal(err)
		}
		if stamp, err := readWorktreeWorkerOwner(path); err == nil {
			t.Fatalf("malformed owner stamp accepted: %+v from %s", stamp, raw)
		}
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
	norm := func(p string) string {
		if resolved, err := filepath.EvalSymlinks(p); err == nil {
			p = resolved
		}
		return filepath.ToSlash(filepath.Clean(p))
	}
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
		func(paths []string) (map[string]bool, error) { return map[string]bool{paths[0]: true}, nil },
		func(string) (bool, error) { return false, nil },
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
		func([]string) (map[string]bool, error) {
			probeCalls++
			return map[string]bool{}, nil
		},
		func(string) (bool, error) {
			probeCalls++
			return true, nil
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
		func([]string) (map[string]bool, error) { return map[string]bool{}, nil },
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
		func([]string) (map[string]bool, error) { return map[string]bool{}, nil },
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

func TestWorktreeColdReapSnapshotsProcessesOnceForLargePlan(t *testing.T) {
	const worktreeCount = 200
	plan := make([]workerworktree.ColdWorktree, 0, worktreeCount)
	for i := 0; i < worktreeCount; i++ {
		plan = append(plan, workerworktree.ColdWorktree{Path: fmt.Sprintf("worker-%03d", i), Eligible: true})
	}
	calls := 0
	got, err := batchColdProcessRefs(plan, false, func(paths []string) (map[string]bool, error) {
		calls++
		if len(paths) != worktreeCount {
			t.Fatalf("snapshot paths=%d want=%d", len(paths), worktreeCount)
		}
		return map[string]bool{}, nil
	})
	if err != nil || calls != 1 || len(got) != 0 {
		t.Fatalf("snapshot calls=%d result=%v err=%v", calls, got, err)
	}
}

func TestWorktreeColdReapSkipsProcessSnapshotWithoutCandidates(t *testing.T) {
	calls := 0
	got, err := batchColdProcessRefs(
		[]workerworktree.ColdWorktree{{Path: "dirty", Eligible: false, Reason: "kept: dirty"}},
		false,
		func([]string) (map[string]bool, error) {
			calls++
			return nil, nil
		},
	)
	if err != nil || calls != 0 || len(got) != 0 {
		t.Fatalf("snapshot calls=%d result=%v err=%v", calls, got, err)
	}
}

func TestLandVerifyFlagParsesGoBuild(t *testing.T) {
	// The hook selector is exercised via the internal Land test with a fake hook;
	// here we only assert the CLI's go-build hook is a valid VerifyHook value.
	var _ workerworktree.VerifyHook = worktreeWorkerGoBuildVerify
}

func TestGoBuildVerifyRecreatesMissingBuildDirectories(t *testing.T) {
	if _, err := exec.LookPath("go"); err != nil {
		t.Skip("go toolchain is not installed")
	}
	wt := t.TempDir()
	if err := os.WriteFile(filepath.Join(wt, "go.mod"), []byte("module example.com/verify\n\ngo 1.26\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(wt, "verify.go"), []byte("package verify\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if ok, detail := worktreeWorkerGoBuildVerify(wt); !ok {
		t.Fatalf("go-build verify failed after healing absent build directories: %s", detail)
	}
	for _, name := range []string{".gocache", ".gotmp"} {
		if info, err := os.Stat(filepath.Join(wt, name)); err != nil || !info.IsDir() {
			t.Fatalf("%s was not recreated: info=%v err=%v", name, info, err)
		}
	}
}

func TestGoBuildVerifyFailsClosedWhenBuildDirectoryCannotBeCreated(t *testing.T) {
	if _, err := exec.LookPath("go"); err != nil {
		t.Skip("go toolchain is not installed")
	}
	wt := t.TempDir()
	gotmp := filepath.Join(wt, ".gotmp")
	if err := os.WriteFile(gotmp, []byte("not a directory"), 0o644); err != nil {
		t.Fatal(err)
	}
	ok, detail := worktreeWorkerGoBuildVerify(wt)
	if ok {
		t.Fatal("go-build verify passed despite an unusable GOTMPDIR")
	}
	if !strings.Contains(detail, "prepare isolated Go build directories") ||
		!strings.Contains(detail, "create GOTMPDIR directory") || !strings.Contains(detail, gotmp) {
		t.Fatalf("failure detail is not actionable: %q", detail)
	}
}
func TestWorktreeColdReapKnownAndUnknownBytes(t *testing.T) {
	tests := []struct {
		name        string
		size        func(string) (int64, error)
		wantBytes   int64
		wantKnown   int
		wantUnknown int
	}{
		{name: "measured empty", size: func(string) (int64, error) { return 0, nil }, wantKnown: 1},
		{name: "measurement failed", size: func(string) (int64, error) { return 0, fmt.Errorf("census incomplete") }, wantUnknown: 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			repo, _, now, floor := newColdReapProbeFixture(t)
			got := worktreeColdReapReportWithOptionsAndProbes(
				repo, false, floor, now, false,
				workerworktree.ColdReapOptions{Concurrency: 1, Size: tt.size},
				func([]string) (map[string]bool, error) { return map[string]bool{}, nil },
				func(string) (bool, error) { return false, nil },
				func(_, path string) workerworktree.Result {
					t.Fatalf("dry-run reached reap for %s", path)
					return workerworktree.Result{}
				},
			)
			if got.WouldReap != 1 || len(got.Worktrees) != 1 {
				t.Fatalf("dry-run classification changed: %+v", got)
			}
			item := got.Worktrees[0]
			if got.EligibleBytes != tt.wantBytes || got.Bytes != tt.wantBytes || got.EligibleBytesKnown != tt.wantKnown || got.EligibleBytesUnknown != tt.wantUnknown {
				t.Fatalf("aggregate byte provenance=%+v", got)
			}
			if item.ReclaimBytes != tt.wantBytes || item.Bytes != tt.wantBytes || item.ReclaimBytesKnown != (tt.wantKnown == 1) || item.BytesKnown != (tt.wantKnown == 1) {
				t.Fatalf("item byte provenance=%+v", item)
			}
			encoded := mustKeys(t, got,
				"bytes", "eligible_bytes", "eligible_bytes_known", "eligible_bytes_unknown", "worktrees",
			)
			rows, ok := encoded["worktrees"].([]any)
			if !ok || len(rows) != 1 {
				t.Fatalf("encoded worktrees=%T %#v", encoded["worktrees"], encoded["worktrees"])
			}
			row, ok := rows[0].(map[string]any)
			if !ok {
				t.Fatalf("encoded worktree=%T %#v", rows[0], rows[0])
			}
			for _, key := range []string{"bytes", "bytes_known", "reclaim_bytes", "reclaim_bytes_known"} {
				if _, ok := row[key]; !ok {
					t.Fatalf("encoded worktree missing compatibility/provenance key %q: %v", key, row)
				}
			}
			if tt.wantUnknown == 1 && strings.Contains(coldReapBytesSummary(got.EligibleBytes, got.EligibleBytesUnknown), "0B") {
				t.Fatalf("unknown bytes rendered as known zero: %q", coldReapBytesSummary(got.EligibleBytes, got.EligibleBytesUnknown))
			}
		})
	}
}

func TestWorktreeColdReapProgressPrecedesFinalReceipt(t *testing.T) {
	repo, _, _, floor := newColdReapProbeFixture(t)
	var stream bytes.Buffer
	worktreeWorkerReapAllColdTo(repo, false, floor, false, &stream, &stream)

	dec := json.NewDecoder(&stream)
	var progress workerworktree.ColdReapProgress
	if err := dec.Decode(&progress); err != nil {
		t.Fatalf("decode progress: %v\nstream=%s", err, stream.String())
	}
	if progress.Completed != 1 || progress.Total != 1 || progress.Eligible != 1 {
		t.Fatalf("first event is not completed ColdReapProgress: %+v", progress)
	}
	var final worktreeColdReapOut
	if err := dec.Decode(&final); err != nil {
		t.Fatalf("decode final receipt after progress: %v\nstream=%s", err, stream.String())
	}
	if final.Mode != "dry-run" || len(final.Worktrees) != 1 {
		t.Fatalf("final receipt=%+v", final)
	}
}

func sameResolvedPath(a, b string) bool {
	resolve := func(path string) string {
		if resolved, err := filepath.EvalSymlinks(path); err == nil {
			path = resolved
		}
		return filepath.Clean(path)
	}
	return resolve(a) == resolve(b)
}

func TestWorktreeColdReapApplyUsesOnePlanningSnapshot(t *testing.T) {
	repo, wt, now, floor := newColdReapProbeFixture(t)
	sizeCalls := 0
	reapCalls := 0
	got := worktreeColdReapReportWithOptionsAndProbes(
		repo, true, floor, now, false,
		workerworktree.ColdReapOptions{
			Concurrency: 1,
			Size: func(path string) (int64, error) {
				sizeCalls++
				if !sameResolvedPath(path, wt) {
					t.Fatalf("planned unexpected path %q want %q", path, wt)
				}
				return 8192, nil
			},
		},
		func(paths []string) (map[string]bool, error) {
			if len(paths) != 1 || !sameResolvedPath(paths[0], wt) {
				t.Fatalf("process snapshot paths=%v want [%s]", paths, wt)
			}
			return map[string]bool{}, nil
		},
		func(string) (bool, error) { return false, nil },
		func(root, path string) workerworktree.Result {
			reapCalls++
			return workerworktree.ForceReap(root, path, nil)
		},
	)
	if sizeCalls != 1 || reapCalls != 1 {
		t.Fatalf("apply recensused or changed the plan: size_calls=%d reap_calls=%d out=%+v", sizeCalls, reapCalls, got)
	}
	if got.WouldReap != 1 || got.Reaped != 1 ||
		got.EligibleBytes != 8192 || got.EligibleBytesKnown != 1 || got.EligibleBytesUnknown != 0 ||
		got.ReapedBytes != 8192 || got.ReapedBytesKnown != 1 || got.ReapedBytesUnknown != 0 {
		t.Fatalf("apply receipt did not preserve planned bytes: %+v", got)
	}
}

func captureWorktreeWorkerStdout(t *testing.T, fn func()) []byte {
	t.Helper()
	read, write, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	old := os.Stdout
	os.Stdout = write
	defer func() { os.Stdout = old }()
	fn()
	if err := write.Close(); err != nil {
		t.Fatal(err)
	}
	raw, err := io.ReadAll(read)
	if err != nil {
		t.Fatal(err)
	}
	if err := read.Close(); err != nil {
		t.Fatal(err)
	}
	return bytes.TrimSpace(raw)
}

func TestWorktreeWorkerListLocalOnlyOutputUnchangedWhenRemoteOmitted(t *testing.T) {
	repo, _, _ := newSingleReapFixture(t)
	raw := captureWorktreeWorkerStdout(t, func() { worktreeWorkerList([]string{"--root", repo, "--json"}) })
	var got map[string]json.RawMessage
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatalf("decode: %v output=%s", err, raw)
	}
	want := []string{"schema", "count", "paths", "inventory", "capacity"}
	if len(got) != len(want) {
		t.Fatalf("local-only keys=%v output=%s", reflect.ValueOf(got).MapKeys(), raw)
	}
	for _, key := range want {
		if _, ok := got[key]; !ok {
			t.Fatalf("missing local-only key %q: %s", key, raw)
		}
	}
	for _, forbidden := range []string{"local", "remote", "hosts", "fetched"} {
		if _, ok := got[forbidden]; ok {
			t.Fatalf("local-only output gained %q: %s", forbidden, raw)
		}
	}
}

func worktreeWorkerTestGit(t *testing.T, dir string, args ...string) string {
	t.Helper()
	cmd := exec.Command("git", args...)
	cmd.Dir = dir
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("git %v: %v\n%s", args, err, out)
	}
	return strings.TrimSpace(string(out))
}

func worktreeWorkerSnapshotClones(t *testing.T) (string, string, string) {
	t.Helper()
	base := t.TempDir()
	bare := filepath.Join(base, "remote.git")
	worktreeWorkerTestGit(t, base, "init", "--bare", bare)
	source := filepath.Join(base, "source")
	worktreeWorkerTestGit(t, base, "clone", bare, source)
	if err := os.WriteFile(filepath.Join(source, "README"), []byte("seed\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	worktreeWorkerTestGit(t, source, "add", "README")
	worktreeWorkerTestGit(t, source, "-c", "user.name=fak-test", "-c", "user.email=fak@example.invalid", "commit", "-m", "seed")
	worktreeWorkerTestGit(t, source, "push", "origin", "HEAD:main")
	clone := filepath.Join(base, "clone")
	worktreeWorkerTestGit(t, base, "clone", "--branch", "main", bare, clone)
	return bare, source, clone
}

func TestWorktreeWorkerRemoteListGroupsLocalAndRemoteProvenance(t *testing.T) {
	_, source, clone := worktreeWorkerSnapshotClones(t)
	now := time.Now().UTC()
	s, err := workerworktree.NewRemoteSnapshot("publisher", now, []workerworktree.SnapshotRow{{Association: workerworktree.SnapshotAssociation{State: "ASSOCIATED", Lane: "lane-a"}, Liveness: workerworktree.SnapshotLiveness{Owner: "LIVE", Lease: "LIVE"}, Cleanliness: workerworktree.SnapshotCleanliness{State: "CLEAN"}, Lifecycle: "READY"}})
	if err != nil {
		t.Fatal(err)
	}
	if got := workerworktree.PublishRemoteSnapshot(source, "origin", s, true, nil); !got.OK {
		t.Fatalf("publish=%+v", got)
	}
	raw := captureWorktreeWorkerStdout(t, func() { worktreeWorkerList([]string{"--root", clone, "--json", "--remote", "origin", "--fetch"}) })
	var got worktreeWorkerRemoteListOut
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatalf("decode: %v output=%s", err, raw)
	}
	if got.Schema != "fak-worker-cross-host-lifecycle/1" || got.Local.Provenance != "LOCAL_LIVE" || !got.Local.Authoritative || got.Local.Freshness != workerworktree.SnapshotFresh {
		t.Fatalf("local=%+v", got.Local)
	}
	if !got.Fetched || got.Remote != "origin" || len(got.Hosts) != 1 {
		t.Fatalf("remote output=%+v", got)
	}
	if got.Hosts[0].Provenance != "REMOTE_SNAPSHOT" || got.Hosts[0].Authoritative || got.Hosts[0].Freshness != workerworktree.SnapshotFresh {
		t.Fatalf("host=%+v", got.Hosts[0])
	}
}

func TestWorktreeWorkerPublishRequiresExplicitMode(t *testing.T) {
	raw := captureWorktreeWorkerStdout(t, func() { worktreeWorkerPublish([]string{"--remote", "origin"}) })
	var got workerworktree.SnapshotPublishResult
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatalf("decode: %v output=%s", err, raw)
	}
	if got.OK || !strings.Contains(got.Reason, "exactly one") {
		t.Fatalf("publish=%+v", got)
	}
}

func newSymptomWorkerFixture(t *testing.T, isRedThenGreen bool) (repo, worktree, base string) {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git unavailable")
	}
	if _, err := exec.LookPath("go"); err != nil {
		t.Skip("go unavailable")
	}
	root := t.TempDir()
	repo = filepath.Join(root, "repo")
	worktree = filepath.Join(root, "worker")
	if err := os.MkdirAll(filepath.Join(repo, "pkg"), 0o755); err != nil {
		t.Fatal(err)
	}
	git := func(dir string, args ...string) string {
		t.Helper()
		cmd := exec.Command("git", append([]string{"-C", dir}, args...)...)
		cmd.Env = append(os.Environ(),
			"GIT_AUTHOR_NAME=t", "GIT_AUTHOR_EMAIL=t@t",
			"GIT_COMMITTER_NAME=t", "GIT_COMMITTER_EMAIL=t@t")
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("git -C %s %s: %v: %s", dir, strings.Join(args, " "), err, out)
		}
		return strings.TrimSpace(string(out))
	}
	git(repo, "init", "-q", "-b", "main")
	git(repo, "config", "user.email", "t@t")
	git(repo, "config", "user.name", "t")
	git(repo, "config", "maintenance.auto", "false")
	git(repo, "config", "gc.auto", "0")
	git(repo, "config", "commit.gpgsign", "false")
	git(repo, "config", "core.hooksPath", "")

	_ = os.WriteFile(filepath.Join(repo, "go.mod"), []byte("module m\n\ngo 1.21\n"), 0o644)
	_ = os.WriteFile(filepath.Join(repo, "pkg", "calc.go"), []byte("package pkg\n\nfunc Calc(n int) int {\n\tif n > 0 { return n }\n\treturn 0\n}\n"), 0o644)
	git(repo, "add", ".")
	git(repo, "commit", "-qm", "init")
	base = git(repo, "rev-parse", "HEAD")

	git(repo, "worktree", "add", "--detach", worktree, base)

	if isRedThenGreen {
		_ = os.WriteFile(filepath.Join(worktree, "pkg", "calc.go"), []byte("package pkg\n\nfunc Calc(n int) int {\n\tif n > 0 { return n }\n\tif n < 0 { return -n }\n\treturn 0\n}\n"), 0o644)
		_ = os.WriteFile(filepath.Join(worktree, "pkg", "calc_test.go"), []byte("package pkg\n\nimport \"testing\"\n\nfunc TestCalcNegative(t *testing.T) {\n\tif Calc(-5) != 5 {\n\t\tt.Fatalf(\"Calc(-5)=%d, want 5\", Calc(-5))\n\t}\n}\n"), 0o644)
	} else {
		_ = os.WriteFile(filepath.Join(worktree, "pkg", "calc.go"), []byte("package pkg\n\nfunc Calc(n int) int {\n\tif n > 0 { return n }\n\tif n < 0 { return -n }\n\treturn 0\n}\n"), 0o644)
		_ = os.WriteFile(filepath.Join(worktree, "pkg", "calc_test.go"), []byte("package pkg\n\nimport \"testing\"\n\nfunc TestCalcPositive(t *testing.T) {\n\tif Calc(5) != 5 {\n\t\tt.Fatalf(\"Calc(5)=%d, want 5\", Calc(5))\n\t}\n}\n"), 0o644)
	}

	git(worktree, "add", ".")
	git(worktree, "commit", "-qm", "fix(calc): fix negative calculation (fak calc)")
	return repo, worktree, base
}

func TestWorkerLandSymptom(t *testing.T) {
	t.Run("tautological fix is rejected with SYMPTOM_UNWITNESSED", func(t *testing.T) {
		repo, worktree, base := newSymptomWorkerFixture(t, false)
		var out, errb bytes.Buffer
		res, code := runWorktreeWorkerLand(&out, &errb, []string{
			"--root", repo,
			"--worktree", worktree,
			"--base-sha", base,
		})
		if res.OK || res.Code != "SYMPTOM_UNWITNESSED" || code == 0 {
			t.Fatalf("expected rejection with SYMPTOM_UNWITNESSED, got res=%+v code=%d err=%s", res, code, errb.String())
		}
	})

	t.Run("red-then-green fix is accepted and lands cleanly", func(t *testing.T) {
		repo, worktree, base := newSymptomWorkerFixture(t, true)
		var out, errb bytes.Buffer
		res, code := runWorktreeWorkerLand(&out, &errb, []string{
			"--root", repo,
			"--worktree", worktree,
			"--base-sha", base,
			"--paths", "pkg/calc.go",
			"--paths", "pkg/calc_test.go",
		})
		if !res.OK || code != 0 {
			t.Fatalf("expected successful landing, got res=%+v code=%d err=%s", res, code, errb.String())
		}
	})

	t.Run("unsafe bypass flag skips symptom check for fix", func(t *testing.T) {
		repo, worktree, base := newSymptomWorkerFixture(t, false)
		var out, errb bytes.Buffer
		res, code := runWorktreeWorkerLand(&out, &errb, []string{
			"--root", repo,
			"--worktree", worktree,
			"--base-sha", base,
			"--paths", "pkg/calc.go",
			"--paths", "pkg/calc_test.go",
			"--unsafe-skip-symptom-witness",
		})
		if !res.OK || code != 0 {
			t.Fatalf("expected bypass to succeed, got res=%+v code=%d err=%s", res, code, errb.String())
		}
	})
}

func newTwoWorkerFixture(t *testing.T) (repo string, wt1, wt2 string, base string) {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git unavailable")
	}
	t.Setenv(workerworktree.PoolCapEnv, "0")
	root := t.TempDir()
	repo = filepath.Join(root, "repo")
	if err := os.MkdirAll(repo, 0o755); err != nil {
		t.Fatal(err)
	}
	git := func(args ...string) string {
		t.Helper()
		cmd := exec.Command("git", append([]string{"-C", repo}, args...)...)
		cmd.Env = append(os.Environ(),
			"GIT_AUTHOR_NAME=t", "GIT_AUTHOR_EMAIL=t@t",
			"GIT_COMMITTER_NAME=t", "GIT_COMMITTER_EMAIL=t@t")
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("git %s: %v: %s", strings.Join(args, " "), err, out)
		}
		return strings.TrimSpace(string(out))
	}
	git("init", "-q", "-b", "main")
	git("config", "user.email", "t@t")
	git("config", "user.name", "t")
	if err := os.WriteFile(filepath.Join(repo, "owned.txt"), []byte("base\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	git("add", "owned.txt")
	git("commit", "-qm", "base")
	base = git("rev-parse", "HEAD")
	prep1 := workerworktree.Prepare(repo, "cmd", "10447-a", base, filepath.Join(root, "workers"), nil)
	if !prep1.OK {
		t.Fatalf("prepare wt1: %+v", prep1)
	}
	prep2 := workerworktree.Prepare(repo, "tools", "10447-b", base, filepath.Join(root, "workers"), nil)
	if !prep2.OK {
		t.Fatalf("prepare wt2: %+v", prep2)
	}
	t.Cleanup(func() {
		_ = workerworktree.ForceReap(repo, prep1.Path, nil)
		_ = workerworktree.ForceReap(repo, prep2.Path, nil)
	})
	return repo, prep1.Path, prep2.Path, base
}

func TestWorktreeWorkerListWorkerFilter(t *testing.T) {
	repo, wt1, wt2, _ := newTwoWorkerFixture(t)

	// Filter by directory basename
	base1 := filepath.Base(wt1)
	raw := captureWorktreeWorkerStdout(t, func() {
		worktreeWorkerList([]string{"--root", repo, "--json", "--worker", base1})
	})
	var got worktreeWorkerLifecycleOut
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatalf("decode: %v\noutput: %s", err, raw)
	}
	if got.Count != 1 || len(got.Paths) != 1 || !sameResolvedPath(got.Paths[0], wt1) {
		t.Fatalf("filter by basename: expected only wt1 (%s), got: %+v", wt1, got)
	}

	// Filter by full path
	raw = captureWorktreeWorkerStdout(t, func() {
		worktreeWorkerList([]string{"--root", repo, "--json", "--worker", wt2})
	})
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatalf("decode: %v\noutput: %s", err, raw)
	}
	if got.Count != 1 || len(got.Paths) != 1 || !sameResolvedPath(got.Paths[0], wt2) {
		t.Fatalf("filter by full path: expected only wt2 (%s), got: %+v", wt2, got)
	}

	// Filter non-existent worker
	raw = captureWorktreeWorkerStdout(t, func() {
		worktreeWorkerList([]string{"--root", repo, "--json", "--worker", "nonexistent-worker"})
	})
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatalf("decode: %v\noutput: %s", err, raw)
	}
	if got.Count != 0 || len(got.Paths) != 0 {
		t.Fatalf("filter nonexistent: expected 0 results, got: %+v", got)
	}

	// Legacy mode filter by basename
	raw = captureWorktreeWorkerStdout(t, func() {
		worktreeWorkerList([]string{"--root", repo, "--worker", base1})
	})
	var legacyGot worktreeWorkerListOut
	if err := json.Unmarshal(raw, &legacyGot); err != nil {
		t.Fatalf("decode legacy: %v\noutput: %s", err, raw)
	}
	if legacyGot.Count != 1 || len(legacyGot.Paths) != 1 || !sameResolvedPath(legacyGot.Paths[0], wt1) {
		t.Fatalf("legacy filter by basename: expected only wt1, got: %+v", legacyGot)
	}
}

func TestWorktreeWorkerListSessionFilter(t *testing.T) {
	repo, wt1, wt2, _ := newTwoWorkerFixture(t)

	// Write lease.json with session IDs
	if err := workerworktree.WriteWorkerLease(wt1, workerworktree.WorkerLease{
		PID:       os.Getpid(),
		SessionID: "sess-alpha-10447",
	}); err != nil {
		t.Fatal(err)
	}
	if err := workerworktree.WriteWorkerLease(wt2, workerworktree.WorkerLease{
		PID:       os.Getpid(),
		SessionID: "sess-beta-10447",
	}); err != nil {
		t.Fatal(err)
	}

	// Filter for sess-alpha
	raw := captureWorktreeWorkerStdout(t, func() {
		worktreeWorkerList([]string{"--root", repo, "--json", "--session", "sess-alpha-10447"})
	})
	var got worktreeWorkerLifecycleOut
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatalf("decode: %v\noutput: %s", err, raw)
	}
	if got.Count != 1 || len(got.Paths) != 1 || !sameResolvedPath(got.Paths[0], wt1) {
		t.Fatalf("filter by session alpha: expected only wt1, got: %+v", got)
	}

	// Filter for sess-beta
	raw = captureWorktreeWorkerStdout(t, func() {
		worktreeWorkerList([]string{"--root", repo, "--json", "--session", "sess-beta-10447"})
	})
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatalf("decode: %v\noutput: %s", err, raw)
	}
	if got.Count != 1 || len(got.Paths) != 1 || !sameResolvedPath(got.Paths[0], wt2) {
		t.Fatalf("filter by session beta: expected only wt2, got: %+v", got)
	}

	// Filter for nonexistent session
	raw = captureWorktreeWorkerStdout(t, func() {
		worktreeWorkerList([]string{"--root", repo, "--json", "--session", "sess-none-10447"})
	})
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatalf("decode: %v\noutput: %s", err, raw)
	}
	if got.Count != 0 || len(got.Paths) != 0 {
		t.Fatalf("filter nonexistent session: expected 0 results, got: %+v", got)
	}
}

func TestWorktreeWorkerListTimeoutReturnsPartialResultWithoutHanging(t *testing.T) {
	repo, wt1, wt2, _ := newTwoWorkerFixture(t)

	oldProbes := worktreeWorkerListProbes
	defer func() { worktreeWorkerListProbes = oldProbes }()

	// Stub Inspect so wt1 completes quickly and wt2 sleeps
	worktreeWorkerListProbes = worktreeWorkerLifecycleProbes{
		Inspect: func(repoRoot, path string) (worktreeWorkerRevisionEvidence, error) {
			if sameResolvedPath(path, wt2) {
				time.Sleep(3 * time.Second)
			}
			return worktreeWorkerRevisionEvidence{
				HeadSHA:     "head-" + filepath.Base(path),
				BaseSHA:     "base",
				Cleanliness: worktreeEvidenceClean,
				DirtyPaths:  []string{},
			}, nil
		},
	}

	start := time.Now()
	raw := captureWorktreeWorkerStdout(t, func() {
		worktreeWorkerList([]string{"--root", repo, "--json", "--timeout", "50ms"})
	})
	elapsed := time.Since(start)

	if elapsed >= 2*time.Second {
		t.Fatalf("command hung for %s, want timeout around 50ms", elapsed)
	}

	var got worktreeWorkerLifecycleOut
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatalf("decode timeout output: %v\noutput: %s", err, raw)
	}
	if !got.Partial || !got.Timeout {
		t.Fatalf("expected partial=true and timeout=true, got partial=%v timeout=%v", got.Partial, got.Timeout)
	}
	if got.Count != 1 || len(got.Inventory) != 1 || !sameResolvedPath(got.Inventory[0].Path, wt1) {
		t.Fatalf("expected partial results containing only wt1, got: %+v", got)
	}
}

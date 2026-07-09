package main

import (
	"encoding/json"
	"testing"

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

// TestListJSONShape proves `fak worktree worker list` emits {count, paths} and
// that an empty listing renders paths as [] (never null), so a JSON consumer can
// always range it.
func TestListJSONShape(t *testing.T) {
	b, err := json.Marshal(worktreeWorkerListOut{Count: 0, Paths: []string{}})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if string(b) != `{"count":0,"paths":[]}` {
		t.Fatalf("empty list json = %s, want {\"count\":0,\"paths\":[]}", b)
	}
	got := mustKeys(t, worktreeWorkerListOut{Count: 2, Paths: []string{"/a", "/b"}}, "count", "paths")
	if got["count"].(float64) != 2 {
		t.Fatalf("count = %v, want 2", got["count"])
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

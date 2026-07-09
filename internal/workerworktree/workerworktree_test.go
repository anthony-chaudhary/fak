package workerworktree

import (
	"path/filepath"
	"strings"
	"testing"
)

// fakeGit records every (root, args) call and replies from a per-verb queue or a
// default, mirroring the Python FakeGit so the whole prepare→land→reap path is
// exercised without a real repo.
type fakeGit struct {
	calls   [][]string
	replies map[string][](struct {
		rc  int
		out string
	})
	byVerb map[string]struct {
		rc  int
		out string
	}
	def struct {
		rc  int
		out string
	}
}

func newFakeGit() *fakeGit {
	return &fakeGit{byVerb: map[string]struct {
		rc  int
		out string
	}{}}
}

func (f *fakeGit) reply(verb string, rc int, out string) *fakeGit {
	f.byVerb[verb] = struct {
		rc  int
		out string
	}{rc, out}
	return f
}

func (f *fakeGit) run(root string, args []string) (int, string) {
	f.calls = append(f.calls, append([]string{}, args...))
	verb := ""
	if len(args) > 0 {
		verb = args[0]
	}
	if r, ok := f.byVerb[verb]; ok {
		return r.rc, r.out
	}
	return f.def.rc, f.def.out
}

func (f *fakeGit) callsWithPrefix(prefix ...string) [][]string {
	var out [][]string
	for _, c := range f.calls {
		if len(c) < len(prefix) {
			continue
		}
		match := true
		for i, p := range prefix {
			if c[i] != p {
				match = false
				break
			}
		}
		if match {
			out = append(out, c)
		}
	}
	return out
}

func contains(args []string, want string) bool {
	for _, a := range args {
		if a == want {
			return true
		}
	}
	return false
}

// ---- pure planners -------------------------------------------------------- //

func TestDirNameIsOneFlatSafeSegment(t *testing.T) {
	name := DirName("tools", "3168")
	if !strings.HasPrefix(name, "fak-worker-wt-tools-") {
		t.Fatalf("dir name %q missing marker/lane prefix", name)
	}
	if strings.ContainsAny(name, "/\\") {
		t.Fatalf("dir name %q is not a flat segment", name)
	}
}

func TestHostileKeyCannotEscapeRoot(t *testing.T) {
	name := DirName("tools", "../../etc/passwd")
	if strings.Contains(name, "/") || strings.Contains(name, "..") {
		t.Fatalf("hostile key leaked into dir name %q", name)
	}
}

func TestOddLaneIsSanitised(t *testing.T) {
	name := DirName("a/b lane", "k")
	if strings.Contains(name, "/") || strings.Contains(name, " ") {
		t.Fatalf("odd lane not sanitised: %q", name)
	}
}

func TestDistinctKeysDistinctDirs(t *testing.T) {
	if DirName("tools", "1") == DirName("tools", "2") {
		t.Fatal("distinct keys must yield distinct dirs")
	}
}

func TestPathUnderExplicitRoot(t *testing.T) {
	root := filepath.FromSlash("/tmp/wtroot")
	p := Path("gateway", "42", root)
	if filepath.Dir(p) != filepath.Clean(root) {
		t.Fatalf("path %q not under explicit root %q", p, root)
	}
	if !strings.HasPrefix(filepath.Base(p), "fak-worker-wt-gateway-") {
		t.Fatalf("path base %q missing marker", filepath.Base(p))
	}
}

func TestDefaultRootHonoursEnv(t *testing.T) {
	custom := filepath.FromSlash("/custom/wt")
	t.Setenv(WorktreeRootEnv, custom)
	if got := DefaultRoot(); got != custom {
		t.Fatalf("DefaultRoot = %q, want %q", got, custom)
	}
}

func TestWorktreeEnvIsolatesBuildIntoWorktree(t *testing.T) {
	wt := filepath.FromSlash("/tmp/wt/fak-worker-wt-tools-abc")
	env := WorktreeEnv(map[string]string{"PATH": "/usr/bin"}, wt)
	if !strings.HasPrefix(env["GOCACHE"], wt) {
		t.Fatalf("GOCACHE %q not inside worktree", env["GOCACHE"])
	}
	if !strings.HasPrefix(env["GOTMPDIR"], wt) {
		t.Fatalf("GOTMPDIR %q not inside worktree", env["GOTMPDIR"])
	}
	if env["DISPATCH_WORKSPACE"] != wt {
		t.Fatalf("DISPATCH_WORKSPACE = %q, want %q", env["DISPATCH_WORKSPACE"], wt)
	}
	if env[WorktreeDirEnv] != wt {
		t.Fatalf("%s = %q, want %q", WorktreeDirEnv, env[WorktreeDirEnv], wt)
	}
	if env["PATH"] != "/usr/bin" {
		t.Fatal("base env not preserved")
	}
}

func TestWorktreeEnvDoesNotMutateBase(t *testing.T) {
	base := map[string]string{"PATH": "/usr/bin"}
	WorktreeEnv(base, filepath.FromSlash("/tmp/wt/fak-worker-wt-x-y"))
	if _, ok := base["GOCACHE"]; ok {
		t.Fatal("base env was mutated")
	}
}

func TestIsWorkerWorktreeOnlyMarkerDirs(t *testing.T) {
	if !IsWorkerWorktree(filepath.FromSlash("/tmp/Fleet/worker-worktrees/fak-worker-wt-tools-deadbeef")) {
		t.Fatal("marker worktree not recognized")
	}
	if IsWorkerWorktree(filepath.FromSlash("/work/fak")) {
		t.Fatal("trunk mis-flagged as worker worktree")
	}
	if IsWorkerWorktree(filepath.FromSlash("/tmp/fak-selfupdate-build-1")) {
		t.Fatal("a foreign scratch worktree mis-flagged as ours")
	}
}

// ---- Prepare -------------------------------------------------------------- //

func TestPrepareDetachedAddAtTrunkHead(t *testing.T) {
	g := newFakeGit().reply("rev-parse", 0, "feedface\n").reply("worktree", 0, "")
	res := Prepare("/r", "tools", "3168", "", t.TempDir(), g.run)
	if !res.OK {
		t.Fatalf("prepare failed: %+v", res)
	}
	if res.BaseSHA != "feedface" {
		t.Fatalf("base sha = %q, want feedface", res.BaseSHA)
	}
	adds := g.callsWithPrefix("worktree", "add")
	if len(adds) != 1 {
		t.Fatalf("want 1 worktree add, got %d: %v", len(adds), g.calls)
	}
	add := adds[0]
	if !contains(add, "--detach") {
		t.Fatalf("add must be --detach: %v", add)
	}
	if add[len(add)-1] != "feedface" {
		t.Fatalf("add must pin at resolved sha, got %v", add)
	}
}

func TestPrepareFailOpenWhenHeadUnresolvable(t *testing.T) {
	g := newFakeGit().reply("rev-parse", 127, "")
	res := Prepare("/r", "tools", "1", "", t.TempDir(), g.run)
	if res.OK {
		t.Fatal("prepare must fail when HEAD unresolvable")
	}
	if len(g.callsWithPrefix("worktree", "add")) != 0 {
		t.Fatal("fail-open: no worktree add should have been attempted")
	}
}

func TestPrepareExplicitBaseSHASkipsRevParse(t *testing.T) {
	g := newFakeGit().reply("worktree", 0, "")
	res := Prepare("/r", "tools", "1", "cafe", t.TempDir(), g.run)
	if !res.OK || res.BaseSHA != "cafe" {
		t.Fatalf("explicit base sha not honored: %+v", res)
	}
	if len(g.callsWithPrefix("rev-parse")) != 0 {
		t.Fatal("rev-parse should be skipped when base sha given")
	}
}

func TestPrepareAddFailureIsFailOpen(t *testing.T) {
	g := newFakeGit().reply("rev-parse", 0, "abc\n").reply("worktree", 1, "fatal: already exists")
	res := Prepare("/r", "tools", "1", "", t.TempDir(), g.run)
	if res.OK {
		t.Fatal("add failure must fail open (ok=false)")
	}
	if !strings.Contains(res.Reason, "fail open") {
		t.Fatalf("reason should name fail-open: %q", res.Reason)
	}
}

// ---- Reap ----------------------------------------------------------------- //

func TestReapReapsOnlyMarkerWorktree(t *testing.T) {
	g := newFakeGit().reply("worktree", 0, "")
	res := Reap("/r", "/wt/fak-worker-wt-tools-abc", g.run)
	if !res.OK {
		t.Fatalf("reap of marker worktree failed: %+v", res)
	}
	removes := g.callsWithPrefix("worktree", "remove")
	if len(removes) != 1 || !contains(removes[0], "--force") {
		t.Fatalf("want one forced remove, got %v", g.calls)
	}
	prunes := 0
	for _, c := range g.calls {
		if len(c) == 2 && c[0] == "worktree" && c[1] == "prune" {
			prunes++
		}
	}
	if prunes != 1 {
		t.Fatalf("want a trailing prune, got %v", g.calls)
	}
}

func TestReapRefusesNonWorkerWorktree(t *testing.T) {
	g := newFakeGit()
	res := Reap("/r", filepath.FromSlash("/work/fak"), g.run)
	if res.OK {
		t.Fatal("reap of non-worker worktree must refuse")
	}
	if len(g.calls) != 0 {
		t.Fatalf("must never touch git for a non-worker path, got %v", g.calls)
	}
}

// ---- Land ----------------------------------------------------------------- //

func TestLandLandsDiffOntoTrunkByPathSignedOff(t *testing.T) {
	g := newFakeGit().
		reply("diff", 0, "diff --git a/x b/x\n@@\n-old\n+new\n").
		reply("apply", 0, "").
		reply("commit", 0, "[main abc] msg")
	res := Land("/trunk", "/wt/fak-worker-wt-tools-abc", "feedface", "/tmp/msg.txt", []string{"x"}, nil, g.run)
	if !res.OK || !res.Committed {
		t.Fatalf("land failed: %+v", res)
	}
	commits := g.callsWithPrefix("commit")
	if len(commits) != 1 {
		t.Fatalf("want one commit, got %v", g.calls)
	}
	c := commits[0]
	if !contains(c, "-s") || !contains(c, "--") || !contains(c, "x") {
		t.Fatalf("commit must be signed-off and path-scoped: %v", c)
	}
}

func TestLandEmptyDiffIsOKButNotCommitted(t *testing.T) {
	g := newFakeGit().reply("diff", 0, "   \n")
	res := Land("/trunk", "/wt/fak-worker-wt-tools-abc", "cafe", "/tmp/msg.txt", nil, nil, g.run)
	if !res.OK || res.Committed {
		t.Fatalf("empty diff should be ok-but-not-committed: %+v", res)
	}
	if len(g.callsWithPrefix("apply")) != 0 || len(g.callsWithPrefix("commit")) != 0 {
		t.Fatalf("no apply/commit on empty diff, got %v", g.calls)
	}
}

func TestLandApplyFailureDoesNotCommit(t *testing.T) {
	g := newFakeGit().
		reply("diff", 0, "diff --git a/x b/x\n@@\n-o\n+n\n").
		reply("apply", 1, "error: patch does not apply")
	res := Land("/trunk", "/wt/fak-worker-wt-tools-abc", "cafe", "/tmp/msg.txt", nil, nil, g.run)
	if res.OK || res.Committed {
		t.Fatalf("apply failure must not commit: %+v", res)
	}
	if len(g.callsWithPrefix("commit")) != 0 {
		t.Fatalf("no commit after failed apply, got %v", g.calls)
	}
}

func TestLandBaseSHAIsTheDiffRefNotHead(t *testing.T) {
	g := newFakeGit().
		reply("diff", 0, "diff --git a/x b/x\n@@\n-old\n+new\n").
		reply("apply", 0, "").
		reply("commit", 0, "[main abc] msg")
	res := Land("/trunk", "/wt/fak-worker-wt-tools-abc", "feedface", "/tmp/msg.txt", []string{"x"}, nil, g.run)
	if !res.Committed {
		t.Fatalf("land should commit: %+v", res)
	}
	diffs := g.callsWithPrefix("diff")
	if len(diffs) != 1 {
		t.Fatalf("want one diff call, got %v", g.calls)
	}
	if !contains(diffs[0], "feedface") || contains(diffs[0], "HEAD") {
		t.Fatalf("diff base must be the pinned sha, not HEAD: %v", diffs[0])
	}
}

func TestLandVerifyFailureRefusesToLand(t *testing.T) {
	g := newFakeGit().
		reply("diff", 0, "diff --git a/x b/x\n@@\n-o\n+n\n").
		reply("apply", 0, "").
		reply("commit", 0, "[main abc] msg")
	var seen string
	verify := func(wt string) (bool, string) {
		seen = wt
		return false, "go build ./... : x.go:1: broke"
	}
	res := Land("/trunk", "/wt/fak-worker-wt-tools-abc", "cafe", "/tmp/msg.txt", nil, verify, g.run)
	if res.OK || res.Committed {
		t.Fatalf("failed verify must refuse the land: %+v", res)
	}
	if len(g.callsWithPrefix("apply")) != 0 || len(g.callsWithPrefix("commit")) != 0 {
		t.Fatalf("failed witness runs before apply/commit — nothing lands: %v", g.calls)
	}
	if seen == "" {
		t.Fatal("verify hook should receive the worktree path")
	}
}

func TestLandDiffErrorFailsOpen(t *testing.T) {
	g := newFakeGit().reply("diff", 127, "")
	res := Land("/trunk", "/wt/fak-worker-wt-tools-abc", "cafe", "/tmp/msg.txt", nil, nil, g.run)
	if res.OK {
		t.Fatalf("a diff git error must fail open (ok=false): %+v", res)
	}
	if len(g.callsWithPrefix("apply")) != 0 {
		t.Fatal("no apply after a diff error")
	}
}

// TestLandDerivesMsgFromWorktreeTipWhenNoFile proves the witness-sweep call site
// (which has no pre-written message file) borrows the worker's own commit subject.
func TestLandDerivesMsgFromWorktreeTipWhenNoFile(t *testing.T) {
	g := newFakeGit().
		reply("diff", 0, "diff --git a/x b/x\n@@\n-o\n+n\n").
		reply("log", 0, "fix(x): resolve thing (#3168) (fak x)\n").
		reply("apply", 0, "").
		reply("commit", 0, "[main abc] msg")
	res := Land("/trunk", "/wt/fak-worker-wt-tools-abc", "cafe", "", []string{"x"}, nil, g.run)
	if !res.Committed {
		t.Fatalf("land with derived message should commit: %+v", res)
	}
	if len(g.callsWithPrefix("log")) == 0 {
		t.Fatalf("empty msg-file should trigger a worktree tip `git log`: %v", g.calls)
	}
	commits := g.callsWithPrefix("commit")
	if len(commits) != 1 || !contains(commits[0], "-F") {
		t.Fatalf("commit must still use -F <derived msg file>: %v", commits)
	}
}

// ---- Land: opt-in post-commit readback (#3547 shared-index race) ---------- //

func TestLandReadbackVerifyPassesWhenTrunkCarriesPaths(t *testing.T) {
	t.Setenv(LandReadbackEnv, "1")
	g := newFakeGit().
		reply("diff", 0, "diff --git a/x b/x\n@@\n-old\n+new\n").
		reply("apply", 0, "").
		reply("commit", 0, "[main abc] msg").
		reply("rev-parse", 0, "abc123def4567\n").
		reply("diff-tree", 0, "x\n")
	res := Land("/trunk", "/wt/fak-worker-wt-tools-abc", "feedface", "/tmp/msg.txt", []string{"x"}, nil, g.run)
	if !res.OK || !res.Committed {
		t.Fatalf("readback should pass when trunk HEAD carries the intended path: %+v", res)
	}
}

func TestLandReadbackVerifyRefusesWhenPathSweptByRace(t *testing.T) {
	t.Setenv(LandReadbackEnv, "1")
	// commit succeeded, but trunk HEAD carries a DIFFERENT file — our `x` was swept
	// into a concurrent commit on the shared index (the #3547 failure).
	g := newFakeGit().
		reply("diff", 0, "diff --git a/x b/x\n@@\n-old\n+new\n").
		reply("apply", 0, "").
		reply("commit", 0, "[main abc] msg").
		reply("rev-parse", 0, "deadbeef12345\n").
		reply("diff-tree", 0, "some/other/file.go\n")
	res := Land("/trunk", "/wt/fak-worker-wt-tools-abc", "feedface", "/tmp/msg.txt", []string{"x"}, nil, g.run)
	if res.OK {
		t.Fatalf("readback must refuse a false-success when the intended path is missing: %+v", res)
	}
	if !strings.Contains(res.Reason, "LAND_READBACK_MISMATCH") || !strings.Contains(res.Reason, "3547") {
		t.Fatalf("refusal must name LAND_READBACK_MISMATCH and cite #3547: %q", res.Reason)
	}
}

func TestLandReadbackDefaultOffLeavesBaselineUnchanged(t *testing.T) {
	t.Setenv(LandReadbackEnv, "0") // explicit off — baseline path
	// A diff-tree that WOULD fail the check must never be consulted when off.
	g := newFakeGit().
		reply("diff", 0, "diff --git a/x b/x\n@@\n-old\n+new\n").
		reply("apply", 0, "").
		reply("commit", 0, "[main abc] msg").
		reply("diff-tree", 0, "totally/unrelated.go\n")
	res := Land("/trunk", "/wt/fak-worker-wt-tools-abc", "feedface", "/tmp/msg.txt", []string{"x"}, nil, g.run)
	if !res.OK || !res.Committed {
		t.Fatalf("baseline (readback off) must land regardless of trunk contents: %+v", res)
	}
	if len(g.callsWithPrefix("diff-tree")) != 0 {
		t.Fatalf("readback OFF must not consult diff-tree: %v", g.calls)
	}
}

func TestLandReadbackFailsOpenOnGitError(t *testing.T) {
	t.Setenv(LandReadbackEnv, "1")
	// HEAD unreadable — the readback cannot run, so it must NOT manufacture a
	// refusal; the commit's own verdict stands (fail-open, the module invariant).
	g := newFakeGit().
		reply("diff", 0, "diff --git a/x b/x\n@@\n-old\n+new\n").
		reply("apply", 0, "").
		reply("commit", 0, "[main abc] msg").
		reply("rev-parse", 127, "")
	res := Land("/trunk", "/wt/fak-worker-wt-tools-abc", "feedface", "/tmp/msg.txt", []string{"x"}, nil, g.run)
	if !res.OK || !res.Committed {
		t.Fatalf("readback must FAIL OPEN on a git error, never refuse: %+v", res)
	}
}

// ---- Count ---------------------------------------------------------------- //

func TestCountOnlyCountsOurWorktrees(t *testing.T) {
	porc := "worktree /work/fak\nHEAD abc\nbranch refs/heads/main\n\n" +
		"worktree /tmp/Fleet/worker-worktrees/fak-worker-wt-tools-deadbeef\nHEAD def\ndetached\n\n" +
		"worktree /tmp/fak-selfupdate-build-1\nHEAD 123\ndetached\n"
	g := newFakeGit().reply("worktree", 0, porc)
	n, paths := Count("/r", g.run)
	if n != 1 {
		t.Fatalf("count = %d, want 1 (only our marker worktree)", n)
	}
	if len(paths) != 1 || !strings.HasSuffix(paths[0], "fak-worker-wt-tools-deadbeef") {
		t.Fatalf("count paths wrong: %v", paths)
	}
}

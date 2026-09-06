package workerworktree

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
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
	envCalls [][]string        // args of calls made through the env-aware runner
	lastEnv  map[string]string // env overlay of the most recent env-aware call
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

// replyOnce enqueues a single-use reply for verb, consumed before byVerb; chained
// calls sequence responses so a test can model state that CHANGES between calls —
// e.g. HEAD moving after a lost CAS, then holding still for the retry (#3570).
func (f *fakeGit) replyOnce(verb string, rc int, out string) *fakeGit {
	if f.replies == nil {
		f.replies = map[string][](struct {
			rc  int
			out string
		}){}
	}
	f.replies[verb] = append(f.replies[verb], struct {
		rc  int
		out string
	}{rc, out})
	return f
}

// replyLandDiff models the three diff reads a path-scoped land performs: the
// declared-path admission list, the patch itself, then the whole diff name list.
func replyLandDiff(g *fakeGit, declaredNames, patch, allNames string) *fakeGit {
	return g.
		replyOnce("diff", 0, declaredNames).
		replyOnce("diff", 0, patch).
		replyOnce("diff", 0, allNames)
}

func gitVerb(args []string) string {
	for i := 0; i < len(args); i++ {
		if args[i] == "-c" && i+1 < len(args) {
			i++
			continue
		}
		return args[i]
	}
	if len(args) > 0 {
		return args[0]
	}
	return ""
}

func stripGlobalFlags(c []string) []string {
	i := 0
	for i < len(c) {
		if c[i] == "-c" && i+1 < len(c) {
			i += 2
			continue
		}
		break
	}
	return c[i:]
}

func (f *fakeGit) run(root string, args []string) (int, string) {
	f.calls = append(f.calls, append([]string{}, args...))
	verb := gitVerb(args)
	if queue := f.replies[verb]; len(queue) > 0 {
		r := queue[0]
		f.replies[verb] = queue[1:]
		return r.rc, r.out
	}
	if r, ok := f.byVerb[verb]; ok {
		return r.rc, r.out
	}
	return f.def.rc, f.def.out
}

// runEnv is the GitEnvRunner face of the same fake: it records the overlay env and
// the args, and replies from the same per-verb table (single-use queue first, like
// run) so an isolated-land sequence (read-tree/apply/write-tree/commit-tree) is
// stubbed exactly like a plain call — including a retry whose re-apply behaves
// differently from the first apply (#3570).
func (f *fakeGit) runEnv(root string, env map[string]string, args []string) (int, string) {
	f.envCalls = append(f.envCalls, append([]string{}, args...))
	f.lastEnv = env
	verb := gitVerb(args)
	// Recovery anchors are an added side effect of isolated landing. Keep legacy
	// CAS response queues scoped to trunk update-ref calls.
	if len(args) > 1 && verb == "update-ref" && args[1] == "--create-reflog" {
		return 0, ""
	}
	if queue := f.replies[verb]; len(queue) > 0 {
		r := queue[0]
		f.replies[verb] = queue[1:]
		return r.rc, r.out
	}
	if r, ok := f.byVerb[verb]; ok {
		return r.rc, r.out
	}
	return f.def.rc, f.def.out
}

func (f *fakeGit) envCallsWithPrefix(prefix ...string) [][]string {
	saved := f.calls
	f.calls = f.envCalls
	out := f.callsWithPrefix(prefix...)
	f.calls = saved
	return out
}

func (f *fakeGit) callsWithPrefix(prefix ...string) [][]string {
	var out [][]string
	for _, raw := range f.calls {
		c := raw
		if len(prefix) > 0 && prefix[0] != "-c" {
			c = stripGlobalFlags(raw)
		}
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
			out = append(out, raw)
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

func TestEnsureBuildDirsRecreatesMissingOwnedDirectories(t *testing.T) {
	wt := t.TempDir()
	env, err := EnsureBuildDirs(wt)
	if err != nil {
		t.Fatalf("EnsureBuildDirs: %v", err)
	}
	for _, name := range []string{"GOCACHE", "GOTMPDIR"} {
		info, statErr := os.Stat(env[name])
		if statErr != nil || !info.IsDir() {
			t.Fatalf("%s directory was not recreated: info=%v err=%v", name, info, statErr)
		}
	}
}

func TestEnsureBuildDirsFailsClosedOnCreationError(t *testing.T) {
	wt := t.TempDir()
	gotmp := filepath.Join(wt, ".gotmp")
	if err := os.WriteFile(gotmp, []byte("not a directory"), 0o644); err != nil {
		t.Fatal(err)
	}
	_, err := EnsureBuildDirs(wt)
	if err == nil {
		t.Fatal("EnsureBuildDirs unexpectedly accepted a non-directory GOTMPDIR")
	}
	if detail := err.Error(); !strings.Contains(detail, "create GOTMPDIR directory") || !strings.Contains(detail, gotmp) {
		t.Fatalf("error %q does not identify the failed variable and path", detail)
	}
}

func TestEnsureBuildDirsDoesNotResurrectMissingWorktree(t *testing.T) {
	wt := filepath.Join(t.TempDir(), "reaped-worker")
	_, err := EnsureBuildDirs(wt)
	if err == nil || !strings.Contains(err.Error(), "stat managed worktree") {
		t.Fatalf("missing worktree error = %v, want actionable stat failure", err)
	}
	if _, statErr := os.Stat(wt); !os.IsNotExist(statErr) {
		t.Fatalf("missing worktree was recreated, stat err=%v", statErr)
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

type reapProofFixture struct {
	repo string
	wt   string
	base string
}

func newReapProofFixture(t *testing.T) reapProofFixture {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	repo := t.TempDir()
	reapProofGit(t, repo, "init", "-q", "-b", "main")
	reapProofGit(t, repo, "config", "user.email", "reap-proof@test")
	reapProofGit(t, repo, "config", "user.name", "reap proof")
	reapProofGit(t, repo, "config", "commit.gpgsign", "false")
	reapProofGit(t, repo, "config", "core.filemode", "true")
	if err := os.WriteFile(filepath.Join(repo, "target.txt"), []byte("base\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(repo, "peer.txt"), []byte("peer base\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	reapProofGit(t, repo, "add", "target.txt", "peer.txt")
	reapProofGit(t, repo, "commit", "-q", "-m", "base")
	base := strings.TrimSpace(reapProofGit(t, repo, "rev-parse", "HEAD"))
	prepared := Prepare(repo, "workerworktree", "9244", base, t.TempDir(), nil)
	if !prepared.OK {
		t.Fatalf("prepare fixture worktree: %+v", prepared)
	}
	return reapProofFixture{repo: repo, wt: prepared.Path, base: base}
}

func reapProofGit(t *testing.T, dir string, args ...string) string {
	t.Helper()
	cmd := exec.Command("git", args...)
	cmd.Dir = dir
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("git %s: %v\n%s", strings.Join(args, " "), err, out)
	}
	return string(out)
}

func writeReapProofFile(t *testing.T, dir, name, content string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func commitReapProofTrunk(t *testing.T, f reapProofFixture) string {
	t.Helper()
	reapProofGit(t, f.repo, "add", "-A")
	reapProofGit(t, f.repo, "commit", "-q", "-m", "supersession")
	return strings.TrimSpace(reapProofGit(t, f.repo, "rev-parse", "HEAD"))
}

func assertReapProofRefused(t *testing.T, f reapProofFixture, supersededBy string) {
	t.Helper()
	res := ReapChecked(f.repo, f.wt, supersededBy, nil)
	if res.OK || res.Code != ReapCodeProofRefused || !res.Preserved {
		t.Fatalf("unsafe reap proof must be refused and preserved: %+v", res)
	}
	if _, err := os.Stat(f.wt); err != nil {
		t.Fatalf("refused worktree was not preserved: %v", err)
	}
}

func TestReapCheckedAcceptsOnlyByteEquivalentDirtyPaths(t *testing.T) {
	t.Run("matching dirty paths ignore unrelated trunk changes", func(t *testing.T) {
		f := newReapProofFixture(t)
		writeReapProofFile(t, f.wt, "target.txt", "landed\n")
		writeReapProofFile(t, f.repo, "target.txt", "landed\n")
		writeReapProofFile(t, f.repo, "peer.txt", "peer advanced on trunk\n")
		supersededBy := commitReapProofTrunk(t, f)

		res := ReapChecked(f.repo, f.wt, supersededBy, nil)
		if !res.OK || !res.Removed || res.Code != ReapCodeVerifiedWorktreeReaped {
			t.Fatalf("byte-equivalent dirty paths should reap: %+v", res)
		}
		if res.SupersededBy != supersededBy {
			t.Fatalf("supersession = %q, want %q", res.SupersededBy, supersededBy)
		}
	})

	t.Run("content mismatch", func(t *testing.T) {
		f := newReapProofFixture(t)
		writeReapProofFile(t, f.wt, "target.txt", "worker bytes\n")
		writeReapProofFile(t, f.repo, "target.txt", "landed bytes\n")
		assertReapProofRefused(t, f, commitReapProofTrunk(t, f))
	})

	t.Run("mode mismatch", func(t *testing.T) {
		f := newReapProofFixture(t)
		writeReapProofFile(t, f.wt, "target.txt", "landed\n")
		writeReapProofFile(t, f.repo, "target.txt", "landed\n")
		reapProofGit(t, f.repo, "add", "target.txt")
		reapProofGit(t, f.repo, "update-index", "--chmod=+x", "target.txt")
		reapProofGit(t, f.repo, "commit", "-q", "-m", "supersession")
		supersededBy := strings.TrimSpace(reapProofGit(t, f.repo, "rev-parse", "HEAD"))
		assertReapProofRefused(t, f, supersededBy)
	})

	t.Run("symlink mismatch", func(t *testing.T) {
		f := newReapProofFixture(t)
		writeReapProofFile(t, f.wt, "target.txt", "destination")
		if err := os.Remove(filepath.Join(f.repo, "target.txt")); err != nil {
			t.Fatal(err)
		}
		if err := os.Symlink("destination", filepath.Join(f.repo, "target.txt")); err != nil {
			t.Skipf("symlinks unavailable: %v", err)
		}
		assertReapProofRefused(t, f, commitReapProofTrunk(t, f))
	})

	t.Run("missing tracked file", func(t *testing.T) {
		f := newReapProofFixture(t)
		if err := os.Remove(filepath.Join(f.wt, "target.txt")); err != nil {
			t.Fatal(err)
		}
		writeReapProofFile(t, f.repo, "target.txt", "landed\n")
		assertReapProofRefused(t, f, commitReapProofTrunk(t, f))
	})

	t.Run("untracked path", func(t *testing.T) {
		f := newReapProofFixture(t)
		writeReapProofFile(t, f.wt, "new.txt", "landed\n")
		writeReapProofFile(t, f.repo, "new.txt", "landed\n")
		assertReapProofRefused(t, f, commitReapProofTrunk(t, f))
	})

	t.Run("missing commit", func(t *testing.T) {
		f := newReapProofFixture(t)
		writeReapProofFile(t, f.wt, "target.txt", "landed\n")
		assertReapProofRefused(t, f, "0000000000000000000000000000000000000000")
	})

	t.Run("commit outside trunk ancestry", func(t *testing.T) {
		f := newReapProofFixture(t)
		writeReapProofFile(t, f.wt, "target.txt", "side bytes\n")
		writeReapProofFile(t, f.repo, "target.txt", "side bytes\n")
		reapProofGit(t, f.repo, "add", "target.txt")
		tree := strings.TrimSpace(reapProofGit(t, f.repo, "write-tree"))
		side := strings.TrimSpace(reapProofGit(t, f.repo, "commit-tree", tree, "-p", f.base, "-m", "side"))
		reapProofGit(t, f.repo, "reset", "--hard", "-q", f.base)
		assertReapProofRefused(t, f, side)
	})
}

// ---- Land ----------------------------------------------------------------- //

func TestLandLandsDiffOntoTrunkByPathSignedOff(t *testing.T) {
	// Exercises the shared-index baseline apply+commit mechanics; force both #3619
	// safety gates off (default-ON since #3619) so this fake need not stub them.
	t.Setenv(IsolatedLandEnv, "0")
	t.Setenv(LandReadbackEnv, "0")
	g := replyLandDiff(newFakeGit(), "x\n", "diff --git a/x b/x\n@@\n-old\n+new\n", "x\n").
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
	g := replyLandDiff(newFakeGit(), "x\n", "diff --git a/x b/x\n@@\n-old\n+new\n", "x\n").
		reply("apply", 0, "").
		reply("commit", 0, "[main abc] msg")
	res := Land("/trunk", "/wt/fak-worker-wt-tools-abc", "feedface", "/tmp/msg.txt", []string{"x"}, nil, g.run)
	if !res.Committed {
		t.Fatalf("land should commit: %+v", res)
	}
	diffs := g.callsWithPrefix("diff")
	var contentDiffs [][]string
	for _, call := range diffs {
		if !contains(call, "--name-only") {
			contentDiffs = append(contentDiffs, call)
		}
	}
	if len(contentDiffs) != 1 {
		t.Fatalf("want one content diff call, got %v", g.calls)
	}
	if !contains(contentDiffs[0], "feedface") || contains(contentDiffs[0], "HEAD") {
		t.Fatalf("diff base must be the pinned sha, not HEAD: %v", contentDiffs[0])
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
	g := replyLandDiff(newFakeGit(), "x\n", "diff --git a/x b/x\n@@\n-o\n+n\n", "x\n").
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
	g := replyLandDiff(newFakeGit(), "x\n", "diff --git a/x b/x\n@@\n-old\n+new\n", "x\n").
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
	g := replyLandDiff(newFakeGit(), "x\n", "diff --git a/x b/x\n@@\n-old\n+new\n", "x\n").
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

func TestLandReadbackForcedOffLeavesBaselineUnchanged(t *testing.T) {
	t.Setenv(LandReadbackEnv, "0") // explicit off — baseline path
	// A diff-tree that WOULD fail the check must never be consulted when off.
	g := replyLandDiff(newFakeGit(), "x\n", "diff --git a/x b/x\n@@\n-old\n+new\n", "x\n").
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
	g := replyLandDiff(newFakeGit(), "x\n", "diff --git a/x b/x\n@@\n-old\n+new\n", "x\n").
		reply("apply", 0, "").
		reply("commit", 0, "[main abc] msg").
		reply("rev-parse", 127, "")
	res := Land("/trunk", "/wt/fak-worker-wt-tools-abc", "feedface", "/tmp/msg.txt", []string{"x"}, nil, g.run)
	if !res.OK || !res.Committed {
		t.Fatalf("readback must FAIL OPEN on a git error, never refuse: %+v", res)
	}
}

// ---- Land layer 2: isolated-index (#3547) --------------------------------- //

// writeMsg materializes a real commit-message file; the isolated path reads it for
// real (composeSignedMsg), unlike the baseline which lets the fake stub `commit -F`.
func writeMsg(t *testing.T, subject string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "msg.txt")
	if err := os.WriteFile(p, []byte(subject+"\n"), 0o644); err != nil {
		t.Fatalf("write msg: %v", err)
	}
	return p
}

// fakeGit stubbing every step of the isolated-land happy path.
func isolatedHappyFake() *fakeGit {
	return newFakeGit().
		reply("symbolic-ref", 0, "refs/heads/main\n").
		reply("rev-parse", 0, "oldhead000\n").
		reply("config", 0, "fak test\n").
		reply("read-tree", 0, "").
		reply("apply", 0, "").
		reply("write-tree", 0, "treeSHA111\n").
		reply("commit-tree", 0, "newc2223334445556667778889990001112223334\n").
		reply("update-ref", 0, "").
		reply("checkout", 0, "")
}

func TestLandIsolatedHappyPathUsesTempIndexAndCASRefUpdate(t *testing.T) {
	g := isolatedHappyFake()
	msg := writeMsg(t, "feat(x): do the thing (fak x)")
	res, handled := landIsolated("/trunk", "/wt", "diff --git a/x b/x\n@@\n-o\n+n\n", msg, []string{"x"}, g.run, g.runEnv)
	if !handled || !res.OK || !res.Committed || !res.Applied {
		t.Fatalf("isolated land should succeed and be handled: handled=%v res=%+v", handled, res)
	}
	if !strings.Contains(res.Reason, "isolated-index") || !strings.Contains(res.Reason, "#3547") {
		t.Fatalf("reason should name the isolated fix: %q", res.Reason)
	}
	// Staging went to a THROWAWAY index, not the shared one.
	if v := g.lastEnv["GIT_INDEX_FILE"]; v == "" {
		t.Fatalf("isolated path must set GIT_INDEX_FILE, env=%v", g.lastEnv)
	}
	// Index seeded from the captured HEAD; the diff staged with --cached (never the worktree).
	if rt := g.envCallsWithPrefix("read-tree", "oldhead000"); len(rt) != 1 {
		t.Fatalf("want one read-tree of captured HEAD, env=%v", g.envCalls)
	}
	if ap := g.envCallsWithPrefix("apply", "--cached"); len(ap) != 1 {
		t.Fatalf("diff must stage with --cached into the temp index, env=%v", g.envCalls)
	}
	// Commit parents the EXACT captured HEAD (stale-base guard).
	ct := g.envCallsWithPrefix("commit-tree")
	if len(ct) != 1 || !contains(ct[0], "-p") || !contains(ct[0], "oldhead000") {
		t.Fatalf("commit-tree must parent captured HEAD: %v", g.envCalls)
	}
	// Branch advanced by a compare-and-swap: new AND old value both pinned.
	ur := g.callsWithPrefix("update-ref", "refs/heads/main")
	if len(ur) != 1 || len(ur[0]) != 4 || !strings.HasPrefix(ur[0][2], "newc222") || ur[0][3] != "oldhead000" {
		t.Fatalf("update-ref must CAS new-over-old-HEAD: %v", g.calls)
	}
	// Shared working tree synced for the landed paths so trunk builders see the change.
	co := g.callsWithPrefix("checkout")
	if len(co) != 1 || !contains(co[0], "x") || !contains(co[0], "--") {
		t.Fatalf("want a path-scoped working-tree sync checkout: %v", g.calls)
	}
	// Detail records the CAS attempts used even on a first-try win (#3570).
	if !strings.Contains(res.Detail, "cas-attempts=1") {
		t.Fatalf("Detail must record attempts used, got %q", res.Detail)
	}
}

func TestLandIsolatedPostMergeVerificationFailureRefusesCAS(t *testing.T) {
	g := newFakeGit().
		reply("symbolic-ref", 0, "refs/heads/main\n").
		reply("rev-parse", 0, "oldhead000\n").
		reply("config", 0, "fak test\n").
		reply("read-tree", 0, "").
		reply("apply", 0, "").
		reply("write-tree", 0, "treeSHA111\n").
		reply("commit-tree", 0, "newc2223334445556667778889990001112223334\n").
		reply("checkout", 0, "") // checkout --detach <newCommit>
	msg := writeMsg(t, "feat(x): do the thing (fak x)")

	verifyCalled := false
	var verifiedPath string
	failingVerify := func(wtPath string) (bool, string) {
		verifyCalled = true
		verifiedPath = wtPath
		return false, "syntax error in merged code"
	}

	res, handled := landIsolated("/trunk", "/wt", "diff --git a/x b/x\n@@\n-o\n+n\n", msg, []string{"x"}, failingVerify, g.run, g.runEnv)
	if !handled {
		t.Fatalf("expected handled=true, got %v", handled)
	}
	if res.OK {
		t.Fatalf("expected res.OK=false, got %+v", res)
	}
	if !verifyCalled || verifiedPath != "/wt" {
		t.Fatalf("expected verify to be called on /wt, called=%v path=%q", verifyCalled, verifiedPath)
	}
	expectedReason := "post-merge compilation verification failed, refusing CAS update: syntax error in merged code"
	if res.Reason != expectedReason {
		t.Fatalf("expected reason %q, got %q", expectedReason, res.Reason)
	}

	// CAS update must be refused and update-ref must never be called.
	ur := g.callsWithPrefix("update-ref", "refs/heads/main")
	if len(ur) != 0 {
		t.Fatalf("expected update-ref never to be called, got: %v", ur)
	}

	// Confirm checkout --detach was called on wtPath with newCommit.
	co := g.callsWithPrefix("checkout", "--detach")
	if len(co) != 1 || len(co[0]) < 3 || !strings.HasPrefix(co[0][2], "newc222") {
		t.Fatalf("expected checkout --detach <newCommit>, got %v", co)
	}
}

func TestLandIsolatedPostMergeVerificationSuccessProceedsWithCAS(t *testing.T) {
	g := newFakeGit().
		reply("symbolic-ref", 0, "refs/heads/main\n").
		reply("rev-parse", 0, "oldhead000\n").
		reply("config", 0, "fak test\n").
		reply("read-tree", 0, "").
		reply("apply", 0, "").
		reply("write-tree", 0, "treeSHA111\n").
		reply("commit-tree", 0, "newc2223334445556667778889990001112223334\n").
		reply("checkout", 0, ""). // checkout --detach <newCommit>
		reply("update-ref", 0, "").
		reply("checkout", 0, "") // checkout <newCommit> -- paths
	msg := writeMsg(t, "feat(x): do the thing (fak x)")

	verifyCalled := false
	passingVerify := func(wtPath string) (bool, string) {
		verifyCalled = true
		return true, ""
	}

	res, handled := landIsolated("/trunk", "/wt", "diff --git a/x b/x\n@@\n-o\n+n\n", msg, []string{"x"}, passingVerify, g.run, g.runEnv)
	if !handled || !res.OK {
		t.Fatalf("expected handled=true res.OK=true, got handled=%v res=%+v", handled, res)
	}
	if !verifyCalled {
		t.Fatalf("expected verify to be called")
	}
	ur := g.callsWithPrefix("update-ref", "refs/heads/main")
	if len(ur) != 1 {
		t.Fatalf("expected update-ref to be called once, got: %v", ur)
	}
}

func TestLandIsolatedDisambiguationRefusalPreservesStateAndWorkerDiff(t *testing.T) {
	g := isolatedHappyFake()
	oldRead := readDisambiguation
	defer func() { readDisambiguation = oldRead }()
	reads := 0
	readDisambiguation = stubDisambiguationReader(func(repo, tree string) DisambiguationWitness {
		reads++
		w := DisambiguationWitness{Tree: tree, Fresh: true, SemanticValid: true, CriticalClean: true, Coverage: 100, FamilyCoverage: map[string]float64{"loop": 100}}
		if reads == 3 {
			w.SemanticValid = false
			w.CriticalClean = false
			w.Detail = "post-apply duplicate grounding"
		}
		return w
	})
	res, handled := landIsolated("/repo", "/worker", "diff --git a/tools/concept_disambiguation_scorecard.data/rows-loop-family.json b/tools/concept_disambiguation_scorecard.data/rows-loop-family.json\n@@\n-old\n+new\n", writeMsg(t, "feat: x"), []string{"tools/concept_disambiguation_scorecard.data/rows-loop-family.json"}, g.run, g.runEnv)
	if !handled || res.OK || res.Committed {
		t.Fatalf("post-apply rejection must be handled before a durable commit: handled=%v result=%+v", handled, res)
	}
	if res.Disambiguation.PostApply.Detail == "" || res.Disambiguation.PostApply.SemanticValid {
		t.Fatalf("machine-readable refusal witness missing: %+v", res.Disambiguation)
	}
	if res.Disambiguation.Before.Tree == "" || res.Disambiguation.Worktree.Tree == "" || res.Disambiguation.PostApply.Tree == "" {
		t.Fatalf("all three refusal witnesses must be retained: %+v", res.Disambiguation)
	}
	countVerb := func(verb string) int {
		n := 0
		for _, call := range g.calls {
			if len(call) > 0 && call[0] == verb {
				n++
			}
		}
		return n
	}
	if countVerb("commit-tree") != 0 || countVerb("update-ref") != 0 || countVerb("reset") != 0 || countVerb("checkout") != 0 {
		t.Fatalf("refusal changed trunk/index/worker state: calls=%v", g.calls)
	}
}

func TestLandIsolatedApplyConflictFallsBackNotCommits(t *testing.T) {
	g := isolatedHappyFake().reply("apply", 1, "error: patch does not apply")
	res, handled := landIsolated("/trunk", "/wt", "diff --git a/x b/x\n@@\n-o\n+n\n", writeMsg(t, "s"), []string{"x"}, g.run, g.runEnv)
	if handled {
		t.Fatalf("an apply conflict must fall back (handled=false), got %+v", res)
	}
	if len(g.envCallsWithPrefix("commit-tree")) != 0 || len(g.callsWithPrefix("update-ref")) != 0 {
		t.Fatalf("conflict must not build/move a commit: env=%v calls=%v", g.envCalls, g.calls)
	}
}

// stubCASSleep silences the lost-CAS retry backoff for the duration of a test so
// retry tests assert sequencing, not wall-clock jitter.
func stubCASSleep(t *testing.T) {
	t.Helper()
	saved := casRetrySleep
	casRetrySleep = func(int) {}
	t.Cleanup(func() { casRetrySleep = saved })
}

func TestLandIsolatedLostCASRetriesReseedFromNewHEADAndLands(t *testing.T) {
	stubCASSleep(t)
	// A peer lands in the CAS gap: the first update-ref loses; HEAD then reads as
	// the peer's new tip and holds still, so the SECOND attempt must re-seed from
	// that new base and win — not fall back into the racy shared-index path (#3570).
	g := isolatedHappyFake().
		replyOnce("rev-parse", 0, "oldhead000\n").
		replyOnce("rev-parse", 0, "newhead111\n").
		replyOnce("update-ref", 1, "fatal: update_ref failed: ref moved").
		replyOnce("update-ref", 0, "")
	res, handled := landIsolated("/trunk", "/wt", "diff --git a/x b/x\n@@\n-o\n+n\n", writeMsg(t, "s"), []string{"x"}, g.run, g.runEnv)
	if !handled || !res.OK || !res.Committed || !res.Applied {
		t.Fatalf("retry after a lost CAS must land, not fall back: handled=%v res=%+v", handled, res)
	}
	// The retry re-seeded the throwaway index from the peer's NEW head…
	if len(g.envCallsWithPrefix("read-tree", "oldhead000")) != 1 || len(g.envCallsWithPrefix("read-tree", "newhead111")) != 1 {
		t.Fatalf("retry must re-seed the index from the re-resolved HEAD: %v", g.envCalls)
	}
	// …re-staged the SAME captured diff into it…
	if len(g.envCallsWithPrefix("apply", "--cached")) != 2 {
		t.Fatalf("each attempt must re-apply the captured diff --cached: %v", g.envCalls)
	}
	// …and re-built the commit as a child of that new head.
	ct := g.envCallsWithPrefix("commit-tree")
	if len(ct) != 2 || !contains(ct[1], "newhead111") {
		t.Fatalf("retry commit-tree must parent the new HEAD: %v", g.envCalls)
	}
	// The second CAS pinned the NEW head as old-value — still compare-and-swap, never force.
	ur := g.callsWithPrefix("update-ref", "refs/heads/main")
	if len(ur) != 2 || len(ur[1]) != 4 || ur[1][3] != "newhead111" {
		t.Fatalf("retry must CAS against the re-resolved HEAD: %v", g.calls)
	}
	// Detail records attempts used so the land record shows the contention.
	if !strings.Contains(res.Detail, "cas-attempts=2") {
		t.Fatalf("Detail must record the attempts used, got %q", res.Detail)
	}
	// Exactly one working-tree sync, for the winning commit only.
	if co := g.callsWithPrefix("checkout"); len(co) != 1 {
		t.Fatalf("want exactly one post-win worktree sync: %v", g.calls)
	}
}

func TestLandIsolatedRetryReapplyConflictFallsBack(t *testing.T) {
	stubCASSleep(t)
	// Attempt 1 stages clean but loses the CAS; the re-apply onto the peer's new
	// HEAD conflicts on the same hunk — a genuine overlap, so the baseline path
	// must adjudicate it (handled=false), exactly as a first-try conflict does.
	g := isolatedHappyFake().
		replyOnce("apply", 0, "").
		replyOnce("apply", 1, "error: patch does not apply").
		replyOnce("update-ref", 1, "fatal: update_ref failed: ref moved")
	res, handled := landIsolated("/trunk", "/wt", "diff --git a/x b/x\n@@\n-o\n+n\n", writeMsg(t, "s"), []string{"x"}, g.run, g.runEnv)
	if handled {
		t.Fatalf("a conflicting re-apply must fall back to the baseline, got %+v", res)
	}
	if len(g.envCallsWithPrefix("commit-tree")) != 1 || len(g.callsWithPrefix("update-ref", "refs/heads/main")) != 1 {
		t.Fatalf("the conflicted retry must not build/CAS a second commit: env=%v calls=%v", g.envCalls, g.calls)
	}
	if len(g.callsWithPrefix("checkout")) != 0 {
		t.Fatalf("no land happened — the shared working tree must stay untouched: %v", g.calls)
	}
}

func TestLandIsolatedLostCASRetryCapHonoredThenFallsBack(t *testing.T) {
	stubCASSleep(t)
	t.Setenv(IsolatedLandRetryEnv, "3")
	// A CAS that NEVER wins (HEAD keeps moving under us): the loop must be bounded
	// by the cap, then fall back — and never sync the shared working tree.
	g := isolatedHappyFake().reply("update-ref", 1, "fatal: update_ref failed: ref moved")
	res, handled := landIsolated("/trunk", "/wt", "diff --git a/x b/x\n@@\n-o\n+n\n", writeMsg(t, "s"), []string{"x"}, g.run, g.runEnv)
	if handled {
		t.Fatalf("exhausted CAS attempts must fall back to baseline, got %+v", res)
	}
	if n := len(g.callsWithPrefix("update-ref")); n != 3 {
		t.Fatalf("retry cap of 3 must yield exactly 3 CAS attempts, got %d: %v", n, g.calls)
	}
	if n := len(g.envCallsWithPrefix("read-tree")); n != 3 {
		t.Fatalf("each attempt must re-seed the throwaway index, want 3 seeds got %d: %v", n, g.envCalls)
	}
	if len(g.callsWithPrefix("checkout")) != 0 {
		t.Fatalf("a lost CAS must not touch the shared working tree: %v", g.calls)
	}
}

func TestIsolatedLandRetryCapDefaultsAndParses(t *testing.T) {
	// t.Setenv registers restoration of any ambient value; then truly unset so the
	// check reads the genuine absent-env default (same pattern as the #3619 pin).
	t.Setenv(IsolatedLandRetryEnv, "")
	os.Unsetenv(IsolatedLandRetryEnv)
	if got := isolatedLandRetryCap(); got != 5 {
		t.Fatalf("absent env must default to 5 attempts, got %d", got)
	}
	for _, bad := range []string{"banana", "0", "-2"} {
		t.Setenv(IsolatedLandRetryEnv, bad)
		if got := isolatedLandRetryCap(); got != 5 {
			t.Fatalf("invalid %q must fall back to the default cap, got %d", bad, got)
		}
	}
	t.Setenv(IsolatedLandRetryEnv, "1")
	if got := isolatedLandRetryCap(); got != 1 {
		t.Fatalf("cap 1 must mean a single attempt (retries off), got %d", got)
	}
}

func TestLandIsolatedDetachedHeadFallsBackImmediately(t *testing.T) {
	g := isolatedHappyFake().reply("symbolic-ref", 1, "") // detached HEAD
	_, handled := landIsolated("/trunk", "/wt", "diff --git a/x b/x\n@@\n-o\n+n\n", writeMsg(t, "s"), []string{"x"}, g.run, g.runEnv)
	if handled {
		t.Fatalf("detached HEAD has no branch ref to CAS — must fall back")
	}
	if len(g.envCalls) != 0 {
		t.Fatalf("must bail before any throwaway-index work: env=%v", g.envCalls)
	}
}

func TestLandIsolatedMissingIdentityFallsBack(t *testing.T) {
	g := isolatedHappyFake().reply("config", 0, "") // no user.name/email → can't honor -s
	_, handled := landIsolated("/trunk", "/wt", "diff --git a/x b/x\n@@\n-o\n+n\n", writeMsg(t, "s"), []string{"x"}, g.run, g.runEnv)
	if handled {
		t.Fatalf("unresolved signoff identity must fall back to baseline")
	}
	if len(g.envCalls) != 0 {
		t.Fatalf("must bail before touching the throwaway index: env=%v", g.envCalls)
	}
}

func TestLandIsolatedForcedOffLeavesBaselineUnchanged(t *testing.T) {
	// The env is the operator escape hatch back to the shared-index baseline now
	// that #3619 flipped both gates default-ON; force both off here.
	t.Setenv(IsolatedLandEnv, "0")
	t.Setenv(LandReadbackEnv, "0")
	g := replyLandDiff(newFakeGit(), "x\n", "diff --git a/x b/x\n@@\n-old\n+new\n", "x\n").
		reply("apply", 0, "").
		reply("commit", 0, "[main abc] msg")
	res := Land("/trunk", "/wt/fak-worker-wt-tools-abc", "feedface", "/tmp/msg.txt", []string{"x"}, nil, g.run)
	if !res.OK || !res.Committed {
		t.Fatalf("baseline land must still work with the gate off: %+v", res)
	}
	if len(g.callsWithPrefix("symbolic-ref")) != 0 || len(g.callsWithPrefix("commit-tree")) != 0 {
		t.Fatalf("gate OFF must never enter the isolated path: %v", g.calls)
	}
	if len(g.callsWithPrefix("commit")) != 1 {
		t.Fatalf("baseline path-scoped commit expected: %v", g.calls)
	}
}

func TestLandIsolatedGateOnRoutesLandThroughIsolatedPath(t *testing.T) {
	t.Setenv(IsolatedLandEnv, "1")
	g := replyLandDiff(isolatedHappyFake(), "x\n", "diff --git a/x b/x\n@@\n-o\n+n\n", "x\n")
	// Inject the fake env-runner into the package seam Land's isolated path uses.
	saved := isolatedGitEnv
	isolatedGitEnv = g.runEnv
	defer func() { isolatedGitEnv = saved }()

	res := Land("/trunk", "/wt/fak-worker-wt-tools-abc", "base", writeMsg(t, "feat(x): thing (fak x)"), []string{"x"}, nil, g.run)
	if !res.OK || !strings.Contains(res.Reason, "isolated-index") {
		t.Fatalf("gate ON must land via the isolated path: %+v", res)
	}
	// The baseline shared-index commit must NOT have run.
	if len(g.callsWithPrefix("commit")) != 0 || len(g.callsWithPrefix("apply")) != 0 {
		t.Fatalf("isolated success must skip the baseline apply+commit: %v", g.calls)
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

func TestLandReportsDroppedOutOfLanePaths(t *testing.T) {
	t.Setenv(IsolatedLandEnv, "0")
	t.Setenv(LandReadbackEnv, "0")
	g := newFakeGit()
	g = replyLandDiff(g, "internal/tools/a.go\n", "patch", "internal/tools/a.go\ndocs/escaped.md\n")
	g.reply("log", 0, "feat(tools): witness lane (fak tools)")
	g.reply("apply", 0, "")
	g.reply("commit", 0, "committed")
	res := Land("/trunk", "/wt/fak-worker-wt-tools-abc", "feedface", "", []string{"internal/tools"}, nil, g.run)
	if !res.OK || !res.Committed {
		t.Fatalf("Land() = %+v", res)
	}
	if res.DroppedOutOfLane != 1 {
		t.Fatalf("DroppedOutOfLane = %d, want 1", res.DroppedOutOfLane)
	}
}

func TestPrepareOwnedBoundedCreatesVerifiedCleanWorktree(t *testing.T) {
	fixture := newReapProofFixture(t)
	wtRoot := t.TempDir()
	res := PrepareOwnedBounded(fixture.repo, "test", "ready", fixture.base, wtRoot, OwnerStamp{PID: os.Getpid(), LeaseID: "lease-ready"}, 10*time.Second)
	if !res.OK || res.Code != "" || res.BaseSHA != fixture.base {
		t.Fatalf("prepare result = %+v", res)
	}
	if got := strings.TrimSpace(reapProofGit(t, res.Path, "rev-parse", "HEAD")); got != fixture.base {
		t.Fatalf("HEAD = %q, want %q", got, fixture.base)
	}
	if got := strings.TrimSpace(reapProofGit(t, res.Path, "status", "--porcelain")); got != "" {
		t.Fatalf("prepared worktree is dirty: %q", got)
	}
	stamp, err := readOwnerStamp(res.Path)
	if err != nil || stamp.LeaseID != "lease-ready" {
		t.Fatalf("owner stamp = %+v, %v", stamp, err)
	}
}

func TestPrepareOwnedBoundedTimeoutEmitsNoOwnerReceipt(t *testing.T) {
	root := t.TempDir()
	wtRoot := t.TempDir()
	base := strings.Repeat("a", 40)
	cancelled := false
	calls := 0
	var cleanupCalls []string
	git := func(root string, rawArgs []string) (int, string) {
		calls++
		args := stripGlobalFlags(rawArgs)
		if len(args) > 1 && args[0] == "rev-parse" {
			return 0, base + "\n"
		}
		if len(args) > 1 && args[0] == "worktree" && args[1] == "add" {
			cancelled = true
			return ReapTimeoutExitCode, context.Canceled.Error()
		}
		if len(args) > 1 && args[0] == "worktree" && (args[1] == "remove" || args[1] == "prune") {
			cleanupCalls = append(cleanupCalls, strings.Join(rawArgs, " "))
		}
		return 0, ""
	}
	res := prepareOwnedWithBackend(root, "test", "timeout", base, wtRoot, git, defaultIsolationBackend, OwnerStamp{PID: 42, LeaseID: "lease"}, true)
	if res.OK || res.Code != "PREPARE_TIMEOUT" || !strings.Contains(res.Reason, "no ready receipt") {
		t.Fatalf("timeout result = %+v", res)
	}
	if calls == 0 || !cancelled {
		t.Fatal("git runner was not called")
	}
	if _, err := os.Stat(OwnerStampPath(res.Path)); !os.IsNotExist(err) {
		t.Fatalf("owner receipt exists after timeout: %v", err)
	}
	wantCleanup := []string{"worktree remove --force " + res.Path, "worktree prune"}
	if strings.Join(cleanupCalls, "\n") != strings.Join(wantCleanup, "\n") {
		t.Fatalf("cleanup calls = %q, want %q", cleanupCalls, wantCleanup)
	}
}

func TestPreparePreservesDirtyReusedWorktree(t *testing.T) {
	t.Run("preserves_dirty_reused_checkout", func(t *testing.T) {
		fixture := newReapProofFixture(t)
		wtRoot := t.TempDir()
		lane := "reused-lane"
		key := "dirty-worker"
		owner := OwnerStamp{PID: os.Getpid(), LeaseID: "lease-dirty-worker"}

		// First prepare: creates clean worktree
		res1 := PrepareOwned(fixture.repo, lane, key, fixture.base, wtRoot, defaultGit, owner)
		if !res1.OK || res1.Reused {
			t.Fatalf("first prepare = %+v", res1)
		}

		// Add tracked modifications
		trackedFile := filepath.Join(res1.Path, "target.txt")
		if err := os.WriteFile(trackedFile, []byte("dirty tracked content\n"), 0o644); err != nil {
			t.Fatal(err)
		}

		// Add untracked file
		untrackedFile := filepath.Join(res1.Path, "new-untracked.txt")
		if err := os.WriteFile(untrackedFile, []byte("untracked content\n"), 0o644); err != nil {
			t.Fatal(err)
		}

		// Second prepare on same key with verifyReady=true (PrepareOwnedBounded verifies readiness)
		res2 := prepareOwnedWithBackend(fixture.repo, lane, key, fixture.base, wtRoot, defaultGit, defaultIsolationBackend, owner, true)
		if res2.OK {
			t.Fatalf("expected readiness refusal for dirty reused worktree, got %+v", res2)
		}
		if res2.Code != "PREPARE_NOT_READY" {
			t.Fatalf("expected Code PREPARE_NOT_READY, got %q", res2.Code)
		}
		if !res2.Reused {
			t.Fatalf("expected Reused=true for same-key prepare, got %+v", res2)
		}

		// Verify existing worktree files remain intact
		trackedData, err := os.ReadFile(trackedFile)
		if err != nil || string(trackedData) != "dirty tracked content\n" {
			t.Fatalf("expected tracked file to be preserved, got %q, err %v", string(trackedData), err)
		}
		untrackedData, err := os.ReadFile(untrackedFile)
		if err != nil || string(untrackedData) != "untracked content\n" {
			t.Fatalf("expected untracked file to be preserved, got %q, err %v", string(untrackedData), err)
		}

		// Verify git administrative registration remains intact (not pruned/removed)
		wtList := reapProofGit(t, fixture.repo, "worktree", "list", "--porcelain")
		found := false
		for _, p := range parseWorktreePaths(wtList) {
			if samePath(p, res1.Path) {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("expected worktree %s to remain registered in git worktree list", res1.Path)
		}
	})

	t.Run("cleans_up_new_unready_worktree", func(t *testing.T) {
		fixture := newReapProofFixture(t)
		wtRoot := t.TempDir()
		lane := "new-lane"
		key := "unready-new-worker"
		owner := OwnerStamp{PID: os.Getpid(), LeaseID: "lease-new-worker"}
		expectedPath := Path(lane, key, wtRoot)

		// Wrap git so that status --porcelain reports dirty during readiness verification of a newly created worktree
		wrappedGit := func(dir string, args []string) (int, string) {
			joined := strings.Join(args, " ")
			if joined == "status --porcelain" && samePath(dir, expectedPath) {
				return 0, " M dirty.go\n"
			}
			return defaultGit(dir, args)
		}

		res := prepareOwnedWithBackend(fixture.repo, lane, key, fixture.base, wtRoot, wrappedGit, defaultIsolationBackend, owner, true)
		if res.OK || res.Code != "PREPARE_NOT_READY" || res.Reused {
			t.Fatalf("expected PREPARE_NOT_READY for unready new worktree, got %+v", res)
		}

		// Verify that cleanup occurred for newly created partial worktree (!res.Reused)
		if _, err := os.Stat(expectedPath); !os.IsNotExist(err) {
			t.Fatalf("expected new worktree at %s to be cleaned up, stat err: %v", expectedPath, err)
		}
		wtList := reapProofGit(t, fixture.repo, "worktree", "list", "--porcelain")
		for _, p := range parseWorktreePaths(wtList) {
			if samePath(p, expectedPath) {
				t.Fatalf("expected worktree %s to be removed from git administrative registration", expectedPath)
			}
		}
	})

	t.Run("preserves_reused_checkout_on_timeout", func(t *testing.T) {
		fixture := newReapProofFixture(t)
		wtRoot := t.TempDir()
		lane := "reused-timeout-lane"
		key := "timeout-worker"
		owner := OwnerStamp{PID: os.Getpid(), LeaseID: "lease-timeout-worker"}

		// First prepare: creates clean worktree
		res1 := PrepareOwnedBounded(fixture.repo, lane, key, fixture.base, wtRoot, owner, 10*time.Second)
		if !res1.OK || res1.Reused {
			t.Fatalf("first prepare = %+v", res1)
		}

		keepFile := filepath.Join(res1.Path, "preserve-me.txt")
		if err := os.WriteFile(keepFile, []byte("timeout preserved content\n"), 0o644); err != nil {
			t.Fatal(err)
		}

		// Wrap git: allow worktree list so MaterializeOwned detects reuse, but return ReapTimeoutExitCode on status --porcelain
		wrappedGit := func(dir string, args []string) (int, string) {
			joined := strings.Join(args, " ")
			if joined == "status --porcelain" && samePath(dir, res1.Path) {
				return ReapTimeoutExitCode, "timeout"
			}
			return defaultGit(dir, args)
		}

		res2 := prepareOwnedWithBackend(fixture.repo, lane, key, fixture.base, wtRoot, wrappedGit, defaultIsolationBackend, owner, true)
		if res2.OK {
			t.Fatalf("expected timeout refusal, got %+v", res2)
		}
		if res2.Code != "PREPARE_TIMEOUT" {
			t.Fatalf("expected Code PREPARE_TIMEOUT, got %q", res2.Code)
		}
		if !res2.Reused {
			t.Fatalf("expected Reused=true for same-key prepare, got %+v", res2)
		}

		// Verify existing worktree files remain intact
		keepData, err := os.ReadFile(keepFile)
		if err != nil || string(keepData) != "timeout preserved content\n" {
			t.Fatalf("expected file to be preserved on timeout, got %q, err %v", string(keepData), err)
		}

		// Verify git administrative registration remains intact
		wtList := reapProofGit(t, fixture.repo, "worktree", "list", "--porcelain")
		found := false
		for _, p := range parseWorktreePaths(wtList) {
			if samePath(p, res1.Path) {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("expected worktree %s to remain registered in git worktree list", res1.Path)
		}
	})
}

func TestVerifyPreparedWorktreeRejectsDirtyAndLockedCheckout(t *testing.T) {
	base := strings.Repeat("b", 40)
	wt := filepath.Join(t.TempDir(), "fak-worker-wt-verify")
	if err := os.MkdirAll(filepath.Join(wt, ".gitdir"), 0o755); err != nil {
		t.Fatal(err)
	}
	git := func(root string, args []string) (int, string) {
		joined := strings.Join(args, " ")
		switch joined {
		case "rev-parse HEAD":
			return 0, base + "\n"
		case "status --porcelain":
			return 0, " M partial.go\n"
		case "rev-parse --git-dir":
			return 0, filepath.Join(wt, ".gitdir")
		}
		return 1, "unexpected"
	}
	res := verifyPreparedWorktree(Result{OK: true, Path: wt, BaseSHA: base}, git)
	if res.OK || res.Code != "PREPARE_NOT_READY" || !strings.Contains(res.Reason, "not clean") {
		t.Fatalf("dirty result = %+v", res)
	}

	wrongHead := strings.Repeat("c", 40)
	git = func(root string, args []string) (int, string) {
		if strings.Join(args, " ") == "rev-parse HEAD" {
			return 0, wrongHead + "\n"
		}
		return 1, "unexpected"
	}
	res = verifyPreparedWorktree(Result{OK: true, Path: wt, BaseSHA: base}, git)
	if res.OK || res.Code != "PREPARE_NOT_READY" || !strings.Contains(res.Reason, "does not match") {
		t.Fatalf("wrong-head result = %+v", res)
	}

	git = func(root string, args []string) (int, string) {
		switch strings.Join(args, " ") {
		case "rev-parse HEAD":
			return 0, base + "\n"
		case "status --porcelain":
			return 0, ""
		case "rev-parse --git-dir":
			return 0, filepath.Join(wt, ".gitdir")
		}
		return 1, "unexpected"
	}
	lock := filepath.Join(wt, ".gitdir", "index.lock")
	if err := os.WriteFile(lock, nil, 0o600); err != nil {
		t.Fatal(err)
	}
	res = verifyPreparedWorktree(Result{OK: true, Path: wt, BaseSHA: base}, git)
	if res.OK || !strings.Contains(res.Reason, "index lock") {
		t.Fatalf("lock result = %+v", res)
	}
}

func TestBoundedGitRunnerCancelsBlockedCommand(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Nanosecond)
	defer cancel()
	time.Sleep(time.Millisecond)
	rc, out := BoundedGitRunner(ctx)(t.TempDir(), []string{"status"})
	if rc != ReapTimeoutExitCode || !strings.Contains(out, "deadline") {
		t.Fatalf("bounded runner = (%d, %q)", rc, out)
	}
}

func TestPrepare_WindowsIndexResetSelfRecoveringAndAtomic(t *testing.T) {
	t.Run("recovers from index reset failure", func(t *testing.T) {
		root := t.TempDir()
		wtRoot := t.TempDir()
		base := "0123456789abcdef0123456789abcdef01234567"
		lane := "tools"
		key := "recovery-test"
		expectedWT := Path(lane, key, wtRoot)

		var cleanupCalls []string
		var decoupledAttempted bool
		var checkoutCalled bool
		var resetCalled bool

		git := func(dir string, args []string) (int, string) {
			joined := strings.Join(args, " ")

			// 1. First worktree add simulates index reset failure on Windows
			if strings.Contains(joined, "worktree add") && !strings.Contains(joined, "--no-checkout") {
				// Simulate partial directory creation by git before failure
				_ = os.MkdirAll(expectedWT, 0o755)
				return 1, "fatal: Could not reset index file to revision 'HEAD'"
			}

			// 2. Immediate cleanup via ForceReap
			if strings.Contains(joined, "worktree remove") || strings.Contains(joined, "worktree prune") {
				cleanupCalls = append(cleanupCalls, joined)
				return 0, ""
			}

			// 3. Windows-safe decoupled retry
			if strings.Contains(joined, "worktree add") && strings.Contains(joined, "--no-checkout") {
				decoupledAttempted = true
				if !strings.Contains(joined, "-c core.longpaths=true") {
					t.Errorf("decoupled retry missing -c core.longpaths=true: %s", joined)
				}
				// Simulate git creating directory on --no-checkout
				_ = os.MkdirAll(expectedWT, 0o755)
				return 0, ""
			}

			// 4. Decoupled checkout and reset
			if strings.Contains(joined, "checkout --force") {
				checkoutCalled = true
				if dir != expectedWT {
					t.Errorf("checkout working directory = %q, want %q", dir, expectedWT)
				}
				if !strings.Contains(joined, "-c core.longpaths=true") {
					t.Errorf("checkout missing -c core.longpaths=true: %s", joined)
				}
				return 0, ""
			}

			if strings.Contains(joined, "reset --hard") {
				resetCalled = true
				if dir != expectedWT {
					t.Errorf("reset working directory = %q, want %q", dir, expectedWT)
				}
				if !strings.Contains(joined, "-c core.longpaths=true") {
					t.Errorf("reset missing -c core.longpaths=true: %s", joined)
				}
				return 0, ""
			}

			return 0, ""
		}

		res := Prepare(root, lane, key, base, wtRoot, git)
		if !res.OK {
			t.Fatalf("expected Prepare to succeed via decoupled retry, got: %+v", res)
		}
		if len(cleanupCalls) == 0 {
			t.Errorf("expected cleanup (ForceReap) to occur after initial failure")
		}
		if !decoupledAttempted {
			t.Errorf("expected decoupled retry with --no-checkout")
		}
		if !checkoutCalled {
			t.Errorf("expected decoupled checkout --force")
		}
		if !resetCalled {
			t.Errorf("expected decoupled reset --hard")
		}
		if res.Path != expectedWT {
			t.Errorf("res.Path = %q, want %q", res.Path, expectedWT)
		}
	})

	t.Run("atomic cleanup when both fail", func(t *testing.T) {
		root := t.TempDir()
		wtRoot := t.TempDir()
		base := "0123456789abcdef0123456789abcdef01234567"
		lane := "tools"
		key := "fail-test"
		expectedWT := Path(lane, key, wtRoot)

		var forceReapCalls []string

		git := func(dir string, args []string) (int, string) {
			joined := strings.Join(args, " ")

			if strings.Contains(joined, "worktree add") {
				_ = os.MkdirAll(expectedWT, 0o755)
				if !strings.Contains(joined, "--no-checkout") {
					return 1, "fatal: Could not reset index file to revision 'HEAD'"
				}
				// Retry also fails
				return 1, "fatal: retry also failed"
			}

			if strings.Contains(joined, "worktree remove") || strings.Contains(joined, "worktree prune") {
				forceReapCalls = append(forceReapCalls, joined)
				return 0, ""
			}

			return 0, ""
		}

		res := Prepare(root, lane, key, base, wtRoot, git)
		if res.OK {
			t.Fatalf("expected Prepare to fail when both attempts fail, got: %+v", res)
		}
		if len(forceReapCalls) == 0 {
			t.Errorf("expected ForceReap to be called")
		}
		if _, err := os.Stat(expectedWT); !os.IsNotExist(err) {
			t.Errorf("worktree directory %s still exists; atomic cleanup failed", expectedWT)
		}
	})
}

func TestWorkerLandTypedTerminalResults(t *testing.T) {
	t.Run("no-op", func(t *testing.T) {
		g := newFakeGit().reply("diff", 0, "   \n")
		res := Land("/trunk", "/wt/fak-worker-wt-test-1", "base123", "/tmp/msg.txt", nil, nil, g.run)
		if !res.OK || res.Code != LandResultNoOp {
			t.Fatalf("want code %q and OK=true, got %+v", LandResultNoOp, res)
		}
	})

	t.Run("conflict", func(t *testing.T) {
		t.Setenv(IsolatedLandEnv, "0")
		t.Setenv(LandReadbackEnv, "0")
		g := newFakeGit().
			reply("diff", 0, "diff --git a/x b/x\n@@\n-old\n+new\n").
			reply("merge-base", 0, "").
			reply("apply", 1, "error: patch does not apply")
		res := Land("/trunk", "/wt/fak-worker-wt-test-1", "base123", "/tmp/msg.txt", nil, nil, g.run)
		if res.OK || res.Code != LandResultConflict {
			t.Fatalf("want code %q and OK=false, got %+v", LandResultConflict, res)
		}
	})

	t.Run("stale-base", func(t *testing.T) {
		g := newFakeGit().
			reply("diff", 0, "diff --git a/x b/x\n@@\n-old\n+new\n").
			reply("merge-base", 1, "")
		res := Land("/trunk", "/wt/fak-worker-wt-test-1", "stalebase123", "/tmp/msg.txt", nil, nil, g.run)
		if res.OK || res.Code != LandResultStaleBase {
			t.Fatalf("want code %q and OK=false, got %+v", LandResultStaleBase, res)
		}
	})

	t.Run("success", func(t *testing.T) {
		t.Setenv(IsolatedLandEnv, "0")
		t.Setenv(LandReadbackEnv, "0")
		g := replyLandDiff(newFakeGit(), "x\n", "diff --git a/x b/x\n@@\n-old\n+new\n", "x\n").
			reply("merge-base", 0, "").
			reply("apply", 0, "").
			reply("commit", 0, "[main abc] msg")
		res := Land("/trunk", "/wt/fak-worker-wt-test-1", "base123", "/tmp/msg.txt", []string{"x"}, nil, g.run)
		if !res.OK || !res.Committed || res.Code != LandResultSuccess {
			t.Fatalf("want code %q and OK=true, got %+v", LandResultSuccess, res)
		}
	})
}

func TestLandDoesNotInheritUnrelatedCommitSubject(t *testing.T) {
	t.Setenv(IsolatedLandEnv, "0")
	t.Setenv(LandReadbackEnv, "0")
	// When HEAD == baseSHA (worker made no commit), Land must NOT run git log -1
	// to borrow base commit message from shared trunk.
	g := replyLandDiff(newFakeGit(), "x\n", "diff --git a/x b/x\n@@\n-old\n+new\n", "x\n").
		reply("merge-base", 0, "").
		reply("rev-parse", 0, "base123\n").
		reply("apply", 0, "").
		reply("commit", 0, "[main abc] msg")
	res := Land("/trunk", "/wt/fak-worker-wt-test-1", "base123", "", []string{"x"}, nil, g.run)
	if !res.OK || !res.Committed {
		t.Fatalf("land failed: %+v", res)
	}
	if len(g.callsWithPrefix("log")) != 0 {
		t.Fatalf("must not call git log when HEAD == baseSHA (would inherit unrelated commit subject); calls=%v", g.calls)
	}
}

func TestMaterializeOwned_IndexLockContentionRetried(t *testing.T) {
	root := t.TempDir()
	wtRoot := t.TempDir()
	base := "0123456789abcdef0123456789abcdef01234567"
	lane := "tools"
	key := "retry-test"

	var addCalls int
	var decoupledAttempted bool

	git := func(dir string, args []string) (int, string) {
		joined := strings.Join(args, " ")
		if strings.Contains(joined, "worktree add") && !strings.Contains(joined, "--no-checkout") {
			addCalls++
			if addCalls == 1 {
				return 1, "fatal: Unable to create '.git/index.lock': File exists."
			}
			return 0, ""
		}
		if strings.Contains(joined, "--no-checkout") {
			decoupledAttempted = true
			return 0, ""
		}
		return 0, ""
	}

	res := gitWorktree{}.MaterializeOwned(root, lane, key, base, wtRoot, git, defaultOwnerStamp(lane))
	if !res.OK {
		t.Fatalf("expected MaterializeOwned to succeed on retry, got: %+v", res)
	}
	if addCalls != 2 {
		t.Fatalf("expected 2 worktree add attempts, got %d", addCalls)
	}
	if decoupledAttempted {
		t.Fatalf("did not expect decoupled --no-checkout fallback when retry succeeds")
	}
}

func TestVerifyPreparedWorktree_IndexLockTransientPoll(t *testing.T) {
	wt := t.TempDir()
	base := "0123456789abcdef0123456789abcdef01234567"
	gitDir := filepath.Join(wt, ".gitdir")
	if err := os.MkdirAll(gitDir, 0o755); err != nil {
		t.Fatal(err)
	}

	git := func(root string, args []string) (int, string) {
		switch strings.Join(args, " ") {
		case "rev-parse HEAD":
			return 0, base + "\n"
		case "status --porcelain":
			return 0, ""
		case "rev-parse --git-dir":
			return 0, gitDir
		}
		return 1, "unexpected"
	}

	lock := filepath.Join(gitDir, "index.lock")
	if err := os.WriteFile(lock, nil, 0o600); err != nil {
		t.Fatal(err)
	}

	// Release the lock after 60ms to simulate transient Windows file handle release
	go func() {
		time.Sleep(60 * time.Millisecond)
		_ = os.Remove(lock)
	}()

	res := verifyPreparedWorktree(Result{OK: true, Path: wt, BaseSHA: base}, git)
	if !res.OK {
		t.Fatalf("expected verifyPreparedWorktree to succeed after lock released, got %+v", res)
	}
}

func TestSweepDeadWorktrees_PreservesActiveWorkerWithStaleHeartbeat(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	root := t.TempDir()
	mustGit(t, root, "init", "-q", "-b", "main")
	mustGit(t, root, "config", "user.email", "active@test")
	mustGit(t, root, "config", "user.name", "active")
	mustGit(t, root, "config", "commit.gpgsign", "false")

	seedFile := filepath.Join(root, "init.txt")
	if err := os.WriteFile(seedFile, []byte("init\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	mustGit(t, root, "add", "init.txt")
	mustGit(t, root, "commit", "-q", "-m", "init")

	wtRoot := t.TempDir()
	prep := Prepare(root, "activelane", "activekey", "", wtRoot, nil)
	if !prep.OK {
		t.Fatalf("prepare failed: %+v", prep)
	}
	wtPath := prep.Path

	// Set lease with an active living PID (os.Getpid()) and heartbeat older than DefaultHeartbeatStaleThreshold
	oldTS := time.Now().Add(-2 * time.Hour)
	activeLease := WorkerLease{
		PID:         os.Getpid(),
		SessionID:   "active-worker-session",
		CreatedAt:   oldTS,
		HeartbeatTS: oldTS,
	}
	if err := WriteWorkerLease(wtPath, activeLease); err != nil {
		t.Fatalf("WriteWorkerLease failed: %v", err)
	}
	_ = writeOwnerStamp(wtPath, OwnerStamp{
		Schema:    ownerStampSchema,
		PID:       os.Getpid(),
		LeaseID:   "active-worker-session",
		CreatedAt: oldTS,
	})

	// Run dead worktree sweep; the active worker must NOT be reaped
	report := SweepDeadWorktrees(root, wtRoot, nil)
	if report.Pruned != 0 {
		t.Fatalf("expected 0 worktrees pruned for active worker, got %d (paths: %v)", report.Pruned, report.Paths)
	}
	if _, err := os.Stat(wtPath); os.IsNotExist(err) {
		t.Fatalf("active worktree %q was reaped even though process is alive", wtPath)
	}

	// Also verify git worktree list still tracks wtPath
	_, listOut := rawGit(t, root, "worktree", "list")
	if !strings.Contains(filepath.ToSlash(listOut), filepath.ToSlash(wtPath)) {
		t.Fatalf("active worktree %q missing from git worktree list:\n%s", wtPath, listOut)
	}
}

func TestSweepDeadWorktrees_PreservesDirtyWorktree(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	root := t.TempDir()
	mustGit(t, root, "init", "-q", "-b", "main")
	mustGit(t, root, "config", "user.email", "dirty@test")
	mustGit(t, root, "config", "user.name", "dirty")
	mustGit(t, root, "config", "commit.gpgsign", "false")

	seedFile := filepath.Join(root, "init.txt")
	if err := os.WriteFile(seedFile, []byte("init\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	mustGit(t, root, "add", "init.txt")
	mustGit(t, root, "commit", "-q", "-m", "init")

	wtRoot := t.TempDir()
	prep := Prepare(root, "dirtylane", "dirtykey", "", wtRoot, nil)
	if !prep.OK {
		t.Fatalf("prepare failed: %+v", prep)
	}
	wtPath := prep.Path

	// Set lease with a dead PID (99999999) and stale timestamp so it would be eligible for reaping if clean
	staleTime := time.Now().Add(-2 * time.Hour)
	deadLease := WorkerLease{
		PID:         99999999,
		SessionID:   "dead-session",
		CreatedAt:   staleTime,
		HeartbeatTS: staleTime,
	}
	if err := WriteWorkerLease(wtPath, deadLease); err != nil {
		t.Fatalf("WriteWorkerLease failed: %v", err)
	}
	_ = writeOwnerStamp(wtPath, OwnerStamp{
		Schema:    ownerStampSchema,
		PID:       99999999,
		LeaseID:   "dead-session",
		CreatedAt: staleTime,
	})

	// Make the worktree dirty with an uncommitted file
	uncommittedFile := filepath.Join(wtPath, "wip.txt")
	uncommittedContent := []byte("valuable worker diff to preserve\n")
	if err := os.WriteFile(uncommittedFile, uncommittedContent, 0o644); err != nil {
		t.Fatal(err)
	}

	// Run sweep; the dirty worktree must be preserved to protect uncommitted worker diffs
	report := SweepDeadWorktrees(root, wtRoot, nil)
	if report.Pruned != 0 {
		t.Fatalf("expected 0 worktrees pruned for dirty worktree, got %d (paths: %v)", report.Pruned, report.Paths)
	}
	if _, err := os.Stat(wtPath); os.IsNotExist(err) {
		t.Fatalf("dirty worktree %q was deleted, should be preserved", wtPath)
	}
	gotContent, err := os.ReadFile(uncommittedFile)
	if err != nil {
		t.Fatalf("uncommitted file %q was lost: %v", uncommittedFile, err)
	}
	if string(gotContent) != string(uncommittedContent) {
		t.Fatalf("uncommitted file content corrupted: got %q, want %q", string(gotContent), string(uncommittedContent))
	}

	// When uncommitted changes are removed, verify that the clean dead worktree is subsequently pruned
	if err := os.Remove(uncommittedFile); err != nil {
		t.Fatal(err)
	}
	cleanReport := SweepDeadWorktrees(root, wtRoot, nil)
	if cleanReport.Pruned != 1 {
		t.Fatalf("expected 1 worktree pruned after becoming clean, got %d", cleanReport.Pruned)
	}
	if _, err := os.Stat(wtPath); !os.IsNotExist(err) {
		t.Fatalf("clean dead worktree %q still exists on disk after sweep", wtPath)
	}
}

func TestSweepDeadWorktrees_PreservesForeignPlatformRegistration(t *testing.T) {
	root := t.TempDir()
	wtRoot := t.TempDir()

	wtName := "fak-worker-wt-testforeign"
	wtAdminDir := filepath.Join(root, ".git", "worktrees", wtName)
	if err := os.MkdirAll(wtAdminDir, 0o755); err != nil {
		t.Fatal(err)
	}

	// Choose a foreign OS path based on the current platform
	foreignPath := "/mnt/c/work/fak/_scratch/fak-worker-wt-foreign/gitdir"
	if runtime.GOOS != "windows" {
		foreignPath = `C:\work\fak\_scratch\fak-worker-wt-foreign\gitdir`
	}

	gitdirFile := filepath.Join(wtAdminDir, "gitdir")
	if err := os.WriteFile(gitdirFile, []byte(foreignPath+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	// Run sweep; the foreign-platform worktree registration must NOT be reaped
	report := SweepDeadWorktrees(root, wtRoot, nil)
	if report.Pruned != 0 {
		t.Fatalf("expected 0 worktrees pruned for foreign platform registration, got %d (paths: %v)", report.Pruned, report.Paths)
	}
	if _, err := os.Stat(wtAdminDir); os.IsNotExist(err) {
		t.Fatalf("foreign worktree admin dir %q was deleted, should be preserved", wtAdminDir)
	}
}

func TestSweepDeadWorktrees_ReapsLocalDeadRegistration(t *testing.T) {
	root := t.TempDir()
	wtRoot := t.TempDir()

	wtName := "fak-worker-wt-testlocaldead"
	wtAdminDir := filepath.Join(root, ".git", "worktrees", wtName)
	if err := os.MkdirAll(wtAdminDir, 0o755); err != nil {
		t.Fatal(err)
	}

	// Choose a local OS path that does not exist
	localDeadPath := filepath.Join(root, "_scratch", "nonexistent", "gitdir")
	gitdirFile := filepath.Join(wtAdminDir, "gitdir")
	if err := os.WriteFile(gitdirFile, []byte(localDeadPath+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	report := SweepDeadWorktrees(root, wtRoot, nil)
	if report.Pruned != 1 {
		t.Fatalf("expected 1 worktree pruned for dead local registration, got %d", report.Pruned)
	}
	if _, err := os.Stat(wtAdminDir); !os.IsNotExist(err) {
		t.Fatalf("dead local worktree admin dir %q was not reaped", wtAdminDir)
	}
}

func TestIsWindowsAbsolutePath(t *testing.T) {
	cases := []struct {
		path string
		want bool
	}{
		{`C:\work\fak\_scratch\fak-worker-wt-test\gitdir`, true},
		{`c:\work\fak\_scratch\fak-worker-wt-test\gitdir`, true},
		{`D:/work/fak/_scratch/fak-worker-wt-test/gitdir`, true},
		{`\\server\share\worktree`, true},
		{`/mnt/c/work/fak/_scratch/fak-worker-wt-test/gitdir`, false},
		{`/home/user/repo`, false},
		{`relative/path`, false},
		{``, false},
	}
	for _, tc := range cases {
		if got := isWindowsAbsolutePath(tc.path); got != tc.want {
			t.Errorf("isWindowsAbsolutePath(%q) = %v, want %v", tc.path, got, tc.want)
		}
	}
}

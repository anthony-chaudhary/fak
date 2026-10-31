package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/safesync"
)

type detachedSyncFixture struct {
	repo, head, target string
	peerIndex, refs    []byte
}

func detachedSyncGitOutput(t *testing.T, repo string, args ...string) []byte {
	t.Helper()
	cmd := exec.Command("git", append([]string{"-C", repo}, args...)...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("git %v: %v\n%s", args, err, out)
	}
	return out
}

// TestMain already isolates fixture Git configuration. These are real registered
// worktrees, not a mock branch resolver or an unregistered directory name alone.
func newDetachedSyncFixture(t *testing.T, worker, attached bool) detachedSyncFixture {
	t.Helper()
	root := t.TempDir()
	origin := filepath.Join(root, "origin")
	if err := os.Mkdir(origin, 0o755); err != nil {
		t.Fatal(err)
	}
	syncGit(t, origin, "init", "-b", "main")
	syncGit(t, origin, "config", "core.autocrlf", "false")
	syncGit(t, origin, "config", "user.name", "test")
	syncGit(t, origin, "config", "user.email", "test@example.com")
	for _, p := range []string{"local.txt", "incoming.txt", "peer.txt"} {
		syncWriteFile(t, filepath.Join(origin, p), "base\n")
	}
	syncGit(t, origin, "add", ".")
	syncGit(t, origin, "commit", "-m", "base")
	clone := filepath.Join(root, "clone")
	syncGit(t, root, "-c", "core.autocrlf=false", "clone", origin, clone)
	syncGit(t, clone, "config", "user.name", "test")
	syncGit(t, clone, "config", "user.email", "test@example.com")
	syncGit(t, clone, "config", "core.autocrlf", "false")
	syncGit(t, clone, "branch", "-m", "seed") // Local main is deliberately absent.
	name := "ordinary"
	if worker {
		name = "fak-worker-wt-default-cas"
	}
	repo := filepath.Join(root, name)
	if attached {
		syncGit(t, clone, "worktree", "add", "-b", "attached", repo, "HEAD")
	} else {
		syncGit(t, clone, "worktree", "add", "--detach", repo, "HEAD")
	}
	syncWriteFile(t, filepath.Join(repo, "local.txt"), "local commit\n")
	syncGit(t, repo, "add", "local.txt")
	syncGit(t, repo, "commit", "-m", "local")
	syncWriteFile(t, filepath.Join(origin, "incoming.txt"), "incoming commit\n")
	syncGit(t, origin, "add", "incoming.txt")
	syncGit(t, origin, "commit", "-m", "incoming")
	syncGit(t, repo, "fetch", "origin", "main")
	syncWriteFile(t, filepath.Join(repo, "peer.txt"), "peer staged\n")
	syncGit(t, repo, "add", "peer.txt")
	syncWriteFile(t, filepath.Join(repo, "peer.txt"), "peer unstaged\n")
	syncWriteFile(t, filepath.Join(repo, "untracked.txt"), "unrelated untracked\n")
	return detachedSyncFixture{
		repo: repo, head: syncRev(t, repo, "HEAD"), target: syncRev(t, repo, "origin/main"),
		peerIndex: detachedSyncGitOutput(t, repo, "ls-files", "--stage", "-z", "--", "peer.txt"),
		refs:      detachedSyncGitOutput(t, repo, "show-ref"),
	}
}

func (f detachedSyncFixture) assertPeers(t *testing.T) {
	t.Helper()
	if got := detachedSyncGitOutput(t, f.repo, "ls-files", "--stage", "-z", "--", "peer.txt"); !bytes.Equal(got, f.peerIndex) {
		t.Fatalf("peer stage changed: %q -> %q", f.peerIndex, got)
	}
	if syncReadFile(t, filepath.Join(f.repo, "peer.txt")) != "peer unstaged\n" || syncReadFile(t, filepath.Join(f.repo, "untracked.txt")) != "unrelated untracked\n" {
		t.Fatal("unrelated peer working bytes changed")
	}
	if got := syncRev(t, f.repo, "origin/main"); got != f.target {
		t.Fatalf("remote-tracking ref changed: %s -> %s", f.target, got)
	}
	if got := strings.TrimSpace(string(detachedSyncGitOutput(t, f.repo, "diff", "--cached", "--name-only"))); got != "peer.txt" {
		t.Fatalf("unexpected staged paths: %s", got)
	}
}

// fak-test:runtime medium est=3s lane=default
// fak-test:justify Uses tiny local real-Git repositories and the actual runSync entrypoint; no network or compiler subprocess.
func TestRunSyncReconcileDetachedWorkerDefaultTarget(t *testing.T) {
	for _, tc := range []struct {
		name, branch, wantBranch string
		worker, attached, apply  bool
	}{
		{name: "omitted worker branch", worker: true, wantBranch: "main", apply: true},
		{name: "explicit main is not reinterpreted", worker: true, branch: "main", wantBranch: "main"},
		{name: "explicit HEAD remains supported", worker: true, branch: "HEAD", wantBranch: "HEAD", apply: true},
		{name: "ordinary detached is not opted in", wantBranch: "main"},
		{name: "attached branch remains attached", worker: true, attached: true, wantBranch: "attached", apply: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := newDetachedSyncFixture(t, tc.worker, tc.attached)
			args := []string{"reconcile", "--repo", f.repo, "--remote", "origin", "--goal", "integrate", "--apply", "--json"}
			if tc.branch != "" {
				args = append(args, "--branch", tc.branch)
			}
			if tc.branch == "HEAD" || tc.attached {
				// These pre-existing explicit/attached routes retain their own branch;
				// unlike the omitted worker case they need an explicit remote goal.
				args[6] = "integrate refs/remotes/origin/main"
			}
			var out, stderr bytes.Buffer
			code := runSync(&out, &stderr, args)
			var got safesync.ReconcileAssessment
			if err := json.Unmarshal(out.Bytes(), &got); err != nil {
				t.Fatalf("decode: %v; exit=%d stdout=%s stderr=%s", err, code, out.String(), stderr.String())
			}
			if got.Branch != tc.wantBranch || got.Target != f.target || (got.TargetRef != "origin/main" && got.TargetRef != "refs/remotes/origin/main") {
				t.Fatalf("remote/default branch changed: %+v", got)
			}
			if got.Route != safesync.RouteDisjointIntegrate || got.Applied != tc.apply {
				t.Fatalf("route/apply=%s/%v want disjoint/%v: %+v", got.Route, got.Applied, tc.apply, got)
			}
			if tc.apply {
				if code != syncExitOK || got.Execution == nil || !got.Execution.Success {
					t.Fatalf("integration failed: exit=%d result=%+v stderr=%s", code, got, stderr.String())
				}
				newHead := syncRev(t, f.repo, "HEAD")
				parents := strings.Fields(string(detachedSyncGitOutput(t, f.repo, "show", "-s", "--format=%P", newHead)))
				if newHead == f.head || !reflect.DeepEqual(parents, []string{f.head, f.target}) || got.Execution.NewHead != newHead {
					t.Fatalf("wrong pinned integration: HEAD=%s parents=%v result=%+v", newHead, parents, got.Execution)
				}
				if syncReadFile(t, filepath.Join(f.repo, "local.txt")) != "local commit\n" || syncReadFile(t, filepath.Join(f.repo, "incoming.txt")) != "incoming commit\n" {
					t.Fatal("integrated worktree does not contain both sources")
				}
			} else {
				if code != syncExitRefused || syncRev(t, f.repo, "HEAD") != f.head || !bytes.Equal(detachedSyncGitOutput(t, f.repo, "show-ref"), f.refs) {
					t.Fatalf("negative case mutated refs or did not refuse: exit=%d result=%+v", code, got)
				}
				if syncReadFile(t, filepath.Join(f.repo, "incoming.txt")) != "base\n" {
					t.Fatal("refused integration changed incoming working path")
				}
			}
			if tc.attached {
				if got := strings.TrimSpace(string(detachedSyncGitOutput(t, f.repo, "symbolic-ref", "HEAD"))); got != "refs/heads/attached" {
					t.Fatalf("attached HEAD changed: %s", got)
				}
				wantRefs := bytes.Replace(f.refs, []byte(f.head+" refs/heads/attached\n"), []byte(syncRev(t, f.repo, "HEAD")+" refs/heads/attached\n"), 1)
				if !bytes.Equal(detachedSyncGitOutput(t, f.repo, "show-ref"), wantRefs) {
					t.Fatal("attached integration changed an unrelated named ref")
				}
			} else {
				cmd := exec.Command("git", "-C", f.repo, "symbolic-ref", "--quiet", "HEAD")
				var exitErr *exec.ExitError
				if err := cmd.Run(); !errors.As(err, &exitErr) || exitErr.ExitCode() != 1 {
					t.Fatalf("expected detached HEAD, got %v", err)
				}
				if !bytes.Equal(detachedSyncGitOutput(t, f.repo, "show-ref"), f.refs) {
					t.Fatal("detached integration changed a named ref")
				}
			}
			f.assertPeers(t)
		})
	}
}

// fak-test:runtime medium est=2s lane=default
// fak-test:justify Uses four tiny real-Git CLI fixtures with injected lookup failures; no network or compiler subprocess.
func TestRunSyncReconcileDetachedWorkerProbeFailure(t *testing.T) {
	for _, tc := range []struct {
		name string
		out  safesync.RunResult
	}{
		{name: "runner error", out: safesync.RunResult{Code: -1, Err: errors.New("injected symbolic-ref lookup failure")}},
		{name: "unexpected exit", out: safesync.RunResult{Code: 128, Stderr: []byte("injected lookup failure")}},
		{name: "empty symbolic ref", out: safesync.RunResult{Code: 0}},
		{name: "ambiguous detached output", out: safesync.RunResult{Code: 1, Stdout: []byte("refs/heads/ambiguous\n")}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := newDetachedSyncFixture(t, true, false)
			old := syncRouteReconciliation
			t.Cleanup(func() { syncRouteReconciliation = old })
			probes, mutations := 0, 0
			syncRouteReconciliation = func(ctx context.Context, opts safesync.ReconcileOptions) (safesync.ReconcileAssessment, error) {
				if opts.Branch != "" {
					t.Fatalf("actual CLI populated omitted branch: %q", opts.Branch)
				}
				opts.Runner = func(ctx context.Context, repo string, args ...string) safesync.RunResult {
					if reflect.DeepEqual(args, []string{"symbolic-ref", "--quiet", "HEAD"}) {
						probes++
						return tc.out
					}
					if len(args) > 0 && (args[0] == "update-ref" || args[0] == "commit-tree" || args[0] == "merge" || args[0] == "read-tree" || args[0] == "checkout") {
						mutations++
					}
					return safesync.RealRunner(ctx, repo, args...)
				}
				return safesync.RouteReconciliation(ctx, opts)
			}
			var out, stderr bytes.Buffer
			code := runSync(&out, &stderr, []string{"reconcile", "--repo", f.repo, "--goal", "integrate", "--apply", "--json"})
			if code != syncExitInternal || probes != 1 || mutations != 0 || !strings.Contains(stderr.String(), "local integration target") {
				t.Fatalf("unknown probe was not fail-closed: exit=%d probes=%d mutations=%d stderr=%s", code, probes, mutations, stderr.String())
			}
			if syncRev(t, f.repo, "HEAD") != f.head || !bytes.Equal(detachedSyncGitOutput(t, f.repo, "show-ref"), f.refs) || syncReadFile(t, filepath.Join(f.repo, "incoming.txt")) != "base\n" {
				t.Fatal("unknown probe changed refs or incoming working path")
			}
			f.assertPeers(t)
		})
	}
}

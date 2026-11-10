package gitgraph

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func git(t testing.TB, dir string, args ...string) string {
	t.Helper()
	cmd := exec.Command("git", args...)
	cmd.Dir = dir
	cmd.Env = append(os.Environ(),
		"GIT_CONFIG_NOSYSTEM=1", "GIT_CONFIG_GLOBAL="+os.DevNull,
		"GIT_AUTHOR_NAME=t", "GIT_AUTHOR_EMAIL=t@example.com",
		"GIT_COMMITTER_NAME=t", "GIT_COMMITTER_EMAIL=t@example.com",
		"GIT_AUTHOR_DATE=2026-01-01T00:00:00Z", "GIT_COMMITTER_DATE=2026-01-01T00:00:00Z")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("git %v: %v\n%s", args, err, out)
	}
	return strings.TrimSpace(string(out))
}

// buildRepo makes a history with two branches, a two-parent merge, an
// octopus merge and an unrelated root, returning every commit id.
func buildRepo(t testing.TB) (string, []string) {
	t.Helper()
	dir := t.TempDir()
	git(t, dir, "init", "-q", "-b", "main")
	commit := func(msg string) string {
		git(t, dir, "commit", "-q", "--allow-empty", "-m", msg)
		return git(t, dir, "rev-parse", "HEAD")
	}
	var all []string
	all = append(all, commit("root"), commit("m1"))
	git(t, dir, "checkout", "-q", "-b", "side")
	all = append(all, commit("s1"), commit("s2"))
	git(t, dir, "checkout", "-q", "-b", "side2", "main")
	all = append(all, commit("t1"))
	git(t, dir, "checkout", "-q", "main")
	all = append(all, commit("m2"))
	git(t, dir, "merge", "-q", "--no-ff", "-m", "merge side", "side")
	all = append(all, git(t, dir, "rev-parse", "HEAD"))
	git(t, dir, "checkout", "-q", "-b", "o1", "side")
	all = append(all, commit("o1"))
	git(t, dir, "checkout", "-q", "main")
	git(t, dir, "merge", "-q", "--no-ff", "-m", "octopus", "side2", "o1")
	all = append(all, git(t, dir, "rev-parse", "HEAD"), commit("m3"))
	git(t, dir, "checkout", "-q", "--orphan", "lone")
	all = append(all, commit("lone root"), commit("lone 2"))
	git(t, dir, "checkout", "-q", "main")
	return dir, all
}

func gitIsAncestor(dir, a, b string) bool {
	cmd := exec.Command("git", "merge-base", "--is-ancestor", a, b)
	cmd.Dir = dir
	return cmd.Run() == nil
}

// assertParity compares every ordered pair against git. Reachable sets come
// from one `git rev-list` per commit (a spawn costs ~0.5 s on a loaded Windows
// host); a sample of pairs is also checked with merge-base --is-ancestor itself.
func assertParity(t *testing.T, dir string, all []string) {
	t.Helper()
	reach := map[string]map[string]bool{}
	for _, c := range all {
		set := map[string]bool{}
		for _, o := range strings.Fields(git(t, dir, "rev-list", c)) {
			set[o] = true
		}
		reach[c] = set
	}
	for i, a := range all {
		for j, b := range all {
			want := reach[b][a]
			if (i*len(all)+j)%11 == 0 {
				if mb := gitIsAncestor(dir, a, b); mb != want {
					t.Fatalf("rev-list and merge-base disagree on %s,%s", a[:7], b[:7])
				}
			}
			got, ok := IsAncestorInProcess(dir, a, b)
			if !ok {
				t.Fatalf("IsAncestorInProcess(%s,%s) fell back with a full graph", a[:7], b[:7])
			}
			if got != want {
				t.Errorf("IsAncestor(%s,%s)=%v, git says %v", a[:7], b[:7], got, want)
			}
			ah, bh, ok := AheadBehindInProcess(dir, a, b)
			if !ok {
				t.Fatalf("AheadBehindInProcess(%s,%s) fell back", a[:7], b[:7])
			}
			wantAhead, wantBehind := 0, 0
			for o := range reach[b] {
				if !reach[a][o] {
					wantAhead++
				}
			}
			for o := range reach[a] {
				if !reach[b][o] {
					wantBehind++
				}
			}
			if ah != wantAhead || bh != wantBehind {
				t.Errorf("AheadBehind(%s,%s)=%d/%d, git says %d/%d", a[:7], b[:7], ah, bh, wantAhead, wantBehind)
			}
		}
	}
	// One direct rev-list --count witness for the left/right orientation.
	a, b := all[2], all[len(all)-3]
	ah, bh, _ := AheadBehindInProcess(dir, a, b)
	if want := git(t, dir, "rev-list", "--left-right", "--count", a+"..."+b); fmt.Sprintf("%d	%d", bh, ah) != want {
		t.Errorf("rev-list --left-right --count %q vs behind/ahead %d/%d", want, bh, ah)
	}
}

func TestIsAncestorParitySingleGraph(t *testing.T) {
	dir, all := buildRepo(t)
	git(t, dir, "commit-graph", "write", "--reachable")
	assertParity(t, dir, all)
}

func TestIsAncestorParitySplitChain(t *testing.T) {
	dir, all := buildRepo(t)
	// Two layers: the first covers main's history, the second adds the rest.
	git(t, dir, "commit-graph", "write", "--split", "--reachable")
	git(t, dir, "checkout", "-q", "main")
	git(t, dir, "commit", "-q", "--allow-empty", "-m", "after split")
	all = append(all, git(t, dir, "rev-parse", "HEAD"))
	git(t, dir, "commit-graph", "write", "--split=no-merge", "--reachable")
	if _, err := os.Stat(filepath.Join(dir, ".git", "objects", "info", "commit-graphs", "commit-graph-chain")); err != nil {
		t.Fatalf("expected a chain: %v", err)
	}
	g, err := Load(filepath.Join(dir, ".git", "objects"))
	if err != nil || len(g.layers) < 2 {
		t.Fatalf("want >=2 layers, got %v err %v", g, err)
	}
	assertParity(t, dir, all)
}

func TestIsAncestorLinkedWorktree(t *testing.T) {
	dir, all := buildRepo(t)
	git(t, dir, "commit-graph", "write", "--reachable")
	wt := filepath.Join(t.TempDir(), "wt")
	git(t, dir, "worktree", "add", "-q", wt, "side")
	sub := filepath.Join(wt, "nested")
	if err := os.MkdirAll(sub, 0o755); err != nil {
		t.Fatal(err)
	}
	if is, ok := IsAncestorInProcess(sub, all[0], all[len(all)-3]); !ok || !is {
		t.Fatalf("worktree lookup: is=%v ok=%v", is, ok)
	}
}

func TestIsAncestorFallback(t *testing.T) {
	ctx := context.Background()
	dir, all := buildRepo(t)
	root, head := all[0], all[len(all)-3]

	// No graph file: no in-process answer, wrapper still matches git.
	if _, ok := IsAncestorInProcess(dir, root, head); ok {
		t.Fatal("answered without a commit-graph")
	}
	if !IsAncestor(ctx, dir, root, head) || IsAncestor(ctx, dir, head, root) {
		t.Fatal("fallback disagrees with git")
	}

	// Commit newer than the graph: stale, fall back.
	git(t, dir, "commit-graph", "write", "--reachable")
	git(t, dir, "commit", "-q", "--allow-empty", "-m", "post-graph")
	fresh := git(t, dir, "rev-parse", "HEAD")
	if _, ok := IsAncestorInProcess(dir, root, fresh); ok {
		t.Fatal("answered for a commit absent from the graph")
	}
	if !IsAncestor(ctx, dir, root, fresh) {
		t.Fatal("fallback for absent commit disagrees with git")
	}
	if _, _, err := AheadBehind(ctx, dir, root, fresh); err != nil {
		t.Fatal(err)
	}

	// Abbreviated ids and refs are git's to resolve.
	if _, ok := IsAncestorInProcess(dir, root[:12], head); ok {
		t.Fatal("answered for an abbreviated id")
	}

	// Corrupt graph: fall back.
	graph := filepath.Join(dir, ".git", "objects", "info", "commit-graph")
	_ = os.Chmod(graph, 0o644) // git writes the graph read-only
	if err := os.WriteFile(graph, []byte("CGPH garbage"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, ok := IsAncestorInProcess(dir, root, head); ok {
		t.Fatal("answered from a corrupt graph")
	}
	if !IsAncestor(ctx, dir, root, head) {
		t.Fatal("fallback after corruption disagrees with git")
	}

	// Replace refs rewrite ancestry: refuse the graph.
	git(t, dir, "commit-graph", "write", "--reachable")
	if _, ok := IsAncestorInProcess(dir, root, head); !ok {
		t.Fatal("rewritten graph not used")
	}
	git(t, dir, "replace", "--graft", head)
	if _, ok := IsAncestorInProcess(dir, root, head); ok {
		t.Fatal("answered despite replace refs")
	}
	if got, want := IsAncestor(ctx, dir, root, head), gitIsAncestor(dir, root, head); got != want {
		t.Fatalf("replace fallback got %v want %v", got, want)
	}
}

func BenchmarkIsAncestor(b *testing.B) {
	dir, all := buildRepo(b)
	git(b, dir, "commit-graph", "write", "--reachable")
	root, head := all[0], all[len(all)-3]
	b.Run("inprocess", func(b *testing.B) {
		for b.Loop() {
			if is, ok := IsAncestorInProcess(dir, root, head); !ok || !is {
				b.Fatal("bad answer")
			}
		}
	})
	b.Run("subprocess", func(b *testing.B) {
		for b.Loop() {
			if !gitIsAncestor(dir, root, head) {
				b.Fatal("bad answer")
			}
		}
	})
}

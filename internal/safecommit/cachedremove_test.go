package safecommit

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func cachedRemoveReply(nameStatus, status string) map[string]reply {
	rep := onTrunkBase()
	rep["diff-cached"] = reply{out: nameStatus, code: 0}
	rep["status"] = reply{out: status, code: 0}
	return rep
}

func cachedRemoveOpts(t *testing.T, root string) Options {
	t.Helper()
	opts := baseOpts()
	opts.Dir = root
	return opts
}

func writeCachedRemoveWorktreeFile(t *testing.T, root string) {
	t.Helper()
	path := filepath.Join(root, "internal", "foo", "bar.go")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("package foo\n"), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestCachedRemoveWorktreePresent_refusesBeforeAdd(t *testing.T) {
	t.Setenv(cachedRemoveEnvVar, "")
	root := t.TempDir()
	writeCachedRemoveWorktreeFile(t, root)
	g := &fakeGit{reply: cachedRemoveReply("D\tinternal/foo/bar.go\n", "D  internal/foo/bar.go\n")}

	res, err := CommitWith(context.Background(), g.run, okLock(nil), cachedRemoveOpts(t, root))
	if err != nil {
		t.Fatalf("unexpected infra error: %v", err)
	}
	if res.Reason != ReasonCachedRemoveWorktreePresent {
		t.Fatalf("want %q, got reason=%q detail=%q", ReasonCachedRemoveWorktreePresent, res.Reason, res.Detail)
	}
	if res.Committed {
		t.Fatalf("cached-remove refusal must not commit, got %+v", res)
	}
	for _, forbidden := range []string{"add", "commit"} {
		if g.sawSubcommand(forbidden) {
			t.Fatalf("cached-remove refusal must not %q; calls=%v", forbidden, g.calls)
		}
	}
	for _, want := range []string{"internal/foo/bar.go", "remove", ".gitignore", "pathspec"} {
		if !strings.Contains(res.Detail, want) {
			t.Fatalf("detail should contain %q, got %q", want, res.Detail)
		}
	}
}

func TestCachedRemoveWorktreePresent_usesRootDiscoveryWhenDirEmpty(t *testing.T) {
	t.Setenv(cachedRemoveEnvVar, "")
	root := t.TempDir()
	writeCachedRemoveWorktreeFile(t, root)
	rep := cachedRemoveReply("D\tinternal/foo/bar.go\n", "D  internal/foo/bar.go\n")
	rep["rev-parse --show-toplevel"] = reply{out: root + "\n", code: 0}
	g := &fakeGit{reply: rep}
	opts := baseOpts()
	opts.Dir = ""

	res, err := CommitWith(context.Background(), g.run, okLock(nil), opts)
	if err != nil {
		t.Fatalf("unexpected infra error: %v", err)
	}
	if res.Reason != ReasonCachedRemoveWorktreePresent {
		t.Fatalf("want %q with discovered root, got %+v", ReasonCachedRemoveWorktreePresent, res)
	}
}

func TestCachedRemoveWorktreePresent_allowsGenuineDelete(t *testing.T) {
	t.Setenv(cachedRemoveEnvVar, "")
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "internal", "foo"), 0o755); err != nil {
		t.Fatal(err)
	}
	g := &fakeGit{reply: cachedRemoveReply("D\tinternal/foo/bar.go\n", "D  internal/foo/bar.go\n")}

	res, err := CommitWith(context.Background(), g.run, okLock(nil), cachedRemoveOpts(t, root))
	if err != nil {
		t.Fatalf("unexpected infra error: %v", err)
	}
	if res.Reason == ReasonCachedRemoveWorktreePresent {
		t.Fatalf("genuine deletion must not be refused as cached-remove; detail=%q", res.Detail)
	}
	if !res.Verified {
		t.Fatalf("genuine deletion should still commit cleanly, got %+v", res)
	}
}

func TestCachedRemoveWorktreePresent_warnProceeds(t *testing.T) {
	t.Setenv(cachedRemoveEnvVar, "warn")
	root := t.TempDir()
	writeCachedRemoveWorktreeFile(t, root)
	g := &fakeGit{reply: cachedRemoveReply("D\tinternal/foo/bar.go\n", "D  internal/foo/bar.go\n")}

	res, err := CommitWith(context.Background(), g.run, okLock(nil), cachedRemoveOpts(t, root))
	if err != nil {
		t.Fatalf("unexpected infra error: %v", err)
	}
	if res.Reason == ReasonCachedRemoveWorktreePresent {
		t.Fatalf("warn must not refuse, got reason=%q", res.Reason)
	}
	if !res.Verified {
		t.Fatalf("warn should proceed to a verified commit, got %+v", res)
	}
	if !strings.Contains(res.Detail, "CACHED_REMOVE_WORKTREE_PRESENT (warn)") {
		t.Fatalf("warn should record the would-be refusal in Detail, got %q", res.Detail)
	}
}

func TestCachedRemoveWorktreePresent_offSkips(t *testing.T) {
	t.Setenv(cachedRemoveEnvVar, "off")
	root := t.TempDir()
	writeCachedRemoveWorktreeFile(t, root)
	g := &fakeGit{reply: cachedRemoveReply("D\tinternal/foo/bar.go\n", "D  internal/foo/bar.go\n")}

	res, err := CommitWith(context.Background(), g.run, okLock(nil), cachedRemoveOpts(t, root))
	if err != nil {
		t.Fatalf("unexpected infra error: %v", err)
	}
	if res.Reason == ReasonCachedRemoveWorktreePresent {
		t.Fatalf("off must skip the guard, got refusal detail=%q", res.Detail)
	}
}

func TestCachedDeletedPaths(t *testing.T) {
	got := cachedDeletedPaths("M\tinternal/foo/keep.go\nD\tinternal/foo/bar.go\nD internal/foo/baz.go\n", []string{"internal/foo"})
	want := []string{"internal/foo/bar.go", "internal/foo/baz.go"}
	if strings.Join(got, "\x00") != strings.Join(want, "\x00") {
		t.Fatalf("cachedDeletedPaths = %v, want %v", got, want)
	}
}

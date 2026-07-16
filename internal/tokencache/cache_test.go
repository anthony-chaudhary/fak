package tokencache

import (
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/clonescan"
)

// TestRoundTripAndOneByteMiss proves the core content-addressed contract: the same
// bytes round-trip byte-identically, and a one-byte change misses (acceptance #1).
func TestRoundTripAndOneByteMiss(t *testing.T) {
	c := New(t.TempDir(), "v1")
	src := "package a\nfunc f(x int) int { return x * 2 }\n"
	keys := []string{"a", "b", "c"}
	spans := [][2]int{{1, 2}, {3, 4}, {5, 6}}

	if _, _, ok := c.Get(src); ok {
		t.Fatal("cold Get should miss")
	}
	c.Put(src, keys, spans)

	gotKeys, gotSpans, ok := c.Get(src)
	if !ok {
		t.Fatal("warm Get should hit")
	}
	if !reflect.DeepEqual(gotKeys, keys) || !reflect.DeepEqual(gotSpans, spans) {
		t.Fatalf("round-trip not byte-identical:\n keys %v vs %v\n spans %v vs %v", gotKeys, keys, gotSpans, spans)
	}

	if _, _, ok := c.Get(src + "x"); ok {
		t.Fatal("a one-byte change must miss")
	}
}

// TestEmptyResultCaches proves a file with no qualifying window is a valid hit (so a
// large data file is not re-lexed every invocation just because it yields no windows).
func TestEmptyResultCaches(t *testing.T) {
	c := New(t.TempDir(), "v1")
	src := "package a\nvar x = 1\n"
	c.Put(src, nil, nil)
	keys, spans, ok := c.Get(src)
	if !ok || len(keys) != 0 || len(spans) != 0 {
		t.Fatalf("empty result should hit with empty slices: ok=%v keys=%v spans=%v", ok, keys, spans)
	}
}

// TestVersionInvalidation proves the tokenizer-version tag is part of the key: an entry
// written under one version misses under a bumped version, even for identical bytes,
// with no stale window served (acceptance #2).
func TestVersionInvalidation(t *testing.T) {
	dir := t.TempDir()
	src := "package a\nfunc f() int { return 1 + 1 }\n"
	keys := []string{"k"}
	spans := [][2]int{{1, 1}}

	New(dir, "tok-v1").Put(src, keys, spans)

	if _, _, ok := New(dir, "tok-v2").Get(src); ok {
		t.Fatal("a bumped tokenizer version must miss prior entries (no stale window served)")
	}
	// The original version still hits — the bump did not corrupt the old entry.
	if _, _, ok := New(dir, "tok-v1").Get(src); !ok {
		t.Fatal("the original version should still hit its own entry")
	}
}

// TestDirUnderGitCommonDir proves Open resolves under `git rev-parse --git-common-dir`
// / fak / token-cache, that the path is inside .git, and that building an index through
// the cache writes nothing into the worktree (acceptance #3).
func TestDirUnderGitCommonDir(t *testing.T) {
	root := initGitRepo(t)

	wc := Open(root)
	if wc == nil {
		t.Fatal("Open returned nil in a real git repo")
	}
	c, ok := wc.(*Cache)
	if !ok {
		t.Fatalf("Open returned %T, want *Cache", wc)
	}
	commonDirAbs := filepath.Join(root, ".git")
	wantDir := TokenCacheDir(commonDirAbs)
	if c.dir != wantDir {
		t.Fatalf("cache dir = %q, want %q (under git-common-dir/fak/token-cache)", c.dir, wantDir)
	}
	if !strings.Contains(filepath.ToSlash(c.dir), "/.git/") {
		t.Fatalf("cache dir %q is not inside .git", c.dir)
	}

	// Snapshot the worktree (excluding .git), build an index through the cache, and
	// assert nothing new landed outside .git.
	before := worktreeFiles(t, root)
	tree := map[string]string{"a.go": "package a\nfunc f(x int) int {\n\tfor x > 0 {\n\t\tx -= 1\n\t}\n\treturn x\n}\n"}
	clonescan.BuildTreeIndex(tree, wc)
	after := worktreeFiles(t, root)
	if !reflect.DeepEqual(before, after) {
		t.Fatalf("building through the cache wrote into the worktree:\n before=%v\n after=%v", before, after)
	}

	// And an entry DID land inside .git.
	ents, err := os.ReadDir(c.dir)
	if err != nil || len(ents) == 0 {
		t.Fatalf("expected at least one cache entry under %q, err=%v ents=%d", c.dir, err, len(ents))
	}
}

// TestUnwritableDirDegrades proves an unwritable cache dir yields an index byte-identical
// to the uncached path: the cache accelerates, it never gates (acceptance #4).
func TestUnwritableDirDegrades(t *testing.T) {
	// Point the cache dir at an existing FILE, so MkdirAll and every read fail.
	f := filepath.Join(t.TempDir(), "not-a-dir")
	if err := os.WriteFile(f, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	broken := New(f, clonescan.TokenizerVersion())

	tree := cloneTree()
	want := clonescan.CandidateKeys(tree["a.go"])
	uncached := clonescan.BuildTreeIndex(tree).Query(want, "a.go", 0)
	degraded := clonescan.BuildTreeIndex(tree, broken).Query(want, "a.go", 0)
	if !reflect.DeepEqual(uncached, degraded) {
		t.Fatalf("unwritable cache changed output:\n uncached=%+v\n degraded=%+v", uncached, degraded)
	}
	// Nothing should have been written (the file is untouched, no dir created).
	if st, err := os.Stat(f); err != nil || st.IsDir() {
		t.Fatalf("the sentinel file was replaced by the cache: err=%v isDir=%v", err, st.IsDir())
	}
}

// TestEnabledFlagOff proves FAK_TOKEN_CACHE=off disables Open (a true nil interface).
func TestEnabledFlagOff(t *testing.T) {
	t.Setenv(FlagEnv, "off")
	if Enabled() {
		t.Fatal("Enabled() should be false when FAK_TOKEN_CACHE=off")
	}
	if wc := Open(initGitRepo(t)); wc != nil {
		t.Fatalf("Open should return nil when disabled, got %T", wc)
	}
}

// TestPruneEnforcesBudget proves the FIFO byte budget removes oldest entries until the
// dir is under budget.
func TestPruneEnforcesBudget(t *testing.T) {
	dir := t.TempDir()
	c := New(dir, "v1")
	// Write several entries with distinct byte payloads.
	for i := 0; i < 20; i++ {
		src := strings.Repeat("x", 100) + string(rune('a'+i))
		c.Put(src, []string{strings.Repeat("k", 200)}, [][2]int{{i, i}})
	}
	sizeBefore := dirSize(t, dir)
	if sizeBefore == 0 {
		t.Fatal("expected entries on disk")
	}
	budget := sizeBefore / 2
	c.prune(budget)
	if got := dirSize(t, dir); got > budget {
		t.Fatalf("prune left %d bytes, over budget %d", got, budget)
	}
}

// cloneTree mirrors the clonescan drift fixture: two files that token-clone plus a
// data-only file.
func cloneTree() map[string]string {
	body := "\nfunc NAME(items []int) int {\n\ttotal := 0\n\tfor i := 0; i < len(items); i++ {\n\t\tif items[i] > 0 {\n\t\t\ttotal += items[i] * 2\n\t\t} else {\n\t\t\ttotal -= items[i]\n\t\t}\n\t}\n\treturn total\n}\n"
	return map[string]string{
		"a.go":    "package a\n" + strings.Replace(body, "NAME", "alpha", 1),
		"b.go":    "package b\n" + strings.Replace(body, "NAME", "beta", 1),
		"data.go": "package a\nvar x = 1\nvar y = 2\n",
	}
}

func initGitRepo(t *testing.T) string {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	root := t.TempDir()
	for _, args := range [][]string{
		{"init", "-q"},
		{"config", "user.email", "t@example.com"},
		{"config", "user.name", "t"},
	} {
		cmd := exec.Command("git", args...)
		cmd.Dir = root
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
	}
	// Resolve symlinks so the common dir path matches (macOS /var -> /private/var).
	real, err := filepath.EvalSymlinks(root)
	if err != nil {
		t.Fatal(err)
	}
	return real
}

// worktreeFiles lists every file under root except those inside .git, sorted.
func worktreeFiles(t *testing.T, root string) []string {
	t.Helper()
	var out []string
	err := filepath.Walk(root, func(p string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(root, p)
		if err != nil {
			return err
		}
		rel = filepath.ToSlash(rel)
		if rel == ".git" || strings.HasPrefix(rel, ".git/") {
			if info.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}
		if !info.IsDir() {
			out = append(out, rel)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	return out
}

func dirSize(t *testing.T, dir string) int64 {
	t.Helper()
	var total int64
	ents, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	for _, de := range ents {
		if de.IsDir() {
			continue
		}
		info, err := de.Info()
		if err != nil {
			continue
		}
		total += info.Size()
	}
	return total
}

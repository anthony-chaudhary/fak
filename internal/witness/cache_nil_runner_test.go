package witness

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// fak-test:runtime fast est=100ms lane=default
func TestVerdictCacheNilRunnerUsesRealGit(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	dir := t.TempDir()
	if _, code, err := gitRunner(ctx, dir, "init", "--quiet", "--template="); err != nil || code != 0 {
		t.Fatalf("git init: code=%d err=%v", code, err)
	}

	r := NewWithRunner(nil, dir)
	if r.symptomProofCacheConfirmed(ctx, "missing-proof") {
		t.Fatal("absent symptom proof reported a cache hit")
	}
	cache, ok := r.verdictCache(ctx)
	want := WitnessCacheDir(filepath.Join(dir, ".git"))
	if !ok || cache == nil {
		t.Fatal("nil runner did not resolve the real repository cache")
	}
	if cache.dir != want || r.cacheDir != want {
		t.Fatalf("cache dirs = %q, %q; want %q", cache.dir, r.cacheDir, want)
	}
	if r.run != nil {
		t.Fatal("cache discovery mutated the resolver runner")
	}
	if _, err := os.Stat(filepath.Join(dir, ".git", "fak")); !os.IsNotExist(err) {
		t.Fatalf("cache miss created cache state or could not be inspected: %v", err)
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestVerdictCacheNilRunnerOutsideRepositoryMisses(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	root, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	dir := filepath.Join(root, "outside")
	if err := os.Mkdir(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	// A TMPDIR inside a checkout must not let Git discover that enclosing repo.
	t.Setenv("GIT_CEILING_DIRECTORIES", root)
	if _, code, err := gitRunner(ctx, dir, "rev-parse", "--git-common-dir"); err != nil || code == 0 {
		t.Fatalf("fixture is not a verified non-repository: code=%d err=%v", code, err)
	}

	r := NewWithRunner(nil, dir)
	for range 2 {
		if r.symptomProofCacheConfirmed(ctx, "missing-proof") {
			t.Fatal("outside-repository proof lookup reported a cache hit")
		}
		if cache, ok := r.verdictCache(ctx); ok || cache != nil || r.cacheDir != "" {
			t.Fatalf("outside-repository cache = %v, %v, %q; want nil, false, empty", cache, ok, r.cacheDir)
		}
	}
	entries, err := os.ReadDir(dir)
	if err != nil || len(entries) != 0 {
		t.Fatalf("outside-repository lookup wrote state or could not be inspected: entries=%v err=%v", entries, err)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestVerdictCachePreservesInjectedRunner(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	commonDir := t.TempDir()
	calls := 0
	run := func(_ context.Context, gotDir string, args ...string) (string, int, error) {
		calls++
		if gotDir != dir || strings.Join(args, " ") != "rev-parse --git-common-dir" {
			t.Fatalf("runner received dir=%q args=%q", gotDir, args)
		}
		return commonDir + "\n", 0, nil
	}
	r := NewWithRunner(run, dir)
	for range 2 {
		if r.symptomProofCacheConfirmed(ctx, "missing-proof") {
			t.Fatal("absent symptom proof reported a cache hit")
		}
		cache, ok := r.verdictCache(ctx)
		if !ok || cache == nil || cache.dir != WitnessCacheDir(commonDir) {
			t.Fatalf("cache = %v, %v; want injected common dir %q", cache, ok, commonDir)
		}
	}
	if calls != 1 {
		t.Fatalf("injected runner called %d times; want once", calls)
	}
}

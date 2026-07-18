package tokencache

// dogfood_test.go — the net-true dogfood witness for #5137 (follow-on to #4330).
//
// #4330 shipped the content-addressed cache with gate tests but never MEASURED it on
// the real tracked tree, so the speedup stayed "not yet". This test loads the actual
// tracked *.go tree of the enclosing repo (the same `git ls-files -- *.go` set
// `fak dup guard` tokenizes) and runs the A/B the issue names:
//
//	A arm  — BuildTreeIndex with no cache (FAK_TOKEN_CACHE=off path).
//	B cold — BuildTreeIndex through an empty cache (every file a miss + Put).
//	B warm — BuildTreeIndex through the warmed cache (the cross-invocation case).
//
// The DETERMINISTIC facts gate the test: the warm run must hit on every file (net-true
// hit-rate 100%) and its index must be byte-identical to the uncached one. The
// WALL-CLOCK numbers are logged as the provenance-labeled witness (WITNESSED — fak
// authored the measurement, single box, per docs/standards/net-true-value.md), net of
// the costs the cache adds: the cold-run Put overhead, the per-Open `git rev-parse
// --git-common-dir` resolve, and the on-disk bytes read back per warm hit. Timing is
// NOT asserted — a loaded CI box must not red the trunk on noise; the verdict line
// reports net-true vs not-yet honestly either way.
//
// Run: go test ./internal/tokencache -run TestDogfoodRealTreeNetTrue -v
// Skipped under -short and outside a real checkout (small trees prove nothing).

import (
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/clonescan"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// dogfoodMinFiles keeps the witness honest: a tree far smaller than the real ~5.7k
// tracked files (a sparse or foreign checkout) measures nothing worth citing.
const dogfoodMinFiles = 1000

// countingWindowCache wraps a WindowCache and counts gets/hits/puts, so the hit rate
// is measured at the seam BuildTreeIndex actually consults — not inferred.
type countingWindowCache struct {
	inner clonescan.WindowCache
	gets  int
	hits  int
	puts  int
}

func (c *countingWindowCache) Get(src string) ([]string, [][2]int, bool) {
	c.gets++
	keys, spans, ok := c.inner.Get(src)
	if ok {
		c.hits++
	}
	return keys, spans, ok
}

func (c *countingWindowCache) Put(src string, keys []string, spans [][2]int) {
	c.puts++
	c.inner.Put(src, keys, spans)
}

// TestDogfoodRealTreeNetTrue is the #5137 witness: real tree, cold vs warm, hit rate
// and wall-clock, net of the cache's own costs.
func TestDogfoodRealTreeNetTrue(t *testing.T) {
	if testing.Short() {
		t.Skip("dogfood run lexes the whole tracked tree; skipped under -short")
	}
	root, tree := realTrackedGoTree(t)
	if len(tree) < dogfoodMinFiles {
		t.Skipf("tracked tree has %d .go files (< %d): not the real tree this witness is about", len(tree), dogfoodMinFiles)
	}
	var srcBytes int64
	for _, src := range tree {
		srcBytes += int64(len(src))
	}

	// Cache dir: a fresh temp dir, so cold really is cold and no peer's shared entries
	// (nor the prune budget) can contaminate either arm. Placement under the real
	// git-common-dir is already gate-tested; its cost is measured separately below.
	cacheDir := t.TempDir()
	version := clonescan.TokenizerVersion()

	// A arm — the exact FAK_TOKEN_CACHE=off path (best of 2 to shave scheduler noise).
	var uncachedIdx *clonescan.TreeIndex
	uncachedDur := bestOf(2, func() { uncachedIdx = clonescan.BuildTreeIndex(tree) })

	// B cold — empty cache: every file misses, lexes, and Puts.
	cold := &countingWindowCache{inner: New(cacheDir, version)}
	var coldDur time.Duration
	{
		start := time.Now()
		clonescan.BuildTreeIndex(tree, cold)
		coldDur = time.Since(start)
	}
	if cold.hits != 0 {
		t.Fatalf("cold run had %d hits in a fresh cache dir", cold.hits)
	}
	if cold.puts != len(tree) {
		t.Fatalf("cold run put %d entries, want one per file (%d)", cold.puts, len(tree))
	}

	// B warm — the cross-invocation case: every file unchanged, every Get a hit.
	warm := &countingWindowCache{inner: New(cacheDir, version)}
	var warmIdx *clonescan.TreeIndex
	warmDur := bestOf(2, func() { warmIdx = clonescan.BuildTreeIndex(tree, warm) })
	if warm.hits != warm.gets {
		t.Fatalf("warm hit rate %d/%d: an unchanged tree must hit on every file", warm.hits, warm.gets)
	}
	if warm.puts != 0 {
		t.Fatalf("warm run wrote %d entries; a fully-warm run must write none", warm.puts)
	}

	// Accelerate-never-gate on the real tree: warm output byte-identical to uncached.
	if !reflect.DeepEqual(uncachedIdx, warmIdx) {
		t.Fatal("warm cached index differs from the uncached index on the real tree")
	}

	// Costs the cache adds (net-true denominators): the per-Open git resolve, and the
	// on-disk bytes a warm run reads back instead of lexing.
	resolveStart := time.Now()
	_, resolvedOK := commonDir(root)
	resolveDur := time.Since(resolveStart)
	entries, cacheBytes := dirEntriesAndBytes(t, cacheDir)
	if entries != len(tree) {
		t.Fatalf("cache dir holds %d entries, want one per file (%d)", entries, len(tree))
	}

	hitRate := 100 * float64(warm.hits) / float64(warm.gets)
	speedup := float64(uncachedDur) / float64(warmDur)
	warmTotal := warmDur + resolveDur // one invocation pays one Open resolve
	verdict := "net-true: warm+resolve beats the uncached re-lex"
	if warmTotal >= uncachedDur {
		verdict = "not yet: warm+resolve does not beat the uncached re-lex on this box"
	}
	t.Logf("dogfood witness (#5137, WITNESSED, single box): files=%d srcMB=%.1f", len(tree), float64(srcBytes)/(1<<20))
	t.Logf("  A  uncached   %v", uncachedDur)
	t.Logf("  B  cold       %v (put overhead %+v)", coldDur, coldDur-uncachedDur)
	t.Logf("  B  warm       %v (speedup %.2fx, hit rate %.1f%% [%d/%d])", warmDur, speedup, hitRate, warm.hits, warm.gets)
	t.Logf("  costs: git-common-dir resolve %v (ok=%v), cache disk %d entries / %.1f MB read per warm run", resolveDur, resolvedOK, entries, float64(cacheBytes)/(1<<20))
	t.Logf("  budget fit: tree needs %.1f MB vs default budget %.1f MB (fits=%v; over-budget means the Open-time prune evicts the working set)", float64(cacheBytes)/(1<<20), float64(defaultMaxBytes)/(1<<20), cacheBytes <= defaultMaxBytes)
	t.Logf("  verdict: %s (warm+resolve %v vs uncached %v)", verdict, warmTotal, uncachedDur)
}

// bestOf runs f n times and returns the fastest wall-clock, the standard noise shave
// for a pure in-memory measurement.
func bestOf(n int, f func()) time.Duration {
	best := time.Duration(0)
	for i := 0; i < n; i++ {
		start := time.Now()
		f()
		d := time.Since(start)
		if best == 0 || d < best {
			best = d
		}
	}
	return best
}

// realTrackedGoTree loads the enclosing repo's tracked *.go files exactly the way
// `fak dup guard` does (git ls-files -- *.go, read each), or skips when there is no
// real checkout to dogfood against.
func realTrackedGoTree(t *testing.T) (root string, tree map[string]string) {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	top := exec.Command("git", "rev-parse", "--show-toplevel")
	windowgate.ConfigureBackgroundCommand(top)
	out, err := top.Output()
	if err != nil {
		t.Skip("not inside a git checkout")
	}
	root = filepath.FromSlash(strings.TrimSpace(string(out)))
	ls := exec.Command("git", "ls-files", "*.go")
	ls.Dir = root
	windowgate.ConfigureBackgroundCommand(ls)
	out, err = ls.Output()
	if err != nil {
		t.Skipf("git ls-files: %v", err)
	}
	tree = make(map[string]string)
	for _, rel := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		rel = strings.TrimSpace(rel)
		if rel == "" {
			continue
		}
		b, err := os.ReadFile(filepath.Join(root, rel))
		if err != nil {
			continue // tracked-but-deleted; same skip dup guard takes
		}
		tree[filepath.ToSlash(rel)] = string(b)
	}
	return root, tree
}

// dirEntriesAndBytes counts the .json entries and their total size in dir.
func dirEntriesAndBytes(t *testing.T, dir string) (int, int64) {
	t.Helper()
	ents, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	n, total := 0, int64(0)
	for _, de := range ents {
		if de.IsDir() || !strings.HasSuffix(de.Name(), ".json") {
			continue
		}
		info, err := de.Info()
		if err != nil {
			continue
		}
		n++
		total += info.Size()
	}
	return n, total
}

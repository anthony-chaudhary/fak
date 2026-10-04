package tokencache

// dogfood_test.go — the net-true dogfood witness for #5137 (follow-on to #4330).
//
// #4330 shipped the content-addressed cache with gate tests but never MEASURED it on
// the real tracked tree, so the speedup stayed "not yet". This test loads the actual
// tracked *.go tree of the enclosing repo (the same `git ls-files -- *.go` set
// `fak dup guard` tokenizes) and runs the A/B the issue names:
//
//	A arm  — BuildTreeIndex with no cache (FAK_TOKEN_CACHE=off path).
//	B cold — BuildTreeIndex through an empty cache (every distinct file a miss + Put).
//	B warm — BuildTreeIndex through the warmed cache (the cross-invocation case).
//
// The DETERMINISTIC facts gate the test: within an explicit finite full-working-set
// fixture budget, the warm run must hit on every file (hit-rate 100%) and its index
// must be byte-identical to the uncached one. This does not assume the production
// defaults retain the whole tree; measured byte and entry fit is reported separately. The cache is
// content-addressed, so byte-identical tracked files share one entry: in the cold run
// only those duplicates may hit, and the cache holds one entry per distinct content. The
// WALL-CLOCK numbers are logged as the provenance-labeled witness (WITNESSED — fak
// authored the measurement, single box, per docs/standards/net-true-value.md), net of
// the costs the cache adds: the cold-run Put overhead, the per-Open `git rev-parse
// --git-common-dir` resolve, and the on-disk bytes read back per warm hit. Timing is
// NOT asserted — a loaded CI box must not red the trunk on noise; the verdict line
// reports net-true vs not-yet honestly either way.
//
// Run: go test ./internal/tokencache -run TestDogfoodRealTreeNetTrue -v
// Skipped under -short and outside a real checkout (small trees prove nothing).
//
// Memory: one real-tree TreeIndex is several GB of live heap (~3.6 GB at 16k files), so
// the arms never hold two at once. Each arm's index is reduced to a streamed canonical
// digest before the next arm builds, and the GC target is tightened for the run; holding
// the uncached and warm indexes side by side for reflect.DeepEqual peaked above 13 GB RSS.

import (
	"crypto/sha256"
	"encoding/binary"
	"hash"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"runtime/debug"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/clonescan"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// dogfoodMinFiles keeps the witness honest: a tree far smaller than the real (16k+)
// tracked files (a sparse or foreign checkout) measures nothing worth citing.
const dogfoodMinFiles = 1000

// dogfoodMaxBytes gives the full-resident fixture a finite envelope. If the actual
// serialized working set outgrows it, the residency assertions must fail visibly.
const dogfoodMaxBytes int64 = 8 << 30

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

func (c *countingWindowCache) Maintain() {
	if maintainer, ok := c.inner.(interface{ Maintain() }); ok {
		maintainer.Maintain()
	}
}

// TestDogfoodRealTreeNetTrue is the #5137 witness: real tree, cold vs warm, hit rate
// and wall-clock, net of the cache's own costs.
func TestDogfoodRealTreeNetTrue(t *testing.T) {
	if testing.Short() {
		t.Skip("dogfood run lexes the whole tracked tree; skipped under -short")
	}
	if raceDetectorEnabled {
		// The verdict is a wall-clock comparison (uncached re-lex vs warm hit), which
		// ThreadSanitizer's instrumentation makes meaningless. It also lexes the whole
		// tracked tree four times, which ran past the race job's per-package timeout
		// and killed the binary -- taking every later package's result down with it.
		t.Skip("net-true wall-clock witness is not meaningful under go test -race instrumentation")
	}
	root, tree := realTrackedGoTree(t)
	if len(tree) < dogfoodMinFiles {
		t.Skipf("tracked tree has %d .go files (< %d): not the real tree this witness is about", len(tree), dogfoodMinFiles)
	}
	var srcBytes int64
	contents := make(map[string]struct{}, len(tree))
	for _, src := range tree {
		srcBytes += int64(len(src))
		contents[src] = struct{}{}
	}
	// The cache is content-addressed (digest of tokenizer version + exact bytes), so
	// byte-identical tracked files (e.g. one fixture repo vendored into two testdata
	// dirs) share a single entry. BuildTreeIndex walks files in sorted order, so in the
	// cold run the first copy misses and Puts and every later copy hits that entry.
	distinct := len(contents)
	duplicates := len(tree) - distinct
	// Full residency is the cache-effectiveness condition being measured, not a
	// promise of the smaller production defaults. Keep Put and final retention
	// active under finite test-only ceilings; measure serialized fit after cold.
	t.Setenv(MaxBytesEnv, strconv.FormatInt(dogfoodMaxBytes, 10))
	t.Setenv(MaxEntriesEnv, strconv.Itoa(distinct))

	// Bound peak RSS: a real-tree index is several GB, and the default GOGC=100 lets the
	// heap reach twice the live set. The deterministic assertions do not depend on it.
	defer debug.SetGCPercent(debug.SetGCPercent(dogfoodGCPercent))

	// Cache dir: a fresh temp dir, so cold really is cold and no peer's shared entries
	// can contaminate either arm. Retention uses the explicit fixture envelope above.
	// Placement under the real
	// git-common-dir is already gate-tested; its cost is measured separately below.
	cacheDir := t.TempDir()
	version := clonescan.TokenizerVersion()

	// A arm — the exact FAK_TOKEN_CACHE=off path (best of 2 to shave scheduler noise).
	// Each rebuild drops the previous index first so only one is ever live.
	var uncachedIdx *clonescan.TreeIndex
	uncachedDur := bestOf(2, func() {
		uncachedIdx = nil
		uncachedIdx = clonescan.BuildTreeIndex(tree)
	})
	uncachedDigest := treeIndexDigest(t, uncachedIdx)
	uncachedIdx = nil
	runtime.GC()
	if uncachedDigest.leaves == 0 {
		t.Fatal("uncached real-tree index digested no values; the byte-identity check would be vacuous")
	}

	// B cold — empty cache: every distinct file misses, lexes, and Puts.
	cold := &countingWindowCache{inner: New(cacheDir, version)}
	var coldDur time.Duration
	{
		start := time.Now()
		clonescan.BuildTreeIndex(tree, cold)
		coldDur = time.Since(start)
	}
	runtime.GC()
	entries, cacheBytes := dirEntriesAndBytes(t, cacheDir)
	t.Logf("cold serialized cache: %d entries / %d bytes; fixture ceilings %d entries / %d bytes", entries, cacheBytes, distinct, dogfoodMaxBytes)
	if entries != distinct || cacheBytes > dogfoodMaxBytes {
		t.Fatalf("cold cache holds %d entries / %d bytes, want all %d distinct contents within %d bytes", entries, cacheBytes, distinct, dogfoodMaxBytes)
	}
	if cold.gets != len(tree) {
		t.Fatalf("cold run consulted the cache %d times, want once per file (%d)", cold.gets, len(tree))
	}
	if cold.hits != duplicates {
		t.Fatalf("cold run had %d hits in a fresh cache dir, want %d (only byte-identical duplicates of an earlier file may hit); retained %d entries / %d bytes within fixture ceilings %d / %d", cold.hits, duplicates, entries, cacheBytes, distinct, dogfoodMaxBytes)
	}
	if cold.puts != distinct {
		t.Fatalf("cold run put %d entries, want one per distinct file content (%d)", cold.puts, distinct)
	}

	// B warm — the cross-invocation case: every file unchanged, every Get a hit.
	warm := &countingWindowCache{inner: New(cacheDir, version)}
	var warmIdx *clonescan.TreeIndex
	warmDur := bestOf(2, func() {
		warmIdx = nil
		warmIdx = clonescan.BuildTreeIndex(tree, warm)
	})
	if warm.hits != warm.gets {
		t.Fatalf("warm hit rate %d/%d: an unchanged tree must hit on every file", warm.hits, warm.gets)
	}
	if warm.puts != 0 {
		t.Fatalf("warm run wrote %d entries; a fully-warm run must write none", warm.puts)
	}

	// Accelerate-never-gate on the real tree: warm output byte-identical to uncached.
	warmDigest := treeIndexDigest(t, warmIdx)
	warmIdx = nil
	runtime.GC()
	if warmDigest != uncachedDigest {
		t.Fatalf("warm cached index differs from the uncached index on the real tree (digest %x over %d values, want %x over %d)",
			warmDigest.sum, warmDigest.leaves, uncachedDigest.sum, uncachedDigest.leaves)
	}

	// Costs the cache adds (net-true denominators): the per-Open git resolve, and the
	// on-disk bytes a warm run reads back instead of lexing.
	resolveStart := time.Now()
	_, resolvedOK := commonDir(root)
	resolveDur := time.Since(resolveStart)
	entries, cacheBytes = dirEntriesAndBytes(t, cacheDir)
	if entries != distinct {
		t.Fatalf("cache dir holds %d entries, want one per distinct file content (%d)", entries, distinct)
	}

	hitRate := 100 * float64(warm.hits) / float64(warm.gets)
	speedup := float64(uncachedDur) / float64(warmDur)
	warmTotal := warmDur + resolveDur // one invocation pays one Open resolve
	verdict := "net-true: warm+resolve beats the uncached re-lex"
	if warmTotal >= uncachedDur {
		verdict = "not yet: warm+resolve does not beat the uncached re-lex on this box"
	}
	t.Logf("dogfood witness (#5137, WITNESSED, single box): files=%d (distinct contents %d) srcMB=%.1f", len(tree), distinct, float64(srcBytes)/(1<<20))
	t.Logf("  A  uncached   %v", uncachedDur)
	t.Logf("  B  cold       %v (put overhead %+v)", coldDur, coldDur-uncachedDur)
	t.Logf("  B  warm       %v (speedup %.2fx, hit rate %.1f%% [%d/%d])", warmDur, speedup, hitRate, warm.hits, warm.gets)
	t.Logf("  costs: git-common-dir resolve %v (ok=%v), cache disk %d entries / %.1f MB read per warm run", resolveDur, resolvedOK, entries, float64(cacheBytes)/(1<<20))
	t.Logf("  fixture budget: %d entries / %.1f MiB; measured full working set %d entries / %.1f MiB", distinct, float64(dogfoodMaxBytes)/(1<<20), entries, float64(cacheBytes)/(1<<20))
	t.Logf("  production-default fit: bytes=%v (%.1f MiB budget), entries=%v (%d budget); an over-budget working set may be evicted during Put, Open, or final maintenance", cacheBytes <= defaultMaxBytes, float64(defaultMaxBytes)/(1<<20), entries <= defaultMaxEntries, defaultMaxEntries)
	t.Logf("  verdict: %s (warm+resolve %v vs uncached %v)", verdict, warmTotal, uncachedDur)
}

// dogfoodGCPercent is the GOGC the dogfood run uses while it builds real-tree indexes:
// the heap may grow 50% past the live set instead of 100%, trading a few extra GC cycles
// for gigabytes of peak RSS on the shared CI runner.
const dogfoodGCPercent = 50

// indexDigest is a canonical fingerprint of a built index plus how many leaf values
// (strings, ints, bools) it covered.
type indexDigest struct {
	sum    [sha256.Size]byte
	leaves int
}

// treeIndexDigest streams a canonical encoding of idx into sha256 so two real-tree
// indexes can be compared without holding both in memory. The encoding mirrors
// reflect.DeepEqual: nil-vs-empty slices and maps differ, map entries are visited in
// sorted key order, and every length is length-prefixed so boundaries cannot alias. It
// reads unexported fields through reflect and fails on any kind it does not encode, so
// a future TreeIndex field can never be skipped silently.
func treeIndexDigest(t *testing.T, idx *clonescan.TreeIndex) indexDigest {
	t.Helper()
	h := sha256.New()
	var d indexDigest
	digestValue(t, h, reflect.ValueOf(idx), &d.leaves)
	copy(d.sum[:], h.Sum(nil))
	return d
}

func digestValue(t *testing.T, h hash.Hash, v reflect.Value, leaves *int) {
	var buf [binary.MaxVarintLen64]byte
	writeInt := func(n int64) { h.Write(buf[:binary.PutVarint(buf[:], n)]) }
	switch v.Kind() {
	case reflect.Pointer:
		if v.IsNil() {
			writeInt(-1)
			return
		}
		writeInt(1)
		digestValue(t, h, v.Elem(), leaves)
	case reflect.Struct:
		writeInt(int64(v.NumField()))
		for i := 0; i < v.NumField(); i++ {
			digestValue(t, h, v.Field(i), leaves)
		}
	case reflect.Slice, reflect.Array:
		if v.Kind() == reflect.Slice && v.IsNil() {
			writeInt(-1)
			return
		}
		writeInt(int64(v.Len()))
		for i := 0; i < v.Len(); i++ {
			digestValue(t, h, v.Index(i), leaves)
		}
	case reflect.Map:
		if v.IsNil() {
			writeInt(-1)
			return
		}
		if v.Type().Key().Kind() != reflect.String {
			t.Fatalf("treeIndexDigest: map key kind %s has no canonical order", v.Type().Key().Kind())
		}
		keys := v.MapKeys()
		sort.Slice(keys, func(i, j int) bool { return keys[i].String() < keys[j].String() })
		writeInt(int64(len(keys)))
		for _, k := range keys {
			digestValue(t, h, k, leaves)
			digestValue(t, h, v.MapIndex(k), leaves)
		}
	case reflect.String:
		s := v.String()
		writeInt(int64(len(s)))
		_, _ = io.WriteString(h, s)
		*leaves++
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		writeInt(v.Int())
		*leaves++
	case reflect.Bool:
		if v.Bool() {
			writeInt(1)
		} else {
			writeInt(0)
		}
		*leaves++
	default:
		t.Fatalf("treeIndexDigest: unsupported kind %s in %s", v.Kind(), v.Type())
	}
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

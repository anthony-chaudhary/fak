package clonescan

import (
	"reflect"
	"testing"
)

// spyCache is an in-memory WindowCache that records how BuildTreeIndex drives it: how
// many Gets hit, and how many Puts (i.e. recomputes) happened. It lets a test prove the
// index reads from the cache on an unchanged tree and recomputes only on a changed file.
type spyCache struct {
	store     map[string]cachedWindows
	gets      int
	hits      int
	puts      int
	maintains int
}

type cachedWindows struct {
	keys  []string
	spans []span
}

func newSpyCache() *spyCache { return &spyCache{store: map[string]cachedWindows{}} }

func (s *spyCache) Get(src string) (keys []string, spans []span, ok bool) {
	s.gets++
	v, ok := s.store[src]
	if ok {
		s.hits++
	}
	return v.keys, v.spans, ok
}

func (s *spyCache) Put(src string, keys []string, spans []span) {
	s.puts++
	s.store[src] = cachedWindows{keys: keys, spans: spans}
}

func (s *spyCache) Maintain() { s.maintains++ }

// cloneTree is two files that token-clone each other (the drift fixture's real logic
// block) plus a pure-data file that qualifies no window — so the tree exercises both a
// hit that carries windows and a hit that is legitimately empty.
func cloneTree() map[string]string {
	body := `
func %s(items []int) int {
	total := 0
	for i := 0; i < len(items); i++ {
		if items[i] > 0 {
			total += items[i] * 2
		} else {
			total -= items[i]
		}
	}
	return total
}
`
	return map[string]string{
		"a.go":    "package a\n" + replaceOnce(body, "%s", "alpha"),
		"b.go":    "package b\n" + replaceOnce(body, "%s", "beta"),
		"data.go": "package a\nvar x = 1\nvar y = 2\n",
	}
}

func replaceOnce(s, old, new string) string {
	for i := 0; i+len(old) <= len(s); i++ {
		if s[i:i+len(old)] == old {
			return s[:i] + new + s[i+len(old):]
		}
	}
	return s
}

// TestBuildTreeIndexUsesCache proves the injected cache is authoritative: the first
// build misses-then-puts every file, the second build is all hits with zero recompute,
// and both indexes answer a query byte-identically to the uncached index.
func TestBuildTreeIndexUsesCache(t *testing.T) {
	tree := cloneTree()
	want := CandidateKeys(tree["a.go"])

	uncached := BuildTreeIndex(tree).Query(want, "a.go", 0)

	sc := newSpyCache()
	first := BuildTreeIndex(tree, sc).Query(want, "a.go", 0)
	if sc.puts != len(tree) {
		t.Fatalf("first build: expected one Put per file (%d), got %d", len(tree), sc.puts)
	}
	if sc.hits != 0 {
		t.Fatalf("first build: expected 0 hits on a cold cache, got %d", sc.hits)
	}
	if sc.maintains != 1 {
		t.Fatalf("first build: expected one coalesced maintenance pass, got %d", sc.maintains)
	}

	putsAfterFirst := sc.puts
	second := BuildTreeIndex(tree, sc).Query(want, "a.go", 0)
	if sc.hits != len(tree) {
		t.Fatalf("second build: expected one hit per file (%d), got %d", len(tree), sc.hits)
	}
	if sc.puts != putsAfterFirst {
		t.Fatalf("second build recomputed: puts went %d -> %d, expected no new Put on an all-hit build", putsAfterFirst, sc.puts)
	}
	if sc.maintains != 2 {
		t.Fatalf("second build: expected one additional maintenance pass, got %d", sc.maintains)
	}

	if !reflect.DeepEqual(uncached, first) || !reflect.DeepEqual(uncached, second) {
		t.Fatalf("cached query output diverged from uncached:\n uncached=%+v\n first=%+v\n second=%+v", uncached, first, second)
	}
}

// TestBuildTreeIndexOneByteChangeMisses proves a changed file is a miss (a fresh Put),
// while its unchanged siblings stay hits — the cache is keyed on exact bytes.
func TestBuildTreeIndexOneByteChangeMisses(t *testing.T) {
	tree := cloneTree()
	sc := newSpyCache()
	BuildTreeIndex(tree, sc) // warm every file
	putsWarm := sc.puts
	hitsWarm := sc.hits

	// Flip one byte in b.go; a.go and data.go are untouched.
	tree["b.go"] += "\n// touched\nfunc extra() { _ = 1 + 1 }\n"
	BuildTreeIndex(tree, sc)

	if got := sc.puts - putsWarm; got != 1 {
		t.Fatalf("expected exactly 1 recompute (the changed file), got %d", got)
	}
	if got := sc.hits - hitsWarm; got != len(tree)-1 {
		t.Fatalf("expected %d unchanged siblings to hit, got %d", len(tree)-1, got)
	}
}

// TestBuildTreeIndexAccelerateNeverGate proves a cache that never hits (the shape of an
// unwritable disk cache) yields a byte-identical index to the nil-cache path.
func TestBuildTreeIndexAccelerateNeverGate(t *testing.T) {
	tree := cloneTree()
	want := CandidateKeys(tree["a.go"])
	nilCache := BuildTreeIndex(tree).Query(want, "a.go", 0)
	brokenCache := BuildTreeIndex(tree, brokenWindowCache{}).Query(want, "a.go", 0)
	if !reflect.DeepEqual(nilCache, brokenCache) {
		t.Fatalf("broken cache changed output:\n nil=%+v\n broken=%+v", nilCache, brokenCache)
	}
}

// brokenWindowCache always misses and drops every Put — the accelerate-never-gate
// degradation a real cache exhibits when its dir is unwritable.
type brokenWindowCache struct{}

func (brokenWindowCache) Get(string) ([]string, []span, bool) { return nil, nil, false }
func (brokenWindowCache) Put(string, []string, []span)        {}

// TestTokenizerVersionStableAndBinding proves the version tag is deterministic across
// calls (so a hit survives across invocations) and non-empty (so it can key a cache).
func TestTokenizerVersionStableAndBinding(t *testing.T) {
	v1, v2 := TokenizerVersion(), TokenizerVersion()
	if v1 != v2 {
		t.Fatalf("TokenizerVersion not deterministic: %q vs %q", v1, v2)
	}
	if v1 == "" {
		t.Fatal("TokenizerVersion is empty; a content-addressed cache cannot key on it")
	}
}

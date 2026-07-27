package hooks

import (
	"context"
	"fmt"
	"sync"
	"testing"
)

// stageddiff_cacherace_test.go — the regression witness for the hazard the #5335 per-gate budget
// CREATED. checkWithinBudget cannot cancel a gate (Gate.Check takes no context), so it ABANDONS
// one that overruns and lets the loop hand the SAME *StagedDiff to the next gate. Both gates then
// read files through it, and every such read writes fileCache. Unsynchronized that is a Go
// `concurrent map writes` — a RUNTIME FATAL, not a recoverable panic — so the timeout path would
// kill the hook it exists to let through, turning a slow commit into a crashed one.
//
// These run under `go test -race` in CI; the map-write fatal also fires WITHOUT -race whenever the
// runtime happens to detect it, so a regression here is loud either way.

// TestFileBytesIsSafeWhenAnAbandonedGateStillReadsTheSameDiff reproduces the exact shape of the
// hazard: an "abandoned" reader keeps calling FileBytes on a StagedDiff while the "next gate"
// reads through it too. Before cacheMu, this raced on fileCache.
func TestFileBytesIsSafeWhenAnAbandonedGateStillReadsTheSameDiff(t *testing.T) {
	d := &StagedDiff{
		Root:      t.TempDir(),
		ctx:       context.Background(),
		Treeish:   ":",
		fileCache: map[string]fileEntry{},
		// No runner: FileBytes falls through to the disk read, which misses and caches
		// {nil,false}. A miss still WRITES the cache, which is what makes it racy.
	}

	const readers, reads = 8, 64
	var wg sync.WaitGroup
	for r := 0; r < readers; r++ {
		wg.Add(1)
		go func(r int) {
			defer wg.Done()
			for i := 0; i < reads; i++ {
				// Overlapping key space on purpose: every reader touches the same handful of
				// paths, so the readers collide on the same map buckets rather than spreading out.
				d.FileBytes(fmt.Sprintf("docs/note-%d.md", i%4))
			}
		}(r)
	}
	wg.Wait()

	// The cache must still be coherent afterward: every key present, and each entry the
	// missing-file verdict FileBytes documents. A lost lock would more likely have crashed the
	// process than corrupt it, but a torn map would show up here.
	for i := 0; i < 4; i++ {
		rel := fmt.Sprintf("docs/note-%d.md", i)
		e, ok := d.cachedFile(rel)
		if !ok {
			t.Fatalf("cache lost %s after concurrent reads", rel)
		}
		if e.exists {
			t.Fatalf("%s: cached exists=true for a file that was never written", rel)
		}
	}
}

// TestNeighborBytesSharesTheGuardedCacheWithFileBytes pins that the DUPLICATION gate's separate
// read path (neighborBytes, which skips `git show` and reads the working tree directly) goes
// through the SAME guarded accessors. It is the one other fileCache writer, so an unguarded
// version of it would reopen the hazard while FileBytes looked safe.
func TestNeighborBytesSharesTheGuardedCacheWithFileBytes(t *testing.T) {
	d := &StagedDiff{
		Root:      t.TempDir(),
		ctx:       context.Background(),
		Treeish:   ":",
		fileCache: map[string]fileEntry{},
	}

	var wg sync.WaitGroup
	for r := 0; r < 8; r++ {
		wg.Add(1)
		go func(r int) {
			defer wg.Done()
			for i := 0; i < 64; i++ {
				rel := fmt.Sprintf("internal/hooks/sib-%d.go", i%4)
				if r%2 == 0 {
					d.neighborBytes(rel)
				} else {
					d.FileBytes(rel)
				}
			}
		}(r)
	}
	wg.Wait()

	// A warmed entry must be served, not re-read: seed one and confirm neighborBytes honors it.
	d.storeFile("internal/hooks/seeded.go", fileEntry{data: []byte("package hooks"), exists: true})
	if got, ok := d.neighborBytes("internal/hooks/seeded.go"); !ok || got != "package hooks" {
		t.Fatalf("neighborBytes ignored the guarded cache: got %q, ok=%v", got, ok)
	}
}

// TestStoreFileOnACacheLessDiffDoesNotPanic covers the hand-built StagedDiff the gate unit tests
// and publicleak_exports.go construct. Several are built without a fileCache; a nil map write is
// a panic, and a panic inside an ABANDONED gate goroutine has no recover above it — it would take
// the whole hook down on the fail-open path.
func TestStoreFileOnACacheLessDiffDoesNotPanic(t *testing.T) {
	d := &StagedDiff{Root: t.TempDir(), ctx: context.Background(), Treeish: ":"} // fileCache nil

	d.storeFile("docs/x.md", fileEntry{data: []byte("x"), exists: true})

	if _, ok := d.cachedFile("docs/x.md"); ok {
		t.Fatalf("a cache-less diff must stay cache-less, not silently grow one")
	}
	// And the read path must still work end to end over the nil cache.
	if _, ok := d.FileBytes("docs/x.md"); ok {
		t.Fatalf("FileBytes resolved a file that was never written")
	}
}

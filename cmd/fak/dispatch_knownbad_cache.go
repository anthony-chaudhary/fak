package main

// dispatch_knownbad_cache.go -- the read-side perf seam of #3471 (the write side, `fak
// knownbad compact`, shipped separately): the dispatch route hot path re-derives the
// known-bad hold on EVERY route/tick, and before this cache each derivation re-read and
// re-parsed the whole append-to-supersede ledger from disk -- monotonic O(N) work per tick
// for a file that changes rarely. The cache keys on the ledger's (mtime, size) stat pair:
// an unchanged stat serves the previously parsed records without touching the file body,
// and ANY append/compact/rotate moves the stat, so a change is picked up on the very next
// tick. It is used ONLY by the dispatch-route read (holdKnownBadForRoute); the knownbad
// CLI verbs (record/claim/resolve/revoke/compact) are read-modify-WRITE paths that must
// see a fresh ledger and keep calling readKnownBadLedger directly.

import (
	"os"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/knownbad"
)

// knownBadLedgerCache is a single-entry, stat-keyed cache over readKnownBadLedger. One
// entry is exactly enough: a dispatch process reads one workspace's ledger, and the path
// is part of the key so a (test or multi-root) caller with a different path simply misses
// and re-reads -- never serves another path's records. The mutex makes it safe under
// concurrent dispatch.
type knownBadLedgerCache struct {
	mu sync.Mutex
	// The stat key of the cached read. exists=false keys the "ledger absent" state, so
	// repeated ticks over a workspace that has never recorded a known-bad also skip the
	// filesystem read (the common case) -- a later first append flips exists and misses.
	path    string
	exists  bool
	modTime time.Time
	size    int64
	// records is the parsed fold for the key above. Callers must treat it as read-only:
	// the route hold (applyKnownBadHold / knownbad.Match) only iterates it.
	records []knownbad.Record
	primed  bool
	// parses counts the REAL read+parse passes (cache misses) -- the observable seam the
	// cache-hit test asserts on. Guarded by mu like everything else here.
	parses int
}

// dispatchKnownBadCache is the process-wide cache instance behind
// readKnownBadLedgerCached. Tests exercise fresh local instances instead.
var dispatchKnownBadCache knownBadLedgerCache

// readKnownBadLedgerCached is the dispatch-route replacement for readKnownBadLedger:
// same signature, same results, but an unchanged (mtime, size) stat serves the cached
// parse. The returned slice is shared with the cache -- read-only for callers.
func readKnownBadLedgerCached(path string) ([]knownbad.Record, error) {
	return dispatchKnownBadCache.read(path)
}

// read stats the ledger and serves the cached records when the (path, exists, mtime,
// size) key is unchanged; otherwise it falls through to readKnownBadLedger and re-keys.
// The stat happens BEFORE the read on a miss, so a write racing between the two can only
// cache NEWER content under an older key -- the next tick's stat then mismatches and
// re-reads; stale records are never served indefinitely. Errors are never cached: a
// failed read leaves the cache unprimed so the next tick retries the real path.
func (c *knownBadLedgerCache) read(path string) ([]knownbad.Record, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	var exists bool
	var modTime time.Time
	var size int64
	fi, statErr := os.Stat(path)
	switch {
	case statErr == nil:
		exists = true
		modTime = fi.ModTime()
		size = fi.Size()
	case !os.IsNotExist(statErr):
		// A stat failure that is not "absent" (permissions, transient I/O): don't guess a
		// key -- do the real read (which surfaces or absorbs the error exactly as the
		// uncached path would) and leave the cache unprimed.
		c.primed = false
		c.parses++
		return readKnownBadLedger(path)
	}

	if c.primed && c.path == path && c.exists == exists && c.size == size && c.modTime.Equal(modTime) {
		return c.records, nil
	}

	c.parses++
	records, err := readKnownBadLedger(path)
	if err != nil {
		c.primed = false
		return nil, err
	}
	c.path = path
	c.exists = exists
	c.modTime = modTime
	c.size = size
	c.records = records
	c.primed = true
	return records, nil
}

// parseCount reports how many real read+parse passes this cache has performed -- the
// witness the cache-hit test reads.
func (c *knownBadLedgerCache) parseCount() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.parses
}

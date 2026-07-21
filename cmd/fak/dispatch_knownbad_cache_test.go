package main

// Tests for the #3471 read-side cache: the dispatch-route knownbad read must serve an
// unchanged ledger from the stat-keyed cache (proven via the parses counter, which only
// the real read path increments), re-read when the ledger grows, and treat a missing
// ledger as cached-empty rather than an error.

import (
	"path/filepath"
	"reflect"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/knownbad"
)

// plantKnownBadRow appends one live, schema-stamped row through the same append helper
// the `fak knownbad record` verb uses, so the planted ledger is byte-for-byte the real
// JSONL format ParseLedger folds.
func plantKnownBadRow(t *testing.T, path, sig string) {
	t.Helper()
	rec := knownbad.Record{
		Schema:           knownbad.Schema,
		Signature:        sig,
		ReasonClass:      "COMPILE_BROKEN",
		TreeGlobs:        []string{"internal/" + sig + "/**"},
		DiscoveredBy:     "cache-test",
		DiscoveredAtUnix: time.Now().Unix(),
		TTLSeconds:       0,
		Status:           knownbad.StatusOpen,
	}
	if err := appendKnownBadRow(path, rec); err != nil {
		t.Fatalf("appendKnownBadRow(%s): %v", sig, err)
	}
}

// TestKnownBadLedgerCacheHitAndInvalidate proves (a) the second read of an unchanged
// ledger is served from cache -- the parses counter, incremented only inside the real
// read+parse path, stays at 1 across two reads returning equal records -- and (b) growing
// the ledger (content+size change) invalidates the key so the next read reflects the new
// rows.
func TestKnownBadLedgerCacheHitAndInvalidate(t *testing.T) {
	path := filepath.Join(t.TempDir(), "known-bad.jsonl")
	plantKnownBadRow(t, path, "sig-a")
	plantKnownBadRow(t, path, "sig-b")

	c := &knownBadLedgerCache{}
	first, err := c.read(path)
	if err != nil {
		t.Fatalf("first read: %v", err)
	}
	if len(first) != 2 {
		t.Fatalf("first read: got %d records, want 2", len(first))
	}
	if got := c.parseCount(); got != 1 {
		t.Fatalf("after first read: parseCount = %d, want 1", got)
	}

	second, err := c.read(path)
	if err != nil {
		t.Fatalf("second read: %v", err)
	}
	if !reflect.DeepEqual(first, second) {
		t.Fatalf("cached read differs from first read:\nfirst:  %+v\nsecond: %+v", first, second)
	}
	if got := c.parseCount(); got != 1 {
		t.Fatalf("after second read of unchanged ledger: parseCount = %d, want 1 (cache HIT must not re-parse)", got)
	}

	// (b) Grow the ledger: a third row changes content AND size, so even a same-second
	// mtime cannot mask the change -- the stat key mismatches and the read re-parses.
	plantKnownBadRow(t, path, "sig-c")
	third, err := c.read(path)
	if err != nil {
		t.Fatalf("read after append: %v", err)
	}
	if len(third) != 3 {
		t.Fatalf("read after append: got %d records, want 3 (cache must invalidate on stat change)", len(third))
	}
	if third[2].Signature != "sig-c" {
		t.Fatalf("read after append: last record = %q, want sig-c", third[2].Signature)
	}
	if got := c.parseCount(); got != 2 {
		t.Fatalf("after append: parseCount = %d, want 2 (exactly one re-parse)", got)
	}
}

// TestKnownBadLedgerCacheMissingLedger proves (c) a missing ledger returns empty without
// error, the absent state is itself cached (repeated ticks over a never-recorded
// workspace skip the filesystem read), and a first append flips the key so the new row
// is seen on the very next read.
func TestKnownBadLedgerCacheMissingLedger(t *testing.T) {
	path := filepath.Join(t.TempDir(), "known-bad.jsonl")

	c := &knownBadLedgerCache{}
	for i := 1; i <= 2; i++ {
		records, err := c.read(path)
		if err != nil {
			t.Fatalf("read %d of missing ledger: %v", i, err)
		}
		if len(records) != 0 {
			t.Fatalf("read %d of missing ledger: got %d records, want 0", i, len(records))
		}
	}
	if got := c.parseCount(); got != 1 {
		t.Fatalf("two reads of a missing ledger: parseCount = %d, want 1 (absent state must be cached)", got)
	}

	plantKnownBadRow(t, path, "sig-late")
	records, err := c.read(path)
	if err != nil {
		t.Fatalf("read after ledger appeared: %v", err)
	}
	if len(records) != 1 || records[0].Signature != "sig-late" {
		t.Fatalf("read after ledger appeared: got %+v, want one sig-late record", records)
	}
}

// TestReadKnownBadLedgerCachedWrapper covers the package-level wrapper the dispatch
// route calls: it must return the same fold as the uncached read for the same path.
func TestReadKnownBadLedgerCachedWrapper(t *testing.T) {
	path := filepath.Join(t.TempDir(), "known-bad.jsonl")
	plantKnownBadRow(t, path, "sig-wrap")

	cached, err := readKnownBadLedgerCached(path)
	if err != nil {
		t.Fatalf("readKnownBadLedgerCached: %v", err)
	}
	direct, err := readKnownBadLedger(path)
	if err != nil {
		t.Fatalf("readKnownBadLedger: %v", err)
	}
	if !reflect.DeepEqual(cached, direct) {
		t.Fatalf("cached wrapper differs from direct read:\ncached: %+v\ndirect: %+v", cached, direct)
	}
}

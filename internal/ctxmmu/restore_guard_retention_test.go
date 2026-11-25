package ctxmmu

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// TestQuarantineLedgerBoundedRetentionAndCompaction verifies that proactive
// compaction bounds active entries well below maxEntries without triggering
// ErrQuarantineLedgerCapacity or setting loadErr (#14521).
func TestQuarantineLedgerBoundedRetentionAndCompaction(t *testing.T) {
	path := filepath.Join(t.TempDir(), "quarantine.jsonl")
	l := NewQuarantineLedger(path)
	l.maxEntries = 24
	l.softCap = 18
	l.compactTarget = 12
	l.ttl = 24 * time.Hour

	now := time.Date(2026, 11, 25, 0, 0, 0, 0, time.UTC)
	l.nowFn = func() time.Time { return now }

	const total = 40
	for i := 0; i < total; i++ {
		now = now.Add(time.Second)
		digest := fmt.Sprintf("%064x", i)
		if err := l.RecordQuarantine(digest); err != nil {
			t.Fatalf("RecordQuarantine(%d) failed: %v", i, err)
		}
		if got := l.Len(); got >= l.maxEntries {
			t.Fatalf("after %d inserts, Len()=%d reached hard cap %d", i+1, got, l.maxEntries)
		}
	}

	if l.loadErr != nil {
		t.Fatalf("unexpected loadErr after bounded retention: %v", l.loadErr)
	}
	if got := l.Len(); got > l.softCap {
		t.Fatalf("final Len()=%d exceeds softCap=%d", got, l.softCap)
	}

	// Newest entry must remain quarantined; oldest entry must have been evicted.
	newest := fmt.Sprintf("%064x", total-1)
	oldest := fmt.Sprintf("%064x", 0)
	if !l.IsQuarantined(newest) {
		t.Fatalf("newest digest %s was evicted", newest)
	}
	if l.IsQuarantined(oldest) {
		t.Fatalf("oldest digest %s should have been evicted by bounded retention", oldest)
	}

	// Reload from disk and verify every WAL line is valid JSON and state matches.
	reloaded := NewQuarantineLedger(path)
	if reloaded.loadErr != nil {
		t.Fatalf("reloaded ledger has loadErr: %v", reloaded.loadErr)
	}
	if !reloaded.IsQuarantined(newest) {
		t.Fatalf("reloaded ledger missing newest digest %s", newest)
	}
	if reloaded.IsQuarantined(oldest) {
		t.Fatalf("reloaded ledger resurrected evicted digest %s", oldest)
	}

	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("open compacted WAL: %v", err)
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	lines := 0
	for scanner.Scan() {
		lines++
		var rec quarantineRecord
		if err := json.Unmarshal(scanner.Bytes(), &rec); err != nil {
			t.Fatalf("line %d torn JSON: %v", lines, err)
		}
	}
	if err := scanner.Err(); err != nil {
		t.Fatalf("scan compacted WAL: %v", err)
	}
	if lines == 0 || lines > l.softCap {
		t.Fatalf("compacted WAL line count = %d, want in (0, %d]", lines, l.softCap)
	}
}

// TestQuarantineLedgerTTLExpirationOnCompaction verifies TTL-expired entries are
// pruned when compaction runs while unexpired entries survive (#14521).
func TestQuarantineLedgerTTLExpirationOnCompaction(t *testing.T) {
	path := filepath.Join(t.TempDir(), "quarantine.jsonl")
	l := NewQuarantineLedger(path)
	l.maxEntries = 16
	l.softCap = 6
	l.compactTarget = 4
	l.ttl = time.Hour

	now := time.Date(2026, 11, 25, 0, 0, 0, 0, time.UTC)
	l.nowFn = func() time.Time { return now }

	staleDigest := fmt.Sprintf("%064x", 1)
	if err := l.RecordQuarantine(staleDigest); err != nil {
		t.Fatalf("record stale: %v", err)
	}

	// Advance past TTL and insert fresh entries to trigger compaction.
	now = now.Add(2 * time.Hour)
	for i := 2; i <= 7; i++ {
		now = now.Add(time.Second)
		if err := l.RecordQuarantine(fmt.Sprintf("%064x", i)); err != nil {
			t.Fatalf("record fresh %d: %v", i, err)
		}
	}

	if l.IsQuarantined(staleDigest) {
		t.Fatalf("TTL-expired digest %s was not pruned during compaction", staleDigest)
	}
	freshDigest := fmt.Sprintf("%064x", 7)
	if !l.IsQuarantined(freshDigest) {
		t.Fatalf("fresh digest %s should remain quarantined", freshDigest)
	}
}

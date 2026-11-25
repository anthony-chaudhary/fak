package ctxmmu

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// TestQuarantineLedgerBoundedRetentionAndCompaction verifies the hard cap holds
// without evicting live denials: once full, new denials are refused with
// ErrQuarantineLedgerCapacity and the earlier live denials keep holding (#14521).
func TestQuarantineLedgerBoundedRetentionAndCompaction(t *testing.T) {
	path := filepath.Join(t.TempDir(), "quarantine.jsonl")
	l := NewQuarantineLedger(path)
	l.maxEntries = 24
	l.softCap = 18
	l.ttl = 24 * time.Hour

	now := time.Date(2026, 11, 25, 0, 0, 0, 0, time.UTC)
	l.nowFn = func() time.Time { return now }

	const total = 40
	accepted := 0
	for i := 0; i < total; i++ {
		now = now.Add(time.Second)
		err := l.RecordQuarantine(fmt.Sprintf("%064x", i))
		if err == nil {
			accepted++
			continue
		}
		if !errors.Is(err, ErrQuarantineLedgerCapacity) {
			t.Fatalf("RecordQuarantine(%d) failed with %v, want capacity refusal", i, err)
		}
	}
	if accepted != l.maxEntries {
		t.Fatalf("accepted %d denials, want exactly maxEntries=%d before refusing", accepted, l.maxEntries)
	}
	if l.loadErr != nil {
		t.Fatalf("unexpected loadErr after bounded retention: %v", l.loadErr)
	}
	if got := l.Len(); got != accepted {
		t.Fatalf("Len()=%d, want %d live denials", got, accepted)
	}

	newest := fmt.Sprintf("%064x", accepted-1)
	oldest := fmt.Sprintf("%064x", 0)
	refused := fmt.Sprintf("%064x", total-1)
	if !l.IsQuarantined(newest) {
		t.Fatalf("newest accepted digest %s was lost", newest)
	}
	if !l.IsQuarantined(oldest) {
		t.Fatalf("oldest live digest %s was evicted to make room (fail-open)", oldest)
	}
	if l.IsQuarantined(refused) {
		t.Fatalf("refused digest %s published as quarantined", refused)
	}

	// Reload from disk and verify every WAL line is valid JSON and state matches.
	reloaded := NewQuarantineLedger(path)
	if reloaded.loadErr != nil {
		t.Fatalf("reloaded ledger has loadErr: %v", reloaded.loadErr)
	}
	if !reloaded.IsQuarantined(newest) || !reloaded.IsQuarantined(oldest) {
		t.Fatalf("reloaded ledger lost a live denial")
	}

	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("open WAL: %v", err)
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
		t.Fatalf("scan WAL: %v", err)
	}
	if lines != accepted {
		t.Fatalf("WAL line count = %d, want %d", lines, accepted)
	}
}

// TestQuarantineLedgerTTLExpirationOnCompaction verifies TTL-expired entries are
// pruned when compaction runs while unexpired entries survive (#14521).
func TestQuarantineLedgerTTLExpirationOnCompaction(t *testing.T) {
	path := filepath.Join(t.TempDir(), "quarantine.jsonl")
	l := NewQuarantineLedger(path)
	l.maxEntries = 16
	l.softCap = 6
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

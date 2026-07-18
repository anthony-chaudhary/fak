package modver

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"testing"
)

// TestAppendGuardedConcurrentStampersNoTornRows is the #2473 witness: two agents
// stamping the same module-versions ledger at once must produce a parseable
// ledger with no torn, lost, or duplicated rows. Each stamper appends a batch of
// full LedgerRows many times; without the advisory lock the concurrent O_APPEND
// writes could interleave mid-row. Under `-race` (WSL) it also proves the guarded
// append shares no unsynchronised state. The assertion is schema-level: every
// non-empty line must be valid fak-module-versions/1 JSON and every emitted row
// must be present exactly once.
func TestAppendGuardedConcurrentStampersNoTornRows(t *testing.T) {
	path := filepath.Join(t.TempDir(), "module-versions.jsonl")

	const (
		stampers = 2
		rounds   = 50
		perRound = 8 // multi-row batches widen the tear window
	)

	// Each (stamper, round, row) triple names a unique module so a missing or
	// duplicated line is detectable; each row is a full LedgerRow so the bytes
	// written per append are wide enough to tear if the append were unguarded.
	rowsFor := func(s, r int) []LedgerRow {
		out := make([]LedgerRow, perRound)
		for i := range out {
			out[i] = LedgerRow{
				Schema:     Schema,
				TS:         "2026-07-18T00:00:00Z",
				Head:       "0123456789abcdef0123456789abcdef01234567",
				Module:     fmt.Sprintf("internal/mod-s%d-r%d-i%d", s, r, i),
				Kind:       "internal",
				Rev:        r + 1,
				Version:    fmt.Sprintf("r%d+g01234567", r+1),
				LastCommit: "01234567",
				LastDate:   "2026-07-18T00:00:00Z",
			}
		}
		return out
	}

	want := map[string]bool{}
	for s := 0; s < stampers; s++ {
		for r := 0; r < rounds; r++ {
			for _, row := range rowsFor(s, r) {
				want[row.Module] = true
			}
		}
	}

	var wg sync.WaitGroup
	errs := make(chan error, stampers*rounds)
	for s := 0; s < stampers; s++ {
		wg.Add(1)
		go func(s int) {
			defer wg.Done()
			for r := 0; r < rounds; r++ {
				lines, err := AppendLines(rowsFor(s, r))
				if err != nil {
					errs <- err
					return
				}
				if err := AppendGuarded(path, lines); err != nil {
					errs <- err
					return
				}
			}
		}(s)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatalf("concurrent stamp failed: %v", err)
		}
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read ledger: %v", err)
	}
	got := map[string]int{}
	for _, line := range bytes.Split(data, []byte{'\n'}) {
		line = bytes.TrimSpace(line)
		if len(line) == 0 {
			continue
		}
		var row LedgerRow
		if err := json.Unmarshal(line, &row); err != nil {
			t.Fatalf("torn/unparseable ledger row %q: %v", line, err)
		}
		if row.Schema != Schema {
			t.Fatalf("row has schema %q, want %q: %q", row.Schema, Schema, line)
		}
		got[row.Module]++
	}

	if len(got) != len(want) {
		t.Fatalf("distinct modules in ledger = %d, want %d", len(got), len(want))
	}
	for mod := range want {
		switch got[mod] {
		case 1: // present exactly once — good
		case 0:
			t.Fatalf("module %q missing from ledger (lost row)", mod)
		default:
			t.Fatalf("module %q appears %d times (duplicate row)", mod, got[mod])
		}
	}
}

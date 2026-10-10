package snapshot

import (
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
)

// A reader racing repeated rewrites must only ever see a whole old or whole new
// envelope. The non-atomic truncate-then-write left an empty or half-written file
// visible, which is how a crash-looping serve tore its session-state snapshot.
//
// fak-test:runtime fast est=300ms lane=default
func TestSessionStateAtomicWriteNeverTorn(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "session-state.snap")
	small := map[string]string{"k": "v"}
	large := map[string]string{"k": strings.Repeat("x", 256<<10)}
	if err := WriteFile(path, KindFleet, "serve", small, nil, 1); err != nil {
		t.Fatal(err)
	}

	var stop atomic.Bool
	var torn atomic.Value
	var wg sync.WaitGroup
	for r := 0; r < 4; r++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for !stop.Load() {
				b, err := os.ReadFile(path)
				if err != nil {
					continue
				}
				if _, err := Parse(b); err != nil {
					torn.Store(err.Error())
					return
				}
			}
		}()
	}
	wrote := 0
	for i := 0; i < 200; i++ {
		body := small
		if i%2 == 0 {
			body = large
		}
		if err := WriteFile(path, KindFleet, "serve", body, nil, int64(i+2)); err == nil {
			wrote++
		}
	}
	stop.Store(true)
	wg.Wait()

	if msg, ok := torn.Load().(string); ok {
		t.Fatalf("a reader saw a torn snapshot mid-rewrite: %s", msg)
	}
	if wrote == 0 {
		t.Fatal("no rewrite succeeded")
	}
	if _, err := ReadFile(path); err != nil {
		t.Fatalf("final snapshot does not parse: %v", err)
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 {
		names := make([]string, 0, len(entries))
		for _, e := range entries {
			names = append(names, e.Name())
		}
		t.Fatalf("dir holds %v, want only the snapshot (no leftover temp files)", names)
	}
}

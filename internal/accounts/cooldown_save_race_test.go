package accounts

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"
)

// The fleet-shared cooldown store is written concurrently by every checkout, launch
// exit and watchdog tick. Save must therefore stage into a name no peer saver can
// also be holding open: a fixed `<path>.tmp` let two savers write their own payloads
// into ONE file from offset 0, and the shorter payload published the longer one's
// leftover tail — the stray trailing `}` seen twice in the live store (#6027).

// TestSaveDoesNotStageIntoTheFixedSiblingTmp pins the invariant deterministically:
// a peer's staging file sitting at the pre-fix path survives a Save untouched.
func TestSaveDoesNotStageIntoTheFixedSiblingTmp(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "account-cooldown.json")
	peer := path + ".tmp"
	const sentinel = "a peer saver's in-flight payload"
	if err := os.WriteFile(peer, []byte(sentinel), 0o644); err != nil {
		t.Fatalf("seed peer staging file: %v", err)
	}

	s, err := LoadCooldownStore(path)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	base := mustTime(t, "2026-08-12T12:00:00Z")
	s.Cool("uuid:aaa", CooldownUsageLimit, "test", base, base.Add(time.Hour))
	if err := s.Save(); err != nil {
		t.Fatalf("save: %v", err)
	}

	got, err := os.ReadFile(peer)
	if err != nil {
		t.Fatalf("peer staging file was consumed by Save (it must stage into a unique name): %v", err)
	}
	if string(got) != sentinel {
		t.Fatalf("peer staging file overwritten by Save: got %q want %q", got, sentinel)
	}
}

// TestConcurrentSavesAlwaysPublishParseableStore exercises the real shape: many
// savers, payloads that differ in LENGTH (RFC3339Nano trims trailing zeros, so the
// reset timestamps alone vary the byte count), racing on one path. Every published
// file must parse — last-writer-wins is acceptable loss, a spliced file is not.
func TestConcurrentSavesAlwaysPublishParseableStore(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "account-cooldown.json")
	base := mustTime(t, "2026-08-12T12:00:00Z")

	const savers, rounds = 8, 40
	var wg sync.WaitGroup
	for i := 0; i < savers; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			for r := 0; r < rounds; r++ {
				s, err := LoadCooldownStore(path)
				if err != nil {
					t.Errorf("saver %d round %d: load published a corrupt store: %v", i, r, err)
					return
				}
				// Vary both the entry count and the timestamp precision so the
				// encoded payloads differ in length between savers.
				for n := 0; n <= i; n++ {
					reset := base.Add(time.Duration(i+r) * time.Nanosecond).Add(time.Hour)
					s.Cool(fmt.Sprintf("uuid:%d-%d", i, n), CooldownUsageLimit, "race", base, reset)
				}
				if err := s.Save(); err != nil {
					t.Errorf("saver %d round %d: save: %v", i, r, err)
					return
				}
			}
		}(i)
	}
	wg.Wait()

	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read published store: %v", err)
	}
	if _, err := LoadCooldownStore(path); err != nil {
		t.Fatalf("published store does not parse (%v); contents:\n%s", err, raw)
	}

	// No staging file may outlive the writes that made it.
	leftovers, err := filepath.Glob(filepath.Join(dir, "*.tmp"))
	if err != nil {
		t.Fatalf("glob: %v", err)
	}
	if len(leftovers) != 0 {
		t.Fatalf("staging files left behind: %v", leftovers)
	}
}

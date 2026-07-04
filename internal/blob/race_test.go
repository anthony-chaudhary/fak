package blob

// race_test.go — regression guard for the store's concurrency design.
//
// Resolve takes only the RLock (it is intentionally concurrent — many goroutines
// may resolve the same shared payload at once, e.g. a K-arm replay), so the resolv
// counter it bumps MUST be atomic: a plain increment loses updates and data-races
// under `go test -race`. The store had exactly this race and fixed it with the
// atomic in Resolve/Stats (see store.go), but every other test is single-goroutine,
// so nothing exercised that seam — a refactor reverting the atomic to `resolv++`
// would keep CI green. This pins it. The assertions are EXACT, so a lost increment
// fails deterministically here even on a box without the race detector; CI's -race
// job catches the data race directly.

import (
	"bytes"
	"context"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func TestConcurrentResolveKeepsResolvCounterExact(t *testing.T) {
	ctx := context.Background()
	s := newStore(0) // unbounded: no eviction, so the shared ref always resolves and counts are deterministic

	payload := makeBytes(InlineMax+64, 42) // > InlineMax → CAS-resident (RefBlob), so Resolve takes the RLock path
	ref, err := s.Put(ctx, payload)
	if err != nil {
		t.Fatalf("setup Put: %v", err)
	}
	if ref.Kind != abi.RefBlob {
		t.Fatalf("setup: want RefBlob so Resolve hits the CAS/RLock path, got Kind=%d", ref.Kind)
	}

	const (
		readers         = 8
		resolvesPerRead = 2000
		writers         = 4
		putsPerWriter   = 1000
	)

	var wg sync.WaitGroup

	// Concurrent resolvers: all read the one shared CAS blob under the RLock, each
	// bumping resolv atomically. This is the exact shape that tripped the race.
	for r := 0; r < readers; r++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < resolvesPerRead; i++ {
				got, err := s.Resolve(ctx, ref)
				if err != nil {
					t.Errorf("concurrent Resolve: %v", err)
					return
				}
				if !bytes.Equal(got, payload) {
					t.Errorf("concurrent Resolve returned corrupt bytes")
					return
				}
			}
		}()
	}

	// Concurrent writers: Put DISTINCT payloads (write Lock, mutating the blobs map)
	// while the readers hold the RLock — exercises the RWMutex on the map itself.
	for w := 0; w < writers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for i := 0; i < putsPerWriter; i++ {
				if _, err := s.Put(ctx, distinctPayload(w*putsPerWriter+i+1, InlineMax+8)); err != nil {
					t.Errorf("concurrent Put: %v", err)
					return
				}
			}
		}(w)
	}

	wg.Wait()

	puts, hits, resolves := s.Stats()
	// resolv is the racy counter: with the atomic it is EXACT; a plain increment
	// would lose updates under this contention (and -race would flag the write).
	if want := int64(readers * resolvesPerRead); resolves != want {
		t.Fatalf("resolves=%d, want %d (lost increments ⇒ resolv is not atomic under the RLock)", resolves, want)
	}
	// Distinct payloads never dedup, so every writer Put is a fresh store, no hits.
	if want := int64(1 + writers*putsPerWriter); puts != want {
		t.Fatalf("puts=%d, want %d", puts, want)
	}
	if hits != 0 {
		t.Fatalf("hits=%d, want 0 (distinct payloads must not dedup)", hits)
	}
	// The shared blob plus every distinct writer payload are all resident (unbounded).
	if want := 1 + writers*putsPerWriter; s.Len() != want {
		t.Fatalf("Len=%d, want %d", s.Len(), want)
	}
}

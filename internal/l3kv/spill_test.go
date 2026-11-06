package l3kv

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"
)

// digestAt renders n as a distinct 64-char lowercase-hex span digest, the
// validKey shape the durable store requires.
func digestAt(n uint64) string { return fmt.Sprintf("%064x", n) }

// errSpillTestDiskFull is the shared sentinel failStore reports, so the test
// can match the propagated fault with errors.Is.
var errSpillTestDiskFull = errors.New("disk full")

// spillGateStore delays every Put behind gate, then writes through to inner
// on a fresh background context — it ignores the caller's (already
// timed-out) context, the way a real disk save can finish after the bounded
// window closed. (Distinct from async_job_test.go's slowStore, which honors
// its context.)
type spillGateStore struct {
	inner Store
	gate  chan struct{}
}

func (s *spillGateStore) Put(ctx context.Context, key string, payload []byte) error {
	<-s.gate
	return s.inner.Put(context.Background(), key, payload)
}

func (s *spillGateStore) Get(ctx context.Context, key string) ([]byte, bool, error) {
	return s.inner.Get(ctx, key)
}

// failStore faults every Put, standing in for a disk that rejects the write.
type failStore struct{ inner Store }

func (s *failStore) Put(ctx context.Context, key string, payload []byte) error {
	return errSpillTestDiskFull
}

func (s *failStore) Get(ctx context.Context, key string) ([]byte, bool, error) {
	return s.inner.Get(ctx, key)
}

// fak-test:runtime fast est=500ms lane=default
func TestSpillEvictionCapsAndRestorePath(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	t.Run("stages-owned-copy-restorable-on-durable-path", func(t *testing.T) {
		store, err := NewDiskStore(t.TempDir())
		if err != nil {
			t.Fatalf("NewDiskStore: %v", err)
		}
		sp, err := NewSpiller(store, DefaultSpillConfig)
		if err != nil {
			t.Fatalf("NewSpiller: %v", err)
		}
		payload := []byte("span-residency-bytes")
		if err := sp.Spill(ctx, digestAt(1), payload); err != nil {
			t.Fatalf("Spill = %v, want nil", err)
		}
		got, found, err := store.Get(ctx, digestAt(1))
		if err != nil || !found {
			t.Fatalf("store.Get = (%q, %v, %v), want staged hit", got, found, err)
		}
		if !bytes.Equal(got, payload) {
			t.Fatalf("restored = %q, want %q", got, payload)
		}
		if st := sp.Stats(); st.Spilled != 1 || st.DiskBytes != int64(len(payload)) {
			t.Fatalf("Stats = %+v, want Spilled=1 DiskBytes=%d", st, len(payload))
		}
	})

	t.Run("entry-cap-skips-oversized", func(t *testing.T) {
		store, err := NewDiskStore(t.TempDir())
		if err != nil {
			t.Fatalf("NewDiskStore: %v", err)
		}
		sp, err := NewSpiller(store, SpillConfig{MaxEntryBytes: 8})
		if err != nil {
			t.Fatalf("NewSpiller: %v", err)
		}
		payload := make([]byte, 16)
		for i := range payload {
			payload[i] = byte(i)
		}
		if err := sp.Spill(ctx, digestAt(2), payload); err != nil {
			t.Fatalf("Spill = %v, want nil (cap skip is not an error)", err)
		}
		if _, found, err := store.Get(ctx, digestAt(2)); err != nil || found {
			t.Fatalf("store.Get = (..., %v, %v), want clean miss", found, err)
		}
		if st := sp.Stats(); st.SkippedOversized != 1 || st.Spilled != 0 {
			t.Fatalf("Stats = %+v, want SkippedOversized=1", st)
		}
	})

	t.Run("disk-budget-skips-past-cap", func(t *testing.T) {
		store, err := NewDiskStore(t.TempDir())
		if err != nil {
			t.Fatalf("NewDiskStore: %v", err)
		}
		sp, err := NewSpiller(store, SpillConfig{MaxDiskBytes: 20})
		if err != nil {
			t.Fatalf("NewSpiller: %v", err)
		}
		// 10 + 11 = 21 strictly exceeds the 20-byte budget; exactly 20 would
		// still fit under the reference contract's ">" check.
		first, second := make([]byte, 10), make([]byte, 11)
		if err := sp.Spill(ctx, digestAt(3), first); err != nil {
			t.Fatalf("first Spill = %v, want nil", err)
		}
		if err := sp.Spill(ctx, digestAt(4), second); err != nil {
			t.Fatalf("second Spill = %v, want nil (budget skip is not an error)", err)
		}
		if _, found, err := store.Get(ctx, digestAt(3)); err != nil || !found {
			t.Fatalf("first record Get = (..., %v, %v), want hit", found, err)
		}
		if _, found, err := store.Get(ctx, digestAt(4)); err != nil || found {
			t.Fatalf("second record Get = (..., %v, %v), want clean miss", found, err)
		}
		if st := sp.Stats(); st.SkippedBudget != 1 || st.Spilled != 1 || st.DiskBytes != 10 {
			t.Fatalf("Stats = %+v, want SkippedBudget=1 Spilled=1 DiskBytes=10", st)
		}
	})

	t.Run("bounded-window-skip", func(t *testing.T) {
		inner, err := NewDiskStore(t.TempDir())
		if err != nil {
			t.Fatalf("NewDiskStore: %v", err)
		}
		gate := make(chan struct{})
		store := &spillGateStore{inner: inner, gate: gate}
		sp, err := NewSpiller(store, SpillConfig{PutTimeout: 50 * time.Millisecond})
		if err != nil {
			t.Fatalf("NewSpiller: %v", err)
		}
		payload := []byte("late-save-bytes")
		start := time.Now()
		if err := sp.Spill(ctx, digestAt(5), payload); err != nil {
			t.Fatalf("Spill = %v, want nil (window skip is not an error)", err)
		}
		if elapsed := time.Since(start); elapsed >= 5*time.Second {
			t.Fatalf("Spill took %v, want the 50 ms window to close long before the gate opens", elapsed)
		}
		if st := sp.Stats(); st.SkippedTimeout != 1 || st.DiskBytes != 0 || st.Spilled != 0 {
			t.Fatalf("Stats after window close = %+v, want SkippedTimeout=1 DiskBytes=0", st)
		}
		// The store ignored its context: the save finishes late as an
		// idempotent overwrite, uncredited by the stats.
		close(gate)
		waitFor(t, "late idempotent spill write to land", 2*time.Second, func() bool {
			_, found, err := inner.Get(ctx, digestAt(5))
			return err == nil && found
		})
		if st := sp.Stats(); st.Spilled != 0 || st.DiskBytes != 0 || st.SkippedTimeout != 1 {
			t.Fatalf("Stats after late write = %+v, want Spilled=0 (uncredited)", st)
		}
	})

	t.Run("checkpoint-slots-refused-without-store", func(t *testing.T) {
		real, err := NewDiskStore(t.TempDir())
		if err != nil {
			t.Fatalf("NewDiskStore: %v", err)
		}
		if _, err := NewSpiller(nil, SpillConfig{CheckpointSlots: 2}); err == nil || !strings.Contains(err.Error(), "checkpoint slots") {
			t.Fatalf("NewSpiller(nil, slots=2) = %v, want refusal naming checkpoint slots", err)
		}
		sp, err := NewSpiller(real, SpillConfig{CheckpointSlots: 2})
		if err != nil {
			t.Fatalf("NewSpiller(store, slots=2) = %v, want ok", err)
		}
		if got := sp.CheckpointSlots(); got != 2 {
			t.Fatalf("CheckpointSlots = %d, want 2", got)
		}
		neg, err := NewSpiller(real, SpillConfig{CheckpointSlots: -3})
		if err != nil {
			t.Fatalf("NewSpiller(store, slots=-3) = %v, want ok (normalized)", err)
		}
		if got := neg.CheckpointSlots(); got != 0 {
			t.Fatalf("CheckpointSlots after negative normalization = %d, want 0", got)
		}
	})

	t.Run("empty-payload-is-noop", func(t *testing.T) {
		sp, err := NewSpiller(newMemStore(), DefaultSpillConfig)
		if err != nil {
			t.Fatalf("NewSpiller: %v", err)
		}
		if err := sp.Spill(ctx, digestAt(6), nil); err != nil {
			t.Fatalf("Spill(nil payload) = %v, want nil", err)
		}
		if st := sp.Stats(); st != (SpillStats{}) {
			t.Fatalf("Stats = %+v, want zero value", st)
		}
	})

	t.Run("caller-may-reuse-slice", func(t *testing.T) {
		store := newMemStore()
		sp, err := NewSpiller(store, DefaultSpillConfig)
		if err != nil {
			t.Fatalf("NewSpiller: %v", err)
		}
		original := []byte("reusable-buffer")
		payload := append([]byte(nil), original...)
		if err := sp.Spill(ctx, digestAt(7), payload); err != nil {
			t.Fatalf("Spill = %v, want nil", err)
		}
		for i := range payload {
			payload[i] = 'X'
		}
		got, found, err := store.Get(ctx, digestAt(7))
		if err != nil || !found || !bytes.Equal(got, original) {
			t.Fatalf("store.Get = (%q, %v, %v), want the original bytes", got, found, err)
		}
	})

	t.Run("fault-propagates", func(t *testing.T) {
		store := &failStore{inner: newMemStore()}
		sp, err := NewSpiller(store, DefaultSpillConfig)
		if err != nil {
			t.Fatalf("NewSpiller: %v", err)
		}
		if err := sp.Spill(ctx, digestAt(8), []byte("doomed")); !errors.Is(err, errSpillTestDiskFull) {
			t.Fatalf("Spill = %v, want the store fault", err)
		}
		if st := sp.Stats(); st.Faults != 1 || st.Spilled != 0 {
			t.Fatalf("Stats = %+v, want Faults=1", st)
		}
	})

	t.Run("concurrent-spills", func(t *testing.T) {
		store, err := NewDiskStore(t.TempDir())
		if err != nil {
			t.Fatalf("NewDiskStore: %v", err)
		}
		sp, err := NewSpiller(store, SpillConfig{MaxDiskBytes: 1 << 20})
		if err != nil {
			t.Fatalf("NewSpiller: %v", err)
		}
		payload := []byte("concurrent-spill")
		const workers, perWorker = 8, 25
		var wg sync.WaitGroup
		for w := 0; w < workers; w++ {
			wg.Add(1)
			go func(w int) {
				defer wg.Done()
				for i := 0; i < perWorker; i++ {
					d := digestAt(uint64(w)*perWorker + uint64(i) + 1)
					if err := sp.Spill(ctx, d, payload); err != nil {
						t.Errorf("Spill(%s) = %v, want nil", d, err)
					}
				}
			}(w)
		}
		wg.Wait()
		want := workers * perWorker
		if st := sp.Stats(); st.Spilled != want || st.DiskBytes != int64(want)*int64(len(payload)) {
			t.Fatalf("Stats = %+v, want Spilled=%d DiskBytes=%d", st, want, want*len(payload))
		}
	})
}

// fak-test:runtime fast est=100ms lane=default
func TestAsyncStoreSpillSeam(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	var nilStore *AsyncStore
	if err := nilStore.Spill(ctx, digestAt(100), []byte("x")); !errors.Is(err, ErrSpillDisabled) {
		t.Fatalf("nil AsyncStore.Spill = %v, want ErrSpillDisabled", err)
	}
	if st := nilStore.SpillStats(); st != (SpillStats{}) {
		t.Fatalf("nil AsyncStore.SpillStats = %+v, want zero value", st)
	}

	store, err := NewDiskStore(t.TempDir())
	if err != nil {
		t.Fatalf("NewDiskStore: %v", err)
	}
	as := NewAsyncStore(store)
	if err := as.Spill(ctx, digestAt(101), []byte("no-spill-yet")); !errors.Is(err, ErrSpillDisabled) {
		t.Fatalf("pre-enable Spill = %v, want ErrSpillDisabled", err)
	}
	if st := as.SpillStats(); st != (SpillStats{}) {
		t.Fatalf("pre-enable SpillStats = %+v, want zero value", st)
	}

	sp, err := as.EnableSpill(DefaultSpillConfig)
	if err != nil {
		t.Fatalf("EnableSpill: %v", err)
	}
	payload := []byte("seam-span-bytes")
	if err := as.Spill(ctx, digestAt(102), payload); err != nil {
		t.Fatalf("enabled Spill = %v, want nil", err)
	}
	got, found, err := store.Get(ctx, digestAt(102))
	if err != nil || !found || !bytes.Equal(got, payload) {
		t.Fatalf("store.Get = (%q, %v, %v), want staged hit", got, found, err)
	}
	if st := as.SpillStats(); st.Spilled != 1 || st.DiskBytes != int64(len(payload)) {
		t.Fatalf("AsyncStore.SpillStats = %+v, want Spilled=1 DiskBytes=%d", st, len(payload))
	}
	if st := sp.Stats(); st.Spilled != 1 {
		t.Fatalf("returned spiller Stats = %+v, want Spilled=1 (same instance behind the seam)", st)
	}
}

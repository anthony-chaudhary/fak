package l3kv

import (
	"context"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

type memoryStore struct {
	mu   sync.Mutex
	data map[string][]byte
}

func newMemoryStore() *memoryStore {
	return &memoryStore{data: map[string][]byte{}}
}

func (m *memoryStore) Put(_ context.Context, key string, payload []byte) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.data[key] = append([]byte(nil), payload...)
	return nil
}

func (m *memoryStore) Get(_ context.Context, key string) ([]byte, bool, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	val, ok := m.data[key]
	if !ok {
		return nil, false, nil
	}
	return append([]byte(nil), val...), true, nil
}

// TestAsyncWatermarkResetDropsStaleCompletions implements the required acceptance test of #10729:
// 1. Dispatch three background storage tasks.
// 2. Trigger Reset().
// 3. Allow tasks to complete and verify their results are discarded (state not mutated).
// 4. Dispatch a subsequent task, verify it completes and mutates state.
func TestAsyncWatermarkResetDropsStaleCompletions(t *testing.T) {
	wm := NewAsyncWatermarkManager()

	job1 := wm.Dispatch("key1", []byte("val1"))
	job2 := wm.Dispatch("key2", []byte("val2"))
	job3 := wm.Dispatch("key3", []byte("val3"))

	if job1.ID != 1 || job2.ID != 2 || job3.ID != 3 {
		t.Fatalf("unexpected job IDs: %d, %d, %d", job1.ID, job2.ID, job3.ID)
	}

	// Trigger Reset: watermark advances to 3
	wmThreshold := wm.Reset()
	if wmThreshold != 3 {
		t.Fatalf("Reset returned watermark=%d, want 3", wmThreshold)
	}

	// Verify all three initial jobs are identified as stale
	if !wm.IsStale(job1.ID) || !wm.IsStale(job2.ID) || !wm.IsStale(job3.ID) {
		t.Fatalf("expected jobs 1, 2, 3 to be stale, got isStale=(%v, %v, %v)",
			wm.IsStale(job1.ID), wm.IsStale(job2.ID), wm.IsStale(job3.ID))
	}

	// Attempt to complete the three stale jobs: callbacks must NOT run
	var stateMutated int32
	committed1 := wm.Complete(job1, func(j AsyncJob) {
		atomic.AddInt32(&stateMutated, 1)
	})
	committed2 := wm.Complete(job2, func(j AsyncJob) {
		atomic.AddInt32(&stateMutated, 1)
	})
	committed3 := wm.Complete(job3, func(j AsyncJob) {
		atomic.AddInt32(&stateMutated, 1)
	})

	if committed1 || committed2 || committed3 {
		t.Fatalf("stale jobs should not commit, got (%v, %v, %v)", committed1, committed2, committed3)
	}
	if atomic.LoadInt32(&stateMutated) != 0 {
		t.Fatalf("stale job completions mutated state! stateMutated=%d, want 0", stateMutated)
	}

	st := wm.Stats()
	if st.DroppedCount != 3 || st.CompletedCount != 0 {
		t.Fatalf("stats after stale completions: dropped=%d completed=%d, want 3/0", st.DroppedCount, st.CompletedCount)
	}

	// Dispatch job 4 after Reset: must be valid and commit
	job4 := wm.Dispatch("key4", []byte("val4"))
	if job4.ID != 4 {
		t.Fatalf("job4 ID = %d, want 4", job4.ID)
	}
	if wm.IsStale(job4.ID) {
		t.Fatalf("job4 should not be stale, watermark=%d", wm.Stats().StaleWatermark)
	}

	committed4 := wm.Complete(job4, func(j AsyncJob) {
		atomic.AddInt32(&stateMutated, 1)
	})
	if !committed4 {
		t.Fatalf("job4 failed to commit")
	}
	if atomic.LoadInt32(&stateMutated) != 1 {
		t.Fatalf("expected stateMutated=1, got %d", stateMutated)
	}

	st = wm.Stats()
	if st.DroppedCount != 3 || st.CompletedCount != 1 {
		t.Fatalf("final stats: dropped=%d completed=%d, want 3/1", st.DroppedCount, st.CompletedCount)
	}
}

// TestAsyncStoreEndToEnd tests AsyncStore PutAsync with concurrent Reset.
func TestAsyncStoreEndToEnd(t *testing.T) {
	mem := newMemoryStore()
	store := NewAsyncStore(mem)

	blocker := make(chan struct{})
	var callbackRan int32

	// Wrap mem to simulate slow disk I/O
	slowMem := &slowStore{inner: mem, block: blocker}
	slowAsyncStore := NewAsyncStore(slowMem)

	// Dispatch slow job
	slowAsyncStore.PutAsync(context.Background(), "slow_key", []byte("slow_val"), func(err error) {
		atomic.AddInt32(&callbackRan, 1)
	})

	// Advance watermark by resetting
	slowAsyncStore.Reset()

	// Unblock slow job
	close(blocker)

	// Wait deterministically for dropped stale job completion
	deadline := time.Now().Add(2 * time.Second)
	for slowAsyncStore.WatermarkManager().Stats().DroppedCount == 0 && time.Now().Before(deadline) {
		runtime.Gosched()
	}

	if atomic.LoadInt32(&callbackRan) != 0 {
		t.Fatalf("callback ran for stale async write after Reset()")
	}
	if slowAsyncStore.WatermarkManager().Stats().DroppedCount != 1 {
		t.Fatalf("expected 1 dropped job, got %d", slowAsyncStore.WatermarkManager().Stats().DroppedCount)
	}

	// Now dispatch fresh job on store without block
	var freshCallbackRan int32
	var freshWg sync.WaitGroup
	freshWg.Add(1)
	store.PutAsync(context.Background(), "fresh_key", []byte("fresh_val"), func(err error) {
		defer freshWg.Done()
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		atomic.AddInt32(&freshCallbackRan, 1)
	})
	freshWg.Wait()

	if atomic.LoadInt32(&freshCallbackRan) != 1 {
		t.Fatalf("fresh callback did not run")
	}

	val, found, err := mem.Get(context.Background(), "fresh_key")
	if err != nil || !found || string(val) != "fresh_val" {
		t.Fatalf("Get fresh_key = (%q, %v, %v), want fresh_val/true/nil", string(val), found, err)
	}
}

type slowStore struct {
	inner Store
	block chan struct{}
}

func (s *slowStore) Put(ctx context.Context, key string, payload []byte) error {
	<-s.block
	return s.inner.Put(ctx, key, payload)
}

func (s *slowStore) Get(ctx context.Context, key string) ([]byte, bool, error) {
	return s.inner.Get(ctx, key)
}

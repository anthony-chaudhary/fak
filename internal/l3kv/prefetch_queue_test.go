package l3kv

import (
	"bytes"
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// gateStore is a Store whose Get blocks on a per-key release channel, so a
// read can be held in flight and its permit observed without real disk I/O.
type gateStore struct {
	mu      sync.Mutex
	data    map[string][]byte
	gates   map[string]chan struct{}
	current int32 // in-flight Gets, atomic
	peak    int32 // max concurrent Gets, atomic
}

func newGateStore(data map[string][]byte) *gateStore {
	return &gateStore{data: data, gates: map[string]chan struct{}{}}
}

func (g *gateStore) setGate(key string, gate chan struct{}) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.gates[key] = gate
}

func (g *gateStore) Put(_ context.Context, key string, payload []byte) error {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.data[key] = append([]byte(nil), payload...)
	return nil
}

func (g *gateStore) Get(_ context.Context, key string) ([]byte, bool, error) {
	cur := atomic.AddInt32(&g.current, 1)
	for {
		old := atomic.LoadInt32(&g.peak)
		if cur <= old || atomic.CompareAndSwapInt32(&g.peak, old, cur) {
			break
		}
	}
	g.mu.Lock()
	val, ok := g.data[key]
	gate := g.gates[key]
	g.mu.Unlock()
	if gate != nil {
		<-gate
	}
	atomic.AddInt32(&g.current, -1)
	if !ok {
		return nil, false, nil
	}
	return append([]byte(nil), val...), true, nil
}

func (g *gateStore) inFlight() int { return int(atomic.LoadInt32(&g.current)) }

func (g *gateStore) peakConcurrency() int { return int(atomic.LoadInt32(&g.peak)) }

// ctxAwareStore is a Store whose Get blocks until either its context is
// cancelled or an external release fires, modelling a cooperative disk read
// that retires on queue shutdown without external release.
type ctxAwareStore struct {
	mu      sync.Mutex
	data    map[string][]byte
	release chan struct{} // optional external release; nil keeps the read held
	current int32         // in-flight Gets, atomic
}

func newCtxAwareStore(data map[string][]byte) *ctxAwareStore {
	return &ctxAwareStore{data: data}
}

func (s *ctxAwareStore) Put(_ context.Context, key string, payload []byte) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.data[key] = append([]byte(nil), payload...)
	return nil
}

func (s *ctxAwareStore) Get(ctx context.Context, key string) ([]byte, bool, error) {
	atomic.AddInt32(&s.current, 1)
	select {
	case <-ctx.Done():
		atomic.AddInt32(&s.current, -1)
		return nil, false, ctx.Err()
	case <-s.release:
	}
	atomic.AddInt32(&s.current, -1)
	s.mu.Lock()
	val, ok := s.data[key]
	s.mu.Unlock()
	if !ok {
		return nil, false, nil
	}
	return append([]byte(nil), val...), true, nil
}

func (s *ctxAwareStore) inFlight() int { return int(atomic.LoadInt32(&s.current)) }

func waitFor(t *testing.T, what string, within time.Duration, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(within)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(2 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for %s", what)
}

// fak-test:runtime fast est=300ms lane=default
func TestRestorePrefetchPermitTimeoutFallback(t *testing.T) {
	t.Parallel()

	t.Run("held read times out then falls back and stages after I/O returns", func(t *testing.T) {
		store := newGateStore(map[string][]byte{"span-a": []byte("payload-a")})
		gateA := make(chan struct{})
		store.setGate("span-a", gateA)
		pq := NewPrefetchQueue(store, PrefetchConfig{Timeout: 100 * time.Millisecond, MaxConcurrentReads: 2})
		t.Cleanup(pq.Stop)

		if err := pq.Enqueue("span-a"); err != nil {
			t.Fatalf("Enqueue = %v, want nil", err)
		}
		waitFor(t, "read admitted onto an I/O worker", 2*time.Second, func() bool {
			return pq.Stats().InFlight == 1
		})

		// The held read outlives its bound: the waiter is refused with the
		// fallback sentinel, not a payload.
		payload, found, err := pq.Restore(context.Background(), "span-a")
		if !errors.Is(err, ErrPrefetchTimeout) {
			t.Fatalf("Restore = (%q, %v, %v), want ErrPrefetchTimeout", payload, found, err)
		}
		if found || payload != nil {
			t.Fatalf("timed-out restore reported a hit: found=%v payload=%q", found, payload)
		}

		// A cancelled waiter does not cancel the read: the permit is kept
		// until the I/O returns.
		if got := pq.Stats().InFlight; got != 1 {
			t.Fatalf("InFlight after waiter timeout = %d, want 1 (permit kept)", got)
		}

		close(gateA)
		waitFor(t, "in-flight read to complete and release its permit", 2*time.Second, func() bool {
			return pq.Stats().InFlight == 0
		})

		// The late-completing read staged its payload for the next restore.
		payload, found, err = pq.Restore(context.Background(), "span-a")
		if err != nil || !found || !bytes.Equal(payload, []byte("payload-a")) {
			t.Fatalf("post-I/O Restore = (%q, %v, %v), want staged hit", payload, found, err)
		}
		if st := pq.Stats(); st.StagedBytes != 0 || st.StagedEntries != 0 {
			t.Fatalf("consumed hit still staged: %+v", st)
		}
	})

	t.Run("concurrent read permits cap in-flight reads", func(t *testing.T) {
		store := newGateStore(map[string][]byte{"a": []byte("A"), "b": []byte("B")})
		gateA, gateB := make(chan struct{}), make(chan struct{})
		store.setGate("a", gateA)
		store.setGate("b", gateB)
		pq := NewPrefetchQueue(store, PrefetchConfig{Timeout: 5 * time.Second, MaxConcurrentReads: 1})
		t.Cleanup(pq.Stop)

		if err := pq.Enqueue("a"); err != nil {
			t.Fatalf("Enqueue(a) = %v", err)
		}
		if err := pq.Enqueue("b"); err != nil {
			t.Fatalf("Enqueue(b) = %v", err)
		}
		waitFor(t, "first read admitted", 2*time.Second, func() bool {
			return pq.Stats().InFlight == 1
		})

		// While the single permit is held by read "a", read "b" must queue,
		// not start: in-flight stays at exactly 1.
		time.Sleep(50 * time.Millisecond)
		if got := pq.Stats().InFlight; got != 1 {
			t.Fatalf("InFlight while permit held = %d, want 1", got)
		}

		close(gateA)
		close(gateB) // b may start only after a releases the single permit
		waitFor(t, "both reads to finish", 2*time.Second, func() bool {
			return pq.Stats().InFlight == 0 && pq.Stats().Completed == 2
		})
		if got := store.peakConcurrency(); got != 1 {
			t.Fatalf("peak concurrent reads = %d, want 1 (permit cap)", got)
		}
		if p, found, err := pq.Restore(context.Background(), "b"); err != nil || !found || !bytes.Equal(p, []byte("B")) {
			t.Fatalf("Restore(b) = (%q, %v, %v), want staged hit", p, found, err)
		}
	})

	t.Run("zero permits disable read ahead", func(t *testing.T) {
		store := newGateStore(map[string][]byte{"k": []byte("direct")})
		pq := NewPrefetchQueue(store, PrefetchConfig{MaxConcurrentReads: 0})
		t.Cleanup(pq.Stop)

		if err := pq.Enqueue("k"); !errors.Is(err, ErrPrefetchDisabled) {
			t.Fatalf("Enqueue = %v, want ErrPrefetchDisabled", err)
		}
		// The demand path still works: Restore reads directly.
		p, found, err := pq.Restore(context.Background(), "k")
		if err != nil || !found || !bytes.Equal(p, []byte("direct")) {
			t.Fatalf("direct Restore = (%q, %v, %v), want hit", p, found, err)
		}
	})

	t.Run("cancelled waiter keeps the read running until I/O returns", func(t *testing.T) {
		store := newGateStore(map[string][]byte{"k": []byte("kept")})
		gate := make(chan struct{})
		store.setGate("k", gate)
		pq := NewPrefetchQueue(store, PrefetchConfig{Timeout: 5 * time.Second, MaxConcurrentReads: 2})
		t.Cleanup(pq.Stop)

		if err := pq.Enqueue("k"); err != nil {
			t.Fatalf("Enqueue = %v", err)
		}
		waitFor(t, "read admitted", 2*time.Second, func() bool {
			return pq.Stats().InFlight == 1
		})

		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		if _, _, err := pq.Restore(ctx, "k"); !errors.Is(err, context.Canceled) {
			t.Fatalf("cancelled Restore = %v, want context.Canceled", err)
		}
		if got := pq.Stats().InFlight; got != 1 {
			t.Fatalf("InFlight after waiter cancel = %d, want 1 (permit kept until I/O returns)", got)
		}

		close(gate)
		waitFor(t, "read to complete", 2*time.Second, func() bool {
			return pq.Stats().InFlight == 0
		})
		if p, found, err := pq.Restore(context.Background(), "k"); err != nil || !found || !bytes.Equal(p, []byte("kept")) {
			t.Fatalf("post-cancel Restore = (%q, %v, %v), want staged hit", p, found, err)
		}
	})

	t.Run("stop cancels queued reads", func(t *testing.T) {
		store := newGateStore(map[string][]byte{"a": []byte("A"), "b": []byte("B")})
		gateA, gateB := make(chan struct{}), make(chan struct{})
		store.setGate("a", gateA)
		store.setGate("b", gateB)
		pq := NewPrefetchQueue(store, PrefetchConfig{Timeout: 5 * time.Second, MaxConcurrentReads: 1})

		if err := pq.Enqueue("a"); err != nil {
			t.Fatalf("Enqueue(a) = %v", err)
		}
		waitFor(t, "read a admitted onto the single permit", 2*time.Second, func() bool {
			return pq.Stats().InFlight == 1
		})
		// b queues behind the held permit: it cannot start, so its
		// cancellation by Stop is deterministic.
		if err := pq.Enqueue("b"); err != nil {
			t.Fatalf("Enqueue(b) = %v", err)
		}

		// The in-flight read a keeps its permit until the I/O returns;
		// Stop returns only once that I/O has.
		go func() { time.Sleep(50 * time.Millisecond); close(gateA) }()
		pq.Stop()

		if err := pq.Enqueue("b"); !errors.Is(err, ErrPrefetchStopped) {
			t.Fatalf("post-Stop Enqueue = %v, want ErrPrefetchStopped", err)
		}
		if _, _, err := pq.Restore(context.Background(), "b"); !errors.Is(err, ErrPrefetchStopped) {
			t.Fatalf("post-Stop Restore = %v, want ErrPrefetchStopped", err)
		}
		if got := pq.Stats().InFlight; got != 0 {
			t.Fatalf("InFlight after Stop = %d, want 0", got)
		}
	})

	t.Run("memory cap drops oversized staged payloads", func(t *testing.T) {
		store := newGateStore(map[string][]byte{"k": []byte("big-payload")})
		pq := NewPrefetchQueue(store, PrefetchConfig{MaxConcurrentReads: 2, MaxStagedBytes: 4})
		t.Cleanup(pq.Stop)

		if err := pq.Enqueue("k"); err != nil {
			t.Fatalf("Enqueue = %v", err)
		}
		waitFor(t, "cap drop to settle", 2*time.Second, func() bool {
			return pq.Stats().DroppedCap == 1
		})
		// Dropped by the cap: a clean miss, so the caller falls back to
		// admission restore or fresh prefill.
		if p, found, err := pq.Restore(context.Background(), "k"); err != nil || found || p != nil {
			t.Fatalf("capped Restore = (%q, %v, %v), want clean miss", p, found, err)
		}
		if st := pq.Stats(); st.StagedBytes != 0 {
			t.Fatalf("StagedBytes after cap drop = %d, want 0", st.StagedBytes)
		}
	})
}

// fak-test:runtime fast est=100ms lane=default
func TestPrefetchStopCancelsStoreRead(t *testing.T) {
	t.Parallel()

	t.Run("stop cancels a blocked context-aware read without external release", func(t *testing.T) {
		store := newCtxAwareStore(map[string][]byte{"span": []byte("payload")})
		pq := NewPrefetchQueue(store, PrefetchConfig{Timeout: 5 * time.Second, MaxConcurrentReads: 2})

		if err := pq.Enqueue("span"); err != nil {
			t.Fatalf("Enqueue = %v, want nil", err)
		}
		waitFor(t, "read admitted onto an I/O worker", 2*time.Second, func() bool {
			return pq.Stats().InFlight == 1
		})

		// Stop must join the worker with nobody releasing the store: the
		// read's context carries the queue's shutdown.
		stopped := make(chan struct{})
		go func() {
			pq.Stop()
			close(stopped)
		}()
		select {
		case <-stopped:
		case <-time.After(5 * time.Second):
			t.Fatal("Stop did not return: the blocked context-aware read was not cancelled")
		}

		if got := store.inFlight(); got != 0 {
			t.Fatalf("store in-flight after Stop = %d, want 0 (read retired)", got)
		}
		if got := pq.Stats().InFlight; got != 0 {
			t.Fatalf("InFlight after Stop = %d, want 0", got)
		}
		// The cancelled read settles as stopped, not as an I/O fault.
		if _, _, err := pq.Restore(context.Background(), "span"); !errors.Is(err, ErrPrefetchStopped) {
			t.Fatalf("post-Stop Restore = %v, want ErrPrefetchStopped", err)
		}
		// Stop is idempotent.
		pq.Stop()
	})

	t.Run("cancelled waiter leaves the shared read running until stop retires it", func(t *testing.T) {
		store := newCtxAwareStore(map[string][]byte{"k": []byte("kept")})
		pq := NewPrefetchQueue(store, PrefetchConfig{Timeout: 5 * time.Second, MaxConcurrentReads: 2})

		if err := pq.Enqueue("k"); err != nil {
			t.Fatalf("Enqueue = %v", err)
		}
		waitFor(t, "read admitted", 2*time.Second, func() bool {
			return pq.Stats().InFlight == 1
		})

		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		if _, _, err := pq.Restore(ctx, "k"); !errors.Is(err, context.Canceled) {
			t.Fatalf("cancelled Restore = %v, want context.Canceled", err)
		}
		// Waiter cancellation never reaches the shared read.
		if got := pq.Stats().InFlight; got != 1 {
			t.Fatalf("InFlight after waiter cancel = %d, want 1 (permit kept)", got)
		}

		// Only the queue's own shutdown retires the read.
		stopped := make(chan struct{})
		go func() {
			pq.Stop()
			close(stopped)
		}()
		select {
		case <-stopped:
		case <-time.After(5 * time.Second):
			t.Fatal("Stop did not return: the read was not cancelled by queue shutdown")
		}
		if got := store.inFlight(); got != 0 {
			t.Fatalf("store in-flight after Stop = %d, want 0", got)
		}
	})
}

// fak-test:runtime fast est=50ms lane=default
func TestAsyncStoreRestorePrefetchSeam(t *testing.T) {
	t.Parallel()

	store := newGateStore(map[string][]byte{"span-x": []byte("x-payload")})
	as := NewAsyncStore(store)
	t.Cleanup(as.StopPrefetch)

	if err := as.Prefetch("span-x"); err != nil {
		t.Fatalf("AsyncStore.Prefetch = %v, want nil", err)
	}
	// Ungated store: the read lands on its I/O worker while we wait, and the
	// staged payload comes back through the seam.
	if p, found, err := as.RestorePrefetched(context.Background(), "span-x"); err != nil || !found || !bytes.Equal(p, []byte("x-payload")) {
		t.Fatalf("AsyncStore.RestorePrefetched = (%q, %v, %v), want staged hit", p, found, err)
	}
	if err := as.Prefetch("span-x"); !errors.Is(err, ErrPrefetchDisabled) && !errors.Is(err, ErrPrefetchStopped) && err != nil {
		// Re-enqueueing an unconsumed-but-settled key is a dedup no-op (nil)
		// or a post-stop refusal; anything else is a contract break.
		t.Fatalf("second Prefetch = %v", err)
	}
}

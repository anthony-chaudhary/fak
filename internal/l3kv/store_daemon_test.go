package l3kv_test

import (
	"bytes"
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/l3kv"
	"github.com/anthony-chaudhary/fak/internal/l3server/metrics"
	"github.com/anthony-chaudhary/fak/internal/l3server/shard"
	"github.com/anthony-chaudhary/fak/internal/l3server/transport/tcp"
)

// startTestServer launches a test in-process L3 TCP server on loopback.
func startTestServer(t *testing.T) string {
	t.Helper()
	mgr, err := shard.NewManager(shard.ManagerConfig{
		NumShards:      2,
		MaxMemoryGB:    1,
		EvictionPolicy: "wtinylfu",
		ModelPageBytes: 5242880,
		MaxLeaseDurMs:  30000,
		IndexCapacity:  4096,
	})
	if err != nil {
		t.Fatalf("shard.NewManager: %v", err)
	}
	mgr.Start()
	t.Cleanup(func() {
		mgr.Stop()
	})

	connReg := metrics.NewConnRegistry()
	srv := tcp.NewServer("127.0.0.1:0", mgr, connReg, time.Now())
	if err := srv.Start(); err != nil {
		t.Fatalf("srv.Start: %v", err)
	}
	t.Cleanup(func() {
		srv.Stop()
	})

	return srv.Addr()
}

func TestDaemonStoreLoopbackIntegration(t *testing.T) {
	addr := startTestServer(t)
	ctx := context.Background()

	ds, err := l3kv.NewDaemonStore(addr)
	if err != nil {
		t.Fatalf("NewDaemonStore(%s): %v", addr, err)
	}
	defer ds.Close()

	if got := ds.Addr(); got != addr {
		t.Errorf("ds.Addr() = %q, want %q", got, addr)
	}

	key := "test-span-digest-01"
	payload := []byte("hello-from-l3store-daemon")

	// Initial Get should be a clean miss.
	got, found, err := ds.Get(ctx, key)
	if err != nil {
		t.Fatalf("Get miss err: %v", err)
	}
	if found || got != nil {
		t.Fatalf("Get miss: got (%q, %v), want (nil, false)", string(got), found)
	}

	// Exists should report false.
	exists, err := ds.Exists(ctx, key)
	if err != nil {
		t.Fatalf("Exists miss err: %v", err)
	}
	if exists {
		t.Errorf("Exists miss: got true, want false")
	}

	// Put the payload.
	if err := ds.Put(ctx, key, payload); err != nil {
		t.Fatalf("Put err: %v", err)
	}

	// Get should hit with bit-exact bytes.
	got, found, err = ds.Get(ctx, key)
	if err != nil {
		t.Fatalf("Get hit err: %v", err)
	}
	if !found {
		t.Fatalf("Get hit: found=false, want true")
	}
	if !bytes.Equal(got, payload) {
		t.Fatalf("Get hit: got %q, want %q", string(got), string(payload))
	}

	// Exists should report true.
	exists, err = ds.Exists(ctx, key)
	if err != nil {
		t.Fatalf("Exists hit err: %v", err)
	}
	if !exists {
		t.Errorf("Exists hit: got false, want true")
	}

	// Overwrite with updated payload.
	updatedPayload := []byte("updated-l3store-payload-bytes")
	if err := ds.Put(ctx, key, updatedPayload); err != nil {
		t.Fatalf("Put overwrite err: %v", err)
	}

	got, found, err = ds.Get(ctx, key)
	if err != nil {
		t.Fatalf("Get overwrite err: %v", err)
	}
	if !found || !bytes.Equal(got, updatedPayload) {
		t.Fatalf("Get overwrite: got %q, want %q", string(got), string(updatedPayload))
	}

	// Delete key.
	if err := ds.Delete(ctx, key); err != nil {
		t.Fatalf("Delete err: %v", err)
	}

	// Get after delete should be a clean miss.
	got, found, err = ds.Get(ctx, key)
	if err != nil {
		t.Fatalf("Get after delete err: %v", err)
	}
	if found || got != nil {
		t.Fatalf("Get after delete: got (%q, %v), want (nil, false)", string(got), found)
	}

	// Exists after delete should be false.
	exists, err = ds.Exists(ctx, key)
	if err != nil {
		t.Fatalf("Exists after delete err: %v", err)
	}
	if exists {
		t.Errorf("Exists after delete: got true, want false")
	}
}

func TestDaemonStoreLargePayload(t *testing.T) {
	addr := startTestServer(t)
	ctx := context.Background()

	ds, err := l3kv.NewDaemonStore(addr)
	if err != nil {
		t.Fatalf("NewDaemonStore: %v", err)
	}
	defer ds.Close()

	// 512KB KV page
	largePayload := make([]byte, 512*1024)
	for i := range largePayload {
		largePayload[i] = byte((i * 13) % 256)
	}

	key := "large-span-layer-17"
	if err := ds.Put(ctx, key, largePayload); err != nil {
		t.Fatalf("Put large: %v", err)
	}

	got, found, err := ds.Get(ctx, key)
	if err != nil {
		t.Fatalf("Get large: %v", err)
	}
	if !found {
		t.Fatalf("Get large: not found")
	}
	if !bytes.Equal(got, largePayload) {
		t.Fatalf("Get large: payload mismatch (got len %d, want %d)", len(got), len(largePayload))
	}
}

func TestDaemonStoreConcurrentAccess(t *testing.T) {
	addr := startTestServer(t)
	ctx := context.Background()

	ds, err := l3kv.NewDaemonStore(addr)
	if err != nil {
		t.Fatalf("NewDaemonStore: %v", err)
	}
	defer ds.Close()

	const numGoroutines = 10
	const opsPerGoroutine = 20

	var wg sync.WaitGroup
	wg.Add(numGoroutines)

	for g := 0; g < numGoroutines; g++ {
		go func(gid int) {
			defer wg.Done()
			for op := 0; op < opsPerGoroutine; op++ {
				k := fmt.Sprintf("conc-key-%d-%d", gid, op)
				v := []byte(fmt.Sprintf("conc-val-%d-%d", gid, op))

				if err := ds.Put(ctx, k, v); err != nil {
					t.Errorf("goroutine %d put: %v", gid, err)
					return
				}
				got, found, err := ds.Get(ctx, k)
				if err != nil {
					t.Errorf("goroutine %d get: %v", gid, err)
					return
				}
				if !found || !bytes.Equal(got, v) {
					t.Errorf("goroutine %d get mismatch: got %s, want %s", gid, string(got), string(v))
					return
				}
				if err := ds.Delete(ctx, k); err != nil {
					t.Errorf("goroutine %d delete: %v", gid, err)
					return
				}
			}
		}(g)
	}

	wg.Wait()
}

func TestDaemonStoreValidationAndContext(t *testing.T) {
	addr := startTestServer(t)
	ds, err := l3kv.NewDaemonStore(addr)
	if err != nil {
		t.Fatalf("NewDaemonStore: %v", err)
	}
	defer ds.Close()

	ctx := context.Background()

	// Empty key checks
	if err := ds.Put(ctx, "", []byte("v")); err == nil {
		t.Error("Put empty key should fail")
	}
	if _, _, err := ds.Get(ctx, ""); err == nil {
		t.Error("Get empty key should fail")
	}
	if err := ds.Delete(ctx, ""); err == nil {
		t.Error("Delete empty key should fail")
	}
	if _, err := ds.Exists(ctx, ""); err == nil {
		t.Error("Exists empty key should fail")
	}

	// Cancelled context checks
	cancCtx, cancel := context.WithCancel(ctx)
	cancel()

	if err := ds.Put(cancCtx, "k", []byte("v")); err == nil {
		t.Error("Put with cancelled context should fail")
	}
	if _, _, err := ds.Get(cancCtx, "k"); err == nil {
		t.Error("Get with cancelled context should fail")
	}
	if err := ds.Delete(cancCtx, "k"); err == nil {
		t.Error("Delete with cancelled context should fail")
	}
	if _, err := ds.Exists(cancCtx, "k"); err == nil {
		t.Error("Exists with cancelled context should fail")
	}

	// Idempotent Close
	if err := ds.Close(); err != nil {
		t.Errorf("Close: %v", err)
	}
	if err := ds.Close(); err != nil {
		t.Errorf("Second Close should be nil: %v", err)
	}
}

type testStagerRestorer struct {
	data      []byte
	positions int
}

func (s *testStagerRestorer) Len() int                    { return 0 }
func (s *testStagerRestorer) Prefill(ids []int) []float32 { return nil }
func (s *testStagerRestorer) Evict(from, n int) int       { return n }
func (s *testStagerRestorer) ModelID() string             { return "test-model" }
func (s *testStagerRestorer) StageSpan(context.Context, string, int, int) (abi.KVResidency, error) {
	return abi.KVResidency{Outcome: abi.KVResidencyOK}, nil
}
func (s *testStagerRestorer) RestoreSpan(context.Context, string) (abi.KVResidency, error) {
	return abi.KVResidency{Outcome: abi.KVResidencyMiss}, nil
}
func (s *testStagerRestorer) StageSpanBytes(from, n int) ([]byte, error) {
	return s.data, nil
}
func (s *testStagerRestorer) RestoreSpanBytes(payload []byte) (int, error) {
	s.data = append([]byte(nil), payload...)
	return s.positions, nil
}

func TestDaemonStoreL3KVBackendIntegration(t *testing.T) {
	addr := startTestServer(t)
	ctx := context.Background()

	ds, err := l3kv.NewDaemonStore(addr)
	if err != nil {
		t.Fatalf("NewDaemonStore: %v", err)
	}
	defer ds.Close()

	payload := []byte("staged-span-kv-bytes-12345")
	inner := &testStagerRestorer{data: payload, positions: 16}

	// Wrap with l3kv.New
	b := l3kv.New(inner, ds)

	digest := "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

	// StageSpan should write to the daemon store and report OK.
	res, err := b.StageSpan(ctx, digest, 0, 16)
	if err != nil {
		t.Fatalf("StageSpan err: %v", err)
	}
	if res.Outcome != abi.KVResidencyOK {
		t.Fatalf("StageSpan outcome = %v, want OK (%s)", res.Outcome, res.Reason)
	}
	if res.BytesMoved != int64(len(payload)) {
		t.Errorf("BytesMoved = %d, want %d", res.BytesMoved, len(payload))
	}

	// Clear inner data to prove restore really recovers from the daemon.
	inner.data = nil

	// RestoreSpan should retrieve from daemon store and reinstall.
	res, err = b.RestoreSpan(ctx, digest)
	if err != nil {
		t.Fatalf("RestoreSpan err: %v", err)
	}
	if res.Outcome != abi.KVResidencyOK {
		t.Fatalf("RestoreSpan outcome = %v, want OK (%s)", res.Outcome, res.Reason)
	}
	if res.Positions != 16 {
		t.Errorf("RestoreSpan positions = %d, want 16", res.Positions)
	}
	if !bytes.Equal(inner.data, payload) {
		t.Fatalf("Restored bytes mismatch: got %q, want %q", string(inner.data), string(payload))
	}
}

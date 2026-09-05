package tcp_test

import (
	"bytes"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/l3server/client"
	"github.com/anthony-chaudhary/fak/internal/l3server/metrics"
	"github.com/anthony-chaudhary/fak/internal/l3server/shard"
	"github.com/anthony-chaudhary/fak/internal/l3server/transport/tcp"
)

func TestServerIntegration(t *testing.T) {
	// Create a shard manager with small config for testing
	mgr, err := shard.NewManager(shard.ManagerConfig{
		NumShards:      2,
		MaxMemoryGB:    1, // 1 GB for test
		EvictionPolicy: "wtinylfu",
		ModelPageBytes: 5242880,
		MaxLeaseDurMs:  30000,
		IndexCapacity:  4096,
	})
	if err != nil {
		t.Fatalf("NewManager: %v", err)
	}
	mgr.Start()
	defer mgr.Stop()

	// Start TCP server on random port
	connReg := metrics.NewConnRegistry()
	srv := tcp.NewServer("127.0.0.1:0", mgr, connReg, time.Now())
	if err := srv.Start(); err != nil {
		t.Fatalf("Start: %v", err)
	}
	defer srv.Stop()

	addr := srv.Addr()
	t.Logf("Server listening on %s", addr)

	// Connect client
	c, err := client.New(addr)
	if err != nil {
		t.Fatalf("client.New: %v", err)
	}
	defer c.Close()

	// SET
	key := []byte("test-key")
	value := []byte("test-value-12345")
	if err := c.Set(key, value, 0); err != nil {
		t.Fatalf("Set: %v", err)
	}

	// GET
	got, err := c.Get(key)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if !bytes.Equal(got, value) {
		t.Errorf("Get: got %q, want %q", got, value)
	}

	// EXISTS
	exists, err := c.Exists(key)
	if err != nil {
		t.Fatalf("Exists: %v", err)
	}
	if !exists {
		t.Error("Exists: expected true")
	}

	// DELETE
	if err := c.Delete(key); err != nil {
		t.Fatalf("Delete: %v", err)
	}

	// GET after DELETE
	got, err = c.Get(key)
	if err != nil {
		t.Fatalf("Get after Delete: %v", err)
	}
	if got != nil {
		t.Errorf("Get after Delete: got %q, want nil", got)
	}

	// MSET/MGET
	keys := [][]byte{[]byte("mk1"), []byte("mk2"), []byte("mk3")}
	vals := [][]byte{[]byte("mv1"), []byte("mv2"), []byte("mv3")}
	if _, err := c.MSet(keys, vals); err != nil {
		t.Fatalf("MSet: %v", err)
	}

	gotVals, founds, err := c.MGet(keys)
	if err != nil {
		t.Fatalf("MGet: %v", err)
	}
	for i := range keys {
		if !founds[i] {
			t.Errorf("MGet: key %q not found", keys[i])
			continue
		}
		if !bytes.Equal(gotVals[i], vals[i]) {
			t.Errorf("MGet: key %q got %q, want %q", keys[i], gotVals[i], vals[i])
		}
	}

	// LEASE
	if err := c.Lease([]byte("mk1"), 5000); err != nil {
		t.Fatalf("Lease: %v", err)
	}

	// PIN/UNPIN
	if err := c.Pin([]byte("mk2")); err != nil {
		t.Fatalf("Pin: %v", err)
	}
	if err := c.Unpin([]byte("mk2")); err != nil {
		t.Fatalf("Unpin: %v", err)
	}

	// Large value (640KB â€” Llama 70B per-layer)
	largeKey := []byte("large-kv-page")
	largeVal := make([]byte, 655360)
	for i := range largeVal {
		largeVal[i] = byte(i % 256)
	}
	if err := c.Set(largeKey, largeVal, 0); err != nil {
		t.Fatalf("Set large: %v", err)
	}
	gotLarge, err := c.Get(largeKey)
	if err != nil {
		t.Fatalf("Get large: %v", err)
	}
	if !bytes.Equal(gotLarge, largeVal) {
		t.Errorf("Get large: value mismatch (len got=%d, want=%d)", len(gotLarge), len(largeVal))
	}

	t.Log("All integration tests passed")
}

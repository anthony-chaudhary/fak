package cache

import (
	"context"
	"path/filepath"
	"sync"
	"testing"
	"time"
)

func TestMemoryBackend(t *testing.T) {
	b := NewMemoryBackend()
	defer b.Close()

	ctx := context.Background()

	// 1. Get nonexistent
	_, ok, err := b.Get(ctx, "k1")
	if err != nil || ok {
		t.Fatalf("expected not found, got ok=%v, err=%v", ok, err)
	}

	// 2. Set and Get
	if err := b.Set(ctx, "k1", []byte("hello"), 0); err != nil {
		t.Fatalf("Set failed: %v", err)
	}
	val, ok, err := b.Get(ctx, "k1")
	if err != nil || !ok || string(val) != "hello" {
		t.Fatalf("Get failed: ok=%v, val=%s, err=%v", ok, string(val), err)
	}

	// 3. TTL expiration
	if err := b.Set(ctx, "exp", []byte("bye"), 10*time.Millisecond); err != nil {
		t.Fatal(err)
	}
	time.Sleep(20 * time.Millisecond)
	_, ok, _ = b.Get(ctx, "exp")
	if ok {
		t.Fatalf("expected key to be expired")
	}

	// 4. Delete
	if err := b.Delete(ctx, "k1"); err != nil {
		t.Fatal(err)
	}
	_, ok, _ = b.Get(ctx, "k1")
	if ok {
		t.Fatalf("expected key to be deleted")
	}
}

func TestFileBackend(t *testing.T) {
	dir := t.TempDir()
	b, err := NewFileBackend(filepath.Join(dir, "cache"))
	if err != nil {
		t.Fatal(err)
	}

	ctx := context.Background()

	// Set and Get
	if err := b.Set(ctx, "session:123", []byte("user_data"), 0); err != nil {
		t.Fatal(err)
	}
	val, ok, err := b.Get(ctx, "session:123")
	if err != nil || !ok || string(val) != "user_data" {
		t.Fatalf("file Get failed: ok=%v, val=%s", ok, string(val))
	}

	b.Close()

	// Reopen same directory and assert persistence
	b2, err := NewFileBackend(filepath.Join(dir, "cache"))
	if err != nil {
		t.Fatal(err)
	}
	defer b2.Close()

	val2, ok2, err := b2.Get(ctx, "session:123")
	if err != nil || !ok2 || string(val2) != "user_data" {
		t.Fatalf("persistence failed after reopen: ok=%v, val=%s", ok2, string(val2))
	}
}

func TestRedisAndCloudflareBackends(t *testing.T) {
	ctx := context.Background()

	// Redis backend
	rb := NewRedisBackend("localhost:6379", "db0")
	defer rb.Close()

	if err := rb.Set(ctx, "k1", []byte("redis_val"), 0); err != nil {
		t.Fatal(err)
	}
	val, ok, _ := rb.Get(ctx, "k1")
	if !ok || string(val) != "redis_val" {
		t.Errorf("redis backend failed")
	}

	// Cloudflare KV backend
	cf := NewCloudflareKVBackend("kv_ns")
	defer cf.Close()

	if err := cf.Set(ctx, "k2", []byte("cf_val"), 0); err != nil {
		t.Fatal(err)
	}
	val2, ok2, _ := cf.Get(ctx, "k2")
	if !ok2 || string(val2) != "cf_val" {
		t.Errorf("cf kv backend failed")
	}
}

func TestNamedRegistry(t *testing.T) {
	dir := t.TempDir()
	reg, err := NewDefaultRegistry(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer reg.Close()

	expectedServices := map[string]string{
		"token":   "memory",
		"session": "file",
		"config":  "memory",
		"oauth":   "file",
		"mcp":     "file",
	}

	for name, wantType := range expectedServices {
		svc, ok := reg.Get(name)
		if !ok {
			t.Fatalf("expected service %q registered", name)
		}
		if svc.Backend.Type() != wantType {
			t.Errorf("service %q has backend type %q, want %q", name, svc.Backend.Type(), wantType)
		}
	}

	// Test unified operation
	tokenSvc, _ := reg.Get("token")
	if err := tokenSvc.Set(context.Background(), "tok1", []byte("secret"), 0); err != nil {
		t.Fatal(err)
	}
	val, ok, _ := tokenSvc.Get(context.Background(), "tok1")
	if !ok || string(val) != "secret" {
		t.Fatalf("token service failed")
	}
}

func TestServiceObserver(t *testing.T) {
	var mu sync.Mutex
	var events []string

	obs := func(tier Tier, op Op, outcome Outcome, size int64, dur time.Duration) {
		mu.Lock()
		defer mu.Unlock()
		opStr := "read"
		if op == OpWrite {
			opStr = "write"
		}
		outStr := "hit"
		if outcome == OutcomeMiss {
			outStr = "miss"
		} else if outcome == OutcomeError {
			outStr = "error"
		}
		events = append(events, opStr+":"+outStr)
	}

	svc := &Service{
		Name:       "test",
		Backend:    NewMemoryBackend(),
		DefaultTTL: 1 * time.Minute,
		Observer:   obs,
	}
	defer svc.Backend.Close()

	ctx := context.Background()

	// 1. Miss
	_, _, _ = svc.Get(ctx, "k_none")

	// 2. Write
	_ = svc.Set(ctx, "k_hit", []byte("data"), 0)

	// 3. Hit
	_, _, _ = svc.Get(ctx, "k_hit")

	mu.Lock()
	defer mu.Unlock()
	if len(events) != 3 {
		t.Fatalf("expected 3 observed events, got %d: %v", len(events), events)
	}
	if events[0] != "read:miss" {
		t.Errorf("event 0 = %s, want read:miss", events[0])
	}
	if events[1] != "write:hit" {
		t.Errorf("event 1 = %s, want write:hit", events[1])
	}
	if events[2] != "read:hit" {
		t.Errorf("event 2 = %s, want read:hit", events[2])
	}
}

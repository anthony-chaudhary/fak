package cache

import (
	"context"
	"testing"
)

func TestCacheBenchmarkSanity(t *testing.T) {
	b := NewMemoryBackend()
	defer b.Close()

	ctx := context.Background()
	if err := b.Set(ctx, "sanity_key", []byte("sanity_val"), 0); err != nil {
		t.Fatalf("Set failed: %v", err)
	}
	val, ok, err := b.Get(ctx, "sanity_key")
	if err != nil || !ok || string(val) != "sanity_val" {
		t.Fatalf("unexpected get: val=%s, ok=%v, err=%v", string(val), ok, err)
	}
}

func BenchmarkCache(b *testing.B) {
	backend := NewMemoryBackend()
	defer backend.Close()

	ctx := context.Background()
	key := "benchmark_key"
	val := []byte("benchmark_payload_value")

	if err := backend.Set(ctx, key, val, 0); err != nil {
		b.Fatalf("initial set failed: %v", err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res, ok, err := backend.Get(ctx, key)
		if err != nil || !ok || len(res) == 0 {
			b.Fatalf("get failed at iteration %d: ok=%v, err=%v", i, ok, err)
		}
	}
}

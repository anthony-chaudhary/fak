package shard

import (
	"bytes"
	"fmt"
	"math/rand"
	"testing"
)

// BenchmarkShardSet measures full shard write path including dispatch, index,
// allocation, and eviction.
func BenchmarkShardSet(b *testing.B) {
	cfg := ShardConfig{
		ID:             0,
		IndexCapacity:  4096,
		MaxMemoryBytes: 64 * 1024 * 1024,
		EvictionPolicy: "wtinylfu",
	}
	s, err := New(cfg)
	if err != nil {
		b.Fatal(err)
	}
	s.Start()
	defer func() {
		s.Stop()
		<-s.Done()
	}()

	val := make([]byte, 4096)
	for i := range val {
		val[i] = byte(i % 256)
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		key := []byte(fmt.Sprintf("bench-%08d", i))
		s.Submit(ShardOp{
			Type:    OpSet,
			Key:     key,
			KeyHash: uint64(i + 1),
			Value:   val,
			Result:  make(chan OpResult, 1),
		})
	}
}

// BenchmarkShardGetHit measures read-path latency for existing keys.
func BenchmarkShardGetHit(b *testing.B) {
	cfg := ShardConfig{
		ID:             0,
		IndexCapacity:  4096,
		MaxMemoryBytes: 64 * 1024 * 1024,
		EvictionPolicy: "wtinylfu",
	}
	s, err := New(cfg)
	if err != nil {
		b.Fatal(err)
	}
	s.Start()
	defer func() {
		s.Stop()
		<-s.Done()
	}()

	// Pre-populate 1000 keys
	const numKeys = 1000
	keys := make([][]byte, numKeys)
	val := bytes.Repeat([]byte{0xAB}, 4096)
	for i := 0; i < numKeys; i++ {
		keys[i] = []byte(fmt.Sprintf("read-%04d", i))
		s.Submit(ShardOp{
			Type:    OpSet,
			Key:     keys[i],
			KeyHash: uint64(i + 1),
			Value:   val,
			Result:  make(chan OpResult, 1),
		})
	}

	rng := rand.New(rand.NewSource(42))
	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		idx := rng.Intn(numKeys)
		s.Submit(ShardOp{
			Type:    OpGet,
			Key:     keys[idx],
			KeyHash: uint64(idx + 1),
			Result:  make(chan OpResult, 1),
		})
	}
}

// BenchmarkShardMixedWorkload measures 80% GET / 20% SET throughput.
func BenchmarkShardMixedWorkload(b *testing.B) {
	cfg := ShardConfig{
		ID:             0,
		IndexCapacity:  4096,
		MaxMemoryBytes: 64 * 1024 * 1024,
		EvictionPolicy: "wtinylfu",
	}
	s, err := New(cfg)
	if err != nil {
		b.Fatal(err)
	}
	s.Start()
	defer func() {
		s.Stop()
		<-s.Done()
	}()

	const numKeys = 1000
	keys := make([][]byte, numKeys)
	val := bytes.Repeat([]byte{0xCD}, 4096)
	for i := 0; i < numKeys; i++ {
		keys[i] = []byte(fmt.Sprintf("mix-%04d", i))
		s.Submit(ShardOp{
			Type:    OpSet,
			Key:     keys[i],
			KeyHash: uint64(i + 1),
			Value:   val,
			Result:  make(chan OpResult, 1),
		})
	}

	rng := rand.New(rand.NewSource(99))
	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		idx := rng.Intn(numKeys)
		if rng.Intn(100) < 80 {
			// GET (80%)
			s.Submit(ShardOp{
				Type:    OpGet,
				Key:     keys[idx],
				KeyHash: uint64(idx + 1),
				Result:  make(chan OpResult, 1),
			})
		} else {
			// SET (20%)
			s.Submit(ShardOp{
				Type:    OpSet,
				Key:     keys[idx],
				KeyHash: uint64(idx + 1),
				Value:   val,
				Result:  make(chan OpResult, 1),
			})
		}
	}
}

// BenchmarkShardEvictionChurn measures throughput under constant eviction pressure.
func BenchmarkShardEvictionChurn(b *testing.B) {
	cfg := ShardConfig{
		ID:             0,
		IndexCapacity:  512,
		MaxMemoryBytes: 16 * 1024 * 1024, // 16MB â€” small to force evictions with 4KB values
		EvictionPolicy: "wtinylfu",
	}
	s, err := New(cfg)
	if err != nil {
		b.Fatal(err)
	}
	s.Start()
	defer func() {
		s.Stop()
		<-s.Done()
	}()

	val := bytes.Repeat([]byte{0xEF}, 4096)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		key := []byte(fmt.Sprintf("churn-%08d", i))
		s.Submit(ShardOp{
			Type:    OpSet,
			Key:     key,
			KeyHash: uint64(i + 1),
			Value:   val,
			Result:  make(chan OpResult, 1),
		})
	}
}

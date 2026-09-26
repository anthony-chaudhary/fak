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

	// Pre-populate a resident set: with a 64MB arena and default class weights
	// the 4096-byte class holds ~227 slots, so all keys must stay well under
	// that to guarantee every GET is a hit (a miss would skip the value copy
	// and make the allocs/op comparison meaningless).
	const numKeys = 200
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

// BenchmarkShardGetHitAlloc measures the coordinate-only read path (OpGetAlloc):
// it resolves the same entry and physical allocation but never materializes the
// value []byte. It must report strictly fewer allocs/op than BenchmarkShardGetHit
// (the copy path) â€” the witness for issue #13519.
func BenchmarkShardGetHitAlloc(b *testing.B) {
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

	// Pre-populate the same resident 200-key / 4KB set as BenchmarkShardGetHit
	// so both paths resolve real hits (a miss would elide the value copy the
	// OP_GET path performs and defeat the allocs/op comparison).
	const numKeys = 200
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
		res := s.Submit(ShardOp{
			Type:    OpGetAlloc,
			Key:     keys[idx],
			KeyHash: uint64(idx + 1),
			Result:  make(chan OpResult, 1),
		})
		if !res.Found || res.AllocInfo == nil {
			b.Fatal("coordinate path miss")
		}
	}
}

// TestGetAllocCoordinateParity verifies the coordinate-only read path returns the
// identical AllocMeta (class/offset/size) as the value-copy path while leaving
// Value nil, and that a missing key resolves to a miss with no coordinates.
func TestGetAllocCoordinateParity(t *testing.T) {
	cfg := ShardConfig{
		ID:             0,
		IndexCapacity:  1024,
		MaxMemoryBytes: 64 * 1024 * 1024,
		EvictionPolicy: "wtinylfu",
	}
	s, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	s.Start()
	defer func() {
		s.Stop()
		<-s.Done()
	}()

	key := []byte("coord-key")
	keyHash := uint64(7)
	val := bytes.Repeat([]byte{0x5A}, 4096)
	if res := s.Submit(ShardOp{
		Type: OpSet, Key: key, KeyHash: keyHash, Value: val, Result: make(chan OpResult, 1),
	}); res.Err != nil {
		t.Fatalf("SET: %v", res.Err)
	}

	copyRes := s.Submit(ShardOp{
		Type: OpGet, Key: key, KeyHash: keyHash, Result: make(chan OpResult, 1),
	})
	coordRes := s.Submit(ShardOp{
		Type: OpGetAlloc, Key: key, KeyHash: keyHash, Result: make(chan OpResult, 1),
	})

	if !copyRes.Found || !coordRes.Found {
		t.Fatalf("both paths must hit (copy=%v coord=%v)", copyRes.Found, coordRes.Found)
	}
	if coordRes.Value != nil {
		t.Errorf("coordinate path must not materialize Value; got %d bytes", len(coordRes.Value))
	}
	if copyRes.AllocInfo == nil || coordRes.AllocInfo == nil {
		t.Fatalf("both paths must populate AllocInfo (copy=%v coord=%v)", copyRes.AllocInfo, coordRes.AllocInfo)
	}
	if *coordRes.AllocInfo != *copyRes.AllocInfo {
		t.Errorf("AllocInfo mismatch: coord=%+v copy=%+v", *coordRes.AllocInfo, *copyRes.AllocInfo)
	}
	if coordRes.AllocInfo.Size != uint64(len(val)) {
		t.Errorf("AllocInfo.Size=%d, want %d", coordRes.AllocInfo.Size, len(val))
	}

	// Miss must report not-found with no coordinates.
	miss := s.Submit(ShardOp{
		Type: OpGetAlloc, Key: []byte("absent"), KeyHash: uint64(9999), Result: make(chan OpResult, 1),
	})
	if miss.Found || miss.AllocInfo != nil {
		t.Errorf("miss must be not-found with nil AllocInfo; got found=%v alloc=%+v", miss.Found, miss.AllocInfo)
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

//go:build cuda && cgo

package compute

import (
	"errors"
	"testing"
)

// fak-test:runtime gpu est=1s lane=cuda
func TestCUDAFailedKVReservationDoesNotGrowPool(t *testing.T) {
	cb := cudaOrSkip(t)
	keep := cb.Upload(NewF32(Default(), []int{2}, []float32{3, 7}), F32)
	defer cb.Free(keep)
	kv := &cudaKV{be: cb, cfg: KVConfig{NumLayers: 1, NumKVHeads: 4, HeadDim: 256}, Kraw: make([]dslice, 1), K: make([]dslice, 1), V: make([]dslice, 1)}
	func() {
		cudaMu.Lock()
		defer cudaMu.Unlock()
		if err := cb.qwen35SequenceReserveKVLocked(kv, 1, 1024); err != nil {
			t.Fatal(err)
		}
	}()
	defer kv.Free()
	beforeKraw, beforeK, beforeV := kv.Kraw[0], kv.K[0], kv.V[0]
	beforePool := cb.CUDADebugPoolStats()
	beforeTransient, beforeLive := cb.cudaAllocationCountsForTest()
	var err error
	func() {
		cudaMu.Lock()
		defer cudaMu.Unlock()
		qwen35GDNStrictAllocFailAfter = 1 // One unpublished allocation, then refusal.
		defer func() { qwen35GDNStrictAllocFailAfter = -1 }()
		err = cb.qwen35SequenceReserveKVLocked(kv, 1, 4096)
	}()
	var allocation *Qwen35GDNAllocationError
	if !errors.As(err, &allocation) {
		t.Fatalf("lost primary allocation refusal: %v", err)
	}
	if kv.Kraw[0] != beforeKraw || kv.K[0] != beforeK || kv.V[0] != beforeV {
		t.Fatal("failed reservation published KV")
	}
	afterTransient, afterLive := cb.cudaAllocationCountsForTest()
	if afterLive != beforeLive || afterTransient != beforeTransient {
		t.Fatalf("owner retained: live %d -> %d transient %d -> %d", beforeLive, afterLive, beforeTransient, afterTransient)
	}
	afterPool := cb.CUDADebugPoolStats()
	if afterPool.PooledBytes > beforePool.PooledBytes {
		t.Fatalf("failed reservation grew pool: %d -> %d", beforePool.PooledBytes, afterPool.PooledBytes)
	}
	got := cb.Read(keep)
	if len(got) != 2 || got[0] != 3 || got[1] != 7 {
		t.Fatalf("unrelated live owner changed: %v", got)
	}
}

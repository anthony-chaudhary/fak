package model

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func TestPrefixSnapshotResidentBytesCoversHostCacheAndQwenState(t *testing.T) {
	cache := NewKVCache(Config{})
	cache.K = [][]float32{{1, 2}}
	cache.Kraw = [][]float32{{3, 4}}
	cache.V = [][]float32{{5, 6}}
	cache.pos = []int{0}
	snap := NewHostPrefixSnapshotForTest(cache)
	if got, want := snap.ResidentBytes(), int64(6*4+8); got != want {
		t.Fatalf("host resident bytes = %d, want %d", got, want)
	}

	be := compute.Default()
	conv := compute.NewF32(be, []int{2}, []float32{1, 2})
	recurrent := compute.NewF32(be, []int{3}, []float32{1, 2, 3})
	snap.Backend = be
	snap.qwen35 = &qwen35HALState{layers: []qwen35HALLayerState{{conv: conv, recurrent: recurrent}}}
	if got, want := snap.ResidentBytes(), int64(6*4+8+5*4); got != want {
		t.Fatalf("hybrid resident bytes = %d, want %d", got, want)
	}
	snap.Close()
}

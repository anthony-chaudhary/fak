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

// fak-test:runtime fast est=10ms lane=default
// The device-capability case uses CPU-owned storage to witness accounting only.
func TestPrefixSnapshotResidencyBytesIncludesV41HostState(t *testing.T) {
	t.Parallel()
	for _, deviceBacked := range []bool{false, true} {
		name := "host"
		if deviceBacked {
			name = "device-capability"
		}
		t.Run(name, func(t *testing.T) {
			snap, be, _ := hostV41GuardSnapshot(t)
			defer snap.Close()
			wantHost, wantDevice := int64(20), int64(0) // Three float32 KV values and one position.
			if deviceBacked {
				snap.Backend = &hostReadRecordingBackend{Backend: be}
				wantHost, wantDevice = 8, 12
			}
			assertBytes := func(phase string, v41Bytes int64) {
				t.Helper()
				host, device := snap.ResidencyBytes()
				if host != wantHost+v41Bytes || device != wantDevice {
					t.Fatalf("%s residency = (%d, %d), want (%d, %d)", phase, host, device, wantHost+v41Bytes, wantDevice)
				}
				if got, want := snap.ResidentBytes(), wantHost+wantDevice+v41Bytes; got != want || host+device != got {
					t.Fatalf("%s resident bytes = %d, want %d and residency sum %d", phase, got, want, host+device)
				}
			}
			assertBytes("without V4.1", 0)

			state, err := NewV41AttentionState(2, 2)
			if err != nil {
				t.Fatal(err)
			}
			key := v41AttentionPublicationKey{sourceLayer: 0, start: 0, end: 1}
			state.kvPublications[key] = []float32{1, 2}
			state.indexPublications[key] = []float32{3}
			snap.v41 = &v41ForwardSnapshot{
				history: []int{1, 2}, attn: state, hadState: true,
				layers: []*V41AttentionState{{
					window: [][]float32{{1, 2}}, partialKV: [][]float32{{3}},
					partialInputs: [][]float32{{4, 5}}, partialPositions: []int{1},
					candidates: []bool{true, false}, topk: [][]int32{{0, 1}},
				}},
			}
			// Registry ring 1024 + history 16 + publications 12 + temporal
			// floats 20 + position 8 + candidates 2 + selections 8.
			assertBytes("writable", 1090)
			snap.PrepareForCache()
			base := state.publicationPrefix
			if base == nil {
				t.Fatal("cache admission did not seal the publication view")
			}
			assertBytes("sealed", 1090)
			// A shadow replaces the base row's charge; a new key adds its own.
			state.kvPublications[key] = []float32{6, 7, 8}
			state.indexPublications[v41AttentionPublicationKey{sourceLayer: 0, start: 1, end: 2}] = []float32{9, 10}
			assertBytes("overlay", 1102)
			if len(base.kv[key]) != 2 || len(base.index) != 1 {
				t.Fatal("residency accounting mutated the sealed publication base")
			}
		})
	}
}

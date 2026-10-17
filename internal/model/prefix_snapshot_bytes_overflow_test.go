package model

import (
	"math"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestPrefixSnapshotV41NestedBytesSaturate(t *testing.T) {
	t.Parallel()
	// Inject cached publication receipts instead of allocating enormous rows.
	// Each state's receipt fits; their sum with history exceeds int64.
	state := func(bytes int64) *V41AttentionState {
		return &V41AttentionState{
			publicationPrefix: &v41PublicationPrefix{kvBytes: bytes},
			kvPublications:    make(map[v41AttentionPublicationKey][]float32),
		}
	}
	snap := &PrefixSnapshot{
		Cache: &KVCache{pos: []int{0}},
		v41: &v41ForwardSnapshot{
			history: []int{1},
			attn:    state(math.MaxInt64 - 8),
			layers:  []*V41AttentionState{state(8)},
		},
	}
	if got := snap.v41ResidentBytes(); got != math.MaxInt64 {
		t.Fatalf("nested V4.1 bytes = %d, want saturated %d", got, int64(math.MaxInt64))
	}
	// Adding the host position must not wrap the saturated continuation charge.
	if got := snap.ResidentBytes(); got != math.MaxInt64 {
		t.Fatalf("complete snapshot bytes = %d, want saturated %d", got, int64(math.MaxInt64))
	}
}

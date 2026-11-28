package modelengine

import (
	"errors"
	"slices"
	"testing"
)

// fak-test:runtime fast est=5ms lane=default
func TestStepMetaPrecomputeImmutableAndShared(t *testing.T) {
	inKV := []int{128, 256}
	inPages := []int32{1, 2, 3, 4}

	snap, err := PrecomputeStepMeta(42, StepRankSpec{
		Rank:           0,
		WorldSize:      2,
		NumDraftTokens: 3,
		KVLengths:      inKV,
		PageIndices:    inPages,
	})
	if err != nil {
		t.Fatal(err)
	}
	// Mutating caller inputs after precompute must not affect the snapshot.
	inKV[0] = 999
	inPages[0] = 999

	// Two consumers of the same step see identical precomputed metadata.
	c1KV := snap.KVLengths()
	c2KV := snap.KVLengths()
	if !slices.Equal(c1KV, []int{128, 256}) || !slices.Equal(c1KV, c2KV) {
		t.Fatalf("consumers saw mismatched or mutated KVLengths: %v vs %v", c1KV, c2KV)
	}
	if !slices.Equal(snap.PageIndices(), []int32{1, 2, 3, 4}) {
		t.Fatalf("PageIndices mutated: %v", snap.PageIndices())
	}
	if snap.TotalKVTokens() != 384 || snap.MaxKVLen() != 256 || !snap.MTPActive() || !snap.Sealed() {
		t.Fatalf("unexpected precomputed totals: total=%d max=%d mtp=%v sealed=%v",
			snap.TotalKVTokens(), snap.MaxKVLen(), snap.MTPActive(), snap.Sealed())
	}

	// Mutating a returned defensive slice must not affect subsequent reads.
	c1KV[0] = -1
	if !slices.Equal(snap.KVLengths(), []int{128, 256}) {
		t.Fatal("snapshot leaked internal slice mutability")
	}

	// Mid-step mutation attempt is refused with ErrStepMetaImmutable.
	if err := snap.TryMutateKVLengths([]int{1, 2}); !errors.Is(err, ErrStepMetaImmutable) {
		t.Fatalf("expected ErrStepMetaImmutable, got %v", err)
	}
}

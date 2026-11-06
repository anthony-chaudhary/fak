package compute

// CPU-oracle witnesses for the MoE align-and-sort / weighted-sum finalize contract
// (moe_align_sort.go, device twin moe_align_sort.cu). The expected values below are
// hand-computed from the vLLM moe_align_block_size semantics pinned by ticket
// oss-port-vllmtop-moe-align-sort-20261103: every cumsum entry is a block multiple,
// pad slots hold the numel sentinel, and num_tokens_post_pad is the sum of the
// per-expert padded counts.

import (
	"errors"
	"reflect"
	"testing"
)

func TestMoeAlignBlockSizeMatchesOracle(t *testing.T) {
	t.Run("all tokens one expert pads to block boundary", func(t *testing.T) {
		// 5 tokens x topk 2, every id routes to expert 3 of 8, block size 4:
		// count 10 pads to 12, so two pad slots sit at the end of expert 3's region.
		ids := make([]int32, 10)
		for i := range ids {
			ids[i] = 3
		}
		got, err := MoeAlignBlockSizes(ids, 8, 4)
		if err != nil {
			t.Fatalf("MoeAlignBlockSizes: %v", err)
		}
		wantSorted := append([]int32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, make([]int32, 24)...)
		for p := 10; p < 34; p++ {
			wantSorted[p] = 10 // numel sentinel
		}
		want := &MoeAlignResult{
			SortedTokenIDs:     wantSorted,
			ExpertIDs:          []int32{3, 3, 3, -1, -1, -1, -1, -1, -1},
			Cumsum:             []int32{0, 0, 0, 0, 12, 12, 12, 12, 12},
			NumTokensPostPad:   12,
			Numel:              10,
			MaxNumTokensPadded: 34, // 10 + 8*(4-1)
			MaxNumMBlocks:      9,  // ceil(34/4)
		}
		assertMoeAlign(t, got, want)
	})

	t.Run("uniform spread interleaves empty experts", func(t *testing.T) {
		// 4 tokens x topk 2 over 8 experts, block size 2: experts 0,2,4,6 each take
		// exactly 2 entries; 1,3,5,7 stay empty between them.
		ids := []int32{0, 4, 2, 6, 0, 4, 2, 6}
		got, err := MoeAlignBlockSizes(ids, 8, 2)
		if err != nil {
			t.Fatalf("MoeAlignBlockSizes: %v", err)
		}
		want := &MoeAlignResult{
			SortedTokenIDs:     []int32{0, 4, 2, 6, 1, 5, 3, 7, 8, 8, 8, 8, 8, 8, 8, 8},
			ExpertIDs:          []int32{0, 2, 4, 6, -1, -1, -1, -1},
			Cumsum:             []int32{0, 2, 2, 4, 4, 6, 6, 8, 8},
			NumTokensPostPad:   8,
			Numel:              8,
			MaxNumTokensPadded: 16, // 8 + 8*(2-1)
			MaxNumMBlocks:      8,
		}
		assertMoeAlign(t, got, want)
	})

	t.Run("two experts land exactly on block boundaries", func(t *testing.T) {
		// 4 tokens x topk 2: expert 0 takes 4, expert 1 takes 4, block size 4. Each
		// region fills its blocks with zero pad slots, so num_tokens_post_pad equals
		// the raw entry count and both regions end exactly on a block boundary.
		ids := []int32{0, 0, 0, 0, 1, 1, 1, 1}
		got, err := MoeAlignBlockSizes(ids, 4, 4)
		if err != nil {
			t.Fatalf("MoeAlignBlockSizes: %v", err)
		}
		wantSorted := append([]int32{0, 1, 2, 3, 4, 5, 6, 7}, make([]int32, 12)...)
		for p := 8; p < 20; p++ {
			wantSorted[p] = 8 // numel sentinel
		}
		want := &MoeAlignResult{
			SortedTokenIDs:     wantSorted,
			ExpertIDs:          []int32{0, 1, -1, -1, -1},
			Cumsum:             []int32{0, 4, 8, 8, 8},
			NumTokensPostPad:   8,
			Numel:              8,
			MaxNumTokensPadded: 20, // 8 + 4*(4-1)
			MaxNumMBlocks:      5,
		}
		assertMoeAlign(t, got, want)
		if got.NumTokensPostPad != int32(len(ids)) || got.NumTokensPostPad != got.Cumsum[len(got.Cumsum)-1] {
			t.Fatalf("num_tokens_post_pad=%d want %d == cumsum tail %d",
				got.NumTokensPostPad, len(ids), got.Cumsum[len(got.Cumsum)-1])
		}
	})

	t.Run("invalid ids are skipped like get_local_expert_id", func(t *testing.T) {
		// -1 padding and out-of-range 999 are dropped; only entries 1 and 3 route to expert 3.
		ids := []int32{-1, 3, 999, 3}
		got, err := MoeAlignBlockSizes(ids, 4, 2)
		if err != nil {
			t.Fatalf("MoeAlignBlockSizes: %v", err)
		}
		want := &MoeAlignResult{
			SortedTokenIDs:     []int32{1, 3, 4, 4, 4, 4, 4, 4},
			ExpertIDs:          []int32{3, -1, -1, -1},
			Cumsum:             []int32{0, 0, 0, 0, 2},
			NumTokensPostPad:   2,
			Numel:              4,
			MaxNumTokensPadded: 8,
			MaxNumMBlocks:      4,
		}
		assertMoeAlign(t, got, want)
	})

	t.Run("empty panel keeps the capacity bound", func(t *testing.T) {
		got, err := MoeAlignBlockSizes(nil, 4, 2)
		if err != nil {
			t.Fatalf("MoeAlignBlockSizes(empty): %v", err)
		}
		want := &MoeAlignResult{
			SortedTokenIDs:     []int32{0, 0, 0, 0},
			ExpertIDs:          []int32{-1, -1},
			Cumsum:             []int32{0, 0, 0, 0, 0},
			NumTokensPostPad:   0,
			Numel:              0,
			MaxNumTokensPadded: 4,
			MaxNumMBlocks:      2,
		}
		assertMoeAlign(t, got, want)
	})

	t.Run("geometry refusals are closed sentinels", func(t *testing.T) {
		ids := []int32{0, 1}
		for _, tc := range []struct {
			name string
			call func() (*MoeAlignResult, error)
		}{
			{"zero experts", func() (*MoeAlignResult, error) { return MoeAlignBlockSizes(ids, 0, 4) }},
			{"zero block", func() (*MoeAlignResult, error) { return MoeAlignBlockSizes(ids, 8, 0) }},
			{"experts beyond one block", func() (*MoeAlignResult, error) { return MoeAlignBlockSizes(ids, MoeAlignMaxExperts+1, 4) }},
		} {
			_, err := tc.call()
			if !errors.Is(err, ErrMoeAlignGeometry) {
				t.Errorf("%s: errors.Is(err, ErrMoeAlignGeometry) = false, err=%v", tc.name, err)
			}
		}
	})
}

func assertMoeAlign(t *testing.T, got, want *MoeAlignResult) {
	t.Helper()
	if got.NumTokensPostPad != want.NumTokensPostPad || got.Numel != want.Numel ||
		got.MaxNumTokensPadded != want.MaxNumTokensPadded || got.MaxNumMBlocks != want.MaxNumMBlocks {
		t.Fatalf("scalars = {postPad:%d numel:%d maxPadded:%d maxBlocks:%d} want {postPad:%d numel:%d maxPadded:%d maxBlocks:%d}",
			got.NumTokensPostPad, got.Numel, got.MaxNumTokensPadded, got.MaxNumMBlocks,
			want.NumTokensPostPad, want.Numel, want.MaxNumTokensPadded, want.MaxNumMBlocks)
	}
	if !reflect.DeepEqual(got.Cumsum, want.Cumsum) {
		t.Errorf("cumsum = %v want %v", got.Cumsum, want.Cumsum)
	}
	if !reflect.DeepEqual(got.ExpertIDs, want.ExpertIDs) {
		t.Errorf("expertIDs = %v want %v", got.ExpertIDs, want.ExpertIDs)
	}
	if !reflect.DeepEqual(got.SortedTokenIDs, want.SortedTokenIDs) {
		t.Errorf("sortedTokenIDs = %v want %v", got.SortedTokenIDs, want.SortedTokenIDs)
	}
}

func TestMoeSumTopKExactValues(t *testing.T) {
	// 2 tokens x topk 2 x hidden 3. Token 0 sums both experts (weights .5 and 2);
	// token 1's second entry is an unroutable expert 999 and must contribute nothing.
	expertOut := []float32{
		1, 2, 3, 4, 5, 6, // token 0: expert rows k0, k1
		7, 8, 9, 10, 11, 12, // token 1: expert rows k0, k1
	}
	weights := []float32{0.5, 2, 3, -1}
	topkIDs := []int32{0, 1, 2, 999}
	got, err := MoeSumTopK(expertOut, weights, topkIDs, 4, 2, 2, 3)
	if err != nil {
		t.Fatalf("MoeSumTopK: %v", err)
	}
	want := []float32{8.5, 11, 13.5, 21, 24, 27}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("MoeSumTopK = %v want %v", got, want)
	}

	for _, tc := range []struct {
		name string
		call func() ([]float32, error)
	}{
		{"short weights", func() ([]float32, error) { return MoeSumTopK(expertOut, weights[:3], topkIDs, 4, 2, 2, 3) }},
		{"short expertOut", func() ([]float32, error) { return MoeSumTopK(expertOut[:5], weights, topkIDs, 4, 2, 2, 3) }},
		{"zero tokens", func() ([]float32, error) { return MoeSumTopK(expertOut, weights, topkIDs, 4, 0, 2, 3) }},
		{"zero experts", func() ([]float32, error) { return MoeSumTopK(expertOut, weights, topkIDs, 0, 2, 2, 3) }},
	} {
		_, err := tc.call()
		if !errors.Is(err, ErrMoeSumGeometry) {
			t.Errorf("%s: errors.Is(err, ErrMoeSumGeometry) = false, err=%v", tc.name, err)
		}
	}
}

package compute

// block_sparse_indexer_test.go — pins the DIRECT-PORTED SGLang DSV4.1 two-level
// block-sparse indexer contract (block_amax.cuh / candidate_block_table.cuh /
// topk_bf16_small.cuh @faaff1eca8876b8b77d8704f080c3ee2064b8b65):
//   - the acceptance invariant: TwoLevelBlockKeys selects the same key SET as the
//     flat DeepSelectTopk on the identically prepared block-maxima keys, for every
//     input (fuzzed, plus a brute-force reference on small rows);
//   - the newest-block-always-selected rule (including its slot-steal semantics);
//   - the identity fast path (nblocks <= topk), including the table builder's
//     input-independence;
//   - the candidate-block-table shape: ascending ids, INT32_MAX pad, duplicate
//     collapse, pool-slot transform, and the page-size sentinel error.
//
// Closed contracts only: exact values, set membership, and errors.Is — no error
// prose assertions.

import (
	"errors"
	"math"
	"math/rand"
	"sort"
	"testing"
)

// blockAmaxSet returns the selected block ids as a comparable map.
func blockAmaxSet(ids []int) map[int]bool {
	s := make(map[int]bool, len(ids))
	for _, id := range ids {
		s[id] = true
	}
	return s
}

// refBlockAmax is the test-local independent level-one reduction: a plain per-block
// max over exactly the seqLen-valid scores, NaN skipped, last block forced +Inf.
// Written separately from BlockAmax so the equivalence test cross-checks the
// production reducer against this scratch reference.
func refBlockAmax(scores []float64, seqLen int) []float64 {
	if seqLen <= 0 {
		return nil
	}
	nblocks := (seqLen + BlockSparseTokensPerBlock - 1) / BlockSparseTokensPerBlock
	out := make([]float64, nblocks)
	for b := 0; b < nblocks; b++ {
		lo := b * BlockSparseTokensPerBlock
		hi := lo + BlockSparseTokensPerBlock
		if hi > seqLen {
			hi = seqLen
		}
		best := math.Inf(-1)
		for i := lo; i < hi; i++ {
			v := scores[i]
			if math.IsNaN(v) || !(v > best) {
				continue
			}
			best = v
		}
		out[b] = best
	}
	out[nblocks-1] = math.Inf(1)
	return out
}

// TestBlockAmaxReducerValues pins the level-one reducer on closed values: full and
// partial blocks, the +Inf newest-block force, NaN-ignore semantics, and the
// zero-length row.
func TestBlockAmaxReducerValues(t *testing.T) {
	t.Run("full_and_partial_blocks", func(t *testing.T) {
		// 20 tokens: blocks [0..7], [8..15], [16..19] (partial).
		scores := []float64{
			1, 9, 2, 3, 4, 5, 6, 7, // block 0 -> 9
			0, -4, -2, -1, -3, 0, 0, 0, // block 1 -> 0
			-8, -9, -6, -7, // block 2 (partial) -> forced +Inf
		}
		got := BlockAmax(scores, 20)
		want := []float64{9, 0, math.Inf(1)}
		if len(got) != len(want) {
			t.Fatalf("BlockAmax length = %d, want %d", len(got), len(want))
		}
		for i := range want {
			if got[i] != want[i] {
				t.Errorf("amax[%d] = %v, want %v", i, got[i], want[i])
			}
		}
	})

	t.Run("newest_block_forced_even_when_lowest", func(t *testing.T) {
		scores := make([]float64, 16)
		for i := range scores {
			scores[i] = float64(i) // block 0 max 7, block 1 max 15...
		}
		// Make the NEWEST block the lowest: block 0 carries the global max.
		scores[0] = 100
		for i := 8; i < 16; i++ {
			scores[i] = -10
		}
		got := BlockAmax(scores, 16)
		if got[0] != 100 {
			t.Errorf("amax[0] = %v, want 100", got[0])
		}
		if got[1] != math.Inf(1) {
			t.Errorf("amax[1] = %v, want +Inf (newest block forced)", got[1])
		}
	})

	t.Run("nan_scores_ignored", func(t *testing.T) {
		nan := math.NaN()
		scores := []float64{nan, 3, nan, 1, 2, nan, 5, 4, 0, 0, 0, 0, 0, 0, 0, 0}
		got := BlockAmax(scores, 16)
		if got[0] != 5 {
			t.Errorf("amax[0] = %v, want 5 (NaN ignored, fmaxf semantics)", got[0])
		}
		if got[1] != math.Inf(1) {
			t.Errorf("amax[1] = %v, want +Inf", got[1])
		}
	})

	t.Run("all_nan_block_reduces_to_neg_inf", func(t *testing.T) {
		nan := math.NaN()
		scores := []float64{nan, nan, nan, nan, nan, nan, nan, nan, 1, 1, 1, 1, 1, 1, 1, 1}
		got := BlockAmax(scores, 16)
		if got[0] != math.Inf(-1) {
			t.Errorf("amax[0] = %v, want -Inf (never selected downstream)", got[0])
		}
	})

	t.Run("zero_length_row", func(t *testing.T) {
		if got := BlockAmax([]float64{1, 2, 3}, 0); len(got) != 0 {
			t.Errorf("BlockAmax(_, 0) = %v, want empty", got)
		}
	})

	t.Run("single_token_row", func(t *testing.T) {
		got := BlockAmax([]float64{-3}, 1)
		if len(got) != 1 || got[0] != math.Inf(1) {
			t.Errorf("BlockAmax([-3], 1) = %v, want [+Inf]", got)
		}
	})
}

// TestBlockAmaxMatchesFlatDeepSelect is the ticket's acceptance invariant: for every
// input, the two-level path selects the same key SET as the flat DeepSelectTopk on
// the identically prepared (newest-forced) block-maxima keys. The fuzz mixes
// uniform, tie-heavy, all-equal, monotone, and negative score distributions across
// lengths that are not multiples of 8, with topk spanning below/equal/above the
// block count so both the general path and the identity fast path are exercised.
func TestBlockAmaxMatchesFlatDeepSelect(t *testing.T) {
	rng := rand.New(rand.NewSource(0x5EED))
	const iters = 400
	for iter := 0; iter < iters; iter++ {
		seqLen := rng.Intn(1500) // includes 0 and non-multiples of 8
		scores := make([]float64, seqLen)
		switch rng.Intn(5) {
		case 0: // uniform
			for i := range scores {
				scores[i] = rng.Float64()*20 - 10
			}
		case 1: // tie-heavy: few distinct values
			for i := range scores {
				scores[i] = float64(rng.Intn(7) - 3)
			}
		case 2: // all equal: pure position tie-break
			v := float64(rng.Intn(5) - 2)
			for i := range scores {
				scores[i] = v
			}
		case 3: // monotone descending
			for i := range scores {
				scores[i] = float64(seqLen-i) / 4
			}
		default: // monotone ascending
			for i := range scores {
				scores[i] = float64(i) / 4
			}
		}
		topk := []int{1, 2, 3, 5, 8, 16, 32}[rng.Intn(7)]

		got := TwoLevelBlockKeys(scores, seqLen, topk)
		gotSet := blockAmaxSet(got)

		// Reference: independently reduced amax, newest block forced, flat engine
		// with a DIFFERENT geometry than production (B=B2=32 vs 1024) to also pin
		// the engine's geometry independence on this consumer.
		refAmax := refBlockAmax(scores, seqLen)
		pool := make([]Candidate, len(refAmax))
		for i, v := range refAmax {
			pool[i] = Candidate{Value: v, Position: i}
		}
		wantSel := DeepSelectTopk(pool, topk, 32, 4)
		wantSet := make(map[int]bool, len(wantSel))
		for _, c := range wantSel {
			wantSet[c.Position] = true
		}

		if len(got) != len(wantSel) {
			t.Fatalf("iter %d (seqLen=%d topk=%d): |selected| = %d, want %d", iter, seqLen, topk, len(got), len(wantSel))
		}
		if !setsEqualInt(gotSet, wantSet) {
			t.Fatalf("iter %d (seqLen=%d topk=%d): key SET mismatch:\n got  %v\n want %v", iter, seqLen, topk, sortedKeys(gotSet), sortedKeys(wantSet))
		}
		// Host contract: ascending, unique.
		for i := 1; i < len(got); i++ {
			if got[i-1] >= got[i] {
				t.Fatalf("iter %d: selection not strictly ascending: %v", iter, got)
			}
		}
	}
}

// TestBlockAmaxMatchesBruteForceReference pins the same invariant against a fully
// independent reference (hand-rolled sort, no DeepSelect engine) on small rows, so
// the fuzz above cannot be circular with the production engine.
func TestBlockAmaxMatchesBruteForceReference(t *testing.T) {
	rng := rand.New(rand.NewSource(0xBEEF))
	for iter := 0; iter < 300; iter++ {
		seqLen := rng.Intn(96)
		scores := make([]float64, seqLen)
		for i := range scores {
			scores[i] = float64(rng.Intn(9) - 4) // heavy ties
		}
		topk := 1 + rng.Intn(12)

		got := TwoLevelBlockKeys(scores, seqLen, topk)

		// Brute force: reduce, force newest, total-order sort by hand, take k.
		refAmax := refBlockAmax(scores, seqLen)
		type key struct {
			v  float64
			id int
		}
		keys := make([]key, len(refAmax))
		for i, v := range refAmax {
			keys[i] = key{v, i}
		}
		sort.Slice(keys, func(a, b int) bool {
			if keys[a].v != keys[b].v {
				return keys[a].v > keys[b].v
			}
			return keys[a].id < keys[b].id
		})
		n := topk
		if n > len(keys) {
			n = len(keys)
		}
		want := make(map[int]bool, n)
		for i := 0; i < n; i++ {
			want[keys[i].id] = true
		}
		if !setsEqualInt(blockAmaxSet(got), want) {
			t.Fatalf("iter %d (seqLen=%d topk=%d): brute-force mismatch:\n got  %v\n want %v",
				iter, seqLen, topk, sortedKeys(blockAmaxSet(got)), sortedKeys(want))
		}
	}
}

// TestBlockAmaxNewestBlockAlwaysSelected pins the recency rule: whenever a row has
// more blocks than topk, the newest block is in the selected set even when its
// natural amax is the worst — and it STEALS a slot from the unforced top-k block
// (upstream semantics, pinned so a "fix" cannot regress the port).
func TestBlockAmaxNewestBlockAlwaysSelected(t *testing.T) {
	t.Run("pinned_steal_case", func(t *testing.T) {
		// 24 tokens, 3 blocks, topk=2. Unforced block maxima: B0=10, B1=9, B2=8,
		// so the unforced top-2 is {B0, B1}. Forcing B2 to +inf selects {B0, B2}:
		// the newest block steals B1's slot.
		scores := make([]float64, 24)
		scores[0] = 10
		scores[8] = 9
		scores[16] = 8
		got := TwoLevelBlockKeys(scores, 24, 2)
		want := map[int]bool{0: true, 2: true}
		if !setsEqualInt(blockAmaxSet(got), want) {
			t.Fatalf("steal case: selected %v, want block ids {0, 2} (newest steals a slot)", got)
		}
	})

	t.Run("fuzz_newest_always_present", func(t *testing.T) {
		rng := rand.New(rand.NewSource(0x0E57))
		for iter := 0; iter < 300; iter++ {
			topk := 1 + rng.Intn(16)
			// seqLen >= 8*topk+1 guarantees nblocks > topk (general path only).
			seqLen := 8*topk + 1 + rng.Intn(400)
			scores := make([]float64, seqLen)
			for i := range scores {
				scores[i] = rng.Float64() * 10
			}
			// Adversarial: drive the newest block to the worst natural amax.
			lastLo := ((seqLen+7)/8 - 1) * 8
			for i := lastLo; i < seqLen; i++ {
				scores[i] = -1e9
			}
			nblocks := (seqLen + 7) / 8
			got := TwoLevelBlockKeys(scores, seqLen, topk)
			newest := nblocks - 1
			if !blockAmaxSet(got)[newest] {
				t.Fatalf("iter %d (seqLen=%d topk=%d): newest block %d not selected: %v",
					iter, seqLen, topk, newest, got)
			}
			if len(got) != topk {
				t.Fatalf("iter %d: |selected| = %d, want %d", iter, len(got), topk)
			}
		}
	})
}

// TestBlockAmaxIdentityFastPath pins the nblocks <= topk rule: every block is
// selected with nothing reduced or selected, and the table builder yields the
// identity table WITHOUT reading its input.
func TestBlockAmaxIdentityFastPath(t *testing.T) {
	t.Run("selection_returns_all_blocks_ascending", func(t *testing.T) {
		// Even with garbage-low scores, nblocks <= topk selects everything.
		scores := make([]float64, 40) // 5 blocks
		for i := range scores {
			scores[i] = -1e9
		}
		for _, topk := range []int{5, 6, 100} {
			got := TwoLevelBlockKeys(scores, 40, topk)
			want := []int{0, 1, 2, 3, 4}
			if len(got) != len(want) {
				t.Fatalf("topk=%d: got %v, want %v", topk, got, want)
			}
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("topk=%d: got %v, want %v", topk, got, want)
				}
			}
		}
	})

	t.Run("table_builder_ignores_input_on_identity_rows", func(t *testing.T) {
		// nblocks=3 (seqLen=24) <= topk=5: the identity table must come out even
		// though `selected` is -1 padding plus out-of-range garbage.
		pageTable := []int32{500, 600, 700, 800}
		garbage := []int{-1, -1, 99, -1, -1}
		ids, pages, err := CandidateBlockTable(garbage, 24, 5, 8, pageTable)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		wantIDs := []int32{0, 1, 2, BlockSparsePadID, BlockSparsePadID}
		// pageSize=8 -> bpp=1: page slot of block t is pageTable[t] itself.
		wantPages := []int32{500, 600, 700, BlockSparsePadID, BlockSparsePadID}
		assertInt32Slice(t, "ids", ids, wantIDs)
		assertInt32Slice(t, "pages", pages, wantPages)
	})

	t.Run("empty_row_all_pad", func(t *testing.T) {
		ids, pages, err := CandidateBlockTable(nil, 0, 4, 8, []int32{1})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		want := []int32{BlockSparsePadID, BlockSparsePadID, BlockSparsePadID, BlockSparsePadID}
		assertInt32Slice(t, "ids", ids, want)
		assertInt32Slice(t, "pages", pages, want)
	})
}

// TestBlockAmaxCandidateBlockTable pins the finaliser's closed contract: ascending
// ids, INT32_MAX past the count, duplicate collapse, -1/out-of-range drop, the
// pool-slot transform pageTable[id/bpp]*bpp + id%bpp, and the page-size sentinel.
func TestBlockAmaxCandidateBlockTable(t *testing.T) {
	t.Run("ascending_with_int32_max_pad_and_dedup", func(t *testing.T) {
		// nblocks=10 (seqLen=80), topk=4: input order scrambled, one duplicate,
		// one -1 pad. Valid ids {7, 2} -> ascending [2, 7], then pad.
		ids, pages, err := CandidateBlockTable([]int{7, -1, 2, 7}, 80, 4, 8, make([]int32, 10))
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		assertInt32Slice(t, "ids", ids, []int32{2, 7, BlockSparsePadID, BlockSparsePadID})
		// pageSize=8 -> bpp=1: slot == pageTable[id]; all zeros here.
		assertInt32Slice(t, "pages", pages, []int32{0, 0, BlockSparsePadID, BlockSparsePadID})
	})

	t.Run("pool_slot_transform", func(t *testing.T) {
		// pageSize=64 -> bpp=8 (pageBits=3). pageTable[p] = 100*p.
		// slot(id) = pageTable[id/8]*8 + id%8 = 100*(id/8)*8 + id%8.
		pageTable := []int32{0, 100, 200, 300}
		ids, pages, err := CandidateBlockTable([]int{9, 0, 15, -1}, 128, 4, 64, pageTable)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		assertInt32Slice(t, "ids", ids, []int32{0, 9, 15, BlockSparsePadID})
		// id 0  -> page 0*8+0 = 0; id 9 -> page 1*8+1 = 801; id 15 -> page 1*8+7 = 807.
		assertInt32Slice(t, "pages", pages, []int32{0, 801, 807, BlockSparsePadID})
	})

	t.Run("out_of_range_ids_dropped", func(t *testing.T) {
		// nblocks=4 (seqLen=32), topk=3: id 100 is invalid, -1 is pad -> only {2}.
		ids, _, err := CandidateBlockTable([]int{100, -1, 2}, 32, 3, 8, make([]int32, 4))
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		assertInt32Slice(t, "ids", ids, []int32{2, BlockSparsePadID, BlockSparsePadID})
	})

	t.Run("page_size_sentinel", func(t *testing.T) {
		for _, bad := range []int{0, 4, 7, 12, 48, -8} {
			_, _, err := CandidateBlockTable(nil, 8, 1, bad, nil)
			if !errors.Is(err, ErrBlockSparsePageSize) {
				t.Errorf("pageSize=%d: err = %v, want ErrBlockSparsePageSize", bad, err)
			}
		}
		// The boundary power of two is accepted.
		if _, _, err := CandidateBlockTable(nil, 8, 1, 8, make([]int32, 1)); err != nil {
			t.Errorf("pageSize=8: unexpected error %v", err)
		}
	})

	t.Run("topk_zero_yields_empty", func(t *testing.T) {
		ids, pages, err := CandidateBlockTable([]int{0}, 8, 0, 8, []int32{1})
		if err != nil || len(ids) != 0 || len(pages) != 0 {
			t.Errorf("topk=0: ids=%v pages=%v err=%v, want empty/nil", ids, pages, err)
		}
	})
}

func setsEqualInt(a, b map[int]bool) bool {
	if len(a) != len(b) {
		return false
	}
	for k := range a {
		if !b[k] {
			return false
		}
	}
	return true
}

func sortedKeys(m map[int]bool) []int {
	out := make([]int, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Ints(out)
	return out
}

func assertInt32Slice(t *testing.T, name string, got, want []int32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s length = %d, want %d (%v)", name, len(got), len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("%s[%d] = %d, want %d (full: %v)", name, i, got[i], want[i], got)
		}
	}
}

package compute

import (
	"errors"
	"math"
	"math/rand"
	"sort"
	"testing"
)

// referenceTwoPassTopK is an independent from-scratch two-pass exact block
// select oracle (pass 1: per-block top-k filtering over fixed-size blocks;
// pass 2: exact global sort under (score desc, index asc) with NaNs ordered last).
// It shares no helpers with SelectTopkTournament.
func referenceTwoPassTopK(scores []float32, k int) []TopkTournamentEntry {
	const blockSize = 64
	type item struct {
		score float32
		index int32
		nan   bool
	}
	less := func(a, b item) bool {
		if a.nan != b.nan {
			return !a.nan
		}
		if a.nan && b.nan {
			return a.index < b.index
		}
		if a.score != b.score {
			return a.score > b.score
		}
		return a.index < b.index
	}

	// Pass 1: collect at most top-k exact survivors per 64-element block.
	var survivors []item
	for start := 0; start < len(scores); start += blockSize {
		end := start + blockSize
		if end > len(scores) {
			end = len(scores)
		}
		block := make([]item, 0, end-start)
		for i := start; i < end; i++ {
			v := scores[i]
			block = append(block, item{
				score: v,
				index: int32(i),
				nan:   math.IsNaN(float64(v)),
			})
		}
		sort.Slice(block, func(i, j int) bool {
			return less(block[i], block[j])
		})
		keep := k
		if len(block) < keep {
			keep = len(block)
		}
		survivors = append(survivors, block[:keep]...)
	}

	// Pass 2: exact sort over merged block survivors.
	sort.Slice(survivors, func(i, j int) bool {
		return less(survivors[i], survivors[j])
	})
	out := make([]TopkTournamentEntry, k)
	for i := 0; i < k; i++ {
		out[i] = TopkTournamentEntry{
			Score: survivors[i].score,
			Index: survivors[i].index,
		}
	}
	return out
}

func assertTopkBitIdentical(t *testing.T, got, want []TopkTournamentEntry) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("len(got)=%d != len(want)=%d", len(got), len(want))
	}
	for i := range want {
		if got[i].Index != want[i].Index {
			t.Fatalf("rank %d index mismatch: got %d, want %d (gotScore=%g wantScore=%g)",
				i, got[i].Index, want[i].Index, got[i].Score, want[i].Score)
		}
		gotNaN := math.IsNaN(float64(got[i].Score))
		wantNaN := math.IsNaN(float64(want[i].Score))
		if gotNaN != wantNaN {
			t.Fatalf("rank %d NaN mismatch: got=%v want=%v", i, got[i].Score, want[i].Score)
		}
		if !wantNaN && math.Float32bits(got[i].Score) != math.Float32bits(want[i].Score) {
			t.Fatalf("rank %d score bits mismatch: got 0x%08x (%g), want 0x%08x (%g)",
				i, math.Float32bits(got[i].Score), got[i].Score,
				math.Float32bits(want[i].Score), want[i].Score)
		}
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestTopkTournamentFailClosedInputs(t *testing.T) {
	t.Parallel()

	if _, _, err := SelectTopkTournament(nil, 1); !errors.Is(err, ErrTopkTournamentEmpty) {
		t.Fatalf("empty scores err = %v, want ErrTopkTournamentEmpty", err)
	}
	if _, _, err := SelectTopkTournament([]float32{1, 2, 3}, 0); !errors.Is(err, ErrTopkTournamentInvalidK) {
		t.Fatalf("k=0 err = %v, want ErrTopkTournamentInvalidK", err)
	}
	if _, _, err := SelectTopkTournament([]float32{1, 2, 3}, 4); !errors.Is(err, ErrTopkTournamentInvalidK) {
		t.Fatalf("k>n err = %v, want ErrTopkTournamentInvalidK", err)
	}
	big := make([]float32, 32)
	if _, _, err := SelectTopkTournament(big, TopkTournamentMaxK+1); !errors.Is(err, ErrTopkTournamentInvalidK) {
		t.Fatalf("k>max err = %v, want ErrTopkTournamentInvalidK", err)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestTopkTournamentBitIdenticalTwoPassOracle(t *testing.T) {
	t.Parallel()

	t.Run("dense fast path without flagged columns", func(t *testing.T) {
		// Well-separated positive/negative scores fitting a single 256-candidate pass:
		// zero low-4-bit collisions -> zero refinement rounds, dense fast path true.
		scores := make([]float32, 128)
		for i := range scores {
			scores[i] = float32(i*3 - 100)
		}
		want := referenceTwoPassTopK(scores, 16)
		got, receipt, err := SelectTopkTournament(scores, 16)
		if err != nil {
			t.Fatalf("SelectTopkTournament: %v", err)
		}
		if !receipt.DenseFastPath {
			t.Fatalf("expected DenseFastPath=true for n=128")
		}
		if receipt.FlaggedLanes != 0 || receipt.RefinementRounds != 0 || receipt.WarpVotes != 1 {
			t.Fatalf("expected zero-collision fast exit, got receipt %+v", receipt)
		}
		assertTopkBitIdentical(t, got, want)
	})

	t.Run("low-4-bit mantissa collision triggers warp-vote early stop", func(t *testing.T) {
		// Lane 0 candidates differ ONLY in the low 4 bits of float32 mantissa
		// (0x3f800000 .. 0x3f80000f). Packed keys collide on the top 28 bits;
		// two adjacent inversions resolve in round 1 and stop early (< 4 rounds).
		scores := make([]float32, 256)
		for i := range scores {
			scores[i] = -10.0 - float32(i)
		}
		// Plant two adjacent elements whose low 4 mantissa bits swap order vs packed index:
		// idx 0 = 1.0 + 1*ULP (packed low nibble 0), idx 1 = 1.0 + 9*ULP (packed low nibble 1).
		// Wait: packed key for idx 0 is 0x3f800000, idx 1 is 0x3f800001; reverse the values
		// so idx 0 has LARGER true value (0x3f800009) while idx 1 has SMALLER (0x3f800002):
		// packed key puts idx 1 (0x3f800001) ABOVE idx 0 (0x3f800000), which only exact
		// refinement fixes!
		scores[0] = math.Float32frombits(0x3f800009)
		scores[1] = math.Float32frombits(0x3f800002)
		scores[2] = math.Float32frombits(0x3f80000c)
		scores[3] = math.Float32frombits(0x3f800004)

		want := referenceTwoPassTopK(scores, 8)
		got, receipt, err := SelectTopkTournament(scores, 8)
		if err != nil {
			t.Fatalf("SelectTopkTournament: %v", err)
		}
		if !receipt.DenseFastPath {
			t.Fatalf("expected DenseFastPath=true for n=256")
		}
		if receipt.FlaggedLanes == 0 {
			t.Fatalf("expected FlaggedLanes > 0 on low-4-bit collision")
		}
		if receipt.RefinementRounds == 0 || receipt.RefinementRounds >= receipt.MaxRefinementRounds {
			t.Fatalf("expected warp-vote early stop (0 < rounds < %d), got %+v",
				receipt.MaxRefinementRounds, receipt)
		}
		assertTopkBitIdentical(t, got, want)
	})

	t.Run("16-way reversed low-4-bit cluster exercises full fallback", func(t *testing.T) {
		scores := make([]float32, 256)
		for i := range scores {
			scores[i] = -5.0 - float32(i)
		}
		// All 16 elements in lane 0 share top 28 bits 0x4040000x, decreasing in true
		// score while local index increases 0..15, so packed-key order is inverted.
		for i := 0; i < 16; i++ {
			scores[i] = math.Float32frombits(0x4040000f - uint32(i))
		}
		want := referenceTwoPassTopK(scores, 16)
		got, receipt, err := SelectTopkTournament(scores, 16)
		if err != nil {
			t.Fatalf("SelectTopkTournament: %v", err)
		}
		if receipt.FlaggedLanes == 0 {
			t.Fatalf("expected FlaggedLanes > 0, got %+v", receipt)
		}
		assertTopkBitIdentical(t, got, want)
	})

	t.Run("multi-group fixtures with negatives, signed zero, Inf, NaN, and near ties", func(t *testing.T) {
		rng := rand.New(rand.NewSource(20261106))
		cases := []struct {
			name string
			n    int
			k    int
		}{
			{"n1", 1, 1},
			{"n17_k5", 17, 5},
			{"n256_k16", 256, 16},
			{"n1000_k7", 1000, 7},
			{"n2048_k16", 2048, 16},
		}
		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				scores := make([]float32, tc.n)
				for i := range scores {
					scores[i] = rng.Float32()*200 - 100
				}
				if tc.n >= 8 {
					scores[0] = float32(math.NaN())
					scores[1] = float32(math.Inf(-1))
					scores[2] = math.Float32frombits(0x80000000) // -0.0
					scores[3] = 0.0
					scores[4] = math.Float32frombits(0x80000001) // -subnormal
					scores[5] = 99.5
					scores[6] = 99.5 // exact tie
					scores[7] = float32(math.Inf(1))
				}
				if tc.n > 1500 {
					// Plant cross-group ties and ULP-level near ties.
					scores[300] = math.Float32frombits(0x42c80003)
					scores[900] = math.Float32frombits(0x42c8000b)
					scores[1500] = math.Float32frombits(0x42c8000b)
				}
				want := referenceTwoPassTopK(scores, tc.k)
				got, receipt, err := SelectTopkTournament(scores, tc.k)
				if err != nil {
					t.Fatalf("SelectTopkTournament: %v", err)
				}
				wantDense := tc.n <= TopkTournamentGroupCapacity
				if receipt.DenseFastPath != wantDense {
					t.Fatalf("DenseFastPath=%v, want %v", receipt.DenseFastPath, wantDense)
				}
				assertTopkBitIdentical(t, got, want)
			})
		}
	})
}

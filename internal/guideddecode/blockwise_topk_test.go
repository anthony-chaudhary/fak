package guideddecode

import (
	"math"
	"testing"
)

// blockwiseTolerance is the fp32 comparison bound for the streaming merge versus
// the full-candidate reference. Both paths sum the same two fp32 terms per pair
// in the same order, so exact equality is expected; a tiny tolerance absorbs a
// future reassociation without letting a real ordering bug pass.
func blockwiseTolerance(a, b float32) float32 {
	scale := float32(math.Abs(float64(a)))
	if s := float32(math.Abs(float64(b))); s > scale {
		scale = s
	}
	if scale == 0 {
		return 1e-6
	}
	return scale * 1e-6
}

// blockwiseFixture is the deterministic start/end/score fixture the witnesses
// share: a handful of valid starts and dead slots, ends that include one behind
// every start (so the end > start guard bites), and scores with exact ties.
func blockwiseFixture() (starts []BlockwiseStartSlot, ends []int, endScores []float32) {
	starts = []BlockwiseStartSlot{
		{Index: 0, Score: 0.5, Valid: true},
		{Index: 2, Score: -1.25, Valid: true},
		{Index: 5, Score: 2.0, Valid: true},
		{Index: 7, Score: 0.0, Valid: false}, // dead slot: must spawn nothing
	}
	ends = []int{0, 1, 2, 3, 4, 5, 6, 7}
	endScores = []float32{3.0, 1.5, 3.0, -0.25, 2.25, 1.5, 4.0, -2.0}
	return starts, ends, endScores
}

// blockwiseSimpleScore is the reference score: start marginal + end score, no
// extra admission predicate. It is the default the witnesses exercise.
func blockwiseSimpleScore(start BlockwiseStartSlot, endIndex int, endScore float32) (float32, bool) {
	return start.Score + endScore, true
}

// blockwiseRunBlocks feeds the fixture through the accumulator one block of
// blockSize ends at a time, in ascending end order, and returns the final result.
func blockwiseRunBlocks(starts []BlockwiseStartSlot, ends []int, endScores []float32, ks, blockSize int) *BlockwiseTopK {
	b := NewBlockwiseTopK(len(starts), ks)
	for q := range starts {
		b.SetStart(q, starts[q])
	}
	for j0 := 0; j0 < len(ends); j0 += blockSize {
		j1 := j0 + blockSize
		if j1 > len(ends) {
			j1 = len(ends)
		}
		b.ScoreBlock(ends[j0:j1], endScores[j0:j1], blockwiseSimpleScore)
	}
	return b
}

// TestBlockwiseTopKMatchesReference is the equivalence witness: for every block
// size, the streaming blockwise ranker produces exactly the reference's per-start
// top-k — same (end, score) pairs in the same descending order. It also pins that
// no end at or before a start index is admitted, and that a dead start slot spawns
// nothing.
func TestBlockwiseTopKMatchesReference(t *testing.T) {
	starts, ends, endScores := blockwiseFixture()
	const ks = 3
	want := BlockwiseTopKReference(starts, ends, endScores, ks, blockwiseSimpleScore)

	for _, blockSize := range []int{1, 2, 3, 4, 5, 8, 16} {
		got := blockwiseRunBlocks(starts, ends, endScores, ks, blockSize).Results()
		if len(got) != len(want) {
			t.Fatalf("blockSize=%d: %d start slots, want %d", blockSize, len(got), len(want))
		}
		for q := range want {
			if len(got[q]) != ks {
				t.Fatalf("blockSize=%d: slot %d returned %d entries, want %d", blockSize, q, len(got[q]), ks)
			}
			for r := range want[q] {
				w := want[q][r]
				g := got[q][r]
				if g.End != w.End {
					t.Fatalf("blockSize=%d slot %d rank %d: end = %d, want %d", blockSize, q, r, g.End, w.End)
				}
				if d := float32(math.Abs(float64(g.Score - w.Score))); d > blockwiseTolerance(g.Score, w.Score) {
					t.Fatalf("blockSize=%d slot %d rank %d: score = %v, want %v", blockSize, q, r, g.Score, w.Score)
				}
			}
		}
	}
	// The dead slot (Valid=false) must have produced sentinel placeholders only.
	dead := blockwiseRunBlocks(starts, ends, endScores, ks, 4).Results()
	for _, p := range dead[3] {
		if p.Score != blockwiseMaskLogit {
			t.Fatalf("dead start slot admitted %+v, want only %v placeholders", p, blockwiseMaskLogit)
		}
	}
	// The start with Index 0 must never pair with end 0 (end > start is required).
	for r, p := range want[0] {
		if p.End <= starts[0].Index {
			t.Fatalf("slot 0 rank %d: end %d is not strictly after start %d", r, p.End, starts[0].Index)
		}
	}
}

// TestBlockwiseTopKStableTieOrder pins the reference's rebaselined tie contract:
// exact-score ties retain concatenation order, so an earlier (lower-index) end
// keeps the better rank when a later block adds an equal score. The fixture gives
// ends 0 and 2 the same end score, so one block merges both at once.
func TestBlockwiseTopKStableTieOrder(t *testing.T) {
	b := NewBlockwiseTopK(1, 2)
	b.SetStart(0, BlockwiseStartSlot{Index: -1, Score: 0, Valid: true})
	b.ScoreBlock([]int{0, 2}, []float32{3.0, 3.0}, blockwiseSimpleScore)
	got := b.Results()[0]
	if got[0].End != 0 || got[1].End != 2 {
		t.Fatalf("tie order = ends [%d %d], want [0 2] (lower index first)", got[0].End, got[1].End)
	}
	if got[0].Score != got[1].Score {
		t.Fatalf("tie scores diverged: %v vs %v", got[0].Score, got[1].Score)
	}

	// A later, equal-scoring block entry must NOT displace the running entry: the
	// running end 0 stays ranked ahead of the newly-merged end 9.
	b2 := NewBlockwiseTopK(1, 2)
	b2.SetStart(0, BlockwiseStartSlot{Index: -1, Score: 0, Valid: true})
	b2.ScoreBlock([]int{0}, []float32{5.0}, blockwiseSimpleScore)
	b2.ScoreBlock([]int{9}, []float32{5.0}, blockwiseSimpleScore)
	got2 := b2.Results()[0]
	if got2[0].End != 0 || got2[1].End != 9 {
		t.Fatalf("cross-block tie order = ends [%d %d], want [0 9] (running entry wins the tie)", got2[0].End, got2[1].End)
	}
}

// TestBlockwiseTopKMemoryBound proves the ranker stays linear in memory: the
// largest block tensor for Q slots / Ks retained ends / blockSize is
// Q*Ks*blockSize, and the same accumulator over a much longer end stream does not
// grow that bound. The [Q,Ks,L] full-matrix alternative is never materialized.
func TestBlockwiseTopKMemoryBound(t *testing.T) {
	const (
		q         = 4
		ks        = 3
		blockSize = 64
		l         = 4096
	)
	bound := MaxBlockwiseIntermediateElems(q, ks, blockSize)
	if bound != q*ks*blockSize {
		t.Fatalf("MaxBlockwiseIntermediateElems = %d, want %d", bound, q*ks*blockSize)
	}
	if bound >= q*ks*l {
		t.Fatalf("bound %d is not linear in L=%d (full matrix is %d)", bound, l, q*ks*l)
	}

	starts := make([]BlockwiseStartSlot, q)
	for i := range starts {
		starts[i] = BlockwiseStartSlot{Index: i, Score: float32(i) * 0.5, Valid: true}
	}
	b := NewBlockwiseTopK(q, ks)
	for i := range starts {
		b.SetStart(i, starts[i])
	}
	ends := make([]int, blockSize)
	scores := make([]float32, blockSize)
	for i := 0; i < l; i++ {
		for j := range ends {
			ends[j] = i*blockSize + j // plateau on one block width
			scores[j] = float32(j%5) - 2
		}
		if _, ok := b.ScoreBlock(ends, scores, blockwiseSimpleScore); !ok {
			t.Fatalf("ScoreBlock refused a well-formed block at i=%d", i)
		}
	}
	if got := b.MaxBlockElements(); got != bound {
		t.Fatalf("MaxBlockElements = %d, want %d", got, bound)
	}
	if b.ConditionalElements() != l*blockSize*q {
		t.Fatalf("ConditionalElements = %d, want %d", b.ConditionalElements(), l*blockSize*q)
	}
	if b.BlocksScored() != l {
		t.Fatalf("BlocksScored = %d, want %d", b.BlocksScored(), l)
	}
}

// TestBlockwiseTopKAdmissionVeto pins the fail-closed scaffolding: an admiss
// predicate that vetoes a pair keeps it out of the result even when it would have
// ranked, and a malformed block (mismatched lengths) is refused rather than
// partially consumed.
func TestBlockwiseTopKAdmissionVeto(t *testing.T) {
	starts, ends, endScores := blockwiseFixture()
	vetoEven := func(_ BlockwiseStartSlot, endIndex int, _ float32) (float32, bool) {
		return 0, endIndex%2 == 1 // admit odd ends only
	}
	b := NewBlockwiseTopK(len(starts), 3)
	for q := range starts {
		b.SetStart(q, starts[q])
	}
	b.ScoreBlock(ends, endScores, vetoEven)
	for q, slot := range b.Results() {
		for _, p := range slot {
			if p.Score != blockwiseMaskLogit && p.End%2 == 0 {
				t.Fatalf("slot %d admitted vetoed even end %d (score %v)", q, p.End, p.Score)
			}
		}
	}
	if _, ok := b.ScoreBlock([]int{1, 2}, []float32{1}, nil); ok {
		t.Fatal("ScoreBlock accepted a mismatched-length block, want refusal")
	}
}

// blockwiseLcg is a tiny deterministic LCG so the differential witness has no
// dependency on math/rand global state and reproduces byte-for-byte.
func blockwiseLcg(state *uint64) uint32 {
	*state = *state*6364136223846793005 + 1442695040888963407
	return uint32(*state >> 33)
}

// TestBlockwiseTopKDifferentialRandomShapes is the stronger equivalence witness:
// across many random (Q, Ks, L, blockSize) shapes and random score patterns, the
// streaming ranker's per-start top-k equals a from-scratch oracle that sorts every
// admissible pair by (score desc, enumeration order). It exercises block seams,
// ties, vetoes, and dead slots that the fixed fixture cannot.
func TestBlockwiseTopKDifferentialRandomShapes(t *testing.T) {
	var state uint64 = 0xC0FFEE
	for trial := 0; trial < 200; trial++ {
		q := 1 + int(blockwiseLcg(&state)%4)
		ks := 1 + int(blockwiseLcg(&state)%4)
		l := 1 + int(blockwiseLcg(&state)%24)
		blockSize := 1 + int(blockwiseLcg(&state)%7)

		starts := make([]BlockwiseStartSlot, q)
		for i := range starts {
			starts[i] = BlockwiseStartSlot{
				Index: int(blockwiseLcg(&state) % uint32(l+1)),
				Score: float32(int(blockwiseLcg(&state)%7) - 3),
				Valid: blockwiseLcg(&state)%5 != 0,
			}
		}
		ends := make([]int, l)
		scores := make([]float32, l)
		for i := range ends {
			ends[i] = i
			scores[i] = float32(int(blockwiseLcg(&state)%5) - 2) // deliberate ties
		}
		admiss := func(_ BlockwiseStartSlot, endIndex int, _ float32) (float32, bool) {
			return 0, endIndex%3 != 0
		}

		b := NewBlockwiseTopK(q, ks)
		for i := range starts {
			b.SetStart(i, starts[i])
		}
		for j0 := 0; j0 < l; j0 += blockSize {
			j1 := j0 + blockSize
			if j1 > l {
				j1 = l
			}
			if _, ok := b.ScoreBlock(ends[j0:j1], scores[j0:j1], admiss); !ok {
				t.Fatalf("trial %d: ScoreBlock refused a valid block", trial)
			}
		}

		oracle := blockwiseOracle(starts, ends, scores, ks, admiss)
		got := b.Results()
		for i := range oracle {
			// Compare only the real (non-sentinel) prefix of the top-k.
			want := oracle[i]
			var realGot []BlockwisePair
			for _, p := range got[i] {
				if p.Score != blockwiseMaskLogit {
					realGot = append(realGot, p)
				}
			}
			if len(realGot) != len(want) {
				t.Fatalf("trial %d slot %d: %d real pairs, want %d", trial, i, len(realGot), len(want))
			}
			for r := range want {
				if realGot[r].End != want[r].End || realGot[r].Score != want[r].Score {
					t.Fatalf("trial %d slot %d rank %d: got %+v, want %+v", trial, i, r, realGot[r], want[r])
				}
			}
		}
	}
}

// blockwiseOracle is an independent from-scratch reference: it enumerates every
// admissible (slot, end) pair, stable-sorts by score descending (the enumerate
// order is the tie order), and takes the top Ks. Unlike BlockwiseTopKReference it
// uses an explicit insertion sort, so the two references differ in mechanism and
// a shared bug cannot hide in both.
func blockwiseOracle(starts []BlockwiseStartSlot, ends []int, endScores []float32, ks int, admiss endScoreFunc) [][]BlockwisePair {
	out := make([][]BlockwisePair, len(starts))
	for q, start := range starts {
		var all []BlockwisePair
		for i, e := range ends {
			if !start.Valid || e <= start.Index {
				continue
			}
			score := start.Score + endScores[i]
			if admiss != nil {
				s, ok := admiss(start, e, endScores[i])
				if !ok {
					continue
				}
				score = s
			}
			all = append(all, BlockwisePair{Start: start.Index, End: e, Score: score})
		}
		// Stable descending insertion sort: insert each element before the first
		// strictly-lower element, so equal scores keep enumerate order.
		for i := 1; i < len(all); i++ {
			key := all[i]
			j := i - 1
			for j >= 0 && all[j].Score < key.Score {
				all[j+1] = all[j]
				j--
			}
			all[j+1] = key
		}
		if len(all) > ks {
			all = all[:ks]
		}
		out[q] = all
	}
	return out
}

// TestBlockwiseTopKDegenerateShapes pins the boundaries: a non-positive Q or Ks
// refuses with a nil accumulator, an empty block is a no-op, and a slot that
// never received a block still reports Ks sentinel placeholders rather than a
// short or panicking result.
func TestBlockwiseTopKDegenerateShapes(t *testing.T) {
	if NewBlockwiseTopK(0, 3) != nil || NewBlockwiseTopK(3, 0) != nil || NewBlockwiseTopK(-1, -1) != nil {
		t.Fatal("NewBlockwiseTopK accepted a non-positive shape, want nil")
	}
	b := NewBlockwiseTopK(2, 2)
	b.SetStart(0, BlockwiseStartSlot{Index: 0, Valid: true})
	if n, ok := b.ScoreBlock(nil, nil, nil); !ok || n != 0 {
		t.Fatalf("empty block = (%d, %v), want (0, true)", n, ok)
	}
	got := b.Results()
	if len(got) != 2 || len(got[1]) != 2 {
		t.Fatalf("results shape = %d slots, want 2 slots of 2", len(got))
	}
	if got[1][0].Score != blockwiseMaskLogit {
		t.Fatalf("untouched slot score = %v, want sentinel %v", got[1][0].Score, blockwiseMaskLogit)
	}
	if b.SetStart(5, BlockwiseStartSlot{}) {
		t.Fatal("SetStart accepted an out-of-range slot, want refusal")
	}
	var nilAcc *BlockwiseTopK
	if _, ok := nilAcc.ScoreBlock(nil, nil, nil); ok {
		t.Fatal("nil accumulator accepted a block, want refusal")
	}
	if nilAcc.Results() != nil || nilAcc.ConditionalElements() != 0 || nilAcc.MaxBlockElements() != 0 {
		t.Fatal("nil accumulator accessors did not fail closed")
	}
}

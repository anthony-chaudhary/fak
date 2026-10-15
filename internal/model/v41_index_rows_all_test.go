package model

import (
	"cmp"
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"slices"
	"testing"
)

// Preserve the prior full-sort algorithm as the focused equivalence oracle.
func v41IndexRowsAllSortedOracle(logits []float32, compressLen, topK, offset int, candidates []bool) ([]int32, error) {
	if compressLen < 0 || compressLen > len(logits) {
		return nil, fmt.Errorf("v41 index rows: compress length %d outside [0,%d]", compressLen, len(logits))
	}
	if topK < 0 {
		return nil, fmt.Errorf("v41 index rows: negative top-k %d", topK)
	}
	if candidates != nil && len(candidates) != len(logits) {
		return nil, fmt.Errorf("v41 index rows: candidate mask length %d, want %d", len(candidates), len(logits))
	}
	if offset < 0 || int64(offset) > math.MaxInt32-int64(max(0, compressLen-1)) {
		return nil, fmt.Errorf("v41 index rows: offset %d cannot represent visible rows as int32", offset)
	}

	ranked := make([]v41RankedScore, len(logits))
	for position, score := range logits {
		if math.IsNaN(float64(score)) {
			return nil, fmt.Errorf("v41 index rows: score %d is NaN", position)
		}
		if position >= compressLen || (candidates != nil && !candidates[position]) {
			score = float32(math.Inf(-1))
		}
		ranked[position] = v41RankedScore{index: position, score: score}
	}
	sortV41Scores(ranked)
	ranked = ranked[:min(topK, len(ranked))]
	slices.SortFunc(ranked, func(a, b v41RankedScore) int { return cmp.Compare(a.index, b.index) })

	rows := make([]int32, len(ranked))
	for i, selected := range ranked {
		if selected.index >= compressLen || math.IsInf(float64(selected.score), -1) {
			rows[i] = -1
			continue
		}
		rows[i] = int32(offset + selected.index)
	}
	return rows, nil
}

// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime fast est=1s lane=default
func TestV41IndexRowsAllParity(t *testing.T) {
	rng := rand.New(rand.NewSource(12640))
	for n := 0; n <= 129; n++ {
		scores := make([]float32, n)
		for i := range scores {
			scores[i] = float32(rng.Intn(7) - 3)
		}
		if n > 3 {
			scores[0] = float32(math.Inf(-1))
			scores[1] = float32(math.Inf(1))
			scores[2] = math.Float32frombits(0x80000000)
		}
		for _, visible := range []int{0, n / 2, n} {
			for _, k := range []int{0, n / 2, n, n + 3} {
				for _, masked := range []bool{false, true} {
					var mask []bool
					if masked {
						mask = make([]bool, n)
						for i := range mask {
							mask[i] = i%3 != 0
						}
					}
					before := append([]float32(nil), scores...)
					got, ge := V41SelectIndexRows(scores, visible, k, 7, mask)
					want, we := v41IndexRowsAllSortedOracle(scores, visible, k, 7, mask)
					if !reflect.DeepEqual(got, want) || fmt.Sprint(ge) != fmt.Sprint(we) {
						t.Fatalf("n=%d visible=%d k=%d mask=%t got=%v/%v want=%v/%v", n, visible, k, masked, got, ge, want, we)
					}
					for i := range before {
						if math.Float32bits(before[i]) != math.Float32bits(scores[i]) {
							t.Fatal("mutated logits")
						}
					}
				}
			}
		}
	}
	for _, tc := range []struct {
		scores             []float32
		visible, k, offset int
		mask               []bool
	}{
		{nil, 0, 0, 0, nil}, {[]float32{}, 0, 3, 0, nil},
		{[]float32{1, float32(math.NaN())}, 0, 2, 0, []bool{false, false}},
		{[]float32{float32(math.NaN()), 1}, 2, 2, math.MaxInt32, nil},
		{[]float32{float32(math.NaN())}, -1, 1, -1, nil},
		{[]float32{float32(math.NaN())}, 1, -1, -1, nil},
		{[]float32{float32(math.NaN())}, 1, 1, -1, []bool{}},
		{[]float32{1}, 1, 1, math.MaxInt32, nil},
	} {
		got, ge := V41SelectIndexRows(tc.scores, tc.visible, tc.k, tc.offset, tc.mask)
		want, we := v41IndexRowsAllSortedOracle(tc.scores, tc.visible, tc.k, tc.offset, tc.mask)
		if !reflect.DeepEqual(got, want) || fmt.Sprint(ge) != fmt.Sprint(we) {
			t.Fatalf("case=%+v got=%v/%v want=%v/%v", tc, got, ge, want, we)
		}
	}
}

// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime fast est=1s lane=default
func TestV41IndexRowsAllAllocations(t *testing.T) {
	scores := make([]float32, 512)
	for i := range scores {
		scores[i] = float32(512 - i)
	}
	var out []int32
	before := testing.AllocsPerRun(100, func() { out, _ = v41IndexRowsAllSortedOracle(scores, 512, 512, 0, nil) })
	after := testing.AllocsPerRun(100, func() { out, _ = V41SelectIndexRows(scores, 512, 512, 0, nil) })
	if len(out) != 512 || before-after != 1 {
		t.Fatalf("sorted=%g direct=%g", before, after)
	}
	t.Logf("all-selected allocations sorted=%g direct=%g", before, after)
}

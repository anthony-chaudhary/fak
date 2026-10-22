package model

import (
	"math"
	"reflect"
	"testing"
)

// fak-test:runtime fast est=10ms lane=default
func TestAllocateSpeculativeVerificationBudget(t *testing.T) {
	t.Parallel()
	t.Run("contract", func(t *testing.T) {
		cases := []struct {
			name          string
			previous      [][]float32
			remap         []int
			depth, budget int
			want          []int
		}{
			{"conditional not marginal", [][]float32{{.9, .1}, {.8, .9}}, []int{0, 1}, 2, 5, []int{2, 3}},
			{"float32 product tie", [][]float32{{0.7199999690055847, 0}, {.9, .8}}, []int{0, 1}, 2, 4, []int{2, 2}},
			{"remapped and fresh", [][]float32{{.9, .1}, {.8, .9}}, []int{1, -1, 0}, 2, 7, []int{3, 1, 3}},
			{"duplicate mappings", [][]float32{{.5, 1}}, []int{0, 0}, 2, 4, []int{3, 1}},
			{"fresh zero tie", [][]float32{{0, 1}}, []int{-1, 0}, 2, 3, []int{2, 1}},
			{"empty previous", nil, []int{-1, -1}, 2, 5, []int{3, 2}},
			{"mandatory target", [][]float32{{1, 1}}, []int{0, -1}, 2, 2, []int{1, 1}},
			{"zero depth", [][]float32{{}}, []int{0, -1}, 0, 2, []int{1, 1}},
			{"empty batch", nil, nil, 2, 0, []int{}},
			{"underflow zero tie", [][]float32{{math.SmallestNonzeroFloat32, .5}}, []int{-1, 0}, 2, 4, []int{2, 2}},
		}
		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				before := make([][]float32, len(tc.previous))
				for i, row := range tc.previous {
					before[i] = append([]float32{}, row...)
				}
				remapBefore := append([]int{}, tc.remap...)
				got, err := AllocateSpeculativeVerificationBudget(tc.previous, tc.remap, tc.depth, tc.budget)
				if err != nil || !reflect.DeepEqual(got, tc.want) {
					t.Fatalf("got %v, %v; want %v", got, err, tc.want)
				}
				for i, row := range tc.previous {
					if !reflect.DeepEqual(row, before[i]) {
						t.Fatal("mutated previous confidence")
					}
				}
				for i, row := range tc.remap {
					if row != remapBefore[i] {
						t.Fatal("mutated remap")
					}
				}
			})
		}
	})
	t.Run("exhaustive prefix oracle", func(t *testing.T) {
		// Enumerate all two-row depth-two inputs over {0, .5, 1}. The oracle
		// enumerates legal prefix lengths, maximizing their summed acceptance
		// and breaking ties lexicographically, independently of candidate sort.
		for encoding := 0; encoding < 81; encoding++ {
			previous := [][]float32{make([]float32, 2), make([]float32, 2)}
			code := encoding
			for i := 0; i < 4; i++ {
				previous[i/2][i%2] = float32(code%3) / 2
				code /= 3
			}
			remap := []int{1, -1, 0}
			scores := [3][2]float64{}
			for lane, row := range remap {
				if row >= 0 {
					scores[lane] = [2]float64{float64(previous[row][0]), float64(previous[row][0] * previous[row][1])}
				}
			}
			for budget := 3; budget <= 9; budget++ {
				best := -1.0
				var want []int
				// Descending lengths make the first optimum favor the lower
				// lane-major flat index, including every zero-confidence tie.
				for a := 2; a >= 0; a-- {
					for b := 2; b >= 0; b-- {
						for c := 2; c >= 0; c-- {
							if a+b+c != budget-3 {
								continue
							}
							counts := []int{a, b, c}
							sum := 0.0
							for lane, count := range counts {
								for j := 0; j < count; j++ {
									sum += scores[lane][j]
								}
							}
							if sum > best {
								best = sum
								want = []int{a + 1, b + 1, c + 1}
							}
						}
					}
				}
				got, err := AllocateSpeculativeVerificationBudget(previous, remap, 2, budget)
				if err != nil || !reflect.DeepEqual(got, want) {
					t.Fatalf("input %v budget %d: got %v, %v; oracle %v", previous, budget, got, err, want)
				}
			}
		}
	})
	t.Run("reject invalid without allocation", func(t *testing.T) {
		maxInt := int(^uint(0) >> 1)
		cases := []struct {
			previous      [][]float32
			remap         []int
			depth, budget int
		}{
			{nil, nil, -1, 0}, {nil, nil, maxInt, 0},
			{nil, []int{-1, -1}, maxInt / 2, 2},
			{nil, []int{-1}, 2, 0}, {nil, []int{-1}, 2, 4},
			{nil, nil, 2, 1}, {nil, []int{-2}, 2, 1}, {nil, []int{0}, 2, 1},
			{[][]float32{{1}}, []int{0}, 2, 1},
			{[][]float32{{-1}}, []int{-1}, 1, 1},
			{[][]float32{{1.1}}, []int{0}, 1, 1},
			{[][]float32{{float32(math.NaN())}}, []int{0}, 1, 1},
			{[][]float32{{float32(math.Inf(1))}}, []int{0}, 1, 1},
			{[][]float32{{float32(math.Inf(-1))}}, []int{0}, 1, 1},
		}
		for i, tc := range cases {
			got, err := AllocateSpeculativeVerificationBudget(tc.previous, tc.remap, tc.depth, tc.budget)
			if err == nil || got != nil {
				t.Fatalf("invalid case %d: got %v, %v", i, got, err)
			}
		}
	})
}

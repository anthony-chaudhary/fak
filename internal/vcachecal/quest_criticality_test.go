package vcachecal

import (
	"errors"
	"math"
	"math/rand"
	"reflect"
	"testing"
)

func questDot(a, b []float64) float64 {
	s := 0.0
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

func questRandVec(r *rand.Rand, dim int) []float64 {
	v := make([]float64, dim)
	for i := range v {
		v[i] = r.NormFloat64() * 2
	}
	return v
}

func TestQuestCriticalityUpperBound(t *testing.T) {
	t.Run("hand_computed", func(t *testing.T) {
		q := []float64{2, -3, 0}
		kmin := []float64{-1, -2, -5}
		kmax := []float64{4, 1, 5}
		// d0: max(8,-2)=8; d1: max(-3,6)=6; d2: 0.
		got, err := QuestCriticality(q, kmin, kmax)
		if err != nil || got != 14 {
			t.Fatalf("got %v, %v; want 14", got, err)
		}
	})

	t.Run("bound_dominates_true_max_random", func(t *testing.T) {
		r := rand.New(rand.NewSource(7))
		for trial := 0; trial < 500; trial++ {
			dim := 1 + r.Intn(16)
			pageLen := 1 + r.Intn(12)
			keys := make([][]float64, pageLen)
			for i := range keys {
				keys[i] = questRandVec(r, dim)
			}
			q := questRandVec(r, dim)
			kmin, kmax, ok := QuestPageBounds(keys)
			if !ok {
				t.Fatalf("trial %d: bounds not ok", trial)
			}
			bound, err := QuestCriticality(q, kmin, kmax)
			if err != nil {
				t.Fatal(err)
			}
			trueMax := math.Inf(-1)
			for _, k := range keys {
				trueMax = math.Max(trueMax, questDot(q, k))
			}
			if bound < trueMax-1e-9 {
				t.Fatalf("trial %d: bound %v < true max %v", trial, bound, trueMax)
			}
		}
	})

	t.Run("single_key_page_is_tight", func(t *testing.T) {
		k := []float64{1.5, -2, 0.25}
		q := []float64{-1, 4, 8}
		kmin, kmax, _ := QuestPageBounds([][]float64{k})
		got, _ := QuestCriticality(q, kmin, kmax)
		if got != questDot(q, k) {
			t.Fatalf("single-key bound %v != q.k %v", got, questDot(q, k))
		}
	})

	t.Run("ranking_by_bound", func(t *testing.T) {
		q := []float64{1, 1}
		mins := [][]float64{{0, 0}, {5, 5}, {1, 1}, {5, 5}}
		maxs := [][]float64{{1, 1}, {6, 6}, {2, 2}, {6, 6}}
		ranked, err := QuestRankPages(q, mins, maxs)
		if err != nil {
			t.Fatal(err)
		}
		var order []int
		for _, p := range ranked {
			order = append(order, p.Page)
		}
		if want := []int{1, 3, 2, 0}; !reflect.DeepEqual(order, want) {
			t.Fatalf("order %v want %v (ties by page index)", order, want)
		}
		if ranked[0].Score != 12 || ranked[3].Score != 2 {
			t.Fatalf("scores %+v", ranked)
		}
	})

	t.Run("ranking_finds_planted_hot_page", func(t *testing.T) {
		r := rand.New(rand.NewSource(11))
		dim, pages, pageLen := 8, 20, 4
		q := questRandVec(r, dim)
		mins := make([][]float64, pages)
		maxs := make([][]float64, pages)
		for p := 0; p < pages; p++ {
			keys := make([][]float64, pageLen)
			for i := range keys {
				keys[i] = make([]float64, dim)
				for d := range keys[i] {
					keys[i][d] = r.Float64()*0.2 - 0.1
				}
			}
			if p == 13 {
				for d := range keys[0] {
					keys[0][d] = 10 * q[d]
				}
			}
			mins[p], maxs[p], _ = QuestPageBounds(keys)
		}
		ranked, err := QuestRankPages(q, mins, maxs)
		if err != nil {
			t.Fatal(err)
		}
		if ranked[0].Page != 13 {
			t.Fatalf("top page %d want 13", ranked[0].Page)
		}
	})

	t.Run("select_keeps_last_page", func(t *testing.T) {
		q := []float64{1}
		mins := [][]float64{{0}, {9}, {3}, {7}, {-5}}
		maxs := [][]float64{{1}, {10}, {4}, {8}, {-4}}
		cases := []struct {
			budget int
			want   []int
		}{
			{0, nil},
			{1, []int{4}},
			{2, []int{1, 4}},
			{3, []int{1, 3, 4}},
			{99, []int{0, 1, 2, 3, 4}},
		}
		for _, c := range cases {
			got, err := QuestSelectPages(q, mins, maxs, c.budget)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, c.want) {
				t.Fatalf("budget %d: got %v want %v", c.budget, got, c.want)
			}
		}
		if got, err := QuestSelectPages(q, nil, nil, 3); err != nil || got != nil {
			t.Fatalf("empty: %v %v", got, err)
		}
	})

	t.Run("dimension_mismatch_errors", func(t *testing.T) {
		if _, err := QuestCriticality([]float64{1, 2}, []float64{0}, []float64{1, 1}); !errors.Is(err, ErrQuestDim) {
			t.Fatalf("want ErrQuestDim, got %v", err)
		}
		if _, err := QuestRankPages([]float64{1}, [][]float64{{0}}, nil); !errors.Is(err, ErrQuestDim) {
			t.Fatalf("want ErrQuestDim rows, got %v", err)
		}
		if _, err := QuestSelectPages([]float64{1}, [][]float64{{0}, {0, 0}}, [][]float64{{1}, {1, 1}}, 2); !errors.Is(err, ErrQuestDim) {
			t.Fatalf("want ErrQuestDim on last page, got %v", err)
		}
	})

	t.Run("page_bounds_edges", func(t *testing.T) {
		if _, _, ok := QuestPageBounds(nil); ok {
			t.Fatal("empty page must not be ok")
		}
		if _, _, ok := QuestPageBounds([][]float64{{1, 2}, {3}}); ok {
			t.Fatal("ragged page must not be ok")
		}
		kmin, kmax, ok := QuestPageBounds([][]float64{{1, -2}, {-3, 4}})
		if !ok || !reflect.DeepEqual(kmin, []float64{-3, -2}) || !reflect.DeepEqual(kmax, []float64{1, 4}) {
			t.Fatalf("bounds %v %v %v", kmin, kmax, ok)
		}
	})
}

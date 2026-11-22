package kv

import (
	"math"
	"math/rand"
	"slices"
	"testing"
)

func questKeys(rng *rand.Rand, tokens, heads, dim int) []float32 {
	out := make([]float32, tokens*heads*dim)
	for i := range out {
		out[i] = float32(rng.NormFloat64() * 4)
	}
	return out
}

func assertQuestMetaEqual(t *testing.T, got, want *QuestPageMinMax) {
	t.Helper()
	if got.SeqLen() != want.SeqLen() || got.NumPages() != want.NumPages() || got.LastPageLen() != want.LastPageLen() {
		t.Fatalf("shape: got seq=%d pages=%d last=%d, want seq=%d pages=%d last=%d",
			got.SeqLen(), got.NumPages(), got.LastPageLen(), want.SeqLen(), want.NumPages(), want.LastPageLen())
	}
	for p := 0; p < want.NumPages(); p++ {
		for h := 0; h < want.numKVHeads; h++ {
			gmax, _ := got.PageMax(p, h)
			wmax, _ := want.PageMax(p, h)
			gmin, _ := got.PageMin(p, h)
			wmin, _ := want.PageMin(p, h)
			if !slices.Equal(gmax, wmax) || !slices.Equal(gmin, wmin) {
				t.Fatalf("page %d head %d: got max=%v min=%v, want max=%v min=%v", p, h, gmax, gmin, wmax, wmin)
			}
		}
	}
}

func TestQuestPageMinMaxSidecar(t *testing.T) {
	t.Run("hand_computed_partial_last_page", func(t *testing.T) {
		m, err := NewQuestPageMinMax(2, 1, 2)
		if err != nil {
			t.Fatal(err)
		}
		for _, k := range [][]float32{{1, -1}, {3, -5}, {-2, 7}} {
			if err := m.AppendToken(k); err != nil {
				t.Fatal(err)
			}
		}
		if m.SeqLen() != 3 || m.NumPages() != 2 || m.LastPageLen() != 1 {
			t.Fatalf("seq=%d pages=%d last=%d", m.SeqLen(), m.NumPages(), m.LastPageLen())
		}
		want := map[int][2][]float32{0: {{3, -1}, {1, -5}}, 1: {{-2, 7}, {-2, 7}}}
		for p, w := range want {
			gmax, _ := m.PageMax(p, 0)
			gmin, _ := m.PageMin(p, 0)
			if !slices.Equal(gmax, w[0]) || !slices.Equal(gmin, w[1]) {
				t.Fatalf("page %d: max=%v min=%v, want %v %v", p, gmax, gmin, w[0], w[1])
			}
		}
	})

	t.Run("rollover_page_does_not_inherit_previous_bounds", func(t *testing.T) {
		m, _ := NewQuestPageMinMax(2, 1, 1)
		_ = m.Append([]float32{100, -100, 0.5})
		gmax, _ := m.PageMax(1, 0)
		gmin, _ := m.PageMin(1, 0)
		if gmax[0] != 0.5 || gmin[0] != 0.5 {
			t.Fatalf("new page leaked prior bounds: max=%v min=%v", gmax, gmin)
		}
	})

	t.Run("incremental_equals_recompute_across_chunkings", func(t *testing.T) {
		const heads, dim = 3, 5
		for _, pageSize := range []int{1, 4, 7, 16} {
			rng := rand.New(rand.NewSource(int64(pageSize)))
			keys := questKeys(rng, 53, heads, dim)
			stride := heads * dim
			m, _ := NewQuestPageMinMax(pageSize, heads, dim)
			for pos := 0; pos < 53; {
				n := 1 + rng.Intn(9) // mix of decode steps and prefill chunks
				if pos+n > 53 {
					n = 53 - pos
				}
				if err := m.Append(keys[pos*stride : (pos+n)*stride]); err != nil {
					t.Fatal(err)
				}
				pos += n
				ref, err := RecomputeQuestPageMinMax(keys[:pos*stride], pageSize, heads, dim)
				if err != nil {
					t.Fatal(err)
				}
				assertQuestMetaEqual(t, m, ref)
			}
		}
	})

	t.Run("bounds_contain_every_key_and_are_attained", func(t *testing.T) {
		rng := rand.New(rand.NewSource(7))
		const pageSize, heads, dim, tokens = 4, 2, 3, 10
		keys := questKeys(rng, tokens, heads, dim)
		m, _ := RecomputeQuestPageMinMax(keys, pageSize, heads, dim)
		for p := 0; p < m.NumPages(); p++ {
			for h := 0; h < heads; h++ {
				pmax, _ := m.PageMax(p, h)
				pmin, _ := m.PageMin(p, h)
				for c := 0; c < dim; c++ {
					hitMax, hitMin := false, false
					for tok := p * pageSize; tok < min((p+1)*pageSize, tokens); tok++ {
						v := keys[(tok*heads+h)*dim+c]
						if v > pmax[c] || v < pmin[c] {
							t.Fatalf("key %v outside [%v,%v]", v, pmin[c], pmax[c])
						}
						hitMax = hitMax || v == pmax[c]
						hitMin = hitMin || v == pmin[c]
					}
					if !hitMax || !hitMin {
						t.Fatalf("page %d head %d ch %d bound not attained", p, h, c)
					}
				}
			}
		}
	})

	t.Run("nan_key_is_ignored_like_cuda_fmax", func(t *testing.T) {
		m, _ := NewQuestPageMinMax(4, 1, 1)
		nan := float32(math.NaN())
		_ = m.Append([]float32{2, nan, -3})
		gmax, _ := m.PageMax(0, 0)
		gmin, _ := m.PageMin(0, 0)
		if gmax[0] != 2 || gmin[0] != -3 {
			t.Fatalf("max=%v min=%v", gmax, gmin)
		}
	})

	t.Run("invalid_inputs_refused_and_state_unchanged", func(t *testing.T) {
		for _, dims := range [][3]int{{0, 1, 1}, {1, 0, 1}, {1, 1, 0}, {-1, 1, 1}} {
			if _, err := NewQuestPageMinMax(dims[0], dims[1], dims[2]); err == nil {
				t.Fatalf("dims %v accepted", dims)
			}
		}
		m, _ := NewQuestPageMinMax(2, 2, 2)
		if err := m.Append([]float32{1, 2, 3}); err == nil {
			t.Fatal("ragged key accepted")
		}
		if err := m.Append(nil); err != nil || m.SeqLen() != 0 || m.NumPages() != 0 || m.LastPageLen() != 0 {
			t.Fatalf("empty append changed state: err=%v seq=%d", err, m.SeqLen())
		}
		if _, err := m.PageMax(0, 0); err == nil {
			t.Fatal("read of absent page accepted")
		}
		_ = m.Append([]float32{1, 2, 3, 4})
		if _, err := m.PageMin(0, 2); err == nil {
			t.Fatal("out-of-range head accepted")
		}
		if _, err := m.PageMin(-1, 0); err == nil {
			t.Fatal("negative page accepted")
		}
		if _, err := RecomputeQuestPageMinMax([]float32{1}, 2, 2, 2); err == nil {
			t.Fatal("recompute accepted ragged keys")
		}
		got, _ := m.PageMax(0, 1)
		got[0] = 99
		again, _ := m.PageMax(0, 1)
		if again[0] != 3 {
			t.Fatalf("PageMax returned aliased storage: %v", again)
		}
	})
}

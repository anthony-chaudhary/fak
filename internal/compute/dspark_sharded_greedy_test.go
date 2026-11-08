package compute

import (
	"errors"
	"math"
	"testing"
)

// fak-test:runtime fast est=10ms lane=default

// dsparkUnfusedReference materializes sums[i] = baseLocal[i] + bias[i] and
// runs a plain argmax scan with the borrowed closed semantics (any NaN ->
// smallest NaN global index; ties -> smallest global index). It is the
// unfused build-step + argmax pipeline the fused path must match without
// ever materializing the sum vector.
func dsparkUnfusedReference(bias, baseLocal []float32, vocabStart int) int {
	var (
		hasNaN bool
		nanIdx int
		best   float32
	)
	argIdx := -1 // -1 means "no live entry yet"; 0 is a real index
	for i := 0; i < len(bias); i++ {
		v := baseLocal[i] + bias[i]
		if v != v {
			if !hasNaN {
				hasNaN = true
				nanIdx = i
			}
			continue
		}
		if argIdx < 0 || v > best { // ties keep the earlier index
			best = v
			argIdx = i
		}
	}
	if hasNaN {
		return nanIdx + vocabStart
	}
	return argIdx + vocabStart
}

// dsparkLCG fills dst with deterministic float32 values in [lo, hi) from a
// fixed-seed 32-bit LCG, so every matrix below is reproducible byte-for-byte.
func dsparkLCG(dst []float32, seed uint32, lo, hi float32) {
	state := seed
	for i := range dst {
		state = state*1664525 + 1013904223
		f := float32(state>>8) / float32(1<<24) // [0,1)
		dst[i] = lo + f*(hi-lo)
	}
}

func TestDSparkShardedGreedyStepSemantics(t *testing.T) {
	cases := []struct {
		name       string
		bias, base []float32
		vocabStart int
		want       int
	}{
		{"plain max", []float32{0, 0, 0}, []float32{1, 5, 3}, 0, 1},
		{"tie keeps lowest index", []float32{0, 0, 0}, []float32{2, 2, 9 - 7}, 0, 0},
		{"vocab start offset", []float32{0, 0, 0}, []float32{1, 5, 3}, 100, 101},
		{"negative values", []float32{-1, -1, -1}, []float32{-9, -4, -7}, 0, 1},
		{"neg inf values still compete", []float32{0, 0}, []float32{dsparkNegInf, dsparkNegInf}, 0, 0},
		{"nan dominates larger finite", []float32{0, 0, 0}, []float32{1, dsparkNaN, 9}, 0, 1},
		{"lowest nan index wins", []float32{dsparkNaN, 0, dsparkNaN}, []float32{9, 1, 7}, 0, 0},
		{"all nan", []float32{dsparkNaN, dsparkNaN}, []float32{0, 0}, 5, 5},
		{"single entry", []float32{3}, []float32{-2}, 7, 7},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := DSparkShardedGreedyStep(tc.bias, tc.base, tc.vocabStart)
			if err != nil {
				t.Fatalf("DSparkShardedGreedyStep: %v", err)
			}
			if got != tc.want {
				t.Fatalf("got %d, want %d", got, tc.want)
			}
			if ref := dsparkUnfusedReference(tc.bias, tc.base, tc.vocabStart); got != ref {
				t.Fatalf("fused %d diverges from unfused reference %d", got, ref)
			}
		})
	}
}

func TestDSparkShardedGreedyStepBlockBoundaries(t *testing.T) {
	const w = DSparkShardedGreedyBlock + 1 // 4097: spans two partial blocks

	t.Run("tie across block boundary keeps lower global index", func(t *testing.T) {
		base := make([]float32, w)
		bias := make([]float32, w)
		for i := range base {
			base[i] = -1
		}
		base[DSparkShardedGreedyBlock-1] = 5 // end of block 0
		base[DSparkShardedGreedyBlock] = 5   // start of block 1
		got, err := DSparkShardedGreedyStep(bias, base, 0)
		if err != nil {
			t.Fatal(err)
		}
		if got != DSparkShardedGreedyBlock-1 {
			t.Fatalf("got %d, want %d", got, DSparkShardedGreedyBlock-1)
		}
	})

	t.Run("winner in final partial block", func(t *testing.T) {
		base := make([]float32, w)
		bias := make([]float32, w)
		for i := range base {
			base[i] = -1
		}
		base[w-1] = 99
		got, err := DSparkShardedGreedyStep(bias, base, 40)
		if err != nil {
			t.Fatal(err)
		}
		if want := w - 1 + 40; got != want {
			t.Fatalf("got %d, want %d", got, want)
		}
	})

	t.Run("base columns beyond bias width are masked", func(t *testing.T) {
		base := make([]float32, 2*DSparkShardedGreedyBlock) // padded partition
		bias := make([]float32, DSparkShardedGreedyBlock)
		for i := range base {
			base[i] = -1
		}
		base[0] = 1
		base[DSparkShardedGreedyBlock] = 1e30 // beyond len(bias): must not win
		got, err := DSparkShardedGreedyStep(bias, base, 0)
		if err != nil {
			t.Fatal(err)
		}
		if got != 0 {
			t.Fatalf("got %d, want 0 (masked column leaked)", got)
		}
	})
}

func TestDSparkShardedGreedyPartialsPaddedShape(t *testing.T) {
	base := make([]float32, 2*DSparkShardedGreedyBlock)
	bias := make([]float32, DSparkShardedGreedyBlock)
	for i := range base {
		base[i] = -1
	}
	base[3] = 7
	partials, err := DSparkShardedGreedyPartials(bias, base, 1000)
	if err != nil {
		t.Fatal(err)
	}
	if len(partials) != 2 {
		t.Fatalf("partials = %d, want 2 (padded partition width fixes the shape)", len(partials))
	}
	if partials[0].Index != 1003 || partials[0].Value != 7 {
		t.Fatalf("partial[0] = %+v, want {Value:7 Index:1003}", partials[0])
	}
	if partials[1].Index != DSparkShardedGreedyInvalidIndex {
		t.Fatalf("partial[1].Index = %d, want invalid sentinel", partials[1].Index)
	}
	if partials[1].Value != dsparkNegInf {
		t.Fatalf("partial[1].Value = %v, want -Inf", partials[1].Value)
	}
}

func TestDSparkShardedGreedyStepFixedMatrixVsUnfused(t *testing.T) {
	const w = 9000 // three partial blocks: 4096 + 4096 + 808
	base := make([]float32, w)
	bias := make([]float32, w)
	dsparkLCG(base, 0x9E3779B9, -10, 10)
	dsparkLCG(bias, 0xC2B2AE35, -1, 1)

	t.Run("no-nan matrix matches unfused reference", func(t *testing.T) {
		got, err := DSparkShardedGreedyStep(bias, base, 0)
		if err != nil {
			t.Fatal(err)
		}
		if want := dsparkUnfusedReference(bias, base, 0); got != want {
			t.Fatalf("fused %d diverges from unfused reference %d", got, want)
		}
	})

	t.Run("injected nan pins lowest nan index", func(t *testing.T) {
		b := append([]float32{}, base...)
		b[1234] = dsparkNaN
		got, err := DSparkShardedGreedyStep(bias, b, 0)
		if err != nil {
			t.Fatal(err)
		}
		if got != 1234 {
			t.Fatalf("got %d, want 1234", got)
		}
	})

	t.Run("injected extreme max in block 1", func(t *testing.T) {
		b := append([]float32{}, base...)
		b[DSparkShardedGreedyBlock] = 1e30
		got, err := DSparkShardedGreedyStep(bias, b, 500)
		if err != nil {
			t.Fatal(err)
		}
		if want := DSparkShardedGreedyBlock + 500; got != want {
			t.Fatalf("got %d, want %d", got, want)
		}
	})

	t.Run("exact tie pinned far apart", func(t *testing.T) {
		b := append([]float32{}, base...)
		biasCopy := append([]float32{}, bias...)
		const target = 1e9 // far outside the LCG range: no accidental third tie
		i, j := 77, 8000
		biasCopy[i], biasCopy[j] = 0, 0
		b[i], b[j] = target, target
		got, err := DSparkShardedGreedyStep(biasCopy, b, 0)
		if err != nil {
			t.Fatal(err)
		}
		if got != i {
			t.Fatalf("got %d, want %d (lower index of exact tie)", got, i)
		}
	})
}

func TestDSparkShardedGreedySentinels(t *testing.T) {
	ok := []float32{1, 2, 3}

	t.Run("empty bias", func(t *testing.T) {
		if _, err := DSparkShardedGreedyStep(nil, ok, 0); !errors.Is(err, ErrDSparkShardedGreedyEmpty) {
			t.Fatalf("err = %v, want ErrDSparkShardedGreedyEmpty", err)
		}
	})
	t.Run("empty base", func(t *testing.T) {
		if _, err := DSparkShardedGreedyStep(ok, nil, 0); !errors.Is(err, ErrDSparkShardedGreedyEmpty) {
			t.Fatalf("err = %v, want ErrDSparkShardedGreedyEmpty", err)
		}
	})
	t.Run("bias wider than base", func(t *testing.T) {
		if _, err := DSparkShardedGreedyStep([]float32{1, 2, 3, 4, 5}, ok, 0); !errors.Is(err, ErrDSparkShardedGreedyShapeMismatch) {
			t.Fatalf("err = %v, want ErrDSparkShardedGreedyShapeMismatch", err)
		}
	})
	t.Run("negative vocab start", func(t *testing.T) {
		if _, err := DSparkShardedGreedyStep(ok, ok, -1); !errors.Is(err, ErrDSparkShardedGreedyVocabStart) {
			t.Fatalf("err = %v, want ErrDSparkShardedGreedyVocabStart", err)
		}
	})
	t.Run("global index overflows int32", func(t *testing.T) {
		if _, err := DSparkShardedGreedyStep(ok, ok, math.MaxInt32-1); !errors.Is(err, ErrDSparkShardedGreedyIndexRange) {
			t.Fatalf("err = %v, want ErrDSparkShardedGreedyIndexRange", err)
		}
	})
	t.Run("finish with only invalid partials", func(t *testing.T) {
		_, err := DSparkShardedGreedyFinish([]DSparkShardedGreedyPartial{
			{Value: dsparkNegInf, Index: DSparkShardedGreedyInvalidIndex},
		})
		if !errors.Is(err, ErrDSparkShardedGreedyNoWinner) {
			t.Fatalf("err = %v, want ErrDSparkShardedGreedyNoWinner", err)
		}
		if _, err := DSparkShardedGreedyFinish(nil); !errors.Is(err, ErrDSparkShardedGreedyNoWinner) {
			t.Fatalf("err = %v, want ErrDSparkShardedGreedyNoWinner", err)
		}
	})
}

// fak-test:runtime fast est=1ms
func TestDSparkShardedGreedyFinishRejectsNegativeIndex(t *testing.T) {
	t.Parallel()
	for _, value := range []float32{1, float32(math.NaN())} {
		for _, index := range []int32{-1, math.MinInt32} {
			_, err := DSparkShardedGreedyFinish([]DSparkShardedGreedyPartial{
				{Value: 2, Index: 0},
				{Value: value, Index: index},
			})
			if !errors.Is(err, ErrDSparkShardedGreedyIndexRange) {
				t.Fatalf("value=%v index=%d: got %v, want index-range refusal", value, index, err)
			}
		}
	}
	// The declared masked sentinel remains ignorable even with a NaN value;
	// zero is a valid token index and must still win the remaining row.
	got, err := DSparkShardedGreedyFinish([]DSparkShardedGreedyPartial{
		{Value: float32(math.NaN()), Index: DSparkShardedGreedyInvalidIndex},
		{Value: 2, Index: 0},
	})
	if err != nil || got != 0 {
		t.Fatalf("masked sentinel plus token zero: got (%d, %v)", got, err)
	}
}

package agent

import (
	"context"
	"errors"
	"math"
	"math/rand"
	"sort"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// inkernel_sampling_test.go is the #13873 witness: the in-kernel sampler must be
// allocation-free across tokens once its per-session workspace is warm, and the
// partial-selection rewrite must draw the identical token as the un-workspaced path
// for a fixed seed across every option combination.

const samplingWitnessVocab = 129000 // DeepSeek V4.1-class LM-head width.

// sampleOptionsCase is one (temp, topP, topK) combination exercised by the parity
// witness. The set spans the full grid: pure argmax, full softmax (both cutoffs off),
// top-k only, top-p only, and the composed top-k -> top-p pipeline.
type sampleOptionsCase struct {
	name string
	temp float64
	topP float64
	topK int
}

// fak-test:runtime fast est=5ms lane=default
func TestWithSeedCopiesAndContextPreservesPresence(t *testing.T) {
	seed := int64(-7)
	var params SampleParams
	WithSeed(&seed)(&params)
	seed = 99
	if params.Seed == nil || *params.Seed != -7 {
		t.Fatalf("WithSeed did not copy input: got %v", params.Seed)
	}
	WithSeed(nil)(&params)
	if params.Seed == nil || *params.Seed != -7 {
		t.Fatalf("WithSeed(nil) changed existing seed: got %v", params.Seed)
	}

	for _, want := range []int64{0, -1} {
		ctx := withNativeTargetSampleSeed(context.Background(), &want)
		got, ok := nativeTargetSampleSeedFromContext(ctx)
		if !ok || got != uint64(want) {
			t.Fatalf("seed %d context = (%d,%v), want (%d,true)", want, got, ok, uint64(want))
		}
	}
	base := context.Background()
	if got := withNativeTargetSampleSeed(base, nil); got != base {
		t.Fatal("nil seed replaced request context")
	}
	if _, ok := nativeTargetSampleSeedFromContext(base); ok {
		t.Fatal("seed unexpectedly present in unseeded context")
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestInkernelExplicitCoinPreservesLegacyRNGState(t *testing.T) {
	logits := []float32{-0.4, 1.2, 0.7, 0.2, -1.0}
	bias := model.LogitBias{0: 0.6, 4: 1.1}
	counts := []int32{0, 2, 0, 1, 0}
	const seed = int64(9137)

	legacyRNG := rand.New(rand.NewSource(seed))
	coinRNG := rand.New(rand.NewSource(seed))
	want := sampleLogitsWithPenaltyWS(logits, 0.8, 0.9, 4, bias, 0.15, 0.25, counts, legacyRNG, &samplerWorkspace{})
	coin := coinRNG.Float64()
	got := sampleLogitsWithPenaltyCoinWS(logits, 0.8, 0.9, 4, bias, 0.15, 0.25, counts, coin, &samplerWorkspace{})
	if want != 2 {
		t.Fatalf("legacy golden token = %d, want 2", want)
	}
	if got != want {
		t.Fatalf("explicit coin selected %d, legacy selected %d", got, want)
	}
	gotState, wantState := legacyRNG.Uint64(), coinRNG.Uint64()
	if wantState != 0x8509b013fc96c7b7 {
		t.Fatalf("legacy golden next RNG = %#x, want %#x", wantState, uint64(0x8509b013fc96c7b7))
	}
	if gotState != wantState {
		t.Fatalf("legacy RNG state changed: got %#x want %#x", gotState, wantState)
	}
}

func sampleOptionsCases() []sampleOptionsCase {
	return []sampleOptionsCase{
		{"argmax", 0, 0, 0},
		{"full_softmax", 1.0, 0, 0},
		{"full_softmax_topP1", 1.0, 1, 0},
		{"topk_small", 1.0, 0, 50},
		{"topk_boundary", 1.0, 0, 64},
		{"topk_large", 1.0, 0, 100},
		{"topp", 1.0, 0.9, 0},
		{"topp_tight", 0.7, 0.5, 0},
		{"composed", 1.0, 0.9, 50},
		{"composed_large_k", 0.8, 0.95, 100},
	}
}

func samplingWitnessLogits(vocab int, seed int64) []float32 {
	rng := rand.New(rand.NewSource(seed))
	logits := make([]float32, vocab)
	for i := range logits {
		logits[i] = float32(rng.NormFloat64() * 2.0)
	}
	return logits
}

// TestInkernelSampleWorkspaceZeroAllocs proves the reusable workspace removes every
// per-token allocation: after one warm-up call sizes its buffers, subsequent samples
// at the V4.1 vocabulary width allocate nothing.
func TestInkernelSampleWorkspaceZeroAllocs(t *testing.T) {
	logits := samplingWitnessLogits(samplingWitnessVocab, 12345)
	var ws samplerWorkspace
	// Warm-up: the first draw sizes probs/order/eff, so the measured runs are steady state.
	drain := rand.New(rand.NewSource(1))
	_ = sampleLogitsWithPenaltyWS(logits, 1.0, 0.9, 50, nil, 0, 0, nil, drain, &ws)

	for _, tc := range sampleOptionsCases() {
		t.Run(tc.name, func(t *testing.T) {
			var counts []int32
			if tc.name == "composed" {
				counts = make([]int32, len(logits))
				counts[7] = 3
			}
			allocs := testing.AllocsPerRun(50, func() {
				_ = sampleLogitsWithPenaltyWS(logits, tc.temp, tc.topP, tc.topK, nil, 0, 0, counts, drain, &ws)
			})
			if allocs > 0 {
				t.Fatalf("sampleLogitsWithPenaltyWS(%s) allocated %v per call, want 0 after warm-up", tc.name, allocs)
			}
		})
	}
}

// TestInkernelSampleWorkspaceSeededParity proves the workspaced path draws the exact
// same token as the historical per-call path for a fixed seed over every option
// combination — the semantic contract the rewrite must not change.
func TestInkernelSampleWorkspaceSeededParity(t *testing.T) {
	logits := samplingWitnessLogits(4096, 999)
	for _, tc := range sampleOptionsCases() {
		t.Run(tc.name, func(t *testing.T) {
			var ws samplerWorkspace
			for seed := int64(0); seed < 200; seed++ {
				want := sampleLogitsWithPenalty(logits, tc.temp, tc.topP, tc.topK, nil, 0, 0, nil, rand.New(rand.NewSource(seed)))
				got := sampleLogitsWithPenaltyWS(logits, tc.temp, tc.topP, tc.topK, nil, 0, 0, nil, rand.New(rand.NewSource(seed)), &ws)
				if got != want {
					t.Fatalf("%s seed %d: workspaced draw %d != reference %d", tc.name, seed, got, want)
				}
			}
		})
	}
}

// TestInkernelSampleWorkspaceGrowsAcrossVocabularies proves a workspace sized for a
// narrow vocabulary keeps parity when a later, wider call reuses it — the served lane
// holds one workspace for a session whose widest LM head may arrive after warm-up.
func TestInkernelSampleWorkspaceGrowsAcrossVocabularies(t *testing.T) {
	var ws samplerWorkspace
	for _, vocab := range []int{64, 4096, samplingWitnessVocab} {
		logits := samplingWitnessLogits(vocab, int64(vocab))
		for seed := int64(0); seed < 40; seed++ {
			want := sampleLogits(logits, 1.0, 0.9, 50, rand.New(rand.NewSource(seed)))
			got := sampleLogitsWS(logits, 1.0, 0.9, 50, rand.New(rand.NewSource(seed)), &ws)
			if got != want {
				t.Fatalf("vocab %d seed %d: got %d want %d", vocab, seed, got, want)
			}
		}
	}
}

// BenchmarkInkernelSample is the issue's named witness: the sampler at production
// vocabulary width with allocation reporting, so the ns/op delta from the
// partial-selection rewrite is recorded alongside the zero-alloc steady state.
func BenchmarkInkernelSample(b *testing.B) {
	logits := samplingWitnessLogits(samplingWitnessVocab, 12345)
	var ws samplerWorkspace
	rng := rand.New(rand.NewSource(42))
	_ = sampleLogitsWithPenaltyWS(logits, 1.0, 0.9, 50, nil, 0, 0, nil, rng, &ws)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = sampleLogitsWithPenaltyWS(logits, 1.0, 0.9, 50, nil, 0, 0, nil, rng, &ws)
	}
}

// BenchmarkInkernelSampleOneShot is the pre-#13873 shape (a fresh workspace per call)
// kept as the named baseline for the recorded delta.
func BenchmarkInkernelSampleOneShot(b *testing.B) {
	logits := samplingWitnessLogits(samplingWitnessVocab, 12345)
	rng := rand.New(rand.NewSource(42))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = sampleLogitsWithPenalty(logits, 1.0, 0.9, 50, nil, 0, 0, nil, rng)
	}
}

// rankedAll is the full-sort oracle: every position ordered by probability
// descending, ties by ascending input position — the closed contract topKRankOrder
// must reproduce as a prefix.
func rankedAll(probs []float64) []int {
	idx := make([]int, len(probs))
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(a, b int) bool {
		if probs[idx[a]] != probs[idx[b]] {
			return probs[idx[a]] > probs[idx[b]]
		}
		return idx[a] < idx[b]
	})
	return idx
}

func sameInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// fak-test:runtime fast est=5ms lane=default
// TestTopKBoundedSelectionOrder pins the borrowed bounded top-k selection
// contract: the O(k)-storage selector returns exactly the top-k of a full sort
// (value-desc, id-asc), stays tie-stable across repeated calls, agrees with the
// existing topKTruncate seam on the kept set and surviving mass, allocates only
// the k indices, and rejects any non-finite row with the package sentinel.
func TestTopKBoundedSelectionOrder(t *testing.T) {
	// Fixed row with deliberate ties across ranks: 0.30 x2, 0.25 x2, 0.10 x2, 0.05 x2.
	probs := []float64{0.05, 0.30, 0.10, 0.30, 0.00, 0.25, 0.05, 0.40, 0.25, 0.10}
	full := rankedAll(probs)

	t.Run("prefix-equality-small", func(t *testing.T) {
		for _, k := range []int{1, 2, 3, 5, 8, 10} {
			got, err := topKRankOrder(probs, k)
			if err != nil {
				t.Fatalf("topKRankOrder(k=%d): unexpected error %v", k, err)
			}
			if want := full[:k]; !sameInts(got, want) {
				t.Fatalf("k=%d: rank order %v, want full-sort prefix %v", k, got, want)
			}
		}
	})

	// 200-position row with heavy structured ties ((i*37)%50 cycles all 50 values
	// four times each) exercises the shared sweep well past the small-k threshold.
	big := make([]float64, 200)
	for i := range big {
		big[i] = float64((i*37)%50) / 100
	}
	bigFull := rankedAll(big)

	t.Run("prefix-equality-large", func(t *testing.T) {
		for _, k := range []int{65, 100, 199} {
			got, err := topKRankOrder(big, k)
			if err != nil {
				t.Fatalf("topKRankOrder(k=%d): unexpected error %v", k, err)
			}
			if want := bigFull[:k]; !sameInts(got, want) {
				t.Fatalf("k=%d: rank order diverges from full-sort prefix at first diff", k)
			}
		}
	})

	t.Run("tie-stability", func(t *testing.T) {
		// All-equal row: rank order must be pure input position, repeat for repeat.
		eq := make([]float64, 80)
		for i := range eq {
			eq[i] = 0.5
		}
		first, err := topKRankOrder(eq, 80)
		if err != nil {
			t.Fatalf("topKRankOrder(equal row): unexpected error %v", err)
		}
		for i := range first {
			if first[i] != i {
				t.Fatalf("equal row position %d = %d, want %d", i, first[i], i)
			}
		}
		second, err := topKRankOrder(eq, 80)
		if err != nil || !sameInts(first, second) {
			t.Fatalf("repeated selection not stable: %v vs %v (err=%v)", first, second, err)
		}
	})

	t.Run("seam-agreement", func(t *testing.T) {
		// The existing truncate seam must keep exactly the rank-ordered set, and its
		// surviving mass must equal the kept values summed in ascending-index order.
		const k = 100
		kept, err := topKRankOrder(big, k)
		if err != nil {
			t.Fatalf("topKRankOrder: unexpected error %v", err)
		}
		row := append([]float64(nil), big...)
		var total float64
		for _, p := range row {
			total += p
		}
		sum := (&samplerWorkspace{}).topKTruncate(row, total, k)
		wantSet := make([]bool, len(row))
		for _, idx := range kept {
			wantSet[idx] = true
		}
		for i := range row {
			if (row[i] != 0) != wantSet[i] {
				t.Fatalf("position %d survivor=%v, rank-order kept=%v", i, row[i] != 0, wantSet[i])
			}
		}
		order := append([]int(nil), kept...)
		sort.Ints(order)
		var wantSum float64
		for _, idx := range order {
			wantSum += big[idx]
		}
		if sum != wantSum {
			t.Fatalf("surviving mass %v, want kept-set sum %v", sum, wantSum)
		}
	})

	t.Run("allocation-contract", func(t *testing.T) {
		// The borrowed storage bound: exactly one allocation, the k-index slice.
		allocs := testing.AllocsPerRun(10, func() {
			_, _ = topKRankOrder(big, 100)
		})
		if allocs != 1 {
			t.Fatalf("topKRankOrder allocated %v per call, want 1 (the k-index slice)", allocs)
		}
	})

	t.Run("non-finite-row-rejects", func(t *testing.T) {
		for name, bad := range map[string]float64{"nan": math.NaN(), "+inf": math.Inf(1), "-inf": math.Inf(-1)} {
			row := append([]float64(nil), big...)
			row[3] = bad
			got, err := topKRankOrder(row, 10)
			if !errors.Is(err, ErrNonFiniteLogits) {
				t.Fatalf("%s row: error = %v, want ErrNonFiniteLogits", name, err)
			}
			if got != nil {
				t.Fatalf("%s row: partial selection %v returned, want none", name, got)
			}
		}
	})

	t.Run("k-clamping", func(t *testing.T) {
		// k beyond the row returns the whole row in rank order; k<=0 selects nothing.
		all, err := topKRankOrder(probs, 100)
		if err != nil {
			t.Fatalf("topKRankOrder(k>len): unexpected error %v", err)
		}
		if !sameInts(all, full) {
			t.Fatalf("k>len: %v, want full rank order %v", all, full)
		}
		if got, err := topKRankOrder(probs, 0); err != nil || got != nil {
			t.Fatalf("k=0: (%v, %v), want (nil, nil)", got, err)
		}
		if got, err := topKRankOrder(nil, 5); err != nil || got != nil {
			t.Fatalf("empty row: (%v, %v), want (nil, nil)", got, err)
		}
	})
}

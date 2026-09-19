package model

import (
	"errors"
	"math"
	"math/rand"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func TestV4ScoredRouteMatchesIndependentOracle(t *testing.T) {
	logits := []float32{-2, -0.5, 0, 1, 3, 0.25, -4, 2}
	bias := []float32{0, 0, 0, -4, 0, 3, 0, 0}
	got, err := v4ScoredRoute(logits, bias, 3, 2.5)
	if err != nil {
		t.Fatal(err)
	}

	// Independent scalar transcription of pinned Gate.forward. It deliberately
	// does not call any production scoring, sorting, or normalization helper.
	score := func(z float32) float64 { return math.Sqrt(math.Log1p(math.Exp(float64(z)))) }
	// Biased selection ranks experts 5, 4, 7. The gathered weights remain
	// unbiased, which is the important noaux_tc contract.
	wantExperts := []int{5, 4, 7}
	denom := score(logits[5]) + score(logits[4]) + score(logits[7])
	for i, expert := range wantExperts {
		if got[i].expert != expert {
			t.Fatalf("pick %d expert=%d, want %d (all=%+v)", i, got[i].expert, expert, got)
		}
		wantWeight := float32(score(logits[expert]) / denom * 2.5)
		if diff := math.Abs(float64(got[i].weight - wantWeight)); diff > 2e-6 {
			t.Fatalf("expert %d weight=%g want=%g diff=%g", expert, got[i].weight, wantWeight, diff)
		}
	}
}

func TestV4ScoredRouteTieAndExtremeFiniteInputs(t *testing.T) {
	got, err := v4ScoredRoute([]float32{1000, -100, 0, 0}, nil, 3, 2.5)
	if err != nil {
		t.Fatal(err)
	}
	want := []int{0, 2, 3} // equal scores select lower expert index first
	for i := range want {
		if got[i].expert != want[i] {
			t.Fatalf("experts=%+v, want order %v", got, want)
		}
		if math.IsNaN(float64(got[i].weight)) || math.IsInf(float64(got[i].weight), 0) {
			t.Fatalf("non-finite weight: %+v", got)
		}
	}
	var sum float32
	for _, pick := range got {
		sum += pick.weight
	}
	if math.Abs(float64(sum-2.5)) > 2e-6 {
		t.Fatalf("scaled normalized sum=%g, want 2.5", sum)
	}
}

func TestV4ScoredRouteFailsClosed(t *testing.T) {
	tests := []struct {
		name   string
		logits []float32
		bias   []float32
		k      int
		scale  float32
	}{
		{"empty", nil, nil, 1, 2.5},
		{"bias width", []float32{1, 2}, []float32{1}, 1, 2.5},
		{"zero topk", []float32{1}, nil, 0, 2.5},
		{"wide topk", []float32{1}, nil, 2, 2.5},
		{"nan logit", []float32{float32(math.NaN())}, nil, 1, 2.5},
		{"inf logit", []float32{float32(math.Inf(1))}, nil, 1, 2.5},
		{"nan bias", []float32{1}, []float32{float32(math.NaN())}, 1, 2.5},
		{"zero scale", []float32{1}, nil, 1, 0},
		{"inf scale", []float32{1}, nil, 1, float32(math.Inf(1))},
		{"zero normalization", []float32{-math.MaxFloat32}, nil, 1, 2.5},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := v4ScoredRoute(tt.logits, tt.bias, tt.k, tt.scale)
			if err == nil {
				t.Fatal("expected fail-closed error")
			}
			var typed *v4RouteError
			if !errors.As(err, &typed) {
				t.Fatalf("error %T is not *v4RouteError: %v", err, err)
			}
		})
	}
}

func TestV4HashRouteIsExplicitlyUnsupported(t *testing.T) {
	err := v4HashRouteUnsupported(2)
	var typed *v4RouteError
	if !errors.As(err, &typed) || typed.Field != "hash_layer" {
		t.Fatalf("got %T %v, want typed hash-layer refusal", err, err)
	}
}

// v4NextAfter32 returns the next representable float32 above v, used to build
// near-ties exactly one ULP apart.
func v4NextAfter32(v float32) float32 {
	return math.Nextafter32(v, float32(math.Inf(1)))
}

// TestV4BitonicTopKMatchesStableSort is the load-bearing equivalence witness:
// over adversarial fixtures the opt-in partial-selection kernel must return the
// exact same expert-index order as the reference stable sort, flag OFF vs ON.
func TestV4BitonicTopKMatchesStableSort(t *testing.T) {
	// want pins the expected top-6 selection order INDEPENDENT of the reference
	// sort: descending value, lower expert index first on ties. It is derived by
	// hand so a bug shared by both the kernel and ref() cannot hide here.
	fixtures := map[string]struct {
		choice []float32
		want   []int
	}{
		"all_equal_384": {func() []float32 {
			v := make([]float32, 384)
			for i := range v {
				v[i] = 3.5
			}
			return v
		}(), []int{0, 1, 2, 3, 4, 5}},
		"near_ties_384": {func() []float32 {
			v := make([]float32, 384)
			base := float32(1.0)
			for i := range v {
				v[i] = base
				base = v4NextAfter32(base)
			}
			return v
		}(), []int{383, 382, 381, 380, 379, 378}},
		"descending_384": {func() []float32 {
			v := make([]float32, 384)
			for i := range v {
				v[i] = float32(384 - i)
			}
			return v
		}(), []int{0, 1, 2, 3, 4, 5}},
		"ascending_384": {func() []float32 {
			v := make([]float32, 384)
			for i := range v {
				v[i] = float32(i)
			}
			return v
		}(), []int{383, 382, 381, 380, 379, 378}},
		"one_giant_outlier_384": {func() []float32 {
			v := make([]float32, 384)
			for i := range v {
				v[i] = 1
			}
			v[271] = 1e30
			return v
		}(), []int{271, 0, 1, 2, 3, 4}},
		"six_way_tie_at_max_384": {func() []float32 {
			v := make([]float32, 384)
			for i := range v {
				v[i] = 0
			}
			for _, i := range []int{10, 40, 133, 200, 300, 383} {
				v[i] = 9
			}
			return v
		}(), []int{10, 40, 133, 200, 300, 383}},
		"all_equal_8": {[]float32{2, 2, 2, 2, 2, 2, 2, 2}, []int{0, 1, 2, 3, 4, 5}},
		"near_ties_8": {[]float32{
			1, v4NextAfter32(1), v4NextAfter32(v4NextAfter32(1)), 0.5, 0.5, 0.25, -3, 7,
		}, []int{7, 2, 1, 0, 3, 4}},
	}
	for name, fx := range fixtures {
		t.Run(name, func(t *testing.T) {
			defer SetV4BitonicTopK(false) // restore the package-global default
			choice := fx.choice
			k := 6
			if k > len(choice) {
				k = len(choice)
			}
			SetV4BitonicTopK(false)
			ref := v4TopKIndices(choice, k)

			SetV4BitonicTopK(true)
			got := v4TopKIndices(choice, k)

			if len(ref) != len(choice) {
				t.Fatalf("reference returned %d indices, want full width %d", len(ref), len(choice))
			}
			// The kernel returns only the selected k; the router consumes only
			// the first k of the reference. Compare the load-bearing prefix.
			if !reflect.DeepEqual(got, ref[:k]) {
				t.Fatalf("bitonic order %v != reference top-k %v", got, ref[:k])
			}
			if len(got) != k {
				t.Fatalf("got %d indices, want %d", len(got), k)
			}
			// Independent contract check: the reference prefix AND the kernel
			// result must both equal the hand-derived order, so a mistaken
			// tie-break shared by both cannot pass.
			if !reflect.DeepEqual(got, fx.want) {
				t.Fatalf("bitonic order %v != expected %v", got, fx.want)
			}
			if !reflect.DeepEqual(ref[:k], fx.want) {
				t.Fatalf("reference top-k %v != expected %v", ref[:k], fx.want)
			}
		})
	}
}

// TestV4BitonicTopKFallbackContract pins the fail-closed contract. The helper
// must return a safe reference result, never a panic or a short/overlong
// slice, when it is handed an out-of-range k (the kernel-error path itself is
// unreachable through the routers, which validate top_k first, but the helper
// is package-visible and guards k directly).
func TestV4BitonicTopKFallbackContract(t *testing.T) {
	defer SetV4BitonicTopK(false) // restore the package-global default
	choice := []float32{5, 1, 4, 4, 9, 0}

	// k == len(choice) is serviceable by the kernel; the result must match.
	SetV4BitonicTopK(false)
	ref := v4TopKIndices(choice, len(choice))
	SetV4BitonicTopK(true)
	got := v4TopKIndices(choice, len(choice))
	if !reflect.DeepEqual(got, ref) {
		t.Fatalf("full-selection order %v != reference %v", got, ref)
	}

	// An out-of-range k must fall through to ref(), which yields the full
	// width, rather than indexing past it or panicking.
	SetV4BitonicTopK(true)
	if out := v4TopKIndices(choice, len(choice)+5); len(out) != len(choice) {
		t.Fatalf("k>len fallback width=%d, want %d", len(out), len(choice))
	}
	if out := v4TopKIndices(choice, 0); len(out) != len(choice) {
		t.Fatalf("k=0 fallback width=%d, want %d", len(out), len(choice))
	}

	// Minimal valid width stays deterministic on both paths.
	minimal := []float32{42}
	SetV4BitonicTopK(false)
	refMin := v4TopKIndices(minimal, 1)
	SetV4BitonicTopK(true)
	gotMin := v4TopKIndices(minimal, 1)
	if !reflect.DeepEqual(gotMin, refMin) || len(gotMin) != 1 || gotMin[0] != 0 {
		t.Fatalf("minimal case order %v (ref %v), want [0]", gotMin, refMin)
	}
}

// TestV41AndV4BitonicTopKWiringPreservesOracleSelection reruns the existing
// independent-oracle scenarios with the kernel flag ON, proving the flag does
// not perturb expert selection or weights.
func TestV41AndV4BitonicTopKWiringPreservesOracleSelection(t *testing.T) {
	defer SetV4BitonicTopK(false) // restore the package-global default
	SetV4BitonicTopK(true)

	t.Run("v41_384", func(t *testing.T) {
		cfg := v41TestCfg()
		logits := make([]float32, V41RouterExperts)
		bias := make([]float32, V41RouterExperts)
		for i := range logits {
			logits[i] = -8
		}
		for i := 100; i <= 105; i++ {
			logits[i] = 5
		}
		logits[200] = 1
		bias[200] = 3

		picks, err := v41Route(logits, bias, cfg)
		if err != nil {
			t.Fatalf("route: %v", err)
		}
		order := v41OracleTopK(logits, bias, V41RouterTopK)
		wantExperts := []int{200, 100, 101, 102, 103, 104}
		if !reflect.DeepEqual(order, wantExperts) {
			t.Fatalf("oracle order=%v want=%v", order, wantExperts)
		}
		for i, p := range picks {
			if p.expert != order[i] {
				t.Fatalf("pick[%d].expert=%d want=%d", i, p.expert, order[i])
			}
		}
	})

	t.Run("v4_8_wide", func(t *testing.T) {
		logits := []float32{-2, -0.5, 0, 1, 3, 0.25, -4, 2}
		bias := []float32{0, 0, 0, -4, 0, 3, 0, 0}
		picks, err := v4ScoredRoute(logits, bias, 3, 2.5)
		if err != nil {
			t.Fatal(err)
		}
		score := func(z float32) float64 { return math.Sqrt(math.Log1p(math.Exp(float64(z)))) }
		wantExperts := []int{5, 4, 7}
		denom := score(logits[5]) + score(logits[4]) + score(logits[7])
		for i, expert := range wantExperts {
			if picks[i].expert != expert {
				t.Fatalf("pick %d expert=%d, want %d (all=%+v)", i, picks[i].expert, expert, picks)
			}
			wantWeight := float32(score(logits[expert]) / denom * 2.5)
			if diff := math.Abs(float64(picks[i].weight - wantWeight)); diff > 2e-6 {
				t.Fatalf("expert %d weight=%g want=%g diff=%g", expert, picks[i].weight, wantWeight, diff)
			}
		}
	})
}

// TestV4PartialTopKFlagSelectsBoundedKernel is the fak#12975 reproduction
// witness. Before the fix the flagged arm delegated to compute.PersistentBitonicTopK,
// a FULL padded sort: at E=384 it materialised a 512-wide entry array and ran
// ~11520 compare-exchanges, so opting in was a pessimisation and no O(E log k)
// partial selection existed on the production path.
//
// The kernels are semantically equivalent (both return the correct top-k), so no
// input value distinguishes them -- only their WORK does. The witness therefore
// pins the flagged arm's deterministic allocation footprint, measured with
// testing.Benchmark's AllocedBytesPerOp (allocation bytes, not wall-clock time,
// so it is host-independent and reproducible). The O(E log k) heap retains a
// k-wide working set and allocates ~152 B/op; the retired full sort padded to
// 512 entries and allocated ~4144 B/op. The threshold 1000 B/op sits an order of
// magnitude above the heap and far below the full sort, so a future swap back to
// a full-sort kernel fails this witness loudly while machine noise cannot.
func TestV4PartialTopKFlagSelectsBoundedKernel(t *testing.T) {
	defer SetV4BitonicTopK(false) // restore the package-global default
	const E, K = V41RouterExperts, V41RouterTopK
	const boundedFootprintBytes = 1000 // heap ~152 B/op; full sort ~4144 B/op

	rng := rand.New(rand.NewSource(12975))
	choice := make([]float32, E)
	for i := range choice {
		choice[i] = rng.Float32()
	}

	// The bounded-heap kernel is exactly k wide, not E.
	partial := v4PartialTopKIndices(choice, K)
	if len(partial) != K {
		t.Fatalf("partial-select width = %d, want %d (a bounded top-k, not a full sort)", len(partial), K)
	}

	// Flag OFF -> the reference full ordering (E wide).
	SetV4BitonicTopK(false)
	ref := v4TopKIndices(choice, K)
	if len(ref) != E {
		t.Fatalf("reference width = %d, want the full E=%d ordering", len(ref), E)
	}

	// Flag ON -> the bounded kernel selects the same experts in the pinned order.
	SetV4BitonicTopK(true)
	got := v4TopKIndices(choice, K)
	if len(got) != K {
		t.Fatalf("flagged width = %d, want %d: the flag did not route through the bounded "+
			"O(E log k) partial selection (the retired bitonic arm returned the full E=%d ordering)",
			len(got), K, E)
	}
	if !reflect.DeepEqual(got, ref[:K]) {
		t.Fatalf("flagged partial-select order %v != reference top-k %v", got, ref[:K])
	}
	if !reflect.DeepEqual(got, partial) {
		t.Fatalf("flagged arm %v did not route through v4PartialTopKIndices %v", got, partial)
	}

	// The load-bearing cost witness: the flagged seam must not carry a full-sort
	// allocation. This is the assertion that FAILS against the retired bitonic
	// delegation (~4144 B/op) and PASSES only with the bounded heap.
	flagged := testing.Benchmark(func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = v4TopKIndices(choice, K)
		}
	})
	if bytes := flagged.AllocedBytesPerOp(); bytes >= boundedFootprintBytes {
		t.Fatalf("flagged top-k allocates %d B/op, want < %d: the flag is still routing through a "+
			"full-sort kernel (the O(E log k) bounded heap allocates ~152 B/op)", bytes, boundedFootprintBytes)
	}

	// The retired kernel's own receipt proves the full-sort shape the flagged
	// arm no longer carries; this keeps the contrast in the same test.
	_, receipt, err := compute.PersistentBitonicTopK(choice, K)
	if err != nil {
		t.Fatalf("retired bitonic kernel: %v", err)
	}
	if receipt.PaddedN <= K {
		t.Fatalf("fixture does not exercise the bitonic padding (padded=%d, K=%d)", receipt.PaddedN, K)
	}
	if receipt.Comparisons <= E {
		t.Fatalf("retired full sort ran %d comparisons, want > E=%d (not a partial select)", receipt.Comparisons, E)
	}

	// Degenerate k must fail closed to the reference width on both the kernel
	// helper and the flagged seam.
	for _, k := range []int{0, -3, E + 1} {
		if w := len(v4PartialTopKIndices(choice, k)); w != E {
			t.Fatalf("degenerate k=%d kernel width = %d, want %d", k, w, E)
		}
		SetV4BitonicTopK(true)
		if w := len(v4TopKIndices(choice, k)); w != E {
			t.Fatalf("degenerate k=%d flagged width = %d, want %d", k, w, E)
		}
	}
}

// BenchmarkV4TopKIndicesE384 measures the E=384 top-k selection three ways: the
// reference full stable sort, the opt-in partial-selection kernel the flag now
// selects (v4PartialTopKIndices), and the retired full bitonic network it
// replaced, so the issue's before/after routing-time criterion is answerable.
//
// The partial-select arm is the O(E log k) bounded heap (fak#12975): at E=384,
// k=6 it beats the reference. The bitonic arm is retained as the documented
// negative: it pads 384 -> 512 and runs a full network (~11520 compare-
// exchanges) against the reference sort's ~3300, so it is slower -- which is
// exactly why the flag now selects the heap and not the bitonic kernel.
func BenchmarkV4TopKIndicesE384(b *testing.B) {
	defer SetV4BitonicTopK(false) // restore the package-global default
	const E, K = V41RouterExperts, V41RouterTopK
	rng := rand.New(rand.NewSource(12970))
	choice := make([]float32, E)
	for i := range choice {
		choice[i] = rng.Float32()
	}

	b.Run("reference_stable_sort", func(b *testing.B) {
		SetV4BitonicTopK(false)
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			_ = v4TopKIndices(choice, K)
		}
	})
	b.Run("partial_select_kernel", func(b *testing.B) {
		SetV4BitonicTopK(true)
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			_ = v4TopKIndices(choice, K)
		}
	})
	b.Run("bitonic_full_sort_retired", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			_, _, _ = compute.PersistentBitonicTopK(choice, K)
		}
	})
}

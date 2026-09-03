package compute

import (
	"math"
	"math/rand"
	"testing"
)

func TestBoundedAsymmetricForget(t *testing.T) {
	minBound := MinBoundedDecay
	const maxBound = MaxBoundedDecay

	// Subtest 1: Table-driven unit tests for reference mathematical values and edge cases.
	t.Run("table_driven_cases", func(t *testing.T) {
		tests := []struct {
			name      string
			aLog      []float32
			fProj     []float32
			dtBias    []float32
			wantDecay []float32
			tol       float64
		}{
			{
				name:      "zero_inputs",
				aLog:      []float32{0.0},
				fProj:     []float32{0.0},
				dtBias:    []float32{0.0},
				wantDecay: []float32{float32(math.Exp(-2.5))},
				tol:       1e-6,
			},
			{
				name:   "unit_positive",
				aLog:   []float32{0.0},
				fProj:  []float32{1.0},
				dtBias: []float32{0.0},
				// scaled = 1.0, sig = 1/(1+exp(-1)) ~ 0.7310586, forget = -5*sig ~ -3.655293, decay ~ 0.0258543
				wantDecay: []float32{float32(math.Exp(-5.0 / (1.0 + math.Exp(-1.0))))},
				tol:       1e-6,
			},
			{
				name:   "unit_negative",
				aLog:   []float32{0.0},
				fProj:  []float32{-1.0},
				dtBias: []float32{0.0},
				// scaled = -1.0, sig = 1/(1+exp(1)) ~ 0.2689414, forget = -5*sig ~ -1.344707, decay ~ 0.26062
				wantDecay: []float32{float32(math.Exp(-5.0 / (1.0 + math.Exp(1.0))))},
				tol:       1e-6,
			},
			{
				name:   "positive_bias_offset",
				aLog:   []float32{0.0},
				fProj:  []float32{0.5},
				dtBias: []float32{0.5},
				// scaled = 1.0, same as unit_positive
				wantDecay: []float32{float32(math.Exp(-5.0 / (1.0 + math.Exp(-1.0))))},
				tol:       1e-6,
			},
			{
				name:   "cancelling_bias_to_zero",
				aLog:   []float32{1.5},
				fProj:  []float32{3.25},
				dtBias: []float32{-3.25},
				// scaled = exp(1.5) * 0 = 0 -> decay = exp(-2.5)
				wantDecay: []float32{float32(math.Exp(-2.5))},
				tol:       1e-6,
			},
			{
				name:   "saturating_positive",
				aLog:   []float32{5.0},
				fProj:  []float32{2.0},
				dtBias: []float32{0.0},
				// scaled ~ 296.8 >> 40 -> sigmoid ~ 1.0 -> decay = exp(-5)
				wantDecay: []float32{minBound},
				tol:       1e-6,
			},
			{
				name:   "saturating_negative",
				aLog:   []float32{5.0},
				fProj:  []float32{-2.0},
				dtBias: []float32{0.0},
				// scaled ~ -296.8 << -40 -> sigmoid ~ 0.0 -> decay = 1.0
				wantDecay: []float32{maxBound},
				tol:       1e-6,
			},
			{
				name:      "extreme_pos_1e4",
				aLog:      []float32{1e4},
				fProj:     []float32{1e4},
				dtBias:    []float32{1e4},
				wantDecay: []float32{minBound},
				tol:       1e-6,
			},
			{
				name:      "extreme_neg_1e4",
				aLog:      []float32{1e4},
				fProj:     []float32{-1e4},
				dtBias:    []float32{-1e4},
				wantDecay: []float32{maxBound},
				tol:       1e-6,
			},
			{
				name:   "extreme_pos_neg_sum_zero_cancelling",
				aLog:   []float32{1e4},
				fProj:  []float32{1e4},
				dtBias: []float32{-1e4},
				// u == 0, exp(1e4)*0 must not produce NaN, scaled=0 -> decay = exp(-2.5)
				wantDecay: []float32{float32(math.Exp(-2.5))},
				tol:       1e-6,
			},
			{
				name:   "extreme_neg_alog_with_pos_proj",
				aLog:   []float32{-1e4},
				fProj:  []float32{1e4},
				dtBias: []float32{1e4},
				// exp(-1e4) * 2e4 = 0 -> scaled=0 -> decay = exp(-2.5)
				wantDecay: []float32{float32(math.Exp(-2.5))},
				tol:       1e-6,
			},
			{
				name:      "extreme_neg_alog_with_neg_proj",
				aLog:      []float32{-1e4},
				fProj:     []float32{-1e4},
				dtBias:    []float32{-1e4},
				wantDecay: []float32{float32(math.Exp(-2.5))},
				tol:       1e-6,
			},
			{
				name:      "ultra_extreme_pos_1e6",
				aLog:      []float32{1e6},
				fProj:     []float32{1e6},
				dtBias:    []float32{0.0},
				wantDecay: []float32{minBound},
				tol:       1e-6,
			},
			{
				name:      "ultra_extreme_neg_1e6",
				aLog:      []float32{-1e6},
				fProj:     []float32{1e6},
				dtBias:    []float32{0.0},
				wantDecay: []float32{float32(math.Exp(-2.5))},
				tol:       1e-6,
			},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				got := BoundedAsymmetricForget(tc.aLog, tc.fProj, tc.dtBias)
				if len(got) != len(tc.wantDecay) {
					t.Fatalf("length mismatch: got %d, want %d", len(got), len(tc.wantDecay))
				}
				for i := range got {
					g := got[i]
					w := tc.wantDecay[i]
					if math.IsNaN(float64(g)) || math.IsInf(float64(g), 0) {
						t.Fatalf("index %d: got non-finite decay %v", i, g)
					}
					if g < minBound || g > maxBound {
						t.Fatalf("index %d: decay %v strictly outside [%v, %v]", i, g, minBound, maxBound)
					}
					if math.Abs(float64(g-w)) > tc.tol {
						t.Fatalf("index %d: got %v, want %v (diff %v > tol %v)", i, g, w, math.Abs(float64(g-w)), tc.tol)
					}
				}
			})
		}
	})

	// Subtest 2: Complete Cartesian grid of extreme inputs (±10^4 and 0 across all 3 operands = 27 cases).
	t.Run("extreme_combinations_grid_27", func(t *testing.T) {
		values := []float32{-10000.0, 0.0, 10000.0}
		for _, a := range values {
			for _, f := range values {
				for _, dt := range values {
					aLog := []float32{a}
					fProj := []float32{f}
					dtBias := []float32{dt}

					got := BoundedAsymmetricForget(aLog, fProj, dtBias)
					if len(got) != 1 {
						t.Fatalf("expected 1 result, got %d", len(got))
					}
					d := got[0]
					if math.IsNaN(float64(d)) {
						t.Fatalf("NaN decay produced for aLog=%v, fProj=%v, dtBias=%v", a, f, dt)
					}
					if math.IsInf(float64(d), 0) {
						t.Fatalf("Inf decay produced for aLog=%v, fProj=%v, dtBias=%v", a, f, dt)
					}
					if d < minBound || d > maxBound {
						t.Fatalf("decay %v out of bounds [%v, %v] for aLog=%v, fProj=%v, dtBias=%v", d, minBound, maxBound, a, f, dt)
					}
				}
			}
		}
	})

	// Subtest 3: Randomized stress test across extreme ranges (±10^4).
	t.Run("randomized_extreme_range_bounds", func(t *testing.T) {
		rng := rand.New(rand.NewSource(10761))
		n := 1024
		aLog := make([]float32, n)
		fProj := make([]float32, n)
		dtBias := make([]float32, n)

		for i := 0; i < n; i++ {
			// Generate numbers spanning -10000 to +10000 with varying exponents
			aLog[i] = (rng.Float32()*2.0 - 1.0) * float32(math.Pow(10, float64(rng.Intn(5))))
			fProj[i] = (rng.Float32()*2.0 - 1.0) * float32(math.Pow(10, float64(rng.Intn(5))))
			dtBias[i] = (rng.Float32()*2.0 - 1.0) * float32(math.Pow(10, float64(rng.Intn(5))))
		}

		got := BoundedAsymmetricForget(aLog, fProj, dtBias)
		if len(got) != n {
			t.Fatalf("got length %d, want %d", len(got), n)
		}

		for i, d := range got {
			if math.IsNaN(float64(d)) {
				t.Fatalf("index %d: NaN decay for aLog=%v, fProj=%v, dtBias=%v", i, aLog[i], fProj[i], dtBias[i])
			}
			if math.IsInf(float64(d), 0) {
				t.Fatalf("index %d: Inf decay for aLog=%v, fProj=%v, dtBias=%v", i, aLog[i], fProj[i], dtBias[i])
			}
			if d < minBound || d > maxBound {
				t.Fatalf("index %d: decay %v strictly outside [%v, %v]", i, d, minBound, maxBound)
			}
		}
	})

	// Subtest 4: Shape and broadcasting behaviors.
	t.Run("broadcasting_and_shapes", func(t *testing.T) {
		// 4a. Matching vectors
		vA := []float32{0.0, 1.0, -1.0, 2.0}
		vF := []float32{0.5, -0.5, 1.0, -2.0}
		vB := []float32{0.1, -0.1, 0.2, -0.2}
		got := BoundedAsymmetricForget(vA, vF, vB)
		if len(got) != 4 {
			t.Fatalf("expected 4 outputs, got %d", len(got))
		}

		// 4b. Head-wise broadcast (tokens=3, heads=2 -> len(fProj)=6, len(aLog)=2, len(dtBias)=2)
		aHeads := []float32{0.0, 1.0}
		dtHeads := []float32{0.1, -0.1}
		fTokens := []float32{0.5, -0.5, 1.0, -1.0, 0.0, 0.0}
		gotSeq := BoundedAsymmetricForget(aHeads, fTokens, dtHeads)
		if len(gotSeq) != 6 {
			t.Fatalf("expected 6 outputs, got %d", len(gotSeq))
		}
		// Verify token 0 head 0 matches single elementwise
		want0 := BoundedAsymmetricForget([]float32{aHeads[0]}, []float32{fTokens[0]}, []float32{dtHeads[0]})
		if math.Abs(float64(gotSeq[0]-want0[0])) > 1e-6 {
			t.Fatalf("broadcast token 0 head 0 mismatch: got %v, want %v", gotSeq[0], want0[0])
		}
		// Verify token 1 head 1 matches single elementwise
		want3 := BoundedAsymmetricForget([]float32{aHeads[1]}, []float32{fTokens[3]}, []float32{dtHeads[1]})
		if math.Abs(float64(gotSeq[3]-want3[0])) > 1e-6 {
			t.Fatalf("broadcast token 1 head 1 mismatch: got %v, want %v", gotSeq[3], want3[0])
		}

		// 4c. Scalar aLog broadcast
		gotScalarA := BoundedAsymmetricForget([]float32{0.0}, []float32{1.0, -1.0}, []float32{0.0, 0.0})
		if len(gotScalarA) != 2 {
			t.Fatalf("expected 2 outputs, got %d", len(gotScalarA))
		}

		// 4d. Scalar dtBias broadcast
		gotScalarB := BoundedAsymmetricForget([]float32{0.0, 0.0}, []float32{1.0, -1.0}, []float32{0.5})
		if len(gotScalarB) != 2 {
			t.Fatalf("expected 2 outputs, got %d", len(gotScalarB))
		}

		// 4e. Nil / empty dtBias
		gotNoBias := BoundedAsymmetricForget([]float32{0.0}, []float32{1.0}, nil)
		wantNoBias := BoundedAsymmetricForget([]float32{0.0}, []float32{1.0}, []float32{0.0})
		if len(gotNoBias) != 1 || math.Abs(float64(gotNoBias[0]-wantNoBias[0])) > 1e-6 {
			t.Fatalf("nil dtBias mismatch: got %v, want %v", gotNoBias, wantNoBias)
		}

		// 4f. Empty fProj
		gotEmpty := BoundedAsymmetricForget([]float32{0.0}, []float32{}, []float32{})
		if gotEmpty == nil || len(gotEmpty) != 0 {
			t.Fatalf("expected empty slice, got %v", gotEmpty)
		}

		// 4g. Nil fProj
		if gotNil := BoundedAsymmetricForget([]float32{0.0}, nil, nil); gotNil != nil {
			t.Fatalf("expected nil for nil fProj, got %v", gotNil)
		}

		// 4h. Empty aLog
		if gotNilA := BoundedAsymmetricForget([]float32{}, []float32{1.0}, nil); gotNilA != nil {
			t.Fatalf("expected nil for empty aLog, got %v", gotNilA)
		}

		// 4i. Incompatible dimensions
		if gotMismatch := BoundedAsymmetricForget([]float32{1.0, 2.0, 3.0}, []float32{1.0, 2.0, 3.0, 4.0, 5.0}, nil); gotMismatch != nil {
			t.Fatalf("expected nil for dimension mismatch, got %v", gotMismatch)
		}
	})

	// Subtest 5: Monotonicity with respect to fProj.
	t.Run("monotonicity", func(t *testing.T) {
		aLog := []float32{0.0}
		dtBias := []float32{0.0}
		// As fProj increases, scaled increases, sigmoid increases, forget becomes more negative,
		// and decay = exp(forget) strictly decreases.
		steps := 50
		prevDecay := float32(2.0)
		for i := 0; i <= steps; i++ {
			f := -10.0 + float32(i)*(20.0/float32(steps))
			d := BoundedAsymmetricForget(aLog, []float32{f}, dtBias)[0]
			if d > prevDecay {
				t.Fatalf("monotonicity violated at f=%v: prevDecay=%v, currentDecay=%v", f, prevDecay, d)
			}
			prevDecay = d
		}
	})

	// Subtest 6: NaN robustness fail-closed.
	t.Run("nan_inf_robustness", func(t *testing.T) {
		nan := float32(math.NaN())
		posInf := float32(math.Inf(1))
		negInf := float32(math.Inf(-1))

		inputs := [][]float32{
			{nan, 0.0, 0.0},
			{0.0, nan, 0.0},
			{0.0, 0.0, nan},
			{posInf, 1.0, 0.0},
			{negInf, 1.0, 0.0},
		}

		for idx, inp := range inputs {
			got := BoundedAsymmetricForget([]float32{inp[0]}, []float32{inp[1]}, []float32{inp[2]})
			if len(got) != 1 {
				t.Fatalf("case %d: expected 1 output", idx)
			}
			d := got[0]
			if math.IsNaN(float64(d)) || math.IsInf(float64(d), 0) {
				t.Fatalf("case %d: non-finite output %v for input %v", idx, d, inp)
			}
			if d < minBound || d > maxBound {
				t.Fatalf("case %d: out of bounds %v for input %v", idx, d, inp)
			}
		}
	})
}

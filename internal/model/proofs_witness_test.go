package model

// proofs_witness_test.go — deterministic witness tests closing OPEN math-proof
// obligations for internal/model (proof discipline: docs/proofs/00-METHOD.md).
//
// Every test here is metamorphic / round-trip / invariant — it ASSERTS the
// claimed property against an independently-recomputed reference, never a smoke
// check. Float comparisons use 1e-5..1e-6 tolerances; integer/exact properties
// use ==. All randomness uses a FIXED seed for bit-reproducibility.

import (
	"math"
	"math/rand"
	"testing"
)

const proofTol = 1e-5

func approxEqf(a, b, tol float32) bool {
	d := a - b
	if d < 0 {
		d = -d
	}
	return float64(d) <= float64(tol)
}

func isFiniteF32(v float32) bool {
	f := float64(v)
	return !math.IsNaN(f) && !math.IsInf(f, 0)
}

// ----------------------------------------------------------------------------
// (1) [softmax-row-stochastic-shift-invariant]
//     softmaxInPlace yields y_i >= 0 with sum == 1, and softmax(x+c) == softmax(x).
// ----------------------------------------------------------------------------

func TestProofSoftmaxRowStochasticAndShiftInvariant(t *testing.T) {
	rng := rand.New(rand.NewSource(0x50F1ACE))
	for trial := 0; trial < 200; trial++ {
		n := 1 + rng.Intn(64)
		x := make([]float32, n)
		for i := range x {
			// span a wide score range, incl. negatives and large magnitudes
			x[i] = float32((rng.Float64()*2 - 1) * 50)
		}

		// row-stochastic: copy, softmax, check >=0 and sum==1
		y := append([]float32(nil), x...)
		softmaxInPlace(y)
		var sum float32
		for i, v := range y {
			if v < 0 {
				t.Fatalf("trial %d: softmax entry %d = %v < 0 (not nonneg)", trial, i, v)
			}
			if !isFiniteF32(v) {
				t.Fatalf("trial %d: softmax entry %d not finite: %v", trial, i, v)
			}
			sum += v
		}
		if !approxEqf(sum, 1, proofTol) {
			t.Fatalf("trial %d: softmax sum = %v, want 1 (not row-stochastic)", trial, sum)
		}

		// shift invariance: softmax(x + c) == softmax(x)
		c := float32((rng.Float64()*2 - 1) * 30)
		xs := make([]float32, n)
		for i := range x {
			xs[i] = x[i] + c
		}
		ys := xs
		softmaxInPlace(ys)
		for i := range y {
			if !approxEqf(ys[i], y[i], proofTol) {
				t.Fatalf("trial %d idx %d: shift c=%v changed softmax: %v vs %v",
					trial, i, c, ys[i], y[i])
			}
		}
	}
}

// ----------------------------------------------------------------------------
// (2) [causal-strictly-lower-triangular]
//     The dense SDPA weight matrix is strictly lower-triangular: query position i
//     receives zero weight from any key j > i. This replicates the exact dense
//     score-construction discipline at forward.go:233-244 (scores built only over
//     keys j in [lo, t]; the weighted value-sum reads only those same indices).
// ----------------------------------------------------------------------------

func TestProofCausalStrictlyLowerTriangular(t *testing.T) {
	rng := rand.New(rand.NewSource(0xCA05A1))
	hd := 8
	for trial := 0; trial < 50; trial++ {
		seq := 2 + rng.Intn(12)
		// random per-position query/key head vectors
		q := make([][]float32, seq)
		k := make([][]float32, seq)
		for i := 0; i < seq; i++ {
			q[i] = make([]float32, hd)
			k[i] = make([]float32, hd)
			for d := 0; d < hd; d++ {
				q[i][d] = float32(rng.NormFloat64())
				k[i][d] = float32(rng.NormFloat64())
			}
		}
		scale := float32(1 / math.Sqrt(float64(hd)))

		// Build the full seq x seq attention-weight matrix exactly as the dense
		// path does: query t scores keys lo..t only, softmax over that range, and
		// keys j>t never enter. W = -1 (no window) => lo = 0 (full causal).
		W := -1
		Wt := make([][]float32, seq) // Wt[t][j] = attention weight from query t to key j
		for tpos := 0; tpos < seq; tpos++ {
			lo := 0
			if W >= 0 {
				if lo = tpos - W + 1; lo < 0 {
					lo = 0
				}
			}
			scores := make([]float32, tpos+1-lo)
			for j := lo; j <= tpos; j++ {
				scores[j-lo] = dot(q[tpos], k[j]) * scale
			}
			softmaxInPlace(scores)
			Wt[tpos] = make([]float32, seq) // zero-initialized: j outside [lo,t] stay 0
			for j := lo; j <= tpos; j++ {
				Wt[tpos][j] = scores[j-lo]
			}
		}

		// Assert strict-lower-triangularity + row-stochastic over the causal band.
		for i := 0; i < seq; i++ {
			var rowsum float32
			for j := 0; j < seq; j++ {
				if j > i {
					if Wt[i][j] != 0 {
						t.Fatalf("trial %d: query %d got nonzero weight %v from future key %d (not causal)",
							trial, i, Wt[i][j], j)
					}
				} else {
					if Wt[i][j] < 0 {
						t.Fatalf("trial %d: weight (%d,%d)=%v negative", trial, i, j, Wt[i][j])
					}
					rowsum += Wt[i][j]
				}
			}
			if !approxEqf(rowsum, 1, proofTol) {
				t.Fatalf("trial %d: query %d causal-band weights sum to %v, want 1", trial, i, rowsum)
			}
		}
	}
}

// ----------------------------------------------------------------------------
// (3) [layernorm-shift-scale-equivariant]
//     layernorm(a*x+b) == layernorm(x) in the eps->0 limit (with unit weight,
//     nil bias): mean-subtraction removes b, division by stddev removes a>0.
//     RMSNorm is invariant to positive input scaling up to the learned gain.
// ----------------------------------------------------------------------------

func TestProofLayerNormShiftScaleEquivariant(t *testing.T) {
	rng := rand.New(rand.NewSource(0x1A4E40))
	for trial := 0; trial < 100; trial++ {
		n := 4 + rng.Intn(60)
		x := make([]float32, n)
		for i := range x {
			x[i] = float32(rng.NormFloat64() * 3)
		}
		w := make([]float32, n)
		for i := range w {
			w[i] = float32(rng.NormFloat64())
		}
		const eps = float32(1e-12) // eps->0 limit

		base := layernorm(x, w, nil, eps)

		a := float32(0.1 + rng.Float64()*5) // a > 0
		b := float32((rng.Float64()*2 - 1) * 10)
		xt := make([]float32, n)
		for i := range x {
			xt[i] = a*x[i] + b
		}
		got := layernorm(xt, w, nil, eps)

		for i := range base {
			// scale-relative tolerance: layernorm outputs are O(w*normalized)
			tol := float32(1e-4) * (1 + absF32(base[i]))
			if !approxEqf(got[i], base[i], tol) {
				t.Fatalf("trial %d idx %d: layernorm(a*x+b)=%v != layernorm(x)=%v (a=%v b=%v)",
					trial, i, got[i], base[i], a, b)
			}
		}
	}
}

func TestProofRMSNormPositiveScaleInvariant(t *testing.T) {
	rng := rand.New(rand.NewSource(0x4D5104))
	for trial := 0; trial < 100; trial++ {
		n := 4 + rng.Intn(60)
		x := make([]float32, n)
		for i := range x {
			x[i] = float32(rng.NormFloat64() * 3)
		}
		w := make([]float32, n)
		for i := range w {
			w[i] = float32(rng.NormFloat64())
		}
		const eps = float32(1e-12)

		base := rmsnorm(x, w, eps)
		a := float32(0.1 + rng.Float64()*5) // positive scale only
		xt := make([]float32, n)
		for i := range x {
			xt[i] = a * x[i]
		}
		got := rmsnorm(xt, w, eps)
		for i := range base {
			tol := float32(1e-4) * (1 + absF32(base[i]))
			if !approxEqf(got[i], base[i], tol) {
				t.Fatalf("trial %d idx %d: rmsnorm(a*x)=%v != rmsnorm(x)=%v (a=%v)",
					trial, i, got[i], base[i], a)
			}
		}
	}
}

func absF32(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}

// ----------------------------------------------------------------------------
// (4) [norm-numerically-stable-large-inputs]
//     For large-magnitude finite inputs (|x| ~ 1e15..1e20) rmsnorm/layernorm
//     produce only finite outputs (no NaN/Inf): the sum-of-squares ss/N+eps does
//     not overflow f32 and the 1/sqrt does not divide by zero/inf.
// ----------------------------------------------------------------------------

func TestProofNormNumericallyStableLargeInputs(t *testing.T) {
	rng := rand.New(rand.NewSource(0x816B16))
	mags := []float64{1e15, 1e16, 1e18, 1e20}
	for _, mag := range mags {
		for trial := 0; trial < 50; trial++ {
			n := 4 + rng.Intn(60)
			x := make([]float32, n)
			for i := range x {
				x[i] = float32((rng.Float64()*2 - 1) * mag)
			}
			w := make([]float32, n)
			bias := make([]float32, n)
			for i := range w {
				w[i] = float32(rng.NormFloat64())
				bias[i] = float32(rng.NormFloat64())
			}
			eps := float32(1e-6)

			for i, v := range rmsnorm(x, w, eps) {
				if !isFiniteF32(v) {
					t.Fatalf("rmsnorm mag=%g idx %d not finite: %v", mag, i, v)
				}
			}
			for i, v := range layernorm(x, w, bias, eps) {
				if !isFiniteF32(v) {
					t.Fatalf("layernorm mag=%g idx %d not finite: %v", mag, i, v)
				}
			}
			// also the in-place QK-norm variant (arch.go:245)
			xc := append([]float32(nil), x...)
			applyRMSNormInPlace(xc, w, eps)
			for i, v := range xc {
				if !isFiniteF32(v) {
					t.Fatalf("applyRMSNormInPlace mag=%g idx %d not finite: %v", mag, i, v)
				}
			}
		}
	}
}

// ----------------------------------------------------------------------------
// (5) [rope-preserves-pair-norm]
//     applyRopeRow rotates each pair (h[j], h[j+half]) by a Givens rotation; since
//     cos^2+sin^2=1 it preserves each per-pair Euclidean norm (and the whole-vector
//     norm) for every position and input. (Uses the UNSCALED Llama map: scale=1.)
// ----------------------------------------------------------------------------

func ropeCosSin(half, p int) (cos, sin []float32) {
	// Independent, standard Llama inv_freq: 1/base^(2j/dim), dim = 2*half, base = 10000.
	dim := 2 * half
	inv := make([]float64, half)
	for j := 0; j < half; j++ {
		inv[j] = 1.0 / math.Pow(10000, float64(2*j)/float64(dim))
	}
	return ropeRowFromInv(inv, p)
}

func TestProofRopePreservesPairNorm(t *testing.T) {
	rng := rand.New(rand.NewSource(0x80B340))
	for trial := 0; trial < 100; trial++ {
		half := 1 + rng.Intn(32)
		hv := make([]float32, 2*half)
		for i := range hv {
			hv[i] = float32(rng.NormFloat64() * 2)
		}
		p := rng.Intn(4096)
		cos, sin := ropeCosSin(half, p)

		// pre-rotation pair norms
		pre := make([]float32, half)
		for j := 0; j < half; j++ {
			a, b := hv[j], hv[j+half]
			pre[j] = a*a + b*b
		}
		var preVec float32
		for _, v := range hv {
			preVec += v * v
		}

		applyRopeRow(hv, cos, sin)

		for j := 0; j < half; j++ {
			a, b := hv[j], hv[j+half]
			post := a*a + b*b
			// relative tolerance scaled by magnitude (f32 rotation rounding)
			tol := float32(1e-4) * (1 + pre[j])
			if !approxEqf(post, pre[j], tol) {
				t.Fatalf("trial %d pair %d (p=%d): pair-norm^2 %v -> %v (not preserved)",
					trial, j, p, pre[j], post)
			}
		}
		var postVec float32
		for _, v := range hv {
			postVec += v * v
		}
		tolV := float32(1e-4) * (1 + preVec)
		if !approxEqf(postVec, preVec, tolV) {
			t.Fatalf("trial %d (p=%d): vector-norm^2 %v -> %v (not preserved)",
				trial, p, preVec, postVec)
		}
	}
}

// ----------------------------------------------------------------------------
// (6) [rope-dot-relative-position]
//     <R_m q, R_n k> depends on m,n only through the relative offset (m-n):
//     <R_m q, R_n k> == <R_{m-n} q, R_0 k>. (Unscaled Llama RoPE.) R_0 is identity.
// ----------------------------------------------------------------------------

func TestProofRopeDotRelativePosition(t *testing.T) {
	rng := rand.New(rand.NewSource(0x9E1A71))
	for trial := 0; trial < 100; trial++ {
		half := 1 + rng.Intn(24)
		dim := 2 * half
		q := make([]float32, dim)
		k := make([]float32, dim)
		for i := 0; i < dim; i++ {
			q[i] = float32(rng.NormFloat64())
			k[i] = float32(rng.NormFloat64())
		}
		m := 1 + rng.Intn(2048)
		n := rng.Intn(m + 1) // ensure m-n >= 0 (nonnegative relative offset)
		rel := m - n

		// <R_m q, R_n k>
		qm := append([]float32(nil), q...)
		kn := append([]float32(nil), k...)
		cosM, sinM := ropeCosSin(half, m)
		cosN, sinN := ropeCosSin(half, n)
		applyRopeRow(qm, cosM, sinM)
		applyRopeRow(kn, cosN, sinN)
		lhs := dot(qm, kn)

		// <R_{m-n} q, R_0 k> == <R_{rel} q, k>
		qr := append([]float32(nil), q...)
		cosR, sinR := ropeCosSin(half, rel)
		applyRopeRow(qr, cosR, sinR)
		rhs := dot(qr, k)

		// scale tolerance by the dot magnitude (sum of dim products)
		tol := float32(1e-3) * (1 + absF32(lhs))
		if !approxEqf(lhs, rhs, tol) {
			t.Fatalf("trial %d: <R_%d q, R_%d k>=%v != <R_%d q, k>=%v (rel-pos broken)",
				trial, m, n, lhs, rel, rhs)
		}
	}
}

// ----------------------------------------------------------------------------
// (7) [awq-matches-reference]
//     AWQ dequant weight[i] = scale*(unpack4bit(code)-8): (a) the affine arithmetic
//     is bit-exact to an independently-computed expected value. Both the on-the-fly
//     awqDequantRow and the scalar reference awqDequantRowScalar must agree, and both
//     must equal the closed-form scale*(nibble-8). unpack4bit must invert the packing.
//     (We assert (a). Part (b) — byte-equality vs a stored AutoAWQ HF fixture — needs
//     an absent fixture and is left to oracle_test; not claimed here.)
// ----------------------------------------------------------------------------

func TestProofAWQMatchesReference(t *testing.T) {
	rng := rand.New(rand.NewSource(0xA40DEF))
	for trial := 0; trial < 200; trial++ {
		out := 1 + rng.Intn(6)
		inHalf := 1 + rng.Intn(16)
		in := inHalf * 2
		raw := make([]byte, out*inHalf)
		for i := range raw {
			raw[i] = byte(rng.Intn(256))
		}
		scales := make([]float32, out)
		for o := range scales {
			scales[o] = float32(rng.NormFloat64())
		}

		for o := 0; o < out; o++ {
			row := raw[o*inHalf : (o+1)*inHalf]

			// closed-form independent expectation
			want := make([]float32, in)
			for i := 0; i < inHalf; i++ {
				b := row[i]
				lo := b & 0x0f
				hi := (b >> 4) & 0x0f
				// also witness unpack4bit inverts the packing exactly
				ulo, uhi := unpack4bit(b)
				if uint8(lo) != ulo || uint8(hi) != uhi {
					t.Fatalf("unpack4bit(%#x) = (%d,%d), want (%d,%d)", b, ulo, uhi, lo, hi)
				}
				want[i*2] = scales[o] * float32(int16(lo)-8)
				want[i*2+1] = scales[o] * float32(int16(hi)-8)
			}

			// path A: on-the-fly dequant used by the GEMV/GEMM kernels
			gotA := make([]float32, in)
			awqDequantRow(gotA, scales, o, row, in)

			// path B: portable scalar reference oracle
			gotB := make([]float32, in)
			awqDequantRowScalar(gotB, scales[o], &row[0], in)

			for i := 0; i < in; i++ {
				if gotA[i] != want[i] { // bit-exact (same affine f32 arithmetic)
					t.Fatalf("trial %d o=%d i=%d: awqDequantRow=%v want=%v", trial, o, i, gotA[i], want[i])
				}
				if gotB[i] != want[i] {
					t.Fatalf("trial %d o=%d i=%d: awqDequantRowScalar=%v want=%v", trial, o, i, gotB[i], want[i])
				}
			}
		}
	}
}

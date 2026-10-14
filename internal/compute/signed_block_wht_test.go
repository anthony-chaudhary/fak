package compute

import (
	"math"
	"math/bits"
	"math/rand"
	"testing"
)

// signedWHTCase builds a deterministic width-keyed rotation for the given widths: each
// sign vector is a seeded ±1 stream so at least one entry is -1.
func signedWHTCase(t *testing.T, block int, widths []int) (*SignedBlockWHT, [][]int) {
	t.Helper()
	rng := rand.New(rand.NewSource(0x5eed))
	values := make([]int, 0)
	got := make([][]int, 0, len(widths))
	for _, w := range widths {
		row := make([]int, w)
		for i := range row {
			if rng.Intn(2) == 0 {
				row[i] = 1
			} else {
				row[i] = -1
			}
		}
		values = append(values, row...)
		got = append(got, row)
	}
	wht, err := NewSignedBlockWHT(block, widths, values)
	if err != nil {
		t.Fatalf("NewSignedBlockWHT(%d, %v, len=%d): %v", block, widths, len(values), err)
	}
	return wht, got
}

func almostEqual(a, b float32, rel, abs float64) bool {
	d := math.Abs(float64(a) - float64(b))
	if d <= abs {
		return true
	}
	mag := math.Max(math.Abs(float64(a)), math.Abs(float64(b)))
	return d <= rel*mag
}

// denseBlockWHT64 is an independent block-diagonal normalized Sylvester Walsh-Hadamard
// transform built from the closed form H[i][j] = (-1)^popcount(i AND j) / sqrt(block),
// NOT from the production butterfly. It is the algebra oracle.
func denseBlockWHT64(vec []float64, block int) []float64 {
	out := make([]float64, len(vec))
	scale := 1.0 / math.Sqrt(float64(block))
	for base := 0; base < len(vec); base += block {
		for i := 0; i < block; i++ {
			var sum float64
			for j := 0; j < block; j++ {
				if bits.OnesCount(uint(i&j))&1 == 0 {
					sum += vec[base+j]
				} else {
					sum -= vec[base+j]
				}
			}
			out[base+i] = sum * scale
		}
	}
	return out
}

func TestSignedBlockWHTConstructorRefusals(t *testing.T) {
	cases := []struct {
		name    string
		block   int
		widths  []int
		values  []int
		wantErr bool
	}{
		{"ok", 4, []int{8}, repSign(8), false},
		{"zero block", 0, []int{8}, repSign(8), true},
		{"non power of two block", 3, []int{6}, repSign(6), true},
		{"no widths", 4, nil, nil, true},
		{"non positive width", 4, []int{0}, nil, true},
		{"duplicate width", 4, []int{8, 8}, repSign(16), true},
		{"width not multiple of block", 4, []int{6}, repSign(6), true},
		{"bad sign value", 4, []int{8}, []int{1, 1, 1, 1, 1, 1, 1, 2}, true},
		{"short values", 4, []int{8}, repSign(7), true},
		{"long values", 4, []int{8}, repSign(9), true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := NewSignedBlockWHT(tc.block, tc.widths, tc.values)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected a typed refusal, got nil")
				}
				var typed ErrSignedBlockWHT
				if !asSignedErr(err, &typed) {
					t.Fatalf("expected ErrSignedBlockWHT, got %T: %v", err, err)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestSignedBlockWHTAggregateWidthOverflow(t *testing.T) {
	_, err := NewSignedBlockWHT(1, []int{1, int(^uint(0) >> 1)}, []int{1})
	var typed ErrSignedBlockWHT
	if !asSignedErr(err, &typed) || typed.Field != "sign" || typed.Value != 1 {
		t.Fatalf("overflowing aggregate widths must return the sign-length refusal, got %T: %v", err, err)
	}
}

func repSign(n int) []int {
	v := make([]int, n)
	for i := range v {
		if i%2 == 0 {
			v[i] = 1
		} else {
			v[i] = -1
		}
	}
	return v
}

func asSignedErr(err error, target *ErrSignedBlockWHT) bool {
	e, ok := err.(ErrSignedBlockWHT)
	if ok {
		*target = e
		return true
	}
	return false
}

func TestSignedBlockWHTRoundTrip(t *testing.T) {
	widths := []int{5120, 6144, 17408}
	wht, _ := signedWHTCase(t, PrismWHTBlock, widths)
	rng := rand.New(rand.NewSource(7))
	for _, w := range widths {
		x := make([]float32, w)
		for i := range x {
			x[i] = float32(rng.NormFloat64())
		}
		orig := append([]float32(nil), x...)
		if err := wht.Forward(x); err != nil {
			t.Fatalf("Forward(width=%d): %v", w, err)
		}
		if err := wht.Inverse(x); err != nil {
			t.Fatalf("Inverse(width=%d): %v", w, err)
		}
		for i := range x {
			if !almostEqual(x[i], orig[i], 2e-3, 2e-3) {
				t.Fatalf("width %d: Inverse(Forward(x))[%d] = %g, want %g", w, i, x[i], orig[i])
			}
		}
	}
}

func TestSignedBlockWHTForwardTwiceNotIdentity(t *testing.T) {
	widths := []int{5120}
	wht, _ := signedWHTCase(t, PrismWHTBlock, widths)
	rng := rand.New(rand.NewSource(11))
	x := make([]float32, widths[0])
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	orig := append([]float32(nil), x...)
	if err := wht.Forward(x); err != nil {
		t.Fatal(err)
	}
	if err := wht.Forward(x); err != nil {
		t.Fatal(err)
	}
	// With explicit signs, fwd(fwd(x)) != x.
	if vectorsClose(x, orig, 1e-3) {
		t.Fatalf("Forward(Forward(x)) unexpectedly equals x; the signed forward is not involutory")
	}
	// And the unsigned inverse alias does not undo the signed forward.
	y := append([]float32(nil), orig...)
	if err := wht.Forward(y); err != nil {
		t.Fatal(err)
	}
	if err := BlockDiagonalWHTInverse(y, PrismWHTBlock); err != nil {
		t.Fatal(err)
	}
	if vectorsClose(y, orig, 1e-3) {
		t.Fatalf("BlockDiagonalWHTInverse(signed Forward(x)) equals x; the unsigned alias must not undo the signed forward")
	}
}

func vectorsClose(a, b []float32, tol float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if !almostEqual(a[i], b[i], tol, tol) {
			return false
		}
	}
	return true
}

func TestSignedBlockWHTMatchesDenseOracle(t *testing.T) {
	widths := []int{5120, 6144, 17408}
	wht, signs := signedWHTCase(t, PrismWHTBlock, widths)
	rng := rand.New(rand.NewSource(19))
	for wi, w := range widths {
		x := make([]float64, w)
		xf := make([]float32, w)
		for i := 0; i < w; i++ {
			v := rng.NormFloat64()
			x[i] = v
			xf[i] = float32(v)
		}
		// Oracle: WHT_block(x ⊙ s), sign before the transform.
		signed := make([]float64, w)
		for i := range signed {
			signed[i] = x[i] * float64(signs[wi][i])
		}
		want := denseBlockWHT64(signed, PrismWHTBlock)
		if err := wht.Forward(xf); err != nil {
			t.Fatalf("Forward(width=%d): %v", w, err)
		}
		for i := range xf {
			if !almostEqual(xf[i], float32(want[i]), 1e-4, 1e-4) {
				t.Fatalf("width %d: Forward[%d] = %g, oracle = %g", w, i, xf[i], want[i])
			}
		}
	}
}

// --- fold / embedding / tied-head contract tests (small block, dense matrices) ---

func sylvester64(n int) [][]float64 {
	h := make([][]float64, n)
	for i := 0; i < n; i++ {
		h[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			if bits.OnesCount(uint(i&j))&1 == 0 {
				h[i][j] = 1
			} else {
				h[i][j] = -1
			}
		}
	}
	return h
}

// blockedH builds the dense block-diagonal normalized WHT of width `width` with `block`.
func blockedH(width, block int) [][]float64 {
	nb := width / block
	hb := sylvester64(block)
	scale := 1.0 / math.Sqrt(float64(block))
	m := make([][]float64, width)
	for k := range m {
		m[k] = make([]float64, width)
	}
	for b := 0; b < nb; b++ {
		for i := 0; i < block; i++ {
			for j := 0; j < block; j++ {
				m[b*block+i][b*block+j] = hb[i][j] * scale
			}
		}
	}
	return m
}

func matVec(m [][]float64, v []float64) []float64 {
	out := make([]float64, len(m))
	for i := range m {
		var s float64
		for j := range v {
			s += m[i][j] * v[j]
		}
		out[i] = s
	}
	return out
}

func TestSignedBlockWHTFoldAndTiedHeadContract(t *testing.T) {
	const block, width, outDim = 4, 8, 3
	signs := repSign(width)
	wht, err := NewSignedBlockWHT(block, []int{width}, signs)
	if err != nil {
		t.Fatal(err)
	}
	H := blockedH(width, block) // normalized block-diagonal WHT
	rng := rand.New(rand.NewSource(23))
	x := make([]float64, width)
	xf := make([]float32, width)
	for i := range x {
		v := rng.NormFloat64()
		x[i] = v
		xf[i] = float32(v)
	}
	W := make([][]float64, outDim)
	for o := range W {
		W[o] = make([]float64, width)
		for i := range W[o] {
			W[o][i] = rng.NormFloat64()
		}
	}
	// Wfolded = W · D · H : scale W's columns by s, then apply H on the right.
	WD := make([][]float64, outDim)
	for o := range WD {
		WD[o] = make([]float64, width)
		for i := range WD[o] {
			WD[o][i] = W[o][i] * float64(signs[i])
		}
	}
	want := matVec(W, x)

	fx := append([]float32(nil), xf...)
	if err := wht.Forward(fx); err != nil {
		t.Fatal(err)
	}
	fxd := make([]float64, width)
	for i := range fx {
		fxd[i] = float64(fx[i])
	}
	for o := 0; o < outDim; o++ {
		got := dotDenseWithBlockedH(WD[o], fxd, H) // (W·D) · H · Forward(x)
		if math.Abs(got-want[o]) > 1e-3 {
			t.Fatalf("folded projection[%d] = %g, want %g", o, got, want[o])
		}
	}
}

// dotDenseWithBlockedH computes (row · H) · forward implicitly: first apply H to `row`
// (row is length width), then dot with the already-forwarded vector.
func dotDenseWithBlockedH(row, forwardX []float64, H [][]float64) float64 {
	rowH := matVec(H, row) // H is symmetric, so row·H = H·row
	var s float64
	for i := range rowH {
		s += rowH[i] * forwardX[i]
	}
	return s
}

func TestSignedBlockWHTEmbeddingContract(t *testing.T) {
	const block, width = 4, 8
	signs := repSign(width)
	wht, err := NewSignedBlockWHT(block, []int{width}, signs)
	if err != nil {
		t.Fatal(err)
	}
	H := blockedH(width, block)
	rng := rand.New(rand.NewSource(29))
	e := make([]float64, width)
	for i := range e {
		e[i] = rng.NormFloat64()
	}
	// rotated = H · D · e (the folded embedding row), then Inverse restores e.
	de := make([]float64, width)
	for i := range de {
		de[i] = e[i] * float64(signs[i])
	}
	rotated := matVec(H, de)
	rf := make([]float32, width)
	for i := range rotated {
		rf[i] = float32(rotated[i])
	}
	if err := wht.Inverse(rf); err != nil {
		t.Fatal(err)
	}
	for i := range rf {
		if math.Abs(float64(rf[i])-e[i]) > 1e-3 {
			t.Fatalf("Inverse(H·D·e)[%d] = %g, want %g", i, rf[i], e[i])
		}
	}
}

func TestSignedBlockWHTSiblingSharesForward(t *testing.T) {
	const block, width, outDim = 4, 8, 2
	signs := repSign(width)
	wht, err := NewSignedBlockWHT(block, []int{width}, signs)
	if err != nil {
		t.Fatal(err)
	}
	H := blockedH(width, block)
	rng := rand.New(rand.NewSource(31))
	x := make([]float64, width)
	xf := make([]float32, width)
	for i := range x {
		v := rng.NormFloat64()
		x[i] = v
		xf[i] = float32(v)
	}
	// Two sibling folded projections of the same width share one Forward(x).
	if err := wht.Forward(xf); err != nil {
		t.Fatal(err)
	}
	fxd := make([]float64, width)
	for i := range xf {
		fxd[i] = float64(xf[i])
	}
	for sib := 0; sib < outDim; sib++ {
		W := make([]float64, width)
		for i := range W {
			W[i] = rng.NormFloat64()
		}
		WD := make([]float64, width)
		for i := range WD {
			WD[i] = W[i] * float64(signs[i])
		}
		got := dotDenseWithBlockedH(WD, fxd, H)
		var want float64
		for i := range W {
			want += W[i] * x[i]
		}
		if math.Abs(got-want) > 1e-3 {
			t.Fatalf("sibling %d: folded projection = %g, want %g", sib, got, want)
		}
	}
}

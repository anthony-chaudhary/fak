//go:build darwin && arm64 && cgo

package compute

import (
	"math"
	"slices"
	"testing"
)

// metalInk2TestBF16 uploads vals ([rows][cols], each exact in BF16) as a
// BF16 device tensor.
func metalInk2TestBF16(t *testing.T, be *metalBackend, vals []float32, rows, cols int) Tensor {
	t.Helper()
	words := make([]float32, (rows*cols+1)/2)
	for i, v := range vals {
		words[i/2] = math.Float32frombits(math.Float32bits(words[i/2]) | (math.Float32bits(v)>>16)<<(16*(i%2)))
	}
	store := metalInk2TestUpload(t, be, words)
	return makeTensor(be, BF16, RowMajor, []int{rows, cols}, nil, store.buf)
}

func metalInk2TestInt(v int) float32 { return math.Float32frombits(uint32(int32(v))) }

// fak-test:runtime integration est=1s lane=default
func TestMetalInkling2DraftConvThroughBackend(t *testing.T) {
	be := metalInk2TestBackend(t)
	const rows, hidden, group, kn, side = 3, 8, 4, 2, 1
	groups := hidden / group
	// Small integers keep every sum exact, so the check is bitwise.
	x, resid := make([]float32, rows*hidden), make([]float32, rows*hidden)
	for i := range x {
		x[i], resid[i] = float32(i%5)-2, float32(i%7)
	}
	dyn := make([]float32, rows*2*kn*groups)
	for i := range dyn {
		dyn[i] = float32(i%3) - 1
	}
	base := make([]float32, 2*kn*hidden)
	for i := range base {
		base[i] = float32(i%4) - 2
	}
	out := metalInk2TestUpload(t, be, resid)
	if err := be.Ink2DFConv(Ink2At(metalInk2TestUpload(t, be, x), 0), Ink2At(metalInk2TestUpload(t, be, dyn), 0),
		metalInk2TestBF16(t, be, base, 2*kn, hidden), Ink2At(out, 0), Ink2At(out, 0), rows, hidden, group, kn, side); err != nil {
		t.Fatal(err)
	}
	got, err := be.Ink2Read(Ink2At(out, 0), rows*hidden)
	if err != nil {
		t.Fatal(err)
	}
	for row := range rows {
		for c := range hidden {
			var y float32
			for tap := 0; tap < kn && tap <= row; tap++ {
				k := side*kn + tap
				y += (base[k*hidden+c] + dyn[row*2*kn*groups+k*groups+c/group]) * x[(row-tap)*hidden+c]
			}
			if want := resid[row*hidden+c] + y; got[row*hidden+c] != want {
				t.Fatalf("conv row %d channel %d = %v, want %v", row, c, got[row*hidden+c], want)
			}
		}
	}
}

// fak-test:runtime integration est=1s lane=default
func TestMetalInkling2DraftSelectThroughBackend(t *testing.T) {
	be := metalInk2TestBackend(t)
	const rows, vocab, rank, k, anchor = 3, 12, 4, 3, 7
	// Small integers keep every score exact; softcap 0 leaves them uncapped.
	logits := make([]float32, rows*vocab)
	for i := range logits {
		logits[i] = float32((i*7)%13) - 6
	}
	proj, pred, succ := make([]float32, rows*rank), make([]float32, vocab*rank), make([]float32, vocab*rank)
	for i := range proj {
		proj[i] = float32(i%3) - 1
	}
	for i := range pred {
		pred[i], succ[i] = float32(i%5)-2, float32((i*3)%4)-1
	}
	wantPath, wantCand := []int32{}, []int32{}
	predecessor := anchor
	for row := range rows {
		l := logits[row*vocab : (row+1)*vocab]
		var cand []int
		for range k {
			best := -1
			for id, v := range l {
				if !slices.Contains(cand, id) && (best < 0 || v > l[best]) {
					best = id
				}
			}
			cand = append(cand, best)
		}
		pick, pickScore := -1, float32(0)
		for _, c := range cand {
			score := l[c]
			for i := range rank {
				score += pred[predecessor*rank+i] * proj[row*rank+i] * succ[c*rank+i]
			}
			if pick < 0 || score > pickScore {
				pick, pickScore = c, score
			}
			wantCand = append(wantCand, int32(c))
		}
		wantPath, predecessor = append(wantPath, int32(pick)), pick
	}
	run := func(logits []float32) (path, cand []int32) {
		scratch := metalInk2TestAlloc(t, be, Ink2DFSelectScratch(rows, k))
		p, c := metalInk2TestAlloc(t, be, rows), metalInk2TestAlloc(t, be, rows*k)
		if err := be.Ink2DFSelect(Ink2At(metalInk2TestUpload(t, be, logits), 0), rows, vocab, vocab, k,
			Ink2At(metalInk2TestUpload(t, be, proj), 0), metalInk2TestBF16(t, be, pred, vocab, rank), metalInk2TestBF16(t, be, succ, vocab, rank),
			Ink2At(metalInk2TestUpload(t, be, []float32{metalInk2TestInt(anchor)}), 0), 0, Ink2At(scratch, 0), Ink2At(p, 0), Ink2At(c, 0)); err != nil {
			t.Fatal(err)
		}
		path, err := be.Ink2ReadInts(Ink2At(p, 0), rows)
		if err != nil {
			t.Fatal(err)
		}
		cand, err = be.Ink2ReadInts(Ink2At(c, 0), rows*k)
		if err != nil {
			t.Fatal(err)
		}
		return path, cand
	}
	if path, cand := run(logits); !slices.Equal(path, wantPath) || !slices.Equal(cand, wantCand) {
		t.Fatalf("select path %v candidates %v, want %v %v", path, cand, wantPath, wantCand)
	}
	// A NaN in row 1 rejects it and the row after; row 0 is unchanged.
	logits[vocab+4] = float32(math.NaN())
	path, cand := run(logits)
	if !slices.Equal(path, []int32{wantPath[0], -1, -1}) || !slices.Equal(cand[:k], wantCand[:k]) ||
		slices.ContainsFunc(cand[k:], func(v int32) bool { return v != -1 }) {
		t.Fatalf("select with a NaN in row 1: path %v candidates %v", path, cand)
	}
}

// fak-test:runtime integration est=1s lane=default
func TestMetalInkling2DraftOpsThroughBackend(t *testing.T) {
	be := metalInk2TestBackend(t)
	const rows, cols, tokens = 3, 8, 2
	// Small integers keep every sum exact, so the check is bitwise.
	w := make([]float32, rows*cols)
	for i := range w {
		w[i] = float32(i%3) - 1
	}
	bf := metalInk2TestBF16(t, be, w, rows, cols)
	x0 := make([]float32, tokens*cols/2)
	x1 := make([]float32, tokens*cols/2)
	for i := range x0 {
		x0[i], x1[i] = float32(i), float32(-2*i)
	}
	y := metalInk2TestAlloc(t, be, tokens*rows)
	if err := be.Ink2DFLinear([]Ink2Ref{Ink2At(metalInk2TestUpload(t, be, x0), 0), Ink2At(metalInk2TestUpload(t, be, x1), 0)},
		bf, Ink2At(y, 0), tokens); err != nil {
		t.Fatal(err)
	}
	got, err := be.Ink2Read(Ink2At(y, 0), tokens*rows)
	if err != nil {
		t.Fatal(err)
	}
	seg := cols / 2
	for tok := range tokens {
		for h := range rows {
			var want float32
			for k := range cols {
				x := x0
				if k >= seg {
					x = x1
				}
				want += w[h*cols+k] * x[tok*seg+k%seg]
			}
			if got[tok*rows+h] != want {
				t.Fatalf("linear token %d row %d = %v, want %v", tok, h, got[tok*rows+h], want)
			}
		}
	}

	// One block row at position 0 sees only its own key, so each head's output
	// is its kv head's value row.
	const heads, kvHeads, dim = 4, 2, 8
	q := make([]float32, heads*dim)
	v := make([]float32, kvHeads*dim)
	for i := range v {
		v[i] = float32(i) * .25
	}
	for i := range q {
		q[i] = float32(i%5) - 2
	}
	out := metalInk2TestAlloc(t, be, heads*dim)
	ring := metalInk2TestAlloc(t, be, 4*kvHeads*dim)
	pos := metalInk2TestUpload(t, be, []float32{math.Float32frombits(0)})
	if err := be.Ink2DFAttend(Ink2At(metalInk2TestUpload(t, be, q), 0), Ink2At(metalInk2TestUpload(t, be, v), 0),
		Ink2At(metalInk2TestUpload(t, be, v), 0), Ink2At(ring, 0), Ink2At(ring, 0), Ink2At(out, 0), Ink2At(pos, 0),
		1, heads, kvHeads, dim, 2048, 4); err != nil {
		t.Fatal(err)
	}
	att, err := be.Ink2Read(Ink2At(out, 0), heads*dim)
	if err != nil {
		t.Fatal(err)
	}
	for h := range heads {
		for e := range dim {
			if want := v[(h/(heads/kvHeads))*dim+e]; att[h*dim+e] != want {
				t.Fatalf("attend head %d dim %d = %v, want %v", h, e, att[h*dim+e], want)
			}
		}
	}
}

// The fused draft linears through the backend equal their separate ops.
//
// fak-test:runtime integration est=1s lane=default
func TestMetalInkling2DraftFusedLinearsThroughBackend(t *testing.T) {
	be := metalInk2TestBackend(t)
	const tokens, cols = 3, 64
	small := func(n, mod int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32((i*7)%mod) - float32(mod/2)
		}
		return v
	}
	x := Ink2At(metalInk2TestUpload(t, be, small(tokens*cols, 5)), 0)
	rows := []int{16, 8, 24}
	var ws []Tensor
	var sep, fused []Ink2Ref
	for i, r := range rows {
		ws = append(ws, metalInk2TestBF16(t, be, small(r*cols+i, 3+i)[i:], r, cols))
		a, b := Ink2At(metalInk2TestUpload(t, be, make([]float32, tokens*r)), 0), Ink2At(metalInk2TestUpload(t, be, make([]float32, tokens*r)), 0)
		if err := be.Ink2DFLinear([]Ink2Ref{x}, ws[i], a, tokens); err != nil {
			t.Fatal(err)
		}
		sep, fused = append(sep, a), append(fused, b)
	}
	if err := be.Ink2DFLinearMulti([]Ink2Ref{x}, ws, fused, tokens); err != nil {
		t.Fatal(err)
	}
	read := func(r Ink2Ref, n int) []float32 {
		v, err := be.Ink2Read(r, n)
		if err != nil {
			t.Fatal(err)
		}
		return v
	}
	for i, r := range rows {
		if !slices.Equal(read(fused[i], tokens*r), read(sep[i], tokens*r)) {
			t.Fatalf("multi output %d differs from Ink2DFLinear", i)
		}
	}
	up := metalInk2TestBF16(t, be, small(rows[0]*cols, 7), rows[0], cols)
	u := Ink2At(metalInk2TestUpload(t, be, make([]float32, tokens*rows[0])), 0)
	if err := be.Ink2DFLinear([]Ink2Ref{x}, up, u, tokens); err != nil {
		t.Fatal(err)
	}
	if err := be.Ink2DFSiluMul(sep[0], u, sep[0], tokens*rows[0]); err != nil {
		t.Fatal(err)
	}
	g := Ink2At(metalInk2TestUpload(t, be, make([]float32, tokens*rows[0])), 0)
	if err := be.Ink2DFLinearSiluMul(x, ws[0], up, g, Ink2Ref{}, tokens); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(read(g, tokens*rows[0]), read(sep[0], tokens*rows[0])) {
		t.Fatal("gated linear differs from Ink2DFLinear and Ink2DFSiluMul")
	}
}

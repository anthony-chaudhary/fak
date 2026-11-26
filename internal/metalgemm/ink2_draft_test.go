//go:build darwin && arm64 && cgo

package metalgemm

import (
	"math"
	"math/rand"
	"slices"
	"testing"
	"time"
)

// The host references below round every product before its add, which is
// what the kernels' FP_CONTRACT OFF evaluates; an explicit float32 conversion
// keeps Go from fusing them.

func ink2DFTestBF16Round(v float32) float32 {
	return math.Float32frombits(math.Float32bits(v) &^ 0xffff)
}

// ink2DFTestBF16 uploads vals ([rows][cols]) as a BF16 weight starting pad
// elements into its buffer, so an odd pad exercises the misaligned binding.
// It returns the weight and the values it holds.
func ink2DFTestBF16(t *testing.T, d *Ink2Device, vals []float32, rows, cols, pad int) (Ink2Weight, []float32) {
	t.Helper()
	held := make([]float32, len(vals))
	half := make([]uint16, pad+len(vals)+1)
	for i, v := range vals {
		held[i] = ink2DFTestBF16Round(v)
		half[pad+i] = uint16(math.Float32bits(held[i]) >> 16)
	}
	words := make([]float32, (len(half)+1)/2)
	for i := range words {
		lo := uint32(half[2*i])
		var hi uint32
		if 2*i+1 < len(half) {
			hi = uint32(half[2*i+1])
		}
		words[i] = math.Float32frombits(lo | hi<<16)
	}
	ref := ink2TestAlloc(t, d, len(words))
	if err := d.Write(ref, words); err != nil {
		t.Fatal(err)
	}
	return Ink2Weight{Buf: ref.Buf, ByteOff: 2 * pad, Type: 30, Rows: rows, Cols: cols, Experts: 1}, held
}

func ink2DFTestInts(t *testing.T, d *Ink2Device, r Ink2Ref, n int) []int32 {
	t.Helper()
	got, err := d.ReadInts(r, n)
	if err != nil {
		t.Fatal(err)
	}
	return got
}

func ink2DFTestLinearHost(x [][]float32, w []float32, outRows, cols, tokens int) (y []float32, mag []float64) {
	seg := cols / len(x)
	y, mag = make([]float32, tokens*outRows), make([]float64, tokens*outRows)
	for t := range tokens {
		for h := range outRows {
			var sum float32
			var m float64
			for k := range cols {
				p := float32(w[h*cols+k] * x[k/seg][t*seg+k%seg])
				sum += p
				m += math.Abs(float64(p))
			}
			y[t*outRows+h], mag[t*outRows+h] = sum, m
		}
	}
	return y, mag
}

// fak-test:runtime integration est=2s lane=default
func TestInk2DraftLinearMatchesHost(t *testing.T) {
	d := ink2TestDevice(t)
	rng := rand.New(rand.NewSource(11))
	// xoff and pad multiples of 4 and 8 with seg a multiple of 8 take the
	// vector kernel; the rest the scalar one.
	for _, tc := range []struct{ segs, seg, outRows, tokens, pad, xoff int }{
		{4, 64, 48, 8, 1, 5}, // the fc tap fuse: four target taps side by side
		{1, 96, 37, 3, 0, 5},
		{2, 40, 5, 1, 3, 5},
		{4, 64, 48, 8, 0, 4},
		{1, 2304, 7, 8, 8, 0}, // nine SIMD groups split the columns; an odd last row pair
		{2, 256, 9, 5, 0, 8},
		{1, 96, 37, 3, 0, 0},
		{3, 120, 4, 2, 0, 4},
		{1, 40, 3, 1, 0, 0},
	} {
		cols := tc.segs * tc.seg
		w, held := ink2DFTestBF16(t, d, ink2ConvTestRand(rng, tc.outRows*cols, 1), tc.outRows, cols, tc.pad)
		var xs [][]float32
		var refs []Ink2Ref
		for range tc.segs {
			v := ink2ConvTestRand(rng, tc.tokens*tc.seg, 2)
			xs, refs = append(xs, v), append(refs, ink2ConvTestPut(t, d, tc.xoff, v))
		}
		y := ink2TestAlloc(t, d, tc.tokens*tc.outRows+8).At(3)
		if err := d.DraftLinear(refs, w, y, tc.tokens); err != nil {
			t.Fatal(err)
		}
		want, mag := ink2DFTestLinearHost(xs, held, tc.outRows, cols, tc.tokens)
		var drift ink2ConvTestDrift
		drift.parity(t, "draft linear", ink2ConvTestRead(t, d, y, len(want)), want, mag)
		t.Logf("segs %d x %d -> %d, %d tokens, pad %d, offset %d: %d of %d inexact, worst %.3f of the bound", tc.segs, tc.seg, tc.outRows, tc.tokens, tc.pad, tc.xoff,
			drift.inexact, len(want), drift.worst)
	}
}

func ink2DFTestConvHost(x, dyn, base, resid []float32, rows, hidden, group, kn, side int) []float32 {
	groups, stride := hidden/group, 2*kn*(hidden/group)
	out := make([]float32, rows*hidden)
	for row := range rows {
		for c := range hidden {
			var y float32
			for tap := 0; tap < kn && tap <= row; tap++ {
				v := x[(row-tap)*hidden+c]
				k := side*kn + tap
				y += float32(base[k*hidden+c] * v)
				y += float32(dyn[row*stride+k*groups+c/group] * v)
			}
			if resid != nil {
				y = resid[row*hidden+c] + y
			}
			out[row*hidden+c] = y
		}
	}
	return out
}

// fak-test:runtime integration est=1s lane=default
func TestInk2DraftConvBitEqualHost(t *testing.T) {
	d := ink2TestDevice(t)
	rng := rand.New(rand.NewSource(12))
	const hidden, group, kn = 96, 16, 2
	for _, rows := range []int{8, 1, 5} {
		for side := range 2 {
			for _, residual := range []bool{false, true} {
				x := ink2ConvTestRand(rng, rows*hidden, 3)
				dyn := ink2ConvTestRand(rng, rows*2*kn*hidden/group, 1)
				base, held := ink2DFTestBF16(t, d, ink2ConvTestRand(rng, 2*kn*hidden, 1), 2*kn, hidden, 1+side)
				var resid []float32
				out := ink2TestAlloc(t, d, rows*hidden)
				rref := Ink2Ref{}
				if residual {
					// In place, as the block's residual add runs.
					resid = ink2ConvTestRand(rng, rows*hidden, 4)
					rref = out
					if err := d.Write(out, resid); err != nil {
						t.Fatal(err)
					}
				}
				if err := d.DraftConv(ink2ConvTestPut(t, d, 3, x), ink2ConvTestPut(t, d, 2, dyn), base, rref, out, rows, hidden, group, kn, side); err != nil {
					t.Fatal(err)
				}
				ink2ConvTestSame(t, "draft conv", ink2ConvTestRead(t, d, out, rows*hidden),
					ink2DFTestConvHost(x, dyn, held, resid, rows, hidden, group, kn, side))
			}
		}
	}
}

// ink2DFTestAttendHost is the host reference's draft attention: the stored
// context is the last positions the ring and window keep, masked per row
// where the row is window or more past them.
func ink2DFTestAttendHost(q, kb, vb, kc, vc []float32, pos, rows, heads, kvHeads, dim, window, cap int) (out []float32, mag []float64) {
	kvw := kvHeads * dim
	keep := min(pos, cap)
	if window > 0 {
		keep = min(keep, window-1)
	}
	first := pos - keep
	out, mag = make([]float32, rows*heads*dim), make([]float64, rows*heads*dim)
	scale := float32(1 / math.Sqrt(float64(dim)))
	scores := make([]float32, keep+rows)
	key := func(k int, v bool) []float32 {
		src, at := kb, (k-keep)*kvw
		if k < keep {
			src, at = kc, ((first+k)%cap)*kvw
			if v {
				src = vc
			}
		} else if v {
			src = vb
		}
		return src[at : at+kvw]
	}
	for row := range rows {
		for h := range heads {
			kvh, qs := h/(heads/kvHeads), (row*heads+h)*dim
			maximum := float32(math.Inf(-1))
			for k := range scores {
				if k < keep && window > 0 && keep+row-k >= window {
					scores[k] = float32(math.Inf(-1))
					continue
				}
				kr := key(k, false)[kvh*dim:]
				var sum float32
				for e := range dim {
					sum += float32(q[qs+e] * kr[e])
				}
				scores[k] = sum * scale
				maximum = max(maximum, scores[k])
			}
			var den float32
			for k := range scores {
				scores[k] = float32(math.Exp(float64(scores[k] - maximum)))
				den += scores[k]
			}
			for k, s := range scores {
				vr := key(k, true)[kvh*dim:]
				for e := range dim {
					out[qs+e] += float32((s / den) * vr[e])
					mag[qs+e] += math.Abs(float64(s/den) * float64(vr[e]))
				}
			}
		}
	}
	return out, mag
}

// fak-test:runtime integration est=2s lane=default
func TestInk2DraftAttendMatchesHost(t *testing.T) {
	d := ink2TestDevice(t)
	rng := rand.New(rand.NewSource(13))
	const heads, kvHeads, dim = 8, 2, 32
	for _, tc := range []struct{ rows, pos, window, cap int }{
		{8, 0, 16, 32}, {8, 5, 16, 32}, {8, 20, 16, 32}, {8, 75, 16, 32}, {3, 40, 0, 24}, {8, 300, 2048, 512}, {8, 2100, 2048, 2048},
	} {
		kvw := kvHeads * dim
		q := ink2ConvTestRand(rng, tc.rows*heads*dim, 1)
		kb, vb := ink2ConvTestRand(rng, tc.rows*kvw, 1), ink2ConvTestRand(rng, tc.rows*kvw, 1)
		kc, vc := ink2ConvTestRand(rng, tc.cap*kvw, 1), ink2ConvTestRand(rng, tc.cap*kvw, 1)
		out := ink2TestAlloc(t, d, tc.rows*heads*dim)
		if err := d.DraftAttend(ink2ConvTestPut(t, d, 1, q), ink2ConvTestPut(t, d, 2, kb), ink2ConvTestPut(t, d, 3, vb),
			ink2ConvTestPut(t, d, 0, kc), ink2ConvTestPut(t, d, 4, vc), out, ink2TestAttendUpload(t, d, ink2TestAttendMeta(tc.pos)),
			tc.rows, heads, kvHeads, dim, tc.window, tc.cap); err != nil {
			t.Fatal(err)
		}
		want, mag := ink2DFTestAttendHost(q, kb, vb, kc, vc, tc.pos, tc.rows, heads, kvHeads, dim, tc.window, tc.cap)
		var drift ink2ConvTestDrift
		drift.parity(t, "draft attend", ink2ConvTestRead(t, d, out, len(want)), want, mag)
		t.Logf("pos %d window %d cap %d: %d of %d inexact, worst %.3f of the bound", tc.pos, tc.window, tc.cap, drift.inexact, len(want), drift.worst)
	}
}

// ink2DFTestSelectHost is the reference selector: an insertion top-k (lower
// id first on equal logits) and the strict-improvement argmax of the capped
// edge scores.
func ink2DFTestSelectHost(logits []float32, rows, stride, vocab, k int, proj, pred, succ []float32, rank, anchor int, softcap float32) (path []int32, cands [][]int) {
	predecessor := anchor
	for row := range rows {
		l := logits[row*stride : row*stride+vocab]
		ids := make([]int, vocab)
		for i := range ids {
			ids[i] = i
		}
		slices.SortStableFunc(ids, func(a, b int) int {
			switch {
			case l[a] > l[b]:
				return -1
			case l[a] < l[b]:
				return 1
			}
			return 0
		})
		cand := ids[:k]
		best, bestScore := -1, float32(0)
		for _, c := range cand {
			score := l[c]
			for i := range rank {
				score += float32(float32(pred[predecessor*rank+i]*proj[row*rank+i]) * succ[c*rank+i])
			}
			if softcap > 0 {
				score = softcap * float32(math.Tanh(float64(score/softcap)))
			}
			if best < 0 || score > bestScore {
				best, bestScore = c, score
			}
		}
		path, cands = append(path, int32(best)), append(cands, slices.Clone(cand))
		predecessor = best
	}
	return path, cands
}

func ink2DFTestSelect(t *testing.T, d *Ink2Device, rng *rand.Rand, rows, vocab, stride, rank, k int, softcap float32, ties bool) {
	t.Helper()
	logits := ink2ConvTestRand(rng, rows*stride, 8)
	if ties {
		// Few distinct values, so equal logits straddle the top-k boundary.
		for i := range logits {
			logits[i] = float32(rng.Intn(6))
		}
	}
	proj := ink2ConvTestRand(rng, rows*rank, 1)
	pred, predHeld := ink2DFTestBF16(t, d, ink2ConvTestRand(rng, vocab*rank, 1), vocab, rank, 1)
	succ, succHeld := ink2DFTestBF16(t, d, ink2ConvTestRand(rng, vocab*rank, 1), vocab, rank, 0)
	anchor := rng.Intn(vocab)
	scratch := ink2TestAlloc(t, d, DraftSelectScratch(rows, k))
	path, cand := ink2TestAlloc(t, d, rows), ink2TestAlloc(t, d, rows*k)
	if err := d.DraftSelect(ink2ConvTestPut(t, d, 2, logits), rows, stride, vocab, k, ink2ConvTestPut(t, d, 1, proj), pred, succ,
		ink2TestAttendUpload(t, d, ink2TestAttendMeta(anchor)), softcap, scratch, path, cand); err != nil {
		t.Fatal(err)
	}
	wantPath, wantCand := ink2DFTestSelectHost(logits, rows, stride, vocab, k, proj, predHeld, succHeld, rank, anchor, softcap)
	gotCand := ink2DFTestInts(t, d, cand, rows*k)
	for row := range rows {
		for j := range k {
			if int(gotCand[row*k+j]) != wantCand[row][j] {
				t.Fatalf("row %d candidates = %v, want %v", row, gotCand[row*k:(row+1)*k], wantCand[row])
			}
		}
	}
	if got := ink2DFTestInts(t, d, path, rows); !slices.Equal(got, wantPath) {
		t.Fatalf("path = %v, want %v", got, wantPath)
	}
}

// fak-test:runtime integration est=2s lane=default
func TestInk2DraftSelectMatchesHost(t *testing.T) {
	d := ink2TestDevice(t)
	rng := rand.New(rand.NewSource(14))
	ink2DFTestSelect(t, d, rng, 7, 5000, 5003, 64, 16, 30, false)
	ink2DFTestSelect(t, d, rng, 7, 3000, 3000, 32, 16, 30, true)
	ink2DFTestSelect(t, d, rng, 3, 700, 701, 16, 5, 0, true)
	ink2DFTestSelect(t, d, rng, 1, 40, 40, 8, 16, 30, false)
}

// fak-test:runtime integration est=1s lane=default
func TestInk2DraftSelectRejectsNonFinite(t *testing.T) {
	d := ink2TestDevice(t)
	const rows, vocab, rank, k = 4, 700, 16, 8
	for _, tc := range []struct {
		name     string
		row, col int
		bad      float32
	}{
		{"NaN", 2, 333, float32(math.NaN())},
		{"+Inf", 0, 12, float32(math.Inf(1))},
		{"-Inf", 3, 699, float32(math.Inf(-1))},
	} {
		rng := rand.New(rand.NewSource(16))
		logits := ink2ConvTestRand(rng, rows*vocab, 8)
		proj := ink2ConvTestRand(rng, rows*rank, 1)
		pred, predHeld := ink2DFTestBF16(t, d, ink2ConvTestRand(rng, vocab*rank, 1), vocab, rank, 0)
		succ, succHeld := ink2DFTestBF16(t, d, ink2ConvTestRand(rng, vocab*rank, 1), vocab, rank, 0)
		wantPath, wantCand := ink2DFTestSelectHost(logits, rows, vocab, vocab, k, proj, predHeld, succHeld, rank, 5, 30)
		logits[tc.row*vocab+tc.col] = tc.bad
		scratch := ink2TestAlloc(t, d, DraftSelectScratch(rows, k))
		path, cand := ink2TestAlloc(t, d, rows), ink2TestAlloc(t, d, rows*k)
		if err := d.DraftSelect(ink2ConvTestPut(t, d, 0, logits), rows, vocab, vocab, k, ink2ConvTestPut(t, d, 0, proj), pred, succ,
			ink2TestAttendUpload(t, d, ink2TestAttendMeta(5)), 30, scratch, path, cand); err != nil {
			t.Fatal(err)
		}
		gotPath, gotCand := ink2DFTestInts(t, d, path, rows), ink2DFTestInts(t, d, cand, rows*k)
		for row := range rows {
			for j := range k {
				want := int32(-1)
				if row < tc.row {
					want = int32(wantCand[row][j])
				}
				if gotCand[row*k+j] != want {
					t.Fatalf("%s at row %d: row %d candidates = %v", tc.name, tc.row, row, gotCand[row*k:(row+1)*k])
				}
			}
			want := int32(-1)
			if row < tc.row {
				want = wantPath[row]
			}
			if gotPath[row] != want {
				t.Fatalf("%s at row %d: path = %v, want the host's %v up to it then -1", tc.name, tc.row, gotPath, wantPath)
			}
		}
	}
}

// ink2DFTestGPU runs encode once per repetition and returns the least GPU time
// of its submitted command buffer.
func ink2DFTestGPU(t *testing.T, d *Ink2Device, encode func() error) time.Duration {
	t.Helper()
	best := math.Inf(1)
	for range 5 {
		before := d.Stats().GPUSeconds
		if err := encode(); err != nil {
			t.Fatal(err)
		}
		if err := d.Flush(); err != nil {
			t.Fatal(err)
		}
		best = min(best, d.Stats().GPUSeconds-before)
	}
	return time.Duration(best * float64(time.Second))
}

// TestInk2DraftBlockGPUTime times each draft op once at the shipped DFlash2
// geometry for one block of 8 rows, and holds the selector to the host there.
//
// fak-test:runtime integration est=10s lane=default
func TestInk2DraftBlockGPUTime(t *testing.T) {
	if testing.Short() {
		t.Skip("allocates the full-vocabulary codebooks")
	}
	d := ink2TestDevice(t)
	rng := rand.New(rand.NewSource(15))
	const (
		rows, hidden, inter, heads, kvHeads, dim = 8, 2304, 6912, 32, 8, 128
		group, kn, window, rank, topK, vocab     = 16, 2, 2048, 256, 16, 201024
		softcap                                  = 30
	)
	taps := make([]Ink2Ref, 4)
	for i := range taps {
		taps[i] = ink2ConvTestPut(t, d, 0, ink2ConvTestRand(rng, rows*hidden, 1))
	}
	fc, _ := ink2DFTestBF16(t, d, ink2ConvTestRand(rng, hidden*4*hidden, .05), hidden, 4*hidden, 0)
	fused := ink2TestAlloc(t, d, rows*hidden)
	fuse := ink2DFTestGPU(t, d, func() error { return d.DraftLinear(taps, fc, fused, rows) })

	gate, _ := ink2DFTestBF16(t, d, ink2ConvTestRand(rng, inter*hidden, .05), inter, hidden, 0)
	gateOut := ink2TestAlloc(t, d, rows*inter)
	gateTime := ink2DFTestGPU(t, d, func() error { return d.DraftLinear(taps[:1], gate, gateOut, rows) })

	dyn := ink2ConvTestPut(t, d, 0, ink2ConvTestRand(rng, rows*2*kn*hidden/group, 1))
	base, _ := ink2DFTestBF16(t, d, ink2ConvTestRand(rng, 2*kn*hidden, 1), 2*kn, hidden, 0)
	convOut := ink2TestAlloc(t, d, rows*hidden)
	conv := ink2DFTestGPU(t, d, func() error { return d.DraftConv(taps[0], dyn, base, taps[1], convOut, rows, hidden, group, kn, 0) })

	kvw, pos := kvHeads*dim, 5000
	q := ink2ConvTestPut(t, d, 0, ink2ConvTestRand(rng, rows*heads*dim, 1))
	kb, vb := ink2ConvTestPut(t, d, 0, ink2ConvTestRand(rng, rows*kvw, 1)), ink2ConvTestPut(t, d, 0, ink2ConvTestRand(rng, rows*kvw, 1))
	kc, vc := ink2ConvTestPut(t, d, 0, ink2ConvTestRand(rng, window*kvw, 1)), ink2ConvTestPut(t, d, 0, ink2ConvTestRand(rng, window*kvw, 1))
	attOut := ink2TestAlloc(t, d, rows*heads*dim)
	posDev := ink2TestAttendUpload(t, d, ink2TestAttendMeta(pos))
	attend := ink2DFTestGPU(t, d, func() error {
		return d.DraftAttend(q, kb, vb, kc, vc, attOut, posDev, rows, heads, kvHeads, dim, window, window)
	})

	props := rows - 1
	logits := ink2ConvTestRand(rng, props*vocab, 8)
	proj := ink2ConvTestRand(rng, props*rank, 1)
	pred, predHeld := ink2DFTestBF16(t, d, ink2ConvTestRand(rng, vocab*rank, 1), vocab, rank, 0)
	succ, succHeld := ink2DFTestBF16(t, d, ink2ConvTestRand(rng, vocab*rank, 1), vocab, rank, 0)
	lref, pref := ink2ConvTestPut(t, d, 0, logits), ink2ConvTestPut(t, d, 0, proj)
	anchor := ink2TestAttendUpload(t, d, ink2TestAttendMeta(1234))
	scratch, path := ink2TestAlloc(t, d, DraftSelectScratch(props, topK)), ink2TestAlloc(t, d, props)
	sel := ink2DFTestGPU(t, d, func() error {
		return d.DraftSelect(lref, props, vocab, vocab, topK, pref, pred, succ, anchor, softcap, scratch, path, Ink2Ref{})
	})
	wantPath, _ := ink2DFTestSelectHost(logits, props, vocab, vocab, topK, proj, predHeld, succHeld, rank, 1234, softcap)
	if got := ink2DFTestInts(t, d, path, props); !slices.Equal(got, wantPath) {
		t.Fatalf("full-vocabulary path = %v, want %v", got, wantPath)
	}
	t.Logf("GPU per op, block of %d: fc fuse %dx%d %v (gate %dx%d %v), conv %v, attend %d ctx %v, select %d rows x %d vocab %v",
		rows, hidden, 4*hidden, fuse, inter, hidden, gateTime, conv, window-1, attend, props, vocab, sel)
}

// The fused draft linears write exactly what their separate launches write,
// on the vector kernels and on the fallback a misaligned weight takes.
//
// fak-test:runtime integration est=3s lane=default
func TestInk2DraftFusedLinearsMatchSeparate(t *testing.T) {
	d := ink2TestDevice(t)
	rng := rand.New(rand.NewSource(17))
	const cols = 512
	for _, pad := range []int{0, 1} {
		for _, tokens := range []int{1, 3, 8} {
			x := ink2ConvTestPut(t, d, 0, ink2ConvTestRand(rng, tokens*cols, 1))
			shapes := []int{70, 16, 33}
			var ws []Ink2Weight
			var sep, fused []Ink2Ref
			for _, r := range shapes {
				w, _ := ink2DFTestBF16(t, d, ink2ConvTestRand(rng, r*cols, .05), r, cols, 8*pad)
				if pad == 1 {
					w, _ = ink2DFTestBF16(t, d, ink2ConvTestRand(rng, r*cols, .05), r, cols, 3)
				}
				ws = append(ws, w)
				a, b := ink2TestAlloc(t, d, tokens*r), ink2TestAlloc(t, d, tokens*r)
				if err := d.DraftLinear([]Ink2Ref{x}, w, a, tokens); err != nil {
					t.Fatal(err)
				}
				sep, fused = append(sep, a), append(fused, b)
			}
			for n := 1; n <= 3; n++ {
				if err := d.DraftLinearMulti([]Ink2Ref{x}, ws[:n], fused[:n], tokens); err != nil {
					t.Fatal(err)
				}
				for i := range n {
					want, got := ink2ConvTestRead(t, d, sep[i], tokens*shapes[i]), ink2ConvTestRead(t, d, fused[i], tokens*shapes[i])
					if !slices.Equal(got, want) {
						t.Fatalf("pad %d, %d tokens, %d weights: output %d differs from DraftLinear", pad, tokens, n, i)
					}
				}
			}
			up, _ := ink2DFTestBF16(t, d, ink2ConvTestRand(rng, shapes[0]*cols, .05), shapes[0], cols, 8*pad)
			u, g := ink2TestAlloc(t, d, tokens*shapes[0]), ink2TestAlloc(t, d, tokens*shapes[0])
			if err := d.DraftLinear([]Ink2Ref{x}, up, u, tokens); err != nil {
				t.Fatal(err)
			}
			if err := d.DraftSiluMul(sep[0], u, sep[0], tokens*shapes[0]); err != nil {
				t.Fatal(err)
			}
			scratch := ink2TestAlloc(t, d, tokens*shapes[0])
			if err := d.DraftLinearSiluMul(x, ws[0], up, g, scratch, tokens); err != nil {
				t.Fatal(err)
			}
			if want, got := ink2ConvTestRead(t, d, sep[0], tokens*shapes[0]), ink2ConvTestRead(t, d, g, tokens*shapes[0]); !slices.Equal(got, want) {
				t.Fatalf("pad %d, %d tokens: gated linear differs from DraftLinear and DraftSiluMul", pad, tokens)
			}
		}
	}
}

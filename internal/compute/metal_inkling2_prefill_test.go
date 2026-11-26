//go:build darwin && arm64 && cgo

package compute

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// The batched ops are held to the decode ops of the same backend, row by row
// and bit for bit: the decode ops carry the host parity.

type metalInk2PTest struct {
	t   *testing.T
	be  *metalBackend
	rng *rand.Rand
}

func (h metalInk2PTest) alloc(n int) Ink2Ref {
	h.t.Helper()
	return Ink2At(metalInk2TestAlloc(h.t, h.be, n), 0)
}

func (h metalInk2PTest) put(v []float32) Ink2Ref {
	h.t.Helper()
	return Ink2At(metalInk2TestUpload(h.t, h.be, v), 0)
}

func (h metalInk2PTest) rand(n int, scale float32) []float32 {
	v := metalInk2TestRand(h.rng, n)
	for i := range v {
		v[i] *= scale
	}
	return v
}

func (h metalInk2PTest) read(r Ink2Ref, n int) []float32 {
	h.t.Helper()
	got, err := h.be.Ink2Read(r, n)
	if err != nil {
		h.t.Fatal(err)
	}
	return got
}

func (h metalInk2PTest) ints(vals []int32) Ink2Ref {
	h.t.Helper()
	bits := make([]float32, len(vals))
	for i, v := range vals {
		bits[i] = math.Float32frombits(uint32(v))
	}
	return h.put(bits)
}

func (h metalInk2PTest) must(err error) {
	h.t.Helper()
	if err != nil {
		h.t.Fatal(err)
	}
}

func (h metalInk2PTest) same(what string, got, want []float32) {
	h.t.Helper()
	if len(got) != len(want) {
		h.t.Fatalf("%s: %d values, want %d", what, len(got), len(want))
	}
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			h.t.Fatalf("%s[%d] = %v (%#x), decode op %v (%#x)", what, i, got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
		}
	}
}

func off(r Ink2Ref, n int) Ink2Ref { return Ink2At(r.T, r.Off+n) }

// fak-test:runtime integration est=3s lane=default
func TestMetalInkling2PrefillRowsEqualDecodeOps(t *testing.T) {
	be := metalInk2TestBackend(t)
	var pd Inkling2PrefillDevice = be
	h := metalInk2PTest{t: t, be: be, rng: rand.New(rand.NewSource(91))}
	const n, T, rows, rank, kn, heads = 96, 7, 10, 6, 4, 4

	// Dense matmul against the decode GEMV.
	wt := metalInk2TestUpload(t, be, h.rand(rows*n, 1))
	wt.Shape = []int{rows, n}
	x := h.put(h.rand(T*n, 1))
	y, yRow := h.alloc(T*rows), h.alloc(T*rows)
	h.must(pd.Ink2PMatMul(wt, x, y, T))
	for tok := range T {
		h.must(be.Ink2MatVec(wt, off(x, tok*n), off(yRow, tok*rows)))
	}
	h.same("matmul", h.read(y, T*rows), h.read(yRow, T*rows))

	// Depth mix: query-major [nq][T][n] against [nq][n] per row.
	const nq = 3
	states := []Ink2Ref{x, h.put(h.rand(T*n, 3)), h.put(h.rand(T*n, .2))}
	queries := h.put(h.rand(nq*n, 1.5))
	mixed, mixRow := h.alloc(nq*T*n), h.alloc(nq*n)
	h.must(pd.Ink2PMix(states, queries, nq, mixed, n, T))
	gotMix := h.read(mixed, nq*T*n)
	for tok := range T {
		row := make([]Ink2Ref, len(states))
		for s := range row {
			row[s] = off(states[s], tok*n)
		}
		h.must(be.Ink2Mix(row, queries, nq, nil, 0, mixRow, n))
		want := h.read(mixRow, nq*n)
		for q := range nq {
			h.same(fmt.Sprintf("mix query %d row %d", q, tok), gotMix[(q*T+tok)*n:][:n], want[q*n:][:n])
		}
	}

	// One controller, down then apply, against the fused decode control.
	msg := h.put(h.rand(T*n, .5))
	down := h.put(h.rand(rank*2*n, float32(1/math.Sqrt(2*n))))
	gain, forget := h.put(h.rand(n*rank, .4)), h.put(h.rand(n*rank, .4))
	g, ctl := h.alloc(T*rank), h.alloc(T*n)
	gRow, ctlRow := h.alloc(rank), h.alloc(n)
	h.must(pd.Ink2PCtrlDown(x, msg, down, g, n, rank, T))
	h.must(pd.Ink2PCtrlApply(x, msg, g, gain, forget, ctl, n, rank, T))
	gotCtl := h.read(ctl, T*n)
	for tok := range T {
		c := Ink2Controls{Cur: []Ink2Ref{off(x, tok*n)}, Msg: []Ink2Ref{off(msg, tok*n)},
			Down: []Ink2Ref{down}, Gain: []Ink2Ref{gain}, Forget: []Ink2Ref{forget}}
		h.must(be.Ink2Control(c, gRow, ctlRow, n, rank, nil, 0))
		h.same(fmt.Sprintf("control row %d", tok), gotCtl[tok*n:][:n], h.read(ctlRow, n))
	}

	// Short conv with dynamic taps and a residual, and the history it leaves.
	convW, dyn := h.put(h.rand(n*kn, .5)), h.put(h.rand(T*heads*kn, .5))
	hist0 := h.rand(n*(kn-1), 1)
	hist, histRow := h.put(hist0), h.put(hist0)
	conv, convRow := h.alloc(T*n), h.alloc(T*n)
	norm := float32(1 / math.Sqrt(2*kn+1))
	h.must(pd.Ink2PConv(x, convW, dyn, hist, conv, msg, n, heads, kn, norm, T))
	for tok := range T {
		h.must(be.Ink2Conv(off(x, tok*n), convW, off(dyn, tok*heads*kn), histRow, off(convRow, tok*n), off(msg, tok*n), n, heads, kn, norm))
	}
	h.same("conv", h.read(conv, T*n), h.read(convRow, T*n))
	h.same("conv history", h.read(hist, n*(kn-1)), h.read(histRow, n*(kn-1)))

	// Routing and the weighted combine.
	const nexp, k = 40, 3
	logits, bias := h.put(h.rand(T*nexp, 4)), h.put(h.rand(nexp, .2))
	ids, wts := h.alloc(T*k), h.alloc(T*k)
	idRow, wtRow := h.alloc(k), h.alloc(k)
	h.must(pd.Ink2PRoute(logits, bias, ids, wts, nexp, k, 2, T))
	gotIDs, err := be.Ink2ReadInts(ids, T*k)
	h.must(err)
	gotWts := h.read(wts, T*k)
	slots := h.put(h.rand(T*k*n, 1))
	combined, combRow := h.alloc(T*n), h.alloc(n)
	h.must(pd.Ink2PCombine(slots, wts, combined, k, n, T))
	gotComb := h.read(combined, T*n)
	for tok := range T {
		h.must(be.Ink2Route(off(logits, tok*nexp), bias, idRow, wtRow, nexp, k, 2))
		want, err := be.Ink2ReadInts(idRow, k)
		h.must(err)
		for s := range k {
			if gotIDs[tok*k+s] != want[s] {
				t.Fatalf("route row %d slot %d = %d, decode op %d", tok, s, gotIDs[tok*k+s], want[s])
			}
		}
		h.same(fmt.Sprintf("route weights row %d", tok), gotWts[tok*k:][:k], h.read(wtRow, k))
		h.must(be.Ink2Combine(off(slots, tok*k*n), off(wts, tok*k), combRow, k, n))
		h.same(fmt.Sprintf("combine row %d", tok), gotComb[tok*n:][:n], h.read(combRow, n))
	}
}

// fak-test:runtime integration est=3s lane=default
func TestMetalInkling2PrefillAttentionEqualsDecodeOps(t *testing.T) {
	be := metalInk2TestBackend(t)
	var pd Inkling2PrefillDevice = be
	h := metalInk2PTest{t: t, be: be, rng: rand.New(rand.NewSource(92))}
	for _, c := range []struct {
		name                             string
		global                           bool
		cap, window, extent, pos0, T, rl int
	}{
		{"global from the first position", true, 64, 0, 24, 0, 9, 5},
		{"global mid prompt", true, 64, 0, 24, 30, 11, 5},
		// the ring keeps the chunk's room past the window, as the driver sizes it
		{"local ring wrapped", false, 16 + 8, 16, 16, 37, 8, 5},
		{"no relative states", true, 32, 0, 0, 3, 6, 0},
	} {
		t.Run(c.name, func(t *testing.T) {
			h.t = t
			p := Ink2AttnParams{Heads: 6, KVHeads: 2, HeadDim: 20, RelLen: c.rl, QKClip: 1.2, RelClip: .9, Eps: 1e-6,
				Global: c.global, Alpha: .3, Floor: 4, Scale: float32(1 / math.Sqrt(20))}
			qn, rn, kvDim := p.Heads*p.HeadDim, p.Heads*p.RelLen, p.KVHeads*p.HeadDim
			q0, r0 := h.rand(c.T*qn, 1), h.rand(max(c.T*rn, 1), 1)
			k, v := h.put(h.rand(c.T*kvDim, 1)), h.put(h.rand(c.T*kvDim, 1))
			qclip, rclip := h.put(h.rand(qn, 2)), h.put(h.rand(max(rn, 1), 2))
			kc0, vc0 := h.rand(c.cap*kvDim, .5), h.rand(c.cap*kvDim, 1)
			sinks, rel := h.put(h.rand(c.T*p.Heads, 1)), h.put(h.rand(max(c.extent*p.RelLen, 1), .1))
			q, r, kc, vc, out := h.put(q0), h.put(r0), h.put(kc0), h.put(vc0), h.alloc(c.T*qn)
			h.must(pd.Ink2PQKVPost(q, r, k, v, qclip, rclip, kc, vc, p, c.cap, c.pos0, c.T))
			h.must(pd.Ink2PAttend(q, r, sinks, rel, kc, vc, out, p, c.extent, c.window, c.cap, c.pos0, c.T))
			got := h.read(out, c.T*qn)

			qs, rs, kcs, vcs, outs := h.put(q0), h.put(r0), h.put(kc0), h.put(vc0), h.alloc(c.T*qn)
			scores, meta := h.alloc(p.Heads*c.cap), h.alloc(4)
			for tok := range c.T {
				h.must(be.Ink2Write(meta, []float32{math.Float32frombits(uint32(c.pos0 + tok))}))
				h.must(be.Ink2QKVPost(off(qs, tok*qn), off(rs, tok*rn), off(k, tok*kvDim), off(v, tok*kvDim), qclip, rclip,
					kcs, vcs, meta, p, c.cap, nil))
				h.must(be.Ink2Attend(off(qs, tok*qn), off(rs, tok*rn), off(sinks, tok*p.Heads), rel, kcs, vcs, scores,
					off(outs, tok*qn), meta, p, c.extent, c.window, c.cap))
				// the position is host-written, so each token is its own submission
				h.read(off(outs, tok*qn), 1)
			}
			h.same("attention", got, h.read(outs, c.T*qn))
			h.same("clipped q", h.read(q, c.T*qn), h.read(qs, c.T*qn))
			h.same("k cache", h.read(kc, c.cap*kvDim), h.read(kcs, c.cap*kvDim))
			h.same("v cache", h.read(vc, c.cap*kvDim), h.read(vcs, c.cap*kvDim))
		})
	}
}

// fak-test:runtime integration est=2s lane=default
func TestMetalInkling2PrefillExpertsEqualDecodeOp(t *testing.T) {
	be := metalInk2TestBackend(t)
	var pd Inkling2PrefillDevice = be
	h := metalInk2PTest{t: t, be: be, rng: rand.New(rand.NewSource(93))}
	const experts, rows, cols, k, T = 6, 5, 44, 3, 11
	const npairs = T * k
	stack := metalInk2TestUpload(t, be, h.rand(experts*rows*cols, 1))
	stack.Shape = []int{experts * rows, cols}
	vals := make([]int32, npairs)
	for i := range vals {
		vals[i] = int32(h.rng.Intn(experts - 1)) // the last expert is never routed
	}
	ids, scale := h.ints(vals), h.put(h.rand(experts, 3))

	// The bucket scratch sits between guard words: only its counts, offsets and
	// order regions may be written.
	need := Ink2PGroupScratch(npairs, experts)
	guard := math.Float32frombits(0xdeadbeef)
	fill := make([]float32, need+2)
	for i := range fill {
		fill[i] = guard
	}
	region := h.put(fill)
	scratch := off(region, 1)
	sortIDs, sortX, sortY := h.put(fill[:npairs]), h.put(make([]float32, npairs*cols)), h.put(make([]float32, npairs*rows))

	tokens := h.rand(T*cols, 1)
	perPair := make([]float32, npairs*cols)
	for i := range npairs {
		copy(perPair[i*cols:(i+1)*cols], tokens[(i/k)*cols:])
	}
	tokRef, pairRef := h.put(tokens), h.put(perPair)
	want := h.alloc(npairs * rows)
	if ok, err := be.Ink2ExpertMatVec(stack, rows, ids, pairRef, cols, want, npairs, scale, nil); err != nil || !ok {
		t.Fatalf("decode expert matvec: ok %v, %v", ok, err)
	}
	wantRows := h.read(want, npairs*rows)

	run := func(what string, f func(y Ink2Ref) (bool, error)) {
		t.Helper()
		y := h.alloc(npairs * rows)
		if ok, err := f(y); err != nil || !ok {
			t.Fatalf("%s: ok %v, %v", what, ok, err)
		}
		h.same(what, h.read(y, npairs*rows), wantRows)
	}
	run("grouped over token rows", func(y Ink2Ref) (bool, error) {
		return pd.Ink2PExpertGrouped(stack, rows, experts, ids, tokRef, k, cols, y, npairs, scale, scratch, false)
	})
	run("grouped over pair rows, bucket reused", func(y Ink2Ref) (bool, error) {
		return pd.Ink2PExpertGrouped(stack, rows, experts, ids, pairRef, 1, cols, y, npairs, scale, scratch, true)
	})
	run("sorted", func(y Ink2Ref) (bool, error) {
		return pd.Ink2PExpertSorted(stack, rows, experts, ids, tokRef, k, cols, y, npairs, scale, scratch, sortIDs, sortX, sortY)
	})

	got, err := be.Ink2ReadInts(region, need+2)
	h.must(err)
	if uint32(got[0]) != 0xdeadbeef || uint32(got[need+1]) != 0xdeadbeef {
		t.Fatalf("bucket scratch guards overwritten: %#x, %#x", uint32(got[0]), uint32(got[need+1]))
	}
	offsets, order := got[1+experts:][:experts+1], got[1+2*experts+1:][:npairs]
	if offsets[experts] != npairs || got[experts] != 0 {
		t.Fatalf("bucket of %d pairs ends at %d with %d on the unused expert", npairs, offsets[experts], got[experts])
	}
	for i := 1; i < npairs; i++ {
		a, b := order[i-1], order[i]
		if vals[a] > vals[b] || (vals[a] == vals[b] && a > b) {
			t.Fatalf("order[%d..%d] = %d, %d: experts %d, %d are not a stable sort", i-1, i, a, b, vals[a], vals[b])
		}
	}

	flat := stack
	flat.Shape = []int{experts * rows * cols}
	if ok, err := pd.Ink2PExpertGrouped(flat, rows, experts, ids, tokRef, k, cols, want, npairs, scale, scratch, false); ok || err != nil {
		t.Fatalf("a stack without rows was not declined: ok %v, %v", ok, err)
	}
	if _, err := pd.Ink2PExpertGrouped(stack, rows, experts+1, ids, tokRef, k, cols, want, npairs, scale, scratch, false); err == nil {
		t.Fatal("a wrong expert count was accepted")
	}
}

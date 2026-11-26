//go:build darwin && arm64 && cgo

package compute

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"os"
	"syscall"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

// The chunk widths a verify row must be bit-identical across.
var metalInk2VerifyWidths = []int{1, 2, 4, 8}

func metalInk2VerifySame(t *testing.T, what string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d values, want %d", what, len(got), len(want))
	}
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("%s: element %d = %v (%#x), want %v (%#x)", what, i, got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
		}
	}
}

func metalInk2VerifyRead(t *testing.T, be *metalBackend, r Ink2Ref, n int) []float32 {
	t.Helper()
	got, err := be.Ink2Read(r, n)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range got {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("element %d = %v: the fixture must stay finite", i, v)
		}
	}
	return got
}

func metalInk2VerifyRef(t *testing.T, be *metalBackend, v []float32) Ink2Ref {
	t.Helper()
	return Ink2At(metalInk2TestUpload(t, be, v), 0)
}

func metalInk2VerifyInts(vals ...int) []float32 {
	out := make([]float32, len(vals))
	for i, v := range vals {
		out[i] = math.Float32frombits(uint32(int32(v)))
	}
	return out
}

// fak-test:runtime integration est=2s lane=default
func TestMetalInkling2VerifyRowsMatchDecodeAtEveryWidth(t *testing.T) {
	be := metalInk2TestBackend(t)
	var vd Inkling2VerifyDevice = be
	rng := rand.New(rand.NewSource(53))
	const rows, cols, maxT = 10, 96, 9

	dense := metalInk2TestUpload(t, be, metalInk2TestRand(rng, rows*cols))
	dense.Shape = []int{rows, cols}

	data, err := syscall.Mmap(-1, 0, 2*os.Getpagesize(), syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		t.Fatal(err)
	}
	defer syscall.Munmap(data)
	const base = 96
	for b := 0; b < rows*cols/32; b++ {
		at := base + 34*b
		binary.LittleEndian.PutUint16(data[at:], uint16(0x3000+rng.Intn(1024)))
		rng.Read(data[at+2 : at+34])
	}
	store, err := metalgemm.NewPackedGGUFStore(data)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	matrix, err := store.Matrix(base, 8, rows, cols, 1)
	if err != nil {
		t.Fatal(err)
	}
	q8, err := MetalInkling2Weight(matrix, 0)
	if err != nil {
		t.Fatal(err)
	}

	x := metalInk2VerifyRef(t, be, metalInk2TestRand(rng, maxT*cols))
	for _, c := range []struct {
		name string
		w    Tensor
		run  func(w Tensor, x, y Ink2Ref, T int) (bool, error)
	}{
		{"dense f32", dense, vd.Ink2VRows},
		{"q8_0 rows", q8, vd.Ink2VRows},
		{"q8_0 shared-codes entry", q8, func(w Tensor, x, y Ink2Ref, T int) (bool, error) {
			return vd.Ink2VQ8Rows(w, x, Ink2Ref{}, Ink2Ref{}, y, T, true)
		}},
	} {
		t.Run(c.name, func(t *testing.T) {
			decode := metalInk2TestAlloc(t, be, maxT*rows)
			for r := range maxT {
				if err := be.Ink2MatVec(c.w, Ink2At(x.T, r*cols), Ink2At(decode, r*rows)); err != nil {
					t.Fatal(err)
				}
			}
			want := metalInk2VerifyRead(t, be, Ink2At(decode, 0), maxT*rows)
			for _, T := range append([]int{maxT}, metalInk2VerifyWidths...) {
				y := metalInk2TestAlloc(t, be, T*rows)
				ok, err := c.run(c.w, x, Ink2At(y, 0), T)
				if err != nil || !ok {
					t.Fatalf("width %d: ok=%v err=%v", T, ok, err)
				}
				metalInk2VerifySame(t, fmt.Sprintf("width %d vs the decode GEMV of each row", T),
					metalInk2VerifyRead(t, be, Ink2At(y, 0), T*rows), want[:T*rows])
			}
		})
	}
	y := metalInk2TestAlloc(t, be, rows)
	if ok, err := vd.Ink2VQ8Rows(dense, x, Ink2Ref{}, Ink2Ref{}, Ink2At(y, 0), 1, true); ok || err != nil {
		t.Fatalf("shared-codes entry on a dense weight: ok=%v err=%v; want a decline", ok, err)
	}
}

// One layer visit's worth of verify ops over a T-row chunk against the decode
// ops run token by token: every row, cache row and history must carry the same
// bits at every width, and a commit must rewind to the serial history.
//
// fak-test:runtime integration est=3s lane=default
func TestMetalInkling2VerifyChunkMatchesSerialDecode(t *testing.T) {
	be := metalInk2TestBackend(t)
	var vd Inkling2VerifyDevice = be
	dev, err := metalInk2Device()
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(59))
	const (
		maxT, n, rank, nq, ns, nj     = 8, 96, 6, 2, 3, 2
		heads, kvHeads, headDim       = 4, 2, 16
		relLen, extent, kn, cap, pos0 = 8, 6, 4, 24, 11
		kvDim, past                   = kvHeads * headDim, kn - 1
		qn, rn                        = heads * headDim, heads * relLen
		eps                           = 1e-6
	)
	// A global slot never wraps inside a chunk: the last row sits below cap.
	p := Ink2AttnParams{Heads: heads, KVHeads: kvHeads, HeadDim: headDim, RelLen: relLen, QKClip: 1.5, RelClip: .8,
		Eps: eps, Global: true, Alpha: .3, Floor: 4, Scale: float32(1 / math.Sqrt(headDim))}
	normK, normV := float32(1/math.Sqrt(2*kn+1)), float32(1/math.Sqrt(kn+1))
	up := func(count int) Ink2Ref { return metalInk2VerifyRef(t, be, metalInk2TestRand(rng, count)) }
	alloc := func(count int) Ink2Ref { return Ink2At(metalInk2TestAlloc(t, be, count), 0) }
	at := func(r Ink2Ref, off int) Ink2Ref { return Ink2At(r.T, r.Off+off) }

	states := []Ink2Ref{up(maxT * n), up(maxT * n), up(maxT * n)}
	queries, norms := up(nq*n), []Ink2Ref{up(n), up(n)}
	cur, msg := []Ink2Ref{up(maxT * n), up(maxT * n)}, []Ink2Ref{up(maxT * n), up(maxT * n)}
	down, gain, forget := []Ink2Ref{up(rank * 2 * n), up(rank * 2 * n)}, []Ink2Ref{up(n * rank), up(n * rank)}, []Ink2Ref{up(n * rank), up(n * rank)}
	xk, xv, wk, wv, dynK := up(maxT*kvDim), up(maxT*kvDim), up(kvDim*kn), up(kvDim*kn), up(maxT*kvHeads*kn)
	hist0 := metalInk2TestRand(rng, 2*kvDim*past)
	q0, r0 := metalInk2TestRand(rng, maxT*qn), metalInk2TestRand(rng, maxT*rn)
	for i := range q0 {
		// head amplitudes straddle the clip threshold
		q0[i] *= float32(1 + i/headDim%heads)
	}
	sinks, qclip, rclip, relBias := up(maxT*heads), up(qn), up(rn), up(extent*relLen)
	kc0, vc0 := metalInk2TestRand(rng, cap*kvDim), metalInk2TestRand(rng, cap*kvDim)
	positions := make([]int, maxT)
	for i := range positions {
		positions[i] = pos0 + i
	}
	meta := metalInk2VerifyRef(t, be, metalInk2VerifyInts(positions...))

	type result struct{ mix, g, ctl, k, v, q, r, kc, vc, attn, hist []float32 }
	// The serial reference, and the conv history after each token count.
	serial := func(T int) (result, [][]float32) {
		mix, g, ctl := alloc(maxT*nq*n), alloc(maxT*nj*rank), alloc(maxT*nj*n)
		k, v, attn, scores := alloc(maxT*kvDim), alloc(maxT*kvDim), alloc(maxT*qn), alloc(2*heads*cap)
		hist, q, r := metalInk2VerifyRef(t, be, hist0), metalInk2VerifyRef(t, be, q0), metalInk2VerifyRef(t, be, r0)
		kc, vc := metalInk2VerifyRef(t, be, kc0), metalInk2VerifyRef(t, be, vc0)
		hists := [][]float32{hist0}
		for i := range T {
			row := make([]Ink2Ref, ns)
			for s := range row {
				row[s] = at(states[s], i*n)
			}
			ctlRow := Ink2Controls{Down: down, Gain: gain, Forget: forget}
			for j := range nj {
				ctlRow.Cur, ctlRow.Msg = append(ctlRow.Cur, at(cur[j], i*n)), append(ctlRow.Msg, at(msg[j], i*n))
			}
			for _, err := range []error{
				be.Ink2Mix(row, queries, nq, norms, eps, at(mix, i*nq*n), n),
				be.Ink2Control(ctlRow, at(g, i*nj*rank), at(ctl, i*nj*n), n, rank, nil, 0),
				be.Ink2Conv(at(xk, i*kvDim), wk, at(dynK, i*kvHeads*kn), hist, at(k, i*kvDim), Ink2Ref{}, kvDim, kvHeads, kn, normK),
				be.Ink2Conv(at(xv, i*kvDim), wv, Ink2Ref{}, at(hist, kvDim*past), at(v, i*kvDim), Ink2Ref{}, kvDim, kvHeads, kn, normV),
				be.Ink2QKVPost(at(q, i*qn), at(r, i*rn), at(k, i*kvDim), at(v, i*kvDim), qclip, rclip, kc, vc, at(meta, i), p, cap, nil),
				be.Ink2Attend(at(q, i*qn), at(r, i*rn), at(sinks, i*heads), relBias, kc, vc, scores, at(attn, i*qn), at(meta, i), p, extent, 0, cap),
			} {
				if err != nil {
					t.Fatal(err)
				}
			}
			hists = append(hists, metalInk2VerifyRead(t, be, hist, 2*kvDim*past))
		}
		rd := func(ref Ink2Ref, count int) []float32 { return metalInk2VerifyRead(t, be, ref, count) }
		return result{rd(mix, T*nq*n), rd(g, T*nj*rank), rd(ctl, T*nj*n), rd(k, T*kvDim), rd(v, T*kvDim), rd(q, T*qn), rd(r, T*rn),
			rd(kc, cap*kvDim), rd(vc, cap*kvDim), rd(attn, T*qn), rd(hist, 2*kvDim*past)}, hists
	}

	for _, T := range metalInk2VerifyWidths {
		want, hists := serial(T)
		L := past + T
		mix, gs, ctls := alloc(nq*T*n), alloc(nj*T*rank), alloc(nj*T*n)
		g, ctl := []Ink2Ref{gs, at(gs, T*rank)}, []Ink2Ref{ctls, at(ctls, T*n)}
		k, v, attn, log := alloc(T*kvDim), alloc(T*kvDim), alloc(T*qn), alloc(2*kvDim*L)
		hist, q, r := metalInk2VerifyRef(t, be, hist0), metalInk2VerifyRef(t, be, q0), metalInk2VerifyRef(t, be, r0)
		kc, vc := metalInk2VerifyRef(t, be, kc0), metalInk2VerifyRef(t, be, vc0)
		before := dev.Stats()
		if ok, err := vd.Ink2VMix(states, queries, nq, norms, eps, mix, n, T); err != nil || !ok {
			t.Fatalf("width %d mix: ok=%v err=%v", T, ok, err)
		}
		for _, err := range []error{
			vd.Ink2PControlMulti(cur, msg, down, gain, forget, g, ctl, n, rank, T),
			vd.Ink2PConvPairLog([]Ink2Ref{xk, xv}, []Ink2Ref{wk, wv}, []Ink2Ref{dynK, {}}, []Ink2Ref{hist, at(hist, kvDim*past)},
				[]Ink2Ref{log, at(log, kvDim*L)}, []Ink2Ref{k, v}, []float32{normK, normV}, kvDim, kvHeads, kn, T),
			vd.Ink2PQKVPostDev(q, r, k, v, qclip, rclip, kc, vc, meta, p, cap, T),
			vd.Ink2PAttendDev(q, r, sinks, relBias, kc, vc, attn, meta, p, extent, 0, cap, T),
		} {
			if err != nil {
				t.Fatalf("width %d: %v", T, err)
			}
		}
		if s := dev.Stats(); s.Submissions != before.Submissions || s.Pending != before.Pending+6 {
			t.Fatalf("width %d: the chunk submitted instead of encoding: %+v after %+v", T, s, before)
		}
		rd := func(ref Ink2Ref, count int) []float32 { return metalInk2VerifyRead(t, be, ref, count) }
		same := func(what string, got, want []float32) {
			t.Helper()
			metalInk2VerifySame(t, fmt.Sprintf("width %d %s", T, what), got, want)
		}
		gotMix, gotG, gotCtl := rd(mix, nq*T*n), rd(gs, nj*T*rank), rd(ctls, nj*T*n)
		for i := range T {
			for qi := range nq {
				same(fmt.Sprintf("mix query %d row %d", qi, i), gotMix[(qi*T+i)*n:][:n], want.mix[(i*nq+qi)*n:][:n])
			}
			for j := range nj {
				same(fmt.Sprintf("controller %d row %d g", j, i), gotG[(j*T+i)*rank:][:rank], want.g[(i*nj+j)*rank:][:rank])
				same(fmt.Sprintf("controller %d row %d", j, i), gotCtl[(j*T+i)*n:][:n], want.ctl[(i*nj+j)*n:][:n])
			}
		}
		same("conv k", rd(k, T*kvDim), want.k)
		same("conv v", rd(v, T*kvDim), want.v)
		same("clipped q", rd(q, T*qn), want.q)
		same("clipped r", rd(r, T*rn), want.r)
		same("k cache", rd(kc, cap*kvDim), want.kc)
		same("v cache", rd(vc, cap*kvDim), want.vc)
		same("attention", rd(attn, T*qn), want.attn)
		same("conv history", rd(hist, 2*kvDim*past), want.hist)

		// the single logged conv with a residual, then both commit forms
		res, out1, hist1, log1 := up(T*kvDim), alloc(T*kvDim), metalInk2VerifyRef(t, be, hist0), alloc(kvDim*L)
		if err := vd.Ink2PConvLog(xk, wk, dynK, hist1, log1, out1, res, kvDim, kvHeads, kn, normK, T); err != nil {
			t.Fatal(err)
		}
		gotOut, resid := rd(out1, T*kvDim), rd(res, T*kvDim)
		for i := range gotOut {
			if w := resid[i] + want.k[i]; math.Float32bits(gotOut[i]) != math.Float32bits(w) {
				t.Fatalf("width %d logged conv with a residual: element %d = %v, want %v", T, i, gotOut[i], w)
			}
		}
		same("logged conv history", rd(hist1, kvDim*past), want.hist[:kvDim*past])
		for keep := 0; keep <= T; keep++ {
			if err := vd.Ink2PConvCommit(hist, log, 2*kvDim, past, L, keep); err != nil {
				t.Fatal(err)
			}
			same(fmt.Sprintf("commit of %d", keep), rd(hist, 2*kvDim*past), hists[keep])
			if err := be.Ink2Write(hist, make([]float32, 2*kvDim*past)); err != nil {
				t.Fatal(err)
			}
			if err := vd.Ink2PConvCommitDev(hist, log, metalInk2VerifyRef(t, be, metalInk2VerifyInts(keep)), 2*kvDim, past, L); err != nil {
				t.Fatal(err)
			}
			same(fmt.Sprintf("device commit of %d", keep), rd(hist, 2*kvDim*past), hists[keep])
		}

		rep := alloc(T * 3 * n)
		if err := vd.Ink2VRepeatRows(cur[0], rep, n, T, 3); err != nil {
			t.Fatal(err)
		}
		gotRep, src := rd(rep, T*3*n), rd(cur[0], T*n)
		for i := range T * 3 {
			same(fmt.Sprintf("repeat pair %d", i), gotRep[i*n:][:n], src[i/3*n:][:n])
		}
	}
}

// fak-test:runtime integration est=1s lane=default
func TestMetalInkling2VerifyGraphSlotsAndBadOperands(t *testing.T) {
	be := metalInk2TestBackend(t)
	var vd Inkling2VerifyDevice = be
	for name, err := range map[string]error{"end slot": vd.Ink2GraphEndSlot(1), "launch slot": vd.Ink2GraphLaunchSlot(1)} {
		if !errors.Is(err, errors.ErrUnsupported) {
			t.Errorf("graph %s: %v, want an unsupported error", name, err)
		}
	}
	buf := metalInk2TestAlloc(t, be, 64)
	r := Ink2At(buf, 0)
	p := Ink2AttnParams{Heads: 2, KVHeads: 1, HeadDim: 4, RelLen: 2, Scale: 1}
	dense := metalInk2TestAlloc(t, be, 16)
	dense.Shape = []int{4, 4}
	_, rowsErr := vd.Ink2VRows(dense, r, r, 1)
	_, mixErr := vd.Ink2VMix(nil, r, 1, nil, 1e-5, r, 4, 1)
	before := metalInk2.dev.Stats()
	for name, err := range map[string]error{
		"rows onto their input":   rowsErr,
		"mix without a state":     mixErr,
		"control without a row":   vd.Ink2PControlMulti(nil, nil, nil, nil, nil, nil, nil, 4, 1, 1),
		"conv onto its input":     vd.Ink2PConvLog(r, r, Ink2Ref{}, r, Ink2Ref{}, r, Ink2Ref{}, 4, 1, 2, 1, 1),
		"conv pair of one conv":   vd.Ink2PConvPairLog([]Ink2Ref{r}, []Ink2Ref{r}, []Ink2Ref{r}, []Ink2Ref{r}, []Ink2Ref{r}, []Ink2Ref{r}, []float32{1}, 4, 1, 2, 1),
		"commit past the log row": vd.Ink2PConvCommit(r, r, 4, 3, 5, 3),
		"repeat onto its input":   vd.Ink2VRepeatRows(r, r, 4, 1, 2),
		"more rows than cache":    vd.Ink2PQKVPostDev(r, r, r, r, r, r, r, r, r, p, 1, 2),
		"attend of no rows":       vd.Ink2PAttendDev(r, r, r, r, r, r, r, r, p, 1, 0, 4, 0),
	} {
		if !errors.Is(err, metalgemm.ErrInk2Shape) {
			t.Errorf("%s: %v, want the operand error", name, err)
		}
	}
	for name, err := range map[string]error{
		"device commit without a keep":   vd.Ink2PConvCommitDev(r, r, Ink2Ref{}, 4, 3, 5),
		"qkv post without a position":    vd.Ink2PQKVPostDev(r, r, r, r, r, r, r, r, Ink2Ref{}, p, 4, 1),
		"attend without a position":      vd.Ink2PAttendDev(r, r, r, r, r, r, r, Ink2Ref{}, p, 1, 0, 4, 1),
		"attend into a host-side tensor": vd.Ink2PAttendDev(r, r, r, r, r, r, Ink2At(NewF32(be, []int{8}, make([]float32, 8)), 0), r, p, 1, 0, 4, 1),
	} {
		if err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
	if s := metalInk2.dev.Stats(); s.Pending != before.Pending || s.Dispatches != before.Dispatches {
		t.Errorf("a rejected op encoded work: %+v (before %+v)", s, before)
	}
}

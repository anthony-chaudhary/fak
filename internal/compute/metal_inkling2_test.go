//go:build darwin && arm64 && cgo

package compute

import (
	"encoding/binary"
	"errors"
	"math"
	"math/rand"
	"os"
	"syscall"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

func metalInk2TestBackend(t *testing.T) *metalBackend {
	t.Helper()
	be, ok := MetalInkling2Backend()
	if !ok {
		t.Skip("requires a Metal device")
	}
	return be.(*metalBackend)
}

func metalInk2TestAlloc(t *testing.T, be *metalBackend, n int) Tensor {
	t.Helper()
	tt, err := be.Ink2Alloc(n)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { be.Free(tt) })
	return tt
}

func metalInk2TestUpload(t *testing.T, be *metalBackend, v []float32) Tensor {
	t.Helper()
	tt := metalInk2TestAlloc(t, be, len(v))
	if err := be.Ink2Write(Ink2At(tt, 0), v); err != nil {
		t.Fatal(err)
	}
	return tt
}

func metalInk2TestRand(r *rand.Rand, n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = r.Float32() - .5
	}
	return v
}

// The CUDA expert conformance bound over a float64 host product.
func metalInk2TestRow(t *testing.T, what string, got float32, w []float64, x []float32, scale float64) {
	t.Helper()
	var dot, mag float64
	for i, wv := range w {
		p := wv * float64(x[i])
		dot += p
		mag += math.Abs(p)
	}
	dot, mag = dot*scale, mag*math.Abs(scale)
	if math.Abs(float64(got)-dot) > 1e-5*mag+1e-6 {
		t.Fatalf("%s: device %v, host %v (|w||x| %v)", what, got, dot, mag)
	}
}

func metalInk2TestF64(v []float32) []float64 {
	out := make([]float64, len(v))
	for i, f := range v {
		out[i] = float64(f)
	}
	return out
}

// fak-test:runtime integration est=1s lane=default
func TestMetalInkling2AllocWriteRead(t *testing.T) {
	be := metalInk2TestBackend(t)
	buf := metalInk2TestAlloc(t, be, 64)
	if buf.Dtype != F32 || len(buf.Shape) != 1 || buf.Shape[0] != 64 || buf.Backend() != Backend(be) {
		t.Fatalf("alloc tensor %v %v on %v", buf.Dtype, buf.Shape, buf.Backend())
	}
	got, err := be.Ink2Read(Ink2At(buf, 0), 64)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range got {
		if v != 0 {
			t.Fatalf("fresh element %d = %v, want 0", i, v)
		}
	}
	src := []float32{3.5, -1, math.Float32frombits(41)}
	if err := be.Ink2Write(Ink2At(buf, 20), src); err != nil {
		t.Fatal(err)
	}
	if got, err = be.Ink2Read(Ink2At(buf, 19), 5); err != nil {
		t.Fatal(err)
	}
	if got[0] != 0 || got[1] != 3.5 || got[2] != -1 || got[4] != 0 {
		t.Fatalf("read back %v", got)
	}
	ints, err := be.Ink2ReadInts(Ink2At(buf, 22), 1)
	if err != nil || ints[0] != 41 {
		t.Fatalf("ints %v, %v; want [41]", ints, err)
	}
	if err := be.Ink2Write(Ink2At(buf, 1<<20), src); err == nil {
		t.Fatal("write past the end was accepted")
	}
	if _, err := be.Ink2Read(Ink2Ref{}, 1); err == nil {
		t.Fatal("read of an absent operand was accepted")
	}
	if err := be.Ink2Write(Ink2At(NewF32(be, []int{3}, src), 0), src); err == nil {
		t.Fatal("write into a host tensor was accepted")
	}
}

// fak-test:runtime integration est=2s lane=default
func TestMetalInkling2MatVecMatchesHostReference(t *testing.T) {
	be := metalInk2TestBackend(t)
	rng := rand.New(rand.NewSource(37))
	arena := metalInk2TestAlloc(t, be, 1<<13)

	// Dense F32 rows in a device buffer, bound as the resident forward binds
	// its small matrices: the alloc tensor with a [rows, cols] shape.
	const rows, cols, experts, k = 6, 41, 4, 3
	wf := metalInk2TestRand(rng, experts*rows*cols)
	x := metalInk2TestRand(rng, cols)
	dense := metalInk2TestUpload(t, be, wf)
	dense.Shape = []int{rows, cols}
	xr, yr := Ink2At(arena, 5), Ink2At(arena, 1024)
	if err := be.Ink2Write(xr, x); err != nil {
		t.Fatal(err)
	}
	if err := be.Ink2MatVec(dense, xr, yr); err != nil {
		t.Fatal(err)
	}
	got, err := be.Ink2Read(yr, rows)
	if err != nil {
		t.Fatal(err)
	}
	w64 := metalInk2TestF64(wf)
	for r := range rows {
		metalInk2TestRow(t, "dense row", got[r], w64[r*cols:(r+1)*cols], x, 1)
	}

	ids := []int{3, 1, 2}
	idBits := make([]float32, k)
	for i, id := range ids {
		idBits[i] = math.Float32frombits(uint32(id))
	}
	scale := []float32{2, -0.5, 1.25, 3}
	idr := Ink2At(metalInk2TestUpload(t, be, idBits), 0)
	scr := Ink2At(metalInk2TestUpload(t, be, scale), 0)
	dense.Shape = []int{experts * rows, cols}
	out := Ink2At(arena, 2048)
	ok, err := be.Ink2ExpertMatVec(dense, rows, idr, xr, 0, out, k, scr, nil)
	if err != nil || !ok {
		t.Fatalf("dense expert matvec: ok=%v err=%v", ok, err)
	}
	if got, err = be.Ink2Read(out, k*rows); err != nil {
		t.Fatal(err)
	}
	for slot, id := range ids {
		for r := range rows {
			metalInk2TestRow(t, "dense routed row", got[slot*rows+r], w64[(id*rows+r)*cols:][:cols], x, float64(scale[id]))
		}
	}

	// A Q8_0 expert stack read in place from a mapping.
	const qrows, qcols = 10, 96
	data, err := syscall.Mmap(-1, 0, 2*os.Getpagesize(), syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		t.Fatal(err)
	}
	defer syscall.Munmap(data)
	const base = 96
	q64 := make([]float64, experts*qrows*qcols)
	for b := 0; b < len(q64)/32; b++ {
		at := base + 34*b
		half := uint16(0x3000 + rng.Intn(1024))
		binary.LittleEndian.PutUint16(data[at:], half)
		d := math.Ldexp(float64(1024+half&1023), int(half>>10&31)-25)
		for j := range 32 {
			code := int8(rng.Intn(256) - 128)
			data[at+2+j] = byte(code)
			q64[32*b+j] = d * float64(code)
		}
	}
	store, err := metalgemm.NewPackedGGUFStore(data)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	matrix, err := store.Matrix(base, 8, qrows, qcols, experts)
	if err != nil {
		t.Fatal(err)
	}
	stack, err := MetalInkling2Weight(matrix, -1)
	if err != nil {
		t.Fatal(err)
	}
	if stack.Dtype != Q8_0 || stack.Shape[0] != experts*qrows || stack.Shape[1] != qcols {
		t.Fatalf("stack tensor %v %v", stack.Dtype, stack.Shape)
	}
	const stride = 128
	xs := metalInk2TestRand(rng, k*stride)
	xsr := Ink2At(arena, 4096)
	if err := be.Ink2Write(xsr, xs); err != nil {
		t.Fatal(err)
	}
	for e := range experts {
		view, err := MetalInkling2Weight(matrix, e)
		if err != nil {
			t.Fatal(err)
		}
		if err := be.Ink2MatVec(view, xsr, Ink2At(arena, 1024+e*qrows)); err != nil {
			t.Fatal(err)
		}
	}
	if got, err = be.Ink2Read(Ink2At(arena, 1024), experts*qrows); err != nil {
		t.Fatal(err)
	}
	for e := range experts {
		for r := range qrows {
			metalInk2TestRow(t, "q8_0 row", got[e*qrows+r], q64[(e*qrows+r)*qcols:][:qcols], xs, 1)
		}
	}
	for _, scaled := range []bool{false, true} {
		sr, by := Ink2Ref{}, func(int) float64 { return 1 }
		if scaled {
			sr, by = scr, func(id int) float64 { return float64(scale[id]) }
		}
		ok, err := be.Ink2ExpertMatVec(stack, qrows, idr, xsr, stride, out, k, sr, nil)
		if err != nil || !ok {
			t.Fatalf("q8_0 expert matvec scaled=%v: ok=%v err=%v", scaled, ok, err)
		}
		if got, err = be.Ink2Read(out, k*qrows); err != nil {
			t.Fatal(err)
		}
		for slot, id := range ids {
			for r := range qrows {
				metalInk2TestRow(t, "q8_0 routed row", got[slot*qrows+r], q64[(id*qrows+r)*qcols:][:qcols], xs[slot*stride:], by(id))
			}
		}
	}

	// The shared-codes Q8_0 entry declines, which sends the driver to Ink2MatVec.
	view, _ := MetalInkling2Weight(matrix, 0)
	if ok, err := be.Ink2MatVecQ8(view, xsr, Ink2Ref{}, Ink2At(arena, 6000), true, out); ok || err != nil {
		t.Fatalf("shared Q8_0 matvec: ok=%v err=%v; want a decline", ok, err)
	}
	if ok, err := be.Ink2ExpertMatVec(stack, qrows, idr, xsr, stride, out, k, Ink2Ref{}, &Ink2Situ{Width: 8, Beta: 1}); ok || err != nil {
		t.Fatalf("fused SITU expert matvec: ok=%v err=%v; want a decline", ok, err)
	}
	if err := be.Ink2MatVec(stack, xsr, out); err == nil {
		t.Fatal("a whole expert stack was accepted as one matrix")
	}
	if _, err := be.Ink2ExpertMatVec(view, qrows, idr, xsr, stride, out, k, Ink2Ref{}, nil); err == nil {
		t.Fatal("one expert's view was accepted as a stack")
	}
	if _, err := MetalInkling2Weight(matrix, experts); err == nil {
		t.Fatal("an expert past the stack was wrapped")
	}
}

// fak-test:runtime integration est=1s lane=default
func TestMetalInkling2TokenIsOneSubmission(t *testing.T) {
	be := metalInk2TestBackend(t)
	dev, err := metalInk2Device()
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(41))
	const n = 24
	a, b, x := metalInk2TestRand(rng, n*n), metalInk2TestRand(rng, n*n), metalInk2TestRand(rng, n)
	wa, wb := metalInk2TestUpload(t, be, a), metalInk2TestUpload(t, be, b)
	wa.Shape, wb.Shape = []int{n, n}, []int{n, n}
	arena := metalInk2TestAlloc(t, be, 256)
	if err := be.Ink2Write(Ink2At(arena, 0), x); err != nil {
		t.Fatal(err)
	}
	before := dev.Stats()
	// The second product consumes the first inside the same command buffer.
	if err := be.Ink2MatVec(wa, Ink2At(arena, 0), Ink2At(arena, 64)); err != nil {
		t.Fatal(err)
	}
	if err := be.Ink2MatVec(wb, Ink2At(arena, 64), Ink2At(arena, 128)); err != nil {
		t.Fatal(err)
	}
	if s := dev.Stats(); s.Pending != 2 || s.Submissions != before.Submissions {
		t.Fatalf("after two ops: %+v (before %+v), want both pending", s, before)
	}
	got, err := be.Ink2Read(Ink2At(arena, 128), n)
	if err != nil {
		t.Fatal(err)
	}
	if s := dev.Stats(); s.Pending != 0 || s.Submissions != before.Submissions+1 || s.Dispatches != before.Dispatches+2 {
		t.Fatalf("after the read fence: %+v (before %+v), want one submission of two dispatches", s, before)
	}
	mid := make([]float32, n)
	for r := range n {
		var acc float64
		for j, v := range x {
			acc += float64(a[r*n+j]) * float64(v)
		}
		mid[r] = float32(acc)
	}
	b64 := metalInk2TestF64(b)
	for r := range n {
		// the host's intermediate is not the device's, so the bound is doubled
		var dot, mag float64
		for j, v := range mid {
			p := b64[r*n+j] * float64(v)
			dot, mag = dot+p, mag+math.Abs(p)
		}
		if math.Abs(float64(got[r])-dot) > 2e-5*mag+1e-6 {
			t.Fatalf("chained row %d: device %v, host %v", r, got[r], dot)
		}
	}
	if _, err := be.Ink2ReadInts(Ink2At(arena, 0), 1); err != nil {
		t.Fatal(err)
	}
	if s := dev.Stats(); s.Submissions != before.Submissions+1 {
		t.Fatalf("an idle fence submitted a command buffer: %+v", s)
	}
}

// fak-test:runtime integration est=1s lane=default
func TestMetalInkling2GraphFallsBackAndBadOperandsReject(t *testing.T) {
	be := metalInk2TestBackend(t)
	var dev Inkling2Device = be
	for name, err := range map[string]error{
		"begin": dev.Ink2GraphBegin(), "end": dev.Ink2GraphEnd(), "launch": dev.Ink2GraphLaunch(),
	} {
		if !errors.Is(err, errors.ErrUnsupported) {
			t.Errorf("graph %s: %v, want an unsupported error", name, err)
		}
	}
	dev.Ink2GraphReset()

	buf := metalInk2TestAlloc(t, be, 64)
	r := Ink2At(buf, 0)
	before := metalInk2.dev.Stats()
	for name, err := range map[string]error{
		"rms":            dev.Ink2RMS(r, r, r, 0, 4, 1e-5),
		"ctrl pair":      dev.Ink2CtrlPair(r, r, r, 0),
		"mix":            dev.Ink2Mix(nil, r, 1, nil, 1e-5, r, 4),
		"control":        dev.Ink2Control(Ink2Controls{}, r, r, 4, 1, nil, 1e-5),
		"conv":           dev.Ink2Conv(r, r, Ink2Ref{}, r, r, Ink2Ref{}, 4, 1, 1, 1),
		"combine conv":   dev.Ink2CombineConv(r, r, 0, r, r, r, Ink2Ref{}, 4, 2, 1),
		"qkv post":       dev.Ink2QKVPost(r, r, r, r, r, r, r, r, r, Ink2AttnParams{}, 1, nil),
		"attend":         dev.Ink2Attend(r, r, r, r, r, r, r, r, r, Ink2AttnParams{}, 1, 0, 1),
		"sigmoid mul":    dev.Ink2SigmoidMul(r, r, 0),
		"route":          dev.Ink2Route(r, r, r, r, 4, 5, 1),
		"expert scale":   dev.Ink2ExpertScale(r, r, r, 0, 4),
		"situ":           dev.Ink2Situ(r, r, 0, 2, 2, 1, 0),
		"moe combine":    dev.Ink2Combine(r, r, r, 0, 4),
		"add":            dev.Ink2Add(r, r, r, 0),
		"softcap argmax": dev.Ink2SoftcapArgmax(r, 0, 30, r),
	} {
		if !errors.Is(err, metalgemm.ErrInk2Shape) {
			t.Errorf("%s: %v, want the operand error", name, err)
		}
	}
	if s := metalInk2.dev.Stats(); s.Pending != before.Pending || s.Dispatches != before.Dispatches {
		t.Errorf("a rejected op encoded work: %+v (before %+v)", s, before)
	}
}

func metalInk2TestClip(x, scale []float32, thr float32) []float64 {
	out := make([]float64, len(x))
	var ss float64
	for i, v := range x {
		out[i] = float64(v)
		if scale != nil {
			out[i] = float64(v * scale[i])
		}
		ss += out[i] * out[i]
	}
	f := min(1, float64(thr)/(math.Sqrt(ss)+1e-6))
	for i := range out {
		out[i] *= f
	}
	return out
}

// The cache rows QKVPost appends are the keys and values Attend reads: grouped
// heads over a ring that wraps, against the host formulas.
//
// fak-test:runtime integration est=1s lane=default
func TestMetalInkling2QKVPostFeedsAttend(t *testing.T) {
	be := metalInk2TestBackend(t)
	const heads, kvHeads, headDim, relLen, extent, window, cap = 4, 2, 12, 5, 2, 3, 3
	const kvDim = kvHeads * headDim
	p := Ink2AttnParams{Heads: heads, KVHeads: kvHeads, HeadDim: headDim, RelLen: relLen,
		QKClip: 1.5, RelClip: 0.6, Eps: 1e-5, Scale: float32(1 / math.Sqrt(headDim))}
	rng := rand.New(rand.NewSource(11))
	qScale, rScale := metalInk2TestRand(rng, heads*headDim), metalInk2TestRand(rng, heads*relLen)
	relBias := metalInk2TestRand(rng, extent*relLen)
	qclip, rclip, bias := metalInk2TestUpload(t, be, qScale), metalInk2TestUpload(t, be, rScale), metalInk2TestUpload(t, be, relBias)
	kcache, vcache := metalInk2TestAlloc(t, be, cap*kvDim), metalInk2TestAlloc(t, be, cap*kvDim)
	scores := metalInk2TestAlloc(t, be, 2*heads*cap)
	// q, r, k, v, sinks, the attention output, then the position word.
	const qAt, rAt, kAt, vAt, sinkAt, outAt, metaAt = 0, 64, 128, 192, 256, 320, 384
	arena := metalInk2TestAlloc(t, be, 448)
	at := func(off int) Ink2Ref { return Ink2At(arena, off) }

	var keys, vals [][]float64
	for pos := range 5 {
		q, r := metalInk2TestRand(rng, heads*headDim), metalInk2TestRand(rng, heads*relLen)
		k, v, sinks := metalInk2TestRand(rng, kvDim), metalInk2TestRand(rng, kvDim), metalInk2TestRand(rng, heads)
		for i := range q {
			q[i], k[i%kvDim] = 6*q[i], 6*k[i%kvDim]
		}
		in := make([]float32, metaAt+1)
		copy(in[qAt:], q)
		copy(in[rAt:], r)
		copy(in[kAt:], k)
		copy(in[vAt:], v)
		copy(in[sinkAt:], sinks)
		in[metaAt] = math.Float32frombits(uint32(pos))
		if err := be.Ink2Write(at(0), in); err != nil {
			t.Fatal(err)
		}
		if err := be.Ink2QKVPost(at(qAt), at(rAt), at(kAt), at(vAt), Ink2At(qclip, 0), Ink2At(rclip, 0),
			Ink2At(kcache, 0), Ink2At(vcache, 0), at(metaAt), p, cap, nil); err != nil {
			t.Fatal(err)
		}
		if err := be.Ink2Attend(at(qAt), at(rAt), at(sinkAt), Ink2At(bias, 0), Ink2At(kcache, 0), Ink2At(vcache, 0),
			Ink2At(scores, 0), at(outAt), at(metaAt), p, extent, window, cap); err != nil {
			t.Fatal(err)
		}
		got, err := be.Ink2Read(at(outAt), heads*headDim)
		if err != nil {
			t.Fatal(err)
		}

		key, val := make([]float64, 0, kvDim), make([]float64, kvDim)
		for h := range kvHeads {
			key = append(key, metalInk2TestClip(k[h*headDim:(h+1)*headDim], nil, p.QKClip)...)
			var ss float64
			for _, x := range v[h*headDim : (h+1)*headDim] {
				ss += float64(x) * float64(x)
			}
			for i, x := range v[h*headDim : (h+1)*headDim] {
				val[h*headDim+i] = float64(x) / math.Sqrt(ss/headDim+float64(p.Eps))
			}
		}
		keys, vals = append(keys, key), append(vals, val)
		n := min(pos+1, window)
		for h := range heads {
			kvh := h / (heads / kvHeads)
			qh := metalInk2TestClip(q[h*headDim:(h+1)*headDim], qScale[h*headDim:(h+1)*headDim], p.QKClip)
			rh := metalInk2TestClip(r[h*relLen:(h+1)*relLen], rScale[h*relLen:(h+1)*relLen], p.RelClip)
			score, top := make([]float64, n), 0.0
			for j := range n {
				kj := keys[pos+1-n+j][kvh*headDim:]
				var dot float64
				for i, x := range qh {
					dot += x * kj[i]
				}
				score[j] = dot*float64(p.Scale) - float64(sinks[h])
				if dist := n - 1 - j; dist < extent {
					for i, x := range rh {
						score[j] += float64(relBias[dist*relLen+i]) * x
					}
				}
				top = max(top, score[j])
			}
			denom := math.Exp(-top)
			for j := range score {
				score[j] = math.Exp(score[j] - top)
				denom += score[j]
			}
			for i := range headDim {
				var want, mag float64
				for j, w := range score {
					x := w / denom * vals[pos+1-n+j][kvh*headDim+i]
					want, mag = want+x, mag+math.Abs(x)
				}
				// the device clips and normalizes in float32 before it attends, so the bound is doubled
				if g := float64(got[h*headDim+i]); math.Abs(g-want) > 2e-5*mag+2e-6 {
					t.Fatalf("position %d head %d dim %d: device %v, host %v", pos, h, i, g, want)
				}
			}
		}
	}
}

// fak-test:runtime integration est=1s lane=default
func TestMetalInkling2StatsAndProfileWrappers(t *testing.T) {
	be := metalInk2TestBackend(t)
	dev, err := metalInk2Device()
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(43))
	const n = 32
	w := metalInk2TestUpload(t, be, metalInk2TestRand(rng, n*n))
	w.Shape = []int{n, n}
	arena := metalInk2TestAlloc(t, be, 128)
	if err := be.Ink2Write(Ink2At(arena, 0), metalInk2TestRand(rng, n)); err != nil {
		t.Fatal(err)
	}
	if err := MetalInkling2ProfileReset(); err != nil {
		t.Fatal(err)
	}
	before, ok := MetalInkling2Stats()
	if !ok {
		t.Fatal("no stats from a live device")
	}
	if err := be.Ink2MatVec(w, Ink2At(arena, 0), Ink2At(arena, 64)); err != nil {
		t.Fatal(err)
	}
	if _, err := be.Ink2Read(Ink2At(arena, 64), n); err != nil {
		t.Fatal(err)
	}
	after, _ := MetalInkling2Stats()
	if d := after.Sub(before); d.Dispatches != 1 || d.Submissions != 1 || d.EncodeSeconds <= 0 || d.WaitSeconds <= 0 || d.GPUSeconds <= 0 {
		t.Fatalf("stats delta %+v, want one timed dispatch in one submission", d)
	}
	p, ok := MetalInkling2Profile()
	if !ok || p.Mode != dev.ProfileMode() {
		t.Fatalf("profile mode %q (ok %v), device %q", p.Mode, ok, dev.ProfileMode())
	}
	if !p.Enabled() {
		if len(p.Rows) != 0 {
			t.Fatalf("profile off collected %+v", p.Rows)
		}
		return
	}
	if len(p.Rows) != 1 || p.Rows[0].Scope != "Ink2MatVec" || p.Rows[0].Kernel != "mv_f32" || p.Rows[0].Detail != "32x32" || p.Rows[0].Dispatches != 1 {
		t.Fatalf("profile rows %+v, want one Ink2MatVec mv_f32 32x32 dispatch", p.Rows)
	}
}

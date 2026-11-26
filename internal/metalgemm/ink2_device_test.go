//go:build darwin && arm64 && cgo

package metalgemm

import (
	"encoding/binary"
	"errors"
	"math"
	"math/rand"
	"os"
	"syscall"
	"testing"
)

func ink2TestDevice(t *testing.T) *Ink2Device {
	t.Helper()
	if !Available() {
		t.Skip("requires a Metal device")
	}
	d, err := NewInk2Device()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(d.Close)
	return d
}

func ink2TestAlloc(t *testing.T, d *Ink2Device, n int) Ink2Ref {
	t.Helper()
	b, err := d.Alloc(n)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { d.Free(b) })
	return Ink2Ref{Buf: b}
}

// fak-test:runtime integration est=1s lane=default
func TestInk2AllocWriteReadRoundTrip(t *testing.T) {
	d := ink2TestDevice(t)
	buf := ink2TestAlloc(t, d, 100)
	got, err := d.Read(buf, 100)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range got {
		if v != 0 {
			t.Fatalf("fresh buffer element %d = %v, want 0", i, v)
		}
	}
	src := []float32{1.5, -2.25, float32(math.Inf(1)), math.Float32frombits(7), math.Float32frombits(0x80000003)}
	if err := d.Write(buf.At(7), src); err != nil {
		t.Fatal(err)
	}
	// An adopted handle addresses the same memory as the one Alloc returned.
	adopted := Ink2Ref{Buf: Ink2AdoptBuffer(buf.Buf.h)}
	got, err = d.Read(adopted.At(6), 7)
	if err != nil {
		t.Fatal(err)
	}
	want := append(append([]float32{0}, src...), 0)
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("element %d = %v, want %v", 6+i, got[i], want[i])
		}
	}
	ints, err := d.ReadInts(buf.At(10), 2)
	if err != nil {
		t.Fatal(err)
	}
	if ints[0] != 7 || ints[1] != math.MinInt32+3 {
		t.Fatalf("ints = %v, want [7 %d]", ints, math.MinInt32+3)
	}
	if err := d.Zero(buf.At(7), 2); err != nil {
		t.Fatal(err)
	}
	if got, _ = d.Read(buf.At(7), 3); got[0] != 0 || got[1] != 0 || !math.IsInf(float64(got[2]), 1) {
		t.Fatalf("after zeroing two elements: %v", got)
	}
	if s := d.Stats(); s.Submissions != 0 || s.Dispatches != 0 {
		t.Fatalf("host transfers submitted GPU work: %+v", s)
	}
	for _, bad := range []struct {
		name string
		err  error
	}{
		{"write past the end", d.Write(buf.At(98), src)},
		{"negative offset", d.Write(buf.At(-1), src)},
		{"absent buffer", d.Write(Ink2Ref{}, src)},
	} {
		if !errors.Is(bad.err, ErrInk2Shape) {
			t.Errorf("%s: %v, want ErrInk2Shape", bad.name, bad.err)
		}
	}
	if _, err := d.Read(buf.At(50), 51); !errors.Is(err, ErrInk2Shape) {
		t.Errorf("read past the end: %v", err)
	}
}

func ink2TestHalf(bits uint16) float64 {
	return math.Ldexp(float64(1024+bits&1023), int(bits>>10&31)-25)
}

// ink2TestMatrix is a packed matrix written into a mapping, with its float64
// decode as the host reference.
type ink2TestMatrix struct {
	weight              Ink2Weight
	stack               *PackedGGUFMatrix
	rows, cols, experts int
	ref                 []float64
}

func (m *ink2TestMatrix) dot(expert, row int, x []float32) (sum, mag float64) {
	for j, w := range m.ref[(expert*m.rows+row)*m.cols:][:m.cols] {
		p := w * float64(x[j])
		sum += p
		mag += math.Abs(p)
	}
	return sum, mag
}

// The CUDA expert conformance bound.
func ink2TestClose(got float32, want, mag float64) bool {
	return math.Abs(float64(got)-want) <= 1e-5*mag+1e-6
}

func ink2TestStore(t *testing.T, pages int) (*PackedGGUFStore, []byte) {
	t.Helper()
	data, err := syscall.Mmap(-1, 0, pages*os.Getpagesize(), syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		t.Fatal(err)
	}
	store, err := NewPackedGGUFStore(data)
	if err != nil {
		syscall.Munmap(data)
		t.Fatal(err)
	}
	t.Cleanup(func() {
		store.Close()
		syscall.Munmap(data)
	})
	return store, data
}

// ink2TestFill writes experts*rows*cols weights of a GGML type at offset and
// returns the matrix and the first byte after it.
func ink2TestFill(t *testing.T, store *PackedGGUFStore, data []byte, rng *rand.Rand, offset int, typ uint32, rows, cols, experts int) (*ink2TestMatrix, int) {
	t.Helper()
	m := &ink2TestMatrix{rows: rows, cols: cols, experts: experts, ref: make([]float64, experts*rows*cols)}
	at := offset
	switch typ {
	case 0:
		for i := range m.ref {
			v := rng.Float32() - .5
			binary.LittleEndian.PutUint32(data[at:], math.Float32bits(v))
			m.ref[i] = float64(v)
			at += 4
		}
	case 30:
		for i := range m.ref {
			bits := math.Float32bits(rng.Float32()-.5) >> 16
			binary.LittleEndian.PutUint16(data[at:], uint16(bits))
			m.ref[i] = float64(math.Float32frombits(bits << 16))
			at += 2
		}
	case 8:
		for i := 0; i < len(m.ref); i += 32 {
			scale := uint16(0x3000 + rng.Intn(1024))
			binary.LittleEndian.PutUint16(data[at:], scale)
			for j := range 32 {
				code := int8(rng.Intn(256) - 128)
				data[at+2+j] = byte(code)
				m.ref[i+j] = ink2TestHalf(scale) * float64(code)
			}
			at += 34
		}
	default:
		t.Fatalf("no synthesizer for GGML type %d", typ)
	}
	stack, err := store.Matrix(offset, typ, rows, cols, experts)
	if err != nil {
		t.Fatal(err)
	}
	m.stack = stack
	return m, (at + 31) &^ 31
}

// fak-test:runtime integration est=2s lane=default
func TestInk2MatVecMatchesHostReference(t *testing.T) {
	d := ink2TestDevice(t)
	store, data := ink2TestStore(t, 8)
	rng := rand.New(rand.NewSource(31))
	const experts = 4
	arena := ink2TestAlloc(t, d, 1<<14)
	for _, c := range []struct {
		name       string
		typ        uint32
		rows, cols int
		xOff       int
	}{
		{"f32 ragged", 0, 7, 37, 3},
		{"bf16", 30, 9, 50, 129},
		{"q8_0", 8, 13, 96, 64},
	} {
		m, _ := ink2TestFill(t, store, data, rng, 64, c.typ, c.rows, c.cols, experts)
		x := make([]float32, c.cols)
		for i := range x {
			x[i] = rng.Float32() - .5
		}
		xr, yr := arena.At(c.xOff), arena.At(4096)
		if err := d.Write(xr, x); err != nil {
			t.Fatal(err)
		}
		for _, e := range []int{0, experts - 1} {
			w, err := m.stack.Ink2Weight(e)
			if err != nil {
				t.Fatal(err)
			}
			if err := d.MatVec(w, xr, yr.At(e*c.rows)); err != nil {
				t.Fatalf("%s expert %d: %v", c.name, e, err)
			}
		}
		got, err := d.Read(yr, experts*c.rows)
		if err != nil {
			t.Fatal(err)
		}
		for _, e := range []int{0, experts - 1} {
			for r := range c.rows {
				want, mag := m.dot(e, r, x)
				if g := got[e*c.rows+r]; !ink2TestClose(g, want, mag) {
					t.Fatalf("%s expert %d row %d: device %v, host %v (|w||x| %v)", c.name, e, r, g, want, mag)
				}
			}
		}

		// Routed slots: per-slot inputs, ids on the device, optional scale.
		const k = 3
		ids := []int{2, 0, 3}
		stride := (c.cols + 63) &^ 63
		xs := make([]float32, k*stride)
		for i := range xs {
			xs[i] = rng.Float32() - .5
		}
		idBits := make([]float32, k)
		for i, id := range ids {
			idBits[i] = math.Float32frombits(uint32(id))
		}
		scale := []float32{1.5, -3, 0.25, 0.75}
		xsr, idr, scr, out := arena.At(8192), arena.At(12000), arena.At(12032), arena.At(12288)
		for _, w := range []struct {
			dst Ink2Ref
			src []float32
		}{{xsr, xs}, {idr, idBits}, {scr, scale}} {
			if err := d.Write(w.dst, w.src); err != nil {
				t.Fatal(err)
			}
		}
		stack, ok := m.stack.Ink2Stack()
		if !ok {
			t.Fatalf("%s: stack split across views", c.name)
		}
		for _, scaled := range []bool{false, true} {
			var sr Ink2Ref
			if scaled {
				sr = scr
			}
			ran, err := d.ExpertMatVec(stack, idr, xsr, stride, out, k, sr, nil)
			if err != nil || !ran {
				t.Fatalf("%s routed scaled=%v: ran %v, %v", c.name, scaled, ran, err)
			}
			got, err := d.Read(out, k*c.rows)
			if err != nil {
				t.Fatal(err)
			}
			for slot, id := range ids {
				for r := range c.rows {
					want, mag := m.dot(id, r, xs[slot*stride:])
					if scaled {
						want, mag = want*float64(scale[id]), mag*math.Abs(float64(scale[id]))
					}
					if g := got[slot*c.rows+r]; !ink2TestClose(g, want, mag) {
						t.Fatalf("%s routed scaled=%v slot %d row %d: device %v, host %v", c.name, scaled, slot, r, g, want)
					}
				}
			}
		}
		// One shared input row for every slot.
		if ran, err := d.ExpertMatVec(stack, idr, xr, 0, out, k, Ink2Ref{}, nil); err != nil || !ran {
			t.Fatalf("%s shared input: ran %v, %v", c.name, ran, err)
		}
		if got, err = d.Read(out, k*c.rows); err != nil {
			t.Fatal(err)
		}
		for slot, id := range ids {
			want, mag := m.dot(id, c.rows-1, x)
			if g := got[(slot+1)*c.rows-1]; !ink2TestClose(g, want, mag) {
				t.Fatalf("%s shared input slot %d: device %v, host %v", c.name, slot, g, want)
			}
		}
		if ran, err := d.ExpertMatVec(stack, idr, xsr, stride, out, k, Ink2Ref{}, &Ink2Situ{Width: 4, Beta: 1}); ran || err != nil {
			t.Fatalf("%s fused SITU: ran %v, %v; want a decline", c.name, ran, err)
		}
	}

	// A dense F32 weight held in a device buffer, as the resident forward
	// uploads its small matrices.
	const rows, cols = 5, 11
	w := make([]float32, rows*cols)
	x := make([]float32, cols)
	for i := range w {
		w[i] = rng.Float32() - .5
	}
	for i := range x {
		x[i] = rng.Float32() - .5
	}
	wr := ink2TestAlloc(t, d, 1+rows*cols).At(1)
	if err := d.Write(wr, w); err != nil {
		t.Fatal(err)
	}
	if err := d.Write(arena.At(1), x); err != nil {
		t.Fatal(err)
	}
	if err := d.MatVec(Ink2DenseF32(wr, rows, cols), arena.At(1), arena.At(4096)); err != nil {
		t.Fatal(err)
	}
	got, err := d.Read(arena.At(4096), rows)
	if err != nil {
		t.Fatal(err)
	}
	for r := range rows {
		var want, mag float64
		for j, v := range x {
			p := float64(w[r*cols+j]) * float64(v)
			want, mag = want+p, mag+math.Abs(p)
		}
		if !ink2TestClose(got[r], want, mag) {
			t.Fatalf("dense row %d: device %v, host %v", r, got[r], want)
		}
	}

	q8, _ := ink2TestFill(t, store, data, rng, 64, 8, 4, 64, 1)
	q8w, _ := q8.stack.Ink2Weight(0)
	for name, err := range map[string]error{
		"unaligned block input": d.MatVec(q8w, arena.At(3), arena.At(4096)),
		"input past the end":    d.MatVec(q8w, arena.At(1<<14-32), arena.At(4096)),
		"output past the end":   d.MatVec(q8w, arena, arena.At(1<<14-2)),
		"absent output":         d.MatVec(q8w, arena, Ink2Ref{}),
		"unknown type":          d.MatVec(Ink2Weight{Buf: arena.Buf, Type: 9, Rows: 1, Cols: 32, Experts: 1}, arena, arena.At(64)),
	} {
		if err == nil {
			t.Errorf("%s: accepted", name)
		}
	}
}

const ink2TestKernels = `#include <metal_stdlib>
using namespace metal;
struct Affine { float scale, bias; };
kernel void test_affine(device const float *x [[buffer(0)]], device float *y [[buffer(1)]],
                        constant Affine &p [[buffer(2)]], device const float *absent [[buffer(3)]],
                        uint i [[thread_position_in_grid]]) {
    y[i] = x[i] * p.scale + p.bias + absent[0];
}
`

// fak-test:runtime integration est=1s lane=default
func TestInk2DispatchesDeferUntilAFence(t *testing.T) {
	d := ink2TestDevice(t)
	const n = 300
	buf := ink2TestAlloc(t, d, 4*n)
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(i) - 7
	}
	if err := d.Write(buf, x); err != nil {
		t.Fatal(err)
	}
	p, err := d.PipelinePrecise(ink2TestKernels, "test_affine")
	if err != nil {
		t.Fatal(err)
	}
	if again, err := d.PipelinePrecise(ink2TestKernels, "test_affine"); err != nil || again != p {
		t.Fatalf("pipeline cache returned %+v, %v; want %+v", again, err, p)
	}
	type affine struct{ scale, bias float32 }
	group := [3]int{min(n, p.MaxThreads()), 1, 1}
	// The second launch reads what the first wrote, in the same command buffer.
	if err := d.DispatchThreads(p, [3]int{n, 1, 1}, group, buf.Span(n), buf.At(n).Span(n), Ink2Const(affine{2, 1}), Ink2Ref{}.Span(0)); err != nil {
		t.Fatal(err)
	}
	if err := d.DispatchThreads(p, [3]int{n, 1, 1}, group, buf.At(n).Span(n), buf.At(2*n).Span(n), Ink2Const(affine{-3, .5}), Ink2Ref{}.Span(0)); err != nil {
		t.Fatal(err)
	}
	if s := d.Stats(); s.Pending != 2 || s.Dispatches != 2 || s.Submissions != 0 {
		t.Fatalf("after two dispatches: %+v, want 2 pending and nothing submitted", s)
	}
	got, err := d.Read(buf.At(2*n), n)
	if err != nil {
		t.Fatal(err)
	}
	if s := d.Stats(); s.Pending != 0 || s.Dispatches != 2 || s.Submissions != 1 {
		t.Fatalf("after the fence: %+v, want one submission of both dispatches", s)
	}
	for i, v := range got {
		if want := (x[i]*2+1)*-3 + .5; v != want {
			t.Fatalf("chained element %d = %v, want %v", i, v, want)
		}
	}
	if _, err := d.ReadInts(buf, 1); err != nil {
		t.Fatal(err)
	}
	if err := d.Flush(); err != nil {
		t.Fatal(err)
	}
	if s := d.Stats(); s.Submissions != 1 {
		t.Fatalf("an idle fence submitted a command buffer: %+v", s)
	}

	// A host write fences first: the launch already encoded sees the old input.
	if err := d.DispatchThreads(p, [3]int{n, 1, 1}, group, buf.Span(n), buf.At(3*n).Span(n), Ink2Const(affine{1, 0}), Ink2Ref{}.Span(0)); err != nil {
		t.Fatal(err)
	}
	if err := d.Write(buf, make([]float32, n)); err != nil {
		t.Fatal(err)
	}
	if s := d.Stats(); s.Pending != 0 || s.Submissions != 2 {
		t.Fatalf("after a write over pending work: %+v, want it submitted", s)
	}
	if got, err = d.Read(buf.At(3*n), n); err != nil {
		t.Fatal(err)
	}
	for i, v := range got {
		if v != x[i] {
			t.Fatalf("element %d = %v, want the input %v from before the write", i, v, x[i])
		}
	}

	if err := d.DispatchThreads(p, [3]int{n, 1, 1}, group, buf.Span(n), buf.At(3*n).Span(n+1), Ink2Const(affine{}), Ink2Ref{}.Span(0)); err == nil {
		t.Fatal("a binding past the end of its buffer was encoded")
	}
	if s := d.Stats(); s.Pending != 0 {
		t.Fatalf("a rejected dispatch stayed pending: %+v", s)
	}
	if _, err := d.Pipeline(ink2TestKernels, "missing"); err == nil {
		t.Fatal("an unknown kernel resolved")
	}
	if _, err := d.Pipeline("kernel void broken(", "broken"); err == nil {
		t.Fatal("a source that does not compile resolved")
	}
}

// fak-test:runtime integration est=1s lane=default
func TestInk2OpsRejectAbsentOperands(t *testing.T) {
	d := ink2TestDevice(t)
	var r Ink2Ref
	for name, err := range map[string]error{
		"rms":            d.RMS(r, r, r, 1, 1, 0),
		"ctrl pair":      d.CtrlPair(r, r, r, 1),
		"mix":            d.Mix(nil, r, 1, nil, 0, r, 1),
		"control":        d.Control(Ink2Controls{}, r, r, 1, 1, nil, 0),
		"conv":           d.Conv(r, r, r, r, r, r, 1, 1, 1, 0),
		"combine conv":   d.CombineConv(r, r, 1, r, r, r, r, 1, 1, 0),
		"qkv post":       d.QKVPost(r, r, r, r, r, r, r, r, r, Ink2AttnParams{}, 1, nil),
		"attend":         d.Attend(r, r, r, r, r, r, r, r, r, Ink2AttnParams{}, 1, 1, 1),
		"sigmoid mul":    d.SigmoidMul(r, r, 1),
		"route":          d.Route(r, r, r, r, 1, 1, 0),
		"expert scale":   d.ExpertScale(r, r, r, 1, 1),
		"situ":           d.Situ(r, r, 1, 1, 1, 0, 0),
		"moe combine":    d.Combine(r, r, r, 1, 1),
		"add":            d.Add(r, r, r, 1),
		"softcap argmax": d.SoftcapArgmax(r, 1, 0, r),
		"matvec":         d.MatVec(Ink2Weight{}, r, r),
	} {
		if !errors.Is(err, ErrInk2Shape) {
			t.Errorf("%s: %v, want ErrInk2Shape", name, err)
		}
	}
	if ran, err := d.MatVecQ8(Ink2Weight{}, r, r, r, true, r); ran || err != nil {
		t.Errorf("shared Q8_0 matvec: ran %v, %v; want a decline", ran, err)
	}
	if !errors.Is(ErrInk2GraphUnsupported, errors.ErrUnsupported) {
		t.Error("graph capture error does not report unsupported")
	}
	if s := d.Stats(); s.Dispatches != 0 || s.Pending != 0 {
		t.Errorf("a rejected op encoded work: %+v", s)
	}
	d.Close()
	if err := d.Flush(); !errors.Is(err, ErrInk2Closed) {
		t.Errorf("flush after close: %v", err)
	}
}

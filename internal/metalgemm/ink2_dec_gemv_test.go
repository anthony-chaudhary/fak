//go:build darwin && arm64 && cgo

package metalgemm

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"os"
	"regexp"
	"slices"
	"strconv"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gpulease"
)

func ink2DecUseLegacy(t *testing.T, set ink2DecLegacySet) {
	t.Helper()
	old := ink2DecLegacy
	ink2DecLegacy = set
	t.Cleanup(func() { ink2DecLegacy = old })
}

func ink2DecHost(t *testing.T, d *Ink2Device, b Ink2Buf, n int) []byte {
	t.Helper()
	d.mu.Lock()
	defer d.mu.Unlock()
	host, err := d.host(Ink2Ref{Buf: b}, n)
	if err != nil {
		t.Fatal(err)
	}
	return host
}

// ink2DecScaleAt lists where a block of a GGML type keeps half-float scales.
var ink2DecScaleAt = map[uint32][]int{8: {0}, 11: {108}, 12: {0, 2}, 13: {0, 2}, 14: {208},
	17: {0}, 18: {0}, 20: {0}, 21: {0}, 22: {0}, 40: nil}

// ink2DecRandom fills dst with random weights of a GGML type under finite,
// moderate scales; NVFP4 takes every scale code, the zero code included.
func ink2DecRandom(t *testing.T, dst []byte, typ uint32, seed uint64) {
	t.Helper()
	s := seed*0x9E3779B97F4A7C15 + 1
	next := func() uint64 {
		s ^= s << 13
		s ^= s >> 7
		s ^= s << 17
		return s
	}
	switch typ {
	case 0:
		for i := 0; i+4 <= len(dst); i += 4 {
			binary.LittleEndian.PutUint32(dst[i:], math.Float32bits(float32(next()>>40)/(1<<24)-.5))
		}
		return
	case 30:
		for i := 0; i+2 <= len(dst); i += 2 {
			binary.LittleEndian.PutUint16(dst[i:], uint16(math.Float32bits(float32(next()>>40)/(1<<24)-.5)>>16))
		}
		return
	case 1:
		for i := 0; i+2 <= len(dst); i += 2 {
			r := next()
			binary.LittleEndian.PutUint16(dst[i:], uint16(r&0x83ff|(8+r>>20&7)<<10))
		}
		return
	}
	at, ok := ink2DecScaleAt[typ]
	if !ok {
		t.Fatalf("no synthesizer for GGML type %d", typ)
	}
	_, blockBytes, err := packedBlock(typ)
	if err != nil {
		t.Fatal(err)
	}
	i := 0
	for ; i+8 <= len(dst); i += 8 {
		binary.LittleEndian.PutUint64(dst[i:], next())
	}
	for ; i < len(dst); i++ {
		dst[i] = byte(next())
	}
	for b := 0; b+blockBytes <= len(dst); b += blockBytes {
		for _, o := range at {
			binary.LittleEndian.PutUint16(dst[b+o:], uint16(0x3000+next()%1024))
		}
	}
}

var ink2DecE2M1 = [16]float64{0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6}

// ink2DecNVFP4 is the float64 value of every weight of an NVFP4 matrix.
func ink2DecNVFP4(raw []byte, n int) []float64 {
	ref := make([]float64, n)
	for i := range ref {
		b, g, j := raw[i/64*36:], i%64/16, i%16
		sc := int(b[g] & 127)
		var d float64
		switch e, m := sc>>3, sc&7; {
		case sc == 127:
		case e == 0:
			d = float64(m) / 512
		default:
			d = math.Ldexp(float64(8+m), e-10)
		}
		q := b[4+8*g+j%8]
		if j >= 8 {
			q >>= 4
		}
		ref[i] = ink2DecE2M1[q&15] * d
	}
	return ref
}

type ink2DecCase struct {
	w   Ink2Weight
	raw []byte
}

func ink2DecWeight(t *testing.T, d *Ink2Device, typ uint32, rows, cols, experts, byteOff int, seed uint64) ink2DecCase {
	t.Helper()
	block, blockBytes, err := packedBlock(typ)
	if err != nil {
		t.Fatal(err)
	}
	if cols%block != 0 {
		t.Fatalf("type %d needs a multiple of %d columns, got %d", typ, block, cols)
	}
	bytes := experts * rows * (cols / block * blockBytes)
	n := (byteOff+bytes+3)/4 + 16
	buf, err := d.Alloc(n)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { d.Free(buf) })
	raw := ink2DecHost(t, d, buf, n)[byteOff:][:bytes]
	ink2DecRandom(t, raw, typ, seed)
	return ink2DecCase{w: Ink2Weight{Buf: buf, ByteOff: byteOff, Type: typ, Rows: rows, Cols: cols, Experts: experts}, raw: raw}
}

func ink2DecIDs(ids []int) []float32 {
	out := make([]float32, len(ids))
	for i, id := range ids {
		out[i] = math.Float32frombits(uint32(id))
	}
	return out
}

type ink2DecOperands struct {
	x, ids, scale, y Ink2Ref
	stride           int
	xs, scales       []float32
	idList           []int
}

func ink2DecPut(t *testing.T, d *Ink2Device, rng *rand.Rand, cols, k, experts, yFloats int) *ink2DecOperands {
	t.Helper()
	a := &ink2DecOperands{stride: (cols + 63) &^ 63}
	arena := ink2TestAlloc(t, d, k*a.stride+4096+experts+yFloats+64)
	a.x, a.ids, a.scale = arena, arena.At(k*a.stride), arena.At(k*a.stride+2048)
	a.y = arena.At(k*a.stride + 4096 + experts)
	a.xs, a.scales = make([]float32, k*a.stride), make([]float32, experts)
	for i := range a.xs {
		a.xs[i] = rng.Float32() - .5
	}
	for i := range a.scales {
		a.scales[i] = 4*rng.Float32() - 2
	}
	a.idList = rng.Perm(experts)[:k]
	for _, w := range []struct {
		dst Ink2Ref
		src []float32
	}{{a.x, a.xs}, {a.ids, ink2DecIDs(a.idList)}, {a.scale, a.scales}} {
		if err := d.Write(w.dst, w.src); err != nil {
			t.Fatal(err)
		}
	}
	return a
}

func ink2DecDiffer(a, b []float32) (differ int) {
	for i := range a {
		if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
			differ++
		}
	}
	return differ
}

// The routed_ kernel then the separate scale pass, the product every routed
// kernel here must reproduce bit for bit.
func ink2DecPrevious(t *testing.T, d *Ink2Device, w Ink2Weight, a *ink2DecOperands, xStride, k int, scaled bool) []float32 {
	t.Helper()
	old := ink2DecLegacy
	ink2DecLegacy = ink2DecLegacySet{routed: true}
	defer func() { ink2DecLegacy = old }()
	var sr Ink2Ref
	if scaled {
		sr = a.scale
	}
	if ran, err := d.ExpertMatVec(w, a.ids, a.x, xStride, a.y, k, sr, nil); err != nil || !ran {
		t.Fatalf("previous routed: ran %v, %v", ran, err)
	}
	got, err := d.Read(a.y, k*w.Rows)
	if err != nil {
		t.Fatal(err)
	}
	return got
}

var ink2DecNVFP4Arms = []ink2DecPlan{
	{kernel: "ink2_dec_routed_nvfp4", rowBits: 2}, {kernel: "ink2_dec_routed_nvfp4w", rowBits: 2}, {kernel: "ink2_dec_routed_nvfp4w", rowBits: 0}, {kernel: "ink2_dec_routed_nvfp4w", rowBits: 1},
	{kernel: "ink2_dec_routed_nvfp4w", rowBits: 3}, {kernel: "ink2_dec_routed_nvfp4u", rowBits: 2}, {kernel: "ink2_dec_routed_nvfp4u", rowBits: 3},
	{kernel: "ink2_dec_routed_nvfp4t", rowBits: 2}, {kernel: "ink2_dec_routed_nvfp4t", rowBits: 3}, {kernel: "ink2_dec_routed_nvfp4t", rowBits: 4},
}

// ink2DecShapedArms lists every compile-time-shaped kernel of a type at cols
// under each threadgroup of 1, 2 and 4 SIMD groups of two rows each.
func ink2DecShapedArms(typ uint32, cols int) (arms []ink2DecPlan) {
	if _, ok := ink2DecShaped[ink2DecShapeKey{typ, cols}]; !ok {
		return nil
	}
	name := ink2RoutedKernel[typ][len("routed_"):]
	for rowBits := range 3 {
		arms = append(arms, ink2DecPlan{fmt.Sprintf("ink2_dec_routed_%sc%dr2", name, cols), rowBits, 2, cols})
	}
	return arms
}

// fak-test:runtime integration est=10s lane=default
func TestInk2DecRoutedMatchesRoutedAndScaleBitForBit(t *testing.T) {
	d := ink2TestDevice(t)
	ink2DecUseLegacy(t, ink2DecLegacySet{})
	rng := rand.New(rand.NewSource(53))
	const experts, k = 6, 4
	compared := 0
	for round := range 4 {
		rows := 61 + 4*round
		for _, typ := range []uint32{0, 1, 30, 8, 11, 12, 13, 14, 17, 18, 20, 21, 22, 40} {
			cols := []int{2304, 1152, 512, 1792}[round]
			if block, _, _ := packedBlock(typ); cols%block != 0 {
				cols = 1280
			}
			// Every type keeps 2-byte alignment and NVFP4 none.
			off := 64 + 2*round
			switch typ {
			case 0:
				off = 64
			case 40:
				off = 64 + round
			}
			m := ink2DecWeight(t, d, typ, rows, cols, experts, off, uint64(rng.Int63()))
			a := ink2DecPut(t, d, rng, cols, k, experts, k*rows)
			arms := append([]ink2DecPlan{ink2DecRoutedPick(typ, cols)}, ink2DecShapedArms(typ, cols)...)
			if typ == 40 {
				arms = append(arms, ink2DecNVFP4Arms...)
			}
			for _, xStride := range []int{a.stride, 0} {
				for _, scaled := range []bool{true, false} {
					want := ink2DecPrevious(t, d, m.w, a, xStride, k, scaled)
					finite := 0
					for _, v := range want {
						if v != 0 && !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0) {
							finite++
						}
					}
					if finite < len(want)/2 {
						t.Fatalf("type %d: only %d of %d previous outputs are finite and nonzero", typ, finite, len(want))
					}
					for _, pl := range arms {
						var sr Ink2Ref
						if scaled {
							sr = a.scale
						}
						if err := d.routedDec(pl, m.w, a.ids, a.x, xStride, a.y, k, sr); err != nil {
							t.Fatalf("type %d %+v: %v", typ, pl, err)
						}
						got, err := d.Read(a.y, k*rows)
						if err != nil {
							t.Fatal(err)
						}
						if n := ink2DecDiffer(got, want); n != 0 {
							t.Errorf("type %d %dx%d %s R%d stride %d scaled %v: %d of %d outputs differ in bits",
								typ, rows, cols, pl.kernel, 1<<pl.rowBits, xStride, scaled, n, len(want))
						}
						compared += len(want)
					}
				}
			}
			if typ != 40 {
				continue
			}
			// The host decode bounds what both forms compute.
			ref := ink2DecNVFP4(m.raw, experts*rows*cols)
			got := ink2DecPrevious(t, d, m.w, a, a.stride, k, true)
			for slot, id := range a.idList {
				for r := range rows {
					var sum, mag float64
					for j, w := range ref[(id*rows+r)*cols:][:cols] {
						p := w * float64(a.xs[slot*a.stride+j])
						sum, mag = sum+p, mag+math.Abs(p)
					}
					s := float64(a.scales[id])
					if g := got[slot*rows+r]; !ink2TestClose(g, sum*s, mag*math.Abs(s)) {
						t.Fatalf("nvfp4 %dx%d slot %d row %d: device %v, host %v", rows, cols, slot, r, g, sum*s)
					}
				}
			}
		}
	}
	t.Logf("%d outputs compared bit for bit", compared)
}

// fak-test:runtime integration est=2s lane=default
func TestInk2DecRoutedFoldsTheScaleDispatchUnderEveryLegacyArm(t *testing.T) {
	d := ink2TestDevice(t)
	rng := rand.New(rand.NewSource(59))
	const experts, k, rows, cols = 5, 3, 45, 1152
	m := ink2DecWeight(t, d, 40, rows, cols, experts, 64, uint64(rng.Int63()))
	a := ink2DecPut(t, d, rng, cols, k, experts, k*rows)
	want := ink2DecPrevious(t, d, m.w, a, a.stride, k, true)
	for _, c := range []struct {
		legacy     ink2DecLegacySet
		dispatches uint64
	}{{ink2DecLegacySet{}, 1}, {ink2DecLegacySet{nvfp4w: true}, 1}, {ink2DecLegacySet{scale: true}, 2},
		{ink2DecLegacySet{scale: true, nvfp4w: true}, 2}, {ink2DecLegacySet{routed: true}, 2}} {
		ink2DecUseLegacy(t, c.legacy)
		before := d.Stats().Dispatches
		if ran, err := d.ExpertMatVec(m.w, a.ids, a.x, a.stride, a.y, k, a.scale, nil); err != nil || !ran {
			t.Fatalf("legacy %+v: ran %v, %v", c.legacy, ran, err)
		}
		if n := d.Stats().Dispatches - before; n != c.dispatches {
			t.Errorf("legacy %+v: %d dispatches, want %d", c.legacy, n, c.dispatches)
		}
		got, err := d.Read(a.y, k*rows)
		if err != nil {
			t.Fatal(err)
		}
		if n := ink2DecDiffer(got, want); n != 0 {
			t.Errorf("legacy %+v: %d of %d outputs differ in bits", c.legacy, n, len(want))
		}
	}
}

// fak-test:runtime unit est=1s lane=default
func TestInk2DecLegacyNames(t *testing.T) {
	if got := parseInk2DecLegacy(" Scale, mix ,nvfp4w"); got != (ink2DecLegacySet{scale: true, nvfp4w: true}) {
		t.Fatalf("parsed %+v", got)
	}
	if got := parseInk2DecLegacy("routed"); got != (ink2DecLegacySet{routed: true}) {
		t.Fatalf("parsed %+v", got)
	}
	if got := parseInk2DecLegacy("routed2"); got != (ink2DecLegacySet{routed2: true}) {
		t.Fatalf("parsed %+v", got)
	}
	ink2DecUseLegacy(t, ink2DecLegacySet{})
	for typ, want := range map[uint32]string{40: ink2DecRoutedNVFP4.kernel, 8: "ink2_dec_routed_q8_0", 0: "ink2_dec_routed_f32", 9: ""} {
		if got := ink2DecRoutedPick(typ, 512).kernel; got != want {
			t.Errorf("type %d: planned %q, want %q", typ, got, want)
		}
	}
	for key, want := range ink2DecShaped {
		if got := ink2DecRoutedPick(key.typ, key.cols); got != want {
			t.Errorf("type %d at %d columns: planned %+v, want %+v", key.typ, key.cols, got, want)
		}
	}
	ink2DecLegacy = ink2DecLegacySet{routed2: true}
	if got := ink2DecRoutedPick(18, 2304).kernel; got != "ink2_dec_routed_iq3_xxs" {
		t.Errorf("routed2 legacy: planned %q", got)
	}
	ink2DecLegacy = ink2DecLegacySet{nvfp4w: true}
	if got := ink2DecRoutedPick(40, 2304).kernel; got != "ink2_dec_routed_nvfp4" {
		t.Errorf("nvfp4w legacy at 2304 columns: planned %q", got)
	}
	if got := ink2DecRoutedPick(40, 512).kernel; got != "ink2_dec_routed_nvfp4" {
		t.Errorf("nvfp4w legacy: planned %q", got)
	}
	ink2DecLegacy = ink2DecLegacySet{routed: true}
	if got := ink2DecRoutedPick(40, 512).kernel; got != "" {
		t.Errorf("routed legacy: planned %q", got)
	}
}

// ---- microbenchmark ----

const ink2DecBenchFillSource = `#include <metal_stdlib>
using namespace metal;
kernel void ink2_dec_bench_fill(device const float4 *src [[buffer(0)]], device float *out [[buffer(1)]],
                                constant uint &per [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    float4 a = 0;
    for (uint j = 0; j < per; j++) a += src[i*per + j];
    if (a.x == 12345.f) out[i & 1023] = a.y;
}
`

const ink2DecBenchFiller = "filler"

type ink2DecBenchArm struct {
	name string
	run  func(i int) error
}

// Times every arm interleaved, forward then backward each repetition, after a
// streaming burst that brings the GPU to speed, and returns the median and
// best GPU microseconds per call.
func ink2DecBenchRun(t *testing.T, d *Ink2Device, arms []ink2DecBenchArm, perSample, reps int, fill func() error) (median, best []float64) {
	t.Helper()
	sample := func(a ink2DecBenchArm, start int) float64 {
		if err := d.ResetProfile(); err != nil {
			t.Fatal(err)
		}
		if err := fill(); err != nil {
			t.Fatal(err)
		}
		for i := range perSample {
			if err := a.run(start + i); err != nil {
				t.Fatalf("%s: %v", a.name, err)
			}
		}
		if err := d.Flush(); err != nil {
			t.Fatal(err)
		}
		p := d.Profile()
		if p.Unsampled != 0 {
			t.Fatalf("%s: %d dispatches without timestamps", a.name, p.Unsampled)
		}
		var gpu float64
		for _, r := range p.Rows {
			if r.Detail != ink2DecBenchFiller {
				gpu += r.GPUSeconds
			}
		}
		return gpu / float64(perSample) * 1e6
	}
	for _, a := range arms {
		sample(a, 0)
	}
	samples := make([][]float64, len(arms))
	for rep := range reps {
		for i := range arms {
			samples[i] = append(samples[i], sample(arms[i], rep*perSample))
		}
		for i := len(arms) - 1; i >= 0; i-- {
			samples[i] = append(samples[i], sample(arms[i], rep*perSample+perSample/2))
		}
	}
	for _, s := range samples {
		slices.Sort(s)
		median, best = append(median, (s[(len(s)-1)/2]+s[len(s)/2])/2), append(best, s[0])
	}
	return median, best
}

// Times the routed kernels at the resident forward's shapes: the routed_
// kernel with its scale pass against every bit-identical arm.
// FAK_INK2_MICROBENCH_REPS sets the repetitions, FAK_INK2_MICROBENCH_ONLY
// (regexp) the arms.
//
// fak-test:runtime slow est=300s
func TestInk2DecRoutedMicrobench(t *testing.T) {
	if os.Getenv("FAK_INK2_MICROBENCH") != "1" {
		t.Skip("set FAK_INK2_MICROBENCH=1 to time the routed kernels on the GPU")
	}
	if !Available() {
		t.Skip("requires a Metal device")
	}
	reps := 8
	if s := os.Getenv("FAK_INK2_MICROBENCH_REPS"); s != "" {
		var err error
		if reps, err = strconv.Atoi(s); err != nil || reps < 1 {
			t.Fatalf("FAK_INK2_MICROBENCH_REPS=%q", s)
		}
	}
	only := regexp.MustCompile(os.Getenv("FAK_INK2_MICROBENCH_ONLY"))
	lease, err := gpulease.Acquire(gpulease.Options{Timeout: 20 * time.Minute, Logf: t.Logf})
	if errors.Is(err, gpulease.ErrTimeout) {
		t.Skipf("NOT RUN: %v", err)
	}
	if err != nil {
		t.Fatal(err)
	}
	defer lease.Release()
	d, err := newInk2Device(Ink2ProfilePass)
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	if d.ProfileMode() != Ink2ProfilePass {
		t.Skip("the device cannot sample GPU time per dispatch")
	}

	const fillBytes, fillPer = 64 << 20, 16
	fillBuf, err := d.Alloc(fillBytes / 4)
	if err != nil {
		t.Fatal(err)
	}
	defer d.Free(fillBuf)

	const experts, k, sets, perSample = 128, 8, 16, 8
	for _, shape := range [][3]int{{40, 2304, 2304}, {40, 2304, 1152}, {18, 2304, 2304}, {22, 2304, 2304},
		{21, 2304, 2304}, {11, 2304, 2304}, {17, 2304, 2304}, {20, 2304, 1152}} {
		typ, rows, cols := uint32(shape[0]), shape[1], shape[2]
		block, blockBytes, err := packedBlock(typ)
		if err != nil {
			t.Fatal(err)
		}
		rowBytes := cols / block * blockBytes
		total := experts * rows * rowBytes
		buf, err := d.Alloc(total/4 + 16)
		if err != nil {
			t.Fatal(err)
		}
		ink2DecRandom(t, ink2DecHost(t, d, buf, total/4)[:total], typ, uint64(cols))
		stride := (cols + 63) &^ 63
		yOff := k*stride + 4096
		arena := ink2TestAlloc(t, d, yOff+k*rows+2048)
		rng := rand.New(rand.NewSource(int64(cols)))
		xs := make([]float32, k*stride)
		for i := range xs {
			xs[i] = rng.Float32() - .5
		}
		var idBits []float32
		for range sets {
			idBits = append(idBits, ink2DecIDs(rng.Perm(experts)[:k])...)
		}
		scales := make([]float32, experts)
		for i := range scales {
			scales[i] = rng.Float32() + .5
		}
		x, ids, scale, y := arena, arena.At(k*stride), arena.At(k*stride+2048), arena.At(yOff)
		for _, w := range []struct {
			dst Ink2Ref
			src []float32
		}{{x, xs}, {ids, idBits}, {scale, scales}} {
			if err := d.Write(w.dst, w.src); err != nil {
				t.Fatal(err)
			}
		}
		stack := Ink2Weight{Buf: buf, Type: typ, Rows: rows, Cols: cols, Experts: experts}
		fillOut := arena.At(yOff + k*rows)
		fill := func() error {
			p, err := d.Pipeline(ink2DecBenchFillSource, "ink2_dec_bench_fill")
			if err != nil {
				return err
			}
			d.profileDetail(ink2DecBenchFiller)
			return d.DispatchThreads(p, [3]int{fillBytes / 16 / fillPer, 1, 1}, [3]int{256, 1, 1},
				Ink2BufferAt(fillBuf, 0, fillBytes), fillOut.Span(1024), Ink2Const(uint32(fillPer)))
		}
		arms := []ink2DecBenchArm{{"routed_nvfp4 + scale pass", func(i int) error {
			old := ink2DecLegacy
			ink2DecLegacy = ink2DecLegacySet{routed: true}
			defer func() { ink2DecLegacy = old }()
			_, err := d.ExpertMatVec(stack, ids.At(i%sets*k), x, stride, y, k, scale, nil)
			return err
		}}}
		plans := append([]ink2DecPlan{{kernel: "ink2_dec_routed_" + ink2RoutedKernel[typ][len("routed_"):], rowBits: 2}},
			ink2DecShapedArms(typ, cols)...)
		if typ == 40 {
			plans = append(ink2DecNVFP4Arms, ink2DecShapedArms(typ, cols)...)
		}
		for _, pl := range plans {
			arms = append(arms, ink2DecBenchArm{fmt.Sprintf("%s G%d S%d", pl.kernel[len("ink2_dec_routed_"):], 1<<pl.rowBits, pl.rowsPerSIMD()), func(i int) error {
				return d.routedDec(pl, stack, ids.At(i%sets*k), x, stride, y, k, scale)
			}})
		}
		arms = slices.DeleteFunc(arms, func(a ink2DecBenchArm) bool { return !only.MatchString(a.name) })
		median, best := ink2DecBenchRun(t, d, arms, perSample, reps, fill)
		bytes := float64(k * rows * rowBytes)
		for i, a := range arms {
			t.Logf("MICROBENCH %s %dx%d k=%d | %-26s | median %7.1f us | best %7.1f us | %6.1f GB/s best | x%.3f median vs routed_",
				ink2RoutedKernel[typ][len("routed_"):], rows, cols, k, a.name, median[i], best[i], bytes/best[i]/1e3, median[0]/median[i])
		}
		d.Free(buf)
	}
}

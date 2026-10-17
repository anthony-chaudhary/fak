package compute

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/computetrace"
)

const (
	cpuRefBenchOut = 1024
	cpuRefBenchIn  = 4096
)

// cpuRefBenchKQuant builds a deterministic raw k-quant weight whose f16 scale fields hold a
// small finite value, so the dequant arithmetic never runs on NaN/Inf.
func cpuRefBenchKQuant(blockBytes int, scaleOffsets ...int) []byte {
	raw := make([]byte, cpuRefBenchOut*(cpuRefBenchIn/q4kSuper)*blockBytes)
	rand.New(rand.NewSource(7)).Read(raw)
	for off := 0; off < len(raw); off += blockBytes {
		for _, so := range scaleOffsets {
			binary.LittleEndian.PutUint16(raw[off+so:], 0x1c00)
		}
	}
	return raw
}

func cpuRefBenchX(n int) Tensor {
	x := make([]float32, n)
	r := rand.New(rand.NewSource(11))
	for i := range x {
		x[i] = r.Float32()*2 - 1
	}
	return NewF32(&cpuBackend{}, []int{n}, x)
}

func cpuRefBenchWeights() map[string]Tensor {
	be := &cpuBackend{}
	shape := []int{cpuRefBenchOut, cpuRefBenchIn}
	return map[string]Tensor{
		"Q4_K": NewQ4K(be, shape, cpuRefBenchKQuant(q4kSuperBlock, 0, 2)),
		"Q5_K": NewQ5K(be, shape, cpuRefBenchKQuant(176, 0, 2)),
		"Q6_K": NewQ6K(be, shape, cpuRefBenchKQuant(210, 208)),
	}
}

func BenchmarkCPURefMatMulKQuant(b *testing.B) {
	be := &cpuBackend{}
	x := cpuRefBenchX(cpuRefBenchIn)
	ws := cpuRefBenchWeights()
	for _, name := range []string{"Q4_K", "Q5_K", "Q6_K"} {
		w := ws[name]
		b.Run(name, func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				be.MatMul(w, x)
			}
		})
	}
}

func BenchmarkCPURefBatchedMatMulQ4K(b *testing.B) {
	const p = 4
	be := &cpuBackend{}
	w := cpuRefBenchWeights()["Q4_K"]
	x := cpuRefBenchX(p * cpuRefBenchIn)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		be.BatchedMatMul(w, x, p)
	}
}

// cpuRefPackedWeights builds one host weight for every packed dtype cpu-ref MatMul admits.
func cpuRefPackedWeights(out, in int) map[Dtype]Tensor {
	be := &cpuBackend{}
	shape := []int{out, in}
	raw := func(blockBytes int) []byte {
		b := make([]byte, out*(in/q4kSuper)*blockBytes)
		rand.New(rand.NewSource(int64(blockBytes))).Read(b)
		return b
	}
	w := make([]float32, out*in)
	for i := range w {
		w[i] = float32(i%7) - 3
	}
	ws := map[Dtype]Tensor{
		Q4_K:    NewQ4K(be, shape, raw(q4kSuperBlock)),
		Q5_K:    NewQ5K(be, shape, raw(176)),
		Q6_K:    NewQ6K(be, shape, raw(210)),
		Q2_K:    NewQ2K(be, shape, raw(q2kSuperBlock)),
		Q3_K:    NewQ3K(be, shape, raw(q3kSuperBlock)),
		IQ2_XXS: NewIQ2XXS(be, shape, raw(iq2xxsSuperBlock)),
		Q2_0:    QuantizeQ2(be, shape, w, q2Block),
	}
	for dt, f := range rawIQFormats {
		if _, ok := ws[dt]; !ok {
			ws[dt] = NewRawIQ(be, dt, shape, raw(f.blockBytes))
		}
	}
	return ws
}

// fak-test:runtime fast est=1s
func TestCPURefMatMulKQuantReadsWeightInPlace(t *testing.T) {
	const out, in, p = 64, 512, 2
	be := &cpuBackend{}
	x := cpuRefBenchX(in)
	xb := cpuRefBenchX(p * in)
	for dt, w := range cpuRefPackedWeights(out, in) {
		if !be.SupportsDeviceWeightDtype(dt) {
			t.Fatalf("%s built but not admitted by SupportsDeviceWeightDtype", dt)
		}
		if allocs := testing.AllocsPerRun(5, func() { be.MatMul(w, x) }); allocs > 3 {
			t.Errorf("%s MatMul allocs/op = %v, want <= 3 (output row + tensor only)", dt, allocs)
		}
		if allocs := testing.AllocsPerRun(5, func() { be.BatchedMatMul(w, xb, p) }); allocs > 3 {
			t.Errorf("%s BatchedMatMul allocs/op = %v, want <= 3 (output panel + tensor only)", dt, allocs)
		}
	}
}

// fak-test:runtime fast est=1s
func TestCPURefMatMulFeedsAttachedObserver(t *testing.T) {
	var got []computetrace.Event
	computetrace.SetObserver(func(e computetrace.Event) { got = append(got, e) })
	defer computetrace.SetObserver(nil)
	be := &cpuBackend{}
	be.MatMul(cpuRefBenchWeights()["Q4_K"], cpuRefBenchX(cpuRefBenchIn))
	if len(got) != 1 || got[0].Operation != "matmul" || got[0].Kernel != Q4_K.String()+"_matmul" {
		t.Fatalf("observer events = %+v, want one Q4_K matmul", got)
	}
}

// cpuRefAttentionFixture seeds a KV store with nPos positions for a GQA geometry and returns a
// query vector, so the Attention ceiling exercises the real per-head loop rather than a trivial case.
func cpuRefAttentionFixture(c *cpuBackend, cfg KVConfig, grp, nPos int) (Tensor, KVStore) {
	w := cfg.NumKVHeads * cfg.HeadDim
	kv := c.NewKV(cfg)
	var s lcg = 101
	for p := 0; p < nPos; p++ {
		kr := randVec(&s, w)
		kv.AppendKV(0, NewF32(c, []int{w}, kr), NewF32(c, []int{w}, append([]float32(nil), kr...)), NewF32(c, []int{w}, randVec(&s, w)), p)
	}
	nH := grp * cfg.NumKVHeads
	q := NewF32(c, []int{nH * cfg.HeadDim}, randVec(&s, nH*cfg.HeadDim))
	return q, kv
}

// TestCPURefAttentionReusesScoresScratch pins the attention scores buffer to one allocation per
// call instead of one per query head: the allocation COUNT must be independent of the query-head
// count. On the parent (scores allocated inside the head loop) it grows by one per head. Every entry
// is overwritten before use and softmaxInPlace mutates in place, so reuse is bit-identical —
// asserted against a per-head-fresh-buffer oracle.
// fak-test:runtime fast est=1s
func TestCPURefAttentionReusesScoresScratch(t *testing.T) {
	c := &cpuBackend{}
	cfg := KVConfig{NumLayers: 1, NumKVHeads: 2, HeadDim: 8, RopeTheta: 10000}
	const nPos = 7
	scale := float32(1.0 / math.Sqrt(float64(cfg.HeadDim)))
	qSmall, kvSmall := cpuRefAttentionFixture(c, cfg, 1, nPos)
	qBig, kvBig := cpuRefAttentionFixture(c, cfg, 8, nPos)

	// Reused buffer => the count is dominated by the fixed output/view envelope and does not track
	// the 14 extra heads. On the parent it would grow by ~14.
	small := testing.AllocsPerRun(20, func() { c.Attention(qSmall, kvSmall, 0, true, 1, scale) })
	big := testing.AllocsPerRun(20, func() { c.Attention(qBig, kvBig, 0, true, 8, scale) })
	if big-small > 4 {
		t.Errorf("Attention allocs/op grows with head count: grp=1 -> %v, grp=8 -> %v (scores buffer not reused)", small, big)
	}
	// Seeded parent shape: a per-head `make([]float32, nPos)` that escapes (as the real scores
	// buffer does when passed to softmaxInPlace) makes the count grow with nH, so this proxy for the
	// pre-fix loop is rejected by the same threshold the check above uses.
	var sink [][]float32
	parent := func(nH int) float64 {
		return testing.AllocsPerRun(20, func() {
			sink = sink[:0]
			for h := 0; h < nH; h++ {
				sink = append(sink, make([]float32, nPos))
			}
		})
	}
	if parent(8)-parent(1) <= 4 {
		t.Fatalf("seeded per-head allocation does not grow with head count; the ceiling would not catch it")
	}
	_ = sink

	// Bit-identity against a reference that allocates a fresh scores buffer per head.
	const grp = 4
	nH := grp * cfg.NumKVHeads
	q, kv := cpuRefAttentionFixture(c, cfg, grp, nPos)
	got := c.Read(c.Attention(q, kv, 0, true, grp, scale))
	ck := kv.(*cpuKV)
	qf := c.f32(q)
	w, hd := cfg.NumKVHeads*cfg.HeadDim, cfg.HeadDim
	want := make([]float32, nH*hd)
	for h := 0; h < nH; h++ {
		kvh := h / grp
		qh := qf[h*hd : (h+1)*hd]
		sc := make([]float32, nPos)
		for j := 0; j < nPos; j++ {
			sc[j] = dot(qh, ck.K[0][j*w+kvh*hd:j*w+(kvh+1)*hd]) * scale
		}
		softmaxInPlace(sc)
		for j := 0; j < nPos; j++ {
			vh := ck.V[0][j*w+kvh*hd : j*w+(kvh+1)*hd]
			for d := 0; d < hd; d++ {
				want[h*hd+d] += sc[j] * vh[d]
			}
		}
	}
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("Attention[%d] drift: got %v want %v", i, got[i], want[i])
		}
	}
}

// TestCPURefQ8MatMulReusesActivationScratch pins the Q8_0 activation quantization to a pooled
// buffer. The activation is wide relative to the output row, so a per-call quantization into fresh
// int8+codes slices would allocate about the activation's own bytes; reuse keeps the per-op bytes to
// the output envelope. The ceiling sits between the two, and a zero block is explicitly cleared so
// reuse stays bit-identical (covered by TestAllocCeilingCPURefMatMulReadsWeightInPlace parity).
// fak-test:runtime fast est=1s
func TestCPURefQ8MatMulReusesActivationScratch(t *testing.T) {
	const out, in, p = 64, 4096, 2
	be := &cpuBackend{}
	x := cpuRefBenchX(in)
	xb := cpuRefBenchX(p * in)
	w := QuantizeQ8(be, []int{out, in}, func() []float32 {
		f := make([]float32, out*in)
		for i := range f {
			f[i] = float32(i%9) - 4
		}
		return f
	}(), 32)

	// A fresh per-call activation alloc is ~in int8 + in/32 float32 scales (576 B here for in=512;
	// ~4.6 KB for in=4096). The ceiling is set at a quarter of that, well above the output envelope.
	scratch := int64(in) + 4*int64(in/32)
	ceiling := scratch / 4
	mm := testing.Benchmark(func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			be.MatMul(w, x)
		}
	})
	if per := mm.AllocedBytesPerOp(); per >= ceiling {
		t.Errorf("Q8_0 MatMul allocates %d B/op >= ceiling %d (activation scratch %d): quantizing fresh per call", per, ceiling, scratch)
	}
	bm := testing.Benchmark(func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			be.BatchedMatMul(w, xb, p)
		}
	})
	if per := bm.AllocedBytesPerOp(); per >= ceiling {
		t.Errorf("Q8_0 BatchedMatMul allocates %d B/op >= ceiling %d (activation scratch %d): quantizing fresh per call", per, ceiling, scratch)
	}
}

// TestCPURefRoPECachesRotationTable pins RoPE to a cached cos/sin table. On the parent ropeRow
// recomputes the table with math.Pow/Cos/Sin and allocates two slices per call; the ceiling here
// allows only the output copy + tensor envelope, so a per-call table turns it red. Bit-identity is
// asserted against a direct ropeRow oracle for two different geometries (cache-key isolation).
// fak-test:runtime fast est=1s
func TestCPURefRoPECachesRotationTable(t *testing.T) {
	c := &cpuBackend{}
	const heads, dim = 4, 16
	const theta = 10000
	var s lcg = 7
	x := randVec(&s, heads*dim)
	tens := NewF32(c, []int{heads, dim}, x)

	// Warm the cache, then measure: only the output copy + tensor envelope remain.
	c.RoPE(tens, 3, heads, dim, theta)
	if allocs := testing.AllocsPerRun(5, func() { c.RoPE(tens, 3, heads, dim, theta) }); allocs > 4 {
		t.Errorf("RoPE allocs/op = %v, want <= 4 (rotation table cached)", allocs)
	}
	// Seeded parent shape: ropeRow allocates a fresh cos+sin pair per call ("2"), which alone is
	// within the cached envelope's budget only if the table is NOT rebuilt; the two allocations
	// above budget are what the cache removes.
	if parent := testing.AllocsPerRun(5, func() { _, _ = ropeRow(theta, dim, 3) }); parent < 2 {
		t.Fatalf("ropeRow allocates %v slices, want >= 2; the cached-table ceiling would not catch a rebuild", parent)
	}

	for _, geo := range []struct {
		dim   int
		theta float64
	}{{16, 10000}, {8, 500000}} {
		nHeads := heads * dim / geo.dim
		xx := randVec(&s, nHeads*geo.dim)
		tt := NewF32(c, []int{nHeads, geo.dim}, xx)
		got := c.Read(c.RoPE(tt, 5, nHeads, geo.dim, geo.theta))
		cos, sin := ropeRow(geo.theta, geo.dim, 5)
		want := append([]float32(nil), xx...)
		for h := 0; h < nHeads; h++ {
			applyRope(want[h*geo.dim:(h+1)*geo.dim], cos, sin)
		}
		for i := range want {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("RoPE dim=%d theta=%g [%d] bits=%08x want %08x", geo.dim, geo.theta, i, math.Float32bits(got[i]), math.Float32bits(want[i]))
			}
		}
	}
}

//go:build darwin && arm64 && cgo

package model

// metal_q4k_test.go — the correctness gate for the Metal q4_k dequant-GEMV/GEMM
// (internal/metalgemm/q4k.m). The GPU kernel reconstructs each weight row's f32 values from
// the SAME resident q4_k super-block bytes the CPU q4kMatRowsRange reference reads, so the two
// must agree up to GPU float-accumulation order. We hold the GPU to the CPU *f32* path
// (q4kMatRowsRange, not the int8-SDOT decode kernel) because the GPU kernel also dequants to
// f32 and dots in float — same arithmetic, only the reduction order differs.
//
// This is the keystone for throughput parity: the CPU int8 path is compute-bound (~23 GB/s,
// ~1.4 tok/s decode ceiling on the M3 Pro) and cannot reach the llama.cpp-Metal bar (7.29
// decode / 51.55 prefill tok/s). A correct q4_k GPU GEMM is the only resident route that fits
// 27B on 36 GB (q4_k_m ≈ 16 GB) AND has the bandwidth + parallel dequant to hit the bar.

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strings"
	"testing"
	"unsafe"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

type q4kMLPOutputSlabTestReceipt struct {
	Schema         string `json:"schema"`
	Engine         string `json:"engine"`
	Backend        string `json:"backend"`
	Quant          string `json:"quant"`
	Calls          uint64 `json:"calls"`
	Allocations    uint64 `json:"allocations"`
	Reuses         uint64 `json:"reuses"`
	HighWaterBytes uint64 `json:"high_water_bytes"`
}

func parseQ4KMLPOutputSlabTestReceipt(t *testing.T, line string) q4kMLPOutputSlabTestReceipt {
	t.Helper()
	line = strings.TrimSpace(line)
	var receipt q4kMLPOutputSlabTestReceipt
	if err := json.Unmarshal([]byte(line), &receipt); err != nil {
		t.Fatalf("decode gate/up slab receipt %q: %v", line, err)
	}
	if receipt.Schema != q4kMLPOutputSlabReceiptSchema || receipt.Engine != "fak-native" || receipt.Backend != "metal" || receipt.Quant != "q4_k" {
		t.Fatalf("gate/up slab receipt identity = %+v", receipt)
	}
	return receipt
}

// randomQ4KTensor builds an [out,in] resident q4_k tensor from deterministic pseudo-random
// super-block bytes. Any byte pattern is a valid q4_k block (the dequant is total), so the CPU
// reference and the GPU kernel interpret identical bytes — the comparison is pure kernel math,
// not a quantizer round-trip.
func randomQ4KTensor(out, in int, seed int64) *q4kTensor {
	if in%qkK != 0 {
		panic("randomQ4KTensor: in not a multiple of 256")
	}
	nblk := in / qkK
	raw := make([]byte, out*nblk*q4kBlockBytes)
	rng := rand.New(rand.NewSource(seed))
	for i := range raw {
		raw[i] = byte(rng.Intn(256))
	}
	// Keep the f16 super-block scales (d, dmin) in a sane finite range so the dot doesn't
	// overflow to Inf/NaN: a uniformly random 16-bit pattern can be a huge/Inf half. Clamp the
	// exponent of the two halves at the head of every 144-B block to a small magnitude.
	for o := 0; o < out; o++ {
		for b := 0; b < nblk; b++ {
			base := (o*nblk + b) * q4kBlockBytes
			// d and dmin: small positive halves (~[1/64, 1/4)) — exponent bits set modestly.
			raw[base+1] = 0x2C | (raw[base+1] & 0x03) // high byte of half d
			raw[base+3] = 0x2C | (raw[base+3] & 0x03) // high byte of half dmin
		}
	}
	return &q4kTensor{out: out, in: in, nblk: nblk, raw: raw}
}

func randomVecF(n int, seed int64) []float32 {
	rng := rand.New(rand.NewSource(seed))
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	return x
}

func alignedBytes(n int) []byte {
	page := os.Getpagesize()
	backing := make([]byte, n+page)
	base := uintptr(unsafe.Pointer(&backing[0]))
	off := int((uintptr(page) - base%uintptr(page)) % uintptr(page))
	return backing[off : off+n]
}

// cosineAndMaxRel reports cosine similarity and the max relative error over the larger
// magnitudes (the small-magnitude entries are dominated by float noise and ignored).
func cosineAndMaxRel(a, b []float32) (cos float64, maxRel float64) {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 0, math.Inf(1)
	}
	cos = dot / (math.Sqrt(na) * math.Sqrt(nb))
	scale := math.Sqrt(na / float64(len(a))) // RMS magnitude
	for i := range a {
		if math.Abs(float64(a[i])) < scale {
			continue
		}
		rel := math.Abs(float64(a[i]-b[i])) / math.Abs(float64(a[i]))
		if rel > maxRel {
			maxRel = rel
		}
	}
	return cos, maxRel
}

func TestMetalTransientQ4KReleaseKeepsRegistryBounded(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("Metal unavailable")
	}
	metalgemm.ResetQ4K()
	defer metalgemm.ResetQ4K()
	raw := make([]byte, q4kBlockBytes)
	for i := 0; i < 8192+16; i++ {
		w := metalgemm.UploadQ4K(raw, 1, qkK)
		if w == nil {
			t.Fatalf("transient upload %d failed; released registry slots leaked", i)
		}
		w.Release()
	}
}

func TestMetalLazyQ4KGemvMatchesResidentAndCachesHandle(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("Metal unavailable")
	}
	metalgemm.ResetQ4K()
	defer metalgemm.ResetQ4K()
	const out, in = 5, 256
	raw := make([]byte, out*(in/qkK)*q4kBlockBytes)
	for i := range raw {
		raw[i] = byte(i*29 + 11)
	}
	resident := quantizeQ4KFromRaw(append([]byte(nil), raw...), out, in)
	lazy := &q4kTensor{out: out, in: in, nblk: in / qkK, lazy: &LazyQ4KRange{Reader: bytes.NewReader(raw), Bytes: len(raw)}}
	x := make([]float32, in)
	for i := range x {
		x[i] = float32((i%19)-9) / 13
	}
	want := q4kMatRows(resident, x)
	m := &Model{q4kw: map[string]*q4kTensor{"w": lazy}}
	defer func() {
		metalQ4KMu.Lock()
		delete(metalQ4KW, m)
		metalQ4KMu.Unlock()
	}()
	var got []float32
	var firstID int
	if !m.withMetalQ4K("w", lazy, func(w *metalgemm.Q4KWeight) {
		firstID = w.ID()
		got = make([]float32, out)
		w.GEMV(x, got)
	}) {
		t.Fatal("lazy Q4_K upload failed")
	}
	cos, maxRel := cosineAndMaxRel(want, got)
	if math.IsNaN(cos) || math.IsNaN(maxRel) || cos < 0.9999 || maxRel > 2e-2 {
		t.Fatalf("lazy Q4_K GEMV cosine=%g maxRel=%g, want cosine >= 0.9999 and maxRel <= 0.02", cos, maxRel)
	}
	metalQ4KMu.Lock()
	cached := metalQ4KW[m]["w"]
	metalQ4KMu.Unlock()
	if cached == nil || cached.ID() != firstID {
		t.Fatal("lazy operation did not retain its model-level Metal handle")
	}
	if !cached.NoCopy() {
		t.Fatal("lazy operation did not promote into no-copy unified-memory residency")
	}
	var secondID int
	if !m.withMetalQ4K("w", lazy, func(w *metalgemm.Q4KWeight) { secondID = w.ID() }) {
		t.Fatal("cached lazy Q4_K handle lookup failed")
	}
	if secondID != firstID {
		t.Fatalf("second lazy lookup handle = %d, want cached handle %d", secondID, firstID)
	}
	if len(lazy.raw) != 0 {
		t.Fatalf("lazy tensor retained a second host payload: %d bytes", len(lazy.raw))
	}
}

func TestIssue9073MetalLazyQ4KUsesMappedSpanWithoutReadAt(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("Metal unavailable")
	}
	metalgemm.ResetQ4K()
	defer metalgemm.ResetQ4K()
	const out, in, offset = 5, 256, 32
	resident := randomQ4KTensor(out, in, 9073)
	span := alignedBytes(os.Getpagesize())
	copy(span[offset:], resident.raw)
	reader := &issue9073ReaderAt{data: append([]byte("fallback"), resident.raw...)}
	lazy := &q4kTensor{out: out, in: in, nblk: 1, lazy: &LazyQ4KRange{
		Reader: reader, Offset: int64(len("fallback")), Bytes: len(resident.raw),
		MappedSpan: span, MappedOffset: offset,
	}}
	m := &Model{q4kw: map[string]*q4kTensor{"w": lazy}}
	w := m.metalQ4KWeight("w", lazy)
	if w == nil || !w.NoCopy() {
		t.Fatalf("mapped upload = %#v, want no-copy handle", w)
	}
	if reader.reads != 0 {
		t.Fatalf("mapped upload performed %d payload ReadAt calls, want zero", reader.reads)
	}
	x := randomVecF(in, 9073)
	want := q4kMatRows(resident, x)
	got := make([]float32, out)
	w.GEMV(x, got)
	if cos, maxRel := cosineAndMaxRel(want, got); cos < 0.9999 || maxRel > 0.02 {
		t.Fatalf("mapped GEMV cosine=%g maxRel=%g", cos, maxRel)
	}
	m.SetWeightCloser(closeFunc(func() error {
		if w.ID() >= 0 {
			t.Fatal("checkpoint unmap reached before borrowing Q4_K handle release")
		}
		return nil
	}))
	if err := m.CloseWeights(); err != nil {
		t.Fatal(err)
	}
}

func TestIssue9073MetalLazyQ4KReadAtFallbackIsByteIdentical(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("Metal unavailable")
	}
	metalgemm.ResetQ4K()
	defer metalgemm.ResetQ4K()
	const out, in = 5, 256
	resident := randomQ4KTensor(out, in, 9074)
	prefix := []byte("fallback")
	reader := &issue9073ReaderAt{data: append(append([]byte(nil), prefix...), resident.raw...)}
	lazy := &q4kTensor{out: out, in: in, nblk: 1, lazy: &LazyQ4KRange{
		Reader: reader, Offset: int64(len(prefix)), Bytes: len(resident.raw),
		MappedSpan: alignedBytes(os.Getpagesize()), MappedOffset: os.Getpagesize() - 32,
	}}
	m := &Model{q4kw: map[string]*q4kTensor{"w": lazy}}
	m.SetWeightCloser(closeFunc(func() error { return nil }))
	defer m.CloseWeights()
	w := m.metalQ4KWeight("w", lazy)
	if w == nil {
		t.Fatal("ReadAt fallback upload failed")
	}
	if reader.reads != 1 {
		t.Fatalf("fallback payload ReadAt calls=%d, want 1", reader.reads)
	}
	x := randomVecF(in, 9074)
	want := q4kMatRows(resident, x)
	got := make([]float32, out)
	w.GEMV(x, got)
	if cos, maxRel := cosineAndMaxRel(want, got); cos < 0.9999 || maxRel > 0.02 {
		t.Fatalf("fallback GEMV cosine=%g maxRel=%g", cos, maxRel)
	}
}

func TestMetalQ4KUploadUsesNoCopyUnifiedMemory(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("no Metal device available")
	}
	defer metalgemm.ResetQ4K()
	const out, in = 1024, 256
	raw := alignedBytes(out * (in / qkK) * q4kBlockBytes)
	if uintptr(unsafe.Pointer(&raw[0]))%uintptr(os.Getpagesize()) != 0 {
		t.Fatal("test fixture did not produce page-aligned source bytes")
	}
	// Row 0, block 0: d=1, dmin=0, scale[0]=1, min[0]=0, q[0]=0. With x[0]=1,
	// changing q[0]'s low nibble to 15 should move y[0] by roughly 15 if Metal aliases raw.
	raw[1] = 0x3c
	raw[4] = 1
	w := metalgemm.UploadQ4K(raw, out, in)
	if w == nil {
		t.Fatalf("UploadQ4K(%d,%d) returned nil", out, in)
	}
	if !w.NoCopy() {
		t.Fatal("UploadQ4K fell back to copied residency; #69 requires no-copy unified-memory weights")
	}
	x := make([]float32, in)
	x[0] = 1
	before := make([]float32, out)
	w.GEMV(x, before)
	raw[16] = 0x0f
	after := make([]float32, out)
	w.GEMV(x, after)
	if delta := after[0] - before[0]; delta < 14 || delta > 16 {
		t.Fatalf("Metal q4_k buffer did not observe resident raw-byte mutation: before=%g after=%g delta=%g", before[0], after[0], delta)
	}
}

func TestMetalQ4KGemvMatchesCPU(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("no Metal device available")
	}
	defer metalgemm.ResetQ4K()
	cases := []struct{ out, in int }{
		{256, 256}, {512, 1024}, {5120, 5120}, // hidden-size GEMV
	}
	for _, c := range cases {
		qt := randomQ4KTensor(c.out, c.in, 42)
		x := randomVecF(c.in, 7)
		ref := make([]float32, c.out)
		q4kMatRowsRange(qt, x, ref, 0, c.out) // CPU f32 reference

		w := metalgemm.UploadQ4K(qt.raw, c.out, c.in)
		if w == nil {
			t.Fatalf("UploadQ4K(%d,%d) returned nil", c.out, c.in)
		}
		got := make([]float32, c.out)
		w.GEMV(x, got)

		cos, maxRel := cosineAndMaxRel(ref, got)
		if cos < 0.9999 || maxRel > 5e-3 {
			t.Errorf("q4k GEMV [%d,%d]: cosine=%.6f maxRel=%.4g (want cos>=0.9999, maxRel<=5e-3)\n  ref[:4]=%v\n  got[:4]=%v",
				c.out, c.in, cos, maxRel, ref[:4], got[:4])
		} else {
			t.Logf("q4k GEMV [%d,%d]: cosine=%.6f maxRel=%.4g OK", c.out, c.in, cos, maxRel)
		}
		_ = w // ResetQ4K (deferred) frees every uploaded buffer
	}
}

// TestMetalQ4KGemvGroupMatchesSingle verifies that GEMVGroup (n weights sharing one activation,
// one command buffer) returns the same result as a single GEMV per weight — the correctness gate
// for the live decode group batching (q/k/v, gate/up). Weights have different out dims to exercise
// the per-weight y-offset packing.
func TestMetalQ4KGemvGroupMatchesSingle(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("no Metal device available")
	}
	defer metalgemm.ResetQ4K()
	in := 1024
	outs := []int{512, 1024, 256, 1536} // different out dims, shared in
	x := randomVecF(in, 9)
	ws := make([]*metalgemm.Q4KWeight, len(outs))
	singles := make([][]float32, len(outs))
	for i, out := range outs {
		qt := randomQ4KTensor(out, in, int64(100+i))
		w := metalgemm.UploadQ4K(qt.raw, out, in)
		if w == nil {
			t.Fatalf("UploadQ4K(%d,%d) returned nil", out, in)
		}
		ws[i] = w
		y := make([]float32, out)
		w.GEMV(x, y)
		singles[i] = y
	}
	group := metalgemm.GEMVGroup(ws, x)
	if len(group) != len(ws) {
		t.Fatalf("GEMVGroup returned %d results, want %d", len(group), len(ws))
	}
	for i := range ws {
		if len(group[i]) != outs[i] {
			t.Fatalf("group[%d] len=%d want %d", i, len(group[i]), outs[i])
		}
		for o := 0; o < outs[i]; o++ {
			if d := group[i][o] - singles[i][o]; d > 1e-3 || d < -1e-3 {
				t.Fatalf("group[%d][%d]=%g != single %g", i, o, group[i][o], singles[i][o])
			}
		}
	}
	t.Logf("GEMVGroup matches single GEMV across %d weights (outs=%v)", len(ws), outs)
}

// TestMetalQ4KPrefillMatchesCPU is the end-to-end wiring gate: the resident-Q4_K hybrid
// prefill with MetalQ4K=true (q4_k-majority GEMMs on the GPU) produces the same logits as the
// CPU path (MetalQ4K=false) on the synthetic hybrid model. CPU GEMV is forced to f32
// (setQ4KSDOTForTest(false)) so the comparison is GPU-f32 vs CPU-f32 for the q4_k majority; the
// Q8 minority (q/k + linear_attn.*) may also route through Metal Q8 GEMM. A wiring bug (wrong
// weight, layout mismatch, the GPU result not flowing into the recurrence) diverges O(1) per layer
// and blows past the bound.
func TestMetalQ4KPrefillMatchesCPU(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("no Metal device available")
	}
	defer metalgemm.ResetQ4K()
	setQ4KSDOTForTest(false)
	t.Cleanup(func() { setQ4KSDOTForTest(true) })

	cfg := qwen35HybridQ4KTestCfg()
	m := NewSynthetic(cfg)
	m.Quantize()
	fillQ4KMajority(t, m, cfg)
	prompt := []int{3, 7, 11, 5, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61}

	cpu := m.NewSession()
	cpu.Q4K = true
	cpuLogits := cpu.Prefill(prompt)

	gpu := m.NewSession()
	gpu.Q4K = true
	gpu.MetalQ4K = true
	gpuLogits := gpu.Prefill(prompt)

	if len(cpuLogits) != len(gpuLogits) {
		t.Fatalf("logit length mismatch: cpu=%d gpu=%d", len(cpuLogits), len(gpuLogits))
	}
	cos, maxRel := cosineAndMaxRel(cpuLogits, gpuLogits)
	if argmaxF(cpuLogits) != argmaxF(gpuLogits) || cos < 0.999 {
		t.Errorf("metal q4k prefill: cpu argmax=%d gpu argmax=%d cosine=%.6f maxRel=%.4g (want same argmax, cos>=0.999)",
			argmaxF(cpuLogits), argmaxF(gpuLogits), cos, maxRel)
	} else {
		t.Logf("metal q4k prefill: argmax match=%d cosine=%.6f maxRel=%.4g OK", argmaxF(gpuLogits), cos, maxRel)
	}
}

// TestMetalQ4KDecodeMatchesCPU verifies the single-residency GPU decode path: with MetalQ4K the
// decode q4_k GEMVs run on the GPU (q4k_gemv) and the CPU q4_k copy is freed after upload. It
// must produce the same greedy decode tokens as the CPU path. Two separate models are built
// because the GPU run frees its model's raw q4_k bytes (single residency), which would break a
// subsequent CPU run on the same model.
func TestMetalQ4KDecodeMatchesCPU(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("no Metal device available")
	}
	defer metalgemm.ResetQ4K()
	setQ4KSDOTForTest(false)
	t.Cleanup(func() { setQ4KSDOTForTest(true) })
	cfg := qwen35HybridQ4KTestCfg()
	prompt := []int{3, 7, 11, 5, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61}

	decode := func(metal bool) ([]float32, []int) {
		m := NewSynthetic(cfg)
		m.Quantize()
		fillQ4KMajority(t, m, cfg)
		s := m.NewSession()
		s.Q4K = true
		s.MetalQ4K = metal
		lg := s.Prefill(prompt)
		var seq []int
		for i := 0; i < 4; i++ {
			n := argmaxF(lg)
			seq = append(seq, n)
			lg = s.Step(n)
		}
		return lg, seq
	}

	_, cpuSeq := decode(false)
	gpuLast, gpuSeq := decode(true)
	_ = gpuLast
	for i := range cpuSeq {
		if cpuSeq[i] != gpuSeq[i] {
			t.Fatalf("decode token %d: cpu=%d gpu=%d (cpu seq=%v gpu seq=%v)", i, cpuSeq[i], gpuSeq[i], cpuSeq, gpuSeq)
		}
	}
	t.Logf("metal q4k decode: greedy token sequence matches CPU = %v", gpuSeq)
}

func argmaxF(v []float32) int {
	bi := 0
	best := float32(-3.4e38)
	for i, x := range v {
		if x > best {
			best, bi = x, i
		}
	}
	return bi
}

func TestMetalQ4KGemmMatchesCPU(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("no Metal device available")
	}
	defer metalgemm.ResetQ4K()
	// Cover: small, the 22-token oracle panel, a non-square MLP-like shape, the real Qwen3.6-27B
	// gate/up prefill shape ([17408,5120] — the dominant prefill GEMM, issue #1085), and P>256
	// (the tiled kernel's token-tile size) so the multi-tile / single-command-buffer path and the
	// new SIMD-group-reduction dot are both exercised.
	cases := []struct{ out, in, P int }{
		{1024, 1024, 6},
		{2048, 1024, 22},
		{17408, 5120, 22}, // real Qwen3.6-27B gate/up panel, small-P occupancy path
		{1024, 512, 300},  // two token tiles (256 + 44)
	}
	for _, c := range cases {
		qt := randomQ4KTensor(c.out, c.in, 99)
		X := randomVecF(c.P*c.in, 11)
		ref := make([]float32, c.P*c.out)
		for tIdx := 0; tIdx < c.P; tIdx++ {
			row := make([]float32, c.out)
			q4kMatRowsRange(qt, X[tIdx*c.in:(tIdx+1)*c.in], row, 0, c.out)
			copy(ref[tIdx*c.out:(tIdx+1)*c.out], row)
		}
		w := metalgemm.UploadQ4K(qt.raw, c.out, c.in)
		if w == nil {
			t.Fatalf("UploadQ4K(%d,%d) returned nil", c.out, c.in)
		}
		got := make([]float32, c.P*c.out)
		w.GEMM(X, c.P, got)
		cos, maxRel := cosineAndMaxRel(ref, got)
		if cos < 0.9999 || maxRel > 5e-3 {
			t.Errorf("q4k GEMM [%d,%d]x%d: cosine=%.6f maxRel=%.4g (want cos>=0.9999, maxRel<=5e-3)", c.out, c.in, c.P, cos, maxRel)
		} else {
			t.Logf("q4k GEMM [%d,%d]x%d: cosine=%.6f maxRel=%.4g OK", c.out, c.in, c.P, cos, maxRel)
		}
		metalgemm.ResetQ4K()
	}
}

// TestMetalGEMMGroupMatchesCPU is the correctness gate for the batched prefill GEMM group
// (metalgemm.GEMMGroup / mg_q4k_gemm_group): n weights of different out dims sharing one activation
// panel X[P,in], all run in ONE command buffer. It must be (a) bit-identical to a single
// Q4KWeight.GEMM per weight (same q4k_gemm kernel, just grouped — this catches a wrong per-weight
// Y-offset or `out`/P binding) and (b) match the CPU f32 q4kMatRowsRange reference to cosine
// >= 0.9999 (same as the single-GEMM gate). The differing out dims + P>256 exercise the per-weight
// [P,out_i] token-major packing at yoff[i]=P*Σout_j and the multi-token-tile loop.
func TestMetalGEMMGroupMatchesCPU(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("no Metal device available")
	}
	defer metalgemm.ResetQ4K()
	in := 5120 // shared activation width (Qwen3.6-27B hidden size)
	// Real prefill group shapes sharing in=H: q(+gate fused)=2*qWidth, k=v=nKV*hd, plus a couple of
	// extra out dims to stress the packing. All share in.
	outs := []int{17408, 1024, 1024, 512}
	for _, P := range []int{16, 53, 300} { // small-P, the measured P=53 turn, and >256 (two token tiles)
		X := randomVecF(P*in, 11)
		ws := make([]*metalgemm.Q4KWeight, len(outs))
		singles := make([][]float32, len(outs))
		refs := make([][]float32, len(outs))
		for i, out := range outs {
			qt := randomQ4KTensor(out, in, int64(200+i))
			w := metalgemm.UploadQ4K(qt.raw, out, in)
			if w == nil {
				t.Fatalf("UploadQ4K(%d,%d) returned nil", out, in)
			}
			ws[i] = w
			// per-weight single GEMM (bit-identical target) + CPU f32 reference
			single := make([]float32, P*out)
			w.GEMM(X, P, single)
			singles[i] = single
			ref := make([]float32, P*out)
			for tIdx := 0; tIdx < P; tIdx++ {
				row := make([]float32, out)
				q4kMatRowsRange(qt, X[tIdx*in:(tIdx+1)*in], row, 0, out)
				copy(ref[tIdx*out:(tIdx+1)*out], row)
			}
			refs[i] = ref
		}
		group := metalgemm.GEMMGroup(ws, X, P)
		if len(group) != len(ws) {
			t.Fatalf("P=%d: GEMMGroup returned %d results, want %d", P, len(group), len(ws))
		}
		for i, out := range outs {
			if len(group[i]) != P*out {
				t.Fatalf("P=%d group[%d] len=%d want %d", P, i, len(group[i]), P*out)
			}
			// (a) bit-identical to the single per-weight GEMM
			for k := 0; k < P*out; k++ {
				if group[i][k] != singles[i][k] {
					t.Fatalf("P=%d group[%d][%d]=%g != single GEMM %g (Y-offset/binding wrong)",
						P, i, k, group[i][k], singles[i][k])
				}
			}
			// (b) matches the CPU f32 reference to the same bound as the single GEMM gate
			cos, maxRel := cosineAndMaxRel(refs[i], group[i])
			if cos < 0.9999 || maxRel > 5e-3 {
				t.Errorf("P=%d GEMMGroup[%d] [%d,%d]: cosine=%.6f maxRel=%.4g (want cos>=0.9999, maxRel<=5e-3)",
					P, i, out, in, cos, maxRel)
			}
		}
		metalgemm.ResetQ4K()
		t.Logf("GEMMGroup P=%d matches single GEMM (bit-identical) + CPU ref across outs=%v", P, outs)
	}
}

// TestMetalQ4KGemmMM32SelectionAndParity pins the exact dispatch contract for the hardware-MMA
// candidate: default-off P32 is scalar, FAK_Q4K_MM's SetGEMMUseMM opt-in selects MM32 only at exact
// P32, and adjacent P31/P33 remain scalar. Both the single and grouped call paths report the same
// typed requested/executed identity and stay within the CPU f32 oracle.
func TestMetalQ4KGemmMM32SelectionAndParity(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("no Metal device available")
	}
	defer metalgemm.ResetQ4K()
	t.Cleanup(func() { metalgemm.SetGEMMUseMM(false) })
	const (
		out = 256
		in  = 512
	)
	qt := randomQ4KTensor(out, in, 9202)
	w := metalgemm.UploadQ4K(qt.raw, out, in)
	if w == nil {
		t.Fatal("UploadQ4K returned nil")
	}
	qt2 := randomQ4KTensor(96, in, 9203)
	w2 := metalgemm.UploadQ4K(qt2.raw, 96, in)
	if w2 == nil {
		t.Fatal("second UploadQ4K returned nil")
	}
	ws := []*metalgemm.Q4KWeight{w, w2}
	scalar := metalgemm.Q4KGEMMIdentity{
		Requested: metalgemm.Q4KGEMMExecutedScalar,
		Executed:  metalgemm.Q4KGEMMExecutedScalar,
	}
	mm32 := metalgemm.Q4KGEMMIdentity{
		Requested: metalgemm.Q4KGEMMExecutedMM32,
		Executed:  metalgemm.Q4KGEMMExecutedMM32,
	}

	runSingle := func(P int, useMM bool, want metalgemm.Q4KGEMMIdentity) ([]float32, []float32) {
		t.Helper()
		metalgemm.SetGEMMUseMM(useMM)
		X := randomVecF(P*in, int64(9202+P))
		got := make([]float32, P*out)
		observation := metalgemm.NewExecutionObservation(metalgemm.ExecutionQ4KGEMM)
		identity := w.GEMMWithEvents(X, P, got, observation)
		if identity != want {
			t.Fatalf("single P=%d useMM=%t identity=%+v want %+v", P, useMM, identity, want)
		}
		snapshot, err := observation.Snapshot()
		if err != nil {
			t.Fatal(err)
		}
		if len(snapshot.Events) != 1 {
			t.Fatalf("P=%d useMM=%t events=%+v, want one dispatch", P, useMM, snapshot.Events)
		}
		event := snapshot.Events[0]
		if !event.Committed || !event.CompletedWait || !event.HostReadback || event.Encoders != 1 {
			t.Fatalf("P=%d useMM=%t lifecycle=%+v", P, useMM, event)
		}
		ref := make([]float32, P*out)
		row := make([]float32, out)
		for token := 0; token < P; token++ {
			q4kMatRowsRange(qt, X[token*in:(token+1)*in], row, 0, out)
			copy(ref[token*out:(token+1)*out], row)
		}
		cos, maxRel := cosineAndMaxRel(ref, got)
		if cos < 0.9999 || maxRel > 5e-3 {
			t.Fatalf("single P=%d useMM=%t identity=%+v cosine=%.6f maxRel=%.4g", P, useMM, want, cos, maxRel)
		}
		return X, got
	}

	runGroup := func(P int, useMM bool, want metalgemm.Q4KGEMMIdentity) {
		t.Helper()
		metalgemm.SetGEMMUseMM(useMM)
		X := randomVecF(P*in, int64(9300+P))
		observation := metalgemm.NewExecutionObservation(metalgemm.ExecutionQ4KGEMMGroup)
		group, identity := metalgemm.GEMMGroupWithEventsIdentity(ws, X, P, observation)
		if identity != want {
			t.Fatalf("group P=%d useMM=%t identity=%+v want %+v", P, useMM, identity, want)
		}
		if len(group) != len(ws) {
			t.Fatalf("group P=%d useMM=%t outputs=%d want %d", P, useMM, len(group), len(ws))
		}
		snapshot, err := observation.Snapshot()
		if err != nil {
			t.Fatal(err)
		}
		if len(snapshot.Events) != 1 {
			t.Fatalf("group P=%d useMM=%t events=%+v want one dispatch", P, useMM, snapshot.Events)
		}
		event := snapshot.Events[0]
		if !event.Committed || !event.CompletedWait || !event.HostReadback || event.Encoders != 1 {
			t.Fatalf("group P=%d useMM=%t lifecycle=%+v", P, useMM, event)
		}
		for i, wi := range ws {
			single := make([]float32, P*wi.Out)
			if identity := wi.GEMM(X, P, single); identity != want {
				t.Fatalf("group P=%d useMM=%t single[%d] identity=%+v want %+v", P, useMM, i, identity, want)
			}
			for j := range single {
				if math.Float32bits(group[i][j]) != math.Float32bits(single[j]) {
					t.Fatalf("group P=%d useMM=%t output[%d][%d]=%g want single %g",
						P, useMM, i, j, group[i][j], single[j])
				}
			}
		}
	}

	runSingle(32, false, scalar)
	runSingle(31, true, scalar)
	X32, single32 := runSingle(32, true, mm32)
	runSingle(33, true, scalar)

	runGroup(32, false, scalar)
	runGroup(31, true, scalar)
	runGroup(32, true, mm32)
	runGroup(33, true, scalar)

	groupObservation := metalgemm.NewExecutionObservation(metalgemm.ExecutionQ4KGEMMGroup)
	group, identity := metalgemm.GEMMGroupWithEventsIdentity(ws, X32, 32, groupObservation)
	if identity != mm32 {
		t.Fatalf("group replay P32 identity=%+v want %+v", identity, mm32)
	}
	for i := range single32 {
		if math.Float32bits(group[0][i]) != math.Float32bits(single32[i]) {
			t.Fatalf("group replay MM32 output[0][%d]=%g want single %g", i, group[0][i], single32[i])
		}
	}
	t.Log("typed identities: single/group default-P32=scalar opt-in-P31=scalar opt-in-P32=mm32 opt-in-P33=scalar")
}

// TestMetalQ4KGemmMM32UnavailableDeclinesBeforeMutation exercises the native fail-closed seam
// deterministically. A requested unavailable P32 candidate returns NotExecuted for single and
// grouped selection with no command buffer and byte-for-byte untouched caller outputs.
func TestMetalQ4KGemmMM32UnavailableDeclinesBeforeMutation(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("no Metal device available")
	}
	defer metalgemm.ResetQ4K()
	const (
		P    = 32
		in   = 256
		out  = 64
		mark = float32(9202)
	)
	w := metalgemm.UploadQ4K(randomQ4KTensor(out, in, 9202).raw, out, in)
	if w == nil {
		t.Fatal("UploadQ4K returned nil")
	}
	X := randomVecF(P*in, 9202)
	single := make([]float32, P*out)
	for i := range single {
		single[i] = mark
	}
	singleObservation := metalgemm.NewExecutionObservation(metalgemm.ExecutionQ4KGEMM)
	mode := metalgemm.Q4KGEMMMode(metalgemm.Q4KGEMMModeMM32Unavailable)
	wantIdentity := metalgemm.Q4KGEMMIdentityForMode(P, mode, metalgemm.Q4KGEMMNotExecuted)
	if identity := w.GEMMWithEventsMode(X, P, single, singleObservation, mode); identity != wantIdentity {
		t.Fatalf("unavailable single identity=%+v want %+v", identity, wantIdentity)
	}
	singleSnapshot, err := singleObservation.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	if len(singleSnapshot.Events) != 0 {
		t.Fatalf("unavailable single events=%+v want none", singleSnapshot.Events)
	}
	for i, v := range single {
		if v != mark {
			t.Fatalf("unavailable single mutated output[%d]=%g want %g", i, v, mark)
		}
	}

	groupBacking := make([]float32, P*out)
	for i := range groupBacking {
		groupBacking[i] = mark
	}
	groupObservation := metalgemm.NewExecutionObservation(metalgemm.ExecutionQ4KGEMMGroup)
	group, identity := metalgemm.GEMMGroupIntoWithEventsMode(
		[]*metalgemm.Q4KWeight{w}, X, P, groupBacking, groupObservation, mode)
	if identity != wantIdentity || group != nil {
		t.Fatalf("unavailable group outputs=%v identity=%+v want nil/%+v", group, identity, wantIdentity)
	}
	groupSnapshot, err := groupObservation.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	if len(groupSnapshot.Events) != 0 {
		t.Fatalf("unavailable group events=%+v want none", groupSnapshot.Events)
	}
	for i, v := range groupBacking {
		if v != mark {
			t.Fatalf("unavailable group mutated output[%d]=%g want %g", i, v, mark)
		}
	}
	t.Log("unavailable MM32: single/group status=not-executed events=0 output=untouched")
}

// BenchmarkMetalQ4KGemmMM32vsScalar is the exact-P32 A/B for q4k_gemm_mm32 against q4k_gemm at
// the real Qwen gate/up matrix shape. It remains opt-in evidence; this benchmark does not flip the
// default.
func BenchmarkMetalQ4KGemmMM32vsScalar(b *testing.B) {
	if !metalgemm.Available() {
		b.Skip("no Metal device available")
	}
	defer metalgemm.ResetQ4K()
	out, in, P := 17408, 5120, 32
	qt := randomQ4KTensor(out, in, 1)
	X := randomVecF(P*in, 2)
	w := metalgemm.UploadQ4K(qt.raw, out, in)
	if w == nil {
		b.Fatal("UploadQ4K returned nil")
	}
	Y := make([]float32, P*out)
	flops := 2 * float64(out) * float64(in) * float64(P)
	run := func(name string, useMM bool) {
		b.Run(name, func(b *testing.B) {
			metalgemm.SetGEMMUseMM(useMM)
			defer metalgemm.SetGEMMUseMM(false)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				w.GEMM(X, P, Y)
			}
			b.StopTimer()
			if s := b.Elapsed().Seconds(); s > 0 {
				b.ReportMetric(flops*float64(b.N)/s/1e9, "GFLOP/s")
				b.ReportMetric(s/float64(b.N)*1e3, "ms/op")
			}
		})
	}
	run("scalar", false)
	run("mma", true)
}

// BenchmarkMetalQ4KGemvGroupVsSingle is the prefill-wall benchmark: the q/k/v group as ONE command
// buffer (GEMMGroup) vs three separate Q4KWeight.GEMM calls. If the group collapses the wall-clock
// materially, the per-op command-buffer submit/sync was the prefill bottleneck (as the P=53 profile
// showed: ~97% GEMM+roundtrip). Shapes are a Qwen3.6-27B full-attn q/k/v panel sharing H=5120.
func BenchmarkMetalGEMMGroupVsSingle(b *testing.B) {
	if !metalgemm.Available() {
		b.Skip("no Metal device available")
	}
	defer metalgemm.ResetQ4K()
	in, P := 5120, 53
	outs := []int{10240, 1024, 1024} // q(+gate) ~2*qWidth, k, v
	ws := make([]*metalgemm.Q4KWeight, len(outs))
	for i, out := range outs {
		ws[i] = metalgemm.UploadQ4K(randomQ4KTensor(out, in, int64(300+i)).raw, out, in)
		if ws[i] == nil {
			b.Fatal("UploadQ4K returned nil")
		}
	}
	X := randomVecF(P*in, 7)
	b.Run("group_one_cmdbuf", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = metalgemm.GEMMGroup(ws, X, P)
		}
	})
	b.Run("separate_gemms", func(b *testing.B) {
		ys := make([][]float32, len(outs))
		for i, out := range outs {
			ys[i] = make([]float32, P*out)
		}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			for j, w := range ws {
				w.GEMM(X, P, ys[j])
			}
		}
	})
}

// BenchmarkMetalQ4KGemv reports the GPU q4_k GEMV throughput at hidden size. Compare against
// the CPU BenchmarkQ4KMatRowsInt8 (~23 GB/s at 12 workers): the GPU should clear it and head
// toward the unified-memory bandwidth that the 7.29 tok/s decode bar implies.
func BenchmarkMetalQ4KGemv(b *testing.B) {
	if !metalgemm.Available() {
		b.Skip("no Metal device available")
	}
	defer metalgemm.ResetQ4K()
	out, in := 5120, 5120
	qt := randomQ4KTensor(out, in, 1)
	x := randomVecF(in, 2)
	w := metalgemm.UploadQ4K(qt.raw, out, in)
	if w == nil {
		b.Fatal("UploadQ4K returned nil")
	}
	y := make([]float32, out)
	weightBytes := float64(out) * float64(in) / 256.0 * 144.0
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		w.GEMV(x, y)
	}
	b.StopTimer()
	secs := b.Elapsed().Seconds()
	if secs > 0 {
		b.ReportMetric(weightBytes*float64(b.N)/secs/1e9, "GB/s")
	}
}

// BenchmarkMetalQ4KGemvTiny isolates the per-dispatch (command-buffer commit→wait) overhead:
// a 256x256 GEMV does ~16 KB of work, so its ns/op is dominated by the fixed launch cost. The
// gap between this and BenchmarkMetalQ4KGemv (5120x5120) attributes time to overhead vs work,
// which decides the decode-wiring strategy (how many dispatches/token are affordable).
func BenchmarkMetalQ4KGemvTiny(b *testing.B) {
	if !metalgemm.Available() {
		b.Skip("no Metal device available")
	}
	defer metalgemm.ResetQ4K()
	out, in := 256, 256
	qt := randomQ4KTensor(out, in, 1)
	x := randomVecF(in, 2)
	w := metalgemm.UploadQ4K(qt.raw, out, in)
	if w == nil {
		b.Fatal("UploadQ4K returned nil")
	}
	y := make([]float32, out)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		w.GEMV(x, y)
	}
}

// BenchmarkMetalQ4KGemvBatch runs n decode GEMVs of one 5120x5120 weight in a SINGLE command
// buffer (one commit→wait), to quantify how much of the per-GEMV cost is the CPU↔GPU
// submission/sync round-trip vs the kernel. If per-GEMV here collapses far below the single-GEMV
// BenchmarkMetalQ4KGemv (~457 µs) toward the kernel rate, the decode wall is the per-op command
// buffer and the fix is the one-command-buffer resident forward (issue #67). n=64 ≈ the
// projection/MLP GEMV count in one decoder layer scaled up.
func BenchmarkMetalQ4KGemvBatch(b *testing.B) {
	if !metalgemm.Available() {
		b.Skip("no Metal device available")
	}
	defer metalgemm.ResetQ4K()
	out, in, n := 5120, 5120, 64
	qt := randomQ4KTensor(out, in, 1)
	Xcat := randomVecF(n*in, 2)
	w := metalgemm.UploadQ4K(qt.raw, out, in)
	if w == nil {
		b.Fatal("UploadQ4K returned nil")
	}
	Ycat := make([]float32, n*out)
	// Trust check: batched row 0 must equal a single GEMV of the same activation row.
	w.GEMVBatch(Xcat, n, Ycat)
	single := make([]float32, out)
	w.GEMV(Xcat[:in], single)
	for o := 0; o < out; o++ {
		if d := Ycat[o] - single[o]; d > 1e-3 || d < -1e-3 {
			b.Fatalf("GEMVBatch row0[%d]=%g != GEMV %g (offset binding wrong)", o, Ycat[o], single[o])
		}
	}
	weightBytes := float64(out) * float64(in) / 256.0 * 144.0
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		w.GEMVBatch(Xcat, n, Ycat)
	}
	b.StopTimer()
	secs := b.Elapsed().Seconds()
	if secs > 0 {
		b.ReportMetric(weightBytes*float64(n)*float64(b.N)/secs/1e9, "GB/s")
		b.ReportMetric(secs/float64(b.N)/float64(n)*1e6, "us/gemv")
	}
}

// BenchmarkMetalQ4KFusedMLP isolates the fused on-GPU MLP (gate→silu·up→down in ONE command
// buffer, intermediate resident) against the same three matmuls run as separate command buffers
// with a CPU SwiGLU between — the decode MLP is ~54% of q4_k_m decode, so this is the noise-free
// measure of the per-MLP-call lever the end-to-end wall-clock is too contended to show cleanly.
// Shapes are Qwen3.6-27B's: H=5120, I=17408.
func BenchmarkMetalQ4KFusedMLP(b *testing.B) {
	if !metalgemm.Available() {
		b.Skip("no Metal device available")
	}
	defer metalgemm.ResetQ4K()
	H, I := 5120, 17408
	gate := metalgemm.UploadQ4K(randomQ4KTensor(I, H, 1).raw, I, H)
	up := metalgemm.UploadQ4K(randomQ4KTensor(I, H, 2).raw, I, H)
	down := metalgemm.UploadQ4K(randomQ4KTensor(H, I, 3).raw, H, I)
	if gate == nil || up == nil || down == nil {
		b.Fatal("UploadQ4K returned nil")
	}
	x := randomVecF(H, 4)
	y := make([]float32, H)
	rowBytes := func(out, in int) float64 { return float64(out) * float64(in) / 256.0 * 144.0 }
	mlpBytes := rowBytes(I, H) + rowBytes(I, H) + rowBytes(H, I) // gate + up + down weight bytes
	// Trust check: fused output equals the separate path on decisive margin.
	g0, u0, inter0 := make([]float32, I), make([]float32, I), make([]float32, I)
	gate.GEMV(x, g0)
	up.GEMV(x, u0)
	for j := 0; j < I; j++ {
		inter0[j] = silu(g0[j]) * u0[j]
	}
	ySep := make([]float32, H)
	down.GEMV(inter0, ySep)
	metalgemm.FusedMLP(gate, up, down, x, y)
	for o := 0; o < H; o++ {
		if d := y[o] - ySep[o]; d > 1e-2 || d < -1e-2 {
			b.Fatalf("FusedMLP[%d]=%g != separate %g", o, y[o], ySep[o])
		}
	}
	b.Run("fused", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			metalgemm.FusedMLP(gate, up, down, x, y)
		}
		b.StopTimer()
		if s := b.Elapsed().Seconds(); s > 0 {
			b.ReportMetric(mlpBytes*float64(b.N)/s/1e9, "GB/s")
			b.ReportMetric(s/float64(b.N)*1e3, "ms/mlp")
		}
	})
	b.Run("separate", func(b *testing.B) {
		g, u, inter := make([]float32, I), make([]float32, I), make([]float32, I)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			gate.GEMV(x, g)
			up.GEMV(x, u)
			for j := 0; j < I; j++ {
				inter[j] = silu(g[j]) * u[j]
			}
			down.GEMV(inter, y)
		}
		b.StopTimer()
		if s := b.Elapsed().Seconds(); s > 0 {
			b.ReportMetric(mlpBytes*float64(b.N)/s/1e9, "GB/s")
			b.ReportMetric(s/float64(b.N)*1e3, "ms/mlp")
		}
	})
}

// BenchmarkMetalQ4KGemmSteady measures the kernel's raw dequant+dot throughput when the
// per-command-buffer launch overhead is amortized over a large batch (P tokens in one
// dispatch). The single-GEMV bench above is overhead-bound (one commit→wait per ~0.5 ms of
// work); this one shows what the q4_k MSL kernel actually sustains — the number that says
// whether the GPU route can clear the CPU int8 ceiling once the forward is batched into one
// command buffer (the forward.m pattern), which is the wiring step to the decode/prefill bar.
func BenchmarkMetalQ4KGemmSteady(b *testing.B) {
	if !metalgemm.Available() {
		b.Skip("no Metal device available")
	}
	defer metalgemm.ResetQ4K()
	// Realistic Qwen3.6 MLP prefill shape: gate/up are [17408,5120], the 22-tok oracle panel.
	out, in, P := 17408, 5120, 22
	qt := randomQ4KTensor(out, in, 1)
	X := randomVecF(P*in, 2)
	w := metalgemm.UploadQ4K(qt.raw, out, in)
	if w == nil {
		b.Fatal("UploadQ4K returned nil")
	}
	Y := make([]float32, P*out)
	// The tiled kernel reads each weight ONCE per token-tile, so model-bytes ≈ the weight size
	// (×ceil(P/256) tiles, the Q4K_BN token-tile width). Report effective model-GB/s = weight
	// bytes moved through the GEMM.
	tiles := float64((P + 255) / 256)
	weightBytes := tiles * float64(out) * float64(in) / 256.0 * 144.0
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		w.GEMM(X, P, Y)
	}
	b.StopTimer()
	secs := b.Elapsed().Seconds()
	if secs > 0 {
		b.ReportMetric(weightBytes*float64(b.N)/secs/1e9, "GB/s")
		// FLOP rate: 2*out*in*P MACs per GEMM — the compute-bound view once weights are read once.
		b.ReportMetric(2*float64(out)*float64(in)*float64(P)*float64(b.N)/secs/1e9, "GFLOP/s")
	}
}

// BenchmarkMetalQ4KGemmPSweep sweeps the prompt length P at the real Qwen3.6-27B gate/up shape
// [17408,5120]. The diagnosis measured a 29-token prefill (small P), but real agentic prefills are
// hundreds-to-thousands of tokens, where the weight (read once per token-tile) is reused across many
// more tokens — so GFLOP/s should climb steeply with P if the kernel is weight-bound and saturate
// if it is compute/occupancy-bound. The curve says which regime the real prefill lives in.
func BenchmarkMetalQ4KGemmPSweep(b *testing.B) {
	if !metalgemm.Available() {
		b.Skip("no Metal device available")
	}
	defer metalgemm.ResetQ4K()
	out, in := 17408, 5120
	qt := randomQ4KTensor(out, in, 1)
	w := metalgemm.UploadQ4K(qt.raw, out, in)
	if w == nil {
		b.Fatal("UploadQ4K returned nil")
	}
	for _, P := range []int{22, 64, 128, 256, 512, 1024, 2048} {
		X := randomVecF(P*in, 2)
		Y := make([]float32, P*out)
		b.Run(fmt.Sprintf("P=%d", P), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				w.GEMM(X, P, Y)
			}
			b.StopTimer()
			if s := b.Elapsed().Seconds(); s > 0 {
				b.ReportMetric(2*float64(out)*float64(in)*float64(P)*float64(b.N)/s/1e9, "GFLOP/s")
				b.ReportMetric(s/float64(b.N)*1e3, "ms/op")
			}
		})
	}
}

// TestMetalQ4KSingleRowGemmDispatchMatchesGEMM pins the P=1 dispatch seam used by cached
// Qwen3.5/Qwen3.8 decode. The optimized path must preserve both single-weight values and grouped
// result ordering relative to the prior one-row GEMM implementation.
func TestMetalQ4KSingleRowGemmDispatchMatchesGEMM(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("no Metal device available")
	}
	defer metalgemm.ResetQ4K()

	const in = 1024
	outs := []int{512, 1024, 256}
	names := []string{"single-row-a", "single-row-b", "single-row-c"}
	x := randomVecF(in, 41)
	m := &Model{q4kw: make(map[string]*q4kTensor)}
	s := m.NewSession()
	s.MetalQ4K = true
	want := make([][]float32, len(names))
	for i, name := range names {
		qt := randomQ4KTensor(outs[i], in, int64(200+i))
		m.q4kw[name] = qt
		w := m.metalQ4KWeight(name, qt)
		if w == nil {
			t.Fatalf("metalQ4KWeight(%q) returned nil", name)
		}
		want[i] = make([]float32, outs[i])
		w.GEMM(x, 1, want[i])
	}

	gotSingle := s.q4kGemmDispatch(names[0], m.q4kw[names[0]], x, 1)
	assertClose := func(label string, got, expected []float32) {
		t.Helper()
		if len(got) != len(expected) {
			t.Fatalf("%s len=%d want %d", label, len(got), len(expected))
		}
		for i := range got {
			if d := got[i] - expected[i]; d > 5e-3 || d < -5e-3 {
				t.Fatalf("%s[%d]=%g want %g", label, i, got[i], expected[i])
			}
		}
	}
	assertClose("single", gotSingle, want[0])

	gotGroup := s.q4kGemmGroupDispatch(names, x, 1)
	if len(gotGroup) != len(names) {
		t.Fatalf("group len=%d want %d", len(gotGroup), len(names))
	}
	for i := range names {
		assertClose(names[i], gotGroup[i], want[i])
	}
}

func TestMetalQ4KMLPOutputSlabReuseIsolationLifecycleAndFallback(t *testing.T) {
	t.Setenv("FAK_Q4K_GATEUP_SLAB", "1")
	if !metalgemm.Available() {
		t.Skip("no Metal device available")
	}
	defer metalgemm.ResetQ4K()

	const (
		in = 256
		P  = 3
	)
	names := []string{
		"blk.0.mlp.gate_proj.weight",
		"blk.0.mlp.up_proj.weight",
	}
	m := &Model{q4kw: make(map[string]*q4kTensor)}
	for i, name := range names {
		m.q4kw[name] = randomQ4KTensor(64, in, int64(9102+i))
	}
	X := randomVecF(P*in, 9102)

	controlSession := m.NewSession()
	controlSession.MetalQ4K = true
	control := controlSession.q4kGemmGroupDispatch(names, X, P)
	if control == nil {
		t.Fatal("allocating control group returned nil")
	}
	if controlSession.q4kMLPOutputSlab != nil {
		t.Fatal("default-off control retained a gate/up slab")
	}
	controlCopy := make([][]float32, len(control))
	for i := range control {
		controlCopy[i] = append([]float32(nil), control[i]...)
	}

	candidateSession := m.NewSession()
	candidateSession.MetalQ4K = true
	candidateSession.Q4KGateUpOutputSlab = true
	var stable *float32
	for call := 0; call < 65; call++ { // 64 compute-layer equivalents, then the next chunk's first layer.
		got := candidateSession.q4kGemmGroupDispatch(names, X, P)
		if got == nil {
			t.Fatalf("candidate call %d returned nil", call)
		}
		if call == 0 {
			stable = &got[0][0]
		} else if &got[0][0] != stable {
			t.Fatalf("candidate call %d backing=%p want stable %p", call, &got[0][0], stable)
		}
		for i := range got {
			for j := range got[i] {
				if math.Float32bits(got[i][j]) != math.Float32bits(controlCopy[i][j]) {
					t.Fatalf("call %d group[%d][%d] bits=%08x want allocating %08x", call, i, j,
						math.Float32bits(got[i][j]), math.Float32bits(controlCopy[i][j]))
				}
			}
			cpu := q4kGemm(m.q4kw[names[i]], X, P)
			cos, maxRel := cosineAndMaxRel(got[i], cpu)
			if cos < 0.9999 || maxRel > 0.05 {
				t.Fatalf("call %d group[%d] vs CPU cosine=%g maxRel=%g", call, i, cos, maxRel)
			}
		}
	}
	if len(candidateSession.q4kMLPOutputSlab) != P*128 || cap(candidateSession.q4kMLPOutputSlab) != P*128 {
		t.Fatalf("candidate slab len/cap=%d/%d want %d/%d", len(candidateSession.q4kMLPOutputSlab), cap(candidateSession.q4kMLPOutputSlab), P*128, P*128)
	}
	if got, want := candidateSession.q4kMLPOutputSlabStats, (q4kMLPOutputSlabStats{Calls: 65, Allocations: 1, Reuses: 64, HighWaterBytes: P * 128 * 4}); got != want {
		t.Fatalf("candidate slab stats = %+v, want %+v", got, want)
	}

	isolationSession := m.NewSession()
	isolationSession.MetalQ4K = true
	isolationSession.Q4KGateUpOutputSlab = true
	isolated := isolationSession.q4kGemmGroupDispatch(names, X, P)
	if isolated == nil {
		t.Fatal("isolated session group returned nil")
	}
	if &isolated[0][0] == stable {
		t.Fatal("two sessions share gate/up slab backing")
	}

	receipt := parseQ4KMLPOutputSlabTestReceipt(t, captureStderr(t, candidateSession.Close))
	if receipt.Calls != 65 || receipt.Allocations != 1 || receipt.Reuses != 64 || receipt.HighWaterBytes != P*128*4 {
		t.Fatalf("candidate close receipt = %+v", receipt)
	}
	if candidateSession.q4kMLPOutputSlab != nil {
		t.Fatal("Session.Close retained gate/up slab")
	}
	if got := captureStderr(t, controlSession.Close); got != "" {
		t.Fatalf("default-off Session.Close wrote %q", got)
	}
	_ = captureStderr(t, isolationSession.Close)

	largeSession := m.NewSession()
	largeSession.MetalQ4K = true
	largeSession.Q4KGateUpOutputSlab = true
	largeP := q4kMLPOutputSlabMaxTokens + 1
	large := largeSession.q4kGemmGroupDispatch(names, randomVecF(largeP*in, 9103), largeP)
	if large == nil {
		t.Fatal("P>512 allocating fallback returned nil")
	}
	if largeSession.q4kMLPOutputSlab != nil {
		t.Fatal("P>512 fallback retained a slab")
	}
	largeSession.Close()

	unsupportedNames := []string{"blk.0.self_attn.q_proj.weight", "blk.0.self_attn.v_proj.weight"}
	for i, name := range unsupportedNames {
		m.q4kw[name] = randomQ4KTensor(32, in, int64(9200+i))
	}
	unsupportedSession := m.NewSession()
	unsupportedSession.MetalQ4K = true
	unsupportedSession.Q4KGateUpOutputSlab = true
	unsupported := unsupportedSession.q4kGemmGroupDispatch(unsupportedNames, X, P)
	if unsupported == nil {
		t.Fatal("unsupported-group allocating fallback returned nil")
	}
	if unsupportedSession.q4kMLPOutputSlab != nil {
		t.Fatal("unsupported group retained a gate/up slab")
	}
	unsupportedSession.Close()
}

func TestMetalQ4KMLPOutputSlabProductionPrefillReusesAcrossLayersAndChunks(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("no Metal device available")
	}
	defer metalgemm.ResetQ4K()

	cfg := qwen35HybridQ4KTestCfg()
	m := NewSynthetic(cfg)
	m.Quantize()
	fillQ4KMajority(t, m, cfg)
	first := []int{3, 7, 11, 5, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61}
	second := []int{67, 71, 73, 79, 83, 89, 97}

	control := m.NewSession()
	control.Q4K = true
	control.MetalQ4K = true
	wantFirst := control.Prefill(first)
	wantSecond := control.Prefill(second)
	if control.q4kMLPOutputSlabStats != (q4kMLPOutputSlabStats{}) {
		t.Fatalf("default-off production prefill stats = %+v", control.q4kMLPOutputSlabStats)
	}

	candidate := m.NewSession()
	candidate.Q4K = true
	candidate.MetalQ4K = true
	candidate.Q4KGateUpOutputSlab = true
	gotFirst := candidate.Prefill(first)
	if got, want := candidate.q4kMLPOutputSlabStats, (q4kMLPOutputSlabStats{Calls: 4, Allocations: 1, Reuses: 3, HighWaterBytes: uint64(len(first) * cfg.IntermediateSize * 2 * 4)}); got != want {
		t.Fatalf("first production chunk stats = %+v, want %+v", got, want)
	}
	stable := &candidate.q4kMLPOutputSlab[0]
	gotSecond := candidate.Prefill(second)
	if &candidate.q4kMLPOutputSlab[0] != stable {
		t.Fatal("second production prefill chunk replaced the session slab backing")
	}
	if got, want := candidate.q4kMLPOutputSlabStats, (q4kMLPOutputSlabStats{Calls: 8, Allocations: 1, Reuses: 7, HighWaterBytes: uint64(len(first) * cfg.IntermediateSize * 2 * 4)}); got != want {
		t.Fatalf("two production chunks stats = %+v, want %+v", got, want)
	}
	if candidate.q4kHybridPrefillChunks != 2 || candidate.q4kHybridPrefillLastBase != len(first) {
		t.Fatalf("production prefill marker = (chunks=%d base=%d), want (2,%d)", candidate.q4kHybridPrefillChunks, candidate.q4kHybridPrefillLastBase, len(first))
	}
	assertFloat32BitsEqual(t, "production slab first-chunk logits", gotFirst, wantFirst)
	assertFloat32BitsEqual(t, "production slab second-chunk logits", gotSecond, wantSecond)
	assertKVCacheQuantCloseTol(t, "production slab chunked cache", control.Cache, candidate.Cache, prefillQ4KKTol(), prefillQ4KVTol())
	assertLinearAttnCacheQuantClose(t, "production slab chunked linear cache", control.Cache.linear, candidate.Cache.linear)

	if got := captureStderr(t, control.Close); got != "" {
		t.Fatalf("default-off production Session.Close wrote %q", got)
	}
	receipt := parseQ4KMLPOutputSlabTestReceipt(t, captureStderr(t, candidate.Close))
	if receipt.Calls != 8 || receipt.Allocations != 1 || receipt.Reuses != 7 || receipt.HighWaterBytes != uint64(len(first)*cfg.IntermediateSize*2*4) {
		t.Fatalf("production close receipt = %+v", receipt)
	}
}

func TestQ4KMLPOutputSlabCloseDefaultOffIsSilent(t *testing.T) {
	s := (&Model{}).NewSession()
	if got := captureStderr(t, s.Close); got != "" {
		t.Fatalf("default-off close wrote %q", got)
	}
}

func TestQ4KMLPOutputSlabCloseReceiptIsStructured(t *testing.T) {
	s := (&Model{}).NewSession()
	s.q4kMLPOutputSlab = make([]float32, 32)
	s.q4kMLPOutputSlabStats = q4kMLPOutputSlabStats{Calls: 64, Allocations: 1, Reuses: 63, HighWaterBytes: q4kMLPOutputSlabMaxBytes}
	receipt := parseQ4KMLPOutputSlabTestReceipt(t, captureStderr(t, s.Close))
	if receipt.Calls != 64 || receipt.Allocations != 1 || receipt.Reuses != 63 || receipt.HighWaterBytes != q4kMLPOutputSlabMaxBytes {
		t.Fatalf("P=512 one-chunk receipt = %+v", receipt)
	}
	if s.q4kMLPOutputSlab != nil {
		t.Fatal("close receipt retained slab backing")
	}
}

func TestQwen38MLPOutputSlabExactAllocationArithmetic(t *testing.T) {
	const (
		computeLayers    = int64(64)
		panelTokens      = int64(512)
		promptTokens     = int64(32_800)
		intermediate     = int64(17_408)
		outputsPerLayer  = int64(2)
		bytesPerFloat32  = int64(4)
		wantSlabBytes    = int64(71_303_168)
		wantTraversal    = int64(4_563_402_752)
		wantTrafficBytes = int64(292_342_988_800)
		gib              = int64(1 << 30)
	)
	slabBytes := panelTokens * intermediate * outputsPerLayer * bytesPerFloat32
	if slabBytes != wantSlabBytes {
		t.Fatalf("P=512 gate/up bytes=%d want %d", slabBytes, wantSlabBytes)
	}
	if slabBytes != q4kMLPOutputSlabMaxBytes {
		t.Fatalf("runtime slab cap=%d want exact checkpoint bytes %d", q4kMLPOutputSlabMaxBytes, slabBytes)
	}
	if full, tail := promptTokens/panelTokens, promptTokens%panelTokens; full != 64 || tail != 32 {
		t.Fatalf("P=32800 decomposition=%d full + %d tail, want 64 + 32", full, tail)
	}
	traversalBytes := computeLayers * slabBytes
	if traversalBytes != wantTraversal || traversalBytes*4 != 17*gib {
		t.Fatalf("64-layer full-chunk traffic=%d bytes, want %d bytes (4.25 GiB)", traversalBytes, wantTraversal)
	}
	trafficBytes := computeLayers * promptTokens * intermediate * outputsPerLayer * bytesPerFloat32
	if trafficBytes != wantTrafficBytes {
		t.Fatalf("64-layer P=32800 allocation traffic=%d want %d", trafficBytes, wantTrafficBytes)
	}
	if trafficBytes*64 != 17_425*gib {
		t.Fatalf("64-layer P=32800 allocation traffic=%d bytes, want exactly 272.265625 GiB (272.266 rounded)", trafficBytes)
	}
}

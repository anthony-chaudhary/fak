//go:build cuda

package compute

import (
	"encoding/binary"
	"math"
	"testing"
)

// cuda_quant_test.go — the `-tags cuda` witness for issue #485 (native Q8_0 / Q4_K device matmul:
// the weight stays NARROW in VRAM — int8 codes / Q4_K super-block bytes — and the GEMM consumes it
// directly, no dequant-to-f32 round trip). The CUDA backend ran F32 SGEMM (#484 added F16 HGEMM);
// it panicked on quantized weights. This file holds each dtype to its OWN recorded Approx cosine
// floor against the cpuref f32 Reference (NOT bit-identity — RequireReference keeps the device off
// the exact rungs), and witnesses that the resident weight is int8/int4-sized, not f32-sized.
//
// PER-DTYPE GATES (distinct numeric paths, distinct recorded floors):
//   - Q8_0 (cudaQ8CosineMin=0.999): the f32 weight is narrowed to int8 codes + per-block f32 scales
//     at H2D (Upload(t, Q8_0) under Caps.UploadDtype); the activation is quantized to int8 ON DEVICE
//     per block; the per-block integer dot is scaled by (weight·activation block scales), F32
//     accumulate. This measures the REAL Q8 weight+activation quant error against the unquantized
//     f32 reference — exactly the model's Q8 lane.
//   - Q4_K (cudaQ4KCosineMin=0.995, its OWN gate instance): the raw GGUF super-block bytes are
//     resident and the dequant (w = d·scale·code − dmin·min, getScaleMinK4 geometry) is FUSED into
//     the GEMM tile. The reference is an f32 dequant of the SAME super-block bytes, so this gate
//     isolates the device dequant-fused tile's arithmetic — it witnesses the getScaleMinK4 geometry
//     is reproduced bit-for-bit on device (a wrong port collapses the cosine). The full-model
//     true-f32 → Q4_K reconstruction cosine is the GPU residual; the recorded floor is set looser
//     than Q8 for the reason the constant's comment in cuda.go gives.
//
// VRAM WITNESS: after Upload, the resident weight's byte count is read directly off the device
// buffer and asserted ≈ the int8/int4 size (codes [+ thin scale band] for Q8_0; super-block bytes
// for Q4_K), a small fraction of the f32 size a dequant-to-f32 upload would have paid.
//
// HARDWARE: the GEMM RUN, the realized per-dtype cosines, the VRAM numbers, and the quantized-vs-f32
// tok/s all need a CUDA node — they are the explicit residual of this build+commit handoff. On the
// win32 dev host (no CUDA toolkit) these skip cleanly; run them on a GPU node via
// tools/run_485_acceptance_on_gpu.sh. The Go + cgo here type-checks under `go vet -tags cuda`.

// ---- Q8_0 gate ------------------------------------------------------------------

// alignActToRow makes the target output channel the clear argmax winner by setting the activation
// equal to that weight row, so y[target] = |W[target]|² dominates every cross term — a robust,
// quant-noise-proof argmax-exact check. The cosine precision gate then runs over the OTHER channels
// (the dominant self-term would otherwise mask the per-channel quant error), so the two halves of
// the "argmax-exact + cosine" gate measure different things and neither hides the other.
func alignActToRow(w []float32, out, in, target int) []float32 {
	x := make([]float32, in)
	copy(x, w[target*in:target*in+in])
	return x
}

func nonTarget(v []float32, out, target int) []float32 {
	r := make([]float32, 0, out-1)
	for o := 0; o < out; o++ {
		if o != target {
			r = append(r, v[o])
		}
	}
	return r
}

// TestCUDAQ8MatMulApproxMatchesRef — native Q8_0 decode GEMV (P=1): an f32 weight narrowed to a
// resident Q8_0 weight (int8 codes + f32 scales) at H2D, the activation quantized on-device, vs the
// cpuref f32 fdot matmul on the SAME source f32 weight. The Approx gate is argmax-exact (the device
// quantized GEMM picks the same winning channel) + cosine ≥ the RECORDED cudaQ8CosineMin (0.999).
func TestCUDAQ8MatMulApproxMatchesRef(t *testing.T) {
	cb := cudaOrSkip(t)
	if !cb.Caps().UploadDtype {
		t.Fatal("#485: cuda backend must advertise Caps.UploadDtype=true (Q8_0 narrowing at H2D)")
	}
	ref := Default() // cpu-ref
	var seed lcg = 485
	g := &seed
	out, in := 320, 256 // in divisible by 32 (Q8 block)
	w := rscale(g, out*in, 0.2)
	const target = 0
	x := alignActToRow(w, out, in, target)

	yRef := ref.Read(ref.MatMul(mkResident(ref, []int{out, in}, w), mkResident(ref, []int{in}, x)))

	wQ8 := cb.Upload(NewF32(cb, []int{out, in}, w), Q8_0)
	if wQ8.Dtype != Q8_0 {
		t.Fatalf("Upload(_, Q8_0) produced Dtype %s, want q8_0", wQ8.Dtype)
	}
	yCu := cb.Read(cb.MatMul(wQ8, mkResident(cb, []int{in}, x)))
	if len(yRef) != out || len(yCu) != out {
		t.Fatalf("shape ref=%d cu=%d want %d", len(yRef), len(yCu), out)
	}

	// argmax-exact: the dominant target channel must win on both paths, and the device Argmax kernel
	// must agree with a host argmax over the device output.
	if a := argmaxF32(yRef); a != target {
		t.Fatalf("reference argmax %d != constructed dominant channel %d", a, target)
	}
	aCu := cb.Argmax(mkResident(cb, []int{out}, yCu))
	if aCu != argmaxF32(yCu) || argmaxF32(yCu) != argmaxF32(yRef) {
		t.Fatalf("Q8 argmax-exact failed: ref=%d cudaHost=%d cudaKernel=%d", argmaxF32(yRef), argmaxF32(yCu), aCu)
	}
	// cosine over the non-dominant channels — the real per-channel Q8 quant precision gate.
	c := cosine(nonTarget(yRef, out, target), nonTarget(yCu, out, target))
	if c < cudaQ8CosineMin {
		t.Fatalf("Q8 MatMul cosine %.6f < recorded Q8 gate %.6f (cudaQ8CosineMin)", c, cudaQ8CosineMin)
	}
	t.Logf("#485 Q8_0 MatMul: cosine=%.8f maxAbs=%.2e gate=%.4f argmax-exact (device=%s tier=%s class=%s)",
		c, maxAbsDelta(yRef, yCu), cudaQ8CosineMin, cb.Name(), cb.Tier(), cb.Class())
}

// TestCUDAQ8BatchedMatMulApproxMatchesRef — native Q8_0 prefill GEMM (P>1): per-row on-device
// activation quant against the resident int8 weight, vs the cpuref f32 BatchedMatMul. cosine over
// the full Y ≥ cudaQ8CosineMin.
func TestCUDAQ8BatchedMatMulApproxMatchesRef(t *testing.T) {
	cb := cudaOrSkip(t)
	ref := Default()
	var seed lcg = 0x485b
	g := &seed
	out, in, P := 320, 256, 8
	w := rscale(g, out*in, 0.2)
	X := rscale(g, P*in, 1.0)

	YRef := ref.Read(ref.BatchedMatMul(mkResident(ref, []int{out, in}, w), mkResident(ref, []int{P, in}, X), P))
	wQ8 := cb.Upload(NewF32(cb, []int{out, in}, w), Q8_0)
	YCu := cb.Read(cb.BatchedMatMul(wQ8, mkResident(cb, []int{P, in}, X), P))
	if len(YRef) != P*out || len(YCu) != P*out {
		t.Fatalf("shape ref=%d cu=%d want %d", len(YRef), len(YCu), P*out)
	}
	c := cosine(YRef, YCu)
	if c < cudaQ8CosineMin {
		t.Fatalf("Q8 BatchedMatMul cosine %.6f < recorded Q8 gate %.6f", c, cudaQ8CosineMin)
	}
	t.Logf("#485 Q8_0 BatchedMatMul (P=%d): cosine=%.8f maxAbs=%.2e gate=%.4f", P, c, maxAbsDelta(YRef, YCu), cudaQ8CosineMin)
}

// ---- Q4_K gate (its OWN instance — distinct numeric path) -----------------------

// getScaleMinK4Test mirrors internal/ggufload getScaleMinK4 bit-for-bit: the 6-bit (scale,min) for
// the j-th 32-elem sub-block, unpacked from the 12 packed scale bytes. The device kernel's
// getScaleMinK4_dev must reproduce exactly this — the Q4_K gate is what witnesses it does.
func getScaleMinK4Test(j int, q []byte) (scale, min uint8) {
	if j < 4 {
		return q[j] & 63, q[j+4] & 63
	}
	return (q[j+4] & 0x0f) | ((q[j-4] >> 6) << 4), (q[j+4] >> 4) | ((q[j] >> 6) << 4)
}

// f16BitsToF32 / f32ToF16Bits are the host f16 round-trip the witness uses to author and read the
// super-block d/dmin. The device reads the SAME stored f16 bits via __half2float, so both paths
// agree on the d/dmin value regardless of the rounding f32ToF16Bits applies (the values authored
// are small positive normals, well inside f16's normal range).
func f16BitsToF32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h & 0x3ff)
	switch {
	case exp == 0 && mant == 0:
		return math.Float32frombits(sign)
	case exp == 0x1f:
		return math.Float32frombits(sign | 0x7f800000 | (mant << 13))
	default:
		// normal (subnormals not authored by this witness): rebias 5-bit exp -> 8-bit.
		return math.Float32frombits(sign | ((exp + 112) << 23) | (mant << 13))
	}
}

func f32ToF16Bits(f float32) uint16 {
	b := math.Float32bits(f)
	sign := uint16((b >> 16) & 0x8000)
	exp := int32((b>>23)&0xff) - 127 + 15
	mant := b & 0x7fffff
	if exp <= 0 {
		return sign // flush tiny to zero (the witness authors values well above this)
	}
	if exp >= 0x1f {
		return sign | 0x7c00 // inf (not authored)
	}
	half := uint16(exp<<10) | uint16(mant>>13)
	if mant&0x1000 != 0 { // round half up — carry into exp is well-defined
		half++
	}
	return sign | half
}

const blockQ4KBytesTest = 144 // f16 d + f16 dmin + 12 packed sub-scales + 128 4-bit codes (256 elems)

// randQ4KBytes authors random but VALID Q4_K super-block bytes for a [out,in] weight (row-major by
// output channel; in divisible by 256). Small positive f16 d/dmin keep the dequantized magnitudes
// sane; the 12 scale bytes and 128 code bytes are pseudo-random (every 6-bit field masks into
// [0,63] and every nibble into [0,15] by construction, so the bytes are always a valid block).
func randQ4KBytes(g *lcg, out, in int) []byte {
	nsb := in / 256
	raw := make([]byte, out*nsb*blockQ4KBytesTest)
	for blk := 0; blk < out*nsb; blk++ {
		base := blk * blockQ4KBytesTest
		d := 0.01 + absf(g.f())*0.03
		dmin := 0.004 + absf(g.f())*0.012
		binary.LittleEndian.PutUint16(raw[base:], f32ToF16Bits(d))
		binary.LittleEndian.PutUint16(raw[base+2:], f32ToF16Bits(dmin))
		for i := 0; i < 12+128; i++ {
			raw[base+4+i] = byte(int32((g.f()+0.5)*255.0) & 0xff)
		}
	}
	return raw
}

// dequantQ4KWeight dequantizes the whole [out,in] Q4_K byte weight to f32, mirroring
// internal/ggufload dequantQ4K exactly — this is the f32 Reference the device Q4_K GEMM is held to.
func dequantQ4KWeight(raw []byte, out, in int) []float32 {
	nsb := in / 256
	rowBytes := nsb * blockQ4KBytesTest
	w := make([]float32, out*in)
	for o := 0; o < out; o++ {
		for sb := 0; sb < nsb; sb++ {
			base := o*rowBytes + sb*blockQ4KBytesTest
			d := f16BitsToF32(binary.LittleEndian.Uint16(raw[base:]))
			dmin := f16BitsToF32(binary.LittleEndian.Uint16(raw[base+2:]))
			scales := raw[base+4 : base+16]
			q := raw[base+16 : base+blockQ4KBytesTest]
			qi, is := 0, 0
			yi := o*in + sb*256
			for j := 0; j < 256; j += 64 {
				sc, m := getScaleMinK4Test(is, scales)
				d1, m1 := d*float32(sc), dmin*float32(m)
				sc, m = getScaleMinK4Test(is+1, scales)
				d2, m2 := d*float32(sc), dmin*float32(m)
				for l := 0; l < 32; l++ {
					w[yi+j+l] = d1*float32(q[qi+l]&0x0f) - m1
				}
				for l := 0; l < 32; l++ {
					w[yi+j+32+l] = d2*float32(q[qi+l]>>4) - m2
				}
				qi += 32
				is += 2
			}
		}
	}
	return w
}

// newQ4KHost wraps raw Q4_K super-block bytes as a host Tensor (one int8 per byte — the byte values
// are preserved through the two's-complement reinterpret). Upload(t, Q4_K) copies them resident.
func newQ4KHost(be Backend, shape []int, raw []byte) Tensor {
	i8 := make([]int8, len(raw))
	for i, b := range raw {
		i8[i] = int8(b)
	}
	q := &QuantSpec{Block: 256, Axis: 2, Bits: 4, Symmetric: false}
	return makeTensor(be, Q4_K, RowMajor, append([]int(nil), shape...), q, &hostBuf{i8: i8})
}

// TestCUDAQ4KMatMulApproxMatchesRef — native Q4_K decode GEMV with the dequant fused into the tile,
// vs the cpuref f32 matmul over an f32 dequant of the SAME super-block bytes. Its OWN gate instance:
// argmax-exact (activation aligned to the dominant channel) + cosine ≥ the RECORDED cudaQ4KCosineMin
// (0.995, looser than Q8 — see cuda.go). This isolates the device dequant-fused tile: a wrong
// getScaleMinK4 / super-block port collapses the cosine.
func TestCUDAQ4KMatMulApproxMatchesRef(t *testing.T) {
	cb := cudaOrSkip(t)
	ref := Default()
	var seed lcg = 0x4b4b
	g := &seed
	out, in := 320, 256 // in divisible by 256 (one super-block per row)
	raw := randQ4KBytes(g, out, in)
	w := dequantQ4KWeight(raw, out, in) // the f32 reference weight (dequant of the bytes)
	const target = 0
	x := alignActToRow(w, out, in, target)

	yRef := ref.Read(ref.MatMul(mkResident(ref, []int{out, in}, w), mkResident(ref, []int{in}, x)))

	wQ4K := cb.Upload(newQ4KHost(cb, []int{out, in}, raw), Q4_K)
	if wQ4K.Dtype != Q4_K {
		t.Fatalf("Upload(_, Q4_K) produced Dtype %s, want q4_k", wQ4K.Dtype)
	}
	yCu := cb.Read(cb.MatMul(wQ4K, mkResident(cb, []int{in}, x)))
	if len(yRef) != out || len(yCu) != out {
		t.Fatalf("shape ref=%d cu=%d want %d", len(yRef), len(yCu), out)
	}

	if a := argmaxF32(yRef); a != target {
		t.Fatalf("reference argmax %d != constructed dominant channel %d", a, target)
	}
	aCu := cb.Argmax(mkResident(cb, []int{out}, yCu))
	if aCu != argmaxF32(yCu) || argmaxF32(yCu) != argmaxF32(yRef) {
		t.Fatalf("Q4_K argmax-exact failed: ref=%d cudaHost=%d cudaKernel=%d", argmaxF32(yRef), argmaxF32(yCu), aCu)
	}
	c := cosine(nonTarget(yRef, out, target), nonTarget(yCu, out, target))
	if c < cudaQ4KCosineMin {
		t.Fatalf("Q4_K MatMul cosine %.6f < recorded Q4_K gate %.6f (cudaQ4KCosineMin)", c, cudaQ4KCosineMin)
	}
	t.Logf("#485 Q4_K MatMul: cosine=%.8f maxAbs=%.2e gate=%.4f argmax-exact (device=%s tier=%s class=%s)",
		c, maxAbsDelta(yRef, yCu), cudaQ4KCosineMin, cb.Name(), cb.Tier(), cb.Class())
}

// TestCUDAQ4KBatchedMatMulApproxMatchesRef — native Q4_K prefill GEMM, dequant fused into the tile,
// vs the cpuref f32 BatchedMatMul over the dequant of the same bytes. cosine over the full Y ≥
// cudaQ4KCosineMin.
func TestCUDAQ4KBatchedMatMulApproxMatchesRef(t *testing.T) {
	cb := cudaOrSkip(t)
	ref := Default()
	var seed lcg = 0x4b4c
	g := &seed
	out, in, P := 320, 256, 8
	raw := randQ4KBytes(g, out, in)
	w := dequantQ4KWeight(raw, out, in)
	X := rscale(g, P*in, 1.0)

	YRef := ref.Read(ref.BatchedMatMul(mkResident(ref, []int{out, in}, w), mkResident(ref, []int{P, in}, X), P))
	wQ4K := cb.Upload(newQ4KHost(cb, []int{out, in}, raw), Q4_K)
	YCu := cb.Read(cb.BatchedMatMul(wQ4K, mkResident(cb, []int{P, in}, X), P))
	if len(YRef) != P*out || len(YCu) != P*out {
		t.Fatalf("shape ref=%d cu=%d want %d", len(YRef), len(YCu), P*out)
	}
	c := cosine(YRef, YCu)
	if c < cudaQ4KCosineMin {
		t.Fatalf("Q4_K BatchedMatMul cosine %.6f < recorded Q4_K gate %.6f", c, cudaQ4KCosineMin)
	}
	t.Logf("#485 Q4_K BatchedMatMul (P=%d): cosine=%.8f maxAbs=%.2e gate=%.4f", P, c, maxAbsDelta(YRef, YCu), cudaQ4KCosineMin)
}

// ---- VRAM witness: resident weight ≈ int8/int4 size, NOT f32 --------------------

// residentWeightBytes reads the resident weight's VRAM footprint straight off the device buffer
// (codes [+ scale band] for Q8_0; super-block bytes for Q4_K) — never a host self-report.
func residentWeightBytes(w Tensor) int {
	if b, ok := w.buf.(*cudaBuf); ok {
		return b.residentBytes()
	}
	return -1
}

// TestCUDAQuantVRAMWitness proves the native quantized upload keeps the weight NARROW: the same f32
// weight uploaded as Q8_0 and as Q4_K is resident at ≈ its int8/int4 size, a small fraction of the
// f32 bytes a dequant-to-f32 upload would have paid. The byte counts are read off the device
// buffers (residentWeightBytes), the whole point of #485 (the VRAM/bandwidth win is kept).
func TestCUDAQuantVRAMWitness(t *testing.T) {
	cb := cudaOrSkip(t)
	var seed lcg = 0x485f
	g := &seed
	out, in := 512, 256
	f32Bytes := out * in * 4

	// Q8_0: int8 codes (out*in) + per-block(32) f32 scales (out*(in/32)*4).
	w := rscale(g, out*in, 0.2)
	wQ8 := cb.Upload(NewF32(cb, []int{out, in}, w), Q8_0)
	gotQ8 := residentWeightBytes(wQ8)
	q8b := wQ8.buf.(*cudaBuf)
	wantCodes := out * in
	wantScales := out * (in / q8DeviceBlock) * 4
	if q8b.n != wantCodes {
		t.Fatalf("Q8_0 resident codes = %d bytes, want int8 size %d", q8b.n, wantCodes)
	}
	if q8b.scalesN != wantScales {
		t.Fatalf("Q8_0 resident scales = %d bytes, want %d", q8b.scalesN, wantScales)
	}
	if gotQ8 >= f32Bytes/3 {
		t.Fatalf("Q8_0 resident %d bytes not << f32 %d (want < f32/3)", gotQ8, f32Bytes)
	}

	// Q4_K: raw super-block bytes (out*(in/256)*144), ≈ 0.56 byte/elem.
	raw := randQ4KBytes(g, out, in)
	wQ4K := cb.Upload(newQ4KHost(cb, []int{out, in}, raw), Q4_K)
	gotQ4K := residentWeightBytes(wQ4K)
	wantQ4K := out * (in / 256) * blockQ4KBytesTest
	if gotQ4K != wantQ4K {
		t.Fatalf("Q4_K resident %d bytes, want super-block size %d", gotQ4K, wantQ4K)
	}
	if gotQ4K >= f32Bytes/4 {
		t.Fatalf("Q4_K resident %d bytes not << f32 %d (want < f32/4)", gotQ4K, f32Bytes)
	}

	t.Logf("#485 VRAM witness ([%d,%d] weight): f32=%d B | Q8_0=%d B (%.2fx smaller) | Q4_K=%d B (%.2fx smaller)",
		out, in, f32Bytes, gotQ8, float64(f32Bytes)/float64(gotQ8), gotQ4K, float64(f32Bytes)/float64(gotQ4K))
}

// ---- throughput: native quantized GEMM vs the F32 SGEMM baseline ----------------
// The acceptance script turns these ns/op into the quantized-vs-f32 GEMM throughput delta (the
// VRAM/bandwidth win should show as faster decode). The baseline is BenchmarkCUDABatchedMatMulF32
// (cuda_fp16_test.go), the SAME dims (fp16BenchDims).

// BenchmarkCUDAQ8BatchedMatMul — the prefill GEMM with a resident Q8_0 weight (on-device activation
// quant + int dot), same dims as the F32/F16 baselines.
func BenchmarkCUDAQ8BatchedMatMul(b *testing.B) {
	cb := cudaTBOrSkip(b)
	out, in, P := fp16BenchDims()
	var seed lcg = 485
	g := &seed
	w := cb.Upload(NewF32(cb, []int{out, in}, rscale(g, out*in, 0.05)), Q8_0)
	X := mkResident(cb, []int{P, in}, rscale(g, P*in, 1.0))
	cb.Read(cb.BatchedMatMul(w, X, P)) // warm: pool + weight cache
	cb.Recycle()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cb.Read(cb.BatchedMatMul(w, X, P))
		cb.Recycle()
	}
	b.StopTimer()
}

// BenchmarkCUDAQ4KBatchedMatMul — the same GEMM with a resident Q4_K weight (dequant fused into the
// tile), same dims.
func BenchmarkCUDAQ4KBatchedMatMul(b *testing.B) {
	cb := cudaTBOrSkip(b)
	out, in, P := fp16BenchDims()
	var seed lcg = 485
	g := &seed
	raw := randQ4KBytes(g, out, in)
	w := cb.Upload(newQ4KHost(cb, []int{out, in}, raw), Q4_K)
	X := mkResident(cb, []int{P, in}, rscale(g, P*in, 1.0))
	cb.Read(cb.BatchedMatMul(w, X, P)) // warm
	cb.Recycle()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cb.Read(cb.BatchedMatMul(w, X, P))
		cb.Recycle()
	}
	b.StopTimer()
}

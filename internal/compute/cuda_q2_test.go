//go:build cuda

package compute

import "testing"

// cuda_q2_test.go — the `-tags cuda` device witness for issue #4872 (native packed-ternary
// Q2_0 device GEMM: the weight stays 0.25 byte/elem in VRAM — 2-bit codes + per-block f32
// scales — and the GEMM consumes it directly, no dequant-to-f32 round trip). It holds the
// device k_q2_0_gemm to the cpuref ternary GEMV over the SAME packed codes+scales:
// argmax-exact + cosine ≥ the RECORDED cudaQ2CosineMin (0.999), and witnesses the resident
// weight is 2-bit-sized. Named *Q2_0GEMM* so the acceptance command is
//   go test -tags cuda ./internal/compute -run Q2_0GEMM
// On a host with no CUDA toolkit these skip cleanly (cudaOrSkip); they RUN on a GPU node.
// randTernaryWeight / dequantQ2Weight / q2Argmax are shared with quant_q2_test.go.

// TestCUDAQ2_0GEMMMatMulApproxMatchesRef — native packed-ternary decode GEMV (P=1): a resident
// Q2_0 weight (2-bit codes + f32 scales) with the signed indicator unpacked on device and
// multiply-accumulated against the f32 activation, vs the cpuref f32 GEMV over an f32 dequant of
// the SAME codes+scales. Approx gate: argmax-exact + cosine ≥ cudaQ2CosineMin.
func TestCUDAQ2_0GEMMMatMulApproxMatchesRef(t *testing.T) {
	cb := cudaOrSkip(t)
	ref := Default() // cpu-ref
	const out, in = 320, 256 // in divisible by 32
	packed, scale := randTernaryWeight(0x4872, out, in, q2Block)
	dense := dequantQ2Weight(packed, scale, out, in, q2Block)
	target := dominantRow(dense, out, in)
	x := alignActToRow(dense, out, in, target)

	yRef := ref.Read(ref.MatMul(mkResident(ref, []int{out, in}, dense), mkResident(ref, []int{in}, x)))

	wQ2 := cb.Upload(NewQ2(cb, []int{out, in}, packed, scale, q2Block), Q2_0)
	if wQ2.Dtype != Q2_0 {
		t.Fatalf("Upload(_, Q2_0) produced Dtype %s, want q2_0", wQ2.Dtype)
	}
	yCu := cb.Read(cb.MatMul(wQ2, mkResident(cb, []int{in}, x)))
	if len(yRef) != out || len(yCu) != out {
		t.Fatalf("shape ref=%d cu=%d want %d", len(yRef), len(yCu), out)
	}

	if a := argmaxF32(yRef); a != target {
		t.Fatalf("reference argmax %d != constructed dominant channel %d", a, target)
	}
	aCu := cb.Argmax(mkResident(cb, []int{out}, yCu))
	if aCu != argmaxF32(yCu) || argmaxF32(yCu) != argmaxF32(yRef) {
		t.Fatalf("Q2_0 argmax-exact failed: ref=%d cudaHost=%d cudaKernel=%d", argmaxF32(yRef), argmaxF32(yCu), aCu)
	}
	c := cosine(nonTarget(yRef, out, target), nonTarget(yCu, out, target))
	if c < cudaQ2CosineMin {
		t.Fatalf("Q2_0 MatMul cosine %.6f < recorded Q2_0 gate %.6f (cudaQ2CosineMin)", c, cudaQ2CosineMin)
	}
	t.Logf("#4872 Q2_0 MatMul: cosine=%.8f maxAbs=%.2e gate=%.4f argmax-exact (device=%s tier=%s class=%s)",
		c, maxAbsDelta(yRef, yCu), cudaQ2CosineMin, cb.Name(), cb.Tier(), cb.Class())
}

// TestCUDAQ2_0GEMMBatchedMatMulApproxMatchesRef — native packed-ternary prefill GEMM (P>1): each
// f32 activation row dotted against the resident 2-bit weight, vs the cpuref f32 BatchedMatMul over
// the dequant of the same codes+scales. cosine over the full Y ≥ cudaQ2CosineMin.
func TestCUDAQ2_0GEMMBatchedMatMulApproxMatchesRef(t *testing.T) {
	cb := cudaOrSkip(t)
	ref := Default()
	const out, in, P = 320, 256, 8
	packed, scale := randTernaryWeight(0x4872b, out, in, q2Block)
	dense := dequantQ2Weight(packed, scale, out, in, q2Block)
	var seed lcg = 0x4872c
	X := rscale(&seed, P*in, 1.0)

	YRef := ref.Read(ref.BatchedMatMul(mkResident(ref, []int{out, in}, dense), mkResident(ref, []int{P, in}, X), P))
	wQ2 := cb.Upload(NewQ2(cb, []int{out, in}, packed, scale, q2Block), Q2_0)
	YCu := cb.Read(cb.BatchedMatMul(wQ2, mkResident(cb, []int{P, in}, X), P))
	if len(YRef) != P*out || len(YCu) != P*out {
		t.Fatalf("shape ref=%d cu=%d want %d", len(YRef), len(YCu), P*out)
	}
	c := cosine(YRef, YCu)
	if c < cudaQ2CosineMin {
		t.Fatalf("Q2_0 BatchedMatMul cosine %.6f < recorded Q2_0 gate %.6f", c, cudaQ2CosineMin)
	}
	t.Logf("#4872 Q2_0 BatchedMatMul (P=%d): cosine=%.8f maxAbs=%.2e gate=%.4f", P, c, maxAbsDelta(YRef, YCu), cudaQ2CosineMin)
}

// TestCUDAQ2_0GEMMVRAMWitness proves the packed-ternary upload keeps the weight at 2-bit density:
// the resident codes are out*in/4 bytes plus a thin per-block f32 scale band — ≈0.25 byte/elem of
// codes, 16× narrower than the f32 weight. Read straight off the device buffer, never self-reported.
func TestCUDAQ2_0GEMMVRAMWitness(t *testing.T) {
	cb := cudaOrSkip(t)
	const out, in = 512, 256
	packed, scale := randTernaryWeight(0x4872f, out, in, q2Block)
	wQ2 := cb.Upload(NewQ2(cb, []int{out, in}, packed, scale, q2Block), Q2_0)

	b := wQ2.buf.(*cudaBuf)
	wantCodes := out * in / 4
	wantScales := out * (in / q2Block) * 4
	if b.n != wantCodes {
		t.Fatalf("Q2_0 resident codes = %d bytes, want 2-bit size %d", b.n, wantCodes)
	}
	if b.scalesN != wantScales {
		t.Fatalf("Q2_0 resident scales = %d bytes, want %d", b.scalesN, wantScales)
	}
	f32Bytes := out * in * 4
	if got := residentWeightBytes(wQ2); got >= f32Bytes/8 {
		t.Fatalf("Q2_0 resident %d bytes not << f32 %d (want < f32/8)", got, f32Bytes)
	}
	t.Logf("#4872 VRAM witness ([%d,%d] weight): f32=%d B | Q2_0=%d B (%.2fx smaller)",
		out, in, f32Bytes, residentWeightBytes(wQ2), float64(f32Bytes)/float64(residentWeightBytes(wQ2)))
}

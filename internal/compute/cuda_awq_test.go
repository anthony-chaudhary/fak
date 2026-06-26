//go:build cuda

package compute

import (
	"testing"
)

// cuda_awq_test.go — the `-tags cuda` witness for issue #926: the AWQ 4-bit device matmul
// (AWQMatMul / AWQBatchedMatMul, kernels fcuda_awq_gemv / fcuda_awq_gemm) gets a cpuref-parity
// gate. AWQ was the ONE device op family with no recorded cosine floor and no acceptance witness
// (CUDA-DEV-SCORECARD cpuref_parity_coverage); this file closes that gap with the SAME
// argmax-exact + cosine gate the Q4_K dequant-fused lane uses (#485), under cudaAWQCosineMin.
//
// AWQ is a 4-bit weight-only format: nibble-packed codes (2/byte) + a PER-CHANNEL f32 scale +
// symmetric zero-point 8 (weight[o,i] = scale[o]·(code−8)). The kernel dequant-fuses the nibble
// into the GEMM tile and accumulates in F32. The reference is an f32 dequant of the SAME packed
// bytes + scales (dequantAWQWeight), so this gate isolates the device tile's arithmetic —
// reduction-order drift against the host AWQ dequant — NOT the true-f32→4-bit reconstruction
// error (the same isolation discipline as the Q4_K gate).
//
// RESIDENCY NOTE: AWQ has no Upload path of its own yet (its binding takes already-resident
// (w, scales) tensors). The witness routes the packed nibble bytes through the Q4_K raw-byte
// upload channel (cb.Upload(newQ4KHost(...), Q4_K)) — a straight H2D of out·in/2 bytes whose Q4_K
// dtype label is COSMETIC: AWQMatMul reads only w.Shape and the device pointer, never the
// dtype/QuantSpec, so the bytes reach the kernel verbatim. A future production uploadAWQ would
// reuse the same dallocWeight primitive with an honest I4 label.
//
// HARDWARE: the realized cosine needs a CUDA node — the explicit residual of this build+commit
// handoff (the win32 dev host has no CUDA toolkit / GPU). On the dev host these skip cleanly; run
// them on a GPU node via tools/run_926_acceptance_on_gpu.sh. The Go here type-checks under
// `go vet -tags cuda`.

// dequantAWQWeight dequantizes a whole [out,in] AWQ byte weight to f32, mirroring the device
// k_awq_gemv / k_awq_gemm nibble unpack bit-for-bit (low nibble = even index, high nibble = odd
// index, zero-point 8). This is the f32 Reference the device AWQ GEMM is held to.
func dequantAWQWeight(packed []byte, scales []float32, out, in int) []float32 {
	w := make([]float32, out*in)
	for o := 0; o < out; o++ {
		scale := scales[o]
		row := o * (in / 2)
		for i := 0; i < in/2; i++ {
			b := packed[row+i]
			w[o*in+i*2] = scale * float32(int16(b&0x0f)-8)
			w[o*in+i*2+1] = scale * float32(int16(b>>4)-8)
		}
	}
	return w
}

// randAWQBytes authors random but VALID AWQ packed bytes for a [out,in] weight (in even): every
// byte's two nibbles are in [0,15] by construction, so any byte is a valid AWQ packed pair.
func randAWQBytes(g *lcg, out, in int) []byte {
	raw := make([]byte, out*(in/2))
	for i := range raw {
		raw[i] = byte(int32((g.f()+0.5)*255.0) & 0xff)
	}
	return raw
}

// randAWQScales authors small positive per-channel f32 scales — the side channel an AWQ weight
// carries. The dynamic range of every channel is kept in full f32, so only the in-nibble code
// rounds (the same structure that keeps the Q8 lane tight against the f32 reference).
func randAWQScales(g *lcg, out int) []float32 {
	s := make([]float32, out)
	for o := 0; o < out; o++ {
		s[o] = 0.01 + absf(g.f())*0.03
	}
	return s
}

// uploadAWQResident makes the resident (w, scales) pair the AWQ binding consumes: w is the
// nibble-packed codes (out·in/2 bytes) at Shape [out,in]; scales is the per-channel f32 [out].
// See the file's RESIDENCY NOTE for why the cosmetic Q4_K label on w is correct-but-cosmetic.
func uploadAWQResident(cb *cudaBackend, packed []byte, scales []float32, out, in int) (Tensor, Tensor) {
	w := cb.Upload(newQ4KHost(cb, []int{out, in}, packed), Q4_K)
	s := cb.Upload(NewF32(cb, []int{out}, scales), F32)
	return w, s
}

// TestCUDAAWQMatMulApproxMatchesRef — native AWQ 4-bit decode GEMV (P=1) with the dequant fused
// into the tile, vs the cpuref f32 matmul over an f32 dequant of the SAME packed bytes + scales.
// Its OWN gate instance: argmax-exact (activation aligned to the dominant channel) + cosine over
// the non-dominant channels ≥ the RECORDED cudaAWQCosineMin (0.995, the same 4-bit dequant-fused
// class as Q4_K). Isolates the device AWQ tile: a wrong nibble unpack or scale apply collapses it.
func TestCUDAAWQMatMulApproxMatchesRef(t *testing.T) {
	cb := cudaOrSkip(t)
	ref := Default()
	var seed lcg = 0x9260
	g := &seed
	out, in := 320, 256 // in even (AWQ packs 2/byte); matches the Q4_K gate dims
	packed := randAWQBytes(g, out, in)
	scales := randAWQScales(g, out)
	w := dequantAWQWeight(packed, scales, out, in) // the f32 reference weight (dequant of the bytes)
	target := dominantRow(w, out, in)
	x := alignActToRow(w, out, in, target)

	yRef := ref.Read(ref.MatMul(mkResident(ref, []int{out, in}, w), mkResident(ref, []int{in}, x)))

	wW, sW := uploadAWQResident(cb, packed, scales, out, in)
	yCu := cb.Read(cb.AWQMatMul(wW, sW, mkResident(cb, []int{in}, x)))
	if len(yRef) != out || len(yCu) != out {
		t.Fatalf("shape ref=%d cu=%d want %d", len(yRef), len(yCu), out)
	}

	if a := argmaxF32(yRef); a != target {
		t.Fatalf("reference argmax %d != constructed dominant channel %d", a, target)
	}
	aCu := cb.Argmax(mkResident(cb, []int{out}, yCu))
	if aCu != argmaxF32(yCu) || argmaxF32(yCu) != argmaxF32(yRef) {
		t.Fatalf("AWQ argmax-exact failed: ref=%d cudaHost=%d cudaKernel=%d", argmaxF32(yRef), argmaxF32(yCu), aCu)
	}
	c := cosine(nonTarget(yRef, out, target), nonTarget(yCu, out, target))
	if c < cudaAWQCosineMin {
		t.Fatalf("AWQ MatMul cosine %.6f < recorded AWQ gate %.6f (cudaAWQCosineMin)", c, cudaAWQCosineMin)
	}
	t.Logf("#926 AWQ MatMul: cosine=%.8f maxAbs=%.2e gate=%.4f argmax-exact (device=%s tier=%s class=%s)",
		c, maxAbsDelta(yRef, yCu), cudaAWQCosineMin, cb.Name(), cb.Tier(), cb.Class())
}

// TestCUDAAWQBatchedMatMulApproxMatchesRef — native AWQ 4-bit prefill GEMM (P>1), dequant fused
// into the tile, vs the cpuref f32 BatchedMatMul over the dequant of the same bytes. cosine over
// the full Y ≥ cudaAWQCosineMin.
func TestCUDAAWQBatchedMatMulApproxMatchesRef(t *testing.T) {
	cb := cudaOrSkip(t)
	ref := Default()
	var seed lcg = 0x9261
	g := &seed
	out, in, P := 320, 256, 8
	packed := randAWQBytes(g, out, in)
	scales := randAWQScales(g, out)
	w := dequantAWQWeight(packed, scales, out, in)
	X := rscale(g, P*in, 1.0)

	YRef := ref.Read(ref.BatchedMatMul(mkResident(ref, []int{out, in}, w), mkResident(ref, []int{P, in}, X), P))
	wW, sW := uploadAWQResident(cb, packed, scales, out, in)
	YCu := cb.Read(cb.AWQBatchedMatMul(wW, sW, mkResident(cb, []int{P, in}, X), P))
	if len(YRef) != P*out || len(YCu) != P*out {
		t.Fatalf("shape ref=%d cu=%d want %d", len(YRef), len(YCu), P*out)
	}
	c := cosine(YRef, YCu)
	if c < cudaAWQCosineMin {
		t.Fatalf("AWQ BatchedMatMul cosine %.6f < recorded AWQ gate %.6f", c, cudaAWQCosineMin)
	}
	t.Logf("#926 AWQ BatchedMatMul (P=%d): cosine=%.8f maxAbs=%.2e gate=%.4f", P, c, maxAbsDelta(YRef, YCu), cudaAWQCosineMin)
}

// BenchmarkCUDAAWQBatchedMatMul — the prefill GEMM with a resident AWQ 4-bit weight (dequant
// fused into the tile), same dims as the F32/F16/Q8/Q4_K baselines (fp16BenchDims). The
// acceptance script turns this ns/op into the AWQ-vs-F32 GEMM throughput delta.
func BenchmarkCUDAAWQBatchedMatMul(b *testing.B) {
	cb := cudaTBOrSkip(b)
	out, in, P := fp16BenchDims()
	var seed lcg = 926
	g := &seed
	packed := randAWQBytes(g, out, in)
	scales := randAWQScales(g, out)
	wW, sW := uploadAWQResident(cb, packed, scales, out, in)
	X := mkResident(cb, []int{P, in}, rscale(g, P*in, 1.0))
	cb.Read(cb.AWQBatchedMatMul(wW, sW, X, P)) // warm: pool + weight cache
	cb.Recycle()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cb.Read(cb.AWQBatchedMatMul(wW, sW, X, P))
		cb.Recycle()
	}
	b.StopTimer()
}

//go:build cuda

package compute

import (
	"math"
	"testing"
)

// cuda_fp16_test.go — the `-tags cuda` witness for issue #484 (fp16 compute path: cuBLAS
// tensor-core HGEMM with weights narrowed to F16 at H2D under Caps.UploadDtype, plus the
// `Layout` repack at H2D). The CUDA backend ran F32 SGEMM only; fp16/tensor-cores is the
// precision axis toward llama.cpp throughput (internal/model/bench_llamacpp.py measures F16).
//
// The witness runs the SAME GEMM two ways and holds the device fp16 path to the cuda backend's
// Approx gate against the cpuref f32 Reference — NOT bit-identity (RequireReference keeps the
// device off the exact rungs):
//   - REFERENCE — cpuref f32 MatMul/BatchedMatMul (the fdot reduction);
//   - DEVICE fp16 — the weight uploaded AS F16 (Upload(t, F16) narrows at H2D), HGEMM on tensor
//     cores (cublasGemmEx, CUDA_R_16F inputs, F32 accumulate), f32 output.
// It asserts the GEMM cosine clears the RECORDED fp16 floor cudaFP16CosineMin (looser than the
// Q8 lane's 0.999 — see the constant's comment for WHY), for BOTH weight layouts: RowMajor (op_T)
// and the ColMajor transpose-repack (op_N). Same logical Y from both layouts is the `Layout`-
// repack-at-H2D witness.
//
// HARDWARE: the actual HGEMM RUN, the realized cosine verdict at cudaFP16CosineMin, and the
// fp16-vs-f32 (and fp16-vs-llama.cpp-F16) tok/s need a CUDA node — they are the explicit residual
// of this build+commit handoff. On the win32 dev host (no CUDA toolkit) these skip cleanly; run
// them on a GPU node via tools/run_484_acceptance_on_gpu.sh. The Go+cgo here type-checks under
// `go vet -tags cuda`.

// uploadF16Layout stages an f32 host weight [out,in] as a resident F16 weight in the requested
// layout — the witness exercises both RowMajor and the ColMajor transpose-repack through the SAME
// Upload(t, F16) seam (only t.Layout differs).
func uploadF16Layout(cb *cudaBackend, out, in int, w []float32, layout Layout) Tensor {
	src := NewF32(cb, []int{out, in}, w)
	src.Layout = layout
	return cb.Upload(src, F16)
}

func maxAbsDelta(a, b []float32) float64 {
	var m float64
	for i := range a {
		if d := math.Abs(float64(a[i] - b[i])); d > m {
			m = d
		}
	}
	return m
}

// TestCUDAMatMulF16ApproxMatchesRef — op-level fp16 first light: a single device HGEMM (decode
// GEMV, P=1) with an F16-resident weight vs the cpuref f32 fdot matmul on the same random W,x,
// for both weight layouts. Approx (fp16 rounds both operands; cuBLAS reduction order differs),
// so the gate is the RECORDED fp16 cosine floor, not equality.
func TestCUDAMatMulF16ApproxMatchesRef(t *testing.T) {
	cb := cudaOrSkip(t)
	if !cb.Caps().UploadDtype {
		t.Fatal("#484: cuda backend must advertise Caps.UploadDtype=true (F16 weight narrowing at H2D)")
	}
	ref := Default() // cpu-ref
	var seed lcg = 484
	g := &seed
	out, in := 320, 256 // multiples of 16 — tensor-core friendly
	w := rscale(g, out*in, 0.2)
	x := rscale(g, in, 1.0)

	yRef := ref.Read(ref.MatMul(mkResident(ref, []int{out, in}, w), mkResident(ref, []int{in}, x)))

	for _, lay := range []struct {
		name   string
		layout Layout
	}{{"rowmajor", RowMajor}, {"colmajor", ColMajor}} {
		wF16 := uploadF16Layout(cb, out, in, w, lay.layout)
		if wF16.Dtype != F16 {
			t.Fatalf("%s: Upload(_, F16) produced Dtype %s, want f16", lay.name, wF16.Dtype)
		}
		yCu := cb.Read(cb.MatMul(wF16, mkResident(cb, []int{in}, x)))
		if len(yRef) != out || len(yCu) != out {
			t.Fatalf("%s: shape ref=%d cu=%d want %d", lay.name, len(yRef), len(yCu), out)
		}
		c := cosine(yRef, yCu)
		if c < cudaFP16CosineMin {
			t.Fatalf("%s: fp16 MatMul cosine %.6f < recorded fp16 gate %.6f (cudaFP16CosineMin)", lay.name, c, cudaFP16CosineMin)
		}
		t.Logf("#484 fp16 MatMul (%s): cosine=%.8f maxAbs=%.2e gate=%.4f (device=%s tier=%s class=%s)",
			lay.name, c, maxAbsDelta(yRef, yCu), cudaFP16CosineMin, cb.Name(), cb.Tier(), cb.Class())
	}
}

// TestCUDABatchedMatMulF16ApproxMatchesRef — the prefill GEMM (P>1), where fp16/tensor-cores pay
// off most: an F16-resident weight vs the cpuref f32 BatchedMatMul, both layouts, held to the
// recorded fp16 cosine floor.
func TestCUDABatchedMatMulF16ApproxMatchesRef(t *testing.T) {
	cb := cudaOrSkip(t)
	if !cb.Caps().UploadDtype {
		t.Fatal("#484: cuda backend must advertise Caps.UploadDtype=true")
	}
	ref := Default()
	var seed lcg = 0x484b
	g := &seed
	out, in, P := 320, 256, 8
	w := rscale(g, out*in, 0.2)
	X := rscale(g, P*in, 1.0)

	YRef := ref.Read(ref.BatchedMatMul(mkResident(ref, []int{out, in}, w), mkResident(ref, []int{P, in}, X), P))

	for _, lay := range []struct {
		name   string
		layout Layout
	}{{"rowmajor", RowMajor}, {"colmajor", ColMajor}} {
		wF16 := uploadF16Layout(cb, out, in, w, lay.layout)
		YCu := cb.Read(cb.BatchedMatMul(wF16, mkResident(cb, []int{P, in}, X), P))
		if len(YRef) != P*out || len(YCu) != P*out {
			t.Fatalf("%s: shape ref=%d cu=%d want %d", lay.name, len(YRef), len(YCu), P*out)
		}
		c := cosine(YRef, YCu)
		if c < cudaFP16CosineMin {
			t.Fatalf("%s: fp16 BatchedMatMul cosine %.6f < recorded fp16 gate %.6f", lay.name, c, cudaFP16CosineMin)
		}
		t.Logf("#484 fp16 BatchedMatMul (%s, P=%d): cosine=%.8f maxAbs=%.2e gate=%.4f",
			lay.name, P, c, maxAbsDelta(YRef, YCu), cudaFP16CosineMin)
	}
	t.Logf("#484 fp16 witness: throughput vs the F32 device path is BenchmarkCUDABatchedMatMul{F16,F32}; "+
		"compare the fp16 GEMM tok/s against the F16 cell of internal/model/bench_llamacpp.py (run on a CUDA node via tools/run_484_acceptance_on_gpu.sh). device=%s tier=%s class=%s",
		cb.Name(), cb.Tier(), cb.Class())
}

// fp16BenchDims — a transformer-ish GEMM (square-ish) at a prefill tile P, where tensor cores
// matter; the F16-vs-F32 ns/op delta is what tools/run_484_acceptance_on_gpu.sh turns into the
// fp16-vs-f32 GEMM throughput ratio and compares against bench_llamacpp.py's F16 number.
func fp16BenchDims() (out, in, P int) { return 4096, 4096, 512 }

// BenchmarkCUDABatchedMatMulF32 — the F32 SGEMM device path (the baseline the fp16 path beats).
func BenchmarkCUDABatchedMatMulF32(b *testing.B) {
	cb := cudaTBOrSkip(b)
	out, in, P := fp16BenchDims()
	var seed lcg = 484
	g := &seed
	w := cb.Upload(NewF32(cb, []int{out, in}, rscale(g, out*in, 0.05)), F32)
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

// BenchmarkCUDABatchedMatMulF16 — the SAME GEMM with an F16-resident weight on tensor-core HGEMM
// (#484). The F16-vs-F32 ns/op delta isolates the tensor-core win (the per-iter Read copies the
// same f32 output bytes in both, so it cancels).
func BenchmarkCUDABatchedMatMulF16(b *testing.B) {
	cb := cudaTBOrSkip(b)
	out, in, P := fp16BenchDims()
	var seed lcg = 484
	g := &seed
	w := cb.Upload(NewF32(cb, []int{out, in}, rscale(g, out*in, 0.05)), F16) // F16 weight
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

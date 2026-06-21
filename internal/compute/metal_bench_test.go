//go:build darwin && metal && cgo

package compute

import "testing"

// metal_bench_test.go — throughput characterization for the Metal backend (issue #300, the
// open "within 2× llama.cpp Metal throughput" lane). GEMM is the decode/prefill bottleneck
// (see modelprof: mlp+qkv+o+head are ~94% of decode time, all memory-bound matmuls), so a
// single-op MatMul probe is the honest first measurement of the device path's wall-clock.
//
// This is a SYNTHETIC single-op probe (one MPSMatrixMultiplication per call vs one cpuref
// fdot matmul), NOT a llama.cpp head-to-head and NOT an end-to-end model number — the model
// engine does not yet route through compute.Backend("metal"). The baseline is cpuref, the
// correctness *reference* (scalar fdot), NOT the optimized NEON/SDOT path the real CPU decode
// uses — so a speedup here is "vs the reference backend", not "vs production CPU".
//
// Measured on an Apple M3 Pro (2026-06-20, machine under concurrent load):
//   GEMM (P=64, prefill): Metal ~40–107× cpuref, ROBUST every run — the GPU has enough work
//     per dispatch to bury the command-buffer commit+wait, and cpuref's scalar GEMM is slow.
//   GEMV (P=1, decode):  HIGHLY VARIABLE, ~0.35–2.0 ms/op = parity-to-~3.9× vs cpuref across
//     runs. The first-light design commits+waits a command buffer PER op, so a small GEMV is
//     dispatch-latency-bound (47 GB/s « the M3 Pro's ~150 GB/s peak) and the latency floats
//     with GPU power state + scheduler contention. This is the concrete motivation for the
//     batched/single-command-buffer-per-forward follow-up; decode throughput is gated on
//     dispatch amortization, not on the matmul itself.
// Run with:
//   CGO_ENABLED=1 go test -tags metal -run '^$' -bench MatMul -benchmem ./internal/compute/
// SetBytes is the weight footprint (out*in*4) so the report reads as weight-GB/s, comparable
// to q8kernel's cross-ISA GEMV bandwidth column. Pin the machine (close other GPU consumers)
// for a stable GEMV number.

const (
	benchOut = 2048
	benchIn  = 2048
)

// benchMatMul times b.N GEMV (P==1) or GEMM (P>1) ops on the given backend at [out,in],
// freeing each output so device residency stays flat across iterations.
func benchMatMul(b *testing.B, be Backend, P int) {
	var seed lcg = 0xBEEF
	g := &seed
	w := mtlMkResident(be, []int{benchOut, benchIn}, mtlRscale(g, benchOut*benchIn, 0.1))
	defer be.Free(w)
	var x Tensor
	if P == 1 {
		x = mtlMkResident(be, []int{benchIn}, mtlRscale(g, benchIn, 1.0))
	} else {
		x = mtlMkResident(be, []int{P, benchIn}, mtlRscale(g, P*benchIn, 1.0))
	}
	defer be.Free(x)
	b.SetBytes(int64(benchOut * benchIn * 4)) // weight bytes streamed per op
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var y Tensor
		if P == 1 {
			y = be.MatMul(w, x)
		} else {
			y = be.BatchedMatMul(w, x, P)
		}
		be.Free(y)
	}
}

func BenchmarkMetalMatMulGEMV(b *testing.B) {
	be := Pick("metal")
	if _, ok := be.(*metalBackend); !ok {
		b.Skip("metal backend not registered (no reachable Metal device)")
	}
	benchMatMul(b, be, 1)
}

func BenchmarkCPURefMatMulGEMV(b *testing.B) { benchMatMul(b, Default(), 1) }

func BenchmarkMetalMatMulGEMM64(b *testing.B) {
	be := Pick("metal")
	if _, ok := be.(*metalBackend); !ok {
		b.Skip("metal backend not registered (no reachable Metal device)")
	}
	benchMatMul(b, be, 64)
}

func BenchmarkCPURefMatMulGEMM64(b *testing.B) { benchMatMul(b, Default(), 64) }

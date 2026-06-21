// Command q8kernel is a self-contained kernel microbenchmark that isolates ONE question
// from all model/WSL/load overhead: in pure Go on this box, which GEMV kernel is fastest
// for the memory-bound batch-1 decode regime —
//
//	f32        : y[o] = Σ w[o,i]·x[i]            (the current f32 path; 4 B/weight, vectorized FMA)
//	int8×int8  : y[o] = Σ blocks dw·dx·Σ(qw·qx)  (the shipped Q8_0 path; 1 B/weight, scalar int dot)
//	int8×f32   : y[o] = s[o]·Σ float32(qw[i])·x[i] (weight-only; 1 B/weight, f32 mul, NO activation quant)
//
// It streams a head-sized weight matrix (49152×576 ≈ 113 MB f32 / 28 MB int8 — far past L3,
// so it is genuinely DRAM-bandwidth-bound like real decode), row-parallel across all cores,
// and reports ms/matmul and effective weight-GB/s. This answers whether descending to int8
// can beat f32 in pure Go WITHOUT touching the model package (no collision), and whether the
// bandwidth win (4× fewer bytes) survives the per-element conversion cost.
package main

import (
	"flag"
	"fmt"
	"os"
	"runtime"
	"sort"
	"sync"
	"time"
)

const qBlk = 32

func parFor(n, workers int, body func(lo, hi int)) {
	if workers <= 1 || n <= 1 {
		body(0, n)
		return
	}
	if workers > n {
		workers = n
	}
	chunk := (n + workers - 1) / workers
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		lo := w * chunk
		if lo >= n {
			break
		}
		hi := lo + chunk
		if hi > n {
			hi = n
		}
		wg.Add(1)
		go func(lo, hi int) { defer wg.Done(); body(lo, hi) }(lo, hi)
	}
	wg.Wait()
}

// --- f32 (the current path) ---
func fdot(r, x []float32) float32 {
	var s0, s1, s2, s3, s4, s5, s6, s7 float32
	n := len(r)
	i := 0
	for ; i+8 <= n; i += 8 {
		s0 += r[i] * x[i]
		s1 += r[i+1] * x[i+1]
		s2 += r[i+2] * x[i+2]
		s3 += r[i+3] * x[i+3]
		s4 += r[i+4] * x[i+4]
		s5 += r[i+5] * x[i+5]
		s6 += r[i+6] * x[i+6]
		s7 += r[i+7] * x[i+7]
	}
	s := ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7))
	for ; i < n; i++ {
		s += r[i] * x[i]
	}
	return s
}

// --- int8×int8 (the shipped Q8_0 qdot8) ---
func qdot8(qw []int8, dw []float32, qx []int8, dx []float32, nblk int) float32 {
	var acc float32
	for b := 0; b < nblk; b++ {
		wb := qw[b*qBlk:]
		xb := qx[b*qBlk:]
		var s0, s1, s2, s3 int32
		for i := 0; i < qBlk; i += 4 {
			s0 += int32(wb[i]) * int32(xb[i])
			s1 += int32(wb[i+1]) * int32(xb[i+1])
			s2 += int32(wb[i+2]) * int32(xb[i+2])
			s3 += int32(wb[i+3]) * int32(xb[i+3])
		}
		acc += float32((s0+s1)+(s2+s3)) * dw[b] * dx[b]
	}
	return acc
}

// --- int8×f32 (weight-only: 8-accumulator, int8 widened on the fly, NO activation quant) ---
func qdotWO(q []int8, x []float32) float32 {
	var s0, s1, s2, s3, s4, s5, s6, s7 float32
	n := len(q)
	i := 0
	for ; i+8 <= n; i += 8 {
		s0 += float32(q[i]) * x[i]
		s1 += float32(q[i+1]) * x[i+1]
		s2 += float32(q[i+2]) * x[i+2]
		s3 += float32(q[i+3]) * x[i+3]
		s4 += float32(q[i+4]) * x[i+4]
		s5 += float32(q[i+5]) * x[i+5]
		s6 += float32(q[i+6]) * x[i+6]
		s7 += float32(q[i+7]) * x[i+7]
	}
	s := ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7))
	for ; i < n; i++ {
		s += float32(q[i]) * x[i]
	}
	return s
}

func medMS(ds []time.Duration) float64 {
	cp := append([]time.Duration(nil), ds...)
	sort.Slice(cp, func(i, j int) bool { return cp[i] < cp[j] })
	return float64(cp[len(cp)/2].Nanoseconds()) / 1e6
}

func main() {
	out := flag.Int("out", 49152, "output rows (49152 = vocab/head; the biggest single matmul)")
	in := flag.Int("in", 576, "inner dim (hidden size)")
	reps := flag.Int("reps", 15, "reps per kernel (median)")
	flag.Parse()
	W := runtime.GOMAXPROCS(0)
	O, I := *out, *in
	nblk := I / qBlk

	// f32 weights + activation
	wf := make([]float32, O*I)
	x := make([]float32, I)
	for i := range wf {
		wf[i] = float32((i*1103515245+12345)&0xffff)/32768 - 1
	}
	for i := range x {
		x[i] = float32((i*48271+7)&0xffff)/32768 - 1
	}
	// int8 weights + per-block scales (values irrelevant to timing)
	qw := make([]int8, O*I)
	dw := make([]float32, O*nblk)
	for i := range qw {
		qw[i] = int8((i % 255) - 127)
	}
	for i := range dw {
		dw[i] = 0.01
	}
	// pre-quantized activation for int8×int8
	qx := make([]int8, I)
	dx := make([]float32, nblk)
	for i := range qx {
		qx[i] = int8((i % 255) - 127)
	}
	for i := range dx {
		dx[i] = 0.02
	}

	yf := make([]float32, O)
	bench := func(name string, body func(lo, hi int)) (float64, float64) {
		// warm
		parFor(O, W, body)
		ds := make([]time.Duration, *reps)
		for r := 0; r < *reps; r++ {
			t := time.Now()
			parFor(O, W, body)
			ds[r] = time.Since(t)
		}
		ms := medMS(ds)
		return ms, ms
	}

	f32Body := func(lo, hi int) {
		for o := lo; o < hi; o++ {
			yf[o] = fdot(wf[o*I:o*I+I], x)
		}
	}
	i8i8Body := func(lo, hi int) {
		for o := lo; o < hi; o++ {
			yf[o] = qdot8(qw[o*I:o*I+I], dw[o*nblk:o*nblk+nblk], qx, dx, nblk)
		}
	}
	woBody := func(lo, hi int) {
		for o := lo; o < hi; o++ {
			yf[o] = dw[o*nblk] * qdotWO(qw[o*I:o*I+I], x)
		}
	}

	f32ms, _ := bench("f32", f32Body)
	i8ms, _ := bench("int8x int8", i8i8Body)
	woms, _ := bench("int8x f32", woBody)

	f32bytes := float64(O*I*4) / 1e9
	i8bytes := float64(O*I+O*nblk*4) / 1e9 // codes + per-block scales
	wobytes := float64(O*I+O*4) / 1e9      // codes + per-row scale (we use dw[o*nblk] as a stand-in)

	fmt.Fprintf(os.Stderr, "q8kernel: GOMAXPROCS=%d  matmul [%d x %d]  reps=%d\n", W, O, I, *reps)
	row := func(name string, ms, gb float64) {
		fmt.Fprintf(os.Stderr, "  %-12s %7.2f ms   %6.1f weight-GB/s   %.2fx vs f32\n", name, ms, gb/(ms/1e3), ms/f32ms)
	}
	row("f32", f32ms, f32bytes)
	row("int8xint8", i8ms, i8bytes)
	row("int8xf32(WO)", woms, wobytes)
	fmt.Fprintf(os.Stderr, "verdict: weight-only int8xf32 is %.2fx f32 (<1.0 = FASTER); int8xint8 is %.2fx f32\n",
		woms/f32ms, i8ms/f32ms)
}

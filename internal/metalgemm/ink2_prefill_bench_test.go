//go:build darwin && arm64 && cgo

package metalgemm

import (
	"math/rand"
	"os"
	"strconv"
	"testing"
)

// ink2PBenchGPU returns the GPU seconds of the launches encode adds, after one
// untimed pass that compiles the kernels and faults the operands in.
func ink2PBenchGPU(t *testing.T, d *Ink2Device, encode func() error) float64 {
	t.Helper()
	best := 0.0
	for pass := range 3 {
		if err := d.Flush(); err != nil {
			t.Fatal(err)
		}
		before := d.Stats().GPUSeconds
		if err := encode(); err != nil {
			t.Fatal(err)
		}
		if err := d.Flush(); err != nil {
			t.Fatal(err)
		}
		if gpu := d.Stats().GPUSeconds - before; pass == 1 || (pass > 1 && gpu < best) {
			best = gpu
		}
	}
	return best
}

// The batched weight kernels against the decode kernels they replace, at the
// shapes of an Inkling-2-Mini chunk: one dense projection over T rows, and
// T*k routed pairs through a 128-expert stack. FAK_INK2P_BENCH=<T> runs it.
// fak-test:runtime slow est=60s
func TestInk2PBenchBatchedWeightKernels(t *testing.T) {
	T, _ := strconv.Atoi(os.Getenv("FAK_INK2P_BENCH"))
	if T < 1 {
		t.Skip("set FAK_INK2P_BENCH to a chunk length to time the batched weight kernels")
	}
	d := ink2TestDevice(t)
	rng := rand.New(rand.NewSource(90))
	const hidden, k, experts = 2304, 8, 128
	page := os.Getpagesize()
	for _, c := range []struct {
		name string
		typ  uint32
	}{{"q8_0", 8}, {"nvfp4", 40}, {"iq3_xxs", 18}, {"q3_K", 11}} {
		t.Run(c.name, func(t *testing.T) {
			block, blockBytes, err := packedBlock(c.typ)
			if err != nil {
				t.Fatal(err)
			}
			const denseRows, expertRows = 3072, 2304
			bytes := (denseRows + experts*expertRows) * (hidden / block * blockBytes)
			store, data := ink2TestStore(t, bytes/page+8)
			dense, next := ink2PTestWeights(t, store, data, rng, 64, c.typ, denseRows, hidden, 1)
			stackM, _ := ink2PTestWeights(t, store, data, rng, next, c.typ, expertRows, hidden, experts)
			w, err := dense.stack.Ink2Weight(0)
			if err != nil {
				t.Fatal(err)
			}
			stack, ok := stackM.stack.Ink2Stack()
			if !ok {
				t.Fatal("stack split across views")
			}
			x := ink2OpsTestRand(rng, T*hidden, .5)
			xr := ink2OpsTestPut(t, d, 0, x)

			serial, batched := ink2TestAlloc(t, d, T*denseRows), ink2TestAlloc(t, d, T*denseRows)
			perToken := ink2PBenchGPU(t, d, func() error {
				for tok := range T {
					if err := d.MatVec(w, xr.At(tok*hidden), serial.At(tok*denseRows)); err != nil {
						return err
					}
				}
				return nil
			})
			rows := ink2PBenchGPU(t, d, func() error { return d.RowsMatMul(w, xr, hidden, batched, denseRows, T) })
			ink2OpsTestSame(t, "dense", ink2OpsTestRead(t, d, batched, T*denseRows), ink2OpsTestRead(t, d, serial, T*denseRows))
			gemm := ink2PBenchGPU(t, d, func() error {
				ok, err := d.RowsGEMM(w, xr, hidden, batched, denseRows, T)
				if err == nil && !ok {
					t.Fatal("rows GEMM declined")
				}
				return err
			})
			elems := float64(T) * denseRows * hidden
			t.Logf("dense [%d][%d] over %d rows: decode GEMV %.4fs (%.1f G weight-rows/s), rows kernel %.4fs (%.1f G/s), %.2fx, GEMM %.4fs (%.1f G/s), %.2fx",
				denseRows, hidden, T, perToken, elems/perToken/1e9, rows, elems/rows/1e9, perToken/rows, gemm, elems/gemm/1e9, perToken/gemm)

			// Each token routes k distinct experts.
			npairs := T * k
			ids := make([]int32, 0, npairs)
			for range T {
				for _, e := range rng.Perm(experts)[:k] {
					ids = append(ids, int32(e))
				}
			}
			perPair := make([]float32, npairs*hidden)
			for i := range npairs {
				copy(perPair[i*hidden:(i+1)*hidden], x[(i/k)*hidden:])
			}
			idr, pr := ink2OpsTestPut(t, d, 0, ink2PTestInts(ids)), ink2OpsTestPut(t, d, 0, perPair)
			counts, offsets, order := ink2TestAlloc(t, d, experts), ink2TestAlloc(t, d, experts+1), ink2TestAlloc(t, d, npairs)
			ys, yb := ink2TestAlloc(t, d, npairs*expertRows), ink2TestAlloc(t, d, npairs*expertRows)
			routed := ink2PBenchGPU(t, d, func() error {
				ok, err := d.ExpertMatVec(stack, idr, pr, hidden, ys, npairs, Ink2Ref{}, nil)
				if err == nil && !ok {
					t.Fatal("decode expert matvec declined")
				}
				return err
			})
			grouped := ink2PBenchGPU(t, d, func() error {
				if err := d.PBucket(idr, counts, offsets, order, npairs, experts); err != nil {
					return err
				}
				return d.PExpertGrouped(stack, idr, order, xr, k, hidden, yb, npairs, Ink2Ref{})
			})
			ink2OpsTestSame(t, "experts", ink2OpsTestRead(t, d, yb, npairs*expertRows), ink2OpsTestRead(t, d, ys, npairs*expertRows))
			groupedGemm := ink2PBenchGPU(t, d, func() error {
				ok, err := d.PExpertGEMM(stack, idr, order, offsets, xr, k, hidden, yb, npairs, Ink2Ref{})
				if err == nil && !ok {
					t.Fatal("grouped GEMM declined")
				}
				return err
			})
			elems = float64(npairs) * expertRows * hidden
			t.Logf("%d pairs through %d experts of [%d][%d]: decode expert GEMV %.4fs (%.1f G/s), grouped kernel %.4fs (%.1f G/s), %.2fx, GEMM %.4fs (%.1f G/s), %.2fx",
				npairs, experts, expertRows, hidden, routed, elems/routed/1e9, grouped, elems/grouped/1e9, routed/grouped,
				groupedGemm, elems/groupedGemm/1e9, routed/groupedGemm)
		})
	}
}

// The bucketing and conv kernels of a chunk against the ones they replace at
// Inkling-2-Mini shapes. FAK_INK2P_BENCH=<T> runs it.
// fak-test:runtime slow est=10s
func TestInk2PBenchBucketAndConv(t *testing.T) {
	T, _ := strconv.Atoi(os.Getenv("FAK_INK2P_BENCH"))
	if T < 32 {
		t.Skip("set FAK_INK2P_BENCH to a chunk length of 32+ to time the bucket and conv kernels")
	}
	d := ink2TestDevice(t)
	rng := rand.New(rand.NewSource(91))
	const hidden, k, experts, kn = 2304, 8, 128, 4
	npairs := T * k
	ids := make([]int32, 0, npairs)
	for range T {
		for _, e := range rng.Perm(experts)[:k] {
			ids = append(ids, int32(e))
		}
	}
	idr := ink2OpsTestPut(t, d, 0, ink2PTestInts(ids))
	counts, offsets, order := ink2TestAlloc(t, d, experts), ink2TestAlloc(t, d, experts+1), ink2TestAlloc(t, d, npairs)
	// Ten of each in one submission, so the submission's own cost stays out of the time.
	const reps = 10
	repeat := func(f func() error) func() error {
		return func() error {
			for range reps {
				if err := f(); err != nil {
					return err
				}
			}
			return nil
		}
	}
	pair := ink2PBenchGPU(t, d, repeat(func() error { return d.PBucket(idr, counts, offsets, order, npairs, experts) })) / reps
	one := ink2PBenchGPU(t, d, repeat(func() error { return d.PBucketSort(idr, counts, offsets, order, npairs, experts) })) / reps
	t.Logf("bucket of %d pairs over %d experts: scan+fill %.1fus, one launch %.1fus", npairs, experts, 1e6*pair, 1e6*one)

	x, w := ink2OpsTestPut(t, d, 0, ink2OpsTestRand(rng, T*hidden, 1)), ink2OpsTestPut(t, d, 0, ink2OpsTestRand(rng, hidden*kn, .5))
	hist, out := ink2TestAlloc(t, d, hidden*(kn-1)), ink2TestAlloc(t, d, T*hidden)
	norm := float32(0.4)
	serial := ink2PBenchGPU(t, d, repeat(func() error { return d.PConv(x, w, Ink2Ref{}, hist, out, Ink2Ref{}, hidden, 1, kn, norm, T) })) / reps
	rows := ink2PBenchGPU(t, d, repeat(func() error { return d.PConvRows(x, w, Ink2Ref{}, hist, out, Ink2Ref{}, hidden, 1, kn, norm, T) })) / reps
	t.Logf("conv of %d rows of %d channels: channel walk %.1fus, thread per row %.1fus", T, hidden, 1e6*serial, 1e6*rows)
}

// The controller stages of a chunk. FAK_INK2P_BENCH=<T> runs it.
// fak-test:runtime slow est=10s
func TestInk2PBenchController(t *testing.T) {
	T, _ := strconv.Atoi(os.Getenv("FAK_INK2P_BENCH"))
	if T < 32 {
		t.Skip("set FAK_INK2P_BENCH to a chunk length of 32+ to time the controller")
	}
	d := ink2TestDevice(t)
	rng := rand.New(rand.NewSource(92))
	const n, rank, reps = 2304, 64, 10
	put := func(size int) Ink2Ref { return ink2OpsTestPut(t, d, 0, ink2OpsTestRand(rng, size, .5)) }
	cur, msg, down, gain, forget := put(T*n), put(T*n), put(rank*2*n), put(n*rank), put(n*rank)
	g, out, scratch := ink2TestAlloc(t, d, T*rank), ink2TestAlloc(t, d, T*n), ink2TestAlloc(t, d, 2*T*n)
	time := func(f func() error) float64 {
		return ink2PBenchGPU(t, d, func() error {
			for range reps {
				if err := f(); err != nil {
					return err
				}
			}
			return nil
		}) / reps
	}
	downGEMM := time(func() error { return d.PCtrlDownGEMM(cur, msg, down, g, scratch, n, rank, T) })
	applyGEMM := time(func() error { return d.PCtrlApplyGEMM(cur, msg, g, gain, forget, out, scratch, n, rank, T) })
	downRows := time(func() error {
		return d.PCtrlDown([]Ink2Ref{cur}, []Ink2Ref{msg}, []Ink2Ref{down}, []Ink2Ref{g}, n, rank, T)
	})
	applyRows := time(func() error {
		return d.PCtrlApply([]Ink2Ref{cur}, []Ink2Ref{msg}, []Ink2Ref{g}, []Ink2Ref{gain}, []Ink2Ref{forget}, []Ink2Ref{out}, n, rank, T)
	})
	part := ink2TestAlloc(t, d, CtrlDownSplitScratch(n, rank, T))
	downTile := time(func() error {
		ok, err := d.PCtrlDownSplit(cur, msg, down, g, part, n, rank, T)
		if err == nil && !ok {
			t.Fatal("split controller down declined")
		}
		return err
	})
	applyTile := time(func() error {
		ok, err := d.PCtrlApplyTile(cur, msg, g, gain, forget, out, n, rank, T)
		if err == nil && !ok {
			t.Fatal("controller apply tile declined")
		}
		return err
	})
	t.Logf("controller of %d rows, width %d, rank %d: down GEMM %.1fus, split %.1fus, decode-order %.1fus; apply GEMM %.1fus, tile %.1fus, decode-order %.1fus",
		T, n, rank, 1e6*downGEMM, 1e6*downTile, 1e6*downRows, 1e6*applyGEMM, 1e6*applyTile, 1e6*applyRows)
}

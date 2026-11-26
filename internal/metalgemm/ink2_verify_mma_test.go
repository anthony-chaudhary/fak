//go:build darwin && arm64 && cgo

package metalgemm

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gpulease"
)

// The MMA kernels sum in another order than the decode GEMV: each row must
// stay within this relative L2 distance of it.
const ink2VMMATestRel = 1e-5

type ink2VMMATestType struct {
	name string
	typ  uint32
}

var ink2VMMATestTypes = []ink2VMMATestType{{"f32", 0}, {"f16", 1}, {"bf16", 30}, {"q8_0", 8}, {"q3_K", 11}, {"q4_K", 12},
	{"q5_K", 13}, {"q6_K", 14}, {"iq2_xs", 17}, {"iq3_xxs", 18}, {"iq4_nl", 20}, {"iq3_s", 21}, {"iq2_s", 22}, {"nvfp4", 40}}

// ink2VMMATestCols is a width of 64-column groups the 32 chains share
// unevenly.
func ink2VMMATestCols(t *testing.T, typ uint32) int {
	t.Helper()
	block, _, err := packedBlock(typ)
	if err != nil {
		t.Fatal(err)
	}
	switch block {
	case 256:
		return 768
	case 64:
		return 576
	}
	return 2112
}

// ink2VMMATestPath pins the widest pass the kernels run on scalar FMAs for the
// rest of the test.
func ink2VMMATestPath(t *testing.T, scalarMax int) {
	t.Helper()
	saved := ink2VerifyMMAScalarMax
	t.Cleanup(func() { ink2VerifyMMAScalarMax = saved })
	ink2VerifyMMAScalarMax = func() int { return scalarMax }
}

func ink2VMMATestNear(t *testing.T, what string, got, want []float32) {
	t.Helper()
	var num, den float64
	for i := range want {
		d := float64(got[i]) - float64(want[i])
		num += d * d
		den += float64(want[i]) * float64(want[i])
	}
	if rel := math.Sqrt(num / max(den, 1e-30)); rel > ink2VMMATestRel || math.IsNaN(rel) {
		t.Fatalf("%s: relative L2 %.3g from the decode GEMV, bound %g", what, rel, ink2VMMATestRel)
	}
}

// fak-test:runtime integration est=10s lane=default
func TestInk2VMMARowsAreWidthInvariantAndNearMatVec(t *testing.T) {
	d := ink2TestDevice(t)
	store, data := ink2TestStore(t, 64)
	rng := rand.New(rand.NewSource(91))
	const rows, maxT = 13, 19
	for _, c := range ink2VMMATestTypes {
		t.Run(c.name, func(t *testing.T) {
			cols := ink2VMMATestCols(t, c.typ)
			xStride, yStride := cols+64, rows+3
			x := ink2OpsTestRand(rng, maxT*xStride, 1)
			xr := ink2VTestPut(t, d, x)
			w, _ := ink2VTestWeight(t, store, data, rng, 64, c.typ, rows, cols)
			decode, alone := ink2TestAlloc(t, d, maxT*rows), ink2TestAlloc(t, d, maxT*rows)
			// alone on the scalar path, then every width on the MMA path and the
			// narrow ones on the scalar path too
			ink2VMMATestPath(t, 4)
			for r := range maxT {
				if err := d.MatVec(w, xr.At(r*xStride), decode.At(r*rows)); err != nil {
					t.Fatal(err)
				}
				if ok, err := d.VerifyRowsMMA(w, xr.At(r*xStride), xStride, alone.At(r*rows), rows, 1); err != nil || !ok {
					t.Fatalf("row %d alone: ok=%v err=%v", r, ok, err)
				}
			}
			gemv := ink2OpsTestRead(t, d, decode, maxT*rows)
			ink2VTestFinite(t, "decode", gemv)
			want := ink2OpsTestRead(t, d, alone, maxT*rows)
			for r := range maxT {
				ink2VMMATestNear(t, fmt.Sprintf("row %d alone", r), want[r*rows:(r+1)*rows], gemv[r*rows:(r+1)*rows])
			}
			for i, T := range []int{1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, maxT} {
				path := "scalar"
				if i == 4 {
					ink2VMMATestPath(t, 0)
				}
				if i >= 4 {
					path = "mma"
				}
				y := ink2VTestPut(t, d, make([]float32, T*yStride))
				if ok, err := d.VerifyRowsMMA(w, xr, xStride, y, yStride, T); err != nil || !ok {
					t.Fatalf("width %d: ok=%v err=%v", T, ok, err)
				}
				got := ink2OpsTestRead(t, d, y, T*yStride)
				for r := range T {
					ink2OpsTestSame(t, fmt.Sprintf("%s width %d row %d", path, T, r), got[r*yStride:r*yStride+rows], want[r*rows:(r+1)*rows])
					for i := rows; i < yStride; i++ {
						if got[r*yStride+i] != 0 {
							t.Fatalf("width %d row %d wrote past its %d outputs", T, r, rows)
						}
					}
				}
			}
		})
	}
	odd, _ := ink2VTestWeight(t, store, data, rng, 64, 0, 2, 20)
	if ok, err := d.VerifyRowsMMA(odd, ink2TestAlloc(t, d, 64), 20, ink2TestAlloc(t, d, 8), 2, 1); ok || err != nil {
		t.Fatalf("20 columns: ok=%v err=%v, want a decline", ok, err)
	}
}

// fak-test:runtime integration est=5s lane=default
func TestInk2VMMAExpertsAreWidthInvariantAndNearExpertMatVec(t *testing.T) {
	d := ink2TestDevice(t)
	store, data := ink2TestStore(t, 256)
	rng := rand.New(rand.NewSource(93))
	const rows, nexp, k, maxT = 13, 6, 3, 19
	scale := make([]float32, nexp)
	for i := range scale {
		scale[i] = 0.5 + rng.Float32()
	}
	sr := ink2VTestPut(t, d, scale)
	// Every third token routes to experts 0..2, so those hold more pairs than
	// one pass covers by the widest chunk.
	ids := make([]int, 0, maxT*k)
	for range maxT {
		perm := rng.Perm(nexp)
		if rng.Intn(3) == 0 {
			perm = []int{0, 1, 2}
		}
		ids = append(ids, perm[:k]...)
	}
	ir := ink2VTestPut(t, d, ink2VTestInts(ids...))
	tab := ink2TestAlloc(t, d, VerifyExpertsTable(len(ids)))
	for _, c := range ink2VMMATestTypes {
		if c.typ == 0 || c.typ == 1 || c.typ == 30 {
			continue
		}
		t.Run(c.name, func(t *testing.T) {
			cols := ink2VMMATestCols(t, c.typ)
			xStride := cols + 32
			x := ink2OpsTestRand(rng, maxT*k*xStride, 1)
			xr := ink2VTestPut(t, d, x)
			ink2VTestWeight(t, store, data, rng, 64, c.typ, rows*nexp, cols)
			m, err := store.Matrix(64, c.typ, rows, cols, nexp)
			if err != nil {
				t.Fatal(err)
			}
			w, ok := m.Ink2Stack()
			if !ok {
				t.Fatal("expert stack split across buffers")
			}
			for _, xDiv := range []int{1, k} {
				npairs := maxT * k
				rep := make([]float32, npairs*xStride)
				for i := range npairs {
					copy(rep[i*xStride:(i+1)*xStride], x[(i/xDiv)*xStride:(i/xDiv+1)*xStride])
				}
				gemv := ink2TestAlloc(t, d, npairs*rows)
				if _, err := d.ExpertMatVec(w, ir, ink2VTestPut(t, d, rep), xStride, gemv, npairs, sr, nil); err != nil {
					t.Fatal(err)
				}
				gv := ink2OpsTestRead(t, d, gemv, npairs*rows)
				ink2VTestFinite(t, "decode experts", gv)
				// Each token's k pairs alone, as a plain step routes them, on the
				// scalar path; then every width on the MMA path.
				ink2VMMATestPath(t, 4)
				alone := ink2TestAlloc(t, d, npairs*rows)
				for tok := range maxT {
					xo := tok * k / xDiv * xStride
					if ok, err := d.VerifyExpertsMMA(w, ir.At(tok*k), xr.At(xo), xDiv, xStride, alone.At(tok*k*rows), k, sr, tab); err != nil || !ok {
						t.Fatalf("token %d alone: ok=%v err=%v", tok, ok, err)
					}
				}
				want := ink2OpsTestRead(t, d, alone, npairs*rows)
				ink2VMMATestNear(t, fmt.Sprintf("xDiv %d tokens alone", xDiv), want, gv)
				ink2VMMATestPath(t, 0)
				for T := 1; T <= maxT; T++ {
					got := ink2TestAlloc(t, d, T*k*rows)
					if ok, err := d.VerifyExpertsMMA(w, ir, xr, xDiv, xStride, got, T*k, sr, tab); err != nil || !ok {
						t.Fatalf("width %d: ok=%v err=%v", T, ok, err)
					}
					ink2OpsTestSame(t, fmt.Sprintf("xDiv %d width %d", xDiv, T), ink2OpsTestRead(t, d, got, T*k*rows), want[:T*k*rows])
				}
			}
		})
	}
}

// The MMA verify kernels against the decode-exact verify kernels at the
// shapes of an Inkling-2-Mini verify chunk, GPU seconds by width.
// FAK_INK2VMMA_BENCH=1 runs it.
// fak-test:runtime slow est=60s
func TestInk2VMMABenchByWidth(t *testing.T) {
	if os.Getenv("FAK_INK2VMMA_BENCH") == "" {
		t.Skip("set FAK_INK2VMMA_BENCH=1 to time the MMA verify kernels")
	}
	lease, err := gpulease.Acquire(gpulease.Options{Timeout: 20 * time.Minute, Logf: t.Logf})
	if err != nil {
		t.Skipf("NOT RUN: %v", err)
	}
	defer lease.Release()
	d := ink2TestDevice(t)
	rng := rand.New(rand.NewSource(95))
	const hidden, k, experts, maxT = 2304, 8, 128, 9
	page := os.Getpagesize()
	widths := []int{1, 2, 4, 8, 9}
	reps := 5
	line := func(name string, f func(T int, mma bool) error) {
		var b strings.Builder
		// interleaved best of reps: the GPU is shared
		g, m := make([]float64, len(widths)), make([]float64, len(widths))
		for range reps {
			for i, T := range widths {
				for arm, best := range [][]float64{g, m} {
					if s := ink2PBenchGPU(t, d, func() error { return f(T, arm == 1) }); best[i] == 0 || s < best[i] {
						best[i] = s
					}
				}
			}
		}
		for i, T := range widths {
			fmt.Fprintf(&b, " w%d gemv %.3fms mma %.3fms;", T, g[i]*1e3, m[i]*1e3)
		}
		t.Logf("%s:%s", name, b.String())
	}
	for _, c := range []struct {
		name       string
		typ        uint32
		rows, cols int
	}{{"q8_0 dense", 8, 3072, hidden}, {"q5_K head", 13, 32768, hidden}, {"bf16 small", 30, 256, hidden}} {
		block, blockBytes, err := packedBlock(c.typ)
		if err != nil {
			t.Fatal(err)
		}
		store, data := ink2TestStore(t, c.rows*(c.cols/block*blockBytes)/page+8)
		w, _ := ink2VTestWeight(t, store, data, rng, 64, c.typ, c.rows, c.cols)
		xr := ink2VTestPut(t, d, ink2OpsTestRand(rng, maxT*c.cols, .5))
		y := ink2TestAlloc(t, d, maxT*c.rows)
		line(fmt.Sprintf("%s [%d][%d]", c.name, c.rows, c.cols), func(T int, mma bool) error {
			if mma {
				_, err := d.VerifyRowsMMA(w, xr, c.cols, y, c.rows, T)
				return err
			}
			return d.VerifyRows(w, xr, c.cols, y, c.rows, T)
		})
	}
	const expertRows = 2304
	block, blockBytes, _ := packedBlock(40)
	store, data := ink2TestStore(t, experts*expertRows*(hidden/block*blockBytes)/page+8)
	ink2VTestWeight(t, store, data, rng, 64, 40, experts*expertRows, hidden)
	m, err := store.Matrix(64, 40, expertRows, hidden, experts)
	if err != nil {
		t.Fatal(err)
	}
	stack, ok := m.Ink2Stack()
	if !ok {
		t.Fatal("stack split across views")
	}
	ids := make([]int, 0, maxT*k)
	for range maxT {
		ids = append(ids, rng.Perm(experts)[:k]...)
	}
	ir := ink2VTestPut(t, d, ink2VTestInts(ids...))
	tab := ink2TestAlloc(t, d, VerifyExpertsTable(len(ids)))
	xr := ink2VTestPut(t, d, ink2OpsTestRand(rng, maxT*hidden, .5))
	y := ink2TestAlloc(t, d, maxT*k*expertRows)
	line(fmt.Sprintf("nvfp4 experts %dx[%d][%d] k=%d", experts, expertRows, hidden, k), func(T int, mma bool) error {
		var err error
		if mma {
			_, err = d.VerifyExpertsMMA(stack, ir, xr, k, hidden, y, T*k, Ink2Ref{}, tab)
		} else {
			_, err = d.VerifyExperts(stack, ir, xr, k, hidden, y, T*k, Ink2Ref{}, tab)
		}
		return err
	})
}

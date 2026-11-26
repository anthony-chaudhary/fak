//go:build darwin && arm64 && cgo

package compute

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"os"
	"syscall"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

// metalInk2VMMANear holds an MMA row within 1e-5 relative L2 of the decode
// GEMV's, whose order it does not keep.
func metalInk2VMMANear(t *testing.T, what string, got, want []float32) {
	t.Helper()
	var num, den float64
	for i := range want {
		d := float64(got[i]) - float64(want[i])
		num, den = num+d*d, den+float64(want[i])*float64(want[i])
	}
	if rel := math.Sqrt(num / max(den, 1e-30)); !(rel <= 1e-5) {
		t.Fatalf("%s: relative L2 %.3g from the decode GEMV", what, rel)
	}
}

// metalInk2VMMAQ8 writes experts*rows*cols random Q8_0 weights and returns the
// store holding them; the caller closes it.
func metalInk2VMMAQ8(t *testing.T, rng *rand.Rand, experts, rows, cols int) (*metalgemm.PackedGGUFStore, *metalgemm.PackedGGUFMatrix) {
	t.Helper()
	const base = 96
	n := base + experts*rows*cols/32*34
	page := os.Getpagesize()
	data, err := syscall.Mmap(-1, 0, (n+page-1)/page*page, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { syscall.Munmap(data) })
	for b := 0; b < experts*rows*cols/32; b++ {
		at := base + 34*b
		binary.LittleEndian.PutUint16(data[at:], uint16(0x3000+rng.Intn(1024)))
		rng.Read(data[at+2 : at+34])
	}
	store, err := metalgemm.NewPackedGGUFStore(data)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(store.Close)
	matrix, err := store.Matrix(base, 8, rows, cols, experts)
	if err != nil {
		t.Fatal(err)
	}
	return store, matrix
}

// In MMA mode a verify row is bit-identical at every width 1..16 and near the
// decode GEMV, the routed pairs too; with the experts switch off the pairs are
// the decode expert GEMV's bits again.
//
// fak-test:runtime integration est=3s lane=default
func TestMetalInkling2VerifyMMARowsAndExpertsAreWidthInvariant(t *testing.T) {
	be := metalInk2TestBackend(t)
	t.Setenv("FAK_INKLING2_VERIFY_MMA", "1")
	if !be.Ink2VerifyMMA() {
		t.Fatal("FAK_INKLING2_VERIFY_MMA=1 does not select the MMA kernels")
	}
	var vd Inkling2VerifyDevice = be
	var moe Inkling2VerifyMoE = be
	rng := rand.New(rand.NewSource(97))
	const rows, cols, maxT, experts, k = 21, 2112, 16, 7, 3

	dense := metalInk2TestUpload(t, be, metalInk2TestRand(rng, rows*cols))
	dense.Shape = []int{rows, cols}
	_, matrix := metalInk2VMMAQ8(t, rng, 1, rows, cols)
	q8, err := MetalInkling2Weight(matrix, 0)
	if err != nil {
		t.Fatal(err)
	}
	x := metalInk2TestRand(rng, maxT*k*cols)
	xr := metalInk2VerifyRef(t, be, x)
	for _, c := range []struct {
		name string
		w    Tensor
		run  func(w Tensor, x, y Ink2Ref, T int) (bool, error)
	}{
		{"dense f32", dense, vd.Ink2VRows},
		{"q8_0 rows", q8, vd.Ink2VRows},
		{"q8_0 shared-codes entry", q8, func(w Tensor, x, y Ink2Ref, T int) (bool, error) {
			return vd.Ink2VQ8Rows(w, x, Ink2Ref{}, Ink2Ref{}, y, T, true)
		}},
	} {
		t.Run(c.name, func(t *testing.T) {
			decode, alone := metalInk2TestAlloc(t, be, maxT*rows), metalInk2TestAlloc(t, be, maxT*rows)
			for r := range maxT {
				if err := be.Ink2MatVec(c.w, Ink2At(xr.T, r*cols), Ink2At(decode, r*rows)); err != nil {
					t.Fatal(err)
				}
				if ok, err := c.run(c.w, Ink2At(xr.T, r*cols), Ink2At(alone, r*rows), 1); err != nil || !ok {
					t.Fatalf("row %d alone: ok=%v err=%v", r, ok, err)
				}
			}
			want := metalInk2VerifyRead(t, be, Ink2At(alone, 0), maxT*rows)
			metalInk2VMMANear(t, "rows alone", want, metalInk2VerifyRead(t, be, Ink2At(decode, 0), maxT*rows))
			for T := 2; T <= maxT; T++ {
				y := metalInk2TestAlloc(t, be, T*rows)
				if ok, err := c.run(c.w, xr, Ink2At(y, 0), T); err != nil || !ok {
					t.Fatalf("width %d: ok=%v err=%v", T, ok, err)
				}
				metalInk2VerifySame(t, fmt.Sprintf("width %d vs each row alone", T), metalInk2VerifyRead(t, be, Ink2At(y, 0), T*rows), want[:T*rows])
			}
		})
	}

	_, em := metalInk2VMMAQ8(t, rng, experts, rows, cols)
	stack, err := MetalInkling2Weight(em, -1)
	if err != nil {
		t.Fatal(err)
	}
	ids := make([]int, 0, maxT*k)
	for range maxT {
		ids = append(ids, rng.Perm(experts)[:k]...)
	}
	idr := metalInk2VerifyRef(t, be, metalInk2VerifyInts(ids...))
	scr := metalInk2VerifyRef(t, be, []float32{2, -0.5, 1.25, 3, 0.75, 1, 1.5})
	npairs := maxT * k
	rep := make([]float32, npairs*cols)
	for i := range npairs {
		copy(rep[i*cols:(i+1)*cols], x[(i/k)*cols:(i/k+1)*cols])
	}
	gemv := Ink2At(metalInk2TestAlloc(t, be, npairs*rows), 0)
	if ok, err := be.Ink2ExpertMatVec(stack, rows, idr, metalInk2VerifyRef(t, be, rep), cols, gemv, npairs, scr, nil); err != nil || !ok {
		t.Fatalf("expert matvec: ok=%v err=%v", ok, err)
	}
	gv := metalInk2VerifyRead(t, be, gemv, npairs*rows)
	run := func(T int) []float32 {
		t.Helper()
		got := Ink2At(metalInk2TestAlloc(t, be, T*k*rows), 0)
		if ok, err := moe.Ink2VExperts(stack, rows, idr, xr, k, cols, got, T*k, scr); err != nil || !ok {
			t.Fatalf("verify experts width %d: ok=%v err=%v", T, ok, err)
		}
		return metalInk2VerifyRead(t, be, got, T*k*rows)
	}
	want := run(maxT)
	metalInk2VMMANear(t, "routed pairs", want, gv)
	for T := 1; T < maxT; T++ {
		metalInk2VerifySame(t, fmt.Sprintf("experts width %d", T), run(T), want[:T*k*rows])
	}
	t.Setenv("FAK_INKLING2_VERIFY_MMA_EXPERTS", "0")
	metalInk2VerifySame(t, "experts on the GEMV kernel", run(maxT), gv)
}

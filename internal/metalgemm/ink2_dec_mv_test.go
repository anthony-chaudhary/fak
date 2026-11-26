//go:build darwin && arm64 && cgo

package metalgemm

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"testing"
)

func ink2MVUseLegacy(t *testing.T, legacy bool) {
	t.Helper()
	old := ink2DecMVLegacy
	ink2DecMVLegacy = legacy
	t.Cleanup(func() { ink2DecMVLegacy = old })
}

// ink2MVArms is the planned kernel of a type under every threadgroup height.
func ink2MVArms(typ uint32, cols int) (arms []ink2MVChoice) {
	for rowBits := range 4 {
		arms = append(arms, ink2MVChoice{ink2MVPick(typ, cols).kernel, rowBits})
	}
	return arms
}

func ink2MVPrevious(t *testing.T, d *Ink2Device, w Ink2Weight, x, y Ink2Ref) []float32 {
	t.Helper()
	old := ink2DecMVLegacy
	ink2DecMVLegacy = true
	defer func() { ink2DecMVLegacy = old }()
	if err := d.MatVec(w, x, y); err != nil {
		t.Fatal(err)
	}
	got, err := d.Read(y, w.Rows)
	if err != nil {
		t.Fatal(err)
	}
	return got
}

// ink2MVHostWeights decodes f32 and bf16 weights for a host reference, or
// returns nil.
func ink2MVHostWeights(raw []byte, typ uint32, n int) []float64 {
	ref := make([]float64, n)
	switch typ {
	case 0:
		for i := range ref {
			ref[i] = float64(math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:])))
		}
	case 30:
		for i := range ref {
			ref[i] = float64(math.Float32frombits(uint32(binary.LittleEndian.Uint16(raw[2*i:])) << 16))
		}
	default:
		return nil
	}
	return ref
}

// fak-test:runtime integration est=15s lane=default
func TestInk2DecMVMatchesMVBitForBit(t *testing.T) {
	d := ink2TestDevice(t)
	ink2MVUseLegacy(t, false)
	rng := rand.New(rand.NewSource(71))
	compared := 0
	for _, typ := range []uint32{0, 1, 30} {
		for _, cols := range []int{2304, 3072} {
			for _, rows := range []int{24, 61, 385} {
				off := 64 + 2*(rows%3)
				if typ == 0 {
					off = 64
				}
				m := ink2DecWeight(t, d, typ, rows, cols, 1, off, uint64(rng.Int63()))
				x := ink2TestAlloc(t, d, cols+rows+64)
				y := x.At(cols + 32)
				xs := make([]float32, cols)
				for i := range xs {
					xs[i] = rng.Float32() - .5
				}
				if err := d.Write(x, xs); err != nil {
					t.Fatal(err)
				}
				want := ink2MVPrevious(t, d, m.w, x, y)
				finite := 0
				for _, v := range want {
					if v != 0 && !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0) {
						finite++
					}
				}
				if finite < len(want)/2 {
					t.Fatalf("type %d: only %d of %d mv_ outputs are finite and nonzero", typ, finite, len(want))
				}
				arms := append(ink2MVArms(typ, cols), ink2MVChoice{})
				for _, pl := range arms {
					if err := d.Zero(y, rows); err != nil {
						t.Fatal(err)
					}
					var err error
					if pl.kernel == "" {
						err = d.MatVec(m.w, x, y)
					} else {
						err = d.matVecDec(pl, m.w, x, y)
					}
					if err != nil {
						t.Fatalf("type %d %+v: %v", typ, pl, err)
					}
					got, err := d.Read(y, rows)
					if err != nil {
						t.Fatal(err)
					}
					if n := ink2DecDiffer(got, want); n != 0 {
						t.Errorf("type %d %dx%d %+v: %d of %d outputs differ in bits", typ, rows, cols, pl, n, rows)
					}
					compared += rows
				}
				ref := ink2MVHostWeights(m.raw, typ, rows*cols)
				if ref == nil {
					continue
				}
				for r := range rows {
					var sum, mag float64
					for j, w := range ref[r*cols:][:cols] {
						p := w * float64(xs[j])
						sum, mag = sum+p, mag+math.Abs(p)
					}
					if !ink2TestClose(want[r], sum, mag) {
						t.Fatalf("type %d %dx%d row %d: device %v, host %v", typ, rows, cols, r, want[r], sum)
					}
				}
			}
		}
	}
	t.Logf("%d outputs compared bit for bit", compared)
}

// fak-test:runtime integration est=2s lane=default
func TestInk2DecMVPicksOnlyTheScalarDecodeShapes(t *testing.T) {
	ink2MVUseLegacy(t, false)
	for _, c := range []struct {
		typ  uint32
		cols int
		want string
	}{{0, 2304, "ink2_dec_mv_f32_2304"}, {30, 2304, "ink2_dec_mv_bf16_2304"}, {1, 3072, "ink2_dec_mv_f16_3072"},
		{30, 1152, ""}, {8, 2304, ""}, {13, 2304, ""}, {14, 3072, ""}, {40, 2304, ""}} {
		if got := ink2MVPick(c.typ, c.cols).kernel; got != c.want {
			t.Errorf("type %d by %d columns: plan %q, want %q", c.typ, c.cols, got, c.want)
		}
	}
	ink2DecMVLegacy = true
	if pl := ink2MVPick(30, 2304); pl.kernel != "" {
		t.Errorf("mvlat legacy still plans %+v", pl)
	}
}

// fak-test:runtime slow est=60s
func TestInk2DecMVMicrobench(t *testing.T) {
	d := ink2OpsBenchDevice(t)
	ink2MVUseLegacy(t, false)
	const sites = 8
	var cases []ink2OpsBenchCase
	for _, c := range []struct {
		typ        uint32
		rows, cols int
	}{{0, 128, 2304}, {30, 24, 2304}, {30, 32, 2304}, {1, 128, 3072}} {
		ws := make([]Ink2Weight, sites)
		for i := range ws {
			ws[i] = ink2DecWeight(t, d, c.typ, c.rows, c.cols, 1, 64, uint64(i+1)).w
		}
		x := ink2OpsBenchFill(t, d, c.cols, 7, 1)
		y := ink2TestAlloc(t, d, c.rows)
		arms := []ink2OpsBenchArm{{"mv_", func(i int) error {
			ink2DecMVLegacy = true
			defer func() { ink2DecMVLegacy = false }()
			return d.MatVec(ws[i], x, y)
		}}}
		for _, pl := range ink2MVArms(c.typ, c.cols) {
			arms = append(arms, ink2OpsBenchArm{fmt.Sprintf("%s R%d", pl.kernel[len("ink2_dec_mv_"):], 1<<pl.rowBits),
				func(i int) error { return d.matVecDec(pl, ws[i], x, y) }})
		}
		cases = append(cases, ink2OpsBenchCase{name: fmt.Sprintf("mv type %d %dx%d", c.typ, c.rows, c.cols), sites: sites, arms: arms})
	}
	ink2OpsBenchRun(t, d, cases, 64, ink2OpsBenchReps(t))
}

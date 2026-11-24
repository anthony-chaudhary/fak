//go:build cuda && cgo

package compute

import (
	"math"
	"math/rand"
	"testing"
)

func ink2BF16(x float32) float64 {
	b := math.Float32bits(x)
	b += 0x7fff + ((b >> 16) & 1)
	return float64(math.Float32frombits(b & 0xffff0000))
}

// The bandwidth-shaped NVFP4 expert GEMV keeps the reference kernel's per-lane
// arithmetic, so the two must agree bit for bit; both are also checked against a
// float64 host dot product.
// fak-test:runtime integration est=2s lane=optin
func TestCUDAInk2ExpertNVFP4FastMatchesReference(t *testing.T) {
	cb := cudaNVFP4OrSkip(t)
	defer ink2SelectNVFP4GEMV(ink2SelectNVFP4GEMV(true))
	r := rand.New(rand.NewSource(29))
	const experts = 10
	for _, c := range []struct {
		name                string
		rows, in, k, xs, sw int
		scaled              bool
	}{
		{"gate_up shared x", 2304, 2304, 8, 0, 0, true},
		{"down per-slot x", 2304, 1152, 8, 1152, 0, true},
		{"ragged rows", 37, 80, 3, 80, 0, false},
		{"widest kb", 19, 4096, 2, 4096, 0, true},
		{"situ staged", 48, 256, 4, 400, 200, false},
	} {
		packed := make([]byte, experts*c.rows*c.in/2)
		r.Read(packed)
		scales := make([]byte, experts*c.rows*c.in/16)
		for i := range scales {
			scales[i] = byte(0x20 + r.Intn(0x30))
		}
		stack := cudaNVFP4Resident(t, cb, []int{experts * c.rows, c.in}, packed, scales, 1)
		xn := c.in
		if c.xs > 0 {
			xn = c.k * c.xs
		}
		x := make([]float32, xn)
		for i := range x {
			x[i] = float32(r.NormFloat64())
		}
		ids := make([]float32, c.k)
		idv := r.Perm(experts)[:c.k]
		for i, id := range idv {
			ids[i] = math.Float32frombits(uint32(id))
		}
		esc := make([]float32, experts)
		for i := range esc {
			esc[i] = 0.5 + r.Float32()
		}
		alloc := func(n int) Tensor {
			tt, err := cb.Ink2Alloc(n)
			if err != nil {
				t.Fatal(err)
			}
			return tt
		}
		dx, dids, des := alloc(xn), alloc(c.k), alloc(experts)
		for _, w := range []struct {
			dst Tensor
			src []float32
		}{{dx, x}, {dids, ids}, {des, esc}} {
			if err := cb.Ink2Write(Ink2At(w.dst, 0), w.src); err != nil {
				t.Fatal(err)
			}
		}
		var scale Ink2Ref
		if c.scaled {
			scale = Ink2At(des, 0)
		}
		var situ *Ink2Situ
		if c.sw > 0 {
			situ = &Ink2Situ{Width: c.sw, Beta: 1.5, LinearBeta: 6}
		}
		run := func(fast bool) []float32 {
			ink2SelectNVFP4GEMV(fast)
			y := alloc(c.k * c.rows)
			ok, err := cb.Ink2ExpertMatVec(stack, c.rows, Ink2At(dids, 0), Ink2At(dx, 0), c.xs, Ink2At(y, 0), c.k, scale, situ)
			if err != nil || !ok {
				t.Fatalf("%s: ok=%v err=%v", c.name, ok, err)
			}
			out, err := cb.Ink2Read(Ink2At(y, 0), c.k*c.rows)
			if err != nil {
				t.Fatal(err)
			}
			return out
		}
		want, got := run(false), run(true)
		for i := range want {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("%s: y[%d] fast %v (%08x), reference %v (%08x)", c.name, i, got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
			}
		}
		if situ != nil {
			continue
		}
		w, err := DecodeNVFP4Reference(experts*c.rows, c.in, packed, scales, 1)
		if err != nil {
			t.Fatal(err)
		}
		for s, id := range idv {
			xo := s * c.xs
			for o := 0; o < c.rows; o++ {
				var dot, mag float64
				row := w[(id*c.rows+o)*c.in : (id*c.rows+o+1)*c.in]
				for i, wv := range row {
					p := float64(wv) * ink2BF16(x[xo+i])
					dot += p
					mag += math.Abs(p)
				}
				if c.scaled {
					dot *= float64(esc[id])
					mag *= float64(esc[id])
				}
				if g := float64(got[s*c.rows+o]); math.Abs(g-dot) > 1e-5*mag+1e-6 {
					t.Fatalf("%s slot %d row %d: device %v, host %v (|w||x| %v)", c.name, s, o, g, dot, mag)
				}
			}
		}
	}
}

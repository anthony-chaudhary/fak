//go:build cuda && cgo

package compute

import (
	"math"
	"math/rand"
	"testing"
)

// The resident decode attend over f16 global-slot caches (FAK_INKLING2_KV=f16) against
// the fp32 attend over the same caches with every value rounded to f16 first, so the
// only difference left is the f16 cache read. Output must agree within the f16 tolerance.
// fak-test:runtime integration est=2s lane=optin
func TestCUDAInk2AttendHalfKVMatchesFP32(t *testing.T) {
	cb := cudaNVFP4OrSkip(t)
	p := Ink2AttnParams{Heads: 24, KVHeads: 8, HeadDim: 128, RelLen: 16, Scale: float32(1 / math.Sqrt(128))}
	kvDim := p.KVHeads * p.HeadDim
	r := rand.New(rand.NewSource(41))
	alloc := func(n int) Tensor {
		tt, err := cb.Ink2Alloc(n)
		if err != nil {
			t.Fatal(err)
		}
		return tt
	}
	write := func(tt Tensor, v []float32) {
		if err := cb.Ink2Write(Ink2At(tt, 0), v); err != nil {
			t.Fatal(err)
		}
	}
	randVec := func(n int, sd float64) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(sd * r.NormFloat64())
		}
		return v
	}
	// packHalf rounds vals to f16 and packs two f16 elements per float32 word, low half
	// first, as the device reads them; rounded is the fp32 view of the same values.
	packHalf := func(vals []float32) (words, rounded []float32) {
		rounded = make([]float32, len(vals))
		bits := make([]uint16, len(vals))
		for i, x := range vals {
			bits[i] = Float32ToFloat16Bits(x)
			rounded[i] = Float16BitsToFloat32(bits[i])
		}
		words = make([]float32, len(vals)/2)
		for i := range words {
			words[i] = math.Float32frombits(uint32(bits[2*i]) | uint32(bits[2*i+1])<<16)
		}
		return words, rounded
	}
	cases := []ink2AttendCase{
		{1024, 1024, 1024, 82}, {1024, 1024, 1024, 2000},
		{4096, 0, 1024, 2000}, {4096, 0, 1024, 5000},
	}
	for _, c := range cases {
		q, rr, sinks := alloc(p.Heads*p.HeadDim), alloc(p.Heads*p.RelLen), alloc(p.Heads)
		relBias, meta := alloc(c.extent*p.RelLen), alloc(1)
		kcF32, vcF32 := alloc(c.cap*kvDim), alloc(c.cap*kvDim)
		kcHalf, vcHalf := alloc(c.cap*kvDim/2), alloc(c.cap*kvDim/2)
		scores := alloc(2 * p.Heads * c.cap)
		qv, rv := randVec(p.Heads*p.HeadDim, 1), randVec(p.Heads*p.RelLen, 1)
		sv := randVec(p.Heads, 1)
		bv := randVec(c.extent*p.RelLen, 0.1)
		kw, kr := packHalf(randVec(c.cap*kvDim, 0.3))
		vw, vr := packHalf(randVec(c.cap*kvDim, 1))
		write(q, qv)
		write(sinks, sv)
		write(relBias, bv)
		write(kcF32, kr)
		write(vcF32, vr)
		write(kcHalf, kw)
		write(vcHalf, vw)
		write(meta, []float32{math.Float32frombits(uint32(c.pos))})
		run := func(half bool) []float32 {
			// the attend's split launches park the head max in the rel-query row
			write(rr, rv)
			out := alloc(p.Heads * p.HeadDim)
			var err error
			if half {
				err = cb.Ink2AttendHalf(Ink2At(q, 0), Ink2At(rr, 0), Ink2At(sinks, 0), Ink2At(relBias, 0), Ink2At(kcHalf, 0),
					Ink2At(vcHalf, 0), Ink2At(scores, 0), Ink2At(out, 0), Ink2At(meta, 0), p, c.extent, c.window, c.cap)
			} else {
				err = cb.Ink2Attend(Ink2At(q, 0), Ink2At(rr, 0), Ink2At(sinks, 0), Ink2At(relBias, 0), Ink2At(kcF32, 0),
					Ink2At(vcF32, 0), Ink2At(scores, 0), Ink2At(out, 0), Ink2At(meta, 0), p, c.extent, c.window, c.cap)
			}
			if err != nil {
				t.Fatal(err)
			}
			got, err := cb.Ink2Read(Ink2At(out, 0), p.Heads*p.HeadDim)
			if err != nil {
				t.Fatal(err)
			}
			return got
		}
		want, got := run(false), run(true)
		for i := range want {
			diff := math.Abs(float64(got[i]) - float64(want[i]))
			if math.IsNaN(diff) || diff > 1e-3*(1+math.Abs(float64(want[i]))) {
				t.Fatalf("cap %d window %d pos %d: out[%d] f16 %v, fp32 reference %v", c.cap, c.window, c.pos, i, got[i], want[i])
			}
		}
	}
}

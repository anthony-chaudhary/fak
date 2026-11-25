//go:build cuda && cgo

package compute

import (
	"math"
	"math/rand"
	"testing"
)

// f16ToF32 widens one IEEE binary16 value.
func f16ToF32(h uint16) float32 {
	sign := uint32(h>>15) << 31
	exp := int(h>>10) & 0x1f
	man := uint32(h & 0x3ff)
	switch {
	case exp == 0:
		v := float32(math.Ldexp(float64(man), -24))
		if sign != 0 {
			v = -v
		}
		return v
	case exp == 31:
		return math.Float32frombits(sign | 0x7f800000 | man<<13)
	}
	return math.Float32frombits(sign | uint32(exp+112)<<23 | man<<13)
}

// The f16-cache append runs the fp32 append's arithmetic and narrows only the stored
// value with round-to-nearest, and the tensor-core attend narrows fp32 cache tiles the
// same way, so a chunk appended and attended over f16 caches gives the fp32 caches'
// output bit for bit: across a ring wrap, a sliding window and the global log-scale.
// fak-test:runtime integration est=10s lane=optin
func TestCUDAInk2PHalfKVMatchesFloat(t *testing.T) {
	cb := cudaNVFP4OrSkip(t)
	r := rand.New(rand.NewSource(43))
	p := Ink2AttnParams{Heads: 24, KVHeads: 8, HeadDim: 128, RelLen: 16, QKClip: 3, RelClip: 2, Eps: 1e-6,
		Alpha: 0.1, Floor: 64, Scale: float32(1 / math.Sqrt(128))}
	kvDim := p.KVHeads * p.HeadDim
	alloc := func(n int) Tensor {
		tt, err := cb.Ink2Alloc(n)
		if err != nil {
			t.Fatal(err)
		}
		return tt
	}
	host := func(n int, sd, mean float64) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(mean + sd*r.NormFloat64())
		}
		return v
	}
	write := func(tt Tensor, v []float32) {
		if err := cb.Ink2Write(Ink2At(tt, 0), v); err != nil {
			t.Fatal(err)
		}
	}
	read := func(tt Tensor, n int) []float32 {
		v, err := cb.Ink2Read(Ink2At(tt, 0), n)
		if err != nil {
			t.Fatal(err)
		}
		return v
	}
	qclip, rclip := alloc(p.Heads*p.HeadDim), alloc(p.Heads*p.RelLen)
	write(qclip, host(p.Heads*p.HeadDim, 0.1, 1))
	write(rclip, host(p.Heads*p.RelLen, 0.1, 1))
	for _, c := range []struct {
		name                      string
		cap, window, extent, pos0 int
		T                         int
		global                    bool
	}{
		{"first chunk global", 1024, 0, 1024, 0, 300, true},
		{"later chunk global", 2048, 0, 1024, 512, 128, true},
		{"local window ring wrap", 1024, 512, 512, 900, 200, false},
	} {
		ap := p
		ap.Global = c.global
		extent := c.extent
		rel := alloc(extent * p.RelLen)
		write(rel, host(extent*p.RelLen, 0.1, 0))
		// the history chunk fills positions 0..pos0-1, then the checked chunk follows
		chunks := []struct{ pos0, T int }{{0, c.pos0}, {c.pos0, c.T}}
		type in struct{ q, r, k, v, sinks []float32 }
		ins := make([]in, len(chunks))
		for i, ch := range chunks {
			ins[i] = in{host(ch.T*p.Heads*p.HeadDim, 1, 0), host(ch.T*p.Heads*p.RelLen, 1, 0),
				host(ch.T*kvDim, 0.4, 0), host(ch.T*kvDim, 1, 0), host(ch.T*p.Heads, 1, 0)}
		}
		run := func(half bool) (out []float32, kc, vc []float32) {
			n := c.cap * kvDim
			if half {
				n = (n + 1) / 2
			}
			kcT, vcT := alloc(n), alloc(n)
			write(kcT, make([]float32, n))
			write(vcT, make([]float32, n))
			for i, ch := range chunks {
				if ch.T == 0 {
					continue
				}
				q, rr, k, v, sinks := alloc(len(ins[i].q)), alloc(len(ins[i].r)), alloc(len(ins[i].k)), alloc(len(ins[i].v)), alloc(len(ins[i].sinks))
				write(q, ins[i].q)
				write(rr, ins[i].r)
				write(k, ins[i].k)
				write(v, ins[i].v)
				write(sinks, ins[i].sinks)
				o := alloc(ch.T * p.Heads * p.HeadDim)
				post, attend := cb.Ink2PQKVPost, cb.Ink2PAttend
				if half {
					post, attend = cb.Ink2PQKVPostHalf, cb.Ink2PAttendHalf
				}
				if err := post(Ink2At(q, 0), Ink2At(rr, 0), Ink2At(k, 0), Ink2At(v, 0), Ink2At(qclip, 0), Ink2At(rclip, 0),
					Ink2At(kcT, 0), Ink2At(vcT, 0), ap, c.cap, ch.pos0, ch.T); err != nil {
					t.Fatalf("%s half %v chunk %d: qkv post: %v", c.name, half, i, err)
				}
				if err := attend(Ink2At(q, 0), Ink2At(rr, 0), Ink2At(sinks, 0), Ink2At(rel, 0), Ink2At(kcT, 0), Ink2At(vcT, 0),
					Ink2At(o, 0), ap, extent, c.window, c.cap, ch.pos0, ch.T); err != nil {
					t.Fatalf("%s half %v chunk %d: attend: %v", c.name, half, i, err)
				}
				if i == len(chunks)-1 {
					out = read(o, ch.T*p.Heads*p.HeadDim)
				}
			}
			return out, read(kcT, n), read(vcT, n)
		}
		want, kf, vf := run(false)
		got, kh, vh := run(true)
		for i := range want {
			if math.IsNaN(float64(want[i])) || math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("%s: out[%d] f16 caches %v, fp32 caches %v", c.name, i, got[i], want[i])
			}
		}
		// every stored half is the fp32 value rounded to nearest
		for name, pair := range map[string][2][]float32{"k": {kf, kh}, "v": {vf, vh}} {
			f, h := pair[0], pair[1]
			for i := range f {
				bits := math.Float32bits(h[i/2])
				x := f16ToF32(uint16(bits >> (16 * uint(i%2))))
				if d := math.Abs(float64(x - f[i])); d > math.Abs(float64(f[i]))/2048+1.0/(1<<25) {
					t.Fatalf("%s: %s cache[%d] f16 %v, fp32 %v", c.name, name, i, x, f[i])
				}
			}
		}
	}
	// a shape off the tensor-core tile has no f16 path
	bad := p
	bad.HeadDim = 64
	tt := alloc(64)
	if err := cb.Ink2PAttendHalf(Ink2At(tt, 0), Ink2At(tt, 0), Ink2At(tt, 0), Ink2At(tt, 0), Ink2At(tt, 0), Ink2At(tt, 0), Ink2At(tt, 0),
		bad, 16, 0, 16, 0, 1); err == nil {
		t.Fatal("f16 attend accepted a head size the tensor-core tile does not cover")
	}
}

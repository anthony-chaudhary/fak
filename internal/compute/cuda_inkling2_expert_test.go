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

// The IQ expert kernels keep k_ink2_expert_iq's per-lane term and block order, so the
// two must agree bit for bit; both are also checked against a float64 host dot over
// DequantRawIQ of the same super-blocks.
// fak-test:runtime integration est=3s lane=optin
func TestCUDAInk2ExpertIQFastMatchesReference(t *testing.T) {
	cb := cudaNVFP4OrSkip(t)
	defer ink2SelectIQGEMV(ink2SelectIQGEMV(true))
	r := rand.New(rand.NewSource(31))
	const experts = 10
	for _, dt := range []Dtype{IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL} {
		for _, c := range []struct{ rows, in, k, xs int }{{2304, 2304, 8, 0}, {2304, 1280, 8, 1280}, {37, 512, 3, 512}} {
			seed := uint64(c.rows*131+c.in) ^ uint64(dt)
			var raw []byte
			if dt == IQ4_NL {
				raw = iqGEMVTestRawFor(dt, experts*c.rows, c.in, seed)
			} else {
				bb, ok := RawIQBlockBytes(dt)
				if !ok {
					t.Fatalf("%s is not a raw i-quant", dt)
				}
				raw = iqGEMVTestRaw(experts*c.rows, c.in, bb, seed)
			}
			stack := cb.Upload(NewRawIQ(cb, dt, []int{experts * c.rows, c.in}, raw), dt)
			xn := c.in
			if c.xs > 0 {
				xn = c.k * c.xs
			}
			x := make([]float32, xn)
			for i := range x {
				x[i] = float32(r.NormFloat64())
			}
			idv := r.Perm(experts)[:c.k]
			ids := make([]float32, c.k)
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
			run := func(fast bool) []float32 {
				ink2SelectIQGEMV(fast)
				y := alloc(c.k * c.rows)
				ok, err := cb.Ink2ExpertMatVec(stack, c.rows, Ink2At(dids, 0), Ink2At(dx, 0), c.xs, Ink2At(y, 0), c.k, Ink2At(des, 0), nil)
				if err != nil || !ok {
					t.Fatalf("%s: ok=%v err=%v", dt, ok, err)
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
					t.Fatalf("%s %dx%d: y[%d] fast %v, reference %v", dt, c.rows, c.in, i, got[i], want[i])
				}
			}
			w := DequantRawIQ(dt, experts*c.rows, c.in, raw)
			for s, id := range idv {
				for o := 0; o < c.rows; o++ {
					var dot, mag float64
					for i, wv := range w[(id*c.rows+o)*c.in : (id*c.rows+o+1)*c.in] {
						p := float64(wv) * float64(x[s*c.xs+i])
						dot += p
						mag += math.Abs(p)
					}
					dot *= float64(esc[id])
					mag *= float64(esc[id])
					if g := float64(got[s*c.rows+o]); math.Abs(g-dot) > 1e-5*mag+1e-6 {
						t.Fatalf("%s %dx%d slot %d row %d: device %v, host %v (|w||x| %v)", dt, c.rows, c.in, s, o, g, dot, mag)
					}
				}
			}
		}
	}
}

// The split attend keeps the reference's per-key score, exp and serial denominator
// order and its 4-group value sums, so the two must agree bit for bit at every depth,
// for local (window) and global (full-cache, wrapped ring) slots.
// fak-test:runtime integration est=2s lane=optin
func TestCUDAInk2AttendSplitMatchesReference(t *testing.T) {
	ink2AttendSplitVsReference(t, true, false, false, ink2SmallCapCases)
}

// The tree denominator and 16 key groups reorder double and float sums, so the
// output is held to a tolerance instead of bit identity.
func TestCUDAInk2AttendTreeNearReference(t *testing.T) {
	ink2AttendSplitVsReference(t, false, true, false, ink2SmallCapCases)
}

// Past the shared-memory staging limit (cap 16416 on an RTX 5080) the exact mode
// tiles the weights and sums the denominator in parallel, so it matches the reference
// to float rounding, for a short prompt in a long cache, a deep one and a wrapped ring.
// fak-test:runtime integration est=2s lane=optin
func TestCUDAInk2AttendSplitLargeCapMatchesReference(t *testing.T) {
	ink2AttendSplitVsReference(t, true, true, false, ink2LargeCapCases)
}

// The same large-cap cases over f16 KV caches, the layout FAK_INKLING2_BATCH_KV=f16 uses.
// fak-test:runtime integration est=2s lane=optin
func TestCUDAInk2AttendSplitLargeCapF16MatchesReference(t *testing.T) {
	ink2AttendSplitVsReference(t, true, true, true, ink2LargeCapCases)
}

type ink2AttendCase struct{ cap, window, extent, pos int }

var ink2SmallCapCases = []ink2AttendCase{
	{1024, 1024, 1024, 0}, {1024, 1024, 1024, 82}, {1024, 1024, 1024, 2000},
	{4096, 0, 1024, 82}, {4096, 0, 1024, 2000}, {4096, 0, 1024, 5000},
}

var ink2LargeCapCases = []ink2AttendCase{
	{16416, 0, 1024, 0}, {16416, 0, 1024, 82}, {16416, 0, 1024, 2047}, {16416, 0, 1024, 9000},
	{16416, 0, 1024, 20000},
}

// ink2AttendSplitVsReference checks the split attend against the reference over cases.
// exact selects the bit-identical output launch; tolerant accepts float rounding
// instead of bit identity; kvHalf stores the caches as f16.
func ink2AttendSplitVsReference(t *testing.T, exact, tolerant, kvHalf bool, cases []ink2AttendCase) {
	cb := cudaNVFP4OrSkip(t)
	defer ink2SelectAttendSplit(ink2SelectAttendSplit(true))
	defer ink2SelectAttendExact(ink2SelectAttendExact(exact))
	r := rand.New(rand.NewSource(37))
	p := Ink2AttnParams{Heads: 24, KVHeads: 8, HeadDim: 128, RelLen: 16, Scale: float32(1 / math.Sqrt(128))}
	kvDim := p.KVHeads * p.HeadDim
	alloc := func(n int) Tensor {
		tt, err := cb.Ink2Alloc(n)
		if err != nil {
			t.Fatal(err)
		}
		return tt
	}
	// two f16 elements pack into one float32 word of the device buffer
	allocKV := func(n int) Tensor {
		if kvHalf {
			return alloc(n / 2)
		}
		return alloc(n)
	}
	fill := func(tt Tensor, n int, sd float64) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(sd * r.NormFloat64())
		}
		if err := cb.Ink2Write(Ink2At(tt, 0), v); err != nil {
			t.Fatal(err)
		}
		return v
	}
	fillKV := func(tt Tensor, n int, sd float64) {
		if !kvHalf {
			fill(tt, n, sd)
			return
		}
		words := make([]float32, n/2)
		for i := range words {
			lo, hi := f16Bits(float32(sd*r.NormFloat64())), f16Bits(float32(sd*r.NormFloat64()))
			words[i] = math.Float32frombits(uint32(lo) | uint32(hi)<<16)
		}
		if err := cb.Ink2Write(Ink2At(tt, 0), words); err != nil {
			t.Fatal(err)
		}
	}
	for _, c := range cases {
		q, rr, sinks, rel := alloc(p.Heads*p.HeadDim), alloc(p.Heads*p.RelLen), alloc(p.Heads), alloc(c.extent*p.RelLen)
		kc, vc, meta := allocKV(c.cap*kvDim), allocKV(c.cap*kvDim), alloc(1)
		scores := alloc(2 * p.Heads * c.cap)
		fill(q, p.Heads*p.HeadDim, 1)
		rv := fill(rr, p.Heads*p.RelLen, 1)
		fill(sinks, p.Heads, 1)
		fill(rel, c.extent*p.RelLen, 0.1)
		fillKV(kc, c.cap*kvDim, 0.3)
		fillKV(vc, c.cap*kvDim, 1)
		if err := cb.Ink2Write(Ink2At(meta, 0), []float32{math.Float32frombits(uint32(c.pos))}); err != nil {
			t.Fatal(err)
		}
		run := func(split bool) []float32 {
			// the split launches park the head max in the rel-query row
			if err := cb.Ink2Write(Ink2At(rr, 0), rv); err != nil {
				t.Fatal(err)
			}
			ink2SelectAttendSplit(split)
			out := alloc(p.Heads * p.HeadDim)
			var err error
			if kvHalf {
				err = cb.Ink2BAttend(Ink2At(q, 0), Ink2At(rr, 0), Ink2At(sinks, 0), Ink2At(rel, 0), Ink2At(kc, 0), Ink2At(vc, 0),
					Ink2At(scores, 0), Ink2At(out, 0), Ink2At(meta, 0), p, c.extent, c.window, c.cap, Ink2SeqStrides{}, true, 1)
			} else {
				err = cb.Ink2Attend(Ink2At(q, 0), Ink2At(rr, 0), Ink2At(sinks, 0), Ink2At(rel, 0), Ink2At(kc, 0), Ink2At(vc, 0),
					Ink2At(scores, 0), Ink2At(out, 0), Ink2At(meta, 0), p, c.extent, c.window, c.cap)
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
			if math.IsNaN(float64(want[i])) {
				t.Fatalf("cap %d window %d pos %d: reference out[%d] is NaN", c.cap, c.window, c.pos, i)
			}
			same := math.Float32bits(got[i]) == math.Float32bits(want[i])
			if tolerant {
				diff := math.Abs(float64(got[i]) - float64(want[i]))
				same = diff <= 1e-6+1e-5*math.Abs(float64(want[i]))
			}
			if !same {
				t.Fatalf("cap %d window %d pos %d: out[%d] split %v, reference %v", c.cap, c.window, c.pos, i, got[i], want[i])
			}
		}
	}
}

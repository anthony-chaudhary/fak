package compute

import (
	"math"
	"math/big"
	"math/rand"
	"testing"
)

// Host port of the activation quantizer in k_ink2p_nvf4_quant (llama.cpp's
// quantize_mmq_nvfp4 algorithm, IEEE-rounded): every float32 step rounds where the
// device rounds, so the codes and nibbles match the kernel bit for bit. They need not
// match fast-math llama.cpp on near-tie blocks.

// ink2pFMA32 is fmaf: a*b+c rounded once to float32. a*b is exact in float64 and
// TwoSum keeps the residual of the float64 sum, so the only double-rounding case, a
// float64 sum landing exactly on a float32 midpoint, is broken by the residual's sign.
func ink2pFMA32(a, b, c float32) float32 {
	p, cf := float64(a)*float64(b), float64(c)
	s := p + cf
	bv := s - p
	res := (p - (s - bv)) + (cf - bv)
	r := float32(s)
	if res == 0 || float64(r) == s {
		return r
	}
	toward := float32(math.Inf(1))
	if s < float64(r) {
		toward = float32(math.Inf(-1))
	}
	if o := math.Nextafter32(r, toward); (float64(r)+float64(o))/2 == s && (res > 0) == (o > r) {
		return o
	}
	return r
}

func ink2pNVF4E2M1RN(x float32) byte {
	var sign byte
	if math.Signbit(float64(x)) {
		sign = 8
	}
	a := math.Abs(float64(x))
	var c byte
	switch {
	case a <= 0.25:
		c = 0
	case a < 0.75:
		c = 1
	case a <= 1.25:
		c = 2
	case a < 1.75:
		c = 3
	case a <= 2.5:
		c = 4
	case a < 3.5:
		c = 5
	case a <= 5:
		c = 6
	default:
		c = 7
	}
	return sign | c
}

// ink2pNVF4E4M3RN is cvt.rn.satfinite.e4m3 for x > 0.
func ink2pNVF4E4M3RN(x float32) byte {
	f := float64(x)
	if f >= 448 {
		return 0x7e
	}
	frac, exp := math.Frexp(f)
	e := exp - 1
	if e < -6 {
		return byte(math.RoundToEven(f * 512))
	}
	m := math.RoundToEven((2*frac - 1) * 8)
	if m == 8 {
		e, m = e+1, 0
	}
	if code := (e+7)<<3 | int(m); code < 0x7e {
		return byte(code)
	}
	return 0x7e
}

func ink2pNVF4Err(v *[16]float32, sc float32) float32 {
	var inv float32
	if sc > 0 {
		inv = 1 / sc
	}
	var err float32
	for k := 0; k < 16; k++ {
		q := ink2pNVF4E2M1RN(v[k] * inv)
		e := float32(float32(math.Abs(float64(v[k]))) - float32(float32(math.Abs(float64(mxE2M1Values[q])))*sc))
		err = ink2pFMA32(e, e, err)
	}
	return err
}

// ink2pNVF4QuantRow returns the packed E2M1 pairs (low nibble = even column), the
// UE4M3 code per 16 columns, and the fp32 row scale.
func ink2pNVF4QuantRow(x []float32) (q, codes []byte, rs float32) {
	var amax float32
	for _, v := range x {
		amax = max(amax, float32(math.Abs(float64(v))))
	}
	rs = amax / float32(6*448)
	var ic float32
	if rs > 0 {
		ic = 1 / rs
	}
	q = make([]byte, len(x)/2)
	codes = make([]byte, len(x)/16)
	for b := range codes {
		vals := x[b*16 : (b+1)*16]
		var v [16]float32
		var a16 float32
		for k := range v {
			v[k] = vals[k] * ic
			a16 = max(a16, float32(math.Abs(float64(v[k]))))
		}
		a6 := a16 / 6
		first := 0
		if a6 > 0 {
			first = int(ink2pNVF4E4M3RN(a6))
		}
		code, sc := first, DecodeE4M3(byte(first))
		best := ink2pNVF4Err(&v, sc)
		for _, off := range []int{-1, 1, -2, 2} {
			tc := first + off
			if tc < 0 || tc > 0x7e {
				continue
			}
			tsc := DecodeE4M3(byte(tc))
			if cur := ink2pNVF4Err(&v, tsc); cur < best {
				best, code, sc = cur, tc, tsc
			}
		}
		var inv float32
		if sc > 0 {
			inv = 1 / sc
		}
		s := ic * inv
		for k := 0; k < 16; k += 2 {
			q[b*8+k/2] = ink2pNVF4E2M1RN(vals[k]*s) | ink2pNVF4E2M1RN(vals[k+1]*s)<<4
		}
		codes[b] = byte(code)
	}
	return q, codes, rs
}

// ink2pNVF4Dequant is the value the W4A4 kernel multiplies for column k of a row.
func ink2pNVF4Dequant(q, codes []byte, rs float32, k int) float64 {
	nib := q[k/2] & 0x0f
	if k%2 == 1 {
		nib = q[k/2] >> 4
	}
	return float64(mxE2M1Values[nib]) * float64(DecodeE4M3(codes[k/16])) * float64(rs)
}

// fak-test:runtime fast est=20ms lane=default
// The host quantizer port rounds ties to even in both formats, saturates, keeps
// all-zero blocks at code 0, and reconstructs Gaussian rows to a few percent.
func TestInk2PNVF4QuantizerReference(t *testing.T) {
	for _, c := range []struct {
		x    float32
		want byte
	}{
		{0.25, 0}, {0.2501, 1}, {0.75, 2}, {1.25, 2}, {1.75, 4}, {2.5, 4}, {3.5, 6}, {5, 6}, {5.01, 7}, {100, 7}, {-0.75, 0xa}, {-7, 0xf},
	} {
		if got := ink2pNVF4E2M1RN(c.x); got != c.want {
			t.Fatalf("e2m1(%v) = %#x, want %#x", c.x, got, c.want)
		}
	}
	for code := 1; code <= 0x7e; code++ {
		if got := ink2pNVF4E4M3RN(DecodeE4M3(byte(code))); int(got) != code {
			t.Fatalf("e4m3 round trip of code %#x = %#x", code, got)
		}
	}
	for _, c := range []struct {
		x    float32
		want byte
	}{
		{1.0625, 0x38}, {1.1875, 0x3a}, {float32(math.Ldexp(1, -10)), 0}, {float32(math.Ldexp(3, -11)), 1}, {460, 0x7e}, {1e9, 0x7e},
	} {
		if got := ink2pNVF4E4M3RN(c.x); got != c.want {
			t.Fatalf("e4m3(%v) = %#x, want %#x", c.x, got, c.want)
		}
	}
	r := rand.New(rand.NewSource(3))
	// (1+u)*(2^-24*(1-u)) + (1+u) with u = 2^-23 rounds in float64 onto the midpoint
	// above 1+u, where float32(float64 sum) ties to the wrong neighbour.
	u := math.Ldexp(1, -23)
	checkInk2PFMA32(t, float32(1+u), float32(math.Ldexp(1-u, -24)), float32(1+u))
	checkInk2PFMA32(t, -float32(1+u), float32(math.Ldexp(1-u, -24)), -float32(1+u))
	for i := 0; i < 20000; i++ {
		a, b := float32(r.NormFloat64()), float32(r.NormFloat64())
		checkInk2PFMA32(t, a, b, float32(r.NormFloat64()*math.Ldexp(1, r.Intn(40)-20)))
	}
	const in = 256
	x := make([]float32, in)
	for i := range x {
		x[i] = float32(r.NormFloat64())
	}
	for i := 32; i < 48; i++ {
		x[i] = 0
	}
	q, codes, rs := ink2pNVF4QuantRow(x)
	if codes[2] != 0 || q[16] != 0 || q[23] != 0 {
		t.Fatalf("zero block quantized to code %#x nibbles %x", codes[2], q[16:24])
	}
	var num, den float64
	for k, v := range x {
		d := ink2pNVF4Dequant(q, codes, rs, k) - float64(v)
		num += d * d
		den += float64(v) * float64(v)
	}
	if rel := math.Sqrt(num / den); rel > 0.12 {
		t.Fatalf("relative reconstruction error %.4f", rel)
	}
}

func checkInk2PFMA32(t *testing.T, a, b, c float32) {
	t.Helper()
	exact := new(big.Float).SetPrec(2000).SetFloat64(float64(a))
	exact.Mul(exact, new(big.Float).SetFloat64(float64(b)))
	exact.Add(exact, new(big.Float).SetFloat64(float64(c)))
	want, _ := exact.Float32()
	if got := ink2pFMA32(a, b, c); math.Float32bits(got) != math.Float32bits(want) {
		t.Fatalf("fma32(%v, %v, %v) = %v, want %v", a, b, c, got, want)
	}
}

package compute

import (
	"bytes"
	"math"
	"math/rand"
	"testing"
)

// The routed-expert acceptance predicates (checkGemv*) replace per-byte decodes with
// IndexByte scans. These witnesses pin the scans to the decoders they stand in for, over
// every byte value, so a device kernel gated by the predicate accepts exactly what the
// scalar oracle accepted before the predicate existed.

// fak-test:runtime fast est=1ms lane=default
func TestE4M3NaNByteIsExactlyNonFiniteDecode(t *testing.T) {
	for v := 0; v < 256; v++ {
		b := byte(v)
		d := float64(DecodeE4M3(b))
		if got, want := e4m3NaNByte(b), math.IsNaN(d) || math.IsInf(d, 0); got != want {
			t.Fatalf("byte %#02x: e4m3NaNByte=%v, DecodeE4M3 non-finite=%v", b, got, want)
		}
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestE8M0ScaleRefusalIsExactlyByteFF(t *testing.T) {
	for v := 0; v < 256; v++ {
		b := byte(v)
		_, err := decodeE8M0Scale(b)
		if got, want := err != nil, b == 0xff; got != want {
			t.Fatalf("byte %#02x: decodeE8M0Scale refused=%v, want %v", b, got, want)
		}
		if gotScan := firstE8M0ScaleError([]byte{127, b}) != nil; gotScan != (b == 0xff) {
			t.Fatalf("byte %#02x: firstE8M0ScaleError refused=%v", b, gotScan)
		}
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestFirstE4M3NaNByteMatchesLinearDecodeScan(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	for trial := 0; trial < 500; trial++ {
		w := make([]byte, 1+rng.Intn(300))
		for k := range w {
			w[k] = byte(rng.Intn(0x7f)) | byte(rng.Intn(2))<<7 // finite bytes only
		}
		for n := rng.Intn(3); n > 0; n-- {
			w[rng.Intn(len(w))] = []byte{0x7f, 0xff}[rng.Intn(2)]
		}
		want := -1
		for k, b := range w {
			d := float64(DecodeE4M3(b))
			if math.IsNaN(d) || math.IsInf(d, 0) {
				want = k
				break
			}
		}
		if got := firstE4M3NaNByte(w); got != want {
			t.Fatalf("trial %d: firstE4M3NaNByte=%d, linear decode scan=%d", trial, got, want)
		}
	}
}

// e8m0FiniteByteO and e8m0FiniteByteI shape one per-byte routed-expert GEMV: one FP8 row
// tile two scale tiles wide, or 32 MXFP4 rows of two scale groups.
const e8m0FiniteByteO, e8m0FiniteByteI = FP8Block32Dim, 2 * FP8Block32Dim

// e8m0FiniteByteCase builds a GEMV whose every E8M0 scale is b and whose every product is
// finite and normal, so a device can be held bit-for-bit to the oracle at EVERY byte,
// including 250..254 where random weights overflow. Weight magnitudes stay below 2, so
// w = d*2^(b-127) is finite at b=254 and exact at b=0; x = +-(1+k/16)*2^(127-b-e), e in
// 0..4, is exact from just under 2^128 down to the 2^-135 subnormal, so each w*x rounds
// the b-independent product d*(1+k/16)*2^-e.
func e8m0FiniteByteCase(tb testing.TB, b byte, mxfp4 bool) (weight, scales []byte, x []float32) {
	tb.Helper()
	const O, I = e8m0FiniteByteO, e8m0FiniteByteI
	seed := int64(b) << 1
	if mxfp4 {
		seed |= 1
	}
	rng := rand.New(rand.NewSource(seed))
	x = make([]float32, I)
	for i := range x {
		v := math.Ldexp(1+float64(rng.Intn(16))/16, 127-int(b)-rng.Intn(5))
		if rng.Intn(2) == 0 {
			v = -v
		}
		if x[i] = float32(v); float64(x[i]) != v {
			tb.Fatalf("byte %#02x: x[%d]=%g is not exactly representable", b, i, v)
		}
	}
	if mxfp4 {
		// Nibbles 0..3 and 8..11: |E2M1| in {0, 0.5, 1, 1.5}.
		nib := func() byte { return byte(rng.Intn(4) | rng.Intn(2)<<3) }
		weight = make([]byte, O*I/2)
		for k := range weight {
			weight[k] = nib() | nib()<<4
		}
		return weight, bytes.Repeat([]byte{b}, O*((I+FP8Block32Dim-1)/FP8Block32Dim)), x
	}
	// E4M3 exponent field 0..7 with either sign: |d| <= 1.875.
	weight = make([]byte, O*I)
	for k := range weight {
		weight[k] = byte(rng.Intn(0x40)) | byte(rng.Intn(2))<<7
	}
	sO, sI := fp8Block32ScaleShape(O, I)
	return weight, bytes.Repeat([]byte{b}, sO*sI), x
}

// TestRoutedExpertE8M0FiniteByteFixtureDiscriminatesEveryByte proves, GPU-free, that the
// per-byte fixture the CUDA suite holds bit-for-bit is not vacuous: every product is zero
// or a finite normal, every output is finite with at least one nonzero, and decoding b as
// its neighbour (254 clamped to 253 or read as 2^126, byte 0 read as 2^-126) changes the
// output bits.
//
// fak-test:runtime fast est=20ms lane=default
func TestRoutedExpertE8M0FiniteByteFixtureDiscriminatesEveryByte(t *testing.T) {
	const O, I = e8m0FiniteByteO, e8m0FiniteByteI
	for v := 0; v <= 254; v++ {
		b := byte(v)
		s, err := decodeE8M0Scale(b)
		if err != nil {
			t.Fatalf("byte %#02x: %v", b, err)
		}
		neighbour := b - 1
		if b == 0 {
			neighbour = 1
		}
		for _, mxfp4 := range []bool{false, true} {
			weight, scales, x := e8m0FiniteByteCase(t, b, mxfp4)
			gemv, format := GemvFP8Block32E8M0, "fp8"
			if mxfp4 {
				gemv, format = GemvMXFP4E8M0, "mxfp4"
			}
			for o := 0; o < O; o++ {
				for i := 0; i < I; i++ {
					var d float32
					if mxfp4 {
						nib := weight[o*I/2+i/2] >> (4 * (i & 1)) & 0x0f
						d = mxE2M1Values[nib]
					} else {
						d = DecodeE4M3(weight[o*I+i])
					}
					w := d * s
					p := math.Float32bits(float32(w * x[i]))
					if exp := p >> 23 & 0xff; p&0x7fffffff != 0 && (exp == 0 || exp == 0xff) {
						t.Fatalf("%s byte %#02x: product [%d,%d] bits %#08x is not a finite normal", format, b, o, i, p)
					}
				}
			}
			y, err := gemv("oracle", O, I, weight, scales, x)
			if err != nil {
				t.Fatalf("%s byte %#02x: %v", format, b, err)
			}
			nonzero := false
			for k, yk := range y {
				if math.IsInf(float64(yk), 0) || math.IsNaN(float64(yk)) {
					t.Fatalf("%s byte %#02x: y[%d]=%g is not finite", format, b, k, yk)
				}
				nonzero = nonzero || yk != 0
			}
			if !nonzero {
				t.Fatalf("%s byte %#02x: every output is zero", format, b)
			}
			alt, err := gemv("oracle", O, I, weight, bytes.Repeat([]byte{neighbour}, len(scales)), x)
			if err != nil {
				t.Fatalf("%s byte %#02x: neighbour %#02x: %v", format, b, neighbour, err)
			}
			differs := false
			for k := range y {
				differs = differs || math.Float32bits(y[k]) != math.Float32bits(alt[k])
			}
			if !differs {
				t.Fatalf("%s byte %#02x: decoding it as %#02x leaves every output bit unchanged", format, b, neighbour)
			}
		}
	}
}

// TestRoutedExpertResidentAcceptanceSplitsOracle holds the resident split to the oracle: the
// upload predicate (check*Weight) plus the per-GEMV x clause accepts exactly the inputs the
// scalar oracle accepts, over every combination of one malformed or well-formed variant per
// operand. A resident weight therefore widens or narrows nothing the byte-in/byte-out path
// refuses.
//
// fak-test:runtime fast est=20ms lane=default
func TestRoutedExpertResidentAcceptanceSplitsOracle(t *testing.T) {
	const O, I = 40, 96
	rng := rand.New(rand.NewSource(5))
	finite := func(n int) []byte {
		b := make([]byte, n)
		for k := range b {
			b[k] = byte(rng.Intn(0x7f)) | byte(rng.Intn(2))<<7
		}
		return b
	}
	set := func(b []byte, k int, v byte) []byte {
		c := append([]byte(nil), b...)
		c[k] = v
		return c
	}
	x := make([]float32, I)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	xs := [][]float32{x, x[:I-1], append(append([]float32(nil), x...), 1), nil}
	for _, format := range []string{"fp8", "mxfp4"} {
		var w, s []byte
		var oracle func(string, int, int, []byte, []byte, []float32) ([]float32, error)
		var weightOK func(string, int, int, []byte, []byte) (int, error)
		prefix := "fp8 block32"
		if format == "fp8" {
			sO, sI := fp8Block32ScaleShape(O, I)
			w, s = finite(O*I), bytes.Repeat([]byte{127}, sO*sI)
			oracle, weightOK = GemvFP8Block32E8M0, checkFP8Block32E8M0Weight
		} else {
			w, s = finite(O*I/2), bytes.Repeat([]byte{127}, O*((I+31)/32))
			oracle, weightOK = GemvMXFP4E8M0, checkMXFP4E8M0Weight
			prefix = "mxfp4"
		}
		ws := [][]byte{w, w[:len(w)-1], append(append([]byte(nil), w...), 0), set(w, 7, 0x7f), set(w, len(w)-1, 0xff)}
		ss := [][]byte{s, s[:len(s)-1], append(append([]byte(nil), s...), 127), set(s, 0, 0xff), set(s, len(s)-1, 0xff)}
		var accepted, xOnlyRefused, refused int
		for _, o := range []int{O, 0, -1, O + 1} {
			for _, i := range []int{I, 0, -I, I + 1} {
				for _, wv := range ws {
					for _, sv := range ss {
						for _, xv := range xs {
							_, oerr := oracle("oracle", o, i, wv, sv, xv)
							_, uerr := weightOK("upload", o, i, wv, sv)
							xerr := routedExpertXError(prefix, "gemv", o, i, xv)
							if (oerr == nil) != (uerr == nil && xerr == nil) {
								t.Fatalf("%s [%d %d] len(w)=%d len(s)=%d len(x)=%d: oracle err=%v, upload err=%v, x err=%v",
									format, o, i, len(wv), len(sv), len(xv), oerr, uerr, xerr)
							}
							switch {
							case oerr == nil:
								accepted++
							case uerr == nil:
								xOnlyRefused++
							default:
								refused++
							}
						}
					}
				}
			}
		}
		if accepted == 0 || xOnlyRefused == 0 || refused == 0 {
			t.Fatalf("%s: vacuous table: accepted=%d x-only refused=%d refused=%d", format, accepted, xOnlyRefused, refused)
		}
	}
}

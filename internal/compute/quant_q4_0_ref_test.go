package compute

import (
	"bytes"
	"math"
	"math/rand"
	"testing"
)

// The oracles below are transcribed from ggml-org/llama.cpp at commit
// authors): quantize_row_q4_0_ref in ggml/src/ggml-quants.c and the portable
// ggml_compute_fp32_to_fp16 in ggml/src/ggml-impl.h. They share no code with the port.

// ggmlFP32ToFP16Oracle is the float-arithmetic fp16 narrowing ggml falls back to
// without F16C; it rounds to nearest even.
func ggmlFP32ToFP16Oracle(f float32) uint16 {
	const scaleToInf = float32(0x1p+112)
	const scaleToZero = float32(0x1p-110)
	abs := math.Float32frombits(math.Float32bits(f) &^ 0x80000000)
	base := float32(float32(abs*scaleToInf) * scaleToZero)

	w := math.Float32bits(f)
	shl1W := w + w
	sign := w & 0x80000000
	bias := shl1W & 0xFF000000
	if bias < 0x71000000 {
		bias = 0x71000000
	}

	base = float32(math.Float32frombits((bias>>1)+0x07800000) + base)
	bits := math.Float32bits(base)
	expBits := (bits >> 13) & 0x00007C00
	mantissaBits := bits & 0x00000FFF
	nonsign := expBits + mantissaBits
	if shl1W > 0xFF000000 {
		return uint16(sign>>16) | 0x7E00
	}
	return uint16(sign>>16) | uint16(nonsign)
}

func ggmlQuantizeRowQ4_0RefOracle(x []float32) []byte {
	const qk = 32
	nb := len(x) / qk
	y := make([]byte, nb*18)
	for i := 0; i < nb; i++ {
		amax := float32(0)
		max := float32(0)
		for j := 0; j < qk; j++ {
			v := x[i*qk+j]
			av := float32(math.Abs(float64(v)))
			if amax < av {
				amax = av
				max = v
			}
		}

		d := max / -8
		id := float32(0)
		if d != 0 {
			id = 1.0 / d
		}

		h := ggmlFP32ToFP16Oracle(d)
		y[i*18] = byte(h)
		y[i*18+1] = byte(h >> 8)

		for j := 0; j < qk/2; j++ {
			x0 := float32(x[i*qk+0+j] * id)
			x1 := float32(x[i*qk+qk/2+j] * id)

			xi0 := int8(int32(float32(x0 + 8.5)))
			if xi0 > 15 {
				xi0 = 15
			}
			xi1 := int8(int32(float32(x1 + 8.5)))
			if xi1 > 15 {
				xi1 = 15
			}

			y[i*18+2+j] = uint8(xi0)
			y[i*18+2+j] |= uint8(xi1) << 4
		}
	}
	return y
}

// Golden blocks are hand-derived from the C reference in float32: d = max/-8,
// id = 1/d, code = min(15, trunc(x*id + 8.5)), byte j = code[j] | code[j+16]<<4.
// Inputs sit away from truncation boundaries except where the sum is exact.
var q4_0RefGoldens = []struct {
	name string
	in   [32]float32
	want [18]byte
}{
	{
		// d = 1 (fp16 0x3C00), so code = min(15, trunc(x + 8.5)).
		name: "negative max magnitude",
		in: [32]float32{
			-8, -7.6, -7.5, -7, -6.25, -5, -4.5, -3, -2, -1, -0.51, -0.5, 0, 0.49, 0.5, 1,
			2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.4, 6.5, 7, 7.4, 7.5, 7.75, 3.25,
		},
		want: [18]byte{
			0x00, 0x3C,
			0xA0, 0xB0, 0xB1, 0xC1, 0xC2, 0xD3, 0xD4, 0xE5,
			0xE6, 0xE7, 0xF7, 0xF8, 0xF8, 0xF8, 0xF9, 0xB9,
		},
	},
	{
		// The first of the tied extremes wins, so max = +4 and d = -0.5 (fp16
		// 0xB800): code = min(15, trunc(-2x + 8.5)), the positive peak lands on 0.
		name: "positive max magnitude with a later negative tie",
		in: [32]float32{
			4, 3.9, 3.75, 3, 2.5, 2, 1, 0.3, 0.25, 0, -0.25, -1, -2, -3, -3.75, -4,
			0.5, -0.5, 1.5, -1.5, 3.5, -3.5, 0.125, -0.125, 2.25, -2.25, 1.75, -1.75, 0.75, -0.75, 3.25, -3.25,
		},
		want: [18]byte{
			0x00, 0xB8,
			0x70, 0x90, 0x51, 0xB2, 0x13, 0xF4, 0x86, 0x87,
			0x48, 0xD8, 0x59, 0xCA, 0x7C, 0xAE, 0x2F, 0xFF,
		},
	},
	{
		// d = float32(0.3): its mantissa rounds up to fp16 0x34CD, where a
		// truncating narrowing would store 0x34CC.
		name: "scale that rounds up in fp16",
		in: [32]float32{
			-2.4, 0, 0.3, 1.2, -1.2, 2.39, 0.14, 0.16, -0.3, 0.6, -0.6, 0.9, -0.9, 1.8, -1.8, 2.0,
			2.1, -2.1, 1.5, -1.5, 0.4, -0.4, 1.0, -1.0, 0.7, -0.7, 2.2, -2.2, 0.05, -0.05, 1.7, -1.7,
		},
		want: [18]byte{
			0xCD, 0x34,
			0xF0, 0x18, 0xD9, 0x3C, 0x94, 0x7F, 0xB8, 0x59,
			0xA7, 0x6A, 0xF6, 0x1B, 0x85, 0x8E, 0xE2, 0x2F,
		},
	},
	{
		// max stays +0, so d = 0/-8 = -0 (fp16 0x8000), id = 0, every code is 8.
		name: "all zero",
		want: [18]byte{
			0x00, 0x80,
			0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88,
			0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88,
		},
	},
}

func TestQuantizeF32ToQ4_0RefMatchesGGML(t *testing.T) {
	var all []float32
	var allWant []byte
	for _, g := range q4_0RefGoldens {
		got, err := QuantizeF32ToQ4_0Ref(g.in[:])
		if err != nil {
			t.Fatalf("%s: %v", g.name, err)
		}
		if !bytes.Equal(got, g.want[:]) {
			t.Errorf("%s: block bytes\n got % X\nwant % X", g.name, got, g.want[:])
		}
		if oracle := ggmlQuantizeRowQ4_0RefOracle(g.in[:]); !bytes.Equal(oracle, g.want[:]) {
			t.Errorf("%s: transcribed C oracle disagrees with the hand-derived golden\n got % X\nwant % X", g.name, oracle, g.want[:])
		}
		all = append(all, g.in[:]...)
		allWant = append(allWant, g.want[:]...)
	}

	got, err := QuantizeF32ToQ4_0Ref(all)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, allWant) {
		t.Errorf("multi-block row\n got % X\nwant % X", got, allWant)
	}

	rng := rand.New(rand.NewSource(0x51a0))
	row := make([]float32, 64*32)
	for i := range row {
		scale := float32(math.Pow(10, float64(rng.Intn(9)-5)))
		row[i] = float32(rng.NormFloat64()) * scale
	}
	got, err = QuantizeF32ToQ4_0Ref(row)
	if err != nil {
		t.Fatal(err)
	}
	if want := ggmlQuantizeRowQ4_0RefOracle(row); !bytes.Equal(got, want) {
		for b := 0; b < len(row)/32; b++ {
			if !bytes.Equal(got[b*18:(b+1)*18], want[b*18:(b+1)*18]) {
				t.Fatalf("random block %d\n got % X\nwant % X", b, got[b*18:(b+1)*18], want[b*18:(b+1)*18])
			}
		}
	}

	if _, err := QuantizeF32ToQ4_0Ref(make([]float32, 33)); err == nil {
		t.Error("a partial block was accepted; the reference asserts k % 32 == 0")
	}
	if _, err := QuantizeF32ToQ4_0Ref(nil); err == nil {
		t.Error("empty input was accepted")
	}
}

func TestQuantizeF32ToQ4_0RefMatchesGGMLRoundTrip(t *testing.T) {
	for _, g := range q4_0RefGoldens {
		blk, err := QuantizeF32ToQ4_0Ref(g.in[:])
		if err != nil {
			t.Fatalf("%s: %v", g.name, err)
		}
		dec := make([]float32, 32)
		if err := DequantizeQ4_0(dec, blk, 32); err != nil {
			t.Fatalf("%s: %v", g.name, err)
		}

		d := math.Abs(float64(Float16BitsToFloat32(uint16(blk[0]) | uint16(blk[1])<<8)))
		// One full step covers the clamp of the extreme opposite the peak (code
		// 16 -> 15); the rest is half a step plus the fp16 error on |code-8| <= 8.
		tol := d * (1 + 8.0/1024)
		for j := 0; j < 16; j++ {
			// DequantizeQ4_0 emits byte j's nibbles as an adjacent pair.
			for _, p := range [2]struct{ src, dec int }{{j, 2 * j}, {j + 16, 2*j + 1}} {
				if diff := math.Abs(float64(g.in[p.src]) - float64(dec[p.dec])); diff > tol {
					t.Errorf("%s: element %d = %g decoded %g, |diff| %g > %g", g.name, p.src, g.in[p.src], dec[p.dec], diff, tol)
				}
			}
		}
	}
}

func TestQuantizeF32ToQ4_0RefMatchesGGMLFP16(t *testing.T) {
	check := func(bits uint32) {
		f := math.Float32frombits(bits)
		if got, want := float32ToFloat16BitsNearestEven(f), ggmlFP32ToFP16Oracle(f); got != want {
			t.Fatalf("fp16(%g, bits %#08x) = %#04x, ggml %#04x", f, bits, got, want)
		}
	}
	// Every float32 exponent, with mantissas on and around the fp16 rounding ties
	// for both the normal (13 dropped bits) and subnormal ranges.
	mantissas := []uint32{
		0, 1, 0x0FFF, 0x1000, 0x1001, 0x2000, 0x2FFF, 0x3000, 0x3001,
		0x19999A, 0x400000, 0x7FEFFF, 0x7FF000, 0x7FF001, 0x7FFFFF,
	}
	for exp := uint32(0); exp < 256; exp++ {
		for _, m := range mantissas {
			check(exp<<23 | m)
			check(0x80000000 | exp<<23 | m)
		}
	}
	rng := rand.New(rand.NewSource(16))
	for i := 0; i < 200000; i++ {
		check(rng.Uint32())
	}
}

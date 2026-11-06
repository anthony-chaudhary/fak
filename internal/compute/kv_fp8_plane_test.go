package compute

import (
	"errors"
	"math"
	"testing"
)

// fak-test:runtime fast est=10ms lane=default
func TestKVFP8PlaneCodec(t *testing.T) {
	t.Run("all_finite_codes_and_midpoints", func(t *testing.T) {
		for sign := byte(0); ; sign = 0x80 {
			for code := byte(0); code <= 0x7e; code++ {
				value := DecodeE4M3(code | sign)
				p, err := QuantizeKVFP8Plane([]float32{448, value}, 1, 1, 2)
				if err != nil || p.Codes[1] != code|sign || p.Scales[0] != 1 {
					t.Fatalf("code=%02x sign=%02x: plane=%+v err=%v", code, sign, p, err)
				}
				if code == 0x7e {
					continue
				}
				mid := float32((DecodeE4M3(code) + DecodeE4M3(code+1)) / 2)
				if sign != 0 {
					mid = -mid
				}
				want := code
				if code&1 != 0 {
					want++
				}
				p, err = QuantizeKVFP8Plane([]float32{448, mid}, 1, 1, 2)
				if err != nil || p.Codes[1] != want|sign {
					t.Fatalf("midpoint code=%02x sign=%02x: got=%+v err=%v", code, sign, p, err)
				}
			}
			if sign != 0 {
				break
			}
		}
	})
	t.Run("exact_division_not_rounded_reciprocal", func(t *testing.T) {
		// 3136/448=7 and 12.6875/7=1.8125 exactly. The latter is
		// the E4M3 midpoint between even 0x3e and odd 0x3f.
		// F32(1/7)=9586981/2^26; multiplying adds 87/2^30 to
		// that midpoint, beyond half an F32 ulp (64/2^30).
		for _, hd := range []int{65, 256} {
			src := make([]float32, hd)
			src[0], src[1], src[hd-1] = 3136, 12.6875, -12.6875
			p, err := QuantizeKVFP8Plane(src, 1, 1, hd)
			if err != nil || len(p.Codes) != hd || len(p.Scales) != 1 {
				t.Fatalf("hd=%d: %+v %v", hd, p, err)
			}
			if math.Float32bits(p.Scales[0]) != 0x40e00000 ||
				p.Codes[0] != 0x7e || p.Codes[1] != 0x3e || p.Codes[hd-1] != 0xbe {
				t.Fatalf("hd=%d: division midpoint changed: %+v", hd, p)
			}
			// Negative control: this fixture must distinguish the shortcut,
			// not merely agree with another calculation of the same division.
			reciprocal := float32(1 / p.Scales[0])
			if math.Float32bits(reciprocal) != 0x3e124925 ||
				rfqEncodeE4M3(float32(src[1]*reciprocal)) != 0x3f ||
				rfqEncodeE4M3(float32(src[hd-1]*reciprocal)) != 0xbf {
				t.Fatalf("hd=%d: reciprocal negative control no longer discriminates", hd)
			}
		}
	})
	t.Run("zero_and_underflow", func(t *testing.T) {
		negativeZero := math.Float32frombits(0x80000000)
		p, err := QuantizeKVFP8Plane([]float32{0, negativeZero}, 1, 1, 2)
		if err != nil || p.Scales[0] != 1 || p.Codes[0] != 0 || p.Codes[1] != 0x80 {
			t.Fatalf("signed zero: %+v %v", p, err)
		}
		p, err = QuantizeKVFP8Plane([]float32{math.SmallestNonzeroFloat32, -math.SmallestNonzeroFloat32}, 1, 1, 2)
		if err != nil || math.Float32bits(p.Scales[0]) != 1 || p.Codes[0] != 0x38 || p.Codes[1] != 0xb8 {
			t.Fatalf("scale underflow: %+v %v", p, err)
		}
	})
	t.Run("independent_heads_and_positions", func(t *testing.T) {
		// Four differently scaled rows: positions2 x heads2 x headDim2.
		p, err := QuantizeKVFP8Plane([]float32{448, 1, 896, 2, 1792, 4, 3584, 8}, 2, 2, 2)
		if err != nil {
			t.Fatal(err)
		}
		for row, want := range []float32{1, 2, 4, 8} {
			if p.Scales[row] != want || p.Codes[2*row] != 0x7e || p.Codes[2*row+1] != 0x38 {
				t.Fatalf("row%d: %+v", row, p)
			}
		}
	})
	t.Run("scale_boundaries", func(t *testing.T) {
		for _, bits := range []uint32{0x3affffff, 0x3b000000, 0x3b000001, 0x04dfffff, 0x04e00000, 0x04e00001, 0x7f7fffff} {
			x := math.Float32frombits(bits)
			p, err := QuantizeKVFP8Plane([]float32{x, -x}, 1, 1, 2)
			want := float32(x / float32(448))
			if err != nil || math.Float32bits(p.Scales[0]) != math.Float32bits(want) {
				t.Fatalf("amax=%08x: %+v %v want scale bits %08x", bits, p, err, math.Float32bits(want))
			}
			if _, err := DequantizeKVFP8Plane(p); err != nil {
				t.Fatal(err)
			}
		}
	})
	t.Run("f32_widen_not_half_scaled", func(t *testing.T) {
		p := KVFP8Plane{Positions: 2, Heads: 1, HeadDim: 1,
			Codes: []byte{0x7e, 0x38}, Scales: []float32{256, math.Float32frombits(0x3f800800)}}
		got, err := DequantizeKVFP8Plane(p)
		if err != nil || math.Float32bits(got[0]) != 0x47e00000 || math.Float32bits(got[1]) != 0x3f800800 {
			t.Fatalf("scaled widen: %v %v", got, err)
		}
	})
	t.Run("refusal_and_empty", func(t *testing.T) {
		for _, bits := range []uint32{0x7f800000, 0xff800000, 0x7fc00000} {
			p, err := QuantizeKVFP8Plane([]float32{1, math.Float32frombits(bits)}, 1, 1, 2)
			if !errors.Is(err, ErrKVFP8NonFinite) || p.Codes != nil || p.Scales != nil {
				t.Fatalf("nonfinite=%08x: %+v %v", bits, p, err)
			}
		}
		for _, shape := range [][3]int{{-1, 1, 1}, {1, 0, 1}, {1, 1, 0}, {1, 1 << 30, 4}, {1 << 30, 2, 1}} {
			if _, err := QuantizeKVFP8Plane(nil, shape[0], shape[1], shape[2]); !errors.Is(err, ErrKVFP8Geometry) {
				t.Fatalf("shape%v: %v", shape, err)
			}
		}
		p, err := QuantizeKVFP8Plane(nil, 0, 4, 256)
		if err != nil || p.Codes != nil || p.Scales != nil {
			t.Fatalf("empty: %+v %v", p, err)
		}
		for _, p := range []KVFP8Plane{
			{Positions: 1, Heads: 1, HeadDim: 1, Codes: []byte{0x38}, Scales: []float32{0}},
			{Positions: 1, Heads: 1, HeadDim: 1, Codes: []byte{0x7f}, Scales: []float32{1}},
			{Positions: 1, Heads: 1, HeadDim: 1, Codes: []byte{0x7e}, Scales: []float32{math.MaxFloat32}},
		} {
			got, err := DequantizeKVFP8Plane(p)
			if err == nil || got != nil {
				t.Fatalf("invalid plane accepted: %+v -> %v %v", p, got, err)
			}
		}
	})
}

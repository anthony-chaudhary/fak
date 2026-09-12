//go:build darwin && arm64 && cgo

package metalgemm

import (
	"math"
	"testing"
)

func v41BF16(x float32) float32 {
	bits := math.Float32bits(x)
	if bits&0x7f800000 != 0x7f800000 {
		bits += 0x7fff + ((bits >> 16) & 1)
	}
	return math.Float32frombits(bits & 0xffff0000)
}

func v41RoPEOracle(values []float32, width, heads, rows, start, stride int, compressed, inverse bool) {
	base := float32(10000)
	if compressed {
		base = 160000
	}
	low := float32(math.Floor(64 * math.Log(65536/(32*2*math.Pi)) / (2 * math.Log(float64(base)))))
	high := float32(math.Ceil(64 * math.Log(65536/(2*math.Pi)) / (2 * math.Log(float64(base)))))
	for row := 0; row < rows; row++ {
		for head := 0; head < heads; head++ {
			for lane := 0; lane < 32; lane++ {
				f := float32(1) / float32(math.Pow(float64(base), float64(float32(lane)/32)))
				if compressed {
					ramp := min(float32(1), max(float32(0), (float32(lane)-low)/(high-low)))
					smooth := 1 - ramp
					f = (f/16)*(1-smooth) + f*smooth
				}
				theta := float32(start+row*stride) * f
				c, s := float32(math.Cos(float64(theta))), float32(math.Sin(float64(theta)))
				if inverse {
					s = -s
				}
				i := (row*heads+head)*width + width - 64 + 2*lane
				re, im := values[i], values[i+1]
				values[i] = v41BF16(re*c - im*s)
				values[i+1] = v41BF16(re*s + im*c)
			}
		}
	}
}

func TestV41RoPEMetalOracle(t *testing.T) {
	if !Available() {
		t.Fatal("physical Metal device required; skip is not a pass")
	}
	t.Logf("physical_device=%s", DeviceName())
	const width, heads, rows, stride = 96, 3, 4, 2
	for _, tc := range []struct {
		name       string
		start      int
		compressed bool
		inverse    bool
	}{
		{"regular", 37, false, false}, {"long-compressed-inverse", 999991, true, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			input := make([]float32, width*heads*rows)
			for i := range input {
				input[i] = float32(math.Sin(float64(i)*0.017) * 0.75)
			}
			want := append([]float32(nil), input...)
			v41RoPEOracle(want, width, heads, rows, tc.start, stride, tc.compressed, tc.inverse)
			got := append([]float32(nil), input...)
			receipt, err := DeepSeekV41RoPE(got, width, heads, rows, tc.start, stride, tc.compressed, tc.inverse)
			if err != nil {
				t.Fatal(err)
			}
			if receipt.Engine != "fak-native" || receipt.Backend != "metal" || receipt.CommandBuffers != 1 || receipt.CPUFallbacks != 0 {
				t.Fatalf("invalid execution receipt: %+v", receipt)
			}
			for i := range got {
				if got[i] != want[i] {
					t.Fatalf("element %d = %08x, want %08x", i, math.Float32bits(got[i]), math.Float32bits(want[i]))
				}
			}
			for row := 0; row < rows; row++ {
				for head := 0; head < heads; head++ {
					for lane := 0; lane < 32; lane++ {
						i := (row*heads+head)*width + width - 64 + 2*lane
						before := math.Hypot(float64(input[i]), float64(input[i+1]))
						after := math.Hypot(float64(got[i]), float64(got[i+1]))
						if math.Abs(after-before) > 0.006 {
							t.Fatalf("unit magnitude drift at row/head/lane %d/%d/%d: %.8f", row, head, lane, after-before)
						}
					}
				}
			}
			t.Logf("receipt engine=%s backend=%s command_buffers=%d cpu_fallbacks=%d", receipt.Engine, receipt.Backend, receipt.CommandBuffers, receipt.CPUFallbacks)
		})
	}

	for _, tc := range []struct {
		name                              string
		width, heads, rows, start, stride int
		values                            []float32
	}{
		{"start-wrap", 64, 1, 1, 1 << 32, 1, make([]float32, 64)},
		{"last-position", 64, 1, 2, 1048575, 1, make([]float32, 128)},
		{"stride-wrap", 64, 1, 1, 0, 1 << 32, make([]float32, 64)},
		{"empty", 64, 1, 1, 0, 1, nil},
		{"length-product", math.MaxInt, 2, 1, 0, 1, make([]float32, 1)},
	} {
		t.Run("reject-"+tc.name, func(t *testing.T) {
			if _, err := DeepSeekV41RoPE(tc.values, tc.width, tc.heads, tc.rows, tc.start, tc.stride, false, false); err == nil {
				t.Fatal("malformed geometry was admitted")
			}
		})
	}
}

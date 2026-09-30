package compute

import (
	"fmt"
	"math"
	"testing"
)

// cpuRopeRoundedOracle implements non-interleaved rotation from its scalar
// definition. A product of two float32 operands is exact in float64; conversion
// back to float32 pins the required rounding before either addition. It calls
// neither the production table builder nor its rotation helper. The two FMA
// candidates identify operands that discriminate fused from rounded products.
func cpuRopeRoundedOracle(x []float32, pos, heads, dim int, theta float64) (rounded, fusedLeft, fusedRight []float32) {
	rounded = make([]float32, len(x))
	fusedLeft = make([]float32, len(x))
	fusedRight = make([]float32, len(x))
	half := dim / 2
	for head := 0; head < heads; head++ {
		for j := 0; j < half; j++ {
			frequency := 1 / math.Pow(theta, float64(2*j)/float64(dim))
			angle := float64(pos) * frequency
			c, s := float32(math.Cos(angle)), float32(math.Sin(angle))
			i, k := head*dim+j, head*dim+j+half
			a, b := float64(x[i]), float64(x[k])
			ac, bs := float32(a*float64(c)), float32(b*float64(s))
			bc, as := float32(b*float64(c)), float32(a*float64(s))
			rounded[i] = float32(float64(ac) - float64(bs))
			rounded[k] = float32(float64(bc) + float64(as))
			fusedLeft[i] = float32(math.FMA(a, float64(c), -float64(bs)))
			fusedRight[i] = float32(math.FMA(-b, float64(s), float64(ac)))
			fusedLeft[k] = float32(math.FMA(b, float64(c), float64(as)))
			fusedRight[k] = float32(math.FMA(a, float64(s), float64(bc)))
		}
	}
	return
}

// fak-test:runtime fast est=1ms lane=default
func TestCPURefRoPEUsesExplicitFloat32Products(t *testing.T) {
	const heads, dim = 2, 8
	const theta = 10000
	// Dense mantissas keep products between representable float32 values.
	// Both heads use mixed signs, and real absolute positions supply the
	// trigonometric coefficients; no synthetic rotation table is injected.
	bits := []uint32{
		0x3fa12345, 0xbfa54321, 0x3fc6789a, 0x3f8bcdef,
		0x3ffedcba, 0xbf923456, 0x3fb45678, 0xbfd23456,
		0xbf912345, 0x3fabcdef, 0xbfe54321, 0x3fb6789a,
		0x3fc12345, 0xbf8fedcb, 0xbfa45678, 0x3fe23456,
	}
	x := make([]float32, len(bits))
	for i, b := range bits {
		x[i] = math.Float32frombits(b)
	}
	var backend Backend = &cpuBackend{}
	input := NewF32(backend, []int{heads, dim}, x)
	for _, pos := range []int{0, 1, 17, 1031} {
		t.Run(fmt.Sprintf("position=%d", pos), func(t *testing.T) {
			want, fusedLeft, fusedRight := cpuRopeRoundedOracle(x, pos, heads, dim, theta)
			sensitive := 0
			for i, value := range want {
				if math.Float32bits(value) != math.Float32bits(fusedLeft[i]) || math.Float32bits(value) != math.Float32bits(fusedRight[i]) {
					sensitive++
				}
			}
			if pos != 0 && sensitive == 0 {
				t.Fatal("fixture does not discriminate fused from explicitly rounded products")
			}
			got := backend.Read(backend.RoPE(input, pos, heads, dim, theta))
			if len(got) != len(want) {
				t.Fatalf("rotation length=%d, want %d", len(got), len(want))
			}
			for i, value := range want {
				if math.Float32bits(got[i]) != math.Float32bits(value) {
					t.Fatalf("RoPE[%d] bits=%08x, rounded oracle=%08x (fused-left=%08x fused-right=%08x; sensitive lanes=%d)",
						i, math.Float32bits(got[i]), math.Float32bits(value), math.Float32bits(fusedLeft[i]), math.Float32bits(fusedRight[i]), sensitive)
				}
			}
			for i, value := range x {
				if math.Float32bits(value) != bits[i] {
					t.Fatalf("functional RoPE mutated input %d", i)
				}
			}
		})
	}
}

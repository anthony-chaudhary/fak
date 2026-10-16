//go:build vulkan

package compute

import (
	"math"
	"testing"
)

// This witnesses the software packed-cache contract, without a device dispatch.
// fak-test:runtime fast est=1s
func TestVulkanFP16KVPreservesNaNClass(t *testing.T) {
	inputBits := []uint32{0x7fc00000, 0xff800001, 0x7f800000, 0xff800000, 0x80000000, 0x3f800000}
	input := make([]float32, len(inputBits))
	for i, bits := range inputBits {
		input[i] = math.Float32frombits(bits)
	}
	cache, err := NewVulkanFP16KVContract(1, 1, len(input))
	if err != nil {
		t.Fatal(err)
	}
	if err := cache.Append(input, input); err != nil {
		t.Fatal(err)
	}
	keys, values, err := cache.ReadPosition(0)
	if err != nil {
		t.Fatal(err)
	}
	for _, row := range [][]float32{keys, values} {
		if len(row) != len(input) {
			t.Fatalf("row length %d, want %d", len(row), len(input))
		}
		for i, got := range row {
			want := input[i]
			if math.IsNaN(float64(want)) {
				if !math.IsNaN(float64(got)) || math.Signbit(float64(got)) != math.Signbit(float64(want)) {
					t.Fatalf("slot %d: NaN %#08x became %#08x", i, inputBits[i], math.Float32bits(got))
				}
			} else if math.Float32bits(got) != inputBits[i] {
				t.Fatalf("slot %d: special/finite control %#08x became %#08x", i, inputBits[i], math.Float32bits(got))
			}
		}
	}
	if cache.Positions() != 1 || cache.FallbackCount() != 0 {
		t.Fatal("cache accounting changed")
	}
}

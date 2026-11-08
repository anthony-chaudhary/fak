package compute

import (
	"bytes"
	"math"
	"reflect"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestNVFP4StackMultirowLayout(t *testing.T) {
	const stacks, rows, cols = 3, 3, 32
	packed := make([]byte, stacks*rows*cols/2)
	scales := make([]byte, stacks*rows*cols/NVFP4BlockSize)
	globals := []float32{0.25, 2, 0.5}
	want := make([]float32, stacks*rows*cols)
	// Distinct scales across all row/expert boundaries expose broadcasting
	// and base-offset errors. Enumerating all 16 codes covers signed zero too.
	for expert := 0; expert < stacks; expert++ {
		for row := 0; row < rows; row++ {
			for group := 0; group < cols/NVFP4BlockSize; group++ {
				slot := (expert*rows+row)*2 + group
				rawScale := byte(0x20 + slot)
				scales[slot] = rawScale
				for within := 0; within < NVFP4BlockSize; within++ {
					column := group*NVFP4BlockSize + within
					index := (expert*rows+row)*cols + column
					code := byte((within + row*3 + expert*5 + group) % 16)
					packed[index/2] |= code << (4 * (column % 2))
					// Existing hand-decoding helpers use exponent/mantissa
					// arithmetic rather than production lookup tables.
					want[index] = testNVFP4E2M1(code) * testNVFP4E4M3(rawScale) * globals[expert]
				}
			}
		}
	}
	packedBefore := bytes.Clone(packed)
	scalesBefore := bytes.Clone(scales)
	globalsBefore := append([]float32(nil), globals...)
	got, err := DecodeNVFP4Stacks(stacks, rows, cols, packed, scales, globals)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(want) {
		t.Fatalf("decoded length %d, want %d", len(got), len(want))
	}
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("expert %d row %d column %d: got %08x want %08x", i/(rows*cols), (i/cols)%rows, i%cols, math.Float32bits(got[i]), math.Float32bits(want[i]))
		}
	}
	if !bytes.Equal(packed, packedBefore) || !bytes.Equal(scales, scalesBefore) || !reflect.DeepEqual(globals, globalsBefore) {
		t.Fatal("stack decode mutated input storage")
	}
}

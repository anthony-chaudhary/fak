package compute

import (
	"errors"
	"math"
	"strconv"
	"testing"
)

// fak-test:runtime fast est=1ms
func TestV41CompressorGeometryBounds(t *testing.T) {
	for _, ratio := range []int{2, 4, V41CompressorPoolMaxRatio} {
		maximum := min(math.MaxInt32/ratio, math.MaxInt/(ratio*4))
		for _, width := range []int{1, maximum} {
			if err := ValidateV41CompressorRingAlloc(ratio, width); err != nil {
				t.Fatalf("valid ring ratio=%d width=%d: %v", ratio, width, err)
			}
			operand := Tensor{Dtype: F32, Layout: RowMajor, Shape: []int{ratio, width}}
			if err := validateV41CompressorPool(operand, operand, ratio, width); err != nil {
				t.Fatalf("valid pool ratio=%d width=%d: %v", ratio, width, err)
			}
		}
		if err := ValidateV41CompressorRingAlloc(ratio, maximum+1); !errors.Is(err, ErrV41CompressorRingGeometry) {
			t.Fatalf("ring above bound ratio=%d: %v", ratio, err)
		}
		operand := Tensor{Dtype: F32, Layout: RowMajor, Shape: []int{ratio, maximum + 1}}
		if err := validateV41CompressorPool(operand, operand, ratio, maximum+1); err == nil {
			t.Fatalf("pool above bound ratio=%d admitted", ratio)
		}
	}
	for _, geometry := range [][2]int{{-1, 1}, {0, 1}, {1, 1}, {V41CompressorPoolMaxRatio + 1, 1}, {2, 0}, {2, -1}} {
		ratio, width := geometry[0], geometry[1]
		if err := ValidateV41CompressorRingAlloc(ratio, width); !errors.Is(err, ErrV41CompressorRingGeometry) {
			t.Fatalf("invalid ring geometry %v: %v", geometry, err)
		}
		if err := validateV41CompressorPool(Tensor{}, Tensor{}, ratio, width); err == nil {
			t.Fatalf("invalid pool geometry %v admitted", geometry)
		}
	}
}

// fak-test:runtime fast est=1ms
func TestV41CompressorGeometryRejectsWrappedProduct(t *testing.T) {
	if strconv.IntSize != 64 {
		t.Skip("uint64 product wrap requires a 64-bit int input; host byte bounds covered separately")
	}
	large := uint64(1) << 62
	width := int(large)
	if err := ValidateV41CompressorRingAlloc(4, width); !errors.Is(err, ErrV41CompressorRingGeometry) {
		t.Fatalf("wrapped ring product admitted: %v", err)
	}
	// Before the division guard, both ratio*width and uint64(ratio)*width
	// wrapped to zero, matching a zero-element operand's metadata.
	empty := Tensor{Dtype: F32, Layout: RowMajor, Shape: []int{0}}
	if err := validateV41CompressorPool(empty, empty, 4, width); err == nil {
		t.Fatal("wrapped pool product admitted zero-element operands")
	}
	if err := ValidateV41CompressorRingStore(empty, empty, empty, empty, 0, 4, width); !errors.Is(err, ErrV41CompressorRingGeometry) {
		t.Fatalf("wrapped ring store product admitted: %v", err)
	}
}

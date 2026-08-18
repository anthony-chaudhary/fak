package kquantbits

import "testing"

func TestF16BitsToF32Bits(t *testing.T) {
	for input, want := range map[uint16]uint32{0: 0, 0x3c00: 0x3f800000, 0x7c00: 0x7f800000} {
		if got := F16BitsToF32Bits(input); got != want {
			t.Errorf("F16BitsToF32Bits(%#x) = %#x, want %#x", input, got, want)
		}
	}
}

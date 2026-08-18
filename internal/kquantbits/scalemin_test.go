package kquantbits

import "testing"

func TestScaleMinK4(t *testing.T) {
	q := []byte{1, 2, 3, 4, 5, 6, 7, 8, 0x90, 0xa0, 0xb0, 0xc0}
	if scale, min := ScaleMinK4(0, q); scale != 1 || min != 5 {
		t.Fatalf("pair 0 = (%d,%d)", scale, min)
	}
	if scale, min := ScaleMinK4(4, q); scale != 0 || min != 9 {
		t.Fatalf("pair 4 = (%d,%d)", scale, min)
	}
}

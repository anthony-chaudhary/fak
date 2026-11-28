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

// fak-test:runtime fast est=50ms lane=default
func TestNUQLUTMetaDecode(t *testing.T) {
	meta := NUQCodebookMeta{
		BitWidth:  2,
		Signposts: []float32{-1.5, -0.5, 0.5, 1.5},
	}
	got, err := DecodeNUQChannel([]uint8{0, 1, 2, 3}, meta, 2.0, 1.0)
	if err != nil {
		t.Fatalf("DecodeNUQChannel: %v", err)
	}
	want := []float32{-2.0, 0.0, 2.0, 4.0}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got[%d] = %v, want %v", i, got[i], want[i])
		}
	}
	if _, err := DecodeNUQChannel([]uint8{4}, meta, 1.0, 0); err == nil {
		t.Fatal("expected error for out-of-range NUQ code")
	}
}

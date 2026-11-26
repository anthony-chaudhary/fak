package compute

import (
	"errors"
	"fmt"
	"testing"
)

// decodedScaleValid is the float definition the byte predicate replaces.
func decodedScaleValid(raw byte) bool {
	value := DecodeE4M3(raw)
	return nvfp4Finite(value) && value >= 0
}

func TestNVFP4ScaleByteValidMatchesDecodeAllBytes(t *testing.T) {
	for raw := 0; raw < 256; raw++ {
		if got, want := nvfp4ScaleByteValid(byte(raw)), decodedScaleValid(byte(raw)); got != want {
			t.Fatalf("byte %#02x: predicate %v, decode %v", raw, got, want)
		}
	}
}

// Every byte value at every position of a word-aligned run, plus a tail, must
// report the same first invalid index as a bytewise decode scan.
func TestNVFP4FirstInvalidScaleMatchesDecodeScan(t *testing.T) {
	for _, length := range []int{0, 1, 7, 8, 9, 16, 23, 24} {
		for position := 0; position < length; position++ {
			for raw := 0; raw < 256; raw++ {
				scales := nvfp4LoadScales(length)
				scales[position] = byte(raw)
				if position+1 < length {
					scales[length-1] = 0xFF
				}
				want := -1
				for index, b := range scales {
					if !decodedScaleValid(b) {
						want = index
						break
					}
				}
				if got := nvfp4FirstInvalidScale(scales); got != want {
					t.Fatalf("len %d pos %d byte %#02x: got %d, want %d", length, position, raw, got, want)
				}
			}
		}
	}
}

func TestNVFP4ValidateScalesErrorText(t *testing.T) {
	scales := nvfp4LoadScales(40)
	scales[3], scales[29] = 0x80, 0x7F
	err := nvfp4ValidateScales(scales, 1)
	want := fmt.Sprintf("%v: block scale 29 must be finite and non-negative", ErrNVFP4Scale)
	if !errors.Is(err, ErrNVFP4Scale) || err.Error() != want {
		t.Fatalf("got %v, want %q", err, want)
	}
	if err := nvfp4ValidateScales(nvfp4LoadScales(40), -1); !errors.Is(err, ErrNVFP4Scale) {
		t.Fatalf("negative tensor scale accepted: %v", err)
	}
}

func TestBytesAsI8CopiesBitPattern(t *testing.T) {
	src := []byte{0x00, 0x7F, 0x80, 0xFF, 0x12}
	got := bytesAsI8(src)
	src[0] = 0x55
	if want := []int8{0, 127, -128, -1, 0x12}; fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("got %v, want %v", got, want)
	}
}

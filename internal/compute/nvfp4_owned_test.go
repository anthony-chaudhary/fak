package compute

import (
	"bytes"
	"errors"
	"math/rand"
	"testing"
	"unsafe"
)

func nvfp4OwnedTestPayload(r *rand.Rand, rows, cols int) []byte {
	weightBytes, scaleBytes := rows*cols/2, rows*cols/NVFP4BlockSize
	payload := make([]byte, weightBytes+scaleBytes)
	r.Read(payload)
	for i := weightBytes; i < len(payload); i++ {
		payload[i] %= 0x7F
	}
	return payload
}

// fak-test:runtime fast est=10ms lane=default
func TestNVFP4OwnedMatchesCopyingConstructor(t *testing.T) {
	be := Default()
	r := rand.New(rand.NewSource(3))
	for _, shape := range [][2]int{{1, 16}, {7, 64}, {33, 2048}} {
		rows, cols := shape[0], shape[1]
		payload := nvfp4OwnedTestPayload(r, rows, cols)
		weightBytes := rows * cols / 2
		copied, err := NewNVFP4(be, []int{rows, cols}, payload[:weightBytes], payload[weightBytes:], 0.5)
		if err != nil {
			t.Fatal(err)
		}
		want := append([]byte(nil), i8View(copied.buf.(HostBuffer).I8())...)
		owned, err := NewNVFP4Owned(be, []int{rows, cols}, payload, 0.5)
		if err != nil {
			t.Fatal(err)
		}
		if got := i8View(owned.buf.(HostBuffer).I8()); !bytes.Equal(got, want) {
			t.Fatalf("[%d %d] owned host payload differs from the copying constructor", rows, cols)
		}
		if unsafe.SliceData(owned.buf.(HostBuffer).I8()) != (*int8)(unsafe.Pointer(&payload[0])) {
			t.Fatalf("[%d %d] owned constructor copied its payload", rows, cols)
		}
		cp, cs, cg := nvfp4WeightPayload(copied)
		op, os, og := nvfp4WeightPayload(owned)
		if !bytes.Equal(cp, op) || !bytes.Equal(cs, os) || cg != og {
			t.Fatalf("[%d %d] upload payloads differ between constructors", rows, cols)
		}
		if owned.Dtype != copied.Dtype || owned.Quant.Block != copied.Quant.Block || owned.Quant.Bits != copied.Quant.Bits {
			t.Fatalf("[%d %d] owned tensor metadata differs", rows, cols)
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestNVFP4OwnedRefusesInvalidPayloads(t *testing.T) {
	be := Default()
	const rows, cols = 4, 32
	weightBytes := rows * cols / 2
	fresh := func() []byte { return make([]byte, weightBytes+rows*cols/NVFP4BlockSize) }
	if _, err := NewNVFP4Owned(be, []int{rows, cols}, fresh()[1:], 1); !errors.Is(err, ErrNVFP4Shape) {
		t.Fatalf("short payload: %v", err)
	}
	if _, err := NewNVFP4Owned(be, []int{rows, 30}, fresh(), 1); !errors.Is(err, ErrNVFP4Shape) {
		t.Fatalf("bad shape: %v", err)
	}
	if _, err := NewNVFP4Owned(be, []int{rows, cols}, fresh(), -1); !errors.Is(err, ErrNVFP4Scale) {
		t.Fatalf("negative tensor scale: %v", err)
	}
	for _, bad := range []byte{0x7F, 0x81, 0xFF} {
		payload := fresh()
		payload[weightBytes+5] = bad
		if _, err := NewNVFP4Owned(be, []int{rows, cols}, payload, 1); !errors.Is(err, ErrNVFP4Scale) {
			t.Fatalf("scale 0x%02X: %v", bad, err)
		}
	}
	payload := fresh()
	payload[weightBytes+3] = 0x80
	if _, err := NewNVFP4Owned(be, []int{rows, cols}, payload, 1); err != nil {
		t.Fatalf("negative zero scale refused: %v", err)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestNVFP4OwnedStillChecksTensorScaleOnRead(t *testing.T) {
	w, err := NewNVFP4Owned(Default(), []int{1, 16}, make([]byte, 9), 1)
	if err != nil {
		t.Fatal(err)
	}
	w.Quant.Scale[0] = -2
	if _, _, _, err := checkedNVFP4WeightPayload(w); !errors.Is(err, ErrNVFP4Scale) {
		t.Fatalf("negative tensor scale after construction: %v", err)
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestNVFP4FirstInvalidScaleParallelMatchesSerial(t *testing.T) {
	r := rand.New(rand.NewSource(11))
	scales := make([]byte, 5*nvfp4ScaleChunk+13)
	for i := range scales {
		scales[i] = byte(r.Intn(0x7F))
	}
	if got := nvfp4FirstInvalidScaleParallel(scales); got != -1 {
		t.Fatalf("valid scales: got %d", got)
	}
	for _, at := range [][]int{{len(scales) - 1}, {3*nvfp4ScaleChunk + 1, 4*nvfp4ScaleChunk + 7}, {17, len(scales) - 2}} {
		buf := append([]byte(nil), scales...)
		for _, index := range at {
			buf[index] = 0x7F
		}
		if got, want := nvfp4FirstInvalidScaleParallel(buf), nvfp4FirstInvalidScale(buf); got != want || got != at[0] {
			t.Fatalf("invalid at %v: parallel %d, serial %d", at, got, want)
		}
	}
}

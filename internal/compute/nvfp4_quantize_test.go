package compute

import (
	"errors"
	"math"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestQuantizeNVFP4ReferenceE2M1BoundariesAndPacking(t *testing.T) {
	values := make([]float32, 32)
	copy(values, []float32{
		0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5,
		-0.25, -0.75, -1.25, -1.75, -2.5, -3.5, -5, 0, 6,
	})
	values[16] = 2688 // pins the tensor scale to 1 and its block scale to 448

	packed, scales, global, err := QuantizeNVFP4Reference(1, 32, values)
	if err != nil {
		t.Fatal(err)
	}
	if global != 1 {
		t.Fatalf("global scale = %v, want 1", global)
	}
	if want := []byte{0x38, 0x7e}; !equalNVFP4Bytes(scales, want) {
		t.Fatalf("block scales = %x, want %x", scales, want)
	}
	wantPackedPrefix := []byte{0x20, 0x42, 0x64, 0x86, 0xaa, 0xcc, 0xee, 0x70}
	if !equalNVFP4Bytes(packed[:len(wantPackedPrefix)], wantPackedPrefix) {
		t.Fatalf("packed prefix = %x, want %x", packed[:len(wantPackedPrefix)], wantPackedPrefix)
	}
	if packed[8] != 0x07 {
		t.Fatalf("anchor byte = %#x, want low-nibble code 7", packed[8])
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestQuantizeNVFP4ReferenceE4M3TiesZeroAndRange(t *testing.T) {
	values := make([]float32, 64)
	values[0] = 6.375                        // block scale 1.0625: tie -> 1.0 (0x38)
	values[16] = 7.125                       // block scale 1.1875: tie -> 1.25 (0x3a)
	values[32] = math.SmallestNonzeroFloat32 // block scale clamps to 2^-9 (0x01)
	values[48] = 2688                        // global scale 1, block scale 448
	_, scales, global, err := QuantizeNVFP4Reference(1, 64, values)
	if err != nil {
		t.Fatal(err)
	}
	if global != 1 || !equalNVFP4Bytes(scales, []byte{0x38, 0x3a, 0x01, 0x7e}) {
		t.Fatalf("global=%v scales=%x", global, scales)
	}

	packed, zeroScales, zeroGlobal, err := QuantizeNVFP4Reference(2, 16, make([]float32, 32))
	if err != nil {
		t.Fatal(err)
	}
	if zeroGlobal != 0 || !equalNVFP4Bytes(zeroScales, []byte{0x38, 0x38}) {
		t.Fatalf("canonical zero global=%v scales=%x", zeroGlobal, zeroScales)
	}
	for index, value := range packed {
		if value != 0 {
			t.Fatalf("canonical zero packed[%d]=%#x", index, value)
		}
	}

	large := make([]float32, 16)
	large[0] = math.MaxFloat32
	p, s, g, err := QuantizeNVFP4Reference(1, 16, large)
	if err != nil {
		t.Fatalf("maximum finite value: %v", err)
	}
	decoded, err := DecodeNVFP4Reference(1, 16, p, s, g)
	if err != nil || !nvfp4Finite(decoded[0]) || decoded[0] <= 0 {
		t.Fatalf("maximum finite roundtrip value=%v err=%v", decoded[0], err)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestQuantizeNVFP4ReferenceRejectsInvalidInput(t *testing.T) {
	valid := make([]float32, 16)
	cases := []struct {
		name       string
		rows, cols int
		values     []float32
		want       error
	}{
		{"zero rows", 0, 16, nil, ErrNVFP4Shape},
		{"partial block", 1, 15, make([]float32, 15), ErrNVFP4Shape},
		{"short values", 1, 16, valid[:15], ErrNVFP4Shape},
		{"NaN", 1, 16, append([]float32{float32(math.NaN())}, valid[1:]...), ErrNVFP4Input},
		{"infinity", 1, 16, append([]float32{float32(math.Inf(1))}, valid[1:]...), ErrNVFP4Input},
		{"global underflow", 1, 16, append([]float32{math.SmallestNonzeroFloat32}, valid[1:]...), ErrNVFP4Scale},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, _, _, err := QuantizeNVFP4Reference(tc.rows, tc.cols, tc.values)
			if !errors.Is(err, tc.want) {
				t.Fatalf("error = %v, want errors.Is(_, %v)", err, tc.want)
			}
		})
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestQuantizeNVFP4DecodeAndCPUMatMul(t *testing.T) {
	be := &cpuBackend{}
	values := []float32{2688, 1344, 896, 448, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	tensor, err := QuantizeNVFP4(be, []int{1, 16}, values)
	if err != nil {
		t.Fatal(err)
	}
	packed, scales, global := nvfp4WeightPayload(tensor)
	decoded, err := DecodeNVFP4Reference(1, 16, packed, scales, global)
	if err != nil {
		t.Fatal(err)
	}
	wantDecoded := []float32{2688, 1344, 896, 448}
	for index, want := range wantDecoded {
		if decoded[index] != want {
			t.Fatalf("decoded[%d]=%v want %v", index, decoded[index], want)
		}
	}
	x := make([]float32, 16)
	copy(x, []float32{1, 2, 3, 4})
	got := be.Read(be.MatMul(tensor, NewF32(be, []int{16}, x)))
	want := float32(2688 + 2*1344 + 3*896 + 4*448)
	if len(got) != 1 || got[0] != want {
		t.Fatalf("CPU MatMul = %v, want [%v]", got, want)
	}
}

func equalNVFP4Bytes(left, right []byte) bool {
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if left[index] != right[index] {
			return false
		}
	}
	return true
}

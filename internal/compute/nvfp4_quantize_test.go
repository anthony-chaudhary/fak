package compute

import (
	"bytes"
	"math"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestNVFP4QuantizeCPUReference(t *testing.T) {
	t.Parallel()
	negativeZero := math.Float32frombits(0x80000000)
	codebook := []float32{0, .5, 1, 1.5, 2, 3, 4, 6, negativeZero, -.5, -1, -1.5, -2, -3, -4, -6}
	// Independent literal bytes, four distinct blocks across two rows, and a
	// nonunit global decode multiplier. Never compute expected codes via decode.
	var input []float32
	for _, factor := range []float32{.5, 1, 1.0 / 1024, 224} {
		for _, v := range codebook {
			input = append(input, v*factor)
		}
	}
	before := append([]float32(nil), input...)
	m, err := QuantizeNVFP4(input, 2, 32, .5)
	if err != nil {
		t.Fatal(err)
	}
	wantPacked := bytes.Repeat([]byte{0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe}, 4)
	if m.Rows != 2 || m.Cols != 32 || m.GlobalScale != .5 || !bytes.Equal(m.Packed, wantPacked) || !bytes.Equal(m.BlockScales, []byte{0x38, 0x40, 0x01, 0x7e}) {
		t.Fatalf("codebook matrix: %+v", m)
	}
	decoded, err := m.Dequantize()
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range input {
		if math.Float32bits(v) != math.Float32bits(before[i]) || math.Float32bits(decoded[i]) != math.Float32bits(before[i]) {
			t.Fatalf("input mutation or exact-code decode mismatch at %d", i)
		}
	}
	// Every E2M1 midpoint, both signs. The final magnitude anchors scale 1.
	ties := []float32{.25, .75, 1.25, 1.75, 2.5, 3.5, 5, 6, -.25, -.75, -1.25, -1.75, -2.5, -3.5, -5, -6}
	m, err = QuantizeNVFP4(ties, 1, 16, 1)
	if err != nil || !bytes.Equal(m.BlockScales, []byte{0x38}) || !bytes.Equal(m.Packed, []byte{0x20, 0x42, 0x64, 0x76, 0xa8, 0xca, 0xec, 0xfe}) {
		t.Fatalf("E2M1 ties: %+v err=%v", m, err)
	}
	// Values immediately on either side of a midpoint must not become ties.
	neighbors := make([]float32, 16)
	neighbors[0], neighbors[1], neighbors[15] = math.Nextafter32(.75, 0), math.Nextafter32(.75, 1), 6
	m, err = QuantizeNVFP4(neighbors, 1, 16, 1)
	if err != nil || !bytes.Equal(m.Packed, []byte{0x21, 0, 0, 0, 0, 0, 0, 0x70}) {
		t.Fatalf("midpoint neighbors: %+v err=%v", m, err)
	}

	for _, tt := range []struct {
		name        string
		amax        float32
		scale, pair byte
	}{
		{"even lower scale tie", 6.375, 0x38, 0xf7},
		{"even upper scale tie", 7.125, 0x3a, 0xf7},
		{"scale tie below", math.Nextafter32(6.375, 0), 0x38, 0xf7},
		{"scale tie above", math.Nextafter32(6.375, 7), 0x39, 0xf7},
		{"zero scale tie", 6.0 / 1024, 0, 0x80},
		{"zero scale tie above", math.Nextafter32(6.0/1024, 1), 1, 0xd5},
		{"minimum scale", 6.0 / 512, 1, 0xf7},
		{"subnormal normal scale tie", 6 * 7.5 / 512, 8, 0xf7},
		{"upper finite scale tie", 6 * 432, 0x7e, 0xf7},
		{"finite scale saturation", 6000, 0x7e, 0xf7},
		{"nonzero scale underflow", math.SmallestNonzeroFloat32, 0, 0x80},
	} {
		t.Run(tt.name, func(t *testing.T) {
			x := make([]float32, 16)
			x[0], x[1] = tt.amax, -tt.amax
			got, err := QuantizeNVFP4(x, 1, 16, 1)
			want := []byte{tt.pair, 0, 0, 0, 0, 0, 0, 0}
			if err != nil || !bytes.Equal(got.BlockScales, []byte{tt.scale}) || !bytes.Equal(got.Packed, want) {
				t.Fatalf("got %+v err=%v, want scale=%02x packed=%x", got, err, tt.scale, want)
			}
		})
	}
	zeros := make([]float32, 16)
	zeros[1] = negativeZero
	for _, global := range []float32{0, 1, math.MaxFloat32} {
		got, err := QuantizeNVFP4(zeros, 1, 16, global)
		if err != nil || !bytes.Equal(got.BlockScales, []byte{0}) || !bytes.Equal(got.Packed, []byte{0x80, 0, 0, 0, 0, 0, 0, 0}) || math.Float32bits(got.GlobalScale) != math.Float32bits(global) {
			t.Fatalf("signed-zero block global=%g: %+v err=%v", global, got, err)
		}
	}
	// Tiny but nonzero globals must not disappear in FP32 scale arithmetic.
	tiny := make([]float32, 16)
	tiny[0], tiny[1], tiny[15] = math.SmallestNonzeroFloat32, -math.SmallestNonzeroFloat32, 6*math.SmallestNonzeroFloat32
	m, err = QuantizeNVFP4(tiny, 1, 16, math.SmallestNonzeroFloat32)
	if err != nil || !bytes.Equal(m.BlockScales, []byte{0x38}) || !bytes.Equal(m.Packed, []byte{0xa2, 0, 0, 0, 0, 0, 0, 0x70}) {
		t.Fatalf("tiny global: %+v err=%v", m, err)
	}
	for _, tt := range []struct {
		name       string
		rows, cols int
		length     int
		global, v  float32
	}{
		{"zero rows", 0, 16, 16, 1, 0},
		{"negative rows", -1, 16, 16, 1, 0},
		{"zero columns", 1, 0, 16, 1, 0},
		{"negative columns", 1, -16, 16, 1, 0},
		{"partial block", 1, 17, 17, 1, 0},
		{"shape overflow", int(^uint(0) >> 1), 16, 16, 1, 0},
		{"short input", 1, 16, 15, 1, 0},
		{"long input", 1, 16, 17, 1, 0},
		{"input NaN", 1, 16, 16, 1, float32(math.NaN())},
		{"input infinity", 1, 16, 16, 1, float32(math.Inf(1))},
		{"input negative infinity", 1, 16, 16, 1, float32(math.Inf(-1))},
		{"global NaN", 1, 16, 16, float32(math.NaN()), 0},
		{"global infinity", 1, 16, 16, float32(math.Inf(1)), 0},
		{"global negative", 1, 16, 16, -1, 0},
		{"global negative zero", 1, 16, 16, negativeZero, 0},
		{"zero global nonzero input", 1, 16, 16, 0, 1},
		{"zero global tiny input", 1, 16, 16, 0, math.SmallestNonzeroFloat32},
		// Finite input rounds up to 6 * .6875 * 2^126, which overflows FP32.
		{"decoded overflow", 1, 16, 16, 0x1p126, math.MaxFloat32},
	} {
		t.Run(tt.name, func(t *testing.T) {
			x := make([]float32, tt.length)
			x[len(x)-1] = tt.v // A late invalid value must expose no partial output.
			got, err := QuantizeNVFP4(x, tt.rows, tt.cols, tt.global)
			if err == nil || got.Rows != 0 || got.Cols != 0 || got.Packed != nil || got.BlockScales != nil || math.Float32bits(got.GlobalScale) != 0 {
				t.Fatalf("got %+v err=%v, want zero matrix and error", got, err)
			}
		})
	}
}

package model

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"math"
	"testing"
)

// v41DenseE4M3Codes are finite E4M3 byte codes (0x7f/0xff are NaN), chosen to
// cross tile boundaries when laid out as a reduced [40,70] weight.
var v41DenseE4M3Codes = []byte{0x38, 0x40, 0x30, 0xb8, 0xc0, 0x34, 0x3c, 0x28}

func v41DenseFixture(rows, cols int, scales []byte) ([]byte, []float32) {
	weights := make([]byte, rows*cols)
	decoded := make([]float32, rows*cols)
	scaleCols := (cols-1)/v41FP8BlockDim + 1
	for o := 0; o < rows; o++ {
		for i := 0; i < cols; i++ {
			b := v41DenseE4M3Codes[(o*cols+i)%len(v41DenseE4M3Codes)]
			weights[o*cols+i] = b
			scale := float32(math.Ldexp(1, int(scales[(o/v41FP8BlockDim)*scaleCols+i/v41FP8BlockDim])-127))
			decoded[o*cols+i] = flashE4M3Oracle(b) * scale
		}
	}
	return weights, decoded
}

// TestV41FP8Dense32x32DecodeSemantics crosses multiple 32x32 tiles on a reduced
// [40,70] fixture and compares against an independent E4M3*E8M0 oracle.
func TestV41FP8Dense32x32DecodeSemantics(t *testing.T) {
	const rows, cols = 40, 70
	scaleRows, scaleCols := (rows-1)/v41FP8BlockDim+1, (cols-1)/v41FP8BlockDim+1
	if scaleRows != 2 || scaleCols != 3 {
		t.Fatalf("fixture scale tiles = %dx%d, want 2x3", scaleRows, scaleCols)
	}
	scales := make([]byte, scaleRows*scaleCols)
	for i := range scales {
		scales[i] = []byte{127, 128, 126, 129, 0, 130}[i%6]
	}
	weights, want := v41DenseFixture(rows, cols, scales)

	got, err := decodeV41DenseFP8("layers.0.attn.wq_a.weight", []int{rows, cols}, weights, scales)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != rows*cols {
		t.Fatalf("decoded length = %d, want %d", len(got), rows*cols)
	}
	for i := range want {
		if gotBits, wantBits := math.Float32bits(got[i]), math.Float32bits(want[i]); gotBits != wantBits {
			t.Fatalf("decoded[%d] bits = %08x, want %08x (%v vs %v)", i, gotBits, wantBits, got[i], want[i])
		}
	}
}

// TestV41FP8DenseMalformedFailsBeforeAllocation proves every malformed input is
// refused before any output slice is produced.
func TestV41FP8DenseMalformedFailsBeforeAllocation(t *testing.T) {
	weightName := "layers.0.attn.wq_a.weight"
	scaleName := "layers.0.attn.wq_a.scale"
	goodWeight := stEntry{Dtype: "F8_E4M3", Shape: []int{40, 70}}
	goodScale := stEntry{Dtype: "F8_E8M0", Shape: []int{2, 3}}
	hdr := func(w, s stEntry) map[string]json.RawMessage {
		wRaw, _ := json.Marshal(w)
		sRaw, _ := json.Marshal(s)
		return map[string]json.RawMessage{weightName: wRaw, scaleName: sRaw}
	}
	scales := []byte{127, 127, 127, 127, 127, 127}
	base := make([]byte, 40*70)
	badNaN := append([]byte(nil), base...)
	badNaN[0] = 0x7f
	badScaleNaN := append([]byte(nil), scales...)
	badScaleNaN[2] = 0xff
	overflowWeights := append([]byte(nil), base...)
	overflowWeights[0] = 0x7e
	overflowScales := append([]byte(nil), scales...)
	overflowScales[0] = 0xfe

	decodeCases := []struct {
		name    string
		shape   []int
		weights []byte
		scales  []byte
	}{
		{name: "rank 3", shape: []int{40, 70, 1}, weights: base, scales: scales},
		{name: "non-positive", shape: []int{0, 70}, weights: base, scales: scales},
		{name: "weight length", shape: []int{40, 70}, weights: base[:len(base)-1], scales: scales},
		{name: "scale length", shape: []int{40, 70}, weights: base, scales: scales[:len(scales)-1]},
		{name: "weight NaN", shape: []int{40, 70}, weights: badNaN, scales: scales},
		{name: "scale NaN", shape: []int{40, 70}, weights: base, scales: badScaleNaN},
		{name: "non-finite product", shape: []int{40, 70}, weights: overflowWeights, scales: overflowScales},
	}
	for _, tt := range decodeCases {
		t.Run("decode "+tt.name, func(t *testing.T) {
			got, err := decodeV41DenseFP8(weightName, tt.shape, tt.weights, tt.scales)
			if got != nil || err == nil {
				t.Fatalf("got=%v err=%v, want nil output and error", got, err)
			}
		})
	}

	pairCases := []struct {
		name   string
		weight stEntry
		scale  stEntry
	}{
		{name: "weight dtype", weight: stEntry{Dtype: "BF16", Shape: goodWeight.Shape}, scale: goodScale},
		{name: "scale dtype", weight: goodWeight, scale: stEntry{Dtype: "F32", Shape: goodScale.Shape}},
		{name: "weight shape", weight: stEntry{Dtype: "F8_E4M3", Shape: []int{40, 69}}, scale: goodScale},
		{name: "scale shape", weight: goodWeight, scale: stEntry{Dtype: "F8_E8M0", Shape: []int{2, 2}}},
	}
	for _, tt := range pairCases {
		t.Run("pair "+tt.name, func(t *testing.T) {
			_, _, err := v41DenseFP8PairEntries(weightName, scaleName, [2]int{40, 70}, hdr(tt.weight, tt.scale))
			if err == nil {
				t.Fatalf("pair validation accepted malformed %s", tt.name)
			}
		})
	}
}

// TestV41FP8ScalarReferenceParity runs an independent scalar dequant loop and
// requires it to match decodeV41DenseFP8 bit-for-bit on a reduced fixture.
func TestV41FP8ScalarReferenceParity(t *testing.T) {
	const rows, cols = 33, 33
	scaleRows, scaleCols := (rows-1)/v41FP8BlockDim+1, (cols-1)/v41FP8BlockDim+1
	scales := make([]byte, scaleRows*scaleCols)
	for i := range scales {
		scales[i] = []byte{0, 127, 128, 131}[i%4]
	}
	weights := make([]byte, rows*cols)
	for i := range weights {
		weights[i] = v41DenseE4M3Codes[i%len(v41DenseE4M3Codes)]
	}

	got, err := decodeV41DenseFP8("layers.1.ffn.shared_experts.w1.weight", []int{rows, cols}, weights, scales)
	if err != nil {
		t.Fatal(err)
	}
	reference := make([]float32, rows*cols)
	for o := 0; o < rows; o++ {
		for i := 0; i < cols; i++ {
			reference[o*cols+i] = fp8E4M3ToF32(weights[o*cols+i]) * float32(math.Ldexp(1, int(scales[(o/v41FP8BlockDim)*scaleCols+i/v41FP8BlockDim])-127))
		}
	}
	for i := range reference {
		if gotBits, wantBits := math.Float32bits(got[i]), math.Float32bits(reference[i]); gotBits != wantBits {
			t.Fatalf("scalar reference mismatch at %d: %08x vs %08x", i, gotBits, wantBits)
		}
	}
}

// TestV41FP4ExpertNibbleAndScaleSemantics exercises the real decoder over a
// temporarily reduced spec map: low-nibble val0 / high-nibble val1 ordering,
// the 16-packed-byte scale group boundary at K=32, and E8M0 byte 0 => 2^-127.
func TestV41FP4ExpertNibbleAndScaleSemantics(t *testing.T) {
	saved := v41ExpertQuantSpecs
	defer func() { v41ExpertQuantSpecs = saved }()

	const rows, packedCols, scaleCols = 2, 32, 2
	v41ExpertQuantSpecs = map[string]v41ExpertQuantSpec{
		"w1": {weightRows: rows, weightCols: packedCols, scaleRows: rows, scaleCols: scaleCols},
		"w2": saved["w2"],
		"w3": saved["w3"],
	}
	weights := make([]byte, rows*packedCols)
	scales := make([]byte, rows*scaleCols)
	scales[0] = 127
	scales[1] = 128
	weights[0] = 0x0f
	weights[16] = 0x11
	scales[2] = 0
	weights[rows/2*packedCols] = 0x22

	name := "layers.0.ffn.experts.0.w1"
	got, shape, err := decodeV41ExpertQuant(name+".weight", name+".scale",
		stEntry{Dtype: "I8", Shape: []int{rows, packedCols}},
		stEntry{Dtype: "F8_E8M0", Shape: []int{rows, scaleCols}},
		weights, scales)
	if err != nil {
		t.Fatal(err)
	}
	if !sameShape(shape, []int{rows, 2 * packedCols}) {
		t.Fatalf("shape = %v, want [%d %d]", shape, rows, 2*packedCols)
	}
	assertBits := func(index int, want float32) {
		t.Helper()
		if gotBits, wantBits := binary.LittleEndian.Uint32(got[index*4:]), math.Float32bits(want); gotBits != wantBits {
			t.Fatalf("decoded[%d] bits=%08x, want %08x (%v)", index, gotBits, wantBits, want)
		}
	}
	assertBits(0, -6)
	assertBits(1, 0)
	assertBits(32, 1)
	assertBits(33, 1)
	assertBits(64, float32(math.Ldexp(1, -127)))
	assertBits(65, float32(math.Ldexp(1, -127)))
}

// TestV41FP4DoesNotSelectV4ProOrFlashSpec proves the V4.1 expert geometry is
// disjoint from the V4-Pro and V4-Flash packed-expert spec maps.
func TestV41FP4DoesNotSelectV4ProOrFlashSpec(t *testing.T) {
	v41Weight, v41Scale := []int{2304, 2560}, []int{2304, 160}
	if _, ok := selectV41ExpertQuantSpec("w1", v41Weight, v41Scale); !ok {
		t.Fatalf("V4.1 w1 spec did not select its own published shape")
	}
	if _, ok := selectV4ExpertQuantSpec("w1", v41Weight, v41Scale); ok {
		t.Fatalf("V4-Pro/Flash selector accepted V4.1 w1 shape %v %v", v41Weight, v41Scale)
	}
	proWeight, proScale := []int{3072, 3584}, []int{3072, 224}
	if _, ok := selectV41ExpertQuantSpec("w1", proWeight, proScale); ok {
		t.Fatalf("V4.1 selector accepted V4-Pro w1 shape")
	}
	flashWeight, flashScale := []int{2048, 2048}, []int{2048, 128}
	if _, ok := selectV41ExpertQuantSpec("w1", flashWeight, flashScale); ok {
		t.Fatalf("V4.1 selector accepted V4-Flash w1 shape")
	}

	name := "layers.0.ffn.experts.0.w1"
	_, _, err := decodeV41ExpertQuant(name+".weight", name+".scale",
		stEntry{Dtype: "I8", Shape: proWeight},
		stEntry{Dtype: "F8_E8M0", Shape: proScale},
		nil, nil)
	if !errors.Is(err, ErrV4ExpertQuantMetadata) {
		t.Fatalf("error = %v, want ErrV4ExpertQuantMetadata for Pro shape", err)
	}
}

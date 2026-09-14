package model

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"math"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
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

// ---- Routed-expert device matmul witnesses -------------------------------------

// v41DeviceExpertBackend is an in-memory stand-in for a gfx1151 device that advertises
// the routed-expert device kernel capability. It is [SW-VERIFIED]: it proves the model
// dispatch reaches a device kernel and preserves packed bytes, NOT physical TOPS/GB·s-1.
type v41DeviceExpertBackend struct {
	compute.Backend
	fp8Calls, fp4Calls int
}

func (b *v41DeviceExpertBackend) GemvExpertFP8Block32E8M0(O, I int, weight, scales []byte, x []float32) ([]float32, error) {
	b.fp8Calls++
	return compute.GemvFP8Block32E8M0("device", O, I, weight, scales, x)
}

func (b *v41DeviceExpertBackend) GemvExpertMXFP4E8M0(O, I int, weight, scales []byte, x []float32) ([]float32, error) {
	b.fp4Calls++
	return compute.GemvMXFP4E8M0("device", O, I, weight, scales, x)
}

// v41ExpertDeviceFixture builds a reduced V4.1 routed-expert (weight, scale) fixture
// under a temporarily reduced spec map and returns the packed bytes plus the spec.
func v41ExpertDeviceFixture(t *testing.T, rows, packedCols, scaleCols int) (name string, weights, scales []byte) {
	t.Helper()
	saved := v41ExpertQuantSpecs
	v41ExpertQuantSpecs = map[string]v41ExpertQuantSpec{
		"w1": {weightRows: rows, weightCols: packedCols, scaleRows: rows, scaleCols: scaleCols},
		"w2": saved["w2"],
		"w3": saved["w3"],
	}
	weights = make([]byte, rows*packedCols)
	for i := range weights {
		weights[i] = byte((i*3 + 1) & 0xff)
	}
	scales = make([]byte, rows*scaleCols)
	for i := range scales {
		scales[i] = byte(126 + i%5)
	}
	return "layers.0.ffn.experts.0.w1", weights, scales
}

// TestV41ExpertDeviceMatmul witnesses the routed-expert device path end-to-end:
// (1) install preserves PACKED bytes (no full-checkpoint dequant) and is observable;
// (2) the device matmul reaches a capable backend kernel and matches the independent
// scalar oracle; (3) an ABSENT device kernel fails CLOSED with a typed error and never
// silently produces a scalar fallback result; (4) the dense path is unchanged.
func TestV41ExpertDeviceMatmul(t *testing.T) {
	saved := v41ExpertQuantSpecs
	defer func() { v41ExpertQuantSpecs = saved }()

	m := &Model{Cfg: Config{NumLayers: 1, DeepSeekV41: &DeepSeekV41Config{}}}

	const rows, packedCols, scaleCols = 8, 32, 2
	name, weights, scales := v41ExpertDeviceFixture(t, rows, packedCols, scaleCols)

	weightEntry := stEntry{Dtype: "I8", Shape: []int{rows, packedCols}}
	scaleEntry := stEntry{Dtype: "F8_E8M0", Shape: []int{rows, scaleCols}}

	// A packed-byte snapshot BEFORE install: install must not mutate the source and must
	// copy, not expand, the bytes.
	weightBefore := append([]byte(nil), weights...)
	if err := installV41ExpertQuantDevice(name+".weight", name+".scale", weightEntry, scaleEntry, weights, scales, m); err != nil {
		t.Fatalf("installV41ExpertQuantDevice: %v", err)
	}
	if !m.V41ExpertDeviceInstalled(name + ".weight") {
		t.Fatalf("routed expert %s was not observed as device-installed", name)
	}
	if !m.V41ExpertDevicePackedPreserved() {
		t.Fatalf("device install did not report packed-byte preservation (full dequant?)")
	}
	if string(weights) != string(weightBefore) {
		t.Fatalf("device install mutated the caller's packed source bytes")
	}
	store := v41ExpertDeviceStore(m, false)
	w := store.weights[name+".weight"]
	if w == nil {
		t.Fatalf("installed expert missing from store")
	}
	if len(w.weight) != rows*packedCols || len(w.scales) != rows*scaleCols {
		t.Fatalf("installed packed sizes = %d/%d, want %d/%d", len(w.weight), len(w.scales), rows*packedCols, rows*scaleCols)
	}
	if w.dtype != "I8" {
		t.Fatalf("installed dtype = %q, want I8 (packed MXFP4)", w.dtype)
	}

	// (2) device matmul reaches the kernel and matches the scalar oracle.
	I := unpackedColsFor(t, packedCols)
	x := make([]float32, I)
	for i := range x {
		x[i] = float32(math.Cos(float64(i)*0.07)) - 0.1
	}
	device := &v41DeviceExpertBackend{Backend: compute.Default()}
	got, err := m.V41ExpertDeviceMatmul(name+".weight", device, x)
	if err != nil {
		t.Fatalf("V41ExpertDeviceMatmul: %v", err)
	}
	if device.fp4Calls != 1 {
		t.Fatalf("device MXFP4 kernel calls = %d, want 1", device.fp4Calls)
	}
	if len(got) != rows {
		t.Fatalf("device matmul returned %d outputs, want %d", len(got), rows)
	}
	// Independent oracle over the PACKED bytes (E2M1 nibbles x E8M0 block scales).
	reference := make([]float32, rows)
	for o := 0; o < rows; o++ {
		var acc float32
		for i := 0; i < I; i++ {
			b := weights[o*packedCols+i/2]
			nib := b & 0x0f
			if i%2 == 1 {
				nib = b >> 4
			}
			exp := int(scales[o*scaleCols+i/v41FP8BlockDim]) - 127
			acc += v4ExpertE2M1Values[nib] * float32(math.Ldexp(1, exp)) * x[i]
		}
		reference[o] = acc
	}
	for o := range reference {
		if math.Float32bits(got[o]) != math.Float32bits(reference[o]) {
			t.Fatalf("device matmul[%d] bits=%08x, want %08x", o, math.Float32bits(got[o]), math.Float32bits(reference[o]))
		}
	}

	// (3) absent device kernel fails CLOSED with a typed error; no scalar fallback.
	absent := compute.Default() // cpu-ref implements no RoutedExpertDeviceKernel
	if _, err := m.V41ExpertDeviceMatmul(name+".weight", absent, x); err == nil {
		t.Fatalf("absent-kernel device matmul returned a result; want a fail-closed error")
	} else if !strings.Contains(err.Error(), "device kernel is absent") {
		t.Fatalf("absent-kernel error = %v, want a typed absent-kernel refusal", err)
	}

	// (4) uninstalled expert fails closed; dense path is untouched.
	if _, err := m.V41ExpertDeviceMatmul("layers.9.ffn.experts.0.w1.weight", device, x); err == nil {
		t.Fatalf("uninstalled expert matmul returned a result; want a metadata error")
	}
	if len(m.q8w) != 0 {
		t.Fatalf("device expert install touched m.q8w (dense path) unexpectedly")
	}
}

// unpackedColsFor mirrors the spec's unpacked column count (packedCols*2).
func unpackedColsFor(t *testing.T, packedCols int) int {
	t.Helper()
	cols, ok := checkedShapeProduct(packedCols, 2)
	if !ok {
		t.Fatalf("unpacked col product overflow")
	}
	return cols
}

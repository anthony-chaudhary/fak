package model

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"sort"
	"testing"
)

// cosSim is the cosine similarity between two equal-length vectors.
func cosSim(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

// TestEXL2BitStreamRoundTrip proves the variable-width packer is bijective for
// every supported bpw, including codes that straddle a uint32 word boundary.
func TestEXL2BitStreamRoundTrip(t *testing.T) {
	for _, bits := range []int{2, 3, 4, 5, 6, 8} {
		maxCode := (1 << uint(bits)) - 1
		// Lay out enough codes to cross several word boundaries at this width.
		n := 200
		words := make([]uint32, (n*bits+31)/32)
		want := make([]int, n)
		bitpos := 0
		for i := 0; i < n; i++ {
			code := (i*7 + bits) & maxCode // deterministic spread over [0,maxCode]
			want[i] = code
			exl2WriteBits(words, bitpos, bits, code)
			bitpos += bits
		}
		bitpos = 0
		for i := 0; i < n; i++ {
			got := exl2ReadBits(words, bitpos, bits)
			bitpos += bits
			if got != want[i] {
				t.Fatalf("bits=%d i=%d: read %d, wrote %d", bits, i, got, want[i])
			}
		}
	}
}

// TestEXL2QuantizeDequantCosine quantizes a random FP32 matrix into a single 4-bit
// group and checks the dequant tracks the original to cosine >= 0.995 (the AWQ-grade
// fidelity gate).
func TestEXL2QuantizeDequantCosine(t *testing.T) {
	out, in := 16, 64
	w := make([]float32, out*in)
	for i := range w {
		w[i] = float32(math.Sin(float64(i)*0.37)) * 0.8
	}
	qt, err := exl2Quantize(w, out, in, []int{4}, []int{in}, nil)
	if err != nil {
		t.Fatalf("quantize: %v", err)
	}
	deq := make([]float32, out*in)
	row := make([]float32, in)
	for o := 0; o < out; o++ {
		exl2DequantRow(row, qt, o)
		copy(deq[o*in:(o+1)*in], row)
	}
	if cs := cosSim(w, deq); cs < 0.995 {
		t.Fatalf("4-bit dequant cosine %.5f < 0.995", cs)
	}
}

// TestEXL2MixedBitGroups exercises the defining EXL2 feature: a single tensor whose
// input axis is split into groups at DIFFERENT bit widths. Higher-bpw groups must
// dequantize more faithfully than lower-bpw ones, and the whole matrix tracks the
// original to a sane cosine.
func TestEXL2MixedBitGroups(t *testing.T) {
	out := 8
	groupBits := []int{2, 4, 8}
	groupSizes := []int{32, 32, 32}
	in := 96
	w := make([]float32, out*in)
	for i := range w {
		w[i] = float32(math.Cos(float64(i) * 0.11)) // span all groups identically
	}
	qt, err := exl2Quantize(w, out, in, groupBits, groupSizes, nil)
	if err != nil {
		t.Fatalf("quantize: %v", err)
	}
	if qt.nGroups() != 3 {
		t.Fatalf("nGroups = %d, want 3", qt.nGroups())
	}
	deq := make([]float32, out*in)
	row := make([]float32, in)
	for o := 0; o < out; o++ {
		exl2DequantRow(row, qt, o)
		copy(deq[o*in:(o+1)*in], row)
	}
	// Per-group MSE should fall monotonically as bpw rises (2 < 4 < 8 bits).
	groupMSE := func(g, lo, hi int) float64 {
		var se float64
		var n int
		for o := 0; o < out; o++ {
			for i := lo; i < hi; i++ {
				d := float64(w[o*in+i] - deq[o*in+i])
				se += d * d
				n++
			}
		}
		return se / float64(n)
	}
	mse2 := groupMSE(0, 0, 32)
	mse4 := groupMSE(1, 32, 64)
	mse8 := groupMSE(2, 64, 96)
	if !(mse2 > mse4 && mse4 > mse8) {
		t.Fatalf("expected MSE to fall with bpw: 2bit=%.3e 4bit=%.3e 8bit=%.3e", mse2, mse4, mse8)
	}
	// The 2-bit group (a coarse 4-level quantizer) caps the aggregate cosine; the
	// mixed-precision mechanism is proven by the monotone per-group MSE above.
	if cs := cosSim(w, deq); cs < 0.97 {
		t.Fatalf("mixed-bit dequant cosine %.5f < 0.97", cs)
	}
}

// TestEXL2ActOrderGEMV checks that the inverse act-order permutation is honored:
// the EXL2 GEMV against a logical-order activation vector matches the FP32 baseline
// to cosine >= 0.995, even though the weights are stored in a permuted order.
func TestEXL2ActOrderGEMV(t *testing.T) {
	out, in := 12, 48
	w := make([]float32, out*in)
	for i := range w {
		w[i] = float32(math.Sin(float64(i)*0.21) + 0.3*math.Cos(float64(i)*0.05))
	}
	// A non-identity permutation: storage column i -> logical channel perm[i].
	perm := make([]int32, in)
	for i := range perm {
		perm[i] = int32((i*5 + 7) % in)
	}
	// Validate it is a permutation (defensive; the test would be meaningless otherwise).
	seen := make([]bool, in)
	for _, p := range perm {
		if seen[p] {
			t.Fatalf("test perm is not bijective")
		}
		seen[p] = true
	}
	qt, err := exl2Quantize(w, out, in, []int{8}, []int{in}, perm)
	if err != nil {
		t.Fatalf("quantize: %v", err)
	}
	x := make([]float32, in)
	for i := range x {
		x[i] = float32(math.Cos(float64(i) * 0.4))
	}
	got := exl2MatRows(qt, x)
	// FP32 baseline: logical-order weight times logical-order activation.
	want := make([]float32, out)
	for o := 0; o < out; o++ {
		var acc float32
		for i := 0; i < in; i++ {
			acc += w[o*in+i] * x[i]
		}
		want[o] = acc
	}
	if cs := cosSim(want, got); cs < 0.995 {
		t.Fatalf("act-order GEMV cosine %.5f < 0.995", cs)
	}
}

// TestEXL2GemmMatchesGEMV checks the prefill GEMM agrees with the per-row GEMV for
// every token (the dequant-once-reuse path must equal the one-row path).
func TestEXL2GemmMatchesGEMV(t *testing.T) {
	out, in, P := 10, 32, 4
	w := make([]float32, out*in)
	for i := range w {
		w[i] = float32(math.Sin(float64(i) * 0.13))
	}
	qt, err := exl2Quantize(w, out, in, []int{4, 6}, []int{16, 16}, nil)
	if err != nil {
		t.Fatalf("quantize: %v", err)
	}
	X := make([]float32, P*in)
	for i := range X {
		X[i] = float32(math.Cos(float64(i) * 0.2))
	}
	Y := exl2Gemm(qt, X, P)
	for tk := 0; tk < P; tk++ {
		want := exl2MatRows(qt, X[tk*in:(tk+1)*in])
		for o := 0; o < out; o++ {
			if g := Y[tk*out+o]; g != want[o] {
				t.Fatalf("GEMM[%d,%d]=%v != GEMV %v", tk, o, g, want[o])
			}
		}
	}
}

// TestEXL2BuildColGroupRejects checks the malformed-partition guards.
func TestEXL2BuildColGroupRejects(t *testing.T) {
	cases := []struct {
		name   string
		groups []exl2Group
		in     int
	}{
		{"gap", []exl2Group{{bits: 4, off: 0, size: 8}, {bits: 4, off: 16, size: 8}}, 24},
		{"overrun", []exl2Group{{bits: 4, off: 0, size: 40}}, 32},
		{"badbits", []exl2Group{{bits: 7, off: 0, size: 32}}, 32},
		{"undercover", []exl2Group{{bits: 4, off: 0, size: 16}}, 32},
		{"nonpos", []exl2Group{{bits: 4, off: 0, size: 0}}, 0},
	}
	for _, c := range cases {
		if _, err := exl2BuildColGroup(c.groups, c.in); err == nil {
			t.Fatalf("%s: expected error, got nil", c.name)
		}
	}
}

// ---- synthetic safetensors writer (test-only) -------------------------------

type stTestTensor struct {
	dtype string
	shape []int
	data  []byte
}

// writeSafetensorsFile writes a minimal valid single-file safetensors checkpoint
// (8-byte header length + JSON header + concatenated tensor data) so LoadEXL2 can be
// driven end to end without a real multi-GB ExLlamaV2 export.
func writeSafetensorsFile(t *testing.T, path string, tensors map[string]stTestTensor) {
	t.Helper()
	type entry struct {
		Dtype       string `json:"dtype"`
		Shape       []int  `json:"shape"`
		DataOffsets []int  `json:"data_offsets"`
	}
	names := make([]string, 0, len(tensors))
	for n := range tensors {
		names = append(names, n)
	}
	sort.Strings(names)
	hdr := map[string]entry{}
	var data []byte
	off := 0
	for _, n := range names {
		tt := tensors[n]
		hdr[n] = entry{Dtype: tt.dtype, Shape: tt.shape, DataOffsets: []int{off, off + len(tt.data)}}
		data = append(data, tt.data...)
		off += len(tt.data)
	}
	hj, err := json.Marshal(hdr)
	if err != nil {
		t.Fatalf("marshal header: %v", err)
	}
	var buf []byte
	var lenBuf [8]byte
	binary.LittleEndian.PutUint64(lenBuf[:], uint64(len(hj)))
	buf = append(buf, lenBuf[:]...)
	buf = append(buf, hj...)
	buf = append(buf, data...)
	if err := os.WriteFile(path, buf, 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

func i16Bytes(v []int) []byte {
	b := make([]byte, len(v)*2)
	for i, x := range v {
		binary.LittleEndian.PutUint16(b[i*2:], uint16(int16(x)))
	}
	return b
}

func i32Bytes(v []int32) []byte {
	b := make([]byte, len(v)*4)
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[i*4:], uint32(x))
	}
	return b
}

func u32Bytes(v []uint32) []byte {
	b := make([]byte, len(v)*4)
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[i*4:], x)
	}
	return b
}

func f32Bytes(v []float32) []byte {
	b := make([]byte, len(v)*4)
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(x))
	}
	return b
}

// TestLoadEXL2RoundTrip builds an EXL2 checkpoint on disk from a quantized matrix,
// loads it via LoadEXL2 (config + mmap safetensors + metadata parse + two-level
// scale reconstruction), and checks the loaded tensor reproduces the in-memory
// dequant exactly and tracks the original FP32 to cosine >= 0.995.
func TestLoadEXL2RoundTrip(t *testing.T) {
	out, in := 8, 64
	groupBits := []int{4, 8}
	groupSizes := []int{32, 32}
	w := make([]float32, out*in)
	for i := range w {
		w[i] = float32(math.Sin(float64(i)*0.17) * 0.6)
	}
	// Non-identity act-order permutation.
	perm := make([]int32, in)
	for i := range perm {
		perm[i] = int32((i*3 + 11) % in)
	}
	qt, err := exl2Quantize(w, out, in, groupBits, groupSizes, perm)
	if err != nil {
		t.Fatalf("quantize: %v", err)
	}

	// Split each effective per-(out,group) scale into a two-level (q_scale,
	// q_scale_max) form with a power-of-two max so the f32 product is exact.
	G := len(groupBits)
	scaleMax := make([]float32, G)
	for g := range scaleMax {
		scaleMax[g] = 2.0
	}
	scaleMul := make([]float32, out*G)
	for o := 0; o < out; o++ {
		for g := 0; g < G; g++ {
			scaleMul[o*G+g] = qt.scales[o*G+g] / scaleMax[g]
		}
	}
	groupsI16 := make([]int, 0, 2*G)
	for g := 0; g < G; g++ {
		groupsI16 = append(groupsI16, groupBits[g], groupSizes[g])
	}

	dir := t.TempDir()
	writeSafetensorsFile(t, filepath.Join(dir, "model.safetensors"), map[string]stTestTensor{
		"blk.q_weight":    {dtype: "I32", shape: []int{out, qt.rowWords}, data: u32Bytes(qt.codes)},
		"blk.q_groups":    {dtype: "I16", shape: []int{2 * G}, data: i16Bytes(groupsI16)},
		"blk.q_invperm":   {dtype: "I32", shape: []int{in}, data: i32Bytes(perm)},
		"blk.q_scale":     {dtype: "F32", shape: []int{out, G}, data: f32Bytes(scaleMul)},
		"blk.q_scale_max": {dtype: "F32", shape: []int{G}, data: f32Bytes(scaleMax)},
	})
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"hidden_size":64,"num_hidden_layers":1}`), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	if !IsEXL2Dir(dir) {
		t.Fatalf("IsEXL2Dir = false on a .q_weight export")
	}
	m, err := LoadEXL2(dir)
	if err != nil {
		t.Fatalf("LoadEXL2: %v", err)
	}
	if m.Count() != 1 {
		t.Fatalf("Count = %d, want 1", m.Count())
	}
	if !m.Has("blk") {
		t.Fatalf("Has(blk) = false")
	}
	go_, gin, gG := m.Shape("blk")
	if go_ != out || gin != in || gG != G {
		t.Fatalf("Shape = (%d,%d,%d), want (%d,%d,%d)", go_, gin, gG, out, in, G)
	}

	loaded := m.tensor("blk")
	deqLoaded := make([]float32, out*in)
	deqMem := make([]float32, out*in)
	row := make([]float32, in)
	for o := 0; o < out; o++ {
		exl2DequantRow(row, loaded, o)
		copy(deqLoaded[o*in:(o+1)*in], row)
		exl2DequantRow(row, qt, o)
		copy(deqMem[o*in:(o+1)*in], row)
	}
	// Power-of-two scale split is exact, so the loaded dequant must match the
	// in-memory quantizer bit-for-bit.
	for i := range deqLoaded {
		if deqLoaded[i] != deqMem[i] {
			t.Fatalf("loaded dequant[%d]=%v != in-memory %v", i, deqLoaded[i], deqMem[i])
		}
	}
	if cs := cosSim(w, deqLoaded); cs < 0.995 {
		t.Fatalf("loaded dequant cosine %.5f < 0.995", cs)
	}

	// The EXL2Model.MatRows compute seam must match the standalone GEMV.
	x := make([]float32, in)
	for i := range x {
		x[i] = float32(math.Cos(float64(i) * 0.3))
	}
	gotM := m.MatRows("blk", x)
	wantM := exl2MatRows(loaded, x)
	for o := 0; o < out; o++ {
		if gotM[o] != wantM[o] {
			t.Fatalf("MatRows[%d]=%v != GEMV %v", o, gotM[o], wantM[o])
		}
	}
}

// TestLoadEXL2Rejects checks the loader fails closed on a non-EXL2 directory.
func TestLoadEXL2Rejects(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{}`), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}
	if _, err := LoadEXL2(dir); err == nil {
		t.Fatalf("expected error with no model.safetensors")
	}
	if IsEXL2Dir(dir) {
		t.Fatalf("IsEXL2Dir = true with no safetensors")
	}
}

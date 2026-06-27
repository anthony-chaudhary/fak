package ggufload

import "testing"

// The IQ4_NL / IQ4_XS expectations below are hand-derived from the llama.cpp GGML
// reference (dequantize_row_iq4_nl / dequantize_row_iq4_xs + the kvalues_iq4nl
// codebook), NOT read back from this package's own dequant — so the test is an
// independent oracle, not a tautology. Every byte of the input block and every
// float of the expected output is written as a literal so the correspondence is
// auditable by eye.

// kvalues_iq4nl (the shared codebook), repeated here so the expectations below do
// not depend on the production table:
//
//	idx:  0    1    2    3    4    5    6    7   8   9  10  11  12  13  14   15
//	val: -127 -104 -83  -65  -49  -35  -22  -10  1  13  25  38  53  69  89  113

func TestDequantIQ4NLKnownAnswer(t *testing.T) {
	// One 32-element block. d = 2.0 (f16 0x4000, little-endian 0x00 0x40). qs[j]
	// packs the low nibble = j and the high nibble = 15-j, so the low nibble walks
	// the codebook up and the high nibble walks it down — catching any low/high
	// swap or position error. y[2j] = d*kvalues[j]; y[2j+1] = d*kvalues[15-j].
	raw := []byte{
		0x00, 0x40, // d = 2.0
		0xF0, 0xE1, 0xD2, 0xC3, 0xB4, 0xA5, 0x96, 0x87,
		0x78, 0x69, 0x5A, 0x4B, 0x3C, 0x2D, 0x1E, 0x0F,
	}
	if len(raw) != blockIQ4NLBytes {
		t.Fatalf("test block is %d bytes, want blockIQ4NLBytes=%d", len(raw), blockIQ4NLBytes)
	}
	want := []float32{
		-254, 226, -208, 178, -166, 138, -130, 106, -98, 76, -70, 50, -44, 26, -20, 2,
		2, -20, 26, -44, 50, -70, 76, -98, 106, -130, 138, -166, 178, -208, 226, -254,
	}
	tns := TensorInfo{Name: "iq4nl.test", Dims: []uint64{32}, Type: TensorIQ4_NL}
	got, err := dequantF32(tns, raw)
	if err != nil {
		t.Fatalf("dequantF32 IQ4_NL: %v", err)
	}
	assertF32Equal(t, got, want)
}

func TestDequantIQ4XSScaleKnownAnswer(t *testing.T) {
	// One 256-element super-block exercising the per-sub-block scale unpacking. d =
	// 1.0 (f16 0x3C00). Every 4-bit code is index 8 (kvalues[8]=1), so each output
	// reduces to dl = d*(ls-32) = ls-32, isolating the 6-bit scale. The eight ls
	// values are chosen so the low 4 bits (scales_l nibbles) span 0..7 and the high
	// 2 bits (scales_h fields) span 0..3:
	//   ib:  0   1   2   3   4   5   6   7
	//   ls: 32  49  18   3  36  53  22   7   -> dl: 0 17 -14 -29 4 21 -10 -25
	raw := make([]byte, blockIQ4XSBytes)
	raw[0], raw[1] = 0x00, 0x3C                             // d = 1.0
	raw[2], raw[3] = 0x1E, 0x1E                             // scales_h = 0x1E1E (high 2 bits per sub-block)
	raw[4], raw[5], raw[6], raw[7] = 0x10, 0x32, 0x54, 0x76 // scales_l (low 4 bits)
	for i := 8; i < blockIQ4XSBytes; i++ {
		raw[i] = 0x88 // both nibbles = index 8 -> kvalues[8] = 1
	}
	dl := []float32{0, 17, -14, -29, 4, 21, -10, -25}
	want := make([]float32, 256)
	for ib := 0; ib < 8; ib++ {
		for j := 0; j < 32; j++ {
			want[ib*32+j] = dl[ib]
		}
	}
	tns := TensorInfo{Name: "iq4xs.scale.test", Dims: []uint64{256}, Type: TensorIQ4_XS}
	got, err := dequantF32(tns, raw)
	if err != nil {
		t.Fatalf("dequantF32 IQ4_XS scale: %v", err)
	}
	assertF32Equal(t, got, want)
}

func TestDequantIQ4XSCodebookKnownAnswer(t *testing.T) {
	// One 256-element super-block exercising the codebook + nibble mapping. d = 1.0
	// and every sub-block scale ls = 33 (scales_l nibbles all 1, scales_h fields all
	// 2 -> 0xAAAA), so dl = d*(33-32) = 1 and each output is exactly kvalues[code].
	// The 16 qs bytes per sub-block reuse the IQ4_NL pattern (low nibble = j, high
	// nibble = 15-j), so y[j] = kvalues[j] and y[j+16] = kvalues[15-j].
	raw := make([]byte, blockIQ4XSBytes)
	raw[0], raw[1] = 0x00, 0x3C                             // d = 1.0
	raw[2], raw[3] = 0xAA, 0xAA                             // scales_h = 0xAAAA -> every high field = 2
	raw[4], raw[5], raw[6], raw[7] = 0x11, 0x11, 0x11, 0x11 // every scales_l nibble = 1
	qs := []byte{
		0xF0, 0xE1, 0xD2, 0xC3, 0xB4, 0xA5, 0x96, 0x87,
		0x78, 0x69, 0x5A, 0x4B, 0x3C, 0x2D, 0x1E, 0x0F,
	}
	for ib := 0; ib < 8; ib++ {
		copy(raw[8+ib*16:], qs)
	}
	sub := []float32{
		-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
		113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127,
	}
	want := make([]float32, 256)
	for ib := 0; ib < 8; ib++ {
		copy(want[ib*32:], sub)
	}
	tns := TensorInfo{Name: "iq4xs.codebook.test", Dims: []uint64{256}, Type: TensorIQ4_XS}
	got, err := dequantF32(tns, raw)
	if err != nil {
		t.Fatalf("dequantF32 IQ4_XS codebook: %v", err)
	}
	assertF32Equal(t, got, want)
}

// TestIQ4PayloadSizingAndNames locks the on-disk block sizes, the type-name strings,
// and the dispatch guards (wrong-length payload, non-multiple element count) that the
// loader relies on to admit or reject an IQ4 tensor.
func TestIQ4PayloadSizingAndNames(t *testing.T) {
	if blockIQ4NLBytes != 18 {
		t.Errorf("blockIQ4NLBytes = %d, want 18 (2 + 32/2)", blockIQ4NLBytes)
	}
	if blockIQ4XSBytes != 136 {
		t.Errorf("blockIQ4XSBytes = %d, want 136 (2 + 2 + 4 + 128)", blockIQ4XSBytes)
	}
	if got := TensorIQ4_NL.String(); got != "IQ4_NL" {
		t.Errorf("TensorIQ4_NL.String() = %q, want IQ4_NL", got)
	}
	if got := TensorIQ4_XS.String(); got != "IQ4_XS" {
		t.Errorf("TensorIQ4_XS.String() = %q, want IQ4_XS", got)
	}

	// tensorPayloadBytes must agree with the per-block sizes across two blocks.
	nl := TensorInfo{Name: "nl", Dims: []uint64{64}, Type: TensorIQ4_NL}
	if got, err := tensorPayloadBytes(nl); err != nil || got != 2*blockIQ4NLBytes {
		t.Errorf("tensorPayloadBytes IQ4_NL(64) = %d, %v; want %d", got, err, 2*blockIQ4NLBytes)
	}
	xs := TensorInfo{Name: "xs", Dims: []uint64{512}, Type: TensorIQ4_XS}
	if got, err := tensorPayloadBytes(xs); err != nil || got != 2*blockIQ4XSBytes {
		t.Errorf("tensorPayloadBytes IQ4_XS(512) = %d, %v; want %d", got, err, 2*blockIQ4XSBytes)
	}

	// A short payload must be rejected, not read out of bounds.
	if _, err := dequantF32(nl, make([]byte, 2*blockIQ4NLBytes-1)); err == nil {
		t.Error("dequantF32 IQ4_NL accepted a short payload, want error")
	}
	if _, err := dequantF32(xs, make([]byte, 2*blockIQ4XSBytes-1)); err == nil {
		t.Error("dequantF32 IQ4_XS accepted a short payload, want error")
	}

	// A non-block-multiple element count must be rejected.
	badNL := TensorInfo{Name: "nl", Dims: []uint64{40}, Type: TensorIQ4_NL}
	if _, err := dequantF32(badNL, make([]byte, blockIQ4NLBytes*2)); err == nil {
		t.Error("dequantF32 IQ4_NL accepted a non-multiple element count, want error")
	}
	badXS := TensorInfo{Name: "xs", Dims: []uint64{300}, Type: TensorIQ4_XS}
	if _, err := dequantF32(badXS, make([]byte, blockIQ4XSBytes*2)); err == nil {
		t.Error("dequantF32 IQ4_XS accepted a non-multiple element count, want error")
	}
}

func assertF32Equal(t *testing.T, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("got %d values, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("element %d = %v, want %v", i, got[i], want[i])
		}
	}
}

package ggufload

import (
	"bytes"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// TestWeightSourceDequantizesQ4_0AndQ4_1 is the issue #489 default-path witness for the
// legacy GGML 4-bit 32-element blocks (Q4_0 = ggml_type 2, Q4_1 = ggml_type 3). These are
// the same 32-element block geometry as the already-supported Q5_0/Q5_1/Q8_0 — NOT the
// out-of-scope Q4_K/Q6_K 256-element super-blocks — and a Q4_0 GGUF is one of the most
// common legacy llama.cpp quantizations, so the reader must ingest it on the
// dequant-to-f32 default path.
//
// The golden f32 bytes are DERIVED from the GGUF/GGML block arithmetic, not invented; the
// derivation is shown inline in q40FixtureBlock / q41FixtureBlock and re-stated here:
//
//	Q4_0 block = f16 d, then qk4/2 bytes; nibble j -> element j, nibble (j>>4) -> element
//	            j+qk4/2; y = (nibble - 8) * d        (dequantize_row_q4_0)
//	Q4_1 block = f16 d, f16 m, then qk4/2 bytes; same interleave; y = nibble*d + m
//	            (dequantize_row_q4_1)
//
// The whole reader path is exercised: header + tensor-directory parse, payload sizing via
// tensorPayloadBytes (Q4_0=18 B/block, Q4_1=20 B/block), and dequant via TensorF32.
func TestWeightSourceDequantizesQ4_0AndQ4_1(t *testing.T) {
	if got := TensorQ4_0.String(); got != "Q4_0" {
		t.Fatalf("TensorQ4_0.String() = %q, want Q4_0", got)
	}
	if got := TensorQ4_1.String(); got != "Q4_1" {
		t.Fatalf("TensorQ4_1.String() = %q, want Q4_1", got)
	}

	q40Block, q40Want := q40FixtureBlock()
	q41Block, q41Want := q41FixtureBlock()
	if len(q40Block) != blockQ4_0Bytes {
		t.Fatalf("Q4_0 block is %d bytes, want %d", len(q40Block), blockQ4_0Bytes)
	}
	if len(q41Block) != blockQ4_1Bytes {
		t.Fatalf("Q4_1 block is %d bytes, want %d", len(q41Block), blockQ4_1Bytes)
	}

	var b bytes.Buffer
	writeMinimalHeader(&b, 2, 1)
	writeKVUint32(&b, "general.alignment", 32)
	writeTensorInfoForTest(&b, "q4_0.weight", []uint64{qk4}, TensorQ4_0, 0)
	writeTensorInfoForTest(&b, "q4_1.weight", []uint64{qk4}, TensorQ4_1, 32)
	padToAlignment(&b, 32)
	dataStart := b.Len()
	b.Write(q40Block)
	padToLen(&b, dataStart+32)
	b.Write(q41Block)

	path := filepath.Join(t.TempDir(), "q4.gguf")
	if err := os.WriteFile(path, b.Bytes(), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	ws, err := OpenWeights(path)
	if err != nil {
		t.Fatalf("OpenWeights: %v", err)
	}
	defer ws.Close()

	assertTensor := func(name string, want []float32) {
		t.Helper()
		got, info, err := ws.TensorF32(name)
		if err != nil {
			t.Fatalf("TensorF32(%s): %v", name, err)
		}
		if len(got) != len(want) {
			t.Fatalf("%s len=%d, want %d (info=%#v)", name, len(got), len(want), info)
		}
		for i := range got {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("%s[%d]=%v bits=%#x, want %v bits=%#x",
					name, i, got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
			}
		}
	}
	assertTensor("q4_0.weight", q40Want)
	assertTensor("q4_1.weight", q41Want)
}

// q40FixtureBlock builds one real Q4_0 block and the f32 values it must dequantize to.
// Scale d = 2.0 (f16 0x4000). For byte j in [0,16): low nibble = j, high nibble = 15-j.
// GGML interleave + recenter-by-8: element[j] = (j-8)*2, element[j+16] = ((15-j)-8)*2.
func q40FixtureBlock() ([]byte, []float32) {
	want := make([]float32, qk4)
	qs := make([]byte, qk4/2)
	for j := 0; j < qk4/2; j++ {
		low := byte(j)
		high := byte(15 - j)
		qs[j] = low | (high << 4)
		want[j] = float32(int(low)-8) * 2.0
		want[j+qk4/2] = float32(int(high)-8) * 2.0
	}
	var b bytes.Buffer
	writeU16ForTest(&b, 0x4000) // d = 2.0
	b.Write(qs)
	return b.Bytes(), want
}

// q41FixtureBlock builds one real Q4_1 block and its f32 values. Scale d = 1.0 (f16
// 0x3c00), min m = 0.5 (f16 0x3800). Same nibble interleave as Q4_0 but affine, no
// recenter: element[j] = j*1 + 0.5, element[j+16] = (15-j)*1 + 0.5.
func q41FixtureBlock() ([]byte, []float32) {
	want := make([]float32, qk4)
	qs := make([]byte, qk4/2)
	for j := 0; j < qk4/2; j++ {
		low := byte(j)
		high := byte(15 - j)
		qs[j] = low | (high << 4)
		want[j] = float32(low)*1.0 + 0.5
		want[j+qk4/2] = float32(high)*1.0 + 0.5
	}
	var b bytes.Buffer
	writeU16ForTest(&b, 0x3c00) // d = 1.0
	writeU16ForTest(&b, 0x3800) // m = 0.5
	b.Write(qs)
	return b.Bytes(), want
}

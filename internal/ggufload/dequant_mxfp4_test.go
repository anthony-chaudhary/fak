package ggufload

import (
	"bytes"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// TestWeightSourceDequantizesMXFP4 is the issue #489 witness for the MXFP4
// (gpt-oss, ggml_type 39) micro-scaled FP4 block — the last dequant-on-load gap
// in the ecosystem loader. An MXFP4 block is a 1-byte E8M0 shared exponent + 16
// bytes of packed 4-bit E2M1 codes (32 elements). The golden f32 bytes are DERIVED
// from the GGML block arithmetic, not invented:
//
//	MXFP4 block = u8 e (E8M0), then qkMXFP4/2 bytes; nibble j -> element j, nibble
//	             (j>>4) -> element j+qkMXFP4/2; y = kvaluesMXFP4[code] * 2^(e-128)
//	             (dequantize_row_mxfp4 with the GGML_E8M0_TO_FP32_HALF pairing)
//
// The whole reader path is exercised: header + tensor-directory parse, payload
// sizing via tensorPayloadBytes (17 B/block) and dequant via TensorF32.
func TestWeightSourceDequantizesMXFP4(t *testing.T) {
	if got := TensorMXFP4.String(); got != "MXFP4" {
		t.Fatalf("TensorMXFP4.String() = %q, want MXFP4", got)
	}

	block, want := mxfp4FixtureBlock()
	if len(block) != blockMXFP4Bytes {
		t.Fatalf("MXFP4 block is %d bytes, want %d", len(block), blockMXFP4Bytes)
	}

	var b bytes.Buffer
	writeMinimalHeader(&b, 1, 1)
	writeKVUint32(&b, "general.alignment", 32)
	writeTensorInfoForTest(&b, "mxfp4.weight", []uint64{qkMXFP4}, TensorMXFP4, 0)
	padToAlignment(&b, 32)
	dataStart := b.Len()
	b.Write(block)
	_ = dataStart

	path := filepath.Join(t.TempDir(), "mxfp4.gguf")
	if err := os.WriteFile(path, b.Bytes(), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	ws, err := OpenWeights(path)
	if err != nil {
		t.Fatalf("OpenWeights: %v", err)
	}
	defer ws.Close()

	got, info, err := ws.TensorF32("mxfp4.weight")
	if err != nil {
		t.Fatalf("TensorF32: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("len=%d, want %d (info=%#v)", len(got), len(want), info)
	}
	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("mxfp4.weight[%d]=%v bits=%#x, want %v bits=%#x",
				i, got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
		}
	}
}

// mxfp4FixtureBlock builds one real MXFP4 block and the f32 values it must
// dequantize to. E8M0 scale byte e = 127 => d = 2^(127-128) = 0.5, which folds
// against the doubled kvaluesMXFP4 table to recover the exact E2M1 value set
// {0,.5,1,1.5,2,3,4,6, 0,-.5,-1,-1.5,-2,-3,-4,-6}. For byte j in [0,16): low
// nibble = j, high nibble = 15-j.
func mxfp4FixtureBlock() ([]byte, []float32) {
	const e = uint8(127) // d = 0.5
	d := float32(0.5)
	want := make([]float32, qkMXFP4)
	qs := make([]byte, qkMXFP4/2)
	for j := 0; j < qkMXFP4/2; j++ {
		low := byte(j)
		high := byte(15 - j)
		qs[j] = low | (high << 4)
		want[j] = kvaluesMXFP4[low] * d
		want[j+qkMXFP4/2] = kvaluesMXFP4[high] * d
	}
	var b bytes.Buffer
	b.WriteByte(e)
	b.Write(qs)
	return b.Bytes(), want
}

// TestE8M0ToF32Half locks the half-scaled E8M0 exponent decode that pairs with
// the doubled kvaluesMXFP4 table.
func TestE8M0ToF32Half(t *testing.T) {
	cases := []struct {
		e    uint8
		want float32
	}{
		{128, 1.0},  // 2^0
		{127, 0.5},  // 2^-1
		{129, 2.0},  // 2^1
		{131, 8.0},  // 2^3
		{125, 0.125}, // 2^-3
	}
	for _, c := range cases {
		if got := e8m0ToF32Half(c.e); got != c.want {
			t.Fatalf("e8m0ToF32Half(%d) = %v, want %v", c.e, got, c.want)
		}
	}
	// The doubled kvalues table must recover the canonical E2M1 magnitudes when
	// scaled by 0.5 (the d that e=127 produces).
	wantE2M1 := [16]float32{0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6}
	for i, w := range wantE2M1 {
		if got := kvaluesMXFP4[i] * 0.5; got != w {
			t.Fatalf("kvaluesMXFP4[%d]*0.5 = %v, want E2M1 %v", i, got, w)
		}
	}
}

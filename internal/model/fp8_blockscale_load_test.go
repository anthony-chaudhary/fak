package model

import (
	"math"
	"path/filepath"
	"testing"
)

// TestAppendSafetensorsFP8BlockScale witnesses the #4360 load-loop wiring end to end:
// a float8_e4m3fn weight paired with an F32 `weight_scale_inv` companion is dequantized
// per 128x128 block into a resident f32 tensor, and the scale companion is CONSUMED —
// it must not survive as a spurious standalone tensor in the manifest. The weight/scale
// values reuse the single-block golden from TestDecodeFP8BlockScaleSingleBlock
// (weight [[1,2],[0.5,-1]] * scaleInv 3 = [[3,6],[1.5,-3]]) so a decode regression here
// and in the pure-math test fail together rather than drifting apart.
func TestAppendSafetensorsFP8BlockScale(t *testing.T) {
	const wName = "model.layers.0.self_attn.q_proj.weight"
	const sName = wName + "_scale_inv"
	tensors := map[string]stTestTensor{
		// [[1.0, 2.0], [0.5, -1.0]] as float8_e4m3fn bytes.
		wName: {dtype: "F8_E4M3", shape: []int{2, 2}, data: []byte{0x38, 0x40, 0x30, 0xB8}},
		// One f32 scale for the single 128x128 block covering the whole 2x2 weight.
		sName: {dtype: "F32", shape: []int{1, 1}, data: f32Bytes([]float32{3.0})},
	}
	path := filepath.Join(t.TempDir(), "model.safetensors")
	writeSafetensorsFile(t, path, tensors)

	sf, err := openSafetensorsFileDataBackedForTest(path)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer sf.Close()
	man := map[string]tensorMeta{}
	var raw []byte
	off := 0
	if err := appendSafetensorsFileInto(sf, man, &raw, &off, Config{}); err != nil {
		t.Fatalf("appendSafetensorsFileInto: %v", err)
	}

	// The scale companion is consumed, never loaded as its own tensor.
	if _, ok := man[sName]; ok {
		t.Fatalf("scale companion %s leaked into the manifest as a standalone tensor", sName)
	}
	// The fp8 weight is present as a dequantized f32 [2,2] tensor.
	meta, ok := man[wName]
	if !ok {
		t.Fatalf("fp8 weight %s missing from manifest", wName)
	}
	if meta.Dtype != "f32" {
		t.Errorf("weight dtype = %q, want f32", meta.Dtype)
	}
	if len(meta.Shape) != 2 || meta.Shape[0] != 2 || meta.Shape[1] != 2 {
		t.Errorf("weight shape = %v, want [2 2]", meta.Shape)
	}
	if meta.Nbytes != 16 {
		t.Fatalf("weight nbytes = %d, want 16 (4 f32)", meta.Nbytes)
	}
	want := []float32{3.0, 6.0, 1.5, -3.0}
	for i, w := range want {
		got := math.Float32frombits(leU32(raw[meta.Offset+i*4:]))
		if got != w {
			t.Errorf("dequant[%d] = %v, want %v", i, got, w)
		}
	}
}

// leU32 reads a little-endian uint32 from b without pulling encoding/binary into this
// test file for a single use.
func leU32(b []byte) uint32 {
	return uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24
}

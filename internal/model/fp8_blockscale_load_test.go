package model

import (
	"math"
	"os"
	"path/filepath"
	"strings"
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

func TestLoadSafetensorsQuantDirFP8BlockScaleQwen36(t *testing.T) {
	const wName = "model.language_model.layers.0.self_attn.q_proj.weight"
	const sName = wName + "_scale_inv"
	path := writeTinySafetensors(t, map[string]tinySTTensor{
		wName: {dtype: "F8_E4M3", shape: []int{1, 32}, data: append([]byte{0x38, 0x40, 0x30, 0xB8}, make([]byte, 28)...)},
		sName: {dtype: "F32", shape: []int{1, 1}, data: f32Bytes([]float32{3})},
	})

	m, err := LoadSafetensorsQuant(path, Config{ModelType: "qwen3_5", HiddenSize: 32, LayerTypes: []string{"linear_attention"}})
	if err != nil {
		t.Fatal(err)
	}
	const canonical = "model.layers.0.self_attn.q_proj.weight"
	q, ok := m.q8w[canonical]
	if !ok {
		t.Fatalf("FP8 weight was not normalized and quantized as %s", canonical)
	}
	if q.out != 1 || q.in != 32 {
		t.Fatalf("quantized shape = [%d,%d], want [1,32]", q.out, q.in)
	}
	if _, ok := m.manifest[sName]; ok {
		t.Fatalf("scale companion %s survived as a runtime tensor", sName)
	}
}

func TestLoadSafetensorsQuantConfigDirRefusesWrongQwen36FP8Geometry(t *testing.T) {
	dir := t.TempDir()
	config := `{
		"architectures":["Qwen3_5ForConditionalGeneration"],
		"model_type":"qwen3_5",
		"text_config":{"model_type":"qwen3_5_text","hidden_size":32,"num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,"head_dim":32,"intermediate_size":64,"vocab_size":32,"layer_types":["linear_attention"]},
		"quantization_config":{"quant_method":"fp8","fmt":"e4m3","activation_scheme":"dynamic","weight_block_size":[64,128]}
	}`
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(config), 0o644); err != nil {
		t.Fatal(err)
	}
	_, err := LoadSafetensorsQuantConfigDir(dir)
	if err == nil || !strings.Contains(err.Error(), "unsupported Qwen3.6 quantization geometry") {
		t.Fatalf("error = %v, want fail-closed geometry refusal", err)
	}
}

func TestLoadSafetensorsQuantDirRefusesIncompleteIndex(t *testing.T) {
	dir := t.TempDir()
	shard := filepath.Join(dir, "model-00001-of-00001.safetensors")
	if err := os.WriteFile(shard, tinySafetensorsBytes(t, map[string]tinySTTensor{
		"model.layers.0.self_attn.q_proj.weight": {dtype: "F32", shape: []int{1, 32}, data: f32Bytes(make([]float32, 32))},
	}), 0o644); err != nil {
		t.Fatal(err)
	}
	index := `{"weight_map":{"model.layers.0.self_attn.q_proj.weight":"model-00001-of-00001.safetensors","model.layers.0.self_attn.k_proj.weight":"model-00001-of-00001.safetensors"}}`
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), []byte(index), 0o644); err != nil {
		t.Fatal(err)
	}
	_, err := LoadSafetensorsQuantDir(dir, Config{HiddenSize: 32})
	if err == nil || !strings.Contains(err.Error(), "index declares tensor model.layers.0.self_attn.k_proj.weight") {
		t.Fatalf("error = %v, want incomplete-index refusal", err)
	}
}

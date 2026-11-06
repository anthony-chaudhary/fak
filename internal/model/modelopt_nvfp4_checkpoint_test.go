package model

import (
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// NVFP4 layout under test (ModelOpt HF export, as consumed by vLLM's
// ModelOptNvFp4LinearMethod): <base>.weight U8 [out,k/2] packs two E2M1 codes
// per byte (even k in the low nibble); <base>.weight_scale F8_E4M3 [out,k/16]
// holds one block scale per 16 values; <base>.weight_scale_2 F32 scalar is the
// per-tensor global scale; value = E2M1 * blockScale * globalScale.
// https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/modelopt.py
// https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/0638a274cf195870564d8e3c164e4f859e5be2a2/modelopt/torch/export/unified_export_hf.py
var nvfp4TestE2M1 = []float32{0, 0.5, 1, 1.5, 2, 3, 4, 6}

// nvfp4TestTriple builds an analytic packed triple: code(r,i) = (r+i)%8, block
// scale 1.0 (E4M3 0x38) for the first 16 columns and 2.0 (0x40) afterwards.
func nvfp4TestTriple(base string, out, k int, hf bool, global float32) map[string]tinySTTensor {
	packed := make([]byte, out*k/2)
	for r := 0; r < out; r++ {
		for i := 0; i < k; i++ {
			idx := r*k + i
			code := byte((r + i) % 8)
			if idx%2 == 0 {
				packed[idx/2] |= code
			} else {
				packed[idx/2] |= code << 4
			}
		}
	}
	scales := make([]byte, out*k/16)
	for r := 0; r < out; r++ {
		for b := 0; b < k/16; b++ {
			scales[r*(k/16)+b] = 0x38
			if b > 0 {
				scales[r*(k/16)+b] = 0x40
			}
		}
	}
	s1, s2 := ".weight_scale", ".weight_scale_2"
	if !hf {
		s1, s2 = ".weights_scaling_factor", ".weights_scaling_factor_2"
	}
	return map[string]tinySTTensor{
		base + ".weight": {dtype: "U8", shape: []int{out, k / 2}, data: packed},
		base + s1:        {dtype: "F8_E4M3", shape: []int{out, k / 16}, data: scales},
		base + s2:        {dtype: "F32", shape: []int{}, data: f32TestBytes([]float32{global})},
	}
}

func nvfp4TestWant(out, k int, global float32) []float32 {
	w := make([]float32, out*k)
	for r := 0; r < out; r++ {
		for i := 0; i < k; i++ {
			scale := float32(1)
			if i >= 16 {
				scale = 2
			}
			w[r*k+i] = nvfp4TestE2M1[(r+i)%8] * scale * global
		}
	}
	return w
}

func mergeST(parts ...map[string]tinySTTensor) map[string]tinySTTensor {
	out := map[string]tinySTTensor{}
	for _, p := range parts {
		for k, v := range p {
			out[k] = v
		}
	}
	return out
}

// writeModelOptShards writes each shard plus a model.safetensors.index.json.
func writeModelOptShards(t *testing.T, dir string, shards map[string]map[string]tinySTTensor) {
	t.Helper()
	weightMap := map[string]string{}
	for shard, tensors := range shards {
		if err := os.WriteFile(filepath.Join(dir, shard), tinySafetensorsBytes(t, tensors), 0o644); err != nil {
			t.Fatal(err)
		}
		for name := range tensors {
			weightMap[name] = shard
		}
	}
	b, err := json.Marshal(map[string]any{"weight_map": weightMap})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), b, 0o644); err != nil {
		t.Fatal(err)
	}
}

func writeTestFile(t *testing.T, dir, name, body string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestReadModelOptNVFP4ConfigForms(t *testing.T) {
	cases := []struct {
		name, config, hfQuant string
		ok                    bool
		err                   error
	}{
		{"config.json", `{"quantization_config":{"quant_method":"modelopt","quant_algo":"NVFP4","group_size":16,"exclude_modules":["lm_head"]}}`, "", true, nil},
		{"hf_quant_config", `{}`, `{"producer":{"name":"modelopt"},"quantization":{"quant_algo":"NVFP4","group_size":16}}`, true, nil},
		{"absent", `{"quantization_config":{"quant_method":"gptq","bits":4}}`, "", false, nil},
		{"fp8 refused", `{"quantization_config":{"quant_method":"modelopt","quant_algo":"FP8"}}`, "", true, ErrModelOptNVFP4Config},
		{"group 32 refused", `{"quantization_config":{"quant_method":"modelopt","quant_algo":"NVFP4","group_size":32}}`, "", true, ErrModelOptNVFP4Config},
		{"disagreement refused", `{"quantization_config":{"quant_method":"modelopt","quant_algo":"NVFP4"}}`, `{"producer":{"name":"modelopt"},"quantization":{"quant_algo":"W4A8_AWQ"}}`, false, ErrModelOptNVFP4Config},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			writeTestFile(t, dir, "config.json", tc.config)
			if tc.hfQuant != "" {
				writeTestFile(t, dir, "hf_quant_config.json", tc.hfQuant)
			}
			got, ok, err := ReadModelOptNVFP4Config(dir)
			if tc.err != nil {
				if !errors.Is(err, tc.err) {
					t.Fatalf("err = %v, want %v", err, tc.err)
				}
				return
			}
			if err != nil || ok != tc.ok {
				t.Fatalf("ok=%v err=%v, want ok=%v", ok, err, tc.ok)
			}
			if ok && (got.QuantAlgo != "NVFP4" || got.GroupSize != compute.NVFP4BlockSize) {
				t.Fatalf("config = %+v", got)
			}
		})
	}
}

func qwen38NameTestConfig() Config {
	return Config{NumLayers: 2, LayerTypes: []string{"linear_attention", "full_attention"}}
}

// fak-test:runtime fast est=10ms lane=default
func TestModelOptNVFP4Qwen38CanonicalNames(t *testing.T) {
	cfg := qwen38NameTestConfig()
	for in, want := range map[string]string{
		"model.language_model.layers.0.linear_attn.in_proj_qkv.weight": "model.layers.0.linear_attn.in_proj_qkv.weight",
		"model.language_model.layers.0.self_attn.qkv_proj.weight":      "model.layers.0.linear_attn.in_proj_qkv.weight",
		"model.language_model.layers.0.self_attn.q_gate_proj.weight":   "model.layers.0.linear_attn.in_proj_z.weight",
		"model.language_model.layers.1.self_attn.q_proj.weight":        "model.layers.1.self_attn.q_proj.weight",
		"model.language_model.layers.1.mlp.down_proj.weight":           "model.layers.1.mlp.down_proj.weight",
	} {
		if got, keep := qwen35CanonicalWeightName(cfg, in); !keep || got != want {
			t.Errorf("%s -> %q keep=%v, want %q", in, got, keep, want)
		}
	}
	if _, keep := qwen35CanonicalWeightName(cfg, "model.visual.blocks.0.attn.qkv.weight"); keep && !RetainVision {
		t.Error("vision tower NVFP4 weight kept on a text load")
	}
	if got, _ := qwen35CanonicalWeightName(Config{NumLayers: 1}, "model.language_model.x.weight"); got != "model.language_model.x.weight" {
		t.Errorf("non-hybrid config renamed %q", got)
	}
}

// fak-test:runtime fast est=30ms lane=default
func TestModelOptNVFP4CheckpointShardedQwen38(t *testing.T) {
	const out, k = 4, 32
	dir := t.TempDir()
	qkv := nvfp4TestTriple("model.language_model.layers.0.linear_attn.in_proj_qkv", out, k, true, 0.5)
	qkv["model.language_model.layers.0.linear_attn.in_proj_qkv.input_scale"] = tinySTTensor{dtype: "F32", shape: []int{}, data: f32TestBytes([]float32{0.25})}
	writeModelOptShards(t, dir, map[string]map[string]tinySTTensor{
		"model-00001-of-00002.safetensors": mergeST(qkv, map[string]tinySTTensor{
			"model.language_model.embed_tokens.weight": {dtype: "F32", shape: []int{2, k}, data: f32TestBytes(make([]float32, 2*k))},
		}),
		"model-00002-of-00002.safetensors": mergeST(
			nvfp4TestTriple("model.language_model.layers.1.self_attn.q_proj", out, k, false, 0.25),
			nvfp4TestTriple("model.visual.blocks.0.attn.qkv", out, k, true, 1),
		),
	})
	ck, err := OpenModelOptNVFP4Checkpoint(dir, qwen38NameTestConfig(), compute.Default())
	if err != nil {
		t.Fatal(err)
	}
	defer ck.Close()
	wantNames := []string{"model.layers.0.linear_attn.in_proj_qkv.weight", "model.layers.1.self_attn.q_proj.weight"}
	if RetainVision {
		t.Skip("RetainVision changes the expected name set")
	}
	if got := ck.Names(); !reflect.DeepEqual(got, wantNames) {
		t.Fatalf("Names = %v, want %v", got, wantNames)
	}
	if !ck.ownsCheckpointTensor("model.language_model.layers.0.linear_attn.in_proj_qkv.input_scale") ||
		ck.ownsCheckpointTensor("model.language_model.embed_tokens.weight") {
		t.Fatal("NVFP4 auxiliary ownership misclassified")
	}
	for name, global := range map[string]float32{wantNames[0]: 0.5, wantNames[1]: 0.25} {
		w, err := ck.Weight(name, compute.NVFP4)
		if err != nil {
			t.Fatal(err)
		}
		hb := w.Buf().(compute.HostBuffer)
		payload := i8AsBytesForTest(hb.I8())
		got, err := compute.DecodeNVFP4Reference(out, k, payload[:out*k/2], payload[out*k/2:], w.Quant.Scale[0])
		if err != nil {
			t.Fatal(err)
		}
		if want := nvfp4TestWant(out, k, global); !reflect.DeepEqual(got, want) {
			t.Fatalf("%s decode = %v, want %v", name, got, want)
		}
	}
}

func i8AsBytesForTest(v []int8) []byte {
	b := make([]byte, len(v))
	for i, x := range v {
		b[i] = byte(x)
	}
	return b
}

// fak-test:runtime fast est=10ms lane=default
func TestModelOptNVFP4CheckpointRefusesSplitTriple(t *testing.T) {
	const out, k = 4, 32
	triple := nvfp4TestTriple("model.layers.0.mlp.up_proj", out, k, true, 1)
	scale2 := "model.layers.0.mlp.up_proj.weight_scale_2"
	moved := map[string]tinySTTensor{scale2: triple[scale2]}
	delete(triple, scale2)
	dir := t.TempDir()
	writeModelOptShards(t, dir, map[string]map[string]tinySTTensor{
		"a.safetensors": triple,
		"b.safetensors": moved,
	})
	if _, err := OpenModelOptNVFP4Checkpoint(dir, Config{NumLayers: 1}, compute.Default()); !errors.Is(err, ErrModelOptNVFP4Checkpoint) {
		t.Fatalf("err = %v, want ErrModelOptNVFP4Checkpoint", err)
	}
}

const modelOptTinyLlamaConfig = `{
	"model_type":"llama",
	"hidden_size":32,
	"num_hidden_layers":1,
	"num_attention_heads":4,
	"num_key_value_heads":4,
	"head_dim":8,
	"intermediate_size":64,
	"vocab_size":2,
	"rms_norm_eps":1e-5,
	"quantization_config":{"quant_method":"modelopt","quant_algo":"NVFP4","group_size":16,"exclude_modules":["lm_head"]}
}`

func writeModelOptTinyCheckpoint(t *testing.T) string {
	t.Helper()
	const out, k = 4, 32
	dir := t.TempDir()
	writeTestFile(t, dir, "config.json", modelOptTinyLlamaConfig)
	writeModelOptShards(t, dir, map[string]map[string]tinySTTensor{
		"model-00001-of-00002.safetensors": {
			"model.embed_tokens.weight": {dtype: "F32", shape: []int{2, k}, data: f32TestBytes(sequenceFloats(2*k, 0.125))},
		},
		"model-00002-of-00002.safetensors": nvfp4TestTriple("model.layers.0.self_attn.q_proj", out, k, true, 0.5),
	})
	return dir
}

// fak-test:runtime fast est=50ms lane=default
func TestLoadModelOptNVFP4DirRoutesProjectionThroughW4A4(t *testing.T) {
	const out, k = 4, 32
	m, err := LoadModelOptNVFP4Dir(writeModelOptTinyCheckpoint(t), nil)
	if err != nil {
		t.Fatal(err)
	}
	const name = "model.layers.0.self_attn.q_proj.weight"
	if m.NVFP4Count() != 1 || !m.hasWeight(name) || !m.has("model.embed_tokens.weight") {
		t.Fatalf("NVFP4Count=%d hasWeight=%v", m.NVFP4Count(), m.hasWeight(name))
	}
	if m.has("model.layers.0.self_attn.q_proj.weight_scale_2") || m.has(name) {
		t.Fatal("NVFP4 triple leaked into the f32 manifest")
	}
	if o, i, ok := m.residentShape(name); !ok || o != out || i != k {
		t.Fatalf("residentShape = (%d,%d,%v)", o, i, ok)
	}
	// x lies on the E2M1 grid with amax 6 in every 16-block, so dynamic NVFP4
	// activation quantization is (near-)exact and y matches the f64 dot product.
	x := make([]float32, k)
	for i := range x {
		x[i] = nvfp4TestE2M1[(3*i+1)%8]
		if i%2 == 1 {
			x[i] = -x[i]
		}
	}
	x[0], x[16] = 6, 6
	w := nvfp4TestWant(out, k, 0.5)
	got := m.residentMatRowsBase(name, x, out, k)
	for r := 0; r < out; r++ {
		var want float64
		for i := 0; i < k; i++ {
			want += float64(w[r*k+i]) * float64(x[i])
		}
		if math.Abs(float64(got[r])-want) > 1e-4*math.Max(1, math.Abs(want)) {
			t.Fatalf("y[%d] = %g, want %g", r, got[r], want)
		}
	}
}

type nvfp4LessBackend struct{ compute.Backend }

// fak-test:runtime fast est=20ms lane=default
func TestLoadModelOptNVFP4DirRefusesBackendWithoutW4A4(t *testing.T) {
	dir := writeModelOptTinyCheckpoint(t)
	_, err := LoadModelOptNVFP4Dir(dir, nvfp4LessBackend{compute.Default()})
	if !errors.Is(err, compute.ErrNVFP4GEMMUnsupported) {
		t.Fatalf("err = %v, want ErrNVFP4GEMMUnsupported", err)
	}
	writeTestFile(t, dir, "config.json", `{"model_type":"llama","hidden_size":32,"num_hidden_layers":1,"num_attention_heads":4,"vocab_size":2}`)
	if _, err := LoadModelOptNVFP4Dir(dir, nil); !errors.Is(err, ErrModelOptNVFP4Config) {
		t.Fatalf("undeclared quantization err = %v, want ErrModelOptNVFP4Config", err)
	}
}

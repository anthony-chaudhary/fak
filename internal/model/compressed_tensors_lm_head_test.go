package model

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// fak-test:runtime fast est=200ms
func TestCompressedTensorsLMHeadTensorQ8(t *testing.T) {
	const descriptor = `{"quant_method":"compressed-tensors","format":"float-quantized","config_groups":{"head":{"targets":["lm_head"],"weights":{"num_bits":8,"type":"float","strategy":"tensor","symmetric":true,"dynamic":false}}}}`
	fixture := func() (Config, map[string]tinySTTensor, []float32) {
		cfg := Config{ModelType: "llama", HiddenSize: 32, IntermediateSize: 32, VocabSize: 32, NumLayers: 1, NumHeads: 1, NumKVHeads: 1, HeadDim: 32, RMSNormEps: 1e-5, RopeTheta: 10000, QuantizationConfig: json.RawMessage(descriptor)}
		tensors := map[string]tinySTTensor{}
		for _, name := range []string{"model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight", "model.layers.0.self_attn.k_proj.weight", "model.layers.0.self_attn.v_proj.weight", "model.layers.0.self_attn.o_proj.weight", "model.layers.0.mlp.gate_proj.weight", "model.layers.0.mlp.up_proj.weight", "model.layers.0.mlp.down_proj.weight"} {
			tensors[name] = tinySTTensor{"F32", []int{32, 32}, f32TestBytes(sequenceFloats(32*32, 0.05))}
		}
		for _, name := range []string{"model.norm.weight", "model.layers.0.input_layernorm.weight", "model.layers.0.post_attention_layernorm.weight"} {
			tensors[name] = tinySTTensor{"F32", []int{32}, f32TestBytes(repeatFloat32(32, 1))}
		}
		codes, decoded := make([]byte, 32*32), make([]float32, 32*32)
		// Independent exact E4M3FN values; the expected model uses the ordinary
		// F32-to-Q8 loader, never the candidate's lookup or scale conversion.
		pattern := []byte{0x38, 0x40, 0x30, 0xb8, 0x00}
		values := []float32{1, 2, 0.5, -1, 0}
		for i := range codes {
			codes[i], decoded[i] = pattern[i%5], values[i%5]*0.25
		}
		tensors["lm_head.weight"] = tinySTTensor{"F8_E4M3", []int{32, 32}, codes}
		tensors["lm_head.weight_scale"] = tinySTTensor{"F32", []int{1}, f32TestBytes([]float32{0.25})}
		return cfg, tensors, decoded
	}
	write := func(cfg Config, tensors map[string]tinySTTensor, sharded bool) string {
		t.Helper()
		dir := t.TempDir()
		config, err := json.Marshal(cfg)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, "config.json"), config, 0o600); err != nil {
			t.Fatal(err)
		}
		if !sharded {
			if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), tinySafetensorsBytes(t, tensors), 0o600); err != nil {
				t.Fatal(err)
			}
			return dir
		}
		body, head := map[string]tinySTTensor{}, map[string]tinySTTensor{}
		weightMap := map[string]string{}
		for name, tensor := range tensors {
			shard := "a-body.safetensors"
			if strings.HasPrefix(name, "lm_head.") {
				head[name], shard = tensor, "z-head.safetensors"
			} else {
				body[name] = tensor
			}
			weightMap[name] = shard
		}
		for name, tensors := range map[string]map[string]tinySTTensor{"a-body.safetensors": body, "z-head.safetensors": head} {
			if err := os.WriteFile(filepath.Join(dir, name), tinySafetensorsBytes(t, tensors), 0o600); err != nil {
				t.Fatal(err)
			}
		}
		index, err := json.Marshal(map[string]any{"weight_map": weightMap})
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), index, 0o600); err != nil {
			t.Fatal(err)
		}
		return dir
	}
	for _, sharded := range []bool{false, true} {
		t.Run(map[bool]string{false: "single", true: "sharded"}[sharded], func(t *testing.T) {
			cfg, tensors, decoded := fixture()
			got, err := LoadSafetensorsQuantConfigDir(write(cfg, tensors, sharded))
			if err != nil {
				t.Fatal(err)
			}
			cfg.QuantizationConfig = nil
			delete(tensors, "lm_head.weight_scale")
			tensors["lm_head.weight"] = tinySTTensor{"F32", []int{32, 32}, f32TestBytes(decoded)}
			want, err := LoadSafetensorsQuantConfigDir(write(cfg, tensors, sharded))
			if err != nil {
				t.Fatal(err)
			}
			if got.headName() != "lm_head.weight" || got.has("lm_head.weight") || got.has("lm_head.weight_scale") || len(got.raw) != len(want.raw) {
				t.Fatal("head did not resolve exclusively to the resident Q8 store")
			}
			a, b := got.q8w["lm_head.weight"], want.q8w["lm_head.weight"]
			if a == nil || !reflect.DeepEqual(a.q, b.q) || !reflect.DeepEqual(a.d, b.d) {
				t.Fatal("Q8 head differs from scaled F32 oracle")
			}
			sg, sw := got.NewSession(), want.NewSession()
			defer sg.Close()
			defer sw.Close()
			sg.Quant, sw.Quant = true, true
			check := func(a, b []float32) {
				t.Helper()
				if len(a) != 32 || len(b) != 32 {
					t.Fatal("missing vocabulary logits")
				}
				for i := range a {
					if math.IsNaN(float64(a[i])) || math.IsInf(float64(a[i]), 0) || math.Float32bits(a[i]) != math.Float32bits(b[i]) {
						t.Fatalf("logit %d differs: %v vs %v", i, a[i], b[i])
					}
				}
			}
			check(sg.Prefill([]int{1, 3, 5}), sw.Prefill([]int{1, 3, 5}))
			check(sg.Step(7), sw.Step(7))
		})
	}
	t.Run("target-resolution", func(t *testing.T) {
		for _, tc := range []struct {
			target, ignore string
			want           bool
		}{
			{"lm_head", "", true}, {"re:.*lm_head$", "", true},
			{"re:head", "", false}, {"Linear", "", false}, {"lm", "", false},
			{"lm_head", "lm_head", false}, {"lm_head", "re:.*head$", false},
		} {
			cfg, _, _ := fixture()
			raw := strings.Replace(descriptor, `["lm_head"]`, `["`+tc.target+`"]`, 1)
			if tc.ignore != "" {
				raw = strings.Replace(raw, `"format":`, `"ignore":["`+tc.ignore+`"],"format":`, 1)
			}
			cfg.QuantizationConfig = json.RawMessage(raw)
			got, err := compressedTensorsLMHeadEnabled(cfg)
			if err != nil || got != tc.want {
				t.Fatalf("target=%q ignore=%q: %v %v", tc.target, tc.ignore, got, err)
			}
		}
		// An unmentioned or ignored float head retains the ordinary loader path.
		for _, raw := range []string{
			strings.Replace(descriptor, `["lm_head"]`, `["Linear"]`, 1),
			strings.Replace(descriptor, `"format":`, `"ignore":["lm_head"],"format":`, 1),
		} {
			cfg, tensors, decoded := fixture()
			cfg.QuantizationConfig = json.RawMessage(raw)
			if m, err := LoadSafetensorsQuantConfigDir(write(cfg, tensors, false)); err == nil || m != nil {
				t.Fatal("unselected FP8 head was admitted from a generic or ignored target")
			}
			delete(tensors, "lm_head.weight_scale")
			tensors["lm_head.weight"] = tinySTTensor{"F32", []int{32, 32}, f32TestBytes(decoded)}
			if _, err := LoadSafetensorsQuantConfigDir(write(cfg, tensors, false)); err != nil {
				t.Fatal(err)
			}
		}
	})
	t.Run("unsupported-metadata", func(t *testing.T) {
		for _, raw := range []string{
			strings.Replace(descriptor, `"tensor"`, `"channel"`, 1),
			strings.Replace(descriptor, `"num_bits":8`, `"num_bits":4`, 1),
			strings.Replace(descriptor, `"type":"float"`, `"type":"int"`, 1),
			strings.Replace(descriptor, `"dynamic":false`, `"dynamic":true`, 1),
			strings.Replace(descriptor, `"symmetric":true`, `"symmetric":false`, 1),
			strings.Replace(descriptor, `"dynamic":false`, `"group_size":128,"dynamic":false`, 1),
			strings.Replace(descriptor, `"weights":`, `"input_activations":{},"weights":`, 1),
			strings.Replace(descriptor, `"format":`, `"kv_cache_scheme":{},"format":`, 1),
			strings.Replace(descriptor, `"float-quantized"`, `"mixed-precision"`, 1),
			strings.Replace(descriptor, `["lm_head"]`, `["re:("]`, 1),
			strings.Replace(descriptor, `"num_bits":8`, `"num_bits":4,"num_bits":8`, 1),
			strings.Replace(descriptor, `"quant_method":"compressed-tensors"`, `"quant_method":"compressed-tensors","quant_method":"fp8"`, 1),
			strings.Replace(descriptor, `"config_groups":{`, `"config_groups":{"other":{"targets":["re:.*lm_head"]},`, 1),
		} {
			cfg, tensors, _ := fixture()
			cfg.QuantizationConfig = json.RawMessage(raw)
			if m, err := LoadSafetensorsQuantConfigDir(write(cfg, tensors, false)); err == nil || m != nil {
				t.Fatalf("admitted metadata %s", raw)
			}
		}
	})
	t.Run("malformed-head", func(t *testing.T) {
		for _, tc := range []string{"missing-head", "missing-scale", "cross-shard", "dtype", "shape", "shape-overflow", "alignment", "short-weight", "scale-shape", "scale-dtype", "scale-nan", "scale-inf", "scale-zero", "scale-negative", "product-overflow", "weight-nan", "input-scale", "zero-point", "tied"} {
			t.Run(tc, func(t *testing.T) {
				cfg, tensors, _ := fixture()
				w, s := tensors["lm_head.weight"], tensors["lm_head.weight_scale"]
				switch tc {
				case "missing-head":
					delete(tensors, "lm_head.weight")
				case "missing-scale", "cross-shard":
					delete(tensors, "lm_head.weight_scale")
				case "dtype":
					w.dtype = "F16"
				case "shape":
					w.shape = []int{16, 64}
				case "shape-overflow":
					cfg.VocabSize = math.MaxInt/32 + 1
					w.shape = []int{cfg.VocabSize, 32}
				case "alignment":
					cfg.HiddenSize = 31
					w.shape = []int{32, 31}
				case "short-weight":
					w.data = w.data[:len(w.data)-1]
				case "scale-shape":
					s.shape = []int{1, 1}
				case "scale-dtype":
					s.dtype = "BF16"
				case "scale-nan":
					s.data = f32TestBytes([]float32{float32(math.NaN())})
				case "scale-inf":
					s.data = f32TestBytes([]float32{float32(math.Inf(1))})
				case "scale-zero":
					s.data = f32TestBytes([]float32{0})
				case "scale-negative":
					s.data = f32TestBytes([]float32{-1})
				case "product-overflow":
					s.data = f32TestBytes([]float32{math.MaxFloat32})
				case "weight-nan":
					w.data[0] = 0x7f
				case "input-scale":
					tensors["lm_head.input_scale"] = s
				case "zero-point":
					tensors["lm_head.weight_zero_point"] = s
				case "tied":
					cfg.TieWordEmbeddings = true
				}
				if tc != "missing-head" {
					tensors["lm_head.weight"] = w
				}
				if tc != "missing-scale" && tc != "cross-shard" {
					tensors["lm_head.weight_scale"] = s
				}
				dir := write(cfg, tensors, tc == "cross-shard")
				if tc == "cross-shard" {
					// The index is valid and both payloads exist, but no shard owns
					// the complete pair. Do not borrow a scale from another shard.
					path := filepath.Join(dir, "model.safetensors.index.json")
					raw, err := os.ReadFile(path)
					if err != nil {
						t.Fatal(err)
					}
					var index struct {
						Weights map[string]string `json:"weight_map"`
					}
					if err := json.Unmarshal(raw, &index); err != nil {
						t.Fatal(err)
					}
					index.Weights["lm_head.weight_scale"] = "zz-scale.safetensors"
					raw, err = json.Marshal(index)
					if err != nil {
						t.Fatal(err)
					}
					if err := os.WriteFile(path, raw, 0o600); err != nil {
						t.Fatal(err)
					}
					if err := os.WriteFile(filepath.Join(dir, "zz-scale.safetensors"), tinySafetensorsBytes(t, map[string]tinySTTensor{"lm_head.weight_scale": s}), 0o600); err != nil {
						t.Fatal(err)
					}
				}
				if m, err := LoadSafetensorsQuantConfigDir(dir); err == nil || m != nil {
					t.Fatalf("admitted %s", tc)
				}
			})
		}
	})
	t.Run("duplicate-shard-tensors", func(t *testing.T) {
		for _, name := range []string{"lm_head.weight", "lm_head.weight_scale"} {
			cfg, tensors, _ := fixture()
			dir := write(cfg, tensors, true)
			body := map[string]tinySTTensor{}
			for key, tensor := range tensors {
				if !strings.HasPrefix(key, "lm_head.") || key == name {
					body[key] = tensor
				}
			}
			if err := os.WriteFile(filepath.Join(dir, "a-body.safetensors"), tinySafetensorsBytes(t, body), 0o600); err != nil {
				t.Fatal(err)
			}
			if m, err := LoadSafetensorsQuantConfigDir(dir); err == nil || m != nil {
				t.Fatalf("accepted duplicate %s in wrong shard", name)
			}
		}
	})
	t.Run("config-overlay", func(t *testing.T) {
		var cfg Config
		if err := json.Unmarshal([]byte(`{"text_config":{"quantization_config":`+descriptor+`}}`), &cfg); err != nil {
			t.Fatal(err)
		}
		if ok, err := compressedTensorsLMHeadEnabled(cfg); err != nil || !ok {
			t.Fatalf("nested descriptor: %v %v", ok, err)
		}
		if err := json.Unmarshal([]byte(`{"text_config":{"quantization_config":`+descriptor+`},"quantization_config":null}`), &cfg); err != nil {
			t.Fatal(err)
		}
		if ok, err := compressedTensorsLMHeadEnabled(cfg); err != nil || ok {
			t.Fatalf("wrapper null did not clear descriptor: %v %v", ok, err)
		}
	})
}

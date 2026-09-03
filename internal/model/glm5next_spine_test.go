package model

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestGLM5NextExactEnvelopeRefusesBeforeForwardSelection(t *testing.T) {
	raw := readGLM5NextFixture(t, "config.json")
	var cfg Config
	if err := json.Unmarshal(raw, &cfg); err != nil {
		t.Fatal(err)
	}
	if !cfg.GLM5Next {
		t.Fatal("exact pinned GLM5Next config was not recognized")
	}
	if cfg.IsQwen35Hybrid() {
		t.Fatal("GLM5Next must not be routed as Qwen3.5 solely because it has linear_attention layers")
	}
	path, err := ClassifyForwardPath(cfg, nil)
	if path != "" {
		t.Fatalf("path = %q, want no forward selected", path)
	}
	var unsupported *GLM5NextUnsupportedError
	if !errors.As(err, &unsupported) {
		t.Fatalf("error = %T %v, want *GLM5NextUnsupportedError", err, err)
	}
	if !strings.Contains(err.Error(), "fak-native") || !strings.Contains(err.Error(), "KDA/DSA/indexer/mHC") {
		t.Fatalf("error is not actionable: %v", err)
	}
}

func TestGLM5NextIdentityRejectsNearMissesAndAliases(t *testing.T) {
	raw := readGLM5NextFixture(t, "config.json")
	var base map[string]any
	if err := json.Unmarshal(raw, &base); err != nil {
		t.Fatal(err)
	}
	cases := map[string]func(map[string]any){
		"alias":              func(v map[string]any) { v["model_type"] = "glm5next" },
		"architecture alias": func(v map[string]any) { v["architectures"] = []any{"GLM5NextForConditionalGeneration"} },
		"wrong cadence": func(v map[string]any) {
			v["text_config"].(map[string]any)["layer_types"].([]any)[3] = "linear_attention"
		},
		"wrong context":     func(v map[string]any) { v["text_config"].(map[string]any)["max_position_embeddings"] = float64(131072) },
		"unsupported quant": func(v map[string]any) { v["quantization_config"] = map[string]any{"quant_method": "awq"} },
		"unsupported format": func(v map[string]any) {
			v["quantization_config"] = map[string]any{"quant_method": "fp8", "fmt": "e5m2", "activation_scheme": "dynamic"}
		},
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			var v map[string]any
			if err := json.Unmarshal(raw, &v); err != nil {
				t.Fatal(err)
			}
			mutate(v)
			candidate, err := json.Marshal(v)
			if err != nil {
				t.Fatal(err)
			}
			if isExactGLM5NextConfig(candidate) {
				t.Fatal("near-miss config recognized as exact GLM5Next")
			}
		})
	}
	_ = base
}

func TestGLM5NextAcceptsUnquantizedBF16Envelope(t *testing.T) {
	raw := readGLM5NextFixture(t, "config.json")
	var v map[string]any
	if err := json.Unmarshal(raw, &v); err != nil {
		t.Fatal(err)
	}
	// Delete quantization_config for the official reference BF16 checkpoint
	delete(v, "quantization_config")
	bf16Bytes, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	if !isExactGLM5NextConfig(bf16Bytes) {
		t.Fatal("unquantized BF16 GLM5Next config was not recognized as exact GLM5Next")
	}

	var cfg Config
	if err := json.Unmarshal(bf16Bytes, &cfg); err != nil {
		t.Fatal(err)
	}
	if !cfg.GLM5Next {
		t.Fatal("unquantized BF16 GLM5Next config did not set cfg.GLM5Next")
	}
	path, err := ClassifyForwardPath(cfg, nil)
	if path != "" {
		t.Fatalf("path = %q, want empty", path)
	}
	var unsupported *GLM5NextUnsupportedError
	if !errors.As(err, &unsupported) {
		t.Fatalf("expected *GLM5NextUnsupportedError, got %T: %v", err, err)
	}
}

func TestGLM5NextTensorInventoryPinned(t *testing.T) {
	var inventory struct {
		Source     string `json:"source"`
		License    string `json:"license"`
		ShardCount int    `json:"shard_count"`
		TotalSize  int64  `json:"total_size_bytes"`
		Tensors    []struct {
			Name, Dtype, Shard string
			Shape              []int
		} `json:"tensors"`
	}
	if err := json.Unmarshal(readGLM5NextFixture(t, "tensor_inventory.json"), &inventory); err != nil {
		t.Fatal(err)
	}
	if inventory.Source != "zai-org/GLM-5.3-Flash@04c4e9e95c5da8862dced7e5056455116f83a7e0" || inventory.License != "MIT" || inventory.ShardCount != 62 || inventory.TotalSize != 328326771576 {
		t.Fatalf("inventory provenance drifted: %+v", inventory)
	}
	want := map[string]struct {
		dtype string
		shape string
		shard string
	}{
		"model.language_model.embed_tokens.weight":                                 {"BF16", "154880,4096", "model-00001-of-00062.safetensors"},
		"model.language_model.layers.0.self_attn.q_proj.weight":                    {"BF16", "8192,4096", "model-00002-of-00062.safetensors"},
		"model.language_model.layers.0.self_attn.k_conv1d.weight":                  {"BF16", "8192,1,4", "model-00002-of-00062.safetensors"},
		"model.language_model.layers.3.self_attn.indexer.index_kpool_compress_ape": {"BF16", "4,128", "model-00032-of-00062.safetensors"},
		"model.language_model.layers.3.mlp.experts.0.down_proj.weight":             {"F8_E4M3", "4096,2048", "model-00031-of-00062.safetensors"},
	}
	for _, got := range inventory.Tensors {
		w, ok := want[got.Name]
		if !ok {
			continue
		}
		parts := make([]string, len(got.Shape))
		for i, n := range got.Shape {
			parts[i] = fmtInt(n)
		}
		if got.Dtype != w.dtype || strings.Join(parts, ",") != w.shape || got.Shard != w.shard {
			t.Fatalf("tensor %s drifted: %+v", got.Name, got)
		}
		delete(want, got.Name)
	}
	if len(want) != 0 {
		t.Fatalf("missing representative tensors: %v", want)
	}
}

func TestGLM5NextGatePreservesExistingQwenAndGLMPaths(t *testing.T) {
	qwen := Config{ModelType: "qwen3_5_text", LayerTypes: []string{"linear_attention", "full_attention"}}
	if path, err := ClassifyForwardPath(qwen, nil); err != nil || path != ForwardQwen35GDN {
		t.Fatalf("Qwen path = %q, %v", path, err)
	}
	glm := Config{ModelType: "glm_moe_dsa", NumExperts: 8, QLoraRank: 8, KVLoraRank: 8}
	if path, err := ClassifyForwardPath(glm, nil); err != nil || path != ForwardGLMDsaMLA {
		t.Fatalf("GLM-5.2 path = %q, %v", path, err)
	}
}

func TestGLM5NextConfigCadenceHelpers(t *testing.T) {
	cfg := DefaultGLM5NextConfig()
	if cfg.NumHiddenLayers != 45 {
		t.Fatalf("NumHiddenLayers = %d, want 45", cfg.NumHiddenLayers)
	}
	if cfg.NRoutedExperts != 288 || cfg.NSharedExperts != 1 || cfg.ExpertsPerToken != 8 {
		t.Fatalf("MoE parameters mismatch: %+v", cfg)
	}
	if cfg.IndexKPool != 4 || cfg.ConvWindowSize != 4 || cfg.HCMult != 4 {
		t.Fatalf("parameters mismatch: %+v", cfg)
	}

	modelCfg := Config{GLM5Next: true}

	var kdaCount, dsaCount int
	for l := 0; l < 45; l++ {
		isKDA := cfg.IsKDALayer(l)
		isDSA := cfg.IsDSALayer(l)
		if isKDA == isDSA {
			t.Fatalf("layer %d must be either KDA or DSA, got KDA=%v DSA=%v", l, isKDA, isDSA)
		}
		if modelCfg.IsGLM5NextKDALayer(l) != isKDA {
			t.Fatalf("Config.IsGLM5NextKDALayer(%d) = %v, want %v", l, modelCfg.IsGLM5NextKDALayer(l), isKDA)
		}
		if modelCfg.IsGLM5NextDSALayer(l) != isDSA {
			t.Fatalf("Config.IsGLM5NextDSALayer(%d) = %v, want %v", l, modelCfg.IsGLM5NextDSALayer(l), isDSA)
		}
		if isKDA {
			kdaCount++
		} else {
			dsaCount++
		}

		isDense := cfg.IsDenseMLPLayer(l)
		isMoE := cfg.IsSparseMoELayer(l)
		if isDense == isMoE {
			t.Fatalf("layer %d must be either dense MLP or sparse MoE, got dense=%v moe=%v", l, isDense, isMoE)
		}
		if modelCfg.IsGLM5NextDenseMLP(l) != isDense {
			t.Fatalf("Config.IsGLM5NextDenseMLP(%d) = %v, want %v", l, modelCfg.IsGLM5NextDenseMLP(l), isDense)
		}
		if modelCfg.IsGLM5NextSparseMoE(l) != isMoE {
			t.Fatalf("Config.IsGLM5NextSparseMoE(%d) = %v, want %v", l, modelCfg.IsGLM5NextSparseMoE(l), isMoE)
		}
	}
	if kdaCount != 34 || dsaCount != 11 {
		t.Fatalf("cadence counts mismatch: kda=%d (want 34), dsa=%d (want 11)", kdaCount, dsaCount)
	}

	// Non-GLM config should return false
	nonGLM := Config{GLM5Next: false}
	for l := 0; l < 45; l++ {
		if nonGLM.IsGLM5NextKDALayer(l) || nonGLM.IsGLM5NextDSALayer(l) || nonGLM.IsGLM5NextDenseMLP(l) || nonGLM.IsGLM5NextSparseMoE(l) {
			t.Fatalf("non-GLM config returned true for layer %d", l)
		}
	}
}

func readGLM5NextFixture(t *testing.T, name string) []byte {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("testdata", "glm5next", name))
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func fmtInt(v int) string {
	if v == 0 {
		return "0"
	}
	var b [20]byte
	i := len(b)
	for v > 0 {
		i--
		b[i] = byte('0' + v%10)
		v /= 10
	}
	return string(b[i:])
}

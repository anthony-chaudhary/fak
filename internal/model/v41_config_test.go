package model

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

const deepSeekV41ConfigSHA256 = "8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879"

func readDeepSeekV41Config(t *testing.T) ([]byte, Config) {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", "deepseek_v41_flash_config.json"))
	if err != nil {
		t.Fatal(err)
	}
	var cfg Config
	if err := json.Unmarshal(raw, &cfg); err != nil {
		t.Fatalf("parse official %s@%s config: %v", DeepSeekV41FlashModelID, DeepSeekV41FlashRevision, err)
	}
	return raw, cfg
}

func TestDeepSeekV41OfficialConfigPinnedAndRetained(t *testing.T) {
	raw, cfg := readDeepSeekV41Config(t)
	sum := sha256.Sum256(raw)
	if got := hex.EncodeToString(sum[:]); got != deepSeekV41ConfigSHA256 {
		t.Fatalf("official config digest=%s want=%s", got, deepSeekV41ConfigSHA256)
	}
	if !cfg.IsDeepSeekV41() || cfg.DeepSeekV41 == nil {
		t.Fatalf("official identity not retained: model_type=%q metadata=%#v", cfg.ModelType, cfg.DeepSeekV41)
	}
	if cfg.NumLayers != 40 || cfg.HiddenSize != 5120 || cfg.NumExperts != 384 {
		t.Fatalf("nested text geometry: layers=%d hidden=%d experts=%d", cfg.NumLayers, cfg.HiddenSize, cfg.NumExperts)
	}
	m := cfg.DeepSeekV41
	cases := map[string]struct{ got, want any }{
		"identity":      {[]string{m.WrapperModelType, m.TextModelType}, []string{"deepseek_v41", "deepseek_v41_text"}},
		"engram layers": {m.EngramLayerIDs, []int{1, 14}},
		"engram rows":   {m.EngramNumEmbeddings, []int{384006168, 384016682}},
		"kv sources":    {m.KVSourceLayerIDs, []int{2, 8, 14, 20}},
		"index sources": {m.IndexSourceLayerIDs, []int{2, 8, 14, 20, 24, 28, 32, 36}},
		"candidate":     {[]int{m.CandidateSourceLayerID, m.CandidateTopKBlocks, m.CandidateBlockSize}, []int{20, 2048, 8}},
		"quant":         {[]any{m.Quantization.Method, m.Quantization.Activation, m.Quantization.WeightBlockSize, m.Quantization.ScaleFormat, m.Quantization.ExpertDtype}, []any{"fp8", "dynamic", []int{32, 32}, "ue8m0", "fp4"}},
		"mHC":           {[]any{m.HCMult, m.HCSinkhornIters, m.HCEps}, []any{4, 20, 1e-6}},
		"compression":   {len(m.CompressRatios), 43},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			if !reflect.DeepEqual(tc.got, tc.want) {
				t.Fatalf("got=%v want=%v", tc.got, tc.want)
			}
		})
	}

	t.Run("Config JSON round trip", func(t *testing.T) {
		encoded, err := json.Marshal(cfg)
		if err != nil {
			t.Fatal(err)
		}
		var roundTrip Config
		if err := json.Unmarshal(encoded, &roundTrip); err != nil {
			t.Fatalf("round-trip official config: %v", err)
		}
		if !roundTrip.IsDeepSeekV41() || roundTrip.DeepSeekV41 == nil {
			t.Fatalf("round trip lost V4.1 identity/metadata: %#v", roundTrip.DeepSeekV41)
		}
		if !reflect.DeepEqual(roundTrip.DeepSeekV41.EngramNumEmbeddings, cfg.DeepSeekV41.EngramNumEmbeddings) {
			t.Fatalf("round trip Engram rows=%v want=%v", roundTrip.DeepSeekV41.EngramNumEmbeddings, cfg.DeepSeekV41.EngramNumEmbeddings)
		}

		mutated := cfg
		mutated.HiddenSize--
		encoded, err = json.Marshal(mutated)
		if err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(encoded, &roundTrip); !errors.Is(err, ErrV41ConfigAdmission) {
			t.Fatalf("marshaled Config reused stale geometry, error=%v want ErrV41ConfigAdmission", err)
		}
	})
}

func TestDeepSeekV41RejectsMalformedCoupledMetadataAndPreservesOtherFamilies(t *testing.T) {
	raw, _ := readDeepSeekV41Config(t)
	mutate := func(t *testing.T, fn func(map[string]any)) []byte {
		t.Helper()
		var doc map[string]any
		if err := json.Unmarshal(raw, &doc); err != nil {
			t.Fatal(err)
		}
		fn(doc["text_config"].(map[string]any))
		out, err := json.Marshal(doc)
		if err != nil {
			t.Fatal(err)
		}
		return out
	}
	bad := map[string][]byte{
		"engram count":       mutate(t, func(x map[string]any) { x["engram_num_embeddings"] = []any{384006168.0} }),
		"engram first rows":  mutate(t, func(x map[string]any) { x["engram_num_embeddings"].([]any)[0] = 384006169.0 }),
		"engram second rows": mutate(t, func(x map[string]any) { x["engram_num_embeddings"].([]any)[1] = 384016683.0 }),
		"kv sources":         mutate(t, func(x map[string]any) { x["kv_source_layer_ids"] = []any{2.0, 8.0, 14.0} }),
		"candidate":          mutate(t, func(x map[string]any) { x["candidate_block_size"] = 0.0 }),
		"compression length": mutate(t, func(x map[string]any) {
			x["compress_ratios"] = x["compress_ratios"].([]any)[:40]
		}),
		"compression value": mutate(t, func(x map[string]any) { x["compress_ratios"].([]any)[2] = 1.0 }),
		"compression theta": mutate(t, func(x map[string]any) { x["compress_rope_theta"] = 10000.0 }),
	}
	for name, malformed := range bad {
		t.Run(name, func(t *testing.T) {
			var cfg Config
			if err := json.Unmarshal(malformed, &cfg); !errors.Is(err, ErrV41ConfigAdmission) {
				t.Fatalf("error=%v want ErrV41ConfigAdmission", err)
			}
		})
	}
	for _, other := range [][]byte{
		[]byte(`{"model_type":"llama","hidden_size":32,"num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1}`),
		[]byte(`{"model_type":"deepseek_v4","architectures":["DeepseekV4ForCausalLM"],"hidden_size":4096,"num_hidden_layers":43,"num_attention_heads":64,"num_key_value_heads":1}`),
	} {
		var cfg Config
		if err := json.Unmarshal(other, &cfg); err != nil || cfg.IsDeepSeekV41() {
			t.Fatalf("unrelated family changed: cfg=%#v err=%v", cfg, err)
		}
	}
}

func TestDeepSeekV41NativeLoadersRefuseBeforeWeightIO(t *testing.T) {
	raw, cfg := readDeepSeekV41Config(t)
	configOnly := t.TempDir()
	if err := os.WriteFile(filepath.Join(configOnly, "config.json"), raw, 0o644); err != nil {
		t.Fatal(err)
	}
	opened := 0
	opener := func(string) (*safetensorsFile, error) {
		opened++
		return nil, errors.New("weight opener reached")
	}
	tests := map[string]func() error{
		"f32 directory": func() error { _, err := Load(configOnly); return err },
		"f32 tensors":   func() error { _, err := NewFromF32Tensors(cfg, nil); return err },
		"quant file": func() error {
			_, err := loadSafetensorsQuantFile(filepath.Join(configOnly, "absent.safetensors"), cfg, opener)
			return err
		},
		"quant directory":        func() error { _, err := loadSafetensorsQuantDir(configOnly, cfg, opener); return err },
		"quant config directory": func() error { _, err := LoadSafetensorsQuantConfigDir(configOnly); return err },
	}
	for name, load := range tests {
		t.Run(name, func(t *testing.T) {
			if err := load(); !errors.Is(err, ErrV41NativeUnsupported) {
				t.Fatalf("error=%v want ErrV41NativeUnsupported", err)
			}
		})
	}
	for _, modelType := range []string{"", "deepseek_v41_text"} {
		t.Run("retained pointer identity "+modelType, func(t *testing.T) {
			copied := cfg
			copied.ModelType = modelType
			if !copied.IsDeepSeekV41() {
				t.Fatalf("retained V4.1 metadata not recognized after model_type normalization to %q", modelType)
			}
			if _, err := loadSafetensorsQuantFile(filepath.Join(configOnly, "absent.safetensors"), copied, opener); !errors.Is(err, ErrV41NativeUnsupported) {
				t.Fatalf("copied config loader error=%v want ErrV41NativeUnsupported", err)
			}
		})
	}
	if opened != 0 {
		t.Fatalf("opened %d weight files before refusal", opened)
	}
}

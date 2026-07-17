package model

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

// writeQuantDirV4Fixture builds the minimal indexed quant-dir snapshot the V4
// loader accepts: a config.json declaring the pinned V4 shape, one tiny shard
// with a routed-expert tensor (skipped by the lazy expert runtime) and an
// lm_head (quantized), and the index binding both. Same fixture shape as
// TestV4LiveExpertLoadSkipsRoutedPayloadsAndConstructsLazily.
func writeQuantDirV4Fixture(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	cfgBytes := []byte(`{"model_type":"deepseek_v4","num_hidden_layers":61,"hidden_size":7168,"n_routed_experts":384,"num_experts_per_tok":6,"moe_intermediate_size":3072,"n_shared_experts":1,"expert_dtype":"fp4","norm_topk_prob":true,"routed_scaling_factor":2.5,"scoring_func":"sqrtsoftplus","topk_method":"noaux_tc","swiglu_limit":10}`)
	if err := os.WriteFile(filepath.Join(dir, "config.json"), cfgBytes, 0o644); err != nil {
		t.Fatal(err)
	}
	expert := "model.layers.3.mlp.experts.0.gate_proj.weight"
	head := "lm_head.weight"
	shard := "model-00001-of-00001.safetensors"
	if err := os.WriteFile(filepath.Join(dir, shard), tinySafetensorsBytes(t, map[string]tinySTTensor{
		expert: {dtype: "F32", shape: []int{1}, data: f32TestBytes([]float32{9})},
		head:   {dtype: "F32", shape: []int{1, 32}, data: f32TestBytes(make([]float32, 32))},
	}), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), []byte(`{"weight_map":{"`+expert+`":"`+shard+`","`+head+`":"`+shard+`"}}`), 0o644); err != nil {
		t.Fatal(err)
	}
	return dir
}

// TestLoadSafetensorsQuantDirAdmitsDeepSeekV4Config closes the #4807 bypass:
// the quant-dir loader must run AdmitDeepSeekV4Config on a deepseek_v4 config
// BEFORE constructing the live model, exactly as newModel does for the f32
// path (weights.go), instead of accepting mis-shaped V4 metadata and failing
// only later at ensureV4LiveExpert.
func TestLoadSafetensorsQuantDirAdmitsDeepSeekV4Config(t *testing.T) {
	dir := writeQuantDirV4Fixture(t)

	t.Run("mis-shaped V4 config is refused at admission", func(t *testing.T) {
		bad := pinnedV4RuntimeConfig()
		bad.NumExpertsPerTok = 5
		if _, err := LoadSafetensorsQuantDir(dir, bad); !errors.Is(err, ErrV4ConfigAdmission) {
			t.Fatalf("LoadSafetensorsQuantDir accepted mis-shaped V4 config, err=%v want ErrV4ConfigAdmission", err)
		}
	})

	t.Run("mis-shaped V4 config is refused on the no-index fallback", func(t *testing.T) {
		fileDir := t.TempDir()
		if err := os.WriteFile(filepath.Join(fileDir, "model.safetensors"), tinySafetensorsBytes(t, map[string]tinySTTensor{
			"lm_head.weight": {dtype: "F32", shape: []int{1, 32}, data: f32TestBytes(make([]float32, 32))},
		}), 0o644); err != nil {
			t.Fatal(err)
		}
		bad := pinnedV4RuntimeConfig()
		bad.NumExperts = 8
		if _, err := LoadSafetensorsQuantDir(fileDir, bad); !errors.Is(err, ErrV4ConfigAdmission) {
			t.Fatalf("no-index quant load accepted mis-shaped V4 config, err=%v want ErrV4ConfigAdmission", err)
		}
	})

	t.Run("pinned V4 config passes admission", func(t *testing.T) {
		m, err := LoadSafetensorsQuantDir(dir, pinnedV4RuntimeConfig())
		if err != nil {
			t.Fatalf("LoadSafetensorsQuantDir rejected the pinned V4 config: %v", err)
		}
		if m.sourceDir != dir {
			t.Fatalf("sourceDir=%q want %q", m.sourceDir, dir)
		}
	})

	t.Run("non-V4 config is unaffected by the gate", func(t *testing.T) {
		fileDir := t.TempDir()
		if err := os.WriteFile(filepath.Join(fileDir, "model.safetensors"), tinySafetensorsBytes(t, map[string]tinySTTensor{
			"lm_head.weight": {dtype: "F32", shape: []int{1, 32}, data: f32TestBytes(make([]float32, 32))},
		}), 0o644); err != nil {
			t.Fatal(err)
		}
		cfg := Config{ModelType: "llama", NumLayers: 1, HiddenSize: 32}
		if _, err := LoadSafetensorsQuantDir(fileDir, cfg); err != nil {
			t.Fatalf("non-V4 quant load regressed: %v", err)
		}
	})
}

package model

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"
)

// safetensorsTestTensor is one header row of a generated fixture shard.
type safetensorsTestTensor struct {
	dtype string
	dims  []int
}

// writeSafetensorsFixture writes a checkpoint directory with an optional index and the named
// shards; each shard carries only a header (no tensor bytes), which is all the preflight reads.
func writeSafetensorsFixture(t *testing.T, dir string, index map[string]string, shards map[string]map[string]safetensorsTestTensor, configName string) {
	t.Helper()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	for name, tensors := range shards {
		hdr := map[string]any{}
		for tn, spec := range tensors {
			hdr[tn] = map[string]any{"dtype": spec.dtype, "shape": spec.dims}
		}
		b, err := json.Marshal(hdr)
		if err != nil {
			t.Fatal(err)
		}
		var buf []byte
		var lenBuf [8]byte
		binary.LittleEndian.PutUint64(lenBuf[:], uint64(len(b)))
		buf = append(buf, lenBuf[:]...)
		buf = append(buf, b...)
		if err := os.WriteFile(filepath.Join(dir, name), buf, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if index != nil {
		b, err := json.Marshal(map[string]any{"metadata": map[string]any{}, "weight_map": index})
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	cfg, err := os.ReadFile("testdata/" + configName)
	if err != nil {
		t.Fatalf("read %s: %v", configName, err)
	}
	if err := os.WriteFile(filepath.Join(dir, "config.json"), cfg, 0o644); err != nil {
		t.Fatal(err)
	}
}

// TestSafetensorsPreflightV41Plan pins the V4.1-shaped census: the native refusal is recorded
// once while the census continues, the MXFP4 pair routes to the packed expert representation,
// the dense FP8 pair routes to Q8_0, and index/header consistency reports the missing entry and
// the orphan without stopping.
func TestSafetensorsPreflightV41Plan(t *testing.T) {
	dir := t.TempDir()
	writeSafetensorsFixture(t, dir,
		map[string]string{
			"layers.0.ffn.experts.0.w1.weight": "model-00001-of-00002.safetensors",
			"layers.0.ffn.experts.0.w1.scale":  "model-00001-of-00002.safetensors",
			"layers.0.attn.wq_a.weight":        "model-00001-of-00002.safetensors",
			"layers.0.attn.wq_a.scale":         "model-00001-of-00002.safetensors",
			"model.norm.weight":                "model-00001-of-00002.safetensors",
			"layers.1.missing.weight":          "model-00002-of-00002.safetensors",
		},
		map[string]map[string]safetensorsTestTensor{
			"model-00001-of-00002.safetensors": {
				"layers.0.ffn.experts.0.w1.weight": {"I8", []int{2304, 2560}},
				"layers.0.ffn.experts.0.w1.scale":  {"F8_E8M0", []int{2304, 160}},
				"layers.0.attn.wq_a.weight":        {"F8_E4M3", []int{1280, 5120}},
				"layers.0.attn.wq_a.scale":         {"F8_E8M0", []int{40, 160}},
				"model.norm.weight":                {"BF16", []int{5120}},
				"layers.0.orphan_tensor.weight":    {"BF16", []int{8}},
			},
			"model-00002-of-00002.safetensors": {},
		},
		"deepseek_v41_flash_config.json")

	plan, err := BuildSafetensorsPreflight(SafetensorsPreflightInput{Dir: dir})
	if err != nil {
		t.Fatalf("BuildSafetensorsPreflight: %v", err)
	}
	if !errors.Is(plan.AdmissionErr, ErrV41NativeUnsupported) {
		t.Fatalf("admission err = %v, want ErrV41NativeUnsupported", plan.AdmissionErr)
	}
	byName := map[string]SafetensorsPreflightTensor{}
	for _, x := range plan.Tensors {
		byName[x.Name] = x
	}
	if got := byName["layers.0.ffn.experts.0.w1.weight"]; got.Role != SafetensorsRoleV41Expert || got.Representation != SafetensorsRepV41Expert {
		t.Errorf("expert weight role/rep = %s/%s, want %s/%s", got.Role, got.Representation, SafetensorsRoleV41Expert, SafetensorsRepV41Expert)
	}
	if got := byName["layers.0.ffn.experts.0.w1.scale"]; got.Role != SafetensorsRoleV41ExpertScale {
		t.Errorf("expert scale role = %s, want %s", got.Role, SafetensorsRoleV41ExpertScale)
	}
	if got := byName["layers.0.attn.wq_a.weight"]; got.Role != SafetensorsRoleV41DenseFP8 || got.Representation != SafetensorsRepQ8 {
		t.Errorf("dense fp8 role/rep = %s/%s, want %s/%s", got.Role, got.Representation, SafetensorsRoleV41DenseFP8, SafetensorsRepQ8)
	}
	if got := byName["model.norm.weight"]; got.Role != SafetensorsRolePlain || got.Representation != SafetensorsRepF32Host {
		t.Errorf("norm role/rep = %s/%s, want plain/f32-host", got.Role, got.Representation)
	}
	if len(plan.Missing) != 1 || plan.Missing[0] != "layers.1.missing.weight" {
		t.Errorf("missing = %v, want [layers.1.missing.weight]", plan.Missing)
	}
	if len(plan.Orphans) != 1 || plan.Orphans[0] != "layers.0.orphan_tensor.weight" {
		t.Errorf("orphans = %v, want [layers.0.orphan_tensor.weight]", plan.Orphans)
	}
	wantDtypes := map[string]int{"I8": 1, "F8_E8M0": 2, "F8_E4M3": 1, "BF16": 2}
	for dt, n := range wantDtypes {
		if plan.DTypes[dt] != n {
			t.Errorf("dtype %s count = %d, want %d", dt, plan.DTypes[dt], n)
		}
	}
	if len(plan.ScaleGaps) != 0 || len(plan.HostGaps) != 0 || len(plan.RoleErrors) != 0 {
		t.Errorf("unexpected gaps: scale=%v host=%v roles=%v", plan.ScaleGaps, plan.HostGaps, plan.RoleErrors)
	}
}

// TestSafetensorsPreflightScaleAndHostGaps pins the generic-config census: an unpaired FP8
// weight becomes both a scale-pair gap and a host-decode gap, an undecodable dtype becomes a
// host-decode gap, and decoded float sources split into Q8_0 resident vs f32 host by role.
func TestSafetensorsPreflightScaleAndHostGaps(t *testing.T) {
	dir := t.TempDir()
	writeSafetensorsFixture(t, dir, nil,
		map[string]map[string]safetensorsTestTensor{
			"model.safetensors": {
				"model.layers.0.self_attn.q_proj.weight": {"F8_E4M3", []int{4096, 4096}},
				"model.layers.0.self_attn.q_norm.weight": {"U8", []int{64}},
				"model.embed_tokens.weight":              {"BF16", []int{32, 4096}},
				"model.layers.0.mlp.up_proj.weight":      {"BF16", []int{8192, 4096}},
			},
		},
		"llama_small_config.json")

	plan, err := BuildSafetensorsPreflight(SafetensorsPreflightInput{Dir: dir})
	if err != nil {
		t.Fatalf("BuildSafetensorsPreflight: %v", err)
	}
	if plan.AdmissionErr != nil {
		t.Fatalf("admission err = %v, want nil", plan.AdmissionErr)
	}
	if len(plan.ScaleGaps) != 1 {
		t.Fatalf("scale gaps = %v, want exactly one", plan.ScaleGaps)
	}
	g := plan.ScaleGaps[0]
	if g.Weight != "model.layers.0.self_attn.q_proj.weight" || g.Sibling != "model.layers.0.self_attn.q_proj.weight_scale_inv" {
		t.Errorf("scale gap = %+v, want q_proj weight/_scale_inv", g)
	}
	if got := plan.HostGaps["host_dequant(U8)"]; len(got) != 1 || got[0] != "model.layers.0.self_attn.q_norm.weight" {
		t.Errorf("host_dequant(U8) = %v, want [q_norm]", got)
	}
	if got := plan.HostGaps["host_dequant(F8_E4M3)"]; len(got) != 1 || got[0] != "model.layers.0.self_attn.q_proj.weight" {
		t.Errorf("host_dequant(F8_E4M3) = %v, want [q_proj]", got)
	}
	byName := map[string]SafetensorsPreflightTensor{}
	for _, x := range plan.Tensors {
		byName[x.Name] = x
	}
	if got := byName["model.embed_tokens.weight"].Representation; got != SafetensorsRepF32Host {
		t.Errorf("embed rep = %s, want f32-host", got)
	}
	if got := byName["model.layers.0.mlp.up_proj.weight"].Representation; got != SafetensorsRepQ8 {
		t.Errorf("up_proj rep = %s, want q8_0", got)
	}
}

// TestDecodedFloatGoesToQ8 pins the header-only mirror of the loader's decoded-float decision:
// fused projections, source MoE arms, and plain rank-2 quant weights become Q8_0; norms,
// embeddings, and unmapped names stay on the host.
func TestDecodedFloatGoesToQ8(t *testing.T) {
	cfg := Config{ModelType: "llama", NumLayers: 2, HiddenSize: 4096, NumHeads: 32, NumExperts: 8}
	cases := []struct {
		name  string
		canon string
		dims  []int
		want  bool
	}{
		{"fused qkv", "model.layers.0.self_attn.qkv_proj.weight", []int{12288, 4096}, true},
		{"fused gate_up", "model.layers.0.mlp.gate_up_proj.weight", []int{16384, 4096}, true},
		{"router", "model.layers.0.mlp.router.weight", []int{8, 4096}, true},
		{"mixtral expert", "model.layers.0.block_sparse_moe.experts.0.w1.weight", []int{8192, 4096}, true},
		{"plain up_proj", "model.layers.0.mlp.up_proj.weight", []int{8192, 4096}, true},
		{"norm stays host", "model.layers.0.input_layernorm.weight", []int{4096}, false},
		{"embedding stays host", "model.embed_tokens.weight", []int{32, 4096}, false},
		{"unmapped stays host", "model.layers.0.unknown_blob.weight", []int{8, 8}, false},
	}
	for _, c := range cases {
		if got := DecodedFloatGoesToQ8(cfg, c.canon, c.dims); got != c.want {
			t.Errorf("%s: DecodedFloatGoesToQ8(%q, %v) = %v, want %v", c.name, c.canon, c.dims, got, c.want)
		}
	}
}

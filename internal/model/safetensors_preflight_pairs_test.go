package model

import (
	"encoding/json"
	"strings"
	"testing"
)

// Pair admission and companion consumption are one boundary: admitting U8 scales
// without validating the same-shard pair would turn the old false refusal into
// false readiness for malformed metadata.
// fak-test:runtime fast est=30ms lane=default
func TestSafetensorsPreflightMXFP4PairBoundaries(t *testing.T) {
	cfg := Config{ModelType: "gpt_oss", NumLayers: 1, NumExperts: 2, HiddenSize: 32, IntermediateSize: 64, NumHeads: 1, NumKVHeads: 1, HeadDim: 32}
	config, err := json.Marshal(cfg)
	if err != nil {
		t.Fatal(err)
	}
	const base = "model.layers.0.mlp.experts.gate_up_proj"
	const blocks, scales = base + "_blocks", base + "_scales"
	const first, second = "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"
	for _, tc := range []struct {
		name      string
		roleError string
		refused   bool
	}{
		{"same-shard", "", false},
		{"missing-scales", "", true},
		{"missing-blocks", "", true},
		{"cross-shard", "", true},
		{"duplicate-blocks", "appears in both", true},
		{"blocks-dtype", "dtypes", true},
		{"scales-dtype", "dtypes", true},
		{"blocks-rank", "shape", true},
		{"scales-grid", "shape", true},
		{"zero-dimension", "shape", true},
		{"decoded-axes", "shape", true},
		{"unclaimed-u8-scale", "", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			a := map[string]safetensorsTestTensor{
				blocks: {"U8", []int{2, 128, 1, 16}},
				scales: {"U8", []int{2, 128, 1}},
			}
			b := map[string]safetensorsTestTensor{"model.norm.weight": {"F32", []int{32}}}
			index := map[string]string{blocks: first, scales: first, "model.norm.weight": second}
			switch tc.name {
			case "missing-scales":
				delete(a, scales)
				delete(index, scales)
			case "missing-blocks":
				delete(a, blocks)
				delete(index, blocks)
			case "cross-shard":
				b[scales] = a[scales]
				delete(a, scales)
				index[scales] = second
			case "duplicate-blocks":
				b[blocks] = a[blocks]
			case "blocks-dtype":
				a[blocks] = safetensorsTestTensor{"F16", []int{2, 128, 1, 16}}
			case "scales-dtype":
				a[scales] = safetensorsTestTensor{"F32", []int{2, 128, 1}}
			case "blocks-rank":
				a[blocks] = safetensorsTestTensor{"U8", []int{2, 128, 16}}
			case "scales-grid":
				a[scales] = safetensorsTestTensor{"U8", []int{2, 1, 128}}
			case "zero-dimension":
				a[blocks] = safetensorsTestTensor{"U8", []int{2, 128, 0, 16}}
			case "decoded-axes":
				a[blocks] = safetensorsTestTensor{"U8", []int{2, 32, 4, 16}}
				a[scales] = safetensorsTestTensor{"U8", []int{2, 32, 4}}
			case "unclaimed-u8-scale":
				a["unclaimed_scales"] = safetensorsTestTensor{"U8", []int{1}}
				index["unclaimed_scales"] = first
			}
			dir := t.TempDir()
			writeSafetensorsFixture(t, dir, index, map[string]map[string]safetensorsTestTensor{first: a, second: b}, "llama_small_config.json")
			plan, err := BuildSafetensorsPreflight(SafetensorsPreflightInput{Dir: dir, HFConfig: config})
			if err != nil {
				t.Fatal(err)
			}
			if plan.AdmissionErr != nil || len(plan.Missing) != 0 || len(plan.Orphans) != 0 {
				t.Fatalf("unexpected admission/index errors: %v missing=%v orphan=%v", plan.AdmissionErr, plan.Missing, plan.Orphans)
			}
			refused := len(plan.RoleErrors)+len(plan.HostGaps)+len(plan.ScaleGaps) != 0
			if refused != tc.refused {
				t.Errorf("refused=%t, want %t: roles=%v host=%v scales=%v", refused, tc.refused, plan.RoleErrors, plan.HostGaps, plan.ScaleGaps)
			}
			if tc.roleError != "" && !strings.Contains(plan.RoleErrors[blocks], tc.roleError) {
				t.Errorf("block error = %q, want %q", plan.RoleErrors[blocks], tc.roleError)
			}
			for _, row := range plan.Tensors {
				if row.Name != blocks && row.Name != scales {
					continue
				}
				if tc.name == "same-shard" && (row.Role != SafetensorsRoleMXFP4 || row.Representation != SafetensorsRepQ8) {
					t.Errorf("%s = %s/%s, want consumed MXFP4/Q8 pair", row.Name, row.Role, row.Representation)
				}
				if tc.name == "cross-shard" && (row.Role != SafetensorsRolePlain || row.Representation != "") {
					t.Errorf("cross-shard %s = %s/%s, want unpaired plain/empty", row.Name, row.Role, row.Representation)
				}
			}
			if tc.name == "unclaimed-u8-scale" {
				got := plan.HostGaps["host_dequant(U8)"]
				if len(got) != 1 || got[0] != "unclaimed_scales" {
					t.Errorf("unclaimed U8 host gap = %v, want only unclaimed_scales", got)
				}
			}
		})
	}
}

// A global companion census must not hide the existing per-shard FP8 loader's
// missing-scale refusal either.
// fak-test:runtime fast est=10ms lane=default
func TestSafetensorsPreflightFP8PairStaysInShard(t *testing.T) {
	const weight = "model.layers.0.self_attn.q_proj.weight"
	const scale = weight + "_scale_inv"
	const first, second = "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"
	for _, split := range []bool{false, true} {
		a := map[string]safetensorsTestTensor{
			weight: {"F8_E4M3", []int{32, 32}},
			scale:  {"F32", []int{1, 1}},
		}
		b := map[string]safetensorsTestTensor{"model.norm.weight": {"F32", []int{32}}}
		index := map[string]string{weight: first, scale: first, "model.norm.weight": second}
		if split {
			b[scale] = a[scale]
			delete(a, scale)
			index[scale] = second
		}
		dir := t.TempDir()
		writeSafetensorsFixture(t, dir, index, map[string]map[string]safetensorsTestTensor{first: a, second: b}, "llama_small_config.json")
		plan, err := BuildSafetensorsPreflight(SafetensorsPreflightInput{Dir: dir})
		if err != nil {
			t.Fatal(err)
		}
		if split {
			if len(plan.ScaleGaps) != 1 || plan.ScaleGaps[0].Weight != weight || plan.ScaleGaps[0].Sibling != scale {
				t.Errorf("cross-shard scale gaps = %v, want missing companion for %s", plan.ScaleGaps, weight)
			}
		} else if len(plan.ScaleGaps)+len(plan.HostGaps)+len(plan.RoleErrors) != 0 {
			t.Errorf("same-shard pair refused: scale=%v host=%v role=%v", plan.ScaleGaps, plan.HostGaps, plan.RoleErrors)
		}
	}
}

// The admitted pair's decoded [E,C,R] axes must agree with the real decoder and
// source-MoE transforms, including source roles which cannot consume rank 3.
// fak-test:runtime fast est=10ms lane=default
func TestSafetensorsPreflightMXFP4DecodedRoleMatchesLoader(t *testing.T) {
	cfg := Config{ModelType: "gpt_oss", NumLayers: 1, NumExperts: 2, HiddenSize: 32, IntermediateSize: 64}
	for _, tc := range []struct {
		name   string
		rows   int
		groups int
	}{
		{"model.layers.0.mlp.experts.gate_up_proj", 128, 1},
		{"model.layers.0.mlp.experts.down_proj", 32, 2},
		{"model.layers.0.mlp.experts.gate_up_proj", 32, 4},
		{"model.layers.0.self_attn.qkv_proj.weight", 128, 1},
		{"model.layers.0.mlp.router.bias", 128, 1},
		{"model.layers.0.block_sparse_moe.experts.0.w1.weight", 128, 1},
		{"model.layers.0.block_sparse_moe.experts.2.w1.weight", 128, 1},
		{"model.layers.0.unmapped", 128, 1},
	} {
		blocks := stEntry{Dtype: "U8", Shape: []int{2, tc.rows, tc.groups, 16}}
		scales := stEntry{Dtype: "U8", Shape: []int{2, tc.rows, tc.groups}}
		shape, err := preflightMXFP4Shape(tc.name, blocks, scales)
		if err != nil {
			t.Fatal(err)
		}
		fb, decoded, err := decodeMXFP4Blocks(tc.name, blocks, scales, make([]byte, 2*tc.rows*tc.groups*16), make([]byte, 2*tc.rows*tc.groups))
		if err != nil || !sameShape(shape, decoded) {
			t.Fatalf("%s: decoded shape=%v err=%v, plan shape=%v", tc.name, decoded, err, shape)
		}
		representation, planErr := preflightMXFP4Representation(cfg, tc.name, shape)
		m := &Model{Cfg: cfg, manifest: map[string]tensorMeta{}, q8w: map[string]*q8Tensor{}}
		var raw []byte
		off := 0
		loadErr := quantizeDecodedTensorInto(tc.name, decoded, fb, m, false, &raw, &off)
		if (planErr != nil) != (loadErr != nil) {
			t.Fatalf("%s: plan error=%v loader error=%v", tc.name, planErr, loadErr)
		}
		if loadErr != nil {
			continue
		}
		want := SafetensorsRepF32Host
		if len(m.q8w) != 0 {
			want = SafetensorsRepQ8
		}
		if representation != want {
			t.Errorf("%s: representation=%q, actual loader representation=%q", tc.name, representation, want)
		}
	}
}

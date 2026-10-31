package model

import (
	"encoding/json"
	"strings"
	"testing"
)

// The FP8 decoded fallback splits valid fused projections into Q8 parts; their
// packed source name is not itself an ordinary isQuantWeight match.
// fak-test:runtime fast est=20ms lane=default
func TestSafetensorsPreflightFP8FusedProjectionMatchesLoader(t *testing.T) {
	cfg := Config{ModelType: "llama", NumLayers: 1, HiddenSize: 32, IntermediateSize: 64, NumHeads: 2, NumKVHeads: 1, HeadDim: 16}
	config, err := json.Marshal(cfg)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name      string
		shape     []int
		q8Parts   int
		malformed bool
	}{
		{"model.layers.0.self_attn.qkv_proj.weight", []int{64, 32}, 3, false},
		{"model.layers.0.self_attn.qkv_proj.weight", []int{63, 32}, 0, true},
		{"model.layers.0.self_attn.qkv_proj.weight", []int{64, 31}, 0, true},
		{"model.layers.0.self_attn.qkv_proj.weight", []int{1, 64, 32}, 0, true},
		{"model.layers.0.mlp.gate_up_proj.weight", []int{128, 32}, 2, false},
		{"model.layers.0.mlp.gate_up_proj.weight", []int{127, 32}, 0, true},
		{"model.layers.0.mlp.up_proj.weight", []int{64, 32}, 1, false},
		{"model.layers.1.self_attn.qkv_proj.weight", []int{64, 32}, 0, false},
	} {
		scaleName := tc.name + "_scale_inv"
		dir := t.TempDir()
		writeSafetensorsFixture(t, dir, nil, map[string]map[string]safetensorsTestTensor{
			"model.safetensors": {
				tc.name:   {"F8_E4M3", tc.shape},
				scaleName: {"F32", []int{1, 1}},
			},
		}, "llama_small_config.json")
		plan, err := BuildSafetensorsPreflight(SafetensorsPreflightInput{Dir: dir, HFConfig: config})
		if err != nil {
			t.Fatal(err)
		}
		if plan.AdmissionErr != nil || len(plan.ScaleGaps)+len(plan.HostGaps) != 0 {
			t.Fatalf("%s: admission=%v scale=%v host=%v", tc.name, plan.AdmissionErr, plan.ScaleGaps, plan.HostGaps)
		}
		msg, refused := plan.RoleErrors[tc.name]
		if refused != tc.malformed || (tc.malformed && !strings.Contains(msg, "shape")) {
			t.Errorf("%s shape %v: role error=%q, malformed=%t", tc.name, tc.shape, msg, tc.malformed)
		}
		wantRep := SafetensorsRepF32Host
		if tc.malformed {
			wantRep = ""
		} else if tc.q8Parts > 0 {
			wantRep = SafetensorsRepQ8
		}
		found := false
		for _, row := range plan.Tensors {
			if row.Name == tc.name {
				found = true
				if row.Role != SafetensorsRoleFP8BlockScale || row.Representation != wantRep {
					t.Errorf("%s shape %v: role/rep=%s/%s, want FP8/%s", tc.name, tc.shape, row.Role, row.Representation, wantRep)
				}
			}
		}
		if !found {
			t.Fatalf("missing plan row %s", tc.name)
		}

		// Compare against the actual FP8 decode/split path with bounded zero
		// weights and a real F32 scale, not another copy of the shape formula.
		weightEntry := stEntry{Dtype: "F8_E4M3", Shape: tc.shape}
		scaleEntry := stEntry{Dtype: "F32", Shape: []int{1, 1}}
		hdr := map[string]json.RawMessage{}
		for name, entry := range map[string]stEntry{tc.name: weightEntry, scaleName: scaleEntry} {
			hdr[name], err = json.Marshal(entry)
			if err != nil {
				t.Fatal(err)
			}
		}
		elems := 1
		for _, dim := range tc.shape {
			elems *= dim
		}
		weights := make([]byte, elems)
		read := func(entry stEntry) ([]byte, error) {
			if entry.Dtype == "F8_E4M3" {
				return weights, nil
			}
			return []byte{0, 0, 128, 63}, nil // F32 scale = 1
		}
		m := &Model{Cfg: cfg, manifest: map[string]tensorMeta{}, q8w: map[string]*q8Tensor{}}
		var raw []byte
		off := 0
		handled, loadErr := quantizeFP8BlockScaleTensorInto(tc.name, hdr, read, m, false, &raw, &off, map[string]bool{})
		if !handled || (loadErr != nil) != tc.malformed {
			t.Fatalf("%s shape %v: loader handled=%t err=%v, malformed=%t", tc.name, tc.shape, handled, loadErr, tc.malformed)
		}
		if len(m.q8w) != tc.q8Parts {
			t.Errorf("%s shape %v: loader Q8 parts=%d, want %d", tc.name, tc.shape, len(m.q8w), tc.q8Parts)
		}
	}
}

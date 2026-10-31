package model

import (
	"encoding/json"
	"strings"
	"testing"
)

// Fused splits reject pre-existing output components, but a plain component
// loaded after the split is an accepted overwrite. Preserve that ordering.
// fak-test:runtime fast est=30ms lane=default
func TestSafetensorsPreflightFusedCollisionMatchesLoadOrder(t *testing.T) {
	cfg := Config{ModelType: "llama", NumLayers: 1, HiddenSize: 32, IntermediateSize: 64, NumHeads: 2, NumKVHeads: 1, HeadDim: 16}
	config, err := json.Marshal(cfg)
	if err != nil {
		t.Fatal(err)
	}
	const first, second = "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"
	for _, tc := range []struct {
		label          string
		fusedSuffix    string
		fusedDtype     string
		component      string
		componentDtype string
		placement      string
		conflict       bool
	}{
		{"qkv-alone", suffixQKVProj, "F8_E4M3", "", "", "same", false},
		{"q-before-qkv", suffixQKVProj, "F8_E4M3", suffixQProj, "F32", "same", true},
		{"v-after-qkv", suffixQKVProj, "F8_E4M3", suffixVProj, "F32", "same", false},
		{"q-earlier-shard", suffixQKVProj, "F8_E4M3", suffixQProj, "F32", "component-first", true},
		{"q-later-shard", suffixQKVProj, "F8_E4M3", suffixQProj, "F32", "fused-first", false},
		{"gate-before-gate-up", suffixGateUpProj, "F8_E4M3", suffixGateProj, "BF16", "same", true},
		{"up-after-gate-up", suffixGateUpProj, "F8_E4M3", suffixUpProj, "F16", "same", false},
		{"f32-q-before-qkv", suffixQKVProj, "F32", suffixQProj, "F32", "same", true},
		{"bf16-gate-before-gate-up", suffixGateUpProj, "BF16", suffixGateProj, "F32", "same", true},
	} {
		t.Run(tc.label, func(t *testing.T) {
			fused := layerPrefix(0) + tc.fusedSuffix
			fusedShape := []int{64, 32}
			if tc.fusedSuffix == suffixGateUpProj {
				fusedShape = []int{128, 32}
			}
			fusedShard, componentShard := first, first
			if tc.placement == "component-first" {
				fusedShard = second
			} else if tc.placement == "fused-first" {
				componentShard = second
			}
			shards := map[string]map[string]safetensorsTestTensor{
				first:  {},
				second: {"model.norm.weight": {"F32", []int{32}}},
			}
			shards[fusedShard][fused] = safetensorsTestTensor{tc.fusedDtype, fusedShape}
			if tc.fusedDtype == "F8_E4M3" {
				shards[fusedShard][fused+"_scale_inv"] = safetensorsTestTensor{"F32", []int{1, 1}}
			}
			if tc.component != "" {
				rows := 32
				if tc.component == suffixVProj || tc.component == suffixKProj {
					rows = 16
				} else if tc.component == suffixGateProj || tc.component == suffixUpProj {
					rows = 64
				}
				shards[componentShard][layerPrefix(0)+tc.component] = safetensorsTestTensor{tc.componentDtype, []int{rows, 32}}
			}
			index := map[string]string{}
			for shard, tensors := range shards {
				for name := range tensors {
					index[name] = shard
				}
			}
			dir := t.TempDir()
			writeSafetensorsFixture(t, dir, index, shards, "llama_small_config.json")
			plan, err := BuildSafetensorsPreflight(SafetensorsPreflightInput{Dir: dir, HFConfig: config})
			if err != nil {
				t.Fatal(err)
			}
			if plan.AdmissionErr != nil || len(plan.Missing)+len(plan.Orphans)+len(plan.HostGaps)+len(plan.ScaleGaps) != 0 {
				t.Fatalf("unrelated refusal: admission=%v missing=%v orphan=%v host=%v scale=%v", plan.AdmissionErr, plan.Missing, plan.Orphans, plan.HostGaps, plan.ScaleGaps)
			}
			msg, refused := plan.RoleErrors[fused]
			if refused != tc.conflict || (refused && !strings.Contains(msg, "already exist")) {
				t.Errorf("fused role error=%q, want conflict=%t", msg, tc.conflict)
			}

			// Exercise the actual per-shard named loader with one shared Q8 store,
			// including the accepted later-plain-overwrite controls.
			m := &Model{Cfg: cfg, manifest: map[string]tensorMeta{}, q8w: map[string]*q8Tensor{}}
			var raw []byte
			off := 0
			var loadErr error
			for _, shard := range []string{first, second} {
				hdr := map[string]json.RawMessage{}
				for name, spec := range shards[shard] {
					hdr[name], err = json.Marshal(stEntry{Dtype: spec.dtype, Shape: spec.dims})
					if err != nil {
						t.Fatal(err)
					}
				}
				read := func(entry stEntry) ([]byte, error) {
					if entry.Dtype == "F32" && sameShape(entry.Shape, []int{1, 1}) {
						return []byte{0, 0, 128, 63}, nil // F32 scale = 1
					}
					elems, width := 1, 1
					for _, dim := range entry.Shape {
						elems *= dim
					}
					if entry.Dtype == "F32" {
						width = 4
					} else if entry.Dtype == "F16" || entry.Dtype == "BF16" {
						width = 2
					}
					return make([]byte, elems*width), nil
				}
				loadErr = quantizeNamedTensorsInto(safetensorsTensorNames(hdr), hdr, read, nil, m, false, &raw, &off)
				if loadErr != nil {
					break
				}
			}
			if (loadErr != nil) != tc.conflict || (loadErr != nil && !strings.Contains(loadErr.Error(), "already exist")) {
				t.Fatalf("actual loader error=%v, want conflict=%t", loadErr, tc.conflict)
			}
		})
	}
}

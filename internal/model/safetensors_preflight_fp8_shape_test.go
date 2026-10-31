package model

import (
	"encoding/json"
	"testing"
)

// A supported dtype/pair is insufficient when the selected FP8 decoder rejects
// the source geometry. Host fallback does not inherit direct-Q8 alignment.
// fak-test:runtime fast est=20ms lane=default
func TestSafetensorsPreflightFP8ShapeMatchesLoader(t *testing.T) {
	const direct = "model.layers.0.mlp.up_proj.weight"
	const host = "unmapped.weight"
	cfg := Config{ModelType: "llama", NumLayers: 1, HiddenSize: 32, IntermediateSize: 64, NumHeads: 2, NumKVHeads: 1, HeadDim: 16}
	config, err := json.Marshal(cfg)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		label  string
		name   string
		shape  []int
		refuse bool
		q8     bool
	}{
		{"direct-valid", direct, []int{2, 32}, false, true},
		{"direct-flat-scale-valid", direct, []int{2, 32}, false, true},
		{"direct-ragged-rows", direct, []int{3, 32}, false, true},
		{"direct-unaligned", direct, []int{2, 31}, true, false},
		{"host-unaligned-valid", host, []int{2, 31}, false, false},
		{"rank-one", direct, []int{32}, true, false},
		{"rank-three", host, []int{1, 2, 32}, true, false},
		{"zero-row", direct, []int{0, 32}, true, false},
		{"negative-column", host, []int{2, -1}, true, false},
		{"overflow", direct, []int{int(^uint(0)>>1)/2 + 1, 2}, true, false},
	} {
		t.Run(tc.label, func(t *testing.T) {
			scale := tc.name + "_scale_inv"
			scaleShape := []int{1, 1}
			if tc.label == "direct-flat-scale-valid" {
				// The actual loader uses decoded scale length, not header axes.
				scaleShape = []int{1}
			}
			dir := t.TempDir()
			writeSafetensorsFixture(t, dir, nil, map[string]map[string]safetensorsTestTensor{
				"model.safetensors": {
					tc.name: {"F8_E4M3", tc.shape},
					scale:   {"F32", scaleShape},
				},
			}, "llama_small_config.json")
			plan, err := BuildSafetensorsPreflight(SafetensorsPreflightInput{Dir: dir, HFConfig: config})
			if err != nil {
				t.Fatal(err)
			}
			if plan.AdmissionErr != nil || len(plan.HostGaps)+len(plan.ScaleGaps) != 0 {
				t.Fatalf("unexpected admission=%v host=%v scale=%v", plan.AdmissionErr, plan.HostGaps, plan.ScaleGaps)
			}
			if msg, refused := plan.RoleErrors[tc.name]; refused != tc.refuse {
				t.Errorf("role error=%q, want refused=%t", msg, tc.refuse)
			}
			wantRep := SafetensorsRepF32Host
			if tc.refuse {
				wantRep = ""
			} else if tc.q8 {
				wantRep = SafetensorsRepQ8
			}
			found := false
			for _, row := range plan.Tensors {
				if row.Name == tc.name {
					found = true
					if row.Role != SafetensorsRoleFP8BlockScale || row.Representation != wantRep {
						t.Errorf("role/representation=%s/%s, want FP8/%s", row.Role, row.Representation, wantRep)
					}
				}
			}
			if !found {
				t.Fatal("missing weight census row")
			}

			// Real format-specific loader, finite zero weights, and F32 scale 1.
			// Invalid/overflowing shapes need no payload allocation: the real
			// decoder rejects their metadata before consuming weight elements.
			var weights []byte
			if n, ok := checkedShapeProduct(tc.shape...); ok && n <= 1024 {
				weights = make([]byte, n)
			}
			hdr := map[string]json.RawMessage{}
			for name, entry := range map[string]stEntry{
				tc.name: {Dtype: "F8_E4M3", Shape: tc.shape},
				scale:   {Dtype: "F32", Shape: scaleShape},
			} {
				hdr[name], err = json.Marshal(entry)
				if err != nil {
					t.Fatal(err)
				}
			}
			read := func(entry stEntry) ([]byte, error) {
				if entry.Dtype == "F8_E4M3" {
					return weights, nil
				}
				return []byte{0, 0, 128, 63}, nil
			}
			m := &Model{Cfg: cfg, manifest: map[string]tensorMeta{}, q8w: map[string]*q8Tensor{}}
			var raw []byte
			off := 0
			handled, loadErr := quantizeFP8BlockScaleTensorInto(tc.name, hdr, read, m, false, &raw, &off, map[string]bool{})
			if !handled || (loadErr != nil) != tc.refuse {
				t.Fatalf("loader handled=%t err=%v, want refused=%t", handled, loadErr, tc.refuse)
			}
			if (len(m.q8w) != 0) != tc.q8 {
				t.Errorf("Q8 tensors=%d, want Q8=%t", len(m.q8w), tc.q8)
			}
		})
	}
}

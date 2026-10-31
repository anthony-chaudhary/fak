package model

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

// Dense E8M0 scales sort before their weights. Even that first visit must
// validate a same-shard owner, not silently describe an orphan scale as Q8.
// fak-test:runtime fast est=20ms lane=default
func TestSafetensorsPreflightDenseScaleOwnerMatchesLoader(t *testing.T) {
	const weight = "layers.0.attn.wq_a.weight"
	const scale = "layers.0.attn.wq_a.scale"
	for _, profile := range []struct {
		name        string
		config      string
		weightShape []int
		scaleShape  []int
		v41         bool
	}{
		{"v4", "deepseek_v4_flash_config.json", []int{1024, 4096}, []int{8, 32}, false},
		{"v41", "deepseek_v41_flash_config.json", []int{1280, 5120}, []int{40, 160}, true},
	} {
		for _, kind := range []string{"valid", "missing-owner", "cross-shard-owner", "scale-dtype", "scale-rows", "scale-columns", "scale-rank", "weight-dtype", "weight-shape"} {
			t.Run(profile.name+"/"+kind, func(t *testing.T) {
				owner := safetensorsTestTensor{"F8_E4M3", append([]int(nil), profile.weightShape...)}
				sibling := safetensorsTestTensor{"F8_E8M0", append([]int(nil), profile.scaleShape...)}
				switch kind {
				case "scale-dtype":
					sibling.dtype = "F32"
				case "scale-rows":
					sibling.dims[0]--
				case "scale-columns":
					sibling.dims[1]--
				case "scale-rank":
					sibling.dims = append([]int{1}, sibling.dims...)
				case "weight-dtype":
					owner.dtype = "F32"
				case "weight-shape":
					owner.dims[0]--
				}
				local := map[string]safetensorsTestTensor{scale: sibling}
				shards := map[string]map[string]safetensorsTestTensor{"model.safetensors": local}
				var index map[string]string
				switch kind {
				case "missing-owner":
				case "cross-shard-owner":
					shards["other.safetensors"] = map[string]safetensorsTestTensor{weight: owner}
					index = map[string]string{scale: "model.safetensors", weight: "other.safetensors"}
				default:
					local[weight] = owner
				}
				dir := t.TempDir()
				writeSafetensorsFixture(t, dir, index, shards, profile.config)
				plan, err := BuildSafetensorsPreflight(SafetensorsPreflightInput{Dir: dir})
				if err != nil {
					t.Fatal(err)
				}
				if profile.v41 {
					if !errors.Is(plan.AdmissionErr, ErrV41NativeUnsupported) {
						t.Fatalf("admission=%v, want preserved V4.1 refusal", plan.AdmissionErr)
					}
				} else if plan.AdmissionErr != nil {
					t.Fatalf("V4 admission=%v", plan.AdmissionErr)
				}
				wantRefused := kind != "valid"
				if got := len(plan.RoleErrors)+len(plan.ScaleGaps) != 0; got != wantRefused {
					t.Errorf("role=%v scale=%v, want refused=%t", plan.RoleErrors, plan.ScaleGaps, wantRefused)
				}
				if len(plan.Missing)+len(plan.Orphans)+len(plan.HostGaps) != 0 {
					t.Errorf("unexpected index/host gaps: missing=%v orphan=%v host=%v", plan.Missing, plan.Orphans, plan.HostGaps)
				}
				if kind == "missing-owner" || kind == "cross-shard-owner" {
					if msg := plan.RoleErrors[scale]; !strings.Contains(msg, "missing "+weight) {
						t.Errorf("scale role error=%q, want missing owner %s", msg, weight)
					}
				}

				// Invoke the real loader's scale-first visit, not a copied validator.
				// It validates/defer pairs using metadata only, including V4.1's
				// not-yet-admitted implementation; no decoder or GPU is exercised.
				hdr := map[string]json.RawMessage{}
				for name, spec := range local {
					hdr[name], err = json.Marshal(stEntry{Dtype: spec.dtype, Shape: spec.dims})
					if err != nil {
						t.Fatal(err)
					}
				}
				read := func(stEntry) ([]byte, error) {
					t.Fatal("scale-first visit unexpectedly read tensor payload")
					return nil, nil
				}
				m := &Model{Cfg: plan.Config}
				var handled bool
				var loadErr error
				if profile.v41 {
					handled, loadErr = quantizeV41DenseFP8TensorInto(scale, hdr, read, m, map[string]bool{})
				} else {
					handled, loadErr = quantizeV4DenseFP8TensorInto(scale, hdr, read, m, map[string]bool{})
				}
				if !handled || (loadErr != nil) != wantRefused {
					t.Errorf("loader handled=%t err=%v, want refused=%t", handled, loadErr, wantRefused)
				}
			})
		}
	}
}

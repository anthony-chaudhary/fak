package model

import (
	"errors"
	"strings"
	"testing"
)

// A valid dtype does not make an invalid scale tile grid usable by the loader.
// fak-test:runtime fast est=10ms lane=default
func TestSafetensorsPreflightDenseScaleShape(t *testing.T) {
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
		t.Run(profile.name, func(t *testing.T) {
			for _, malformed := range []bool{false, true} {
				scaleShape := append([]int(nil), profile.scaleShape...)
				if malformed {
					scaleShape[0]--
				}
				const weight = "layers.0.attn.wq_a.weight"
				const scale = "layers.0.attn.wq_a.scale"
				dir := t.TempDir()
				writeSafetensorsFixture(t, dir, nil, map[string]map[string]safetensorsTestTensor{
					"model.safetensors": {
						weight: {"F8_E4M3", profile.weightShape},
						scale:  {"F8_E8M0", scaleShape},
					},
				}, profile.config)
				plan, err := BuildSafetensorsPreflight(SafetensorsPreflightInput{Dir: dir})
				if err != nil {
					t.Fatal(err)
				}
				if profile.v41 {
					if !errors.Is(plan.AdmissionErr, ErrV41NativeUnsupported) {
						t.Fatalf("admission = %v, want preserved V4.1 refusal", plan.AdmissionErr)
					}
				} else if plan.AdmissionErr != nil {
					t.Fatalf("V4 admission = %v", plan.AdmissionErr)
				}
				msg, refused := plan.RoleErrors[scale]
				if refused != malformed || (malformed && !strings.Contains(msg, "shape")) {
					t.Errorf("malformed=%t: scale role error = %q, want shape refusal only for malformed grid", malformed, msg)
				}
				if !malformed && (len(plan.RoleErrors) != 0 || len(plan.ScaleGaps) != 0 || len(plan.HostGaps) != 0) {
					t.Errorf("valid pair gaps: role=%v scale=%v host=%v", plan.RoleErrors, plan.ScaleGaps, plan.HostGaps)
				}
			}
		})
	}
}

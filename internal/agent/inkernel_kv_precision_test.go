package agent

import (
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

func kvPrecisionDenseCfg() model.Config {
	return model.Config{NumLayers: 2, NumHeads: 4, NumKVHeads: 2, HeadDim: 32, HiddenSize: 128, IntermediateSize: 256, RopeTheta: 10000}
}

func kvPrecisionHybridCfg() model.Config {
	cfg := kvPrecisionDenseCfg()
	cfg.LayerTypes = []string{"linear_attention", "full_attention"}
	return cfg
}

// TestKVPrecisionDeviceHybridQ8RefusesTyped is the Raider crash-loop shape: a CUDA
// device planner over a Qwen hybrid with --kv-precision q8_0 must refuse with the
// typed error at construction and at the request seam, never panic the process.
// fak-test:runtime fast est=10ms lane=default
func TestKVPrecisionDeviceHybridQ8RefusesTyped(t *testing.T) {
	cfg := kvPrecisionHybridCfg()
	if !cfg.IsQwen35Hybrid() {
		t.Fatal("fixture is not a hybrid config")
	}
	be := compute.Default()
	p := NewInKernelPlannerWithConfig(&model.Model{Cfg: cfg}, nil, "kv-precision-device-hybrid", true, be, false, InKernelPlannerConfig{KVPrecision: model.KVPrecisionQ8_0})
	err := p.KVPrecisionConfigError()
	var typed *KVPrecisionUnrealizedError
	if !errors.Is(err, ErrKVPrecisionUnrealized) || !errors.As(err, &typed) {
		t.Fatalf("KVPrecisionConfigError = %v, want *KVPrecisionUnrealizedError", err)
	}
	if typed.Path != KVPrecisionPathDeviceHAL || typed.Precision != model.KVPrecisionQ8_0 {
		t.Fatalf("refusal = %+v, want device-hal q8_0", typed)
	}
	_, callErr := recoverDeviceCall(func() (struct{}, error) {
		p.configureNativeSession(p.m.NewSession())
		return struct{}{}, nil
	})
	if !errors.Is(callErr, ErrKVPrecisionUnrealized) {
		t.Fatalf("request seam = %v, want typed refusal", callErr)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestValidateKVPrecisionMatrix(t *testing.T) {
	be := compute.Default()
	cases := []struct {
		name    string
		cfg     model.Config
		prec    model.KVPrecision
		backend compute.Backend
		metal   bool
		path    string
	}{
		{"unset", kvPrecisionHybridCfg(), "", be, false, ""},
		{"f32-device-hybrid", kvPrecisionHybridCfg(), model.KVPrecisionFP32, be, false, ""},
		{"q8-host-dense", kvPrecisionDenseCfg(), model.KVPrecisionQ8_0, nil, false, ""},
		{"q8-device-dense", kvPrecisionDenseCfg(), model.KVPrecisionQ8_0, be, false, KVPrecisionPathDeviceHAL},
		{"q8-metal-dense", kvPrecisionDenseCfg(), model.KVPrecisionQ8_0, nil, true, KVPrecisionPathMetal},
		{"q8-host-hybrid", kvPrecisionHybridCfg(), model.KVPrecisionQ8_0, nil, false, KVPrecisionPathHost},
		{"f16-host-dense", kvPrecisionDenseCfg(), model.KVPrecisionFP16, nil, false, KVPrecisionPathHost},
		{"f16-device-hybrid", kvPrecisionHybridCfg(), model.KVPrecisionFP16, be, false, ""},
		{"bf16-host-dense", kvPrecisionDenseCfg(), model.KVPrecisionBF16, nil, false, KVPrecisionPathHost},
	}
	for _, tc := range cases {
		err := ValidateKVPrecision(tc.cfg, tc.prec, tc.backend, tc.metal)
		if tc.path == "" {
			if err != nil {
				t.Fatalf("%s: unexpected refusal %v", tc.name, err)
			}
			continue
		}
		var typed *KVPrecisionUnrealizedError
		if !errors.As(err, &typed) || !errors.Is(err, ErrKVPrecisionUnrealized) || typed.Path != tc.path {
			t.Fatalf("%s: got %v, want typed refusal on %s", tc.name, err, tc.path)
		}
	}
}

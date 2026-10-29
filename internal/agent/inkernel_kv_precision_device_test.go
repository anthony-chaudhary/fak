package agent

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

type kvPrecisionNamedBackend struct {
	compute.Backend
	name string
}

func (b kvPrecisionNamedBackend) Name() string { return b.name }

// fak-test:runtime fast est=5ms lane=default
func TestInKernelKVPrecisionDeviceResolution(t *testing.T) {
	t.Setenv("FAK_PAGED_KV", "")
	dense := model.Config{NumLayers: 2, NumKVHeads: 2, HeadDim: 8}
	hybrid := dense
	hybrid.LayerTypes = []string{"linear_attention", "full_attention"}
	cuda := kvPrecisionNamedBackend{Backend: compute.Default(), name: "cuda"}
	vulkan := kvPrecisionNamedBackend{Backend: compute.Default(), name: "vulkan"}
	for _, tc := range []struct {
		name     string
		cfg      model.Config
		backend  compute.Backend
		want     model.KVPrecision
		realized model.KVPrecision
		fallback bool
	}{
		{"cuda fp16", dense, cuda, model.KVPrecisionFP16, model.KVPrecisionFP16, false},
		{"cuda bf16", dense, cuda, model.KVPrecisionBF16, model.KVPrecisionBF16, false},
		{"vulkan fp16 falls back", dense, vulkan, model.KVPrecisionFP16, model.KVPrecisionFP32, true},
		{"cuda hybrid realizes bf16", hybrid, cuda, model.KVPrecisionBF16, model.KVPrecisionBF16, false},
		{"cuda hybrid realizes fp16", hybrid, cuda, model.KVPrecisionFP16, model.KVPrecisionFP16, false},
		{"vulkan hybrid falls back", hybrid, vulkan, model.KVPrecisionFP16, model.KVPrecisionFP32, true},
		{"host keeps its own checks", dense, nil, model.KVPrecisionFP16, model.KVPrecisionFP16, false},
		{"q8_0 keeps its own checks", dense, cuda, model.KVPrecisionQ8_0, model.KVPrecisionQ8_0, false},
		{"unset stays unset", dense, cuda, "", "", false},
	} {
		got, note := pickDeviceKVTier(tc.cfg, tc.backend, tc.want)
		if got != tc.realized || (note != "") != tc.fallback {
			t.Fatalf("%s: got %q note %q; want %q fallback=%v", tc.name, got, note, tc.realized, tc.fallback)
		}
		if tc.fallback && !strings.Contains(note, "source=device-kv-fallback") {
			t.Fatalf("%s: fallback note %q must record its source", tc.name, note)
		}
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestInKernelKVPrecisionDeviceBudgetTier(t *testing.T) {
	for prec, want := range map[model.KVPrecision]compute.KVPrecision{
		model.KVPrecisionFP16: compute.KVPrecisionF16,
		model.KVPrecisionBF16: compute.KVPrecisionBF16,
		model.KVPrecisionQ8_0: compute.KVPrecisionQ8,
		"":                    compute.KVPrecisionF32,
	} {
		if got := computeKVPrecisionFor(prec); got != want {
			t.Fatalf("computeKVPrecisionFor(%q) = %v, want %v", prec, got, want)
		}
	}
}

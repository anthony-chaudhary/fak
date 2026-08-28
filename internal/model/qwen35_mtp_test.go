package model

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

func qwen35MTPTestConfig() Config {
	return Config{
		ModelType:                 "qwen3_5_text",
		LayerTypes:                []string{"linear_attention"},
		MTPNumHiddenLayers:        1,
		MTPUseDedicatedEmbeddings: false,
	}
}

func completeQwen35MTPManifest() map[string]tensorMeta {
	man := make(map[string]tensorMeta, len(qwen35MTPRequiredTensors))
	for _, name := range qwen35MTPRequiredTensors {
		man[name] = tensorMeta{Dtype: "F32", Shape: []int{1}, Nbytes: 4}
	}
	return man
}

func TestQwen35MTPConfigParsesNestedMetadata(t *testing.T) {
	var cfg Config
	err := json.Unmarshal([]byte(`{
		"model_type": "qwen3_5",
		"text_config": {
			"model_type": "qwen3_5_text",
			"layer_types": ["linear_attention"],
			"mtp_num_hidden_layers": 1,
			"mtp_use_dedicated_embeddings": false
		}
	}`), &cfg)
	if err != nil {
		t.Fatalf("unmarshal nested Qwen MTP config: %v", err)
	}
	if cfg.MTPNumHiddenLayers != 1 || cfg.NumMTPLayers() != 1 {
		t.Fatalf("MTP depth = field:%d derived:%d, want 1/1", cfg.MTPNumHiddenLayers, cfg.NumMTPLayers())
	}
	if cfg.MTPUseDedicatedEmbeddings {
		t.Fatal("MTPUseDedicatedEmbeddings = true, want explicit nested false")
	}
}

func TestQwen35MTPAdmissionDefaultsEligibleSubstrateOn(t *testing.T) {
	mode, err := qwen35MTPAdmission(qwen35MTPTestConfig(), completeQwen35MTPManifest(), false)
	if err != nil {
		t.Fatalf("admit complete Qwen MTP substrate: %v", err)
	}
	if !mode.Eligible || !mode.Enabled || mode.Engine != "fak-native" || mode.Reason != "eligible-default-on" {
		t.Fatalf("mode = %+v, want eligible native substrate enabled by default", mode)
	}
}

func TestLoadedModelReportsQwen35MTPMode(t *testing.T) {
	m := &Model{Cfg: qwen35MTPTestConfig(), manifest: completeQwen35MTPManifest()}
	mode, err := m.Qwen35MTPMode(false)
	if err != nil {
		t.Fatalf("loaded model mode: %v", err)
	}
	if !mode.Enabled || mode.Engine != "fak-native" {
		t.Fatalf("loaded model mode = %+v, want enabled fak-native substrate", mode)
	}
}

func TestQwen35MTPAdmissionRejectsIneligibleConfig(t *testing.T) {
	cases := []struct {
		name string
		cfg  Config
		want string
	}{
		{name: "no metadata", cfg: Config{ModelType: "qwen3_5_text", LayerTypes: []string{"linear_attention"}}, want: "unsupported-mtp-depth"},
		{name: "dedicated embeddings", cfg: Config{ModelType: "qwen3_5_text", LayerTypes: []string{"linear_attention"}, MTPNumHiddenLayers: 1, MTPUseDedicatedEmbeddings: true}, want: "dedicated-mtp-embeddings-unsupported"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			mode, err := qwen35MTPAdmission(tc.cfg, completeQwen35MTPManifest(), false)
			if err != nil {
				t.Fatalf("admission: %v", err)
			}
			if mode.Eligible || mode.Enabled || mode.Engine != "fak-native" || mode.Reason != tc.want {
				t.Fatalf("mode = %+v, want ineligible fak-native reason %q", mode, tc.want)
			}
		})
	}
}

func TestQwen35MTPAdmissionExplicitDisableUsesOrdinaryNativeTarget(t *testing.T) {
	mode, err := qwen35MTPAdmission(qwen35MTPTestConfig(), completeQwen35MTPManifest(), true)
	if err != nil {
		t.Fatalf("disable complete Qwen MTP substrate: %v", err)
	}
	if !mode.Eligible || mode.Enabled || mode.Engine != "fak-native" || mode.Reason != "explicitly-disabled" {
		t.Fatalf("mode = %+v, want eligible disabled ordinary fak-native target", mode)
	}
	if strings.Contains(mode.Engine, "llama") || strings.Contains(mode.Engine, "external") {
		t.Fatalf("disabled mode engine = %q, want no external fallback", mode.Engine)
	}
}

func TestQwen35MTPAdmissionRejectsIncompleteNamespace(t *testing.T) {
	man := completeQwen35MTPManifest()
	delete(man, "mtp.fc.weight")
	_, err := qwen35MTPAdmission(qwen35MTPTestConfig(), man, false)
	var incomplete *Qwen35MTPIncompleteError
	if !errors.As(err, &incomplete) {
		t.Fatalf("error = %v, want *Qwen35MTPIncompleteError", err)
	}
	if len(incomplete.Missing) != 1 || incomplete.Missing[0] != "mtp.fc.weight" {
		t.Fatalf("missing = %v, want [mtp.fc.weight]", incomplete.Missing)
	}
}

func TestMaterializeQwen35MTPRespectsRetentionAndCompleteness(t *testing.T) {
	orig := RetainMTP
	defer func() { RetainMTP = orig }()

	cfg := qwen35MTPTestConfig()
	RetainMTP = false
	dropped := completeQwen35MTPManifest()
	if err := materializeQwen35Tensors(cfg, dropped); err != nil {
		t.Fatalf("materialize with retention off: %v", err)
	}
	for name := range dropped {
		if strings.HasPrefix(name, "mtp.") {
			t.Fatalf("retention off preserved %q", name)
		}
	}

	RetainMTP = true
	retained := completeQwen35MTPManifest()
	if err := materializeQwen35Tensors(cfg, retained); err != nil {
		t.Fatalf("materialize complete retained head: %v", err)
	}
	for _, name := range qwen35MTPRequiredTensors {
		if _, ok := retained[name]; !ok {
			t.Fatalf("retention on dropped required tensor %q", name)
		}
	}

	partial := completeQwen35MTPManifest()
	delete(partial, "mtp.norm.weight")
	var incomplete *Qwen35MTPIncompleteError
	if err := materializeQwen35Tensors(cfg, partial); !errors.As(err, &incomplete) {
		t.Fatalf("partial retained head error = %v, want *Qwen35MTPIncompleteError", err)
	}
}

package model

import (
	"errors"
	"fmt"
	"strings"
	"testing"
)

func qwen35MTPLoadConfig() Config {
	return Config{
		Name:                      "Qwen3.8-27B synthetic",
		ModelType:                 "qwen3_5_text",
		LayerTypes:                []string{"linear_attention"},
		HiddenSize:                4,
		HeadDim:                   2,
		NumHeads:                  2,
		NumKVHeads:                1,
		IntermediateSize:          6,
		VocabSize:                 8,
		MTPNumHiddenLayers:        1,
		MTPUseDedicatedEmbeddings: false,
		AttnOutputGate:            true,
	}
}

func qwen35MTPLoadTensors(t *testing.T, cfg Config) ([]NamedTensorF32, map[string]tinySTTensor) {
	t.Helper()
	shapes, err := qwen35MTPExpectedShapes(cfg)
	if err != nil {
		t.Fatalf("derive MTP fixture shapes: %v", err)
	}
	named := make([]NamedTensorF32, 0, len(qwen35MTPRequiredTensors))
	safe := make(map[string]tinySTTensor, len(qwen35MTPRequiredTensors))
	for i, name := range qwen35MTPRequiredTensors {
		shape := append([]int(nil), shapes[name]...)
		n := 1
		for _, dim := range shape {
			n *= dim
		}
		data := make([]float32, n)
		for j := range data {
			data[j] = float32(i+1) + float32(j)/1024
		}
		named = append(named, NamedTensorF32{Name: name, Shape: shape, Data: data})
		safe[name] = tinySTTensor{dtype: "F32", shape: shape, data: f32TestBytes(data)}
	}
	return named, safe
}

func retainMTPForTest(t *testing.T, retain bool) {
	t.Helper()
	orig := RetainMTP
	RetainMTP = retain
	t.Cleanup(func() { RetainMTP = orig })
}

func TestQwen38MTPSafetensorsLoadRetainsValidatedHead(t *testing.T) {
	retainMTPForTest(t, true)
	cfg := qwen35MTPLoadConfig()
	_, safe := qwen35MTPLoadTensors(t, cfg)

	m, err := LoadSafetensors(writeTinySafetensors(t, safe), cfg)
	if err != nil {
		t.Fatalf("load valid Qwen3.8 MTP safetensors fixture: %v", err)
	}
	for i, name := range qwen35MTPRequiredTensors {
		meta, ok := m.manifest[name]
		if !ok {
			t.Fatalf("retained manifest missing %s", name)
		}
		if got := m.tensor(name)[0]; got != float32(i+1) {
			t.Fatalf("retained %s first value = %g, want %d", name, got, i+1)
		}
		if !strings.EqualFold(meta.Dtype, "f32") {
			t.Fatalf("retained %s dtype = %q, want decoded F32", name, meta.Dtype)
		}
	}
	mode, err := m.Qwen35MTPMode(false)
	if err != nil {
		t.Fatalf("MTP mode: %v", err)
	}
	if !mode.Enabled || mode.Engine != "fak-native" {
		t.Fatalf("MTP mode = %+v, want enabled fak-native", mode)
	}
}

func TestQwen38MTPGGUFNormalizedLoadRetainsValidatedHead(t *testing.T) {
	retainMTPForTest(t, true)
	cfg := qwen35MTPLoadConfig()
	// GGUF uses nextn_predict_layers rather than the HF mtp_num_hidden_layers key.
	cfg.MTPNumHiddenLayers = 0
	cfg.NumNextNPredictLayers = 1
	cfg.Name = "Qwen3.8-27B-Q4_K_M synthetic GGUF"
	named, _ := qwen35MTPLoadTensors(t, cfg)
	named = append(named, NamedTensorF32{
		Name:  "model.embed_tokens.weight",
		Shape: []int{cfg.VocabSize, cfg.HiddenSize},
		Data:  make([]float32, cfg.VocabSize*cfg.HiddenSize),
	})

	m, err := NewFromF32Tensors(cfg, named)
	if err != nil {
		t.Fatalf("load valid GGUF-normalized Qwen3.8 MTP fixture: %v", err)
	}
	for i, name := range qwen35MTPRequiredTensors {
		if got := m.tensor(name)[0]; got != float32(i+1) {
			t.Fatalf("retained %s first value = %g, want %d", name, got, i+1)
		}
	}
	forward, err := m.NewQwen35MTPForward()
	if err != nil {
		t.Fatalf("bind retained GGUF-normalized head to native forward: %v", err)
	}
	forward.Close()
}

func TestQwen38MTPLoadRejectsIncompleteSafetensors(t *testing.T) {
	retainMTPForTest(t, true)
	cfg := qwen35MTPLoadConfig()
	_, safe := qwen35MTPLoadTensors(t, cfg)
	delete(safe, "mtp.fc.weight")

	_, err := LoadSafetensors(writeTinySafetensors(t, safe), cfg)
	var incomplete *Qwen35MTPIncompleteError
	if !errors.As(err, &incomplete) {
		t.Fatalf("error = %v, want *Qwen35MTPIncompleteError", err)
	}
	if fmt.Sprint(incomplete.Missing) != "[mtp.fc.weight]" || !strings.Contains(err.Error(), "ordinary fak-native target decode") {
		t.Fatalf("incomplete error = %v, want missing tensor and native downgrade action", err)
	}
}

func TestQwen38MTPLoadRejectsIncompatibleShapeAcrossLoaderSeams(t *testing.T) {
	retainMTPForTest(t, true)
	cfg := qwen35MTPLoadConfig()
	named, safe := qwen35MTPLoadTensors(t, cfg)

	badShape := []int{8, 4}
	for i := range named {
		if named[i].Name == "mtp.fc.weight" {
			named[i].Shape = badShape
			break
		}
	}
	entry := safe["mtp.fc.weight"]
	entry.shape = badShape
	safe["mtp.fc.weight"] = entry

	loads := []struct {
		name string
		load func() error
	}{
		{name: "safetensors", load: func() error {
			_, err := LoadSafetensors(writeTinySafetensors(t, safe), cfg)
			return err
		}},
		{name: "gguf-normalized", load: func() error {
			_, err := NewFromF32Tensors(cfg, named)
			return err
		}},
	}
	for _, tc := range loads {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.load()
			var artifact *Qwen35MTPArtifactError
			if !errors.As(err, &artifact) {
				t.Fatalf("error = %v, want *Qwen35MTPArtifactError", err)
			}
			if artifact.Kind != "incompatible-shape" || artifact.Tensor != "mtp.fc.weight" || artifact.Want != "[4 8]" || artifact.Got != "[8 4]" {
				t.Fatalf("artifact error = %+v, want actionable fc shape refusal", artifact)
			}
		})
	}
}

func TestQwen38MTPLoadRejectsMalformedOrIncompatibleMetadata(t *testing.T) {
	retainMTPForTest(t, true)
	base := qwen35MTPLoadConfig()
	named, _ := qwen35MTPLoadTensors(t, base)

	cases := []struct {
		name      string
		mutate    func(*Config)
		wantKind  string
		wantField string
	}{
		{name: "missing depth", mutate: func(c *Config) { c.MTPNumHiddenLayers = 0 }, wantKind: "incompatible-metadata", wantField: "MTP depth"},
		{name: "negative depth", mutate: func(c *Config) { c.MTPNumHiddenLayers = -1 }, wantKind: "malformed-metadata", wantField: "MTP depth"},
		{name: "conflicting depth keys", mutate: func(c *Config) { c.NumNextNPredictLayers = 2 }, wantKind: "malformed-metadata", wantField: "MTP depth"},
		{name: "unsupported depth", mutate: func(c *Config) { c.MTPNumHiddenLayers = 2 }, wantKind: "incompatible-metadata", wantField: "MTP depth"},
		{name: "dedicated embeddings", mutate: func(c *Config) { c.MTPUseDedicatedEmbeddings = true }, wantKind: "incompatible-metadata", wantField: "mtp_use_dedicated_embeddings"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := base
			tc.mutate(&cfg)
			_, err := NewFromF32Tensors(cfg, named)
			var artifact *Qwen35MTPArtifactError
			if !errors.As(err, &artifact) {
				t.Fatalf("error = %v, want *Qwen35MTPArtifactError", err)
			}
			if artifact.Kind != tc.wantKind || artifact.Field != tc.wantField || !strings.Contains(err.Error(), "ordinary fak-native target decode") {
				t.Fatalf("artifact error = %+v (%v), want kind=%q field=%q and native downgrade action", artifact, err, tc.wantKind, tc.wantField)
			}
		})
	}
}

func TestQwen38MTPLoadRejectsUnknownRetainedLayout(t *testing.T) {
	retainMTPForTest(t, true)
	cfg := qwen35MTPLoadConfig()
	named, _ := qwen35MTPLoadTensors(t, cfg)
	named = append(named, NamedTensorF32{Name: "mtp.layers.1.norm.weight", Shape: []int{1}, Data: []float32{1}})

	_, err := NewFromF32Tensors(cfg, named)
	var artifact *Qwen35MTPArtifactError
	if !errors.As(err, &artifact) || artifact.Kind != "incompatible-layout" || artifact.Got != "mtp.layers.1.norm.weight" {
		t.Fatalf("error = %v (%+v), want typed unknown-layout refusal", err, artifact)
	}
}

func TestQwen38MTPExplicitDowngradeKeepsOrdinaryNativeTargetLoad(t *testing.T) {
	retainMTPForTest(t, false)
	cfg := qwen35MTPLoadConfig()
	_, safe := qwen35MTPLoadTensors(t, cfg)
	entry := safe["mtp.fc.weight"]
	entry.shape = []int{8, 4}
	safe["mtp.fc.weight"] = entry

	m, err := LoadSafetensors(writeTinySafetensors(t, safe), cfg)
	if err != nil {
		t.Fatalf("ordinary target load with explicit MTP retention off: %v", err)
	}
	mode, err := m.Qwen35MTPMode(false)
	if err != nil {
		t.Fatalf("ordinary target mode: %v", err)
	}
	if mode.Enabled || mode.Engine != "fak-native" || mode.Reason != "mtp-tensors-not-retained" {
		t.Fatalf("mode = %+v, want explicit ordinary fak-native target downgrade", mode)
	}
	if strings.Contains(strings.ToLower(mode.Engine), "llama") || strings.Contains(strings.ToLower(mode.Engine), "external") {
		t.Fatalf("downgrade engine = %q, external fallback is forbidden", mode.Engine)
	}
}

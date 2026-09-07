package model

import (
	"errors"
	"fmt"
	"strings"
	"testing"
)

func generateSyntheticGLM5NextManifest() map[string]tensorMeta {
	m := make(map[string]tensorMeta)
	m["model.embed_tokens.weight"] = tensorMeta{Shape: []int{151552, 4096}}
	m["model.norm.weight"] = tensorMeta{Shape: []int{4096}}

	for l := 0; l < 45; l++ {
		prefix := fmt.Sprintf("model.layers.%d", l)
		m[prefix+".input_layernorm.weight"] = tensorMeta{Shape: []int{4096}}

		if l%4 != 3 {
			m[prefix+".self_attn.linear_attn.in_proj.weight"] = tensorMeta{Shape: []int{8192, 4096}}
		} else {
			m[prefix+".self_attn.q_proj.weight"] = tensorMeta{Shape: []int{256, 4096}}
		}

		if l < 3 {
			m[prefix+".mlp.gate_proj.weight"] = tensorMeta{Shape: []int{12288, 4096}}
		} else {
			m[prefix+".mlp.router.weight"] = tensorMeta{Shape: []int{288, 4096}}
		}
	}
	return m
}

func generateUpstreamPrefixGLM5NextManifest() map[string]tensorMeta {
	m := make(map[string]tensorMeta)
	m["model.language_model.embed_tokens.weight"] = tensorMeta{Shape: []int{154880, 4096}}
	m["model.language_model.norm.weight"] = tensorMeta{Shape: []int{4096}}

	for l := 0; l < 45; l++ {
		prefix := fmt.Sprintf("model.language_model.layers.%d", l)
		m[prefix+".input_layernorm.weight"] = tensorMeta{Shape: []int{4096}}

		if l%4 != 3 {
			m[prefix+".self_attn.linear_attn.in_proj.weight"] = tensorMeta{Shape: []int{8192, 4096}}
		} else {
			m[prefix+".self_attn.q_proj.weight"] = tensorMeta{Shape: []int{256, 4096}}
		}

		if l < 3 {
			m[prefix+".mlp.gate_proj.weight"] = tensorMeta{Shape: []int{12288, 4096}}
		} else {
			m[prefix+".mlp.router.weight"] = tensorMeta{Shape: []int{288, 4096}}
		}
	}
	return m
}

func generateUpstreamLayoutGLM5NextManifest(prefix string) map[string]tensorMeta {
	if prefix == "" {
		prefix = "model."
	}
	m := make(map[string]tensorMeta)
	m[prefix+"embed_tokens.weight"] = tensorMeta{Shape: []int{154880, 4096}}
	m[prefix+"norm.weight"] = tensorMeta{Shape: []int{4096}}

	for l := 0; l < 45; l++ {
		layerPrefix := fmt.Sprintf("%slayers.%d", prefix, l)
		m[layerPrefix+".input_layernorm.weight"] = tensorMeta{Shape: []int{4096}}

		if l%4 != 3 {
			m[layerPrefix+".self_attn.q_proj.weight"] = tensorMeta{Shape: []int{8192, 4096}}
			m[layerPrefix+".self_attn.k_conv1d.weight"] = tensorMeta{Shape: []int{8192, 1, 4}}
		} else {
			m[layerPrefix+".self_attn.q_proj.weight"] = tensorMeta{Shape: []int{256, 4096}}
		}

		if l < 3 {
			m[layerPrefix+".mlp.gate_proj.weight"] = tensorMeta{Shape: []int{12288, 4096}}
		} else {
			m[layerPrefix+".mlp.experts.0.down_proj.weight"] = tensorMeta{Shape: []int{4096, 2048}}
		}
	}
	return m
}

func TestValidateGLM5NextManifest(t *testing.T) {
	valid := generateSyntheticGLM5NextManifest()
	if err := ValidateGLM5NextManifest(valid); err != nil {
		t.Fatalf("expected valid manifest to pass, got error: %v", err)
	}

	t.Run("missing embed_tokens", func(t *testing.T) {
		m := generateSyntheticGLM5NextManifest()
		delete(m, "model.embed_tokens.weight")
		err := ValidateGLM5NextManifest(m)
		if err == nil || !strings.Contains(err.Error(), "model.embed_tokens.weight") {
			t.Fatalf("expected missing embed error, got: %v", err)
		}
	})

	t.Run("missing KDA tensor on layer 0", func(t *testing.T) {
		m := generateSyntheticGLM5NextManifest()
		delete(m, "model.layers.0.self_attn.linear_attn.in_proj.weight")
		err := ValidateGLM5NextManifest(m)
		if err == nil || !strings.Contains(err.Error(), "KDA layer 0") {
			t.Fatalf("expected KDA layer 0 error, got: %v", err)
		}
	})

	t.Run("missing DSA tensor on layer 3", func(t *testing.T) {
		m := generateSyntheticGLM5NextManifest()
		delete(m, "model.layers.3.self_attn.q_proj.weight")
		err := ValidateGLM5NextManifest(m)
		if err == nil || !strings.Contains(err.Error(), "DSA layer 3") {
			t.Fatalf("expected DSA layer 3 error, got: %v", err)
		}
	})

	t.Run("missing dense MLP on layer 2", func(t *testing.T) {
		m := generateSyntheticGLM5NextManifest()
		delete(m, "model.layers.2.mlp.gate_proj.weight")
		err := ValidateGLM5NextManifest(m)
		if err == nil || !strings.Contains(err.Error(), "dense MLP layer 2") {
			t.Fatalf("expected dense MLP layer 2 error, got: %v", err)
		}
	})

	t.Run("missing sparse MoE on layer 3", func(t *testing.T) {
		m := generateSyntheticGLM5NextManifest()
		delete(m, "model.layers.3.mlp.router.weight")
		err := ValidateGLM5NextManifest(m)
		if err == nil || !strings.Contains(err.Error(), "sparse MoE layer 3") {
			t.Fatalf("expected sparse MoE layer 3 error, got: %v", err)
		}
	})
}

func TestValidateGLM5NextManifestUpstreamPrefix(t *testing.T) {
	manifest := generateUpstreamPrefixGLM5NextManifest()
	if err := ValidateGLM5NextManifest(manifest); err != nil {
		t.Fatalf("expected upstream prefix manifest to pass, got: %v", err)
	}
}

func TestValidateGLM5NextManifestUpstreamLayout(t *testing.T) {
	t.Run("canonical prefix with upstream layout", func(t *testing.T) {
		manifest := generateUpstreamLayoutGLM5NextManifest("model.")
		if err := ValidateGLM5NextManifest(manifest); err != nil {
			t.Fatalf("expected upstream layout manifest to pass, got: %v", err)
		}
	})

	t.Run("upstream prefix with upstream layout", func(t *testing.T) {
		manifest := generateUpstreamLayoutGLM5NextManifest("model.language_model.")
		if err := ValidateGLM5NextManifest(manifest); err != nil {
			t.Fatalf("expected upstream layout + prefix manifest to pass, got: %v", err)
		}
	})

	t.Run("upstream KDA missing k_conv1d", func(t *testing.T) {
		manifest := generateUpstreamLayoutGLM5NextManifest("model.")
		delete(manifest, "model.layers.0.self_attn.k_conv1d.weight")
		err := ValidateGLM5NextManifest(manifest)
		if err == nil || !strings.Contains(err.Error(), "KDA layer 0") {
			t.Fatalf("expected KDA layer 0 error, got: %v", err)
		}
	})

	t.Run("upstream MoE missing experts.0.down_proj", func(t *testing.T) {
		manifest := generateUpstreamLayoutGLM5NextManifest("model.")
		delete(manifest, "model.layers.3.mlp.experts.0.down_proj.weight")
		err := ValidateGLM5NextManifest(manifest)
		if err == nil || !strings.Contains(err.Error(), "sparse MoE layer 3") {
			t.Fatalf("expected sparse MoE layer 3 error, got: %v", err)
		}
	})
}

func TestClassifyForwardPathGLM5Next(t *testing.T) {
	cfg := Config{GLM5Next: true}

	t.Run("nil manifest returns unsupported error", func(t *testing.T) {
		path, err := ClassifyForwardPath(cfg, nil)
		if path != "" {
			t.Fatalf("expected empty path, got %q", path)
		}
		var unsupported *GLM5NextUnsupportedError
		if !errors.As(err, &unsupported) {
			t.Fatalf("expected *GLM5NextUnsupportedError, got %T: %v", err, err)
		}
	})

	t.Run("valid canonical manifest returns ForwardGLM5Next", func(t *testing.T) {
		manifest := generateSyntheticGLM5NextManifest()
		path, err := ClassifyForwardPath(cfg, manifest)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if path != ForwardGLM5Next {
			t.Fatalf("expected path %q, got %q", ForwardGLM5Next, path)
		}
	})

	t.Run("valid upstream manifest returns ForwardGLM5Next", func(t *testing.T) {
		manifest := generateUpstreamLayoutGLM5NextManifest("model.language_model.")
		path, err := ClassifyForwardPath(cfg, manifest)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if path != ForwardGLM5Next {
			t.Fatalf("expected path %q, got %q", ForwardGLM5Next, path)
		}
	})

	t.Run("invalid manifest returns validation error", func(t *testing.T) {
		manifest := generateSyntheticGLM5NextManifest()
		delete(manifest, "model.embed_tokens.weight")
		path, err := ClassifyForwardPath(cfg, manifest)
		if path != "" {
			t.Fatalf("expected empty path on error, got %q", path)
		}
		if err == nil || !strings.Contains(err.Error(), "embed_tokens.weight") {
			t.Fatalf("expected embed_tokens error, got: %v", err)
		}
	})
}

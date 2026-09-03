package model

import (
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

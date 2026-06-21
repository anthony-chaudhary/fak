package ggufload

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func TestComputeSourceServesDequantizedF32Weights(t *testing.T) {
	path := filepath.Join(t.TempDir(), "canonical.gguf")
	if err := os.WriteFile(path, tinyCanonicalModelGGUF(t), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	ws, err := OpenWeights(path)
	if err != nil {
		t.Fatalf("OpenWeights: %v", err)
	}
	defer ws.Close()

	be := compute.Default()
	if be == nil {
		t.Fatal("no default compute backend registered")
	}
	src, err := NewComputeSource(be, ws)
	if err != nil {
		t.Fatalf("NewComputeSource: %v", err)
	}
	if got := src.Config(); got.HiddenSize != 2 || got.VocabSize != 3 || got.ModelType != "qwen2" {
		t.Fatalf("bad config from adapter: %#v", got)
	}

	cases := []struct {
		name  string
		shape []int
		data  []float32
	}{
		{"model.embed_tokens.weight", []int{3, 2}, []float32{1, 2, 3, 4, 5, 6}},
		{"model.norm.weight", []int{2}, []float32{7, 8}},
		{"lm_head.weight", []int{3, 2}, []float32{9, 10, 11, 12, 13, 14}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tensor, err := src.Weight(tc.name, compute.F32)
			if err != nil {
				t.Fatalf("Weight(%s): %v", tc.name, err)
			}
			if tensor.Dtype != compute.F32 {
				t.Fatalf("%s dtype=%v, want f32", tc.name, tensor.Dtype)
			}
			if len(tensor.Shape) != len(tc.shape) {
				t.Fatalf("%s shape=%v, want %v", tc.name, tensor.Shape, tc.shape)
			}
			for i := range tc.shape {
				if tensor.Shape[i] != tc.shape[i] {
					t.Fatalf("%s shape=%v, want %v", tc.name, tensor.Shape, tc.shape)
				}
			}
			host, ok := be.Host(tensor)
			if !ok {
				t.Fatalf("%s not host-addressable", tc.name)
			}
			if len(host) != len(tc.data) {
				t.Fatalf("%s len=%d, want %d", tc.name, len(host), len(tc.data))
			}
			for i := range tc.data {
				if math.Float32bits(host[i]) != math.Float32bits(tc.data[i]) {
					t.Fatalf("%s[%d]=%v, want %v", tc.name, i, host[i], tc.data[i])
				}
			}
		})
	}
}

func TestComputeSourceUnpermutesRotaryQKWeights(t *testing.T) {
	path := filepath.Join(t.TempDir(), "qk.gguf")
	qHF := sequenceF32ForTest(0, 16)
	kHF := sequenceF32ForTest(100, 16)
	if err := os.WriteFile(path, tinyRotaryQKGGUF(t, qHF, kHF), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	ws, err := OpenWeights(path)
	if err != nil {
		t.Fatalf("OpenWeights: %v", err)
	}
	defer ws.Close()

	be := compute.Default()
	src, err := NewComputeSource(be, ws)
	if err != nil {
		t.Fatalf("NewComputeSource: %v", err)
	}

	cases := []struct {
		name string
		want []float32
	}{
		{"model.layers.0.self_attn.q_proj.weight", qHF},
		{"model.layers.0.self_attn.k_proj.weight", kHF},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tensor, err := src.Weight(tc.name, compute.F32)
			if err != nil {
				t.Fatalf("Weight(%s): %v", tc.name, err)
			}
			host, ok := be.Host(tensor)
			if !ok {
				t.Fatalf("%s not host-addressable", tc.name)
			}
			if len(host) != len(tc.want) {
				t.Fatalf("%s len=%d, want %d", tc.name, len(host), len(tc.want))
			}
			for i := range tc.want {
				if math.Float32bits(host[i]) != math.Float32bits(tc.want[i]) {
					t.Fatalf("%s[%d]=%v, want %v (unpermuted HF order)", tc.name, i, host[i], tc.want[i])
				}
			}
		})
	}
}

func TestComputeSourceRejectsMissAndNonF32(t *testing.T) {
	path := filepath.Join(t.TempDir(), "canonical.gguf")
	if err := os.WriteFile(path, tinyCanonicalModelGGUF(t), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	ws, err := OpenWeights(path)
	if err != nil {
		t.Fatalf("OpenWeights: %v", err)
	}
	defer ws.Close()
	src, err := NewComputeSource(compute.Default(), ws)
	if err != nil {
		t.Fatalf("NewComputeSource: %v", err)
	}

	if _, err := src.Weight("model.embed_tokens.weight", compute.Q8_0); err == nil {
		t.Fatal("Weight accepted a non-f32 request on the dequant-default seam")
	}
	if _, err := src.Weight("model.layers.0.does_not_exist.weight", compute.F32); err == nil {
		t.Fatal("Weight accepted a missing weight name")
	}
}

func TestNewComputeSourceValidatesArgs(t *testing.T) {
	path := filepath.Join(t.TempDir(), "canonical.gguf")
	if err := os.WriteFile(path, tinyCanonicalModelGGUF(t), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	ws, err := OpenWeights(path)
	if err != nil {
		t.Fatalf("OpenWeights: %v", err)
	}
	defer ws.Close()

	if _, err := NewComputeSource(nil, ws); err == nil {
		t.Fatal("NewComputeSource accepted a nil backend")
	}
	if _, err := NewComputeSource(compute.Default(), nil); err == nil {
		t.Fatal("NewComputeSource accepted a nil weight source")
	}
}

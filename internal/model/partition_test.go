package model

import (
	"fmt"
	"testing"
)

// partition_test.go — pipeline-parallel layer partitioning for the lean quant
// loader. A 753B GLM-5.2 checkpoint cannot fit one node's host RAM even at Q8
// (~830 GB); a pipeline-parallel worker must load only ITS band of layers. The
// LayerWindow load option is the seam: WithLayerWindow(lo, hi) keeps only the
// resident matmul weights whose `model.layers.N.*` index is in [lo, hi). It
// defaults to all layers, so every existing load is bit-identical (no-op).
//
// This is the foundational increment toward native multi-GPU GLM-5.2 serving
// (GLM-5.2-NATIVE-ENGINE-GAP gaps #1 device/parallelism and #2 bounded load):
// it bounds resident memory per worker and gives each worker a disjoint layer
// band, without touching the forward path.

// TestLayerWindowFullIsNoOp proves a full-coverage window loads the identical
// q8 store as the plain loader — the Llama-style no-op gate for this option, so
// adding it cannot perturb any existing checkpoint's residency or numerics.
func TestLayerWindowFullIsNoOp(t *testing.T) {
	dir, cfg := writeTinyGLMDsaShardedSafetensorsDir(t, "BF16", false, true, true, true)

	plain, err := LoadSafetensorsQuantDir(dir, cfg)
	if err != nil {
		t.Fatalf("LoadSafetensorsQuantDir (plain): %v", err)
	}
	windowed, err := LoadSafetensorsQuantDir(dir, cfg, WithLayerWindow(0, cfg.NumLayers))
	if err != nil {
		t.Fatalf("LoadSafetensorsQuantDir (full window): %v", err)
	}

	if len(plain.q8w) != len(windowed.q8w) {
		t.Fatalf("full window q8 count = %d, plain = %d; must be identical (no-op)",
			len(windowed.q8w), len(plain.q8w))
	}
	for name, want := range plain.q8w {
		got := windowed.q8w[name]
		if got == nil {
			t.Fatalf("full window dropped q8 tensor %s; window [0,%d) must keep all", name, cfg.NumLayers)
		}
		assertQ8TensorEqualGLM(t, name, got, want)
	}
}

// TestLayerWindowKeepsOnlyBandLayers proves a partial window [1,2) keeps only
// layer-1 resident matmul weights and drops layer-0 entirely — what a
// pipeline-parallel worker for the second band must hold.
func TestLayerWindowKeepsOnlyBandLayers(t *testing.T) {
	dir, cfg := writeTinyGLMDsaShardedSafetensorsDir(t, "BF16", false, true, true, true)
	if cfg.NumLayers != 2 {
		t.Fatalf("fixture NumLayers = %d, test assumes 2", cfg.NumLayers)
	}

	band, err := LoadSafetensorsQuantDir(dir, cfg, WithLayerWindow(1, 2))
	if err != nil {
		t.Fatalf("LoadSafetensorsQuantDir (band [1,2)): %v", err)
	}

	l0 := fmt.Sprintf("model.layers.%d.", 0)
	l1 := fmt.Sprintf("model.layers.%d.", 1)
	var keptL1, keptL0 int
	for name := range band.q8w {
		switch {
		case len(name) >= len(l0) && name[:len(l0)] == l0:
			keptL0++
		case len(name) >= len(l1) && name[:len(l1)] == l1:
			keptL1++
		}
	}
	if keptL0 != 0 {
		t.Fatalf("band [1,2) kept %d layer-0 q8 tensors; want 0 (out of band)", keptL0)
	}
	if keptL1 == 0 {
		t.Fatalf("band [1,2) kept no layer-1 q8 tensors; the worker has nothing to run")
	}
	// A non-layer weight outside any band (untied lm_head) is layer-agnostic and
	// must still load — a pipeline worker that owns the head needs it.
	if _, ok := band.q8w["lm_head.weight"]; !ok {
		t.Fatalf("band load dropped lm_head.weight; layer-agnostic weights must survive the window")
	}
}

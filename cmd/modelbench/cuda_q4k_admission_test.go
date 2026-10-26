package main

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// fak-test:runtime fast est=1ms lane=default
func TestQ4KAdmitsCUDABackend(t *testing.T) {
	tests := []struct {
		name  string
		edit  func(*benchFlags)
		valid bool
	}{
		{name: "cuda", valid: true},
		{name: "legacy", edit: func(f *benchFlags) { *f.backendName = "legacy" }, valid: true},
		{name: "vulkan", edit: func(f *benchFlags) { *f.backendName = "vulkan" }, valid: true},
		{name: "unknown backend", edit: func(f *benchFlags) { *f.backendName = "unknown" }},
		{name: "cpu reference", edit: func(f *benchFlags) { *f.backendName = "cpu-ref" }},
		{name: "missing gguf", edit: func(f *benchFlags) { *f.gguf = "" }},
		{name: "hf", edit: func(f *benchFlags) { *f.hf = "fixture-hf" }},
		{name: "lean", edit: func(f *benchFlags) { *f.lean = true }},
		{name: "verify", edit: func(f *benchFlags) { *f.verify = true }},
		{name: "legacy metal", edit: func(f *benchFlags) { *f.metal = true }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := testCompleteBenchFlags()
			*f.gguf, *f.q4k, *f.backendName = "fixture.gguf", true, "cuda"
			if tt.edit != nil {
				tt.edit(f)
			}
			if err := validateFlagCombinations(f); (err == nil) != tt.valid {
				t.Fatalf("valid=%v, want %v; error=%v", err == nil, tt.valid, err)
			}
		})
	}
}

// fak-test:runtime fast est=200ms lane=default
func TestQ4KCUDAUsesPackedHALPath(t *testing.T) {
	t.Setenv("FAK_PAGED_KV", "0")
	t.Setenv("FAK_GGUF_LOAD_WORKERS", "1")
	f := testCompleteBenchFlags()
	*f.gguf, *f.q4k, *f.backendName = benchMixedQuantGGUF(t), true, "cuda"
	if err := validateFlagCombinations(f); err != nil {
		t.Fatal(err)
	}
	m, _, err := loadModel(f, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer m.CloseWeights()
	if !m.HasQ4K("model.layers.0.mlp.up_proj.weight") || m.HasQ8("model.layers.0.mlp.up_proj.weight") {
		t.Fatal("CUDA load did not retain packed Q4_K weights")
	}
	if m.HasKQuant("lm_head.weight") || !m.HasQ8("lm_head.weight") {
		t.Fatal("CUDA load did not convert unsupported dense Q2_K to Q8")
	}
	be := &benchQuantBackend{benchObservedBackend: benchObservedBackend{Backend: compute.Default()}}
	s := newBenchSession(m, f, be)
	defer s.Close()
	if !s.Quant || !s.Q4K {
		t.Fatal("CUDA Q4_K flags did not reach the HAL session")
	}
	prefill := append([]float32(nil), s.Prefill([]int{0, 1})...)
	decode := s.Step(2)
	if len(prefill) != 4 || len(decode) != 4 || !allFinite(prefill) || !allFinite(decode) || be.matmuls == 0 || len(be.q4Payload) == 0 || len(be.q2Payload) != 0 {
		t.Fatal("CUDA-selected load did not execute packed Q4_K through the supplied HAL backend")
	}
}

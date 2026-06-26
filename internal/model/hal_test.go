package model

import (
	"math"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type uploadClassRecordingBackend struct {
	compute.Backend
	classes []compute.MemoryClass
	sites   []string
}

func (r *uploadClassRecordingBackend) UploadClass(t compute.Tensor, as compute.Dtype, class compute.MemoryClass, site string) compute.Tensor {
	r.classes = append(r.classes, class)
	r.sites = append(r.sites, site)
	return r.Backend.Upload(t, as)
}

func (r *uploadClassRecordingBackend) DSASparseAttend(q, selK, selV compute.Tensor, nSel, nH, qkHead, vHead int, scale float32) compute.Tensor {
	return r.Backend.(compute.DSASparseBackend).DSASparseAttend(q, selK, selV, nSel, nH, qkHead, vHead, scale)
}

func (r *uploadClassRecordingBackend) DSAIndexSelect(indexQ, indexK, weights compute.Tensor, nKeys, nH, indexDim, queryPos, topK int, scale float32) []int {
	return r.Backend.(compute.DSAIndexBackend).DSAIndexSelect(indexQ, indexK, weights, nKeys, nH, indexDim, queryPos, topK, scale)
}

func TestHALSessionMatchesLegacyCPUReference(t *testing.T) {
	cfg := Config{
		HiddenSize:        16,
		NumLayers:         2,
		NumHeads:          4,
		NumKVHeads:        2,
		HeadDim:           4,
		IntermediateSize:  32,
		VocabSize:         64,
		RMSNormEps:        1e-5,
		RopeTheta:         10000,
		TieWordEmbeddings: true,
		EOSTokenID:        63,
	}
	m := NewSynthetic(cfg)
	be := compute.Pick("cpu-ref")
	if !compute.RequireReference(be) {
		t.Fatal("cpu-ref must be a reference backend")
	}

	prompt := []int{3, 7, 11, 19, 23}
	legacy := m.NewSession()
	hal := m.NewBackendSession(be)
	assertSameF32(t, "prefill", legacy.Prefill(prompt), hal.Prefill(prompt))

	for i, id := range []int{5, 13, 21} {
		assertSameF32(t, "step", legacy.Step(id), hal.Step(id))
		if legacy.Cache.Len() != len(prompt)+i+1 {
			t.Fatalf("legacy cache length drifted at step %d", i)
		}
		if hal.halKV.Len() != len(prompt)+i+1 {
			t.Fatalf("HAL KV length drifted at step %d: got %d", i, hal.halKV.Len())
		}
	}

	want := m.NewSession().Generate(prompt, 4)
	got := m.NewBackendSession(be).Generate(prompt, 4)
	if len(got) != len(want) {
		t.Fatalf("Generate length mismatch: got %d want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("Generate[%d]=%d want %d", i, got[i], want[i])
		}
	}
}

func TestHALHostUploadsCarryMemoryClass(t *testing.T) {
	cfg := Config{
		HiddenSize:        8,
		NumLayers:         1,
		NumHeads:          2,
		NumKVHeads:        1,
		HeadDim:           4,
		IntermediateSize:  16,
		VocabSize:         16,
		RMSNormEps:        1e-5,
		RopeTheta:         10000,
		TieWordEmbeddings: true,
	}
	m := NewSynthetic(cfg)
	be := &uploadClassRecordingBackend{Backend: compute.Default()}
	s := &Session{M: m, Backend: be, halW: map[string]compute.Tensor{}}

	_ = s.weightHAL("model.norm.weight")
	_ = s.uploadHostF32([]int{cfg.HiddenSize}, make([]float32, cfg.HiddenSize), compute.MemoryActivation, "hal-token-input")

	if len(be.classes) != 2 {
		t.Fatalf("classed upload calls = %d, want 2 (%+v)", len(be.classes), be.classes)
	}
	if be.classes[0] != compute.MemoryWeights || !strings.Contains(be.sites[0], "hal-weight model.norm.weight") {
		t.Fatalf("weight upload recorded class/site = %s/%q, want weights/hal-weight", be.classes[0], be.sites[0])
	}
	if be.classes[1] != compute.MemoryActivation || be.sites[1] != "hal-token-input" {
		t.Fatalf("activation upload recorded class/site = %s/%q, want activation/hal-token-input", be.classes[1], be.sites[1])
	}
}

func TestBackendKernelRuntimeUploadsCarryActivationClass(t *testing.T) {
	cfg := Config{
		HiddenSize:        8,
		NumLayers:         1,
		NumHeads:          2,
		NumKVHeads:        1,
		HeadDim:           4,
		IntermediateSize:  16,
		VocabSize:         16,
		RMSNormEps:        1e-5,
		RopeTheta:         10000,
		TieWordEmbeddings: true,
	}
	m := NewSynthetic(cfg)
	be := &uploadClassRecordingBackend{Backend: compute.Default()}
	s := &Session{M: m, Backend: be, halW: map[string]compute.Tensor{}}
	k := backendKernel{s: s}

	_ = k.mul(layerName(0, "self_attn.q_proj.weight"), make([]float32, cfg.HiddenSize), cfg.NumHeads*cfg.HeadDim, cfg.HiddenSize)
	if _, ok := k.sparseAttend(
		make([]float32, cfg.NumHeads*cfg.HeadDim),
		make([]float32, 2*cfg.NumHeads*cfg.HeadDim),
		make([]float32, 2*cfg.NumHeads*cfg.HeadDim),
		2, cfg.NumHeads, cfg.HeadDim, cfg.HeadDim, 1,
	); !ok {
		t.Fatal("cpu-ref backend should support DSA sparse attention")
	}
	if _, ok := k.indexSelect(
		make([]float32, cfg.NumHeads*3),
		make([]float32, 2*3),
		make([]float32, cfg.NumHeads),
		2, cfg.NumHeads, 3, 0, 1, 1,
	); !ok {
		t.Fatal("cpu-ref backend should support DSA index select")
	}
	_ = s.glmDsaHead(make([]float32, cfg.HiddenSize))

	for _, want := range []string{
		"glm-dsa-activation " + layerName(0, "self_attn.q_proj.weight"),
		"glm-dsa-sparse-query",
		"glm-dsa-sparse-selected-k",
		"glm-dsa-sparse-selected-v",
		"glm-dsa-index-query",
		"glm-dsa-index-keys",
		"glm-dsa-index-weights",
		"glm-dsa-lm-head-activation",
	} {
		if !recordedUploadClass(be, compute.MemoryActivation, want) {
			t.Fatalf("missing activation upload site %q in classes=%v sites=%v", want, be.classes, be.sites)
		}
	}
}

func recordedUploadClass(be *uploadClassRecordingBackend, class compute.MemoryClass, site string) bool {
	for i := range be.classes {
		if be.classes[i] == class && be.sites[i] == site {
			return true
		}
	}
	return false
}

func assertSameF32(t *testing.T, label string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s length mismatch: got %d want %d", label, len(got), len(want))
	}
	for i := range want {
		// Byte-exact on amd64 (fmaCrossPathTol==0); within FMA noise on arches where gc
		// auto-fuses multiply-add — the legacy session and the HAL backend are two distinct Go
		// code paths over the same f32 math, so on arm64 they diverge sub-ULP (~1e-6), exactly
		// like the reposition/profiler rungs (see fmatol_other_test.go). The token-level
		// Generate check below stays exact, so a real divergence is still caught.
		d := got[i] - want[i]
		if d < 0 {
			d = -d
		}
		if float64(d) > fmaCrossPathTol {
			t.Fatalf("%s[%d] drift: got %v want %v (|Δ|=%.3e > %.0e, bits %x vs %x)",
				label, i, got[i], want[i], d, fmaCrossPathTol, math.Float32bits(got[i]), math.Float32bits(want[i]))
		}
	}
}

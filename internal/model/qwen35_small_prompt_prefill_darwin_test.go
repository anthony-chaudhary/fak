//go:build darwin && arm64 && cgo

package model

import (
	"strconv"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

// TestQwen35MetalResidentSmallPromptPrefillMatchesTokenLoop is the GPU half of the #13694
// small-P gate: on the resident-Q4_K Metal session a fresh 2..15-token prompt now takes the
// batched hybrid prefill (one batched dispatch per projection group, no token loop) and must
// match the CPU per-token decode loop's greedy token, logits (cos>=0.999, the
// TestMetalQ4KPrefillMatchesCPU bar) and cache length. The observed Q4_K route must name a
// Metal GEMM, not the token loop. The CPU reference uses its own model because the GPU
// session's single-residency upload may free the model's raw q4_k bytes.
// fak-test:runtime medium est=20s lane=default
func TestQwen35MetalResidentSmallPromptPrefillMatchesTokenLoop(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("no Metal device available")
	}
	defer metalgemm.ResetQ4K()
	setQ4KSDOTForTest(false)
	t.Cleanup(func() { setQ4KSDOTForTest(true) })
	cfg := qwen35HybridQ4KTestCfg()
	build := func() *Model {
		m := NewSynthetic(cfg)
		m.Quantize()
		fillQ4KMajority(t, m, cfg)
		return m
	}
	cpuModel, gpuModel := build(), build()
	base := []int{3, 7, 11, 5, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59}
	for _, P := range []int{2, 3, 9, 15} {
		prompt := base[:P]
		label := "metal resident small-P prefill P=" + strconv.Itoa(P)

		ref := cpuModel.NewSession()
		ref.Q4K = true
		var refHidden []float32
		for _, id := range prompt {
			refHidden = ref.tokenHiddenQ(id, ref.Cache.Len())
		}
		want := ref.headResident(refHidden)

		gpu := gpuModel.NewSession()
		gpu.Q4K, gpu.MetalQ4K = true, true
		if !gpu.qwen35HybridMetalResidentPrefill() {
			t.Fatalf("%s: session is not the Metal resident lane", label)
		}
		got := gpu.Prefill(prompt)
		assertQ4KHybridPrefillMarker(t, gpu, 1, 0)
		if obs := gpu.Q4KPrefillGEMMObservation(); obs == Q4KPrefillGEMMTokenLoop || obs == Q4KPrefillGEMMNone || obs == Q4KPrefillGEMMCPU {
			t.Fatalf("%s observed Q4_K route %q, want a Metal GEMM identity", label, obs)
		}
		if gpu.Cache.Len() != P {
			t.Fatalf("%s cache len = %d, want %d", label, gpu.Cache.Len(), P)
		}
		if argmax(want) != argmax(got) {
			t.Fatalf("%s greedy token = %d, want %d", label, argmax(got), argmax(want))
		}
		assertCosineAtLeast(t, label+" logits", want, got, 0.999)
		ref.Close()
		gpu.Close()
	}
}

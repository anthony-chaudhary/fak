//go:build darwin && arm64 && cgo

// fak-test:runtime slow est=60s

package ggufload

import (
	"os"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
	"github.com/anthony-chaudhary/fak/internal/model"
)

func greedyQwen35Decode(t *testing.T, m *model.Model, prompt []int, steps int) (toks []int, graphCB, dispatchCB int) {
	t.Helper()
	s := m.NewSession()
	defer s.Close()
	s.Q4K = true
	s.MetalQ4K = true
	next := argmaxF32(s.Prefill(prompt))
	s.ResetMetalCommandBuffers()
	for i := 0; i < steps; i++ {
		toks = append(toks, next)
		next = argmaxF32(s.Step(next))
	}
	return toks, s.MetalGraphCommandBuffers(), s.MetalDispatchCommandBuffers()
}

// A sub-64-layer Qwen3.5-family hybrid (e.g. Qwen3.5-4B, 24 linear / 8 full layers) must
// auto-admit the resident one-command-buffer decode and greedy-decode the same tokens as the
// per-op Metal path.
func TestQwen35SmallHybridResidentDecodeGreedyParity(t *testing.T) {
	path := os.Getenv("FAK_TEST_QWEN35_HYBRID_GGUF")
	if path == "" {
		t.Skip("FAK_TEST_QWEN35_HYBRID_GGUF not set")
	}
	if !metalgemm.Available() {
		t.Skip("no Metal device")
	}
	m, err := LoadModelQ4K(path)
	if err != nil {
		t.Fatal(err)
	}
	if !m.Cfg.IsQwen35Hybrid() || m.Cfg.NumLayers == 64 {
		t.Skipf("want a non-64-layer Qwen3.5-family hybrid, got layers=%d", m.Cfg.NumLayers)
	}
	prompt := []int{9707, 11, 1124, 374, 264, 1273, 315, 279, 8213, 4566, 13, 576, 3974, 13876, 38835, 34208}
	const steps = 32

	model.SetQwen35ResidentDecodeAuto(false)
	want, _, perOpCB := greedyQwen35Decode(t, m, prompt, steps)
	model.SetQwen35ResidentDecodeAuto(true)
	got, graphCB, dispatchCB := greedyQwen35Decode(t, m, prompt, steps)

	if graphCB == 0 {
		t.Fatalf("resident decode was not admitted: graph=%d dispatch=%d command buffers", graphCB, dispatchCB)
	}
	if perCB := graphCB + dispatchCB; perCB >= perOpCB {
		t.Fatalf("resident decode used %d command buffers, per-op path %d", perCB, perOpCB)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("greedy token %d diverged: resident=%d per-op=%d\nresident=%v\nper-op=%v", i, got[i], want[i], got, want)
		}
	}
}

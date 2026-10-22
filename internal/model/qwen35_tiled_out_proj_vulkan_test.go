//go:build vulkan && (windows || linux) && cgo

package model

import (
	"fmt"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type vulkanValueHeadTileStats interface {
	VulkanDebugQwen35ValueHeadTileSnapshot() (kernelCalls, copyCalls int64)
}

// fak-test:runtime fast est=5s
func TestVulkanQwen35TiledOutProjMatchesCPUOnSequenceAndDecode(t *testing.T) {
	be, _ := requiredVulkanSequenceBackend(t)
	stats, ok := be.(vulkanValueHeadTileStats)
	if !ok {
		t.Fatalf("physical Vulkan backend %T exposes no value-head tiling counters", be)
	}
	cfg := qwen35HybridTestCfg()
	cfg.PartialRotaryFactor = .5
	cpu := NewSynthetic(cfg).NewSession()
	m := NewSynthetic(cfg)
	tiledLayers := len(tileSyntheticOutProjForTest(t, m))
	device, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatal(err)
	}
	defer device.Close()

	kernel0, copy0 := stats.VulkanDebugQwen35ValueHeadTileSnapshot()
	prompt := []int{3, 7, 11, 5, 17, 19, 23}
	want := append([]float32(nil), cpu.Prefill(prompt)...)
	got := append([]float32(nil), device.Prefill(prompt)...)
	if status, present := device.Qwen35SequencePrefillRouteStatus(); !present || status.EffectivePath != compute.Qwen35SequencePrefillPath || status.FallbackActive {
		t.Fatalf("prefill did not take the whole-sequence device route: %+v present=%t", status, present)
	}
	compareVulkanSequenceVector(t, "tiled_sequence_prefill_logits", got, want)
	follow := []int{29, 31, 37}
	for _, id := range follow {
		want = append([]float32(nil), cpu.Step(id)...)
		got = append([]float32(nil), device.Step(id)...)
		compareVulkanSequenceVector(t, fmt.Sprintf("tiled_decode_%d_logits", id), got, want)
		if argmaxF32(got) != argmaxF32(want) {
			t.Fatalf("token %d greedy mismatch", id)
		}
	}
	kernel1, copy1 := stats.VulkanDebugQwen35ValueHeadTileSnapshot()
	passes := (kernel1 - kernel0) + (copy1 - copy0)
	if wantPasses := int64(tiledLayers * (1 + len(follow))); passes != wantPasses {
		t.Fatalf("value-head tiling passes = %d (kernel %d, copy %d), want %d", passes, kernel1-kernel0, copy1-copy0, wantPasses)
	}
	t.Logf("device=%s tiled layers=%d tiling passes kernel=%d copy=%d", be.Tier(), tiledLayers, kernel1-kernel0, copy1-copy0)

	// Negative control: the permuted columns without the tile flag diverge on the device too.
	unflagged := NewSynthetic(cfg)
	tileSyntheticOutProjForTest(t, unflagged)
	unflagged.qwen35TiledOutProj = nil
	plain, err := unflagged.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatal(err)
	}
	defer plain.Close()
	wrong := plain.Prefill(prompt)
	if d := maxAbsDelta(wrong, NewSynthetic(cfg).NewSession().Prefill(prompt)); d < 1e-2 {
		t.Fatalf("permuted out_proj without tiling matched on the device (max|delta|=%g); the witness is vacuous", d)
	}
}

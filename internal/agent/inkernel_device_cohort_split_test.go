package agent

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// A dense layer split runs a per-lane host tail, so the all-layer device cohort must decline it;
// full placement (N == NumLayers) still coalesces.
//
// fak-test:runtime fast est=50ms lane=default
func TestInKernelQwenDeviceCohortDeclinesDenseLayerSplit(t *testing.T) {
	t.Setenv("FAK_INKERNEL_BATCH", "")
	p := deviceCohortPlanner(t, true)
	base := p.backend.(capacityProbeBackend)
	p.backend = &cohortCapabilityBackend{capacityProbeBackend: base, path: compute.Qwen35IndependentDecodeBatchPath}
	p.batchDecode = true
	if !p.coalescesQwenDecode() {
		t.Fatal("precondition: the full-placement device cohort must coalesce")
	}
	if p.m.Cfg.NumLayers < 2 {
		t.Fatalf("fixture needs at least two layers to split, has %d", p.m.Cfg.NumLayers)
	}
	p.denseGPULayers = 1
	if p.coalescesQwenDecode() {
		t.Fatal("a dense layer split must not join the all-layer device cohort")
	}
	p.denseGPULayers = p.m.Cfg.NumLayers
	if !p.coalescesQwenDecode() {
		t.Fatal("--gpu-layers == NumLayers is full placement and must still coalesce")
	}
}

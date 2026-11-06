package agent

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// fak-test:runtime fast est=5ms lane=default
func TestCUDADequantScratchPricesMoEOnlyWidth(t *testing.T) {
	moe := model.Config{
		HiddenSize: 64, MoEIntermediateSize: 96, VocabSize: 128, NumLayers: 2,
		NumHeads: 4, NumKVHeads: 2, HeadDim: 16,
	}
	dense := moe
	dense.IntermediateSize, dense.MoEIntermediateSize = moe.MoEIntermediateSize, 0
	cuda := kvPrecisionNamedBackend{Backend: compute.Default(), name: "cuda"}
	plan := func(cfg model.Config) compute.MemoryPlan {
		p := &InKernelPlanner{m: &model.Model{Cfg: cfg}, backend: cuda}
		return p.cudaDequantScratchDemand(compute.DequantGEMMScratchRows)
	}

	got, want := plan(moe).Total(), plan(dense).Total()
	if want <= 0 || got != want {
		t.Fatalf("MoE-only width dequant scratch = %d, want %d (priced at the expert width)", got, want)
	}
}

//go:build cuda

package model

import (
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// TestDeviceHopPipelineCUDAMatchesMonolithic runs the planned three-band hybrid pipeline
// with every boundary crossing real CUDA devices (compute.CUDAHop) and requires logits
// bit-identical to the monolithic forward, by the auto route and by forced pinned bounce.
// With two devices the third band reuses device 0. Pending hardware step on the laptop.
//
// fak-test:runtime slow est=5s
func TestDeviceHopPipelineCUDAMatchesMonolithic(t *testing.T) {
	n, err := compute.CUDADeviceCount()
	if err != nil || n < 2 {
		t.Skipf("needs >=2 CUDA devices (have %d, err %v)", n, err)
	}
	m := qwen35ThreePeriodModel(t)
	plan := planThreeDevices(t, m.Cfg)
	for i := range plan.Bands {
		plan.Bands[i].Device %= n
	}
	ids := []int{3, 1, 4, 1, 5, 9, 2, 6}
	full := m.Forward(ids).Logits
	for _, hop := range []compute.CUDAHop{compute.NewCUDAHopFromEnv(), {ForceBounce: true}} {
		tr := NewDeviceHopTransport(hop, plan)
		logits, err := RunPipelineWith(ids, stagesFromPlan(m, plan), tr)
		if err != nil {
			t.Fatalf("force=%v: %v", hop.ForceBounce, err)
		}
		for p := range full {
			for i := range full[p] {
				if math.Float32bits(logits[p][i]) != math.Float32bits(full[p][i]) {
					t.Fatalf("force=%v: logit[%d][%d] differs from monolithic", hop.ForceBounce, p, i)
				}
			}
		}
		for _, r := range tr.Receipts {
			t.Logf("force=%v %d->%d route=%s bytes=%d", hop.ForceBounce, r.Src, r.Dst, r.Route, r.Bytes)
		}
	}
}

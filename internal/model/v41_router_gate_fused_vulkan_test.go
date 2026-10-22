//go:build vulkan && (windows || linux) && cgo

// Physical witness on any Vulkan device with the SPIR-V bundle built:
//   FAK_VULKAN_REQUIRE_DEVICE=1 go test -tags vulkan ./internal/model -run '^TestV41RouterGateFusedVulkan$' -count=1
// The backend is process-shared, so this witness runs serially.

package model

import (
	"math"
	"os"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// The native V4.1 step routes through the fused router gate with the picks
// and weight bits of the separate projection + router top-k, in one device
// dispatch instead of two.
// fak-test:runtime medium est=10s lane=default
func TestV41RouterGateFusedVulkan(t *testing.T) {
	require := os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1"
	be, ok := compute.Lookup("vulkan")
	if !ok || be == nil || be.Name() != "vulkan" || !be.Caps().DeviceMemory {
		if require {
			t.Fatal("required actual Vulkan device is unavailable")
		}
		t.Skip("actual Vulkan device unavailable")
	}
	gate, gateOK := be.(compute.V41RouterGateTopKBackend)
	router, routerOK := be.(compute.V41RouterTopKBackend)
	if !gateOK || !routerOK || !router.V41RouterTopKAvailable() || !gate.V41RouterGateTopKAvailable() {
		if require {
			t.Fatal("router modules are absent from the runtime SPIR-V directory or the fused gate failed its device self-check")
		}
		t.Skip("router modules unavailable")
	}
	m := v41FullStepPatchedNorms(t)
	t.Cleanup(func() {
		if err := m.CloseWeights(); err != nil {
			t.Errorf("CloseWeights: %v", err)
		}
	})
	s := v41DenseTestSession(t, m, be)
	st := s.v41State()
	scope := s.beginBackendExecution()
	t.Cleanup(scope.finish)
	scratch := &v41ProjScratch{projectionRoute: st.projectionRoute}
	bias := func() []float32 { return m.tensor(layerName(0, "ffn.gate.e_score_correction_bias")) }
	cfg := v41DefaultRouterConfig()
	dispatches := func() uint64 {
		for _, w := range compute.DeviceWorkSnapshot() {
			if w.Backend == "vulkan" {
				return w.Dispatches
			}
		}
		t.Fatal("vulkan device work reader is not registered")
		return 0
	}
	route := func(enabled string, x []float32) ([]routePick, uint64) {
		t.Setenv(v41RouterGateFusedEnv, enabled)
		before := dispatches()
		picks, err := m.v41RouterPicksForStep(0, x, m.Cfg.HiddenSize, cfg, true, scratch, bias)
		if err != nil {
			t.Fatalf("route with %s=%s: %v", v41RouterGateFusedEnv, enabled, err)
		}
		return picks, dispatches() - before
	}
	const rows = 32
	for row := 0; row < rows; row++ {
		x := make([]float32, m.Cfg.HiddenSize)
		for i := range x {
			h := uint32(i)*2654435761 + uint32(row+1)*40503
			h ^= h >> 15
			h *= 2246822519
			h ^= h >> 13
			x[i] = float32(int32(h>>8)-(1<<23)) / float32(1<<22)
		}
		fused, fusedDispatches := route("1", x)
		separate, separateDispatches := route("0", x)
		if fusedDispatches != 1 || separateDispatches != 2 {
			t.Fatalf("row %d dispatches fused=%d separate=%d, want 1 and 2", row, fusedDispatches, separateDispatches)
		}
		if len(fused) != cfg.TopK || len(separate) != cfg.TopK {
			t.Fatalf("row %d pick counts fused=%d separate=%d", row, len(fused), len(separate))
		}
		for slot := range separate {
			if fused[slot].expert != separate[slot].expert || math.Float32bits(fused[slot].weight) != math.Float32bits(separate[slot].weight) {
				t.Fatalf("row %d slot %d fused (%d, %#08x) separate (%d, %#08x)", row, slot,
					fused[slot].expert, math.Float32bits(fused[slot].weight), separate[slot].expert, math.Float32bits(separate[slot].weight))
			}
		}
	}
	if s.halClosed || s.halFailure != nil {
		t.Fatalf("session closed during routing: %v", s.halFailure)
	}
	t.Logf("V41_ROUTER_GATE_MODEL rows=%d bit_identical=%d dispatches_per_route fused=1 separate=2", rows, rows)
}

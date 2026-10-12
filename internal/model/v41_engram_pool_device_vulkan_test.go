//go:build vulkan && (windows || linux) && cgo

// Run from the repository root with the coupled V11/71 shader bundle:
// FAK_VULKAN_SPIRV="$PWD/internal/compute/spirv" FAK_VULKAN_REQUIRE_DEVICE=1 go test -tags vulkan ./internal/model -run '^TestV41EngramGateAndCompressorPoolVulkan$' -count=1 -timeout=10m
// The synthetic payload witnesses the session seams on a physical GPU, not a checkpoint.

package model

import (
	"os"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// fak-test:justify why=integration when=changed:internal/model/**
// fak-test:runtime slow est=30s lane=optin
// Refs #13811, #13812. The device session runs the Engram gate/add on the
// resident panel and the compressor gated pool on the backend; the control
// session on the same backend keeps every other seam and only routes these two
// ops through their host arms. Logits must agree within the session parity
// tolerance the other V4.1 seams use.
func TestV41EngramGateAndCompressorPoolVulkan(t *testing.T) {
	if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") != "1" {
		t.Skip("physical Engram/pool seam witness requires explicit device opt-in")
	}
	be, ok := compute.Lookup("vulkan")
	if !ok || be == nil || be.Name() != "vulkan" || !be.Caps().DeviceMemory {
		t.Fatal("required actual Vulkan device unavailable")
	}
	t.Logf("Vulkan device: %s", be.Tier())
	gate, okGate := be.(compute.V41EngramGateBackend)
	pool, okPool := be.(compute.V41CompressorPoolBackend)
	if !okGate || !gate.SupportsV41EngramGate() || !okPool || !pool.SupportsV41CompressorPool() {
		t.Fatal("registered Vulkan build lacks the V4.1 Engram gate or compressor pool module (rebuild the coupled V11/71 bundle)")
	}
	for _, compressed := range []bool{false, true} {
		name := "plain"
		if compressed {
			name = "compressed"
		}
		t.Run(name, func(t *testing.T) {
			m, _ := v41IncrementalEngramFixture(t, compressed)
			controlModel, _ := v41IncrementalEngramFixture(t, compressed)
			s, err := m.NewBackendSessionChecked(be)
			if err != nil {
				t.Fatal(err)
			}
			defer s.Close()
			control, err := controlModel.NewBackendSessionChecked(be)
			if err != nil {
				t.Fatal(err)
			}
			defer control.Close()
			ops := s.v41State().gatherOps
			if s.v41State().mhcDevice == nil || ops.engramGate == nil || ops.compressorPool == nil {
				t.Fatal("Vulkan session did not bind the mHC, Engram gate and compressor pool seams")
			}
			gateCalls, poolCalls := 0, 0
			bound, boundPool := ops.engramGate, ops.compressorPool
			ops.engramGate = func(l int, streams *v41DeviceStreams, rows *v41EngramGatheredRows, positions int, g v41EngramGeometry, eps float32) (bool, error) {
				gateCalls++
				return bound(l, streams, rows, positions, g, eps)
			}
			ops.compressorPool = func(l int, kv, scores []float32, ratio, width int) ([]float32, error) {
				poolCalls++
				return boundPool(l, kv, scores, ratio, width)
			}
			controlOps := control.v41State().gatherOps
			controlOps.engramGate, controlOps.compressorPool = nil, nil

			prefix, next := v41IncrementalPrefix(m.Cfg, 4)
			diffs := []float64{
				v41MHCVulkanMaxDiff(t, s.Prefill(prefix), control.Prefill(prefix)),
				v41MHCVulkanMaxDiff(t, s.Step(next), control.Step(next)),
				v41MHCVulkanMaxDiff(t, s.Prefill([]int{2, 5}), control.Prefill([]int{2, 5})),
			}
			t.Logf("device-vs-host Engram/pool logits max diff prefill=%g step=%g suffix=%g (gate calls %d, pool calls %d)", diffs[0], diffs[1], diffs[2], gateCalls, poolCalls)
			for _, d := range diffs {
				if d > 1e-4 {
					t.Fatalf("Engram gate/pool seams moved logits by %g (> 1e-4)", d)
				}
			}
			if gateCalls == 0 || (poolCalls > 0) != compressed || s.BackendSessionClosed() {
				t.Fatalf("gate calls=%d pool calls=%d closed=%t", gateCalls, poolCalls, s.BackendSessionClosed())
			}
		})
	}
}

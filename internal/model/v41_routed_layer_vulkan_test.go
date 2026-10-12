//go:build vulkan && (windows || linux) && cgo

// Run serially from the repository root with the coupled shader bundle:
// FAK_VULKAN_SPIRV="$PWD/internal/compute/spirv" FAK_VULKAN_REQUIRE_DEVICE=1 FAK_VULKAN_DISPATCH_PROFILE=1 go test -tags vulkan ./internal/model -run '^TestV41RoutedLayerBatchVulkan$' -count=1 -v -timeout=20m
// Synthetic pinned Q2_K/Q3_K slate; per-step submits are logged, and only the
// batched-vs-per-pick ordering is asserted, never a count or a wall time.

package model

import (
	"fmt"
	"math"
	"os"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type v41RoutedLayerVulkanRun struct {
	bound   bool
	logits  [][]float32
	submits []uint64
	wallMS  []float64
}

// TestV41RoutedLayerBatchVulkan decodes the pinned slate on the device with
// the batched routed layer off and on (#13820). Every prefill and decode
// logit is bitwise equal, and each batched decode step submits fewer times.
// fak-test:justify why=integration when=changed:internal/model/**
// fak-test:runtime slow est=60s lane=optin
func TestV41RoutedLayerBatchVulkan(t *testing.T) {
	if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") != "1" {
		t.Skip("physical batched routed-layer witness requires explicit device opt-in")
	}
	be, ok := compute.Lookup("vulkan")
	if !ok || be == nil || be.Name() != "vulkan" || !be.Caps().DeviceMemory {
		t.Fatal("required actual Vulkan device unavailable")
	}
	if deferred, ok := be.(compute.V41DeferredStatusBackend); !ok || !deferred.SupportsV41DeferredStatus(qkK) {
		t.Fatal("required Vulkan build lacks the deferred V4.1 status records")
	}
	snap, profiled, err := compute.CaptureBackendExecutionSnapshot(be)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("Vulkan device: %s profiled=%t", snap.Identity.Device, profiled && os.Getenv("FAK_VULKAN_DISPATCH_PROFILE") == "1")
	const steps = 6
	prompt := []int{1, 2, 3}
	for _, compressed := range []bool{false, true} {
		t.Run(fmt.Sprintf("compressed_%t", compressed), func(t *testing.T) {
			runs := map[bool]*v41RoutedLayerVulkanRun{}
			for _, batch := range []bool{false, true} {
				t.Run(fmt.Sprintf("batch_%t", batch), func(t *testing.T) {
					if !batch {
						t.Setenv(v41LayerExpertBatchEnv, "0")
					}
					m := v41PinnedSlateFixture(t, compressed)
					defer func() {
						if err := m.CloseWeights(); err != nil {
							t.Error(err)
						}
					}()
					s, err := m.NewBackendSessionChecked(be)
					if err != nil {
						t.Fatal(err)
					}
					defer s.Close()
					run := &v41RoutedLayerVulkanRun{bound: s.v41State().routedLayer != nil}
					last := append([]float32(nil), s.Prefill(prompt)...)
					run.logits = append(run.logits, last)
					for i := 0; i < steps; i++ {
						before, _, _ := compute.CaptureBackendExecutionSnapshot(be)
						opened := time.Now()
						last = append([]float32(nil), s.Step(argmaxF32(last))...)
						wall := float64(time.Since(opened).Microseconds()) / 1000
						after, _, _ := compute.CaptureBackendExecutionSnapshot(be)
						delta, err := compute.BackendExecutionDelta(before, after)
						if err != nil {
							t.Fatal(err)
						}
						run.logits = append(run.logits, last)
						run.submits = append(run.submits, delta.Counters.DispatchSubmits)
						run.wallMS = append(run.wallMS, wall)
						t.Logf("v41_routed_layer compressed=%t batch=%t step=%d submits=%d dispatches=%d d2h_count=%d wall_ms=%.3f token=%d",
							compressed, batch, i, delta.Counters.DispatchSubmits, delta.Counters.ComputeDispatches, delta.Counters.D2HCount, wall, argmaxF32(last))
					}
					if s.HostFallbackObserved() {
						t.Error("device decode observed a host fallback")
					}
					runs[batch] = run
				})
			}
			off, on := runs[false], runs[true]
			if off == nil || on == nil {
				t.Fatal("an arm did not complete")
			}
			if off.bound || !on.bound {
				t.Fatalf("batched routed layer bound off=%t on=%t, want false/true", off.bound, on.bound)
			}
			for i := range off.logits {
				if len(on.logits[i]) != len(off.logits[i]) {
					t.Fatalf("window %d logits width %d vs %d", i, len(on.logits[i]), len(off.logits[i]))
				}
				for j := range off.logits[i] {
					if math.Float32bits(on.logits[i][j]) != math.Float32bits(off.logits[i][j]) {
						t.Fatalf("window %d logit %d batched=%g per-pick=%g: not bitwise equal", i, j, on.logits[i][j], off.logits[i][j])
					}
				}
			}
			var offSum, onSum uint64
			for i := range off.submits {
				offSum += off.submits[i]
				onSum += on.submits[i]
				if off.submits[i] > 0 && on.submits[i] >= off.submits[i] {
					t.Errorf("decode step %d batched submits=%d not below per-pick %d", i, on.submits[i], off.submits[i])
				}
			}
			t.Logf("v41_routed_layer_summary compressed=%t steps=%d submits_per_step per_pick=%.1f batched=%.1f bitwise_equal_windows=%d",
				compressed, steps, float64(offSum)/steps, float64(onSum)/steps, len(off.logits))
		})
	}
}

//go:build vulkan && (windows || linux) && cgo

// Physical opt-in witness on Linux or Windows with Vulkan and cgo:
// FAK_VULKAN_REQUIRE_DEVICE=1 FAK_VULKAN_DISPATCH_PROFILE=1 go test -tags vulkan ./internal/model -run '^TestV41IncrementalDeviceExpertHalo$' -count=1 -timeout=5m
// The backend is process-shared, so this witness deliberately runs serially.

package model

import (
	"math"
	"os"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type v41IncrementalHaloBackend struct {
	compute.Backend
	matmuls int
}

func (b *v41IncrementalHaloBackend) MatMul(w, x compute.Tensor) compute.Tensor {
	b.matmuls++
	return b.Backend.MatMul(w, x)
}
func (b *v41IncrementalHaloBackend) SupportsDeviceWeightDtype(dt compute.Dtype) bool {
	return compute.BackendSupportsDeviceWeightDtype(b.Backend, dt)
}

// fak-test:runtime slow est=30s lane=optin
func TestV41IncrementalDeviceExpertHalo(t *testing.T) {
	be, ok := compute.Lookup("vulkan")
	if !ok || be == nil || be.Name() != "vulkan" || !be.Caps().DeviceMemory {
		if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
			t.Fatal("required actual Vulkan device is unavailable")
		}
		t.Skip("actual Vulkan device unavailable")
	}
	if os.Getenv("FAK_VULKAN_DISPATCH_PROFILE") != "1" {
		if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
			t.Fatal("required physical receipt needs FAK_VULKAN_DISPATCH_PROFILE=1 before process start")
		}
		t.Skip("physical dispatch profile is not enabled")
	}
	profile, ok := be.(interface {
		VulkanDebugDispatchProfileSnapshot() compute.VulkanDispatchProfile
	})
	if !ok {
		t.Fatal("actual Vulkan backend lacks dispatch profile witness")
	}
	identity, available, err := compute.CaptureBackendExecutionSnapshot(be)
	if err != nil || !available {
		t.Fatalf("actual Vulkan backend identity unavailable: available=%t err=%v", available, err)
	}
	device := strings.ToLower(identity.Identity.Device)
	if !strings.Contains(device, "radeon") || !strings.Contains(device, "8060s") {
		t.Fatalf("physical witness requires Radeon 8060S, got %q", identity.Identity.Device)
	}
	b := &v41IncrementalHaloBackend{Backend: be}
	m := v41IncrementalExpertFixture(t, false, false)
	s, err := m.NewBackendSessionChecked(b)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	if len(s.Prefill([]int{1, 2, 3})) == 0 || !s.v41IncrementalEligible() {
		t.Fatal("actual device session did not seed incremental state")
	}
	for _, suffix := range []bool{false, true} {
		opening, available, err := compute.CaptureBackendExecutionSnapshot(be)
		if err != nil || !available || opening.Identity != identity.Identity {
			t.Fatalf("physical execution opening identity invalid: available=%t err=%v", available, err)
		}
		before, n := profile.VulkanDebugDispatchProfileSnapshot(), b.matmuls
		phase, ids := "decode", []int{4}
		var got []float32
		if suffix {
			phase, ids = "prefill", []int{5, 6}
			got = s.Prefill(ids)
		} else {
			got = s.Step(4)
		}
		closing, available, err := compute.CaptureBackendExecutionSnapshot(be)
		if err != nil || !available {
			t.Fatalf("physical execution closing identity unavailable: available=%t err=%v", available, err)
		}
		observation, err := compute.BackendExecutionDelta(opening, closing)
		if err != nil {
			t.Fatal(err)
		}
		rows := len(ids) * m.Cfg.NumExpertsPerTok * m.Cfg.NumLayers
		if b.matmuls-n != 3*rows {
			t.Errorf("actual Vulkan expert MatMuls=%d, want %d", b.matmuls-n, 3*rows)
		}
		after := profile.VulkanDebugDispatchProfileSnapshot()
		if after.Q2KMatmulDispatches <= before.Q2KMatmulDispatches {
			t.Error("incremental route issued no physical Q2_K dispatches")
		}
		p := v41IncrementalExpertPhase(t, m, phase)
		if p["incremental_device_gate_up_calls"] != float64(rows) || p["incremental_device_down_calls"] != float64(rows) || p["incremental_device_dispatch_nanos"] <= 0 {
			t.Error("actual Vulkan callback attribution absent or inconsistent")
		}
		history := []int{1, 2, 3, 4}
		if suffix {
			history = append(history, 5, 6)
		}
		oracle := lastLogits(v41IncrementalExpertFixture(t, false, false).Forward(history))
		if len(got) != len(oracle) || len(got) == 0 {
			t.Fatal("physical incremental logits width invalid")
		}
		var dot, a2, b2 float64
		for i, x := range got {
			if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
				t.Fatal("physical logits must be finite")
			}
			a, c := float64(x), float64(oracle[i])
			dot += a * c
			a2 += a * a
			b2 += c * c
		}
		if a2 == 0 || b2 == 0 || dot/math.Sqrt(a2*b2) < 0.999 {
			t.Error("physical incremental logits lost host oracle cosine >=0.999")
		}
		t.Logf("backend=%s device=%q driver=%q runtime=%q phase=%s routed_rows=%d physical_q2_dispatches=%d compute_dispatches=%d dispatch_submits=%d h2d_bytes=%d d2h_bytes=%d", observation.Identity.Backend, observation.Identity.Device, observation.Identity.Driver, observation.Identity.Runtime, phase, rows, after.Q2KMatmulDispatches-before.Q2KMatmulDispatches, observation.Counters.ComputeDispatches, observation.Counters.DispatchSubmits, observation.Counters.H2DBytes, observation.Counters.D2HBytes)
	}
}

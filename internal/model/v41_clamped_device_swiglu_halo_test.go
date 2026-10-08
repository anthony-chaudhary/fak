//go:build vulkan && (windows || linux) && cgo

// Run with: FAK_VULKAN_REQUIRE_DEVICE=1 FAK_VULKAN_DISPATCH_PROFILE=1 go test -tags vulkan ./internal/model -run '^TestV41ClampedDeviceSwiGLUHalo$' -count=1 -timeout=5m
// The Vulkan backend is process-shared; run this physical witness serially.

package model

import (
	"encoding/json"
	"math"
	"os"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// Embedding the closed Backend interface intentionally hides SwiGLUWithLimit
// while every admitted projection still reaches the same physical backend.
type v41ClampedHaloHostBackend struct {
	compute.Backend
	matmuls int
}

func (b *v41ClampedHaloHostBackend) MatMul(w, x compute.Tensor) compute.Tensor {
	b.matmuls++
	return b.Backend.MatMul(w, x)
}
func (b *v41ClampedHaloHostBackend) SupportsDeviceWeightDtype(dt compute.Dtype) bool {
	return compute.BackendSupportsDeviceWeightDtype(b.Backend, dt)
}

type v41ClampedHaloDeviceBackend struct {
	*v41ClampedHaloHostBackend
	limited int
}

func (b *v41ClampedHaloDeviceBackend) SwiGLUWithLimit(g, u compute.Tensor, limit float32) compute.Tensor {
	operation, ok := b.Backend.(interface {
		SwiGLUWithLimit(compute.Tensor, compute.Tensor, float32) compute.Tensor
	})
	if !ok {
		panic("required physical limited activation is unavailable")
	}
	b.limited++
	return operation.SwiGLUWithLimit(g, u, limit)
}

type v41ClampedHaloRun struct {
	logits      []float32
	observation compute.BackendExecutionObservation
	activation  map[string]float64
}

// fak-test:runtime slow est=30s lane=optin
func TestV41ClampedDeviceSwiGLUHalo(t *testing.T) {
	if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") != "1" {
		t.Skip("physical witness requires explicit device opt-in")
	}
	if os.Getenv("FAK_VULKAN_DISPATCH_PROFILE") != "1" {
		t.Fatal("physical witness requires dispatch profile before process start")
	}
	be, ok := compute.Lookup("vulkan")
	if !ok || be == nil || be.Name() != "vulkan" || !be.Caps().DeviceMemory {
		t.Fatal("required actual Vulkan device is unavailable")
	}
	profile, ok := be.(interface {
		VulkanDebugDispatchProfileSnapshot() compute.VulkanDispatchProfile
	})
	if !ok {
		t.Fatal("actual Vulkan backend lacks dispatch profile witness")
	}
	if _, ok := be.(interface {
		SwiGLUWithLimit(compute.Tensor, compute.Tensor, float32) compute.Tensor
	}); !ok {
		t.Fatal("actual Vulkan backend lacks limited activation operation")
	}
	identity, available, err := compute.CaptureBackendExecutionSnapshot(be)
	if err != nil || !available {
		t.Fatalf("actual Vulkan identity unavailable: available=%t error=%v", available, err)
	}
	device := strings.ToLower(identity.Identity.Device)
	if !strings.Contains(device, "radeon") || !strings.Contains(device, "8060s") || !strings.Contains(device, "radv") {
		t.Fatalf("physical witness requires Radeon8060S, got %q", identity.Identity.Device)
	}
	if !identity.TransferCountersObserved {
		t.Fatal("physical transfer counters unavailable")
	}
	// Saturating operands reject an old shader that silently ignores the limit.
	gateValues, upValues := []float32{4, -4, 4, -4}, []float32{4, 4, -4, -4}
	gate := be.Upload(compute.NewF32(compute.Default(), []int{4}, gateValues), compute.F32)
	defer be.Free(gate)
	up := be.Upload(compute.NewF32(compute.Default(), []int{4}, upValues), compute.F32)
	defer be.Free(up)
	limitedOperation := be.(interface {
		SwiGLUWithLimit(compute.Tensor, compute.Tensor, float32) compute.Tensor
	})
	activation := limitedOperation.SwiGLUWithLimit(gate, up, 0.01)
	defer be.Free(activation)
	gotActivation := be.Read(activation)
	if len(gotActivation) != 4 {
		t.Fatal("physical limited activation oracle width invalid")
	}
	var activationMaximumDiff float64
	for i, value := range gotActivation {
		g := min(gateValues[i], float32(0.01))
		u := max(float32(-0.01), min(upValues[i], float32(0.01)))
		want := g / (1 + float32(math.Exp(float64(-g)))) * u
		diff := math.Abs(float64(value - want))
		activationMaximumDiff = max(activationMaximumDiff, diff)
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || diff > 1e-6 {
			t.Errorf("physical saturated activation[%d] difference=%g want <=1e-6", i, diff)
		}
	}
	t.Logf("physical_activation_oracle width=4 limit=0.01 gate_upper=true gate_negative_unbounded=true up_both_bounds=true maximum_diff=%g", activationMaximumDiff)
	run := func(suffix, capable bool) v41ClampedHaloRun {
		m := v41IncrementalExpertFixture(t, false, false)
		m.Cfg.SwigluLimit = 0.01
		if len(m.Cfg.DeepSeekV41.EngramLayerIDs) != 0 {
			t.Fatal("reduced activation witness unexpectedly enables Engram")
		}
		host := &v41ClampedHaloHostBackend{Backend: be}
		device := &v41ClampedHaloDeviceBackend{v41ClampedHaloHostBackend: host}
		var selected compute.Backend = host
		if capable {
			selected = device
		}
		s, err := m.NewBackendSessionChecked(selected)
		if err != nil {
			t.Fatal(err)
		}
		s.v41State().denseProjection = nil
		defer s.Close()
		if len(s.Prefill([]int{1, 2, 3})) == 0 || !s.v41IncrementalEligible() {
			t.Fatal("physical session did not seed incremental state")
		}
		phase, ids := "decode", []int{4}
		if suffix {
			phase, ids = "prefill", []int{4, 5}
		}
		opening, available, err := compute.CaptureBackendExecutionSnapshot(be)
		if err != nil || !available || opening.Identity != identity.Identity {
			t.Fatalf("physical opening identity invalid: available=%t error=%v", available, err)
		}
		profileBefore := profile.VulkanDebugDispatchProfileSnapshot()
		activationBefore := v41ClampedActivationPhase(t, m, phase)
		matmuls, limited := host.matmuls, device.limited
		var logits []float32
		if suffix {
			logits = s.Prefill(ids)
		} else {
			logits = s.Step(4)
		}
		closing, available, err := compute.CaptureBackendExecutionSnapshot(be)
		if err != nil || !available {
			t.Fatalf("physical closing identity unavailable: available=%t error=%v", available, err)
		}
		observation, err := compute.BackendExecutionDelta(opening, closing)
		if err != nil {
			t.Fatal(err)
		}
		profileAfter := profile.VulkanDebugDispatchProfileSnapshot()
		rows := len(ids) * m.Cfg.NumLayers * m.Cfg.NumExpertsPerTok
		if host.matmuls-matmuls != 3*rows {
			t.Errorf("physical MatMuls=%d want %d", host.matmuls-matmuls, 3*rows)
		}
		if profileAfter.Q2KMatmulDispatches-profileBefore.Q2KMatmulDispatches != uint64(2*rows) {
			t.Error("physical gate/up Q2_K dispatch count differs from selected rows")
		}
		activationAfter := v41ClampedActivationPhase(t, m, phase)
		activation := map[string]float64{}
		for key, value := range activationAfter {
			activation[key] = value - activationBefore[key]
		}
		wantDevice, wantHost, wantActivationReads := rows, 0, rows
		if !capable {
			wantDevice, wantHost, wantActivationReads = 0, rows, 2*rows
		}
		if device.limited-limited != wantDevice || activation["expert_activation_device_calls"] != float64(wantDevice) || activation["expert_activation_host_calls"] != float64(wantHost) {
			t.Errorf("selected activation counters=%v limited_calls=%d", activation, device.limited-limited)
		}
		if activation["expert_activation_readback_bytes"] != float64(4*wantActivationReads*m.Cfg.MoEIntermediateSize) || activation["expert_activation_nanos"] <= 0 {
			t.Errorf("selected activation bytes/elapsed invalid: %v", activation)
		}
		wantD2HBytes := uint64(4 * (wantActivationReads*m.Cfg.MoEIntermediateSize + rows*m.Cfg.HiddenSize))
		if observation.Counters.D2HBytes != wantD2HBytes || observation.Counters.D2HCount != uint64(wantActivationReads+rows) {
			t.Errorf("physical D2H=%d bytes/%d reads want %d/%d", observation.Counters.D2HBytes, observation.Counters.D2HCount, wantD2HBytes, wantActivationReads+rows)
		}
		for _, name := range []string{"w1.weight", "w3.weight", "w2.weight"} {
			for expert := 0; expert < m.Cfg.NumExperts; expert++ {
				weight := layerName(0, "ffn.experts."+itoa(expert)+"."+name)
				if m.has(weight) {
					t.Fatal("routed compressed weight gained a host F32 manifest")
				}
				if staged, ok := s.halW["kquant-raw:"+weight]; ok {
					want := compute.Q2_K
					if name == "w2.weight" {
						want = compute.Q3_K
					}
					if staged.Dtype != want {
						t.Fatalf("staged projection dtype=%s want %s", staged.Dtype, want)
					}
				}
			}
		}
		raw, err := json.Marshal(m.V41ExpertFaultAttribution())
		if err != nil {
			t.Fatal(err)
		}
		var phases map[string]json.RawMessage
		if err := json.Unmarshal(raw, &phases); err != nil {
			t.Fatal(err)
		}
		t.Logf("backend=%s device=%q driver=%q runtime=%q phase=%s capable=%t tokens=%d routed_rows=%d q2_dispatches=%d compute_dispatches=%d d2h_bytes=%d d2h_reads=%d activation_delta=%v default_phase_json=%s", observation.Identity.Backend, observation.Identity.Device, observation.Identity.Driver, observation.Identity.Runtime, phase, capable, len(ids), rows, profileAfter.Q2KMatmulDispatches-profileBefore.Q2KMatmulDispatches, observation.Counters.ComputeDispatches, observation.Counters.D2HBytes, observation.Counters.D2HCount, activation, phases[phase])
		return v41ClampedHaloRun{logits, observation, activation}
	}
	for _, suffix := range []bool{false, true} {
		treatment, control := run(suffix, true), run(suffix, false)
		if len(treatment.logits) == 0 || len(treatment.logits) != len(control.logits) {
			t.Fatal("matched physical logits width invalid")
		}
		var dot, a2, b2, maximumDiff float64
		for i, value := range treatment.logits {
			a, b := float64(value), float64(control.logits[i])
			if math.IsNaN(a) || math.IsInf(a, 0) || math.IsNaN(b) || math.IsInf(b, 0) {
				t.Fatal("matched physical logits must be finite")
			}
			dot += a * b
			a2 += a * a
			b2 += b * b
			maximumDiff = max(maximumDiff, math.Abs(a-b))
		}
		if a2 == 0 || b2 == 0 {
			t.Fatal("matched physical cosine is vacuous")
		}
		cosine := dot / math.Sqrt(a2*b2)
		if cosine < 0.999 {
			t.Errorf("matched physical activation cosine=%g want >=0.999", cosine)
		}
		if 2*treatment.activation["expert_activation_readback_bytes"] != control.activation["expert_activation_readback_bytes"] {
			t.Error("physical selected activation readback bytes did not halve")
		}
		saved := control.observation.Counters.D2HBytes - treatment.observation.Counters.D2HBytes
		if saved != uint64(treatment.activation["expert_activation_readback_bytes"]) {
			t.Error("physical D2H savings disagree with single activation-row ledger")
		}
		t.Logf("matched_physical suffix=%t logits_width=%d cosine=%g maximum_diff=%g saved_d2h_bytes=%d", suffix, len(treatment.logits), cosine, maximumDiff, saved)
	}
}

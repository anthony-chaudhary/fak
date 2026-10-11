//go:build vulkan && (windows || linux) && cgo

// Physical opt-in witness on Linux or Windows with Vulkan and cgo:
// FAK_VULKAN_REQUIRE_DEVICE=1 go test -tags vulkan ./internal/model -run '^TestV41RoutedSwiGLUVulkan' -count=1 -timeout=10m
// FAK_VULKAN_EXPECT_DEVICE=<substring> pins the device identity. A skip is no witness.

package model

import (
	"math"
	"os"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func v41RoutedSwiGLUVulkanBackend(t *testing.T) (compute.Backend, compute.V41RoutedSwiGLUBackend) {
	t.Helper()
	require := os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1"
	be, ok := compute.Lookup("vulkan")
	if !ok || be == nil || be.Name() != "vulkan" || !be.Caps().DeviceMemory {
		if require {
			t.Fatal("required actual Vulkan device is unavailable")
		}
		t.Skip("actual Vulkan device unavailable")
	}
	op, ok := be.(compute.V41RoutedSwiGLUBackend)
	if !ok || !op.V41RoutedSwiGLUAvailable() {
		if require {
			t.Fatal("required Vulkan build lacks the V4.1 routed SwiGLU module or binary32 float controls")
		}
		t.Skip("V4.1 routed SwiGLU module unavailable")
	}
	identity, available, err := compute.CaptureBackendExecutionSnapshot(be)
	if err != nil || !available {
		t.Fatalf("actual Vulkan identity unavailable: available=%t err=%v", available, err)
	}
	if expected := os.Getenv("FAK_VULKAN_EXPECT_DEVICE"); expected != "" && !strings.Contains(strings.ToLower(identity.Identity.Device), strings.ToLower(expected)) {
		t.Fatalf("Vulkan device %q does not match required %q", identity.Identity.Device, expected)
	}
	t.Logf("Vulkan device: %s", identity.Identity.Device)
	return be, op
}

// v41RoutedOperandBounds is the host BF16 down operand over the device gate/up
// outputs, and the interval the device lane must round into: Vulkan bounds exp
// at 3+2|x| ULP and division at 2.5 ULP; add, multiply and the BF16 bit
// rounding are exact. 2^-100 admits only the exp-overflow tail (|g| > 88.7).
func v41RoutedOperandBounds(gate, up, limit, weight float32) (want, lo, hi float32) {
	g, u := v41RoundBF16(gate), v41RoundBF16(up)
	if limit > 0 {
		g = min(g, limit)
		u = max(-limit, min(u, limit))
	}
	pre := silu(g) * u * weight
	tol := (9.5+2*math.Abs(float64(g)))*0x1p-23*math.Abs(float64(pre)) + 0x1p-100
	return v41RoundBF16(pre), v41RoundBF16(float32(float64(pre) - tol)), v41RoundBF16(float32(float64(pre) + tol))
}

// fak-test:runtime slow est=30s lane=optin
// Ref #13815. On the pinned layout (Q2_K gate/up, Q3_K down) the fused routed
// expert is exactly the device down projection of its own fused operand, and
// every operand lane is within the activation bound of the host BF16 clamp
// SwiGLU over the same device gate/up GEMM outputs.
func TestV41RoutedSwiGLUVulkanExpert(t *testing.T) {
	be, op := v41RoutedSwiGLUVulkanBackend(t)
	for _, limit := range []float64{10, 0.05} {
		m := v41IncrementalExpertFixture(t, true, false)
		m.Cfg.SwigluLimit = limit
		cfg := m.Cfg
		H, I := cfg.HiddenSize, cfg.MoEIntermediateSize
		s, err := m.NewBackendSessionChecked(be)
		if err != nil {
			t.Fatal(err)
		}
		routed := s.v41RoutedExpertFunc()
		if routed == nil {
			t.Fatal("full-profile Vulkan session left the fused routed seam unbound")
		}
		x := make([]float32, H)
		for i := range x {
			x[i] = v41RoundBF16(float32(math.Sin(float64(i)*0.37)) * 1.7)
		}
		lanes, exact, clamped := 0, 0, 0
		for expert := 0; expert < cfg.NumExperts; expert++ {
			stem := "ffn.experts." + itoa(expert)
			for _, weight := range []float32{.501953125, 2.5} {
				got, outcome, err := routed(0, stem, x, weight)
				if outcome != v41RoutedExpertHandled || err != nil || len(got) != H {
					t.Fatalf("limit=%g expert=%d: fused seam outcome=%v err=%v width=%d", limit, expert, outcome, err, len(got))
				}
				gateW, releaseGate, upW, releaseUp, input, ok := expertInputDeviceAdmitted(s, layerName(0, stem+".w1.weight"), layerName(0, stem+".w3.weight"), x, H)
				if !ok {
					t.Fatal("reconstruction could not admit gate/up")
				}
				downW, releaseDown, ok := s.q4kExpertDownDeviceWeight(layerName(0, stem+".w2.weight"))
				if !ok {
					t.Fatal("reconstruction could not admit down")
				}
				xd := s.uploadHostF32([]int{H}, input, compute.MemoryActivation, "routed SwiGLU witness input")
				g, u := be.MatMul(gateW, xd), be.MatMul(upW, xd)
				gate := append([]float32(nil), be.Read(g)...)
				up := append([]float32(nil), be.Read(u)...)
				activation, err := op.V41RoutedSwiGLU(g, u, float32(limit), weight)
				if err != nil {
					t.Fatalf("limit=%g expert=%d: reconstruction activation: %v", limit, expert, err)
				}
				operand := append([]float32(nil), be.Read(activation)...)
				y := be.MatMul(downW, activation)
				want := append([]float32(nil), be.Read(y)...)
				for _, tensor := range []compute.Tensor{y, activation, u, g, xd} {
					be.Free(tensor)
				}
				releaseDown()
				releaseUp()
				releaseGate()
				if !reflect.DeepEqual(got, want) {
					t.Fatalf("limit=%g expert=%d weight=%g: fused output is not the device down projection of its operand", limit, expert, weight)
				}
				if len(gate) != I || len(up) != I || len(operand) != I {
					t.Fatal("device projection widths differ from the intermediate size")
				}
				for i, value := range operand {
					lanes++
					if bf16 := v41RoundBF16(gate[i]); limit > 0 && (bf16 > float32(limit) || math.Abs(float64(v41RoundBF16(up[i]))) > limit) {
						clamped++
					}
					hostLane, lo, hi := v41RoutedOperandBounds(gate[i], up[i], float32(limit), weight)
					if math.Float32bits(value) == math.Float32bits(hostLane) {
						exact++
						continue
					}
					if !(value >= lo && value <= hi) {
						t.Fatalf("limit=%g expert=%d lane %d: device operand %g outside [%g,%g] (host %g)", limit, expert, i, value, lo, hi, hostLane)
					}
				}
			}
		}
		if limit < 1 && clamped == 0 {
			t.Fatalf("limit=%g clamped no lane; the witness would not see an ignored limit", limit)
		}
		t.Logf("routed_swiglu_expert limit=%g lanes=%d exact=%d clamped=%d", limit, lanes, exact, clamped)
		s.Close()
	}
}

// fak-test:runtime slow est=60s lane=optin
// Ref #13815. Every routed row of a full-profile session's prefill and decode
// runs the fused activation on the device: no host activation and only the
// 16-byte status readback per row.
func TestV41RoutedSwiGLUVulkanSession(t *testing.T) {
	be, _ := v41RoutedSwiGLUVulkanBackend(t)
	m := v41IncrementalExpertFixture(t, true, false)
	m.Cfg.SwigluLimit = 0.05
	s, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	for _, step := range []struct {
		phase string
		ids   []int
	}{{"prefill", []int{1, 2, 3}}, {"decode", []int{4}}, {"prefill", []int{5, 6}}} {
		before := v41ClampedActivationPhase(t, m, step.phase)
		var logits []float32
		if step.phase == "decode" {
			logits = s.Step(step.ids[0])
		} else {
			logits = s.Prefill(step.ids)
		}
		if len(logits) == 0 {
			t.Fatalf("%s produced no logits", step.phase)
		}
		for i, v := range logits {
			if !finite32(v) {
				t.Fatalf("%s logit %d is nonfinite", step.phase, i)
			}
		}
		after := v41ClampedActivationPhase(t, m, step.phase)
		rows := float64(len(step.ids) * m.Cfg.NumExpertsPerTok * m.Cfg.NumLayers)
		device := after["expert_activation_device_calls"] - before["expert_activation_device_calls"]
		host := after["expert_activation_host_calls"] - before["expert_activation_host_calls"]
		readback := after["expert_activation_readback_bytes"] - before["expert_activation_readback_bytes"]
		if device != rows || host != 0 || readback != 16*rows {
			t.Fatalf("%s %v: activation device=%g host=%g readback=%g, want %g/0/%g", step.phase, step.ids, device, host, readback, rows, 16*rows)
		}
		t.Logf("routed_swiglu_session phase=%s tokens=%d device_activations=%g host_activations=%g readback_bytes=%g", step.phase, len(step.ids), device, host, readback)
	}
}

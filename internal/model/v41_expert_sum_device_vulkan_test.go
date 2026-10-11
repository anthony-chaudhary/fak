//go:build vulkan && (windows || linux) && cgo

// Physical opt-in witness on any Vulkan device (Halo 8060S or a discrete card):
// FAK_VULKAN_REQUIRE_DEVICE=1 go test -tags vulkan ./internal/model -run '^TestV41DeviceExpertSumVulkan$' -count=1 -timeout=5m
// The backend is process-shared, so this witness runs serially.

package model

import (
	"math"
	"math/rand"
	"os"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// fak-test:runtime slow est=20s lane=optin
func TestV41DeviceExpertSumVulkan(t *testing.T) {
	be, ok := compute.Lookup("vulkan")
	if !ok || be == nil || be.Name() != "vulkan" || !be.Caps().DeviceMemory {
		if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
			t.Fatal("required actual Vulkan device is unavailable")
		}
		t.Skip("actual Vulkan device unavailable")
	}
	identity, available, err := compute.CaptureBackendExecutionSnapshot(be)
	if err != nil || !available {
		t.Fatalf("actual Vulkan backend identity unavailable: available=%t err=%v", available, err)
	}
	t.Logf("device=%q", identity.Identity.Device)
	const H = 4096
	rng := rand.New(rand.NewSource(13668))
	s := &Session{M: &Model{}, Backend: be}
	routedSum, sharedAdd := s.v41RoutedSumFunc(), s.v41SharedAddFunc()
	if routedSum == nil || sharedAdd == nil {
		t.Fatal("device sum seams did not bind on an actual Vulkan backend")
	}
	for trial, picks := range [][]routePick{
		{{expert: 301, weight: 0.4}, {expert: 7}, {expert: 199}, {expert: 42}, {expert: 383}, {expert: 0}},
		{{expert: 9}, {expert: 3}, {expert: 9}, {expert: 3}, {expert: 1}, {expert: 9}},
	} {
		outputs := v41SumFixture(rng, len(picks), H)
		outputs[0][0], outputs[1][0] = float32(math.Copysign(0, -1)), float32(math.Copysign(0, -1))
		outputs[0][1], outputs[1][1], outputs[2][1] = -1e8, 1e8, 1
		want, err := v41FullRoutedSum(trial, picks, outputs, H)
		if err != nil {
			t.Fatal(err)
		}
		got, err := v41FullRoutedSumWith(trial, picks, outputs, H, routedSum)
		if err != nil {
			t.Fatalf("device routed sum: %v", err)
		}
		if !v41SameBits(got, want) {
			t.Fatalf("trial %d: device routed sum differs from host bits", trial)
		}
		shared := v41SumFixture(rng, 1, H)[0]
		wantMoE, err := v41SharedExpertAdd(want, shared, v41DefaultRouterConfig())
		if err != nil {
			t.Fatal(err)
		}
		gotMoE, err := v41SharedExpertAddWith(trial, got, shared, v41DefaultRouterConfig(), sharedAdd)
		if err != nil {
			t.Fatalf("device shared add: %v", err)
		}
		if !v41SameBits(gotMoE, wantMoE) {
			t.Fatalf("trial %d: device shared add differs from host bits", trial)
		}
	}
	// Subnormal operands: report, and bound, a flush-to-zero difference.
	sub := [][]float32{{math.Float32frombits(1), math.Float32frombits(3), 1}, {math.Float32frombits(2), -math.Float32frombits(3), 2}}
	picks := []routePick{{expert: 1}, {expert: 0}}
	want, _ := v41FullRoutedSum(0, picks, sub, 3)
	got, err := v41FullRoutedSumWith(0, picks, sub, 3, routedSum)
	if err != nil {
		t.Fatalf("subnormal routed sum: %v", err)
	}
	flushed := !v41SameBits(got, want)
	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) && (math.Abs(float64(want[i])) >= 0x1p-126 || math.Abs(float64(got[i])) >= 0x1p-126) {
			t.Fatalf("subnormal fixture differs beyond a flush at %d: got=%g want=%g", i, got[i], want[i])
		}
	}
	t.Logf("V41_DEVICE_EXPERT_SUM_COMPLETE H=%d subnormal_flush=%t", H, flushed)
}

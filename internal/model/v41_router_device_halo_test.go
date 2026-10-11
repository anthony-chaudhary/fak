//go:build vulkan && (windows || linux) && cgo

// Physical opt-in witness on any Vulkan device (Halo 8060S or a discrete card).
// The router module is not in the trusted shader registry yet, so compile it
// into the runtime SPIR-V directory first:
//   glslc -O --target-env=vulkan1.2 -fshader-stage=comp internal/compute/shaders/v41_router_topk.comp -o "$FAK_VULKAN_SPIRV/v41_router_topk.spv"
//   FAK_VULKAN_REQUIRE_DEVICE=1 go test -tags vulkan ./internal/model -run '^TestV41RouterTopKVulkan$' -count=1 -timeout=5m
// The backend is process-shared, so this witness runs serially.

package model

import (
	"errors"
	"math"
	"math/rand"
	"os"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// fak-test:runtime slow est=30s lane=optin
func TestV41RouterTopKVulkan(t *testing.T) {
	be, ok := compute.Lookup("vulkan")
	if !ok || be == nil || be.Name() != "vulkan" || !be.Caps().DeviceMemory {
		if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
			t.Fatal("required actual Vulkan device is unavailable")
		}
		t.Skip("actual Vulkan device unavailable")
	}
	op, ok := be.(compute.V41RouterTopKBackend)
	if !ok || !op.V41RouterTopKAvailable() {
		if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
			t.Fatal("v41_router_topk.spv is absent from the runtime SPIR-V directory or failed to build")
		}
		t.Skip("optional router module unavailable")
	}
	identity, available, err := compute.CaptureBackendExecutionSnapshot(be)
	if err != nil || !available {
		t.Fatalf("actual Vulkan backend identity unavailable: available=%t err=%v", available, err)
	}
	t.Logf("device=%q; weights abs/rel 2e-5, picks exact", identity.Identity.Device)
	const tol = 2e-5
	cfg := v41DefaultRouterConfig()
	rng := rand.New(rand.NewSource(13668))
	compare := func(label string, got, want []routePick) {
		t.Helper()
		if len(got) != len(want) {
			t.Fatalf("%s: %d picks, want %d", label, len(got), len(want))
		}
		for i := range want {
			d := math.Abs(float64(got[i].weight) - float64(want[i].weight))
			if got[i].expert != want[i].expert || d > tol+tol*math.Abs(float64(want[i].weight)) {
				t.Fatalf("%s slot %d: device %+v, host %+v", label, i, got[i], want[i])
			}
		}
	}
	upload := func(shape []int, v []float32) compute.Tensor {
		return uploadHostF32Class(be, shape, v, compute.MemoryActivation, "V4.1 router witness")
	}
	const rows = 256
	worst := 0.0
	for _, full := range []bool{false, true} {
		for _, withBias := range []bool{false, true} {
			panel := make([]float32, 0, rows*cfg.Experts)
			rowsLogits := make([][]float32, rows)
			var bias []float32
			for r := 0; r < rows; r++ {
				logits, b := v41RouterFixture(rng)
				if withBias && bias == nil {
					bias = b
				}
				switch r {
				case 0: // exact ties keep the lower expert first
					logits[17], logits[260] = 9, 9
					if withBias {
						bias[17], bias[260] = 0.25, 0.25
					}
				case 1: // extremes across every score branch
					logits[3], logits[4], logits[5], logits[6] = -100, -30, -19.5, 60
				}
				rowsLogits[r] = logits
				panel = append(panel, logits...)
			}
			lt := upload([]int{rows, cfg.Experts}, panel)
			biasRow := []float32{0}
			if withBias {
				biasRow = bias
			}
			bt := upload([]int{len(biasRow)}, biasRow)
			got, err := v41DeviceRoute(op, lt, bt, rows, cfg, full, withBias)
			be.Free(lt)
			be.Free(bt)
			if err != nil {
				t.Fatalf("full=%t bias=%t: %v", full, withBias, err)
			}
			for r := range rowsLogits {
				want, err := v41RouteForGeometry(rowsLogits[r], bias, cfg, full)
				if err != nil {
					t.Fatal(err)
				}
				compare("batched", got[r], want)
				for i := range want {
					worst = math.Max(worst, math.Abs(float64(got[r][i].weight)-float64(want[i].weight)))
				}
			}
			s := &Session{M: &Model{}, Backend: be}
			one, err := v41RouteForGeometryWith(0, rowsLogits[2], bias, cfg, full, s.v41RouterTopKFunc())
			if err != nil {
				t.Fatalf("session seam: %v", err)
			}
			want, _ := v41RouteForGeometry(rowsLogits[2], bias, cfg, full)
			compare("session", one, want)
		}
	}
	bad, bias := v41RouterFixture(rng)
	bad[11] = float32(math.NaN())
	s := &Session{M: &Model{}, Backend: be}
	_, err = v41RouteForGeometryWith(0, bad, bias, cfg, true, s.v41RouterTopKFunc())
	var route *v4RouteError
	if !errors.As(err, &route) || route.Field != "logits" || !s.BackendSessionClosed() {
		t.Fatalf("non-finite logit: %v", err)
	}
	t.Logf("V41_ROUTER_TOPK_COMPLETE rows=%d max_abs_weight_delta=%.3g", rows, worst)
}

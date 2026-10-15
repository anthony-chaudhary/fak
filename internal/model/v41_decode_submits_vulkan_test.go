//go:build vulkan && (windows || linux) && cgo

// Run serially from the repository root with the coupled shader bundle:
// FAK_VULKAN_SPIRV="$PWD/internal/compute/spirv" FAK_VULKAN_REQUIRE_DEVICE=1 FAK_VULKAN_DISPATCH_PROFILE=1 go test -tags vulkan ./internal/model -run '^TestV41DecodeSubmits$' -count=1 -v -timeout=30m
// Add FAK_V41_DECODE_SUBMITS_CONTEXTS=256,512 for longer contexts (minutes per fixture).
// Synthetic fixtures; per-step counters and wall time are logged, never bounded.
// H2D classes are native-accounted MemoryClass uploads, not tensor roles or
// completed-device traffic. Physical completed-run evidence still requires a
// successful enclosing quiescent execution window.

package model

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// v41DecodeSubmitsContextsEnv overrides the context lengths, e.g. "256,512".
// Cold prefill cost grows faster than linearly with context on the per-op
// device seams, so the default stays short enough for one leased run.
const v41DecodeSubmitsContextsEnv = "FAK_V41_DECODE_SUBMITS_CONTEXTS"

func v41DecodeSubmitsContexts(t *testing.T) []int {
	t.Helper()
	raw := os.Getenv(v41DecodeSubmitsContextsEnv)
	if raw == "" {
		return []int{4, 64}
	}
	var out []int
	for _, field := range strings.Split(raw, ",") {
		n, err := strconv.Atoi(strings.TrimSpace(field))
		if err != nil || n < 1 {
			t.Fatalf("%s=%q: want comma-separated positive token counts", v41DecodeSubmitsContextsEnv, raw)
		}
		out = append(out, n)
	}
	return out
}

// TestV41DecodeSubmits is the device half of the #13820/#13821 witness. Per
// decode step it logs DispatchSubmits, ComputeDispatches, H2D/D2H bytes and
// counts and wall time, split from prefill, at several context lengths. It
// asserts counter presence and exact split accounting only.
// fak-test:justify why=integration when=changed:internal/model/**
// fak-test:runtime slow est=30s lane=optin
func TestV41DecodeSubmits(t *testing.T) {
	if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") != "1" {
		t.Skip("physical per-step submit witness requires explicit device opt-in")
	}
	if os.Getenv("FAK_VULKAN_DISPATCH_PROFILE") != "1" {
		t.Skip("set FAK_VULKAN_DISPATCH_PROFILE=1 before process start; dispatch and submit counters stay zero without it")
	}
	be, ok := compute.Lookup("vulkan")
	if !ok || be == nil || be.Name() != "vulkan" || !be.Caps().DeviceMemory {
		t.Fatal("required actual Vulkan device unavailable")
	}
	snap, available, err := compute.CaptureBackendExecutionSnapshot(be)
	if err != nil || !available {
		t.Fatalf("backend execution snapshot unavailable: available=%t err=%v", available, err)
	}
	if !snap.H2DAttributionObserved {
		t.Fatal("startup-profiled native witness requires observed H2D attribution")
	}
	if expected := os.Getenv("FAK_VULKAN_EXPECT_DEVICE"); expected != "" && snap.Identity.Device != expected {
		t.Fatalf("physical device=%q want=%q", snap.Identity.Device, expected)
	}
	t.Logf("Vulkan device: %s driver=%s runtime=%s", snap.Identity.Device, snap.Identity.Driver, snap.Identity.Runtime)

	contexts := v41DecodeSubmitsContexts(t)
	var summary []string
	defer func() {
		for _, line := range summary {
			t.Log(line)
		}
	}()
	for _, fx := range append(v41DecodeSubmitsFixtures(), v41DecodeSubmitsPinnedFixtures()...) {
		t.Run(fx.name, func(t *testing.T) {
			for _, ctx := range contexts {
				t.Run(fmt.Sprintf("ctx_%d", ctx), func(t *testing.T) {
					m := fx.build(t)
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
					bound, host := v41DeviceParitySeams(s.v41State())
					if len(bound) == 0 {
						t.Fatal("Vulkan session bound no V4.1 device seam; the counters would describe a host decode")
					}
					t.Logf("device seams bound=%v host=%v", bound, host)

					label := fmt.Sprintf("fixture=%s layers=%d ctx=%d", fx.name, m.Cfg.NumLayers, ctx)
					tr := v41DecodeSubmitsMeasure(t, label, s, be, v41DecodeSubmitsPrompt(ctx, m.Cfg.VocabSize), v41DecodeSubmitsSteps)
					v41DecodeSubmitsCheck(t, tr)

					wall, c, n := v41DecodeSubmitsSteady(tr)
					pre, last := tr.windows[0], tr.windows[len(tr.windows)-1]
					incremental := 0
					for _, w := range tr.windows {
						if w.route == "incremental" {
							incremental++
						}
					}
					summary = append(summary, fmt.Sprintf("v41_decode_submits_summary fixture=%s layers=%d ctx=%d prefill_wall_ms=%.1f prefill_submits=%d prefill_dispatches=%d prefill_h2d_bytes=%d prefill_d2h_bytes=%d steady_steps=%d step_wall_ms=%.3f step_submits=%.1f step_dispatches=%.1f step_h2d_bytes=%.0f step_h2d_count=%.1f step_d2h_bytes=%.0f step_d2h_count=%.1f incremental_steps=%d/%d decode_live_bytes_delta=%d host_fallback=%t",
						fx.name, m.Cfg.NumLayers, ctx, float64(pre.wall.Microseconds())/1000,
						pre.delta.Counters.DispatchSubmits, pre.delta.Counters.ComputeDispatches, pre.delta.Counters.H2DBytes, pre.delta.Counters.D2HBytes,
						n, wall, c[0], c[1], c[2], c[3], c[4], c[5], incremental, len(tr.windows)-1,
						int64(last.liveBytes)-int64(pre.liveBytes), s.HostFallbackObserved()))
				})
			}
		})
	}
}

// v41DecodeSubmitsPinnedFixtures adds the pinned Q2_K/Q3_K slate, whose routed
// experts run on the device; the F32 fixtures above keep them on the host.
func v41DecodeSubmitsPinnedFixtures() []v41DecodeSubmitsFixture {
	var out []v41DecodeSubmitsFixture
	for _, compressed := range []bool{false, true} {
		out = append(out, v41DecodeSubmitsFixture{fmt.Sprintf("pinned_q2k_q3k_layers3_compressed_%t", compressed), func(t *testing.T) *Model {
			return v41PinnedSlateFixture(t, compressed)
		}})
	}
	return out
}

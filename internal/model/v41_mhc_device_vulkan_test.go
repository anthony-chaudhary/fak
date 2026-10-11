//go:build vulkan && (windows || linux) && cgo

// Run from the repository root with the coupled shader bundle:
// FAK_VULKAN_SPIRV="$PWD/internal/compute/spirv" FAK_VULKAN_REQUIRE_DEVICE=1 go test -tags vulkan ./internal/model -run '^TestV41MHCDeviceVulkan$' -count=1 -timeout=10m
// The synthetic payload witnesses the session seam on a physical GPU, not a checkpoint.

package model

import (
	"fmt"
	"math"
	"os"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// fak-test:justify why=integration when=changed:internal/model/**
// fak-test:runtime slow est=30s lane=optin
func TestV41MHCDeviceVulkan(t *testing.T) {
	if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") != "1" {
		t.Skip("physical mHC seam witness requires explicit device opt-in")
	}
	be, ok := compute.Lookup("vulkan")
	if !ok || be == nil || be.Name() != "vulkan" || !be.Caps().DeviceMemory {
		t.Fatal("required actual Vulkan device unavailable")
	}
	t.Logf("Vulkan device: %s", be.Tier())
	if op, ok := be.(compute.V41MHCBackend); !ok || !op.SupportsV41MHC() {
		t.Fatal("registered Vulkan build lacks the V4.1 mHC module (rebuild the coupled V6/63 bundle)")
	}

	t.Run("seam ops at production width", func(t *testing.T) {
		m := v41FullStepPatchedNorms(t)
		s, err := m.NewBackendSessionChecked(be)
		if err != nil {
			t.Fatal(err)
		}
		defer s.Close()
		d := s.v41MHCDeviceSeam()
		if d == nil || d.op == nil || d.refusal != nil {
			t.Fatal("Vulkan session did not select the mHC device seam")
		}
		const H = 4096
		x := make([]float32, H)
		streams := make([][]float32, 4)
		for i := range x {
			x[i] = float32((i*37)%101-50) * 0.0213
		}
		x[0], x[1] = math.Float32frombits(0x3f808000), math.Float32frombits(0x3f818000) // BF16 ties
		for h := range streams {
			streams[h] = make([]float32, H)
			for i := range streams[h] {
				streams[h][i] = v41RoundBF16(float32((i*13+h*7)%89-44) * 0.0377)
			}
		}
		wantInit, err := v41FullInitialStreams(x)
		if err != nil {
			t.Fatal(err)
		}
		gotInit, err := d.initialStreams(x)
		if err != nil {
			t.Fatal(err)
		}
		v41MHCVulkanBits(t, "initial", gotInit, wantInit)

		mixes, scale, base := make([]float32, 24), []float32{0.75, 1.25, 1.5}, make([]float32, 24)
		for i := range mixes {
			mixes[i] = float32((i*37)%41-20) * 0.173
			base[i] = float32(i%5-2) * 0.0625
		}
		wantMix, err := v41MHCSplit(mixes, scale, base, 4, hcItersOrDefault(m.Cfg), hcEpsOrDefault(m.Cfg))
		if err != nil {
			t.Fatal(err)
		}
		gotMix, err := d.split(0, mixes, scale, base, 4, hcItersOrDefault(m.Cfg), hcEpsOrDefault(m.Cfg))
		if err != nil {
			t.Fatal(err)
		}
		worst := 0
		for _, pair := range [][2][]float32{{gotMix.pre, wantMix.pre}, {gotMix.post, wantMix.post}, {gotMix.comb, wantMix.comb}} {
			for i := range pair[1] {
				ulps := int(math.Float32bits(pair[0][i])) - int(math.Float32bits(pair[1][i]))
				if ulps < 0 {
					ulps = -ulps
				}
				worst = max(worst, ulps)
			}
		}
		// Bound and basis: compute.v41MHCSplitULPBound (TestVulkanV41MHC).
		if worst > 32 {
			t.Fatalf("split coefficients differ by %d ulps (> 32)", worst)
		}
		t.Logf("split max ulp=%d", worst)

		// Pre and post consume the host coefficients so their bit witness is
		// independent of the split tolerance.
		wantPre, err := v41MHCPreBF16(0, streams, wantMix.pre)
		if err != nil {
			t.Fatal(err)
		}
		gotPre, err := d.preBF16(0, streams, wantMix.pre)
		if err != nil {
			t.Fatal(err)
		}
		v41MHCVulkanBits(t, "pre", [][]float32{gotPre}, [][]float32{wantPre})
		wantPost, err := v41MHCPostBF16(0, x, streams, wantMix)
		if err != nil {
			t.Fatal(err)
		}
		gotPost, err := d.postBF16(0, x, streams, wantMix)
		if err != nil {
			t.Fatal(err)
		}
		v41MHCVulkanBits(t, "post", gotPost, wantPost)
	})

	t.Run("session forward isolates the mHC seam", func(t *testing.T) {
		m, controlModel := v41FullStepPatchedNorms(t), v41FullStepPatchedNorms(t)
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
		if s.v41State().mhcDevice == nil {
			t.Fatal("device session did not bind the mHC seam")
		}
		// Same backend and every other seam; only the mHC ops stay on the host.
		control.v41State().mhcDevice = nil
		ids := []int{1, 2}
		got, want := s.Prefill(ids), control.Prefill(ids)
		diff := v41MHCVulkanMaxDiff(t, got, want)
		gotStep, wantStep := s.Step(3), control.Step(3)
		stepDiff := v41MHCVulkanMaxDiff(t, gotStep, wantStep)
		t.Logf("device-vs-host mHC logits max diff: prefill=%g step=%g", diff, stepDiff)
		// Bit-exact pre/post leave only the split's few-ulp coefficients, which
		// reach the logits through BF16 stream publications (TICKET-02d class).
		// Measured 0 on RX 7600 (2026-10-10); the bound is the session parity
		// tolerance the other V4.1 seams use.
		if diff > 1e-4 || stepDiff > 1e-4 {
			t.Fatalf("mHC seam moved logits by prefill=%g step=%g (> 1e-4)", diff, stepDiff)
		}
	})
}

func v41MHCVulkanBits(t *testing.T, name string, got, want [][]float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d rows want %d", name, len(got), len(want))
	}
	for r := range want {
		if len(got[r]) != len(want[r]) {
			t.Fatalf("%s[%d]: %d values want %d", name, r, len(got[r]), len(want[r]))
		}
		for i := range want[r] {
			if got[r][i] != want[r][i] {
				t.Fatalf("%s[%d][%d]: got %g want %g", name, r, i, got[r][i], want[r][i])
			}
		}
	}
}

func v41MHCVulkanMaxDiff(t *testing.T, got, want []float32) float64 {
	t.Helper()
	if len(got) != len(want) || len(want) == 0 {
		t.Fatalf("logit widths %d/%d", len(got), len(want))
	}
	worst := 0.0
	for i := range want {
		if !finite32(got[i]) || !finite32(want[i]) {
			panic(fmt.Sprintf("non-finite logit %d: %g/%g", i, got[i], want[i]))
		}
		worst = max(worst, math.Abs(float64(got[i]-want[i])))
	}
	return worst
}

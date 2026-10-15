//go:build vulkan && (windows || linux) && cgo

package model

import (
	"fmt"
	"os"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// TestV41DecodeResidentH2DVulkan is the device half of the #13821 KV-mirror
// witness: with the mirror on, steady per-step H2D bytes are equal at ctx 4
// and 64, and every logit equals the per-step upload path bit for bit.
// fak-test:justify why=integration when=changed:internal/model/**
// fak-test:runtime slow est=30s lane=optin
func TestV41DecodeResidentH2DVulkan(t *testing.T) {
	if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") != "1" || os.Getenv("FAK_VULKAN_DISPATCH_PROFILE") != "1" {
		t.Skip("physical KV-mirror witness requires FAK_VULKAN_REQUIRE_DEVICE=1 and FAK_VULKAN_DISPATCH_PROFILE=1")
	}
	be, ok := compute.Lookup("vulkan")
	if !ok || be == nil || be.Name() != "vulkan" || !be.Caps().DeviceMemory {
		t.Fatal("required actual Vulkan device unavailable")
	}
	if expected := os.Getenv("FAK_VULKAN_EXPECT_DEVICE"); expected != "" {
		snap, available, err := compute.CaptureBackendExecutionSnapshot(be)
		if err != nil || !available || snap.Identity.Device != expected {
			t.Fatalf("physical device=%q want=%q (available=%t err=%v)", snap.Identity.Device, expected, available, err)
		}
	}
	decode := func(t *testing.T, compressed bool, ctx int, disabled bool) v41DecodeSubmitsTrace {
		prior := v41ResidentKVDisabled
		v41ResidentKVDisabled = disabled
		defer func() { v41ResidentKVDisabled = prior }()
		m := v41DecodeSubmitsLayeredFixture(t, 1, compressed)
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
		if s.v41State().sharedAttentionFunc() == nil {
			t.Fatal("Vulkan session bound no shared attention")
		}
		label := fmt.Sprintf("resident compressed=%t ctx=%d mirror=%t", compressed, ctx, !disabled)
		tr := v41DecodeSubmitsMeasure(t, label, s, be, v41DecodeSubmitsPrompt(ctx, m.Cfg.VocabSize), v41DecodeSubmitsSteps)
		v41DecodeSubmitsCheck(t, tr)
		if s.HostFallbackObserved() {
			t.Error("session observed a host fallback")
		}
		return tr
	}
	stepH2D := func(tr v41DecodeSubmitsTrace) []uint64 {
		var out []uint64
		for _, w := range tr.windows {
			if w.phase == "decode" && w.step > 0 {
				out = append(out, w.delta.Counters.H2DBytes)
			}
		}
		return out
	}
	for _, compressed := range []bool{false, true} {
		t.Run(fmt.Sprintf("compressed_%t", compressed), func(t *testing.T) {
			steady := map[int][]uint64{}
			for _, ctx := range []int{4, 64} {
				on, off := decode(t, compressed, ctx, false), decode(t, compressed, ctx, true)
				for i := range on.windows {
					if !reflect.DeepEqual(on.windows[i].logits, off.windows[i].logits) {
						t.Fatalf("ctx=%d %s step %d logits differ between the mirror and the per-step upload path", ctx, on.windows[i].phase, on.windows[i].step)
					}
				}
				steady[ctx] = stepH2D(on)
				t.Logf("v41_resident_kv_summary compressed=%t ctx=%d steady_h2d_bytes mirror=%v upload=%v", compressed, ctx, steady[ctx], stepH2D(off))
			}
			if !reflect.DeepEqual(steady[4], steady[64]) {
				t.Fatalf("steady mirror H2D bytes depend on context: ctx4=%v ctx64=%v", steady[4], steady[64])
			}
		})
	}
}

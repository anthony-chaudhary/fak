//go:build vulkan && (windows || linux) && cgo

package model

import (
	"os"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// On the physical Vulkan session a warm step takes every named site's names
// from the table, keeps every store lookup, allocates less than the layerName
// reference and returns the same logits bit for bit.
// fak-test:justify why=integration when=changed:internal/model/**
// fak-test:runtime slow est=30s lane=optin
func TestV41LayerNamesVulkan(t *testing.T) {
	if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") != "1" {
		t.Skip("physical layer-name witness requires explicit device opt-in")
	}
	be, ok := compute.Lookup("vulkan")
	if !ok || be == nil || be.Name() != "vulkan" || !be.Caps().DeviceMemory {
		t.Fatal("required actual Vulkan device unavailable")
	}
	const steps = 5
	for _, compressed := range []bool{false, true} {
		name := "plain"
		if compressed {
			name = "compressed"
		}
		t.Run(name, func(t *testing.T) {
			open := func(off bool) (*Model, *Session) {
				m := v41PinnedSlateFixture(t, compressed)
				m.v41NamesOff = off
				s, err := m.NewBackendSessionChecked(be)
				if err != nil {
					t.Fatal(err)
				}
				return m, s
			}
			memoModel, memo := open(false)
			defer memoModel.CloseWeights()
			defer memo.Close()
			refModel, ref := open(true)
			defer refModel.CloseWeights()
			defer ref.Close()

			prompt := v41DecodeSubmitsPrompt(4, memoModel.Cfg.VocabSize)
			got, want := memo.Prefill(prompt), ref.Prefill(prompt)
			for i := 0; i < steps; i++ {
				v41AssertLogitsBitEqual(t, i, got, want)
				next := argmaxF32(want)
				if i < 2 {
					got, want = memo.Step(next), ref.Step(next)
					continue
				}
				v41AssertWarmNamedStep(t, i, memoModel, func() { got = memo.Step(next) }, func() { want = ref.Step(next) })
			}
			v41AssertLogitsBitEqual(t, steps, got, want)
			v41AssertWarmStepAllocs(t, memo, ref, argmaxF32(want), true)
		})
	}
}

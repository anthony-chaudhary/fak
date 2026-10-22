//go:build vulkan && (windows || linux) && cgo

package model

import (
	"math"
	"os"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/computetrace"
)

// fak-test:justify why=integration when=changed:internal/model/**
// fak-test:runtime slow est=10s lane=optin
// The same synthetic V4.1 forward on one physical GPU with the mHC prenorm on
// (default) and off (FAK_V41_MHC_PRENORM=0): every prefill and step logit
// must keep its bits, since only the projection's dispatch layout moved. The
// on arm must prove the prenorm ran on the session's own mix weights: the
// device admits the uploaded weight, and every mix projection's kernel record
// names the prenorm module with that weight's dtype and shape. The off arm
// records none.
func TestV41MHCPrenormVulkanForwardBitIdentical(t *testing.T) {
	if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") != "1" {
		t.Skip("physical mHC prenorm witness requires explicit device opt-in")
	}
	be, ok := compute.Lookup("vulkan")
	if !ok || be == nil || be.Name() != "vulkan" || !be.Caps().DeviceMemory {
		t.Fatal("required actual Vulkan device unavailable")
	}
	op, ok := be.(compute.V41MHCPrenormBackend)
	if !ok || !op.V41MHCPrenormAvailable() {
		t.Fatal("registered Vulkan build does not serve the mHC prenorm")
	}
	var events []computetrace.Event
	computetrace.SetObserver(func(e computetrace.Event) {
		if e.Phase == "kernel" && len(e.Shapes) == 2 {
			events = append(events, e)
		}
	})
	t.Cleanup(func() { computetrace.SetObserver(nil) })
	type arm struct {
		prefill, step []float32
		mixes         []compute.Tensor
		prenorm       int
		admitted      bool
	}
	run := func(t *testing.T, off bool) arm {
		env := ""
		if off {
			env = "0"
		}
		t.Setenv(v41MHCPrenormEnv, env)
		events = events[:0]
		m := v41FullStepPatchedNorms(t)
		H := m.Cfg.HiddenSize
		s, err := m.NewBackendSessionChecked(be)
		if err != nil {
			t.Fatal(err)
		}
		defer s.Close()
		if s.v41State().mhcDevice == nil {
			t.Fatal("device session did not bind the mHC seam")
		}
		var a arm
		a.prefill = append([]float32(nil), s.Prefill([]int{1, 2})...)
		a.step = append([]float32(nil), s.Step(3)...)
		a.admitted = true
		for l := 0; l < m.Cfg.NumLayers; l++ {
			for _, leaf := range []string{"mhc.mixes.weight", "mhc.ffn_mixes.weight"} {
				_, transposed, ok := m.v41MHCWeightLayoutNamed(l, leaf)
				if !ok {
					t.Fatalf("layer %d %s has no admitted mix layout", l, leaf)
				}
				w, err := s.v41MHCMixWeight(l, leaf, H, transposed)
				if err != nil || w.Buf() == nil {
					t.Fatalf("layer %d %s: session mix weight unavailable: %v", l, leaf, err)
				}
				a.mixes = append(a.mixes, w)
				a.admitted = a.admitted && op.SupportsV41MHCPrenormWeight(w, H)
			}
		}
		// Only [24, 4H] mix projections; other kernels are not this seam's.
		mixEvents := events[:0]
		for _, e := range events {
			if reflect.DeepEqual(e.Shapes[0], []int{compute.V41MHCCoeffWidth, compute.V41MHCStreams * H}) {
				mixEvents = append(mixEvents, e)
			}
		}
		events = mixEvents
		for _, e := range events {
			if e.Kernel == e.WeightDType+"_v41_mhc_prenorm" {
				a.prenorm++
			}
		}
		// Two prompt rows, then one step row, through every layer's attention
		// and FFN mix: 4 projections per layer, each one record.
		calls := 4 * m.Cfg.NumLayers
		if !off {
			if !a.admitted || a.prenorm != calls || len(events) != calls {
				t.Fatalf("prenorm arm: real weights admitted=%t, %d prenorm of %d mix kernel records, want %d prenorm", a.admitted, a.prenorm, len(events), calls)
			}
			for _, e := range events {
				found := false
				for _, w := range a.mixes {
					found = found || (e.WeightDType == w.Dtype.String() && reflect.DeepEqual(e.Shapes[0], w.Shape))
				}
				if !found {
					t.Fatalf("prenorm record %s %v does not describe a session mix weight", e.WeightDType, e.Shapes[0])
				}
			}
		} else if a.prenorm != 0 {
			t.Fatalf("FAK_V41_MHC_PRENORM=0 still recorded %d prenorm kernels", a.prenorm)
		}
		t.Logf("off=%t: %d session mix weights (%v), admitted=%t, mix kernel records=%d, prenorm=%d", off, len(a.mixes), a.mixes[0].Dtype, a.admitted, len(events), a.prenorm)
		return a
	}
	var on, off arm
	t.Run("prenorm", func(t *testing.T) { on = run(t, false) })
	t.Run("matmul", func(t *testing.T) { off = run(t, true) })
	for _, pair := range []struct {
		name      string
		got, want []float32
	}{{"prefill", on.prefill, off.prefill}, {"step", on.step, off.step}} {
		if len(pair.got) == 0 || len(pair.got) != len(pair.want) {
			t.Fatalf("%s: %d logits want %d", pair.name, len(pair.got), len(pair.want))
		}
		for i := range pair.want {
			if math.Float32bits(pair.got[i]) != math.Float32bits(pair.want[i]) {
				t.Fatalf("%s logit %d: prenorm %g (%#08x) != matmul %g (%#08x)", pair.name, i, pair.got[i], math.Float32bits(pair.got[i]), pair.want[i], math.Float32bits(pair.want[i]))
			}
		}
	}
}

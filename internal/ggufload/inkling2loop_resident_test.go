package ggufload

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

func TestInkling2SlotLayerMapsLoopPassesToCoreLayers(t *testing.T) {
	hp := model.Inkling2LoopHParams{Layers: 24, CoreStart: 6, CoreEnd: 18, Passes: 3}
	cases := map[int]int{0: 0, 5: 5, 6: 6, 17: 17, 23: 23, 24: 6, 35: 17, 36: 6, 47: 17}
	for slot, want := range cases {
		if got := inkling2SlotLayer(hp, slot); got != want {
			t.Errorf("slot %d -> layer %d, want %d", slot, got, want)
		}
	}
}

func TestInkling2IndexedExpertCoversCheckpointExpertTypes(t *testing.T) {
	for _, dt := range []compute.Dtype{compute.NVFP4, compute.IQ2_XS, compute.IQ2_S, compute.IQ3_XXS, compute.IQ3_S, compute.IQ4_NL, compute.Q3_K} {
		if !inkling2IndexedExpert(dt) {
			t.Errorf("%s has no indexed expert kernel", dt)
		}
	}
	if inkling2IndexedExpert(compute.Q8_0) {
		t.Error("Q8_0 experts must take the host-routed fallback")
	}
}

func TestInkling2ResidentNeedsDeviceOps(t *testing.T) {
	be, ok := compute.Lookup("cpu-ref")
	if !ok {
		t.Skip("cpu-ref backend not registered")
	}
	m := &Inkling2LoopModel{Weights: &model.Inkling2LoopWeights{Layers: make([]model.Inkling2LoopLayerWeights, 1)}}
	if inkling2ResidentEligible(m, be) {
		t.Fatal("a backend without inkling2 device ops must keep the host-driven forward")
	}
	if _, err := newInkling2Resident(m, be); err == nil {
		t.Fatal("newInkling2Resident accepted a backend without inkling2 device ops")
	}
}

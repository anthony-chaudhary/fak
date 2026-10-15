package model

import "testing"

// Warm name lookups must not retain the caller's temporary expert suffix.
// fak-test:runtime fast est=10ms lane=default
func TestLayerNameWarmExpertAllocations(t *testing.T) {
	layerNames.mu.Lock()
	previous := layerNames.m
	layerNames.m = nil
	layerNames.mu.Unlock()
	defer func() {
		layerNames.mu.Lock()
		layerNames.m = previous
		layerNames.mu.Unlock()
	}()

	for _, expert := range []int{1, 99, 100, 383} {
		stem := "ffn.experts." + itoa(expert)
		want1 := layerName(39, stem+".w1.weight")
		want3 := layerName(39, stem+".w3.weight")
		want2 := layerName(39, stem+".w2.weight")
		allocs := testing.AllocsPerRun(100, func() {
			stem := "ffn.experts." + itoa(expert)
			if layerName(39, stem+".w1.weight") != want1 ||
				layerName(39, stem+".w3.weight") != want3 ||
				layerName(39, stem+".w2.weight") != want2 {
				panic("warm layer name changed")
			}
		})
		if allocs != 0 {
			t.Fatalf("expert %d warm name lookups allocated %g objects, want zero", expert, allocs)
		}
	}
}

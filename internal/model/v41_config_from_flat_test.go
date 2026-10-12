package model

import "testing"

func TestV41ConfigFromFlatCarriesCompressRopeTheta(t *testing.T) {
	cfg := Config{CompressRatios: []int{0, 0, 4}, CompressRopeTheta: 160000, IndexerTypes: []string{"dense", "full", "shared"}}
	d41 := v41ConfigFromFlat(cfg)
	if d41.CompressRopeTheta != 160000 {
		t.Fatalf("CompressRopeTheta = %g, want 160000", d41.CompressRopeTheta)
	}
	cfg.DeepSeekV41 = d41
	if got := v41RopeThetaForLayer(cfg, 2); got != 160000 {
		t.Fatalf("compressed layer theta = %g, want 160000", got)
	}
}

package model

import (
	"math"
	"reflect"
	"testing"
)

// The configuration is the retained official V4.1 fixture; its rotary fields
// also match the pinned GGUF header. The independent scalar oracle transcribes
// model.py:369-389,680-696 at dba1be0a40aa45a94ad051997016db3960a90277.
// This witnesses layer policy and the YaRN equation at host table precision,
// not PyTorch bitwise parity, tensor loading, or hardware execution.
// fak-test:runtime fast est=10ms lane=default
func TestV41RopePinnedLayerRegimes(t *testing.T) {
	t.Parallel()
	_, cfg := readDeepSeekV41Config(t)
	if cfg.QKRopeHeadDim != 64 || cfg.HeadDim != 512 || cfg.RopeTheta != 10000 ||
		cfg.DeepSeekV41.CompressRopeTheta != 160000 || cfg.RopeFactor != 16 || cfg.RopeOrigContext != 65536 {
		t.Fatal("rotary fixture no longer matches the pinned header")
	}
	check := func(label string, gotCos, gotSin, wantCos, wantSin []float32) {
		t.Helper()
		if len(gotCos) != 32 || len(gotSin) != 32 || len(wantCos) != 32 || len(wantSin) != 32 {
			t.Fatalf("%s: table width does not match rotary_dim/2", label)
		}
		for j := range gotCos {
			if !Finite32(gotCos[j]) || !Finite32(gotSin[j]) ||
				math.Abs(float64(gotCos[j]-wantCos[j])) > 1e-6 || math.Abs(float64(gotSin[j]-wantSin[j])) > 1e-6 {
				t.Fatalf("%s pair=%d: got (%g,%g), want (%g,%g)", label, j, gotCos[j], gotSin[j], wantCos[j], wantSin[j])
			}
			if norm := float64(gotCos[j])*float64(gotCos[j]) + float64(gotSin[j])*float64(gotSin[j]); math.Abs(norm-1) > 2e-7 {
				t.Fatalf("%s pair=%d: non-unit rotation magnitude squared=%g", label, j, norm)
			}
		}
	}
	for _, layer := range []int{0, 1, 2, 19, 20, 39, 40, 42} {
		for _, pos := range []int{0, 1, 7, 65536, 1048575} {
			gotCos, gotSin := v41RopeTableForLayer(cfg, layer, pos)
			wantCos, wantSin := v41OracleRopeTable(t, cfg, layer, pos)
			check("pinned layer", gotCos, gotSin, wantCos, wantSin)
			if pos == 0 {
				for j := range gotCos {
					if gotCos[j] != 1 || gotSin[j] != 0 {
						t.Fatalf("layer=%d pair=%d: position zero must be identity", layer, j)
					}
				}
			}
		}
	}

	// Explicit controls distinguish every policy axis, independently of the oracle:
	// ratio 0 has bare base theta; ratios 1 and 2 share compressed theta and YaRN.
	plainCos, plainSin := v41RopeTableForLayer(cfg, 0, 7)
	compressedCos, compressedSin := v41RopeTableForLayer(cfg, 2, 7)
	ratioOneCos, ratioOneSin := v41RopeTableForLayer(cfg, 20, 7)
	check("ratio one", ratioOneCos, ratioOneSin, compressedCos, compressedSin)
	if reflect.DeepEqual(plainCos, compressedCos) && reflect.DeepEqual(plainSin, compressedSin) {
		t.Fatal("base/compressed layer control did not discriminate")
	}
	for _, j := range []int{0, 15, 20, 25, 31} {
		bare := 7 / math.Pow(10000, float64(2*j)/64)
		if plainCos[j] != float32(math.Cos(bare)) || plainSin[j] != float32(math.Sin(bare)) {
			t.Fatalf("ratio zero pair=%d did not disable YaRN", j)
		}
		// The pinned compressed-base correction bounds are exactly [15,25].
		ramp := math.Max(0, math.Min(1, float64(j-15)/10))
		freq := 1 / math.Pow(160000, float64(2*j)/64)
		angle := 7 * (freq*(1-ramp) + (freq/16)*ramp)
		if math.Abs(float64(compressedCos[j])-math.Cos(angle)) > 1e-6 ||
			math.Abs(float64(compressedSin[j])-math.Sin(angle)) > 1e-6 {
			t.Fatalf("compressed pair=%d missed the unscaled/interpolated/scaled band", j)
		}
	}
	bareCompressed := 7 / math.Pow(160000, float64(40)/64)
	fullyScaled := bareCompressed / 16
	if math.Abs(float64(compressedSin[20])-math.Sin(bareCompressed)) < 1e-5 ||
		math.Abs(float64(compressedSin[20])-math.Sin(fullyScaled)) < 1e-5 {
		t.Fatal("interpolation control cannot reject absent YaRN or uniform scaling")
	}

	// The shared helper must receive the rotary width and selected layer base.
	// Nested HF-only amplitude/base/truncation fields must not replace V4.1 policy,
	// and table construction must not mutate the caller's parameter storage.
	noTruncation := false
	rp := RopeScaling{Type: "yarn", Factor: 16, OriginalMaxPositionEmbeddings: 65536,
		BetaFast: 32, BetaSlow: 1, RopeTheta: 1e9, AttentionFactor: 3, Truncate: &noTruncation}
	for _, shape := range []string{"flat", "classic", "default"} {
		variant := cfg
		variant.LongRope, variant.RopeParameters = nil, nil
		variant.PartialRotaryFactor = 0.5
		switch shape {
		case "classic":
			block := rp
			variant.LongRope = &block
		case "default":
			variant.RopeParameters = RopeParameters{"default": rp}
		}
		gotCos, gotSin := v41RopeTableForLayer(variant, 2, 7)
		check(shape, gotCos, gotSin, compressedCos, compressedSin)
		if variant.LongRope != nil && !reflect.DeepEqual(*variant.LongRope, rp) {
			t.Fatal("table construction mutated classic parameters")
		}
		if variant.RopeParameters != nil && !reflect.DeepEqual(variant.RopeParameters["default"], rp) {
			t.Fatal("table construction mutated nested parameters")
		}
	}
}

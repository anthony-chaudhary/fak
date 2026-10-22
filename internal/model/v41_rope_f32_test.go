package model

import (
	"fmt"
	"math"
	"testing"
)

// v41RopeF32InvFreq emulates DeepInfer's float32 inverse-frequency build for
// one V4.1 layer: float32 pow and reciprocal, then the YaRN blend in its
// upstream operand order with every step rounded to float32. It uses correctly
// rounded host pow and makes no bitwise CUDA powf or torch parity claim.
// Production keeps its float64 table; this unwired oracle only measures the gap.
func v41RopeF32InvFreq(cfg Config, layer int) []float32 {
	dim := cfg.QKRopeHeadDim
	theta := v41RopeThetaForLayer(cfg, layer)
	inv := make([]float32, dim/2)
	pos := make([]float32, dim/2)
	for j := range inv {
		x := float32(float32(2*j) / float32(dim))
		pos[j] = float32(math.Pow(float64(float32(theta)), float64(x)))
		inv[j] = float32(1) / pos[j]
	}
	if v41CompressRatioAt(cfg, layer) <= 0 || cfg.RopeScaling != "yarn" {
		return inv
	}
	rp := cfg.effectiveRopeParameters()
	factor := rp.Factor
	if factor == 0 {
		factor = cfg.RopeFactor
	}
	origCtx := rp.OriginalMaxPositionEmbeddings
	if origCtx == 0 {
		origCtx = cfg.RopeOrigContext
	}
	betaFast, betaSlow := rp.BetaFast, rp.BetaSlow
	if betaFast == 0 {
		betaFast = 32
	}
	if betaSlow == 0 {
		betaSlow = 1
	}
	if factor == 0 || origCtx == 0 {
		return inv
	}
	low, high := yarnCorrectionRange(betaFast, betaSlow, dim, theta, float64(origCtx), true)
	if low == high {
		high += 0.001
	}
	for j := range inv {
		ramp := min(max(float32(float32(float32(j)-float32(low))/float32(high-low)), 0), 1)
		// Upstream blends with mask = 1-ramp and weights interpolation by 1-mask;
		// in float32 that is not the same value as ramp.
		mask := float32(1) - ramp
		interp := float32(1) / float32(float32(factor)*pos[j])
		inv[j] = float32(float32(interp*float32(1-mask)) + float32(inv[j]*mask))
	}
	return inv
}

func v41RopeF32Oracle(cfg Config, layer, p int) (cos, sin []float32) {
	inv := v41RopeF32InvFreq(cfg, layer)
	cos, sin = make([]float32, len(inv)), make([]float32, len(inv))
	for j, f := range inv {
		angle := float32(float32(p) * f)
		cos[j] = float32(math.Cos(float64(angle)))
		sin[j] = float32(math.Sin(float64(angle)))
	}
	return cos, sin
}

// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime fast est=5ms lane=default
func TestV41RopeF32DiscriminatingAnchors(t *testing.T) {
	t.Parallel()
	_, cfg := readDeepSeekV41Config(t)
	if cfg.QKRopeHeadDim != 64 || cfg.RopeTheta != 10000 || v41CompressRatioAt(cfg, 0) != 0 {
		t.Fatal("rotary fixture no longer matches the pinned header")
	}
	for _, layer := range []int{0, 1, 2, 19, 20, 39, 40, 42} {
		cos, sin := v41RopeF32Oracle(cfg, layer, 0)
		for j := range cos {
			if cos[j] != 1 || sin[j] != 0 {
				t.Fatalf("layer=%d pair=%d: position zero must be identity", layer, j)
			}
		}
	}
	const p = 1 << 20
	inv := v41RopeF32InvFreq(cfg, 0)
	if inv[0] != 1 || math.Float32bits(inv[16]) != 0x3c23d70a {
		t.Fatalf("float32 inverse frequencies drifted: inv[0]=%g inv[16]=%#08x", inv[0], math.Float32bits(inv[16]))
	}
	cos, sin := v41RopeF32Oracle(cfg, 0, p)
	prodCos, prodSin := v41RopeTableForLayer(cfg, 0, p)
	if cos[0] != prodCos[0] || sin[0] != prodSin[0] || cos[0] != float32(math.Cos(p)) {
		t.Fatal("unit-frequency slot must match the float64 table and the exact angle")
	}
	for _, tc := range []struct {
		slot  int
		angle float64
	}{{16, 10485.759765625}, {8, 104857.6015625}} {
		if cos[tc.slot] != float32(math.Cos(tc.angle)) || sin[tc.slot] != float32(math.Sin(tc.angle)) {
			t.Fatalf("slot %d is not the float32-rounded angle %g", tc.slot, tc.angle)
		}
	}
	if prodCos[16] != float32(math.Cos(p/100.0)) || prodCos[16] == cos[16] {
		t.Fatal("production table must keep its float64 angle and differ from the float32 emulation")
	}
}

// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime fast est=5ms lane=default
func TestV41RopeF32YaRNBandStructure(t *testing.T) {
	t.Parallel()
	_, cfg := readDeepSeekV41Config(t)
	bareCfg := cfg
	bareCfg.RopeScaling = ""
	for _, layer := range []int{2, 20} {
		yarn, bare := v41RopeF32InvFreq(cfg, layer), v41RopeF32InvFreq(bareCfg, layer)
		for j := range yarn {
			switch {
			case j <= 15:
				if yarn[j] != bare[j] {
					t.Fatalf("layer=%d pair=%d: unscaled band changed", layer, j)
				}
			case j >= 25:
				if float32(yarn[j]*16) != bare[j] {
					t.Fatalf("layer=%d pair=%d: interpolated band is not bare/16", layer, j)
				}
			default:
				if !(yarn[j] > bare[j]/16 && yarn[j] < bare[j]) {
					t.Fatalf("layer=%d pair=%d: ramp value %g outside (%g, %g)", layer, j, yarn[j], bare[j]/16, bare[j])
				}
			}
		}
	}
	a, b := v41RopeF32InvFreq(cfg, 0), v41RopeF32InvFreq(bareCfg, 0)
	for j := range a {
		if a[j] != b[j] {
			t.Fatalf("ratio-zero pair=%d depends on the YaRN flag", j)
		}
	}
}

func v41RopeF32WrapAngle(a float64) float64 {
	a = math.Mod(a+math.Pi, 2*math.Pi)
	if a < 0 {
		a += 2 * math.Pi
	}
	return a - math.Pi
}

// fak-test:justify why=contract when=changed:internal/model/**
// fak-test:runtime fast est=10ms lane=default
func TestV41RopeF32AngleParity(t *testing.T) {
	t.Parallel()
	_, cfg := readDeepSeekV41Config(t)
	for _, layer := range []int{0, 2, 20} {
		for _, p := range []int{1, 7} {
			gotCos, gotSin := v41RopeF32Oracle(cfg, layer, p)
			wantCos, wantSin := v41RopeTableForLayer(cfg, layer, p)
			for j := range gotCos {
				if math.Abs(float64(gotCos[j]-wantCos[j])) > 1e-6 || math.Abs(float64(gotSin[j]-wantSin[j])) > 1e-6 {
					t.Fatalf("layer=%d p=%d pair=%d: regime or YaRN policy diverged", layer, p, j)
				}
			}
		}
		previous := -1.0
		for _, p := range []int{4096, 65536, 150000, 1048575} {
			gotCos, gotSin := v41RopeF32Oracle(cfg, layer, p)
			wantCos, wantSin := v41RopeTableForLayer(cfg, layer, p)
			half := 0.5 * math.Ldexp(1, int(math.Floor(math.Log2(float64(p))))-23)
			bound := half + 6*float64(p)*0x1p-24 + 1e-6
			var maxAngle, maxCos, maxSin float64
			for j := range gotCos {
				if norm := float64(gotCos[j])*float64(gotCos[j]) + float64(gotSin[j])*float64(gotSin[j]); math.Abs(norm-1) > 2e-7 {
					t.Fatalf("layer=%d p=%d pair=%d: non-unit magnitude squared=%g", layer, p, j, norm)
				}
				d := math.Abs(v41RopeF32WrapAngle(math.Atan2(float64(gotSin[j]), float64(gotCos[j])) -
					math.Atan2(float64(wantSin[j]), float64(wantCos[j]))))
				if d > bound {
					t.Fatalf("layer=%d p=%d pair=%d: |dangle|=%g exceeds float32 bound %g", layer, p, j, d, bound)
				}
				maxAngle = max(maxAngle, d)
				maxCos = max(maxCos, math.Abs(float64(gotCos[j]-wantCos[j])))
				maxSin = max(maxSin, math.Abs(float64(gotSin[j]-wantSin[j])))
			}
			if maxAngle < previous {
				t.Fatalf("layer=%d p=%d: max angle delta %g fell below %g at a shorter position", layer, p, maxAngle, previous)
			}
			previous = maxAngle
			if p == 1048575 && maxAngle <= 1e-3 {
				t.Fatalf("layer=%d: float32 emulation is indistinguishable at p=%d (max %g)", layer, p, maxAngle)
			}
			t.Logf("SW-VERIFIED f32-emulated vs f64 host rope %s: max|dangle|=%.3g rad max|dcos|=%.3g max|dsin|=%.3g",
				fmt.Sprintf("layer=%d ratio=%d p=%d", layer, v41CompressRatioAt(cfg, layer), p), maxAngle, maxCos, maxSin)
		}
	}
}

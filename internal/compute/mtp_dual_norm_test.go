package compute

import (
	"math"
	"testing"
)

// mtpDualNormRows builds deterministic adversarial rows: a wide dynamic range, sign
// flips, and exact zeros, so a reduction-order or eps mistake shows up bitwise.
func mtpDualNormRows(n int, seed float32) (x, w []float32) {
	x, w = make([]float32, n), make([]float32, n)
	for i := range x {
		mag := float32(math.Pow(10, float64(i%7)-3)) * (seed + float32(i%5))
		if i%3 == 0 {
			mag = -mag
		}
		if i%11 == 0 {
			mag = 0
		}
		x[i] = mag
		w[i] = 0.5 + float32(i%9)*0.125
	}
	return x, w
}

// TestMTPDualNormOracle [SW-VERIFIED, host-only] pins that the single-pass dual-sum
// contract is bit-identical to composing two row norms, over adversarial eps and
// widths including ragged (embed != hidden) and non-multiple-of-lane shapes. It
// witnesses the arithmetic a fused kernel must meet, not any device behavior.
// fak-test:runtime fast est=20ms lane=default
func TestMTPDualNormOracle(t *testing.T) {
	widths := [][2]int{{1, 1}, {7, 7}, {8, 8}, {9, 9}, {63, 63}, {64, 64}, {1024, 1024}, {17, 64}, {64, 17}}
	epss := []float32{0, 1e-30, 1e-6, 1, 1e6}
	for _, wd := range widths {
		e, we := mtpDualNormRows(wd[0], 1.5)
		h, wh := mtpDualNormRows(wd[1], 2.5)
		for _, eps := range epss {
			want, err := MTPTwoNormComposition(e, h, we, wh, eps)
			if err != nil {
				t.Fatalf("composition(%v, eps=%g): %v", wd, eps, err)
			}
			got, err := MTPDualSumNorm(e, h, we, wh, eps)
			if err != nil {
				t.Fatalf("dual sum(%v, eps=%g): %v", wd, eps, err)
			}
			if len(got) != len(want) {
				t.Fatalf("widths %v eps %g: len %d, want %d", wd, eps, len(got), len(want))
			}
			for i := range want {
				if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
					t.Fatalf("widths %v eps %g idx %d: dual-sum %v != composition %v (not bit-identical)", wd, eps, i, got[i], want[i])
				}
			}
		}
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestMTPDualNormRefusesBadInputs(t *testing.T) {
	x, w := mtpDualNormRows(8, 1)
	for name, fn := range map[string]func() error{
		"empty embed":    func() error { _, err := MTPDualSumNorm(nil, x, nil, w, 1e-6); return err },
		"width mismatch": func() error { _, err := MTPDualSumNorm(x, x, w[:4], w, 1e-6); return err },
		"negative eps":   func() error { _, err := MTPDualSumNorm(x, x, w, w, -1); return err },
		"nan eps":        func() error { _, err := MTPTwoNormComposition(x, x, w, w, float32(math.NaN())); return err },
	} {
		if fn() == nil {
			t.Fatalf("%s: expected refusal", name)
		}
	}
}

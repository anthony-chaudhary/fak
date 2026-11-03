package model

import (
	"fmt"
	"math"
	"testing"
)

// fak-test:runtime fast est=100ms lane=default
func TestQwen35PrefillNormInto(t *testing.T) {
	equalBits := func(label string, got, want []float32) {
		t.Helper()
		if len(got) != len(want) {
			t.Fatalf("%s lengths: got=%d want=%d", label, len(got), len(want))
		}
		for i := range got {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("%s[%d]: got=%08x want=%08x", label, i, math.Float32bits(got[i]), math.Float32bits(want[i]))
			}
		}
	}
	// The untouched allocating implementation is the scalar-order oracle.
	// Include configured fallbacks, signed zero, NaN payloads and infinities.
	inputs := [][]float32{
		nil,
		{0, math.Float32frombits(0x80000000), 1, -2, 3.25},
		{math.Float32frombits(0x7fc12345), 2, -1},
		{math.Float32frombits(0x7fa12345), 2, -1},
		{float32(math.Inf(1)), float32(math.Inf(-1)), 0},
	}
	configs := []Config{
		{NormGain1p: true},
		{LayerNorm: true, NormGain1p: true},
		{LayerNorm: true},
		{},
	}
	for ci, cfg := range configs {
		for xi, x := range inputs {
			w := make([]float32, len(x))
			for i := range w {
				w[i] = float32(i-2) * .25
			}
			oldX, oldW := append([]float32(nil), x...), append([]float32(nil), w...)
			for _, eps := range []float32{1e-5, 0, -1} {
				for _, width := range []int{0, 1, len(x), len(x) + 2} {
					got, want := make([]float32, width), make([]float32, width)
					for i := range got {
						got[i], want[i] = 123.5, 123.5
					}
					copy(want, rmsnormCfg(x, w, eps, cfg))
					qwen35PrefillNormIntoDisjoint(got, x, w, eps, cfg)
					equalBits(fmt.Sprintf("cfg%d/input%d/width%d", ci, xi, width), got, want)
					equalBits("input unchanged", x, oldX)
					equalBits("weight unchanged", w, oldW)
				}
			}
		}
	}
	// Only dst must be disjoint. Shared read-only x/w is still legal.
	shared := []float32{1, -2, 3, .5}
	got, want := make([]float32, 3), make([]float32, 3)
	cfg := Config{NormGain1p: true}
	copy(want, rmsnormCfg(shared[:3], shared[1:], 1e-5, cfg))
	qwen35PrefillNormIntoDisjoint(got, shared[:3], shared[1:], 1e-5, cfg)
	equalBits("shared read operands", got, want)

	panicText := func(f func()) (text string) {
		defer func() {
			if r := recover(); r != nil {
				text = fmt.Sprint(r)
			}
		}()
		f()
		return ""
	}
	for _, width := range []int{0, 1, 4} {
		x, w := []float32{1, 2, 3}, []float32{.25}
		got, want := make([]float32, width), make([]float32, width)
		for i := range got {
			got[i], want[i] = 77, 77
		}
		old := panicText(func() { copy(want, rmsnormCfg(x, w, 1e-5, cfg)) })
		current := panicText(func() { qwen35PrefillNormIntoDisjoint(got, x, w, 1e-5, cfg) })
		if old == "" || current != old {
			t.Fatalf("short weights: got panic %q want %q", current, old)
		}
		equalBits("panic leaves destination unchanged", got, want)
	}
	// One explicit, bounded software allocation witness. This does not measure
	// whole-model latency or RSS and remains unrun until this command executes.
	x, w, dst := make([]float32, 5120), make([]float32, 5120), make([]float32, 5120)
	for i := range x {
		x[i], w[i] = float32(i%7-3)*.125, float32(i%5-2)*.25
	}
	baseline := testing.AllocsPerRun(8, func() { copy(dst, rmsnormCfg(x, w, 1e-5, cfg)) })
	selected := testing.AllocsPerRun(8, func() { qwen35PrefillNormIntoDisjoint(dst, x, w, 1e-5, cfg) })
	if baseline < 1 {
		t.Fatalf("allocating oracle no longer establishes this premise: %g allocs", baseline)
	}
	if qwen35PrefillNormIntoQualification {
		if selected != 0 {
			t.Fatalf("selected destination path allocated: %g", selected)
		}
	} else if selected != baseline {
		t.Fatalf("default allocation behavior changed: got=%g want=%g", selected, baseline)
	}
}

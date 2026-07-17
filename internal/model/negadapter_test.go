package model

import (
	"reflect"
	"testing"
)

func negAdapterFixture(t testing.TB) *NegAdapter {
	t.Helper()
	adapter, err := NewNegAdapter(
		NegationProbeArtifact{
			Version: "fak-negation-probe/1", Layer: 0,
			Weights: []float64{2, 0}, Bias: 0, Threshold: .55,
		},
		&LoRAAdapter{
			Adapter: "negation", Target: "residual.layer.0", In: 2, Out: 2,
			Rank: 1, Alpha: 1,
			A: []float32{1, 0},
			B: []float32{0, 1},
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	return adapter
}

func runNegAdapterForward(hidden []float32, depth int, hook ResidualHook) []float32 {
	out := append([]float32(nil), hidden...)
	cfg := Config{EnableResidualHook: hook != nil, residualHook: hook}
	for layer := 0; layer < depth; layer++ {
		composeBlockAtLayer(layer, PreNorm, out, identityNorm(), identityNorm(), 1e-5, cfg, zeroSublayer, zeroSublayer)
	}
	return out
}

func TestNegAdapterApplyAndZeroCostBypass(t *testing.T) {
	adapter := negAdapterFixture(t)

	negative := []float32{-2, 1}
	before := append([]float32(nil), negative...)
	allocs := testing.AllocsPerRun(100, func() {
		copy(negative, before)
		decision := adapter.Apply(0, negative)
		if decision != (NegAdapterRoute{}) {
			t.Fatalf("detector-negative decision=%+v", decision)
		}
	})
	if allocs != 0 {
		t.Fatalf("detector-negative adapter allocations=%g, want zero", allocs)
	}
	if !reflect.DeepEqual(negative, before) {
		t.Fatalf("detector-negative path changed bits: got=%v want=%v", negative, before)
	}

	negated := []float32{2, -1}
	decision := adapter.Apply(0, negated)
	if !decision.DetectorFired || !decision.AdapterApplied {
		t.Fatalf("detector-positive decision=%+v", decision)
	}
	if decision.AdapterMACs != 4 || !reflect.DeepEqual(negated, []float32{2, 1}) {
		t.Fatalf("adapter result=%v decision=%+v", negated, decision)
	}
}

func TestNegAdapterUsesResidualHookForwardPath(t *testing.T) {
	adapter := negAdapterFixture(t)
	got := runNegAdapterForward([]float32{2, -1}, 1, adapter.Hook())
	if !reflect.DeepEqual(got, []float32{2, 1}) {
		t.Fatalf("forward residual=%v, want [2 1]", got)
	}
}

func TestNegAdapterInversionAccuracyAndBounds(t *testing.T) {
	adapter := negAdapterFixture(t)
	type sample struct {
		hidden  []float32
		negated bool
	}
	eval := []sample{
		{[]float32{1.5, -.6}, true}, {[]float32{1.7, -.7}, true},
		{[]float32{1.9, -.8}, true}, {[]float32{2.1, -.9}, true},
		{[]float32{-1.5, .6}, false}, {[]float32{-1.7, .7}, false},
		{[]float32{-1.9, .8}, false}, {[]float32{-2.1, .9}, false},
	}
	baseHook := func(_ int, hidden []float32) { hidden[1] += .4 }
	accuracy := func(depth int, hook ResidualHook) float64 {
		correct := 0
		for _, item := range eval {
			out := runNegAdapterForward(item.hidden, depth, hook)
			// The affirmative world-state is represented by a positive answer axis.
			if out[1] > 0 {
				correct++
			}
		}
		return float64(correct) / float64(len(eval))
	}

	baseEqualDepth := accuracy(1, baseHook)
	routed := accuracy(1, adapter.Hook())
	baseThreeLayers := accuracy(3, baseHook)
	if routed <= baseEqualDepth || routed != 1 {
		t.Fatalf("routed accuracy %.3f does not beat equal-depth base %.3f", routed, baseEqualDepth)
	}
	if baseThreeLayers != routed {
		t.Fatalf("three-layer base accuracy %.3f, want routed %.3f", baseThreeLayers, routed)
	}
	if adapter.Adapter.Rank > MaxNegAdapterRank || adapter.ParameterCount() != 4 {
		t.Fatalf("rank=%d params=%d", adapter.Adapter.Rank, adapter.ParameterCount())
	}
	t.Log("negation inversion benchmark (deterministic held-out activation fixture)")
	t.Log("route                 depth  accuracy  trainable_params  added_MAC/token")
	t.Logf("base-emulation        1      %.3f     0                 0", baseEqualDepth)
	t.Logf("base-emulation        3      %.3f     0                 0", baseThreeLayers)
	t.Logf("detector+rank-1-LoRA  1      %.3f     %d                 %d", routed, adapter.ParameterCount(), 4)
}

func TestNegAdapterRejectsUnboundedOrMismatchedAdapters(t *testing.T) {
	probe := NegationProbeArtifact{Version: "fak-negation-probe/1", Layer: 0, Weights: []float64{1, 0}, Threshold: .5}
	tests := []*LoRAAdapter{
		{Adapter: "wide", Target: "x", In: 2, Out: 2, Rank: MaxNegAdapterRank + 1, Alpha: 1, A: make([]float32, (MaxNegAdapterRank+1)*2), B: make([]float32, 2*(MaxNegAdapterRank+1))},
		{Adapter: "shape", Target: "x", In: 2, Out: 3, Rank: 1, Alpha: 1, A: make([]float32, 2), B: make([]float32, 3)},
	}
	for _, candidate := range tests {
		if _, err := NewNegAdapter(probe, candidate); err == nil {
			t.Fatalf("NewNegAdapter(%s) unexpectedly succeeded", candidate.Adapter)
		}
	}
}

var negAdapterBenchmarkSink float32

func BenchmarkNegAdapterRouting(b *testing.B) {
	adapter := negAdapterFixture(b)
	cases := []struct {
		name   string
		in     []float32
		hook   ResidualHook
		params int
		macs   int
	}{
		{"base-emulation", []float32{2, -1}, func(_ int, hidden []float32) { hidden[1] += .4 }, 0, 0},
		{"detector-negative-bypass", []float32{-2, 1}, adapter.Hook(), adapter.ParameterCount(), 0},
		{"detector-positive-adapter", []float32{2, -1}, adapter.Hook(), adapter.ParameterCount(), 4},
	}
	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			b.ReportMetric(float64(tc.params), "params")
			b.ReportMetric(float64(tc.macs), "adapter-MAC/token")
			b.ReportAllocs()
			var sink float32
			for i := 0; i < b.N; i++ {
				out := runNegAdapterForward(tc.in, 1, tc.hook)
				sink += out[1]
			}
			negAdapterBenchmarkSink = sink
		})
	}
}

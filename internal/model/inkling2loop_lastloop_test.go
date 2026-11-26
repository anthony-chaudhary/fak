package model

import (
	"context"
	"math"
	"reflect"
	"slices"
	"testing"
)

func inkling2LastLoopPair(t *testing.T, batched bool) (per, last *Inkling2LoopHost) {
	t.Helper()
	hp, w := inkling2TinyModel()
	if batched {
		w.Batcher = &inkling2ReverseBatcher{}
	}
	per, err := NewInkling2LoopHost(hp, w)
	if err != nil {
		t.Fatal(err)
	}
	last, err = NewInkling2LoopHost(hp, w)
	if err != nil {
		t.Fatal(err)
	}
	if err := last.SetLastLoopKV(true); err != nil {
		t.Fatal(err)
	}
	return per, last
}

// fak-test:runtime fast est=50ms lane=default
func TestInkling2LoopLastLoopKVPanelPrefillMatchesPerLoop(t *testing.T) {
	per, last := inkling2LastLoopPair(t, true)
	prompt := []int{1, 4, 2, 7, 3, 9}
	if len(prompt) > last.prefillWidth() {
		t.Fatalf("prompt %d spans more than one panel of %d", len(prompt), last.prefillWidth())
	}
	want, err := per.Prefill(context.Background(), prompt)
	if err != nil {
		t.Fatal(err)
	}
	got, err := last.Prefill(context.Background(), prompt)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) || !reflect.DeepEqual(last.conv, per.conv) {
		t.Fatal("empty-cache panel prefill: last-loop logits or conv state differ from per-loop")
	}
	hp := per.HP
	core := hp.CoreEnd - hp.CoreStart
	if per.KVRows() != per.Visits()*len(prompt) || last.KVRows() != hp.Layers*len(prompt) {
		t.Fatalf("KV rows per=%d last=%d, want %d and %d", per.KVRows(), last.KVRows(), per.Visits()*len(prompt), hp.Layers*len(prompt))
	}
	for ip := hp.CoreStart; ip < hp.CoreEnd; ip++ {
		final := hp.Layers + (hp.Passes-2)*core + (ip - hp.CoreStart)
		if !reflect.DeepEqual(last.keys[ip], per.keys[final]) || !reflect.DeepEqual(last.values[ip], per.values[final]) {
			t.Fatalf("layer %d: last-loop history is not the final pass's K/V", ip)
		}
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestInkling2LoopLastLoopKVFirstStepMatchesAndDecodeIsMeasured(t *testing.T) {
	per, last := inkling2LastLoopPair(t, false)
	if !reflect.DeepEqual(last.Step(1), per.Step(1)) {
		t.Fatal("first token from an empty cache differs between KV modes")
	}
	// Teacher-forced on the per-loop greedy path: agreement and mean KL of
	// the next-token distribution are the acceptance measure, not equality.
	tok, agree, steps, kl, differs := 1, 0, 24, 0.0, false
	for i := 0; i < steps; i++ {
		p, q := per.Step(tok), last.Step(tok)
		if !slices.Equal(p, q) {
			differs = true
		}
		for _, v := range q {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("step %d: non-finite last-loop logit", i)
			}
		}
		if inkling2Argmax(p) == inkling2Argmax(q) {
			agree++
		}
		kl += inkling2KL(p, q)
		tok = inkling2Argmax(p)
	}
	if !differs {
		t.Fatal("decode logits never differ: last-loop KV is not in effect")
	}
	if last.KVRows() != last.HP.Layers*last.Pos() || per.KVRows() != per.Visits()*per.Pos() {
		t.Fatalf("KV rows per=%d last=%d at pos %d", per.KVRows(), last.KVRows(), last.Pos())
	}
	t.Logf("last-loop vs per-loop decode: greedy agreement %d/%d, mean KL %.3g nats, KV rows %d vs %d",
		agree, steps, kl/float64(steps), last.KVRows(), per.KVRows())
}

// fak-test:runtime fast est=50ms lane=default
func TestInkling2LoopLastLoopKVSavedStateSurvivesOverwrite(t *testing.T) {
	_, last := inkling2LastLoopPair(t, true)
	if _, err := last.Prefill(context.Background(), []int{3, 1, 4, 1, 5}); err != nil {
		t.Fatal(err)
	}
	saved := last.Save()
	want := [][]float32{slices.Clone(last.Step(2)), slices.Clone(last.Step(6))}
	last.Load(saved)
	if !reflect.DeepEqual([][]float32{last.Step(2), last.Step(6)}, want) {
		t.Fatal("replay from a saved last-loop state differs: a pass overwrote saved rows")
	}
	if err := last.SetLastLoopKV(false); err == nil {
		t.Fatal("KV mode changed after tokens were cached")
	}
	last.Reset()
	if err := last.SetLastLoopKV(false); err != nil {
		t.Fatal(err)
	}
	last.Load(saved)
	if !reflect.DeepEqual(last.Step(2), want[0]) {
		t.Fatal("a loaded last-loop state did not restore its KV mode")
	}
}

func inkling2KL(p, q []float32) float64 {
	lp, lq := inkling2LogSoftmax(p), inkling2LogSoftmax(q)
	var kl float64
	for i := range lp {
		kl += math.Exp(lp[i]) * (lp[i] - lq[i])
	}
	return kl
}

func inkling2LogSoftmax(x []float32) []float64 {
	m := math.Inf(-1)
	for _, v := range x {
		m = math.Max(m, float64(v))
	}
	var s float64
	for _, v := range x {
		s += math.Exp(float64(v) - m)
	}
	out := make([]float64, len(x))
	for i, v := range x {
		out[i] = float64(v) - m - math.Log(s)
	}
	return out
}

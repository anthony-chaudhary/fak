package model

import (
	"context"
	"errors"
	"reflect"
	"testing"
)

type inklingPromptTestBatcher struct {
	inkling2ReverseBatcher
	maxJobs   int
	interrupt func()
}

func (b *inklingPromptTestBatcher) MatVecBatch(jobs []Inkling2LoopMatVecJob) {
	b.maxJobs = max(b.maxJobs, len(jobs))
	b.inkling2ReverseBatcher.MatVecBatch(jobs)
	if b.batches == 8 && b.interrupt != nil {
		b.interrupt()
	}
}

func checkInklingPromptState(t *testing.T, first, second *Inkling2LoopHost) {
	t.Helper()
	if first.pos != second.pos || !reflect.DeepEqual(first.Features, second.Features) ||
		!reflect.DeepEqual(first.keys, second.keys) || !reflect.DeepEqual(first.values, second.values) ||
		!reflect.DeepEqual(first.positions, second.positions) || !reflect.DeepEqual(first.conv, second.conv) {
		t.Fatal("prefill slots, feature taps or position differ from token order")
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestInkling2PromptPanelsPreserveCausalState(t *testing.T) {
	hp, w := inkling2TinyModel()
	b := &inklingPromptTestBatcher{}
	w.Batcher = b
	scalarWeights := *w
	scalarWeights.Batcher = nil
	want, _ := NewInkling2LoopHost(hp, &scalarWeights)
	got, _ := NewInkling2LoopHost(hp, w)
	got.FeatureVisits = []int{0, got.Visits() / 2, got.Visits()}
	want.FeatureVisits = got.FeatureVisits
	// Begin from occupied cache slots; span two panels and the local window.
	got.Step(2)
	want.Step(2)
	tokens := make([]int, got.prefillWidth()+hp.SlidingWindow+3)
	for i := range tokens {
		tokens[i] = (i * 3) % hp.VocabUnpadded
	}
	g, err := got.Prefill(context.Background(), tokens)
	if err != nil {
		t.Fatal(err)
	}
	var v []float32
	for i, tok := range tokens {
		v = want.PrefillToken(tok, i+1 == len(tokens))
	}
	if !reflect.DeepEqual(g, v) || b.maxJobs < got.prefillWidth() {
		t.Fatalf("prefill projection mismatch or no batch: max jobs %d", b.maxJobs)
	}
	checkInklingPromptState(t, want, got)
	if !reflect.DeepEqual(got.Step(5), want.Step(5)) {
		t.Fatal("decode after prefill differs")
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestInkling2PromptPanelInterruptLeavesStateReusable(t *testing.T) {
	for _, abort := range []string{"cancel", "panic"} {
		t.Run(abort, func(t *testing.T) {
			hp, w := inkling2TinyModel()
			got, _ := NewInkling2LoopHost(hp, w)
			want, _ := NewInkling2LoopHost(hp, w)
			got.Step(2)
			want.Step(2)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			b := &inklingPromptTestBatcher{interrupt: cancel}
			stop := errors.New("projection failure")
			if abort == "panic" {
				b.interrupt = func() { panic(stop) }
			}
			w.Batcher = b
			func() {
				if abort == "panic" {
					defer func() {
						if r := recover(); r != stop {
							t.Fatalf("projection panic = %v", r)
						}
					}()
				}
				_, err := got.Prefill(ctx, []int{1, 3, 5})
				if abort == "cancel" && !errors.Is(err, context.Canceled) {
					t.Fatalf("prefill: %v", err)
				}
			}()
			checkInklingPromptState(t, want, got)
			g, err := got.Prefill(context.Background(), []int{1, 3, 5})
			if err != nil {
				t.Fatal(err)
			}
			want.Step(1)
			want.Step(3)
			if !reflect.DeepEqual(g, want.Step(5)) {
				t.Fatal("retry produced different logits")
			}
			checkInklingPromptState(t, want, got)
		})
	}
}

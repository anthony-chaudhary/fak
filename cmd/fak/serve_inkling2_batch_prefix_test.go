package main

import (
	"context"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

// With the cohorts' prefix store on, a lone request runs as a one-row cohort so it
// reads and feeds the store, and its cached prompt tokens reach the caller's timings.
func TestInkling2BatchLoneRequestRunsOneRowCohortWithPrefixStore(t *testing.T) {
	clk := newFakeInkling2Clock()
	var widths []int
	e := newInkling2LoopBatchEngineWith(
		func(context.Context, []int, int, map[int]bool, func(int) bool) (agent.Inkling2LoopGeneration, error) {
			t.Error("a lone request must not run the serialized engine while the prefix store is on")
			return agent.Inkling2LoopGeneration{}, nil
		},
		func(reqs []ggufload.Inkling2LoopBatchRequest) ([]ggufload.Inkling2LoopBatchResult, error) {
			widths = append(widths, len(reqs))
			reqs[0].Emit(7)
			return []ggufload.Inkling2LoopBatchResult{{Generated: 1, Restored: 3}}, nil
		},
		testInkling2Admission(clk), 4096)
	e.loneCohort = true
	go e.run()
	defer close(e.queue)
	done := make(chan agent.Inkling2LoopGeneration, 1)
	go func() {
		gen, err := e.Generate(context.Background(), []int{1, 2, 3, 4}, 4, nil, func(int) bool { return true })
		if err != nil {
			t.Error(err)
		}
		done <- gen
	}()
	clk.waitArmed(t)
	clk.advance(inkling2BatchGapDefault)
	gen := <-done
	if len(widths) != 1 || widths[0] != 1 || gen.CachedPromptTokens != 3 || e.scalarGroups.Load() != 0 {
		t.Fatalf("widths %v gen %+v scalar groups %d; want one 1-row cohort reporting 3 cached tokens", widths, gen, e.scalarGroups.Load())
	}
}

// A one-row cohort refused device memory before it ran falls back to the serialized
// engine instead of halving forever.
func TestInkling2BatchOneRowRefusalFallsBackToScalar(t *testing.T) {
	clk := newFakeInkling2Clock()
	scalar := 0
	e := newInkling2LoopBatchEngineWith(
		func(_ context.Context, _ []int, _ int, _ map[int]bool, emit func(int) bool) (agent.Inkling2LoopGeneration, error) {
			scalar++
			emit(7)
			return agent.Inkling2LoopGeneration{Generated: 1}, nil
		},
		func(reqs []ggufload.Inkling2LoopBatchRequest) ([]ggufload.Inkling2LoopBatchResult, error) {
			return nil, &ggufload.Inkling2BatchMemoryError{Sequences: len(reqs), Need: 1 << 30, Free: 1}
		},
		testInkling2Admission(clk), 4096)
	e.loneCohort = true
	go e.run()
	defer close(e.queue)
	done := make(chan error, 1)
	go func() {
		_, err := e.Generate(context.Background(), []int{1}, 4, nil, func(int) bool { return true })
		done <- err
	}()
	clk.waitArmed(t)
	clk.advance(inkling2BatchGapDefault)
	if err := <-done; err != nil || scalar != 1 {
		t.Fatalf("err %v scalar runs %d; want the refused one-row cohort served on the serialized engine", err, scalar)
	}
}

// The startup message names the batch prefix store when it is on and refuses a
// malformed size, or a store cohort prefill being off would leave inert, rather
// than serving without it.
func TestInkling2BatchPrefixNote(t *testing.T) {
	env := func(v string, more ...string) func(string) (string, bool) {
		return func(k string) (string, bool) {
			if k == "FAK_INKLING2_BATCH_PREFIX_CACHE_MIB" && v != "" {
				return v, true
			}
			for i := 0; i+1 < len(more); i += 2 {
				if more[i] == k {
					return more[i+1], true
				}
			}
			return "", false
		}
	}
	if note, err := inkling2BatchPrefixNote(nil, env("")); note != "" || err != nil {
		t.Fatalf("off: %q %v", note, err)
	}
	if note, err := inkling2BatchPrefixNote(nil, env("2048")); err != nil || !strings.Contains(note, "batch prefix cache=2048MiB") {
		t.Fatalf("on: %q %v", note, err)
	}
	if _, err := inkling2BatchPrefixNote(nil, env("big")); err == nil {
		t.Fatal("a malformed size must refuse at startup")
	}
	// the store restores rows mid-prompt, which lockstep prompt feeding cannot do
	if _, err := inkling2BatchPrefixNote(nil, env("2048", "FAK_INKLING2_COHORT_PREFILL", "0")); err == nil || !strings.Contains(err.Error(), "needs cohort prefill") {
		t.Errorf("cohort prefill off: err %v, want a startup refusal: the store would be inert and lone requests lockstep", err)
	}
	if _, err := inkling2BatchPrefixNote(nil, env("2048", "FAK_INKLING2_BATCH_KV", "f16")); err == nil {
		t.Errorf("f16 batch KV on nil backend: err nil, want a startup refusal")
	}
}

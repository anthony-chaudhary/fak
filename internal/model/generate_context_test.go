package model

import (
	"context"
	"errors"
	"reflect"
	"testing"
)

type cancelAfterErrChecks struct {
	context.Context
	cancelAt int
	calls    int
}

func (c *cancelAfterErrChecks) Err() error {
	if c.calls >= c.cancelAt {
		return context.Canceled
	}
	c.calls++
	return nil
}

func TestGenerateContextMatchesGenerate(t *testing.T) {
	cfg := Config{
		HiddenSize: 64, NumLayers: 2, NumHeads: 4, NumKVHeads: 2, HeadDim: 16,
		IntermediateSize: 128, VocabSize: 200, RMSNormEps: 1e-5, RopeTheta: 10000,
		TieWordEmbeddings: true, EOSTokenID: -1,
	}
	m := NewSynthetic(cfg)
	prompt := []int{3, 17, 41, 5}
	const n = 8

	want := m.NewSession().Generate(prompt, n)
	got, err := m.NewSession().GenerateContext(context.Background(), prompt, n)
	if err != nil {
		t.Fatalf("GenerateContext: %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("GenerateContext %v != Generate %v", got, want)
	}
}

func TestGenerateContextStopsBeforeNextStepOnCancel(t *testing.T) {
	cfg := Config{
		HiddenSize: 64, NumLayers: 2, NumHeads: 4, NumKVHeads: 2, HeadDim: 16,
		IntermediateSize: 128, VocabSize: 200, RMSNormEps: 1e-5, RopeTheta: 10000,
		TieWordEmbeddings: true, EOSTokenID: -1,
	}
	m := NewSynthetic(cfg)
	prompt := []int{3, 17, 41, 5}
	ctx := &cancelAfterErrChecks{Context: context.Background(), cancelAt: 5}
	s := m.NewSession()

	got, err := s.GenerateContext(ctx, prompt, 8)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("GenerateContext err = %v, want context.Canceled", err)
	}
	if len(got) != 2 {
		t.Fatalf("GenerateContext emitted %d tokens, want 2", len(got))
	}
	if wantLen := len(prompt) + 1; s.Cache.Len() != wantLen {
		t.Fatalf("cache len after cancel = %d, want %d (no extra Step after cancellation)", s.Cache.Len(), wantLen)
	}
}

func TestGenerateBatchContextMatchesGenerateBatch(t *testing.T) {
	cfg := Config{
		HiddenSize: 64, NumLayers: 2, NumHeads: 4, NumKVHeads: 2, HeadDim: 16,
		IntermediateSize: 128, VocabSize: 200, RMSNormEps: 1e-5, RopeTheta: 10000,
		TieWordEmbeddings: true, EOSTokenID: -1,
	}
	m := NewSynthetic(cfg)
	prompts := [][]int{
		{3, 17, 41, 5},
		{11, 19, 23, 29, 31},
		{7, 13, 37, 43, 47, 53},
	}
	const n = 8

	want := m.NewBatchSession(len(prompts)).GenerateBatch(prompts, n)
	got, err := m.NewBatchSession(len(prompts)).GenerateBatchContext(context.Background(), prompts, n)
	if err != nil {
		t.Fatalf("GenerateBatchContext: %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("GenerateBatchContext %v != GenerateBatch %v", got, want)
	}
}

func TestGenerateBatchContextStopsBeforeNextStepOnCancel(t *testing.T) {
	cfg := Config{
		HiddenSize: 64, NumLayers: 2, NumHeads: 4, NumKVHeads: 2, HeadDim: 16,
		IntermediateSize: 128, VocabSize: 200, RMSNormEps: 1e-5, RopeTheta: 10000,
		TieWordEmbeddings: true, EOSTokenID: -1,
	}
	m := NewSynthetic(cfg)
	prompts := [][]int{
		{3, 17, 41, 5},
		{11, 19, 23, 29, 31},
		{7, 13, 37, 43, 47, 53},
	}
	ctx := &cancelAfterErrChecks{Context: context.Background(), cancelAt: 5}
	bs := m.NewBatchSession(len(prompts))

	got, err := bs.GenerateBatchContext(ctx, prompts, 8)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("GenerateBatchContext err = %v, want context.Canceled", err)
	}
	for b := range prompts {
		if len(got[b]) != 2 {
			t.Fatalf("lane %d emitted %d tokens, want 2", b, len(got[b]))
		}
		if wantLen := len(prompts[b]) + 1; bs.Seqs[b].Cache.Len() != wantLen {
			t.Fatalf("lane %d cache len after cancel = %d, want %d (no extra StepBatchActive after cancellation)",
				b, bs.Seqs[b].Cache.Len(), wantLen)
		}
	}
}

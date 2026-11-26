package main

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

// inkling2ReusedMetalTarget is a fake native session that reports the prompt
// prefix it reused when opened, as the darwin session does.
type inkling2ReusedMetalTarget struct {
	*inkling2FakeMetalTarget
	reused int
}

func (f *inkling2ReusedMetalTarget) Reused() int { return f.reused }

func TestInkling2MetalPromptSessionOpensOnFirstPrefillWithWholePrompt(t *testing.T) {
	target := &inkling2ReusedMetalTarget{inkling2FakeMetalTarget: &inkling2FakeMetalTarget{next: []int32{7, 8}}, reused: 3}
	var (
		opens    int
		capacity int
		opened   []int32
	)
	s := &inkling2MetalPromptSession{capacity: 4096, open: func(c int, ids []int32) (inkling2MetalTarget, error) {
		opens++
		capacity, opened = c, append([]int32(nil), ids...)
		return target, nil
	}}

	if _, err := s.Step(1); err == nil {
		t.Fatal("step before prefill accepted")
	}
	if err := s.Close(); err != nil || opens != 0 {
		t.Fatalf("close before prefill = %v with %d opens, want none", err, opens)
	}
	if _, err := s.Prefill(context.Background(), nil); err == nil || opens != 0 {
		t.Fatalf("empty prompt = %v with %d opens, want an error before any open", err, opens)
	}

	s = &inkling2MetalPromptSession{capacity: 4096, open: s.open}
	next, err := s.Prefill(context.Background(), []int{5, 6, 7})
	if err != nil || next != 7 {
		t.Fatalf("prefill = %d, %v; want 7", next, err)
	}
	if opens != 1 || capacity != 4096 || !inkling2EqualInt32s(opened, []int32{5, 6, 7}) {
		t.Fatalf("opened %d times at capacity %d with %v; want once at 4096 with the whole prompt", opens, capacity, opened)
	}
	if s.Reused() != 3 {
		t.Fatalf("reused = %d, want the target's 3", s.Reused())
	}
	if got, err := s.Step(7); err != nil || got != 8 || len(target.commits) != 1 {
		t.Fatalf("step = %d, %v with commits %v; want 8 with one commit", got, err, target.commits)
	}
	if err := s.Close(); err != nil || !target.closed {
		t.Fatalf("close = %v, closed %v; want the target closed", err, target.closed)
	}
}

func TestInkling2MetalEngineReportsReusedPromptTokens(t *testing.T) {
	target := &inkling2ReusedMetalTarget{inkling2FakeMetalTarget: &inkling2FakeMetalTarget{next: []int32{11}}, reused: 2}
	eng := newInkling2LoopEngineFrom(func() (ggufload.Inkling2LoopSession, error) {
		return &inkling2MetalPromptSession{capacity: 64, open: func(int, []int32) (inkling2MetalTarget, error) {
			return target, nil
		}}, nil
	})
	res, err := eng.Generate(context.Background(), []int{5, 6}, 1, nil, func(int) bool { return true })
	if err != nil {
		t.Fatal(err)
	}
	if res.CachedPromptTokens != 2 || res.Generated != 1 {
		t.Fatalf("generation = %+v, want 2 cached prompt tokens and one token", res)
	}
	if len(target.prefill) != 2 || target.prefill[0] != 5 || target.prefill[1] != 6 {
		t.Fatalf("prefill ids = %v, want the whole prompt", target.prefill)
	}
}

func inkling2EqualInt32s(a, b []int32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

package main

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

// inkling2ReusedSession is a fake session that reports the prompt prefix it
// reused, verifying that inkling2LoopEngine surfaces Reused() on its result.
type inkling2ReusedSession struct {
	*inkling2FakeSession
	reused int
}

func (s *inkling2ReusedSession) Reused() int { return s.reused }

func TestInkling2MetalEngineReportsReusedPromptTokens(t *testing.T) {
	sess := &inkling2ReusedSession{inkling2FakeSession: &inkling2FakeSession{canned: []int{11}}, reused: 2}
	eng := newInkling2LoopEngineFrom(func() (ggufload.Inkling2LoopSession, error) {
		return sess, nil
	})
	defer eng.Close()
	res, err := eng.Generate(context.Background(), []int{5, 6}, 1, nil, func(int) bool { return true })
	if err != nil {
		t.Fatal(err)
	}
	if res.CachedPromptTokens != 2 || res.Generated != 1 {
		t.Fatalf("generation = %+v, want 2 cached prompt tokens and one token", res)
	}
	if !inkling2EqualInts(sess.prompt, []int{5, 6}) {
		t.Fatalf("prefill ids = %v, want the whole prompt", sess.prompt)
	}
}

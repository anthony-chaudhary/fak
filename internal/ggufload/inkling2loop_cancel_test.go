package ggufload

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
)

type inkling2CancelFakeSession struct {
	steps atomic.Int32
}

func (s *inkling2CancelFakeSession) Prefill(ctx context.Context, _ []int) (int, error) {
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	return 7, nil
}

func (s *inkling2CancelFakeSession) Step(_ int) (int, error) {
	s.steps.Add(1)
	return 7, nil
}

func (s *inkling2CancelFakeSession) Close() error { return nil }

// fak-test:runtime fast est=1s lane=default
func TestInkling2LoopSessionDecodeCancelsMidDecode(t *testing.T) {
	t.Parallel()
	const maxTokens = 500
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sess := &inkling2CancelFakeSession{}
	emitted := 0
	turn := &Inkling2LoopTurn{
		MaxTokens: maxTokens,
		Emit: func(int) bool {
			emitted++
			if emitted == 3 {
				cancel()
			}
			return true
		},
	}
	st, err := Inkling2LoopSessionDecode(ctx, sess, []int{1, 2, 3}, turn)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Inkling2LoopSessionDecode err = %v, want context.Canceled", err)
	}
	if emitted > 5 || turn.Generated > 5 {
		t.Fatalf("emitted=%d turn.Generated=%d after cancel at 3, want <= 5", emitted, turn.Generated)
	}
	if st.DecodeSteps > 4 || int(sess.steps.Load()) > 4 {
		t.Fatalf("DecodeSteps=%d session.steps=%d after cancel at 3, want <= 4", st.DecodeSteps, sess.steps.Load())
	}
}

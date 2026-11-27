package main

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

// inkling2CancelTestSession is a lightweight session that produces tokens up to
// maxTokens without terminating early, counting every Step call so cancellation
// tests can prove the worker stopped immediately.
type inkling2CancelTestSession struct {
	steps  atomic.Int32
	closed atomic.Bool
	logits []float32
}

func (s *inkling2CancelTestSession) Prefill(ctx context.Context, _ []int) (int, error) {
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	return 'a', nil
}

func (s *inkling2CancelTestSession) Step(_ int) (int, error) {
	s.steps.Add(1)
	return 'a', nil
}

func (s *inkling2CancelTestSession) LastLogits() ([]float32, error) {
	return s.logits, nil
}

func (s *inkling2CancelTestSession) Close() error {
	s.closed.Store(true)
	return nil
}

// fak-test:runtime fast est=1s lane=default
func TestInkling2ServeGenerateCancelsMidDecode(t *testing.T) {
	t.Parallel()
	const maxTokens = 500
	logits := make([]float32, 128)
	logits['a'] = 10.0

	for _, tc := range []struct {
		name   string
		logits bool
	}{
		{name: "greedy", logits: false},
		{name: "sampled", logits: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			sess := &inkling2CancelTestSession{logits: logits}
			eng := newInkling2LoopEngineFrom(func() (ggufload.Inkling2LoopSession, error) {
				return sess, nil
			})
			defer eng.Close()

			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()

			emitted := 0
			emit := func(int) bool {
				emitted++
				if emitted == 3 {
					cancel()
				}
				return true
			}
			sample := func([]float32) (int, error) { return 'a', nil }

			var err error
			if tc.logits {
				_, err = eng.GenerateLogits(ctx, []int{'p', 'q'}, maxTokens, nil, sample, emit)
			} else {
				_, err = eng.Generate(ctx, []int{'p', 'q'}, maxTokens, nil, emit)
			}
			if !errors.Is(err, context.Canceled) {
				t.Fatalf("err = %v, want context.Canceled", err)
			}
			if emitted > 5 {
				t.Fatalf("emitted %d tokens after cancel at 3, want <= 5", emitted)
			}
			if gotSteps := int(sess.steps.Load()); gotSteps > 4 {
				t.Fatalf("session executed %d Step calls after cancel at 3, want <= 4", gotSteps)
			}
		})
	}
}

// fak-test:runtime fast est=1s lane=default
func TestInkling2ServeBatchCancelsMidDecode(t *testing.T) {
	t.Parallel()
	const maxTokens = 500
	logits := make([]float32, 128)
	logits['a'] = 10.0

	for _, tc := range []struct {
		name   string
		logits bool
	}{
		{name: "scalar_greedy", logits: false},
		{name: "scalar_logits", logits: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			sess := &inkling2CancelTestSession{logits: logits}
			scalar := newInkling2LoopEngineFrom(func() (ggufload.Inkling2LoopSession, error) {
				return sess, nil
			})
			defer scalar.Close()

			adm := inkling2Admission{gap: time.Millisecond, maxWait: 5 * time.Millisecond, max: 4, clock: inkling2RealClock{}}
			batchEng := newInkling2LoopBatchEngineWith(scalar.Generate, nil, adm, 1024)
			batchEng.scalarGenerateLogits = scalar.GenerateLogits
			go batchEng.run()
			defer close(batchEng.queue)

			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()

			var emitted atomic.Int32
			emit := func(int) bool {
				if emitted.Add(1) == 3 {
					cancel()
				}
				return true
			}
			sample := func([]float32) (int, error) { return 'a', nil }

			var err error
			if tc.logits {
				_, err = batchEng.GenerateLogits(ctx, []int{'p', 'q'}, maxTokens, nil, sample, emit)
			} else {
				_, err = batchEng.Generate(ctx, []int{'p', 'q'}, maxTokens, nil, emit)
			}
			if !errors.Is(err, context.Canceled) {
				t.Fatalf("err = %v, want context.Canceled", err)
			}
			// Give the worker goroutine a brief moment to settle, then verify it halted
			// instead of running to maxTokens.
			time.Sleep(10 * time.Millisecond)
			if got := int(emitted.Load()); got > 5 {
				t.Fatalf("emitted %d tokens after cancel at 3, want <= 5", got)
			}
			if gotSteps := int(sess.steps.Load()); gotSteps > 4 {
				t.Fatalf("session executed %d Step calls after cancel at 3, want <= 4", gotSteps)
			}
		})
	}
}

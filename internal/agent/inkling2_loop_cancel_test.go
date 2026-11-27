package agent

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
)

// inkling2CancelAfterNGen simulates a generator that does not poll ctx itself
// and relies solely on emit returning false to halt generation. It cancels ctx
// synchronously after emitting cancelAt tokens so the test is deterministic.
type inkling2CancelAfterNGen struct {
	cancelAt int32
	cancel   context.CancelFunc
	emitted  *atomic.Int32
}

func (g inkling2CancelAfterNGen) Generate(_ context.Context, _ []int, maxNew int, _ map[int]bool, emit func(int) bool) (Inkling2LoopGeneration, error) {
	var gen Inkling2LoopGeneration
	for i := 0; i < maxNew; i++ {
		n := g.emitted.Add(1)
		gen.Generated++
		if !emit('a') {
			gen.Stopped = true
			return gen, nil
		}
		if n == g.cancelAt {
			g.cancel()
		}
	}
	return gen, nil
}

// fak-test:runtime fast est=1s lane=default
func TestInkling2LoopPlannerCancelsMidDecode(t *testing.T) {
	t.Parallel()
	const maxTokens = 500
	msgs := []Message{{Role: RoleUser, Content: "hello"}}

	t.Run("CompleteStream", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		var emitted atomic.Int32
		gen := inkling2CancelAfterNGen{
			cancelAt: 3,
			cancel:   cancel,
			emitted:  &emitted,
		}
		p := inkling2StreamPlanner(t, gen)
		_, err := p.CompleteStream(ctx, func(string) error { return nil }, msgs, nil, WithMaxTokens(maxTokens))
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("CompleteStream err = %v, want context.Canceled", err)
		}
		if got := int(emitted.Load()); got > 5 {
			t.Fatalf("CompleteStream emitted %d tokens after cancel at 3, want <= 5", got)
		}
	})

	t.Run("Complete", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		var emitted atomic.Int32
		gen := inkling2CancelAfterNGen{
			cancelAt: 3,
			cancel:   cancel,
			emitted:  &emitted,
		}
		p := inkling2StreamPlanner(t, gen)
		_, err := p.Complete(ctx, msgs, nil, WithMaxTokens(maxTokens))
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("Complete err = %v, want context.Canceled", err)
		}
		if got := int(emitted.Load()); got > 5 {
			t.Fatalf("Complete emitted %d tokens after cancel at 3, want <= 5", got)
		}
	})
}

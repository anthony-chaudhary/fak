// fak-test:runtime fast est=1s lane=default
package agent

import (
	"context"
	"math/rand"
	"testing"
)

func TestRollingTokenCycleBreaker(t *testing.T) {
	t.Run("ignores_short_span_under_16", func(t *testing.T) {
		var cb rollingTokenCycleBreaker
		for rep := 0; rep < 5; rep++ {
			for tok := 0; tok < 15; tok++ {
				if cb.Observe(100 + tok) {
					t.Fatalf("triggered on 15-token span at rep=%d tok=%d", rep, tok)
				}
			}
		}
	})

	t.Run("ignores_3x_repeat_triggers_on_4x", func(t *testing.T) {
		var cb rollingTokenCycleBreaker
		for rep := 0; rep < 4; rep++ {
			for tok := 0; tok < 16; tok++ {
				got := cb.Observe(200 + tok)
				want := rep == 3 && tok == 15
				if got != want {
					t.Fatalf("rep=%d tok=%d: got=%v want=%v", rep, tok, got, want)
				}
			}
		}
	})

	t.Run("triggers_after_prefix_on_longer_span", func(t *testing.T) {
		var cb rollingTokenCycleBreaker
		for i := 0; i < 40; i++ {
			if cb.Observe(1000 + i) {
				t.Fatalf("unexpected trigger in unique prefix at %d", i)
			}
		}
		for rep := 0; rep < 4; rep++ {
			for tok := 0; tok < 24; tok++ {
				got := cb.Observe(5000 + tok)
				want := rep == 3 && tok == 23
				if got != want {
					t.Fatalf("rep=%d tok=%d: got=%v want=%v", rep, tok, got, want)
				}
			}
		}
	})

	t.Run("triggers_even_after_history_compaction", func(t *testing.T) {
		var cb rollingTokenCycleBreaker
		for i := 0; i < 2*cycleBreakerMaxHistory+50; i++ {
			if cb.Observe(10000 + i) {
				t.Fatalf("unexpected trigger in unique stream at %d", i)
			}
		}
		for rep := 0; rep < 4; rep++ {
			for tok := 0; tok < 16; tok++ {
				got := cb.Observe(50 + tok)
				want := rep == 3 && tok == 15
				if got != want {
					t.Fatalf("rep=%d tok=%d: got=%v want=%v", rep, tok, got, want)
				}
			}
		}
	})
}

func TestDecodeLaneCycleBreakerStopsLoop(t *testing.T) {
	const vocab = 32
	logits := make([]float32, vocab)
	var emitted []int
	ln := &decodeLane{
		logits: logits,
		rng:    rand.New(rand.NewSource(1)),
		maxNew: 200,
		vocab:  vocab,
		emit: func(id int) bool {
			emitted = append(emitted, id)
			return false
		},
	}

	step := 0
	for !ln.done {
		for i := range logits {
			logits[i] = -10
		}
		target := step % 16
		logits[target] = 10
		_, advance := ln.decodeOne(context.Background())
		step++
		if !advance {
			break
		}
	}

	if !ln.stopped || !ln.done {
		t.Fatalf("expected stopped=true done=true, got stopped=%v done=%v", ln.stopped, ln.done)
	}
	if len(emitted) != 64 {
		t.Fatalf("expected cycle breaker to stop at 4*16=64 tokens, got %d", len(emitted))
	}
}

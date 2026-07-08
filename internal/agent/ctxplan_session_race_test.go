package agent

import (
	"context"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
)

// TestSessionPlannerConcurrentRenderTurnIsRaceFree reproduces the crash the gateway saw on the
// stateless OpenAI wire: every keyless request pins to ONE shared trace ("default"), so many
// UNRELATED conversations hit the SAME SessionPlanner concurrently (sessionPlannerFor releases
// sessionPlannerMu before RenderTurn). Two goroutines mutating one planner's store/index/fprints
// at once tore the append-only-contract state apart — divergesFromIngested indexed messages[i]
// against a concurrently-reset fprints slice and panicked with index-out-of-range, surfacing at
// ctxplan_session.go RenderTurn -> gateway maybePlanMessages -> handleAnthropicMessages.
//
// Run with -race to catch the data race; the panic reproduces even without -race under load.
func TestSessionPlannerConcurrentRenderTurnIsRaceFree(t *testing.T) {
	ctx := context.Background()
	sp := NewSessionPlanner(64)

	// Each goroutine sends an INDEPENDENT, divergent history (the stateless-wire shape) so every
	// turn trips divergesFromIngested and races the reset against another goroutine's ingest.
	const goroutines = 16
	const turns = 40
	var wg sync.WaitGroup
	var empty int64 // RenderTurn calls that returned no messages — a torn view; must stay 0.
	wg.Add(goroutines)
	for g := 0; g < goroutines; g++ {
		go func(g int) {
			defer wg.Done()
			for i := 0; i < turns; i++ {
				history := []Message{
					{Role: RoleSystem, Content: "system"},
					{Role: RoleUser, Content: "conversation-" + strconv.Itoa(g) + "-turn-" + strconv.Itoa(i)},
				}
				if out := sp.RenderTurn(ctx, history); len(out) == 0 {
					atomic.AddInt64(&empty, 1)
				}
			}
		}(g)
	}
	wg.Wait()

	// Beyond "did not panic / no -race report": every concurrent RenderTurn must
	// still return a non-empty view for its 2-message history. A torn store/index
	// could silently materialize an empty turn instead of crashing, and that
	// corruption is exactly what this reproducer must also catch.
	if empty != 0 {
		t.Fatalf("RenderTurn returned an empty view on %d/%d concurrent calls (torn state)",
			empty, int64(goroutines*turns))
	}
}

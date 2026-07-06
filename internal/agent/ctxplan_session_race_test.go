package agent

import (
	"context"
	"strconv"
	"sync"
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
	wg.Add(goroutines)
	for g := 0; g < goroutines; g++ {
		go func(g int) {
			defer wg.Done()
			for i := 0; i < turns; i++ {
				history := []Message{
					{Role: RoleSystem, Content: "system"},
					{Role: RoleUser, Content: "conversation-" + strconv.Itoa(g) + "-turn-" + strconv.Itoa(i)},
				}
				sp.RenderTurn(ctx, history)
			}
		}(g)
	}
	wg.Wait()
}

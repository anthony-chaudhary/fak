package agent

import (
	"context"
	"fmt"
	"strings"
	"testing"
)

// BenchmarkContextPlanScaling is the committed witness behind the per-session-planner perf
// win (the cost half of TestSessionPlannerBoundedMatchesStatelessFullScan's correctness
// half). It measures CUMULATIVE per-turn planning cost over a growing conversation — the
// live-loop shape, where each turn replans on the whole prefix-so-far — for the two paths
// the gateway can take:
//
//   - stateless: CtxViewPlanner.RenderTurn rebuilds a lossless MemStore from the FULL
//     message list and full-scans it every turn. O(N) per turn → Θ(N²) cumulative.
//   - incremental: ONE persistent SessionPlanner ingests only each turn's NEW messages and
//     probes a bounded candidate set. O(c) amortized per turn → Θ(c·N) cumulative.
//
// Run: go test ./internal/agent/ -run X -bench ContextPlanScaling -benchmem
// The incremental arm's ns/op should fall further below the stateless arm's as turns grow
// (the O(N²) vs O(c·N) signature); at 800 turns the gap is several-fold on a dev box.
func BenchmarkContextPlanScaling(b *testing.B) {
	for _, turns := range []int{50, 200, 800} {
		msgs := benchSession(turns)
		const budget = 2000

		b.Run(fmt.Sprintf("stateless/turns=%d", turns), func(b *testing.B) {
			ctx := context.Background()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				p := &CtxViewPlanner{Enabled: true, Budget: budget}
				for n := 2; n <= len(msgs); n += 2 { // replay growing prefixes (the live loop)
					_, _ = p.RenderTurn(ctx, msgs[:n])
				}
			}
		})

		b.Run(fmt.Sprintf("incremental/turns=%d", turns), func(b *testing.B) {
			ctx := context.Background()
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				sp := NewSessionPlanner(budget)
				for n := 2; n <= len(msgs); n += 2 {
					_ = sp.RenderTurn(ctx, msgs[:n])
				}
			}
		})
	}
}

// benchSession builds a system prompt + `turns` user/assistant exchanges with realistic
// content lengths, the input both planner arms replay on growing prefixes.
func benchSession(turns int) []Message {
	msgs := make([]Message, 0, 1+turns*2)
	msgs = append(msgs, Message{Role: RoleSystem, Content: "You are a coding agent. Follow the policy and use the tools."})
	pad := strings.Repeat("context ", 40)
	for i := 0; i < turns; i++ {
		msgs = append(msgs, Message{Role: RoleUser, Content: fmt.Sprintf("user turn %d: %s", i, pad)})
		msgs = append(msgs, Message{Role: RoleAssistant, Content: fmt.Sprintf("assistant turn %d reply: %s", i, pad)})
	}
	return msgs
}

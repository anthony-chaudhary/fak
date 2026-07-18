package gateway

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/sessionctl"
)

func TestApplyBudgetResetNextWitnessMatchesFreshAdmission(t *testing.T) {
	const child = "next-reset-child"
	seed := []agent.Message{{Role: agent.RoleSystem, Content: "exact carryover recap"}}
	for _, tc := range []struct {
		name      string
		proceed   bool
		wantApply bool
		wantWhy   string
	}{
		{name: "fresh-admitted", proceed: true, wantApply: true},
		{name: "fresh-refused", wantWhy: "fresh child admission refused: child paused"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			sessionctl.ReadBudgetResetNextRecords(child)
			t.Cleanup(func() { sessionctl.ReadBudgetResetNextRecords(child) })
			s := &Server{
				resetOnBudget: func(context.Context, string, []agent.Message) (string, []agent.Message, bool) {
					return child, seed, true
				},
				decideSession: func(context.Context, string) SessionVerdict {
					return SessionVerdict{Proceed: tc.proceed, State: SessionState{TraceID: child, Run: "paused", Reason: "child paused"}}
				},
			}
			trace, messages, _, ok, canceled, reset := s.applyBudgetReset(context.Background(), SessionState{TraceID: "parent", ContinuationID: child}, []agent.Message{{Role: agent.RoleUser, Content: "live request"}})
			if !reset || canceled || ok != tc.proceed || trace != child {
				t.Fatalf("trace=%q reset=%v ok=%v canceled=%v", trace, reset, ok, canceled)
			}
			if len(messages) != 2 || messages[0].Content != seed[0].Content {
				t.Fatalf("messages=%+v", messages)
			}
			rows := sessionctl.ReadBudgetResetNextRecords(child)
			if len(rows) != 1 {
				t.Fatalf("Next rows=%d want 1: %+v", len(rows), rows)
			}
			got := rows[0]
			if got.Move.Payload != seed[0].Content || got.Applied != tc.wantApply || got.Refusal != tc.wantWhy {
				t.Fatalf("Next=%+v want applied=%v refusal=%q", got, tc.wantApply, tc.wantWhy)
			}
		})
	}
}

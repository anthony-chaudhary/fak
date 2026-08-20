package session

import (
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/ctxplan"
)

func TestBeginProviderSessionStartsFreshConversationWithoutLaunderingEnvelope(t *testing.T) {
	tbl := NewTable()
	t0 := time.Unix(1_800_000_000, 0)
	budget := Budget{
		TurnsLeft: 7, TokensLeft: 900, ContextTokensLeft: 120, ContextTokensCap: 1000,
		ClarificationQueriesLeft: 2, ClarificationQueriesCap: 5,
		SpendMicroCentsLeft: 30, SpendMicroCentsCap: 100,
		ToolCallsLeft: 4, ToolCallsCap: 9,
	}
	if _, ok := tbl.SetBudget("old", budget); !ok {
		t.Fatal("set budget")
	}
	if _, ok := tbl.SetPriority("old", 3); !ok {
		t.Fatal("set priority")
	}
	if _, ok := tbl.SetPace("old", Pace{MaxTokensPerTurn: 80, MinTurnGapMs: 10}); !ok {
		t.Fatal("set pace")
	}
	if _, ok := tbl.StartTimeBudget("old", time.Hour, t0); !ok {
		t.Fatal("start time budget")
	}
	cur := tbl.Get("old")
	cur.Goal = Goal{ID: "conversation-goal"}
	cur.Assumptions = []Assumption{{Key: "a", Source: "inferred"}}
	cur.Pins = []string{"span-1"}
	cur.PendingTurn = PendingTurn{Attempt: 2, StartedAtUnixNano: 1}
	cur.ObjectivePin = ctxplan.ObjectivePin{PinID: "pin", Digest: "digest"}
	cur.Cost = CostRing{Count: 1}
	if _, ok := tbl.CompareAndSet("old", cur.Rev, cur); !ok {
		t.Fatal("stage conversation-local state")
	}

	boundary := ProviderSessionBoundary{Provider: "Claude", Source: "CLEAR", ProviderSessionID: "provider-new"}
	child, applied := tbl.BeginProviderSessionAt("old", "claude-abc", boundary, t0.Add(10*time.Minute))
	if !applied {
		t.Fatal("provider boundary was not applied")
	}
	old := tbl.Get("old")
	if old.Run != Stopped || old.Reason != ReasonProviderSessionClear {
		t.Fatalf("old state = run %s reason %q", old.Run, old.Reason)
	}
	if child.Run != Running || child.TraceID != "claude-abc" {
		t.Fatalf("child state = %+v", child)
	}
	if child.Budget.ContextTokensLeft != 1000 {
		t.Fatalf("context left = %d, want fresh cap 1000", child.Budget.ContextTokensLeft)
	}
	if child.Budget.TurnsLeft != 7 || child.Budget.TokensLeft != 900 ||
		child.Budget.ClarificationQueriesLeft != 2 || child.Budget.SpendMicroCentsLeft != 30 ||
		child.Budget.ToolCallsLeft != 4 {
		t.Fatalf("cumulative budget changed across clear: %+v", child.Budget)
	}
	if child.Priority != 3 || child.Pace.MaxTokensPerTurn != 80 || child.Pace.MinTurnGapMs != 10 {
		t.Fatalf("operator drive did not carry: priority=%d pace=%+v", child.Priority, child.Pace)
	}
	if child.Time.Elapsed(t0.Add(10*time.Minute)) != 10*time.Minute || !child.Time.Bounded() {
		t.Fatalf("wall-clock envelope did not carry: %+v", child.Time)
	}
	if child.Goal.ID != "" || len(child.Assumptions) != 0 || len(child.Pins) != 0 ||
		!child.PendingTurn.IsZero() || child.ObjectivePin.PinID != "" || !child.ResetTransaction.IsZero() ||
		child.CacheAffinity.AffinityKey != "" || child.Cost.Count != 0 {
		t.Fatalf("conversation-local state crossed provider clear: %+v", child)
	}
	if child.ProviderBoundary.Schema != ProviderSessionBoundarySchema ||
		child.ProviderBoundary.Provider != "claude" || child.ProviderBoundary.Source != "clear" ||
		child.ProviderBoundary.PreviousTrace != "old" || child.ProviderBoundary.ProviderSessionID != "provider-new" {
		t.Fatalf("boundary = %+v", child.ProviderBoundary)
	}

	again, applied := tbl.BeginProviderSessionAt("old", "claude-abc", boundary, t0.Add(11*time.Minute))
	if applied || again.Rev != child.Rev {
		t.Fatalf("duplicate boundary mutated child: applied=%v rev=%d want=%d", applied, again.Rev, child.Rev)
	}
}

func TestDescriptorRoundTripPreservesProviderBoundary(t *testing.T) {
	boundary := ProviderSessionBoundary{Schema: ProviderSessionBoundarySchema, Provider: "codex", Source: "clear", PreviousTrace: "old", ProviderSessionID: "thread"}
	st := DefaultState("new")
	st.ProviderBoundary = boundary
	st.Rev = 4
	restored := descriptorFromState(st).RestoredState()
	if restored.ProviderBoundary != boundary {
		t.Fatalf("restored boundary = %+v, want %+v", restored.ProviderBoundary, boundary)
	}
}

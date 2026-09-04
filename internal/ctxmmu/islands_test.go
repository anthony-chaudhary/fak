package ctxmmu

import (
	"testing"
)

func TestCtxMMUIslandsWiring(t *testing.T) {
	m := New()

	t.Run("Compactor and SlidingWindow wired into MMU", func(t *testing.T) {
		comp := m.Compactor()
		if comp == nil {
			t.Fatal("expected non-nil Compactor from MMU.Compactor()")
		}
		if comp.mmu != m {
			t.Fatalf("expected compactor MMU to match m")
		}

		sw := m.SlidingWindow()
		if sw == nil {
			t.Fatal("expected non-nil SlidingWindow from MMU.SlidingWindow()")
		}

		// Add pages and verify scan/compact works
		sw.AddPrefixSystem([]byte("System prompt instructions"))
		sw.AddUserTurn(1, []byte("User request"))
		sw.AddToolResult(1, "bash", []byte("1234567890"))
		if sw.PageCount() != 3 {
			t.Fatalf("expected 3 pages, got %d", sw.PageCount())
		}
		report, err := sw.Compact()
		if err != nil {
			t.Fatalf("compact failed: %v", err)
		}
		if !report.PrefixWarm {
			t.Fatal("expected prefix to remain warm")
		}
	})

	t.Run("CompactPositiveState wired into MMU and Compactor", func(t *testing.T) {
		turns := []TurnRecord{
			{Role: "user", Content: "Run database migration"},
			{Role: "assistant", ToolCallName: "exec_sql", ToolCallArgs: "DROP TABLE users"},
			{Role: "tool", ToolCallName: "exec_sql", Content: "[ERROR] Policy block: destructive database operation prohibited", IsFailure: true},
			{Role: "assistant", Content: "I apologize, that was my mistake. Let me run the safe schema check instead.", VerifiedFact: "table users exists"},
			{Role: "assistant", ToolCallName: "inspect_schema", ToolCallArgs: "SELECT 1"},
			{Role: "tool", ToolCallName: "inspect_schema", Content: "OK"},
		}

		// Test MMU.CompactPositive
		hist := m.CompactPositive(turns, "Run database migration")
		if hist == nil {
			t.Fatal("expected non-nil PositiveCompactedHistory from MMU.CompactPositive")
		}
		if hist.ShedTurnsCount == 0 {
			t.Fatal("expected failed turns to be shed")
		}
		if len(hist.VerifiedFacts) == 0 || hist.VerifiedFacts[0] != "table users exists" {
			t.Fatalf("expected verified facts preserved, got %v", hist.VerifiedFacts)
		}

		// Test Compactor.CompactPositive
		c := m.Compactor()
		hist2 := c.CompactPositive(turns, "Run database migration")
		if hist2 == nil || hist2.ShedTurnsCount != hist.ShedTurnsCount {
			t.Fatalf("Compactor.CompactPositive mismatch: %+v vs %+v", hist2, hist)
		}
	})
}

package agent

import "testing"

func TestFoldWidthExcludesClientSuppressedTurns(t *testing.T) {
	report := FoldWidth([]WidthObservation{
		{Lane: "cmd", Engine: "claude", Model: "sonnet", ToolCalls: 2, Success: true},
		{Lane: "cmd", Engine: "claude", Model: "sonnet", ToolCalls: 1, Success: true},
		{Lane: "cmd", Engine: "claude", Model: "sonnet", ToolCalls: 5, Suppressed: true},
		{Lane: "cmd", Engine: "claude", Model: "sonnet", ToolCalls: 0},
	})
	if len(report.Series) != 1 {
		t.Fatalf("report=%+v", report)
	}
	r := report.Series[0]
	if r.AssistantTurns != 4 || r.EligibleToolTurns != 2 || r.SuppressedToolTurns != 1 || r.ToolCalls != 3 || r.BatchedTurns != 1 {
		t.Fatalf("series=%+v", r)
	}
	if r.BatchedTurnRate != 0.5 || r.MeanToolCalls != 1.5 || r.ClientSuppressedRate != 1.0/3.0 || r.OutcomeRate != 0.5 || r.ToolTurnShare != 0.75 {
		t.Fatalf("rates=%+v", r)
	}
}

func TestFoldWidthDistinguishesFusedItemsPerCall(t *testing.T) {
	report := FoldWidth([]WidthObservation{
		// The first MCP call is a fused fak_read batch; the second is the legacy
		// single-path form. Both remain tool calls, while the item axis records the
		// six useful reads they performed between them.
		{Lane: "gateway", Engine: "mcp", Model: "m", ToolCalls: 1, ToolItems: 5, Success: true},
		{Lane: "gateway", Engine: "mcp", Model: "m", ToolCalls: 1, Success: true},
	})
	if len(report.Series) != 1 {
		t.Fatalf("report=%+v", report)
	}
	r := report.Series[0]
	if r.ToolCalls != 2 || r.ToolItems != 6 || r.ItemsPerToolCall != 3 {
		t.Fatalf("fused item axis lost: %+v", r)
	}
	if r.MeanToolCalls != 1 || r.BatchedTurnRate != 0 {
		t.Fatalf("call-width series must remain call-shaped: %+v", r)
	}
}

func TestWidthRegressionRatchetTripsOnlyOnDownwardStep(t *testing.T) {
	if got := DetectWidthRegression(0.6, 0.3, 0.2); !got.Regressed {
		t.Fatalf("downward step=%+v", got)
	}
	if got := DetectWidthRegression(0.1, 0.1, 0.2); got.Regressed {
		t.Fatalf("low steady width alarmed=%+v", got)
	}
	if got := DetectWidthRegression(0.4, 0.6, 0.2); got.Regressed {
		t.Fatalf("improvement alarmed=%+v", got)
	}
}

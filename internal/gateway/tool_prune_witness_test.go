package gateway

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/journal"
)

func TestPrunedToolProposalWitnessLogsOncePerTrace(t *testing.T) {
	srv := newTestServer(t)
	var logs []string
	srv.logf = func(format string, args ...any) {
		line := fmt.Sprintf(format, args...)
		if strings.Contains(line, "pruned tool definition later proposed") {
			logs = append(logs, line)
		}
	}

	srv.recordInboundPrunedToolDefinitions("trace-a", []string{"deny_pruned"})
	calls := []agent.ToolCall{{
		ID:       "c1",
		Type:     "function",
		Function: agent.Func{Name: "deny_pruned", Arguments: `{"secret":"must-not-log"}`},
	}}

	for i := 0; i < 2; i++ {
		_, _, _ = srv.adjudicateProposed(context.Background(), calls, "trace-a")
	}
	_, _, _ = srv.adjudicateProposed(context.Background(), calls, "trace-b")

	if len(logs) != 1 {
		t.Fatalf("pruned proposal logs = %d, want exactly one for trace-a: %v", len(logs), logs)
	}
	if !strings.Contains(logs[0], "trace=trace-a") || !strings.Contains(logs[0], "tool=deny_pruned") {
		t.Fatalf("log does not carry trace/tool witness: %q", logs[0])
	}
	if strings.Contains(logs[0], "must-not-log") || strings.Contains(logs[0], "secret") {
		t.Fatalf("log leaked raw arguments: %q", logs[0])
	}
	if got := srv.metrics.inboundPrunedToolProposalSnapshot(); got != 1 {
		t.Fatalf("pruned proposal metric = %d, want 1", got)
	}
}

func TestPrunedToolProposalMetricRenders(t *testing.T) {
	m := newGatewayMetrics(time.Now())
	m.observeInboundPrunedToolProposal(0)
	m.observeInboundPrunedToolProposal(2)

	var b strings.Builder
	m.writeCompactionMetrics(&b)
	out := b.String()
	if !strings.Contains(out, "fak_gateway_inbound_tools_pruned_then_proposed_total 2") {
		t.Fatalf("metrics missing pruned-then-proposed count:\n%s", out)
	}
}

func TestPrunedToolDefinitionsJournalOncePerTraceAndName(t *testing.T) {
	j := journal.OpenMemory()
	prev := activeJournal
	activeJournal = func() *journal.Journal { return j }
	t.Cleanup(func() { activeJournal = prev })

	srv := newTestServer(t)
	srv.recordInboundPrunedToolDefinitions("trace-a", []string{"customer_lookup", "customer_lookup", "other_tool"})
	srv.recordInboundPrunedToolDefinitions("trace-a", []string{"customer_lookup"})

	rows := j.Recent(10)
	if len(rows) != 2 {
		t.Fatalf("journal rows=%d want 2: %+v", len(rows), rows)
	}
	got := map[string]bool{}
	for _, row := range rows {
		if row.Kind != journal.KindToolDefinitionPruned || row.TraceID != "trace-a" || row.Verdict != "ADVISORY" {
			t.Fatalf("row=%+v", row)
		}
		got[row.Tool] = true
	}
	if !got["customer_lookup"] || !got["other_tool"] {
		t.Fatalf("journaled tools=%v", got)
	}
}

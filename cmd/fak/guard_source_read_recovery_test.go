package main

import (
	"context"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/session"
)

func TestGuardQuarantinedReadRecoveryRotatesRepeatedTraceShape(t *testing.T) {
	messages := []agent.Message{
		{Role: agent.RoleAssistant, ToolCalls: []agent.ToolCall{{ID: "r1", Function: agent.Func{Name: "Read", Arguments: `{"file_path":"internal/agent/loop.go"}`}}}},
		{Role: agent.RoleTool, ToolCallID: "r1", Content: "quarantined: TRUST_VIOLATION"},
		{Role: agent.RoleAssistant, ToolCalls: []agent.ToolCall{{ID: "r2", Function: agent.Func{Name: "Read", Arguments: `{"file_path":"internal/agent/loop.go"}`}}}},
		{Role: agent.RoleTool, ToolCallID: "r2", Content: "quarantined: TRUST_VIOLATION"},
		{Role: agent.RoleAssistant, ToolCalls: []agent.ToolCall{{ID: "r3", Function: agent.Func{Name: "Read", Arguments: `{"file_path":"internal/agent/loop.go"}`}}}},
		{Role: agent.RoleTool, ToolCallID: "r3", Content: "quarantined: TRUST_VIOLATION"},
	}
	got, receipt, ok := guardQuarantinedReadRecovery(messages)
	if !ok {
		t.Fatal("repeated quarantined Read was not recognized")
	}
	if receipt.Schema != guardSourceReadRecoverySchema || receipt.Repeat != 3 || receipt.Checkpoint != "parked-repeated-quarantine" {
		t.Fatalf("receipt=%+v", receipt)
	}
	for _, want := range []string{guardSourceReadRecoverySchema, "Do not replay that Read call", "Get-Content -LiteralPath 'internal/agent/loop.go'", "Select-Object -First 240", "typed park artifact"} {
		if !strings.Contains(got, want) {
			t.Fatalf("recovery missing %q:\n%s", want, got)
		}
	}
}

func TestGuardQuarantinedReadRecoveryRequiresRepeat(t *testing.T) {
	messages := []agent.Message{
		{Role: agent.RoleAssistant, ToolCalls: []agent.ToolCall{{ID: "r1", Function: agent.Func{Name: "Read", Arguments: `{"file_path":"internal/agent/loop.go"}`}}}},
		{Role: agent.RoleTool, ToolCallID: "r1", Content: "quarantined: TRUST_VIOLATION"},
	}
	if got, receipt, ok := guardQuarantinedReadRecovery(messages); ok || got != "" || receipt.Schema != "" {
		t.Fatalf("single quarantine should not rotate: ok=%v got=%q receipt=%+v", ok, got, receipt)
	}
}

func TestGuardQuarantinedReadRecoveryDoesNotRepeatAfterTypedReceipt(t *testing.T) {
	messages := []agent.Message{
		{Role: agent.RoleSystem, Content: "SOURCE_READ_RECOVERY\nschema=" + guardSourceReadRecoverySchema},
		{Role: agent.RoleAssistant, ToolCalls: []agent.ToolCall{{ID: "r1", Function: agent.Func{Name: "Read", Arguments: `{"file_path":"cmd/fak/guard.go"}`}}}},
		{Role: agent.RoleTool, ToolCallID: "r1", Content: "TRUST_VIOLATION"},
		{Role: agent.RoleAssistant, ToolCalls: []agent.ToolCall{{ID: "r2", Function: agent.Func{Name: "Read", Arguments: `{"file_path":"cmd/fak/guard.go"}`}}}},
		{Role: agent.RoleTool, ToolCallID: "r2", Content: "TRUST_VIOLATION"},
	}
	if got, receipt, ok := guardQuarantinedReadRecovery(messages); ok || got != "" || receipt.Schema != "" {
		t.Fatalf("typed recovery must be one-shot: ok=%v got=%q receipt=%+v", ok, got, receipt)
	}
}

func TestGuardBudgetRestartCarriesTypedSourceReadRecovery(t *testing.T) {
	const trace = "guard-read-recovery"
	var child string
	t.Cleanup(func() {
		serveSessions.Reset(trace)
		if child != "" {
			serveSessions.Reset(child)
		}
	})
	serveSessions.SetBudget(trace, session.Budget{TurnsLeft: session.Unbounded, TokensLeft: session.Unbounded, ContextTokensLeft: 5})
	st := debitSession(context.Background(), trace, gateway.SessionUsage{ContextTokens: 6})
	child = st.ContinuationID
	r := newGuardBudgetRestarter(true, 50, 0, t.TempDir(), nil)
	messages := []agent.Message{
		{Role: agent.RoleAssistant, ToolCalls: []agent.ToolCall{{ID: "r1", Function: agent.Func{Name: "Read", Arguments: `{"file_path":"cmd/fak/guard.go"}`}}}},
		{Role: agent.RoleTool, ToolCallID: "r1", Content: "TRUST_VIOLATION"},
		{Role: agent.RoleAssistant, ToolCalls: []agent.ToolCall{{ID: "r2", Function: agent.Func{Name: "Read", Arguments: `{"file_path":"cmd/fak/guard.go"}`}}}},
		{Role: agent.RoleTool, ToolCallID: "r2", Content: "TRUST_VIOLATION"},
	}
	r.OnBudgetExhausted(context.Background(), st, messages)
	ev := <-r.events
	if ev.SourceReadRecovery == nil || ev.SourceReadRecovery.Checkpoint != "parked-repeated-quarantine" || !strings.Contains(ev.SeedText, guardSourceReadRecoverySchema) || strings.Count(ev.SeedText, "SOURCE_READ_RECOVERY") != 1 {
		t.Fatalf("event=%+v seed=%q", ev, ev.SeedText)
	}
}

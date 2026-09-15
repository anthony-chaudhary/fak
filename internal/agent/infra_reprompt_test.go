package agent

import (
	"context"
	"errors"
	"net"
	"strings"
	"testing"
	"time"
)

// infraRepromptPlanner is a scripted planner: it returns the next (*Completion, error)
// from script and counts how many times Complete was called, so a test can witness the
// infra-reprompt arm's call budget directly.
type infraRepromptPlanner struct {
	script []infraStep
	calls  int
}

type infraStep struct {
	comp *Completion
	err  error
}

func (p *infraRepromptPlanner) Model() string { return "infra-reprompt-test" }

func (p *infraRepromptPlanner) Complete(_ context.Context, _ []Message, _ []ToolDef, _ ...SampleOpt) (*Completion, error) {
	idx := p.calls
	p.calls++
	if idx >= len(p.script) {
		return &Completion{
			Message:      Message{Role: RoleAssistant, Content: "fallback done"},
			FinishReason: "stop",
			Usage:        Usage{CompletionTokens: 1},
		}, nil
	}
	return p.script[idx].comp, p.script[idx].err
}

func infraFinal(text string) infraStep {
	return infraStep{comp: &Completion{
		Message:      Message{Role: RoleAssistant, Content: text},
		FinishReason: "stop",
		Usage:        Usage{CompletionTokens: 1},
	}}
}

func infraStatus(status int, body string) infraStep {
	return infraStep{err: &UpstreamStatusError{Status: status, Body: body}}
}

func TestClassifyCompleteError(t *testing.T) {
	cases := []struct {
		name   string
		err    error
		class  infraClass
		reason string
	}{
		{"503", &UpstreamStatusError{Status: 503, Body: "overloaded"}, infraReprompt, "UPSTREAM_STATUS"},
		{"529", &UpstreamStatusError{Status: 529, Body: "overloaded"}, infraReprompt, "UPSTREAM_STATUS"},
		{"429", &UpstreamStatusError{Status: 429, Body: "rate limited"}, infraReprompt, "UPSTREAM_STATUS"},
		{"429 limit reason", &UpstreamStatusError{Status: 429, Body: "rate limited", LimitReason: "LIMIT_SESSION"}, infraReprompt, "LIMIT_SESSION"},
		{"retry ceiling", &RetryCeilingError{Cause: &UpstreamStatusError{Status: 429, Body: "x"}, Wait: time.Hour, Ceiling: time.Minute}, infraReprompt, "RETRY_CEILING"},
		{"plain error", errors.New("boom"), infraReprompt, "TRANSPORT_EXHAUSTED"},
		{"404", &UpstreamStatusError{Status: 404, Body: "unknown model"}, infraTerminal, "UPSTREAM_TERMINAL"},
		{"dial dns", &net.DNSError{Err: "no such host", Name: "nope.example", IsNotFound: true}, infraTerminal, "UPSTREAM_UNREACHABLE"},
		{"context canceled", context.Canceled, infraTerminal, "CONTEXT_DONE"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			class, reason := classifyCompleteError(tc.err)
			if class != tc.class {
				t.Fatalf("class = %v, want %v", class, tc.class)
			}
			if reason != tc.reason {
				t.Fatalf("reason = %q, want %q", reason, tc.reason)
			}
		})
	}
}

func TestInfraRepromptContinues(t *testing.T) {
	p := &infraRepromptPlanner{script: []infraStep{
		infraStatus(503, "overloaded"),
		infraFinal("recovered final"),
	}}
	m, err := RunArm(context.Background(), p, "task", false, 5, nil, WithInfraRepromptBudget(3))
	if err != nil {
		t.Fatalf("RunArm returned error: %v", err)
	}
	if p.calls != 2 {
		t.Fatalf("planner calls = %d, want 2", p.calls)
	}
	if m.InfraReprompts != 1 {
		t.Fatalf("InfraReprompts = %d, want 1", m.InfraReprompts)
	}
	if m.FinalAnswer != "recovered final" {
		t.Fatalf("FinalAnswer = %q, want %q", m.FinalAnswer, "recovered final")
	}
}

func TestInfraRepromptBudgetExhausts(t *testing.T) {
	p := &infraRepromptPlanner{script: []infraStep{
		infraStatus(503, "overloaded"),
		infraStatus(503, "overloaded"),
		infraStatus(503, "overloaded"),
		infraStatus(503, "overloaded"),
		infraStatus(503, "overloaded"),
	}}
	m, err := RunArm(context.Background(), p, "task", false, 10, nil, WithInfraRepromptBudget(2))
	if err == nil {
		t.Fatalf("expected terminal error after budget exhaustion, got nil")
	}
	if p.calls != 3 {
		t.Fatalf("planner calls = %d, want 3 (1 initial + 2 reprompts)", p.calls)
	}
	if m.InfraReprompts != 2 {
		t.Fatalf("InfraReprompts = %d, want 2 (anti-spin bound)", m.InfraReprompts)
	}
}

func TestInfraRepromptTerminalStops(t *testing.T) {
	p := &infraRepromptPlanner{script: []infraStep{
		infraStatus(404, "unknown model"),
		infraFinal("should never run"),
	}}
	m, err := RunArm(context.Background(), p, "task", false, 5, nil, WithInfraRepromptBudget(3))
	if err == nil {
		t.Fatalf("expected terminal error, got nil")
	}
	if p.calls != 1 {
		t.Fatalf("planner calls = %d, want 1", p.calls)
	}
	if m.InfraReprompts != 0 {
		t.Fatalf("InfraReprompts = %d, want 0", m.InfraReprompts)
	}
}

func TestInfraRepromptNegativeBudgetIsHistorical(t *testing.T) {
	p := &infraRepromptPlanner{script: []infraStep{
		infraStatus(503, "overloaded"),
		infraFinal("should never run"),
	}}
	m, err := RunArm(context.Background(), p, "task", false, 5, nil, WithInfraRepromptBudget(-1))
	if err == nil {
		t.Fatalf("expected historical hard-stop error, got nil")
	}
	// Byte-for-byte historical: the disabled arm surfaces the caller's original %w wrap,
	// not a bare unwrapped error.
	if !strings.Contains(err.Error(), "arm turn 1") {
		t.Fatalf("disabled arm must preserve the historical %%w wrap; got %q", err.Error())
	}
	if p.calls != 1 {
		t.Fatalf("planner calls = %d, want 1", p.calls)
	}
	if m.InfraReprompts != 0 {
		t.Fatalf("InfraReprompts = %d, want 0", m.InfraReprompts)
	}
}

func TestInfraContinuationTextClosedToken(t *testing.T) {
	const secret = "SECRET-UPSTREAM-BODY-42"
	text := infraContinuationText("UPSTREAM_STATUS")
	if !strings.Contains(text, "UPSTREAM_STATUS") {
		t.Fatalf("continuation must name the closed reason token; got %q", text)
	}
	if strings.Contains(text, secret) {
		t.Fatalf("continuation must not carry upstream body text; got %q", text)
	}
	blank := infraContinuationText("")
	if !strings.Contains(blank, "INFRA_HICCUP") {
		t.Fatalf("empty reason must fall back to INFRA_HICCUP; got %q", blank)
	}
	// The spliced continuation must be a user turn.
	if RoleUser != "user" {
		t.Fatalf("unexpected RoleUser value %q", RoleUser)
	}
	msg := Message{Role: RoleUser, Content: infraContinuationText("UPSTREAM_STATUS")}
	if msg.Role != RoleUser {
		t.Fatalf("continuation message role = %q, want %q", msg.Role, RoleUser)
	}
	if !strings.Contains(msg.Content, "UPSTREAM_STATUS") {
		t.Fatalf("continuation message content must carry the closed token")
	}
}

func TestNoRepromptOnCleanRun(t *testing.T) {
	p := &infraRepromptPlanner{script: []infraStep{infraFinal("clean final")}}
	m, err := RunArm(context.Background(), p, "task", false, 5, nil, WithInfraRepromptBudget(3))
	if err != nil {
		t.Fatalf("RunArm returned error: %v", err)
	}
	if m.InfraReprompts != 0 {
		t.Fatalf("InfraReprompts = %d, want 0 on a clean run", m.InfraReprompts)
	}
	if m.FinalAnswer != "clean final" {
		t.Fatalf("FinalAnswer = %q, want %q", m.FinalAnswer, "clean final")
	}
}

func TestInfraRepromptDoesNotBumpGoalAnchor(t *testing.T) {
	anchor := NewGoalAnchor("keep going")
	p := &infraRepromptPlanner{script: []infraStep{
		infraStatus(503, "overloaded"),
		infraFinal("recovered final"),
	}}
	_, err := RunArm(context.Background(), p, "task", false, 5, nil, WithGoalAnchor(anchor), WithInfraRepromptBudget(3))
	if err != nil {
		t.Fatalf("RunArm returned error: %v", err)
	}
	if anchor.RecoveryTurnCount != 0 {
		t.Fatalf("RecoveryTurnCount = %d, want 0 (infra reprompt is not a goal recovery turn)", anchor.RecoveryTurnCount)
	}
}

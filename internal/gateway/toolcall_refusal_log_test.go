package gateway

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
)

func TestToolCallRefusalLogCarriesSafeFinishMetadata(t *testing.T) {
	for _, branch := range []string{"buffered conformance", "stream before headers", "stream after headers"} {
		for _, finish := range []string{"length", "secret-finish\nforged-log=true"} {
			t.Run(branch+"/"+finish, func(t *testing.T) {
				var logs []string
				s := &Server{model: "test-model", logf: func(f string, a ...any) { logs = append(logs, fmt.Sprintf(f, a...)) }}
				comp := &agent.Completion{ToolCallsDropped: true, ToolCallsDroppedReason: abi.ReasonOversize, FinishReason: finish, Usage: agent.Usage{CompletionTokens: 24}, Message: agent.Message{Role: agent.RoleAssistant, Content: "secret-completion-and-tool-arguments"}}
				rec := httptest.NewRecorder()
				terminal := 0
				switch branch {
				case "buffered conformance":
					if s.validateChatCompletionConformance(rec, nil, comp, comp.Message, false, false, false, nil) {
						t.Fatal("refusal admitted")
					}
				case "stream before headers":
					if !s.failClosedOnUnparsedToolCalls(rec, comp, false, "pre", "mid", func() { terminal++ }) {
						t.Fatal("refusal not handled")
					}
				case "stream after headers":
					rec.WriteHeader(http.StatusOK)
					if !s.failClosedOnUnparsedToolCalls(rec, comp, true, "pre", "mid", func() { terminal++; _, _ = rec.Write([]byte("terminal")) }) {
						t.Fatal("refusal not handled")
					}
				}
				if len(logs) != 1 {
					t.Fatalf("logs=%v", logs)
				}
				expected := "length"
				if finish != "length" {
					expected = "unknown"
				}
				for _, field := range []string{"reason=OVERSIZE", "finish_reason=" + expected, "completion_tokens=24"} {
					if !strings.Contains(logs[0], field) {
						t.Errorf("missing %q in %q", field, logs[0])
					}
				}
				for _, secret := range []string{"secret-finish", "forged-log", comp.Message.Content} {
					if strings.Contains(logs[0], secret) {
						t.Errorf("private payload leaked into log: %q", logs[0])
					}
				}
				if comp.FinishReason != finish {
					t.Fatal("logging mutated completion metadata")
				}
				if branch == "stream after headers" {
					if rec.Code != http.StatusOK || terminal != 1 || rec.Body.String() != "terminal" {
						t.Fatalf("stream wire changed: %d %d %q", rec.Code, terminal, rec.Body.String())
					}
				} else {
					var envelope struct {
						Error struct {
							Code string `json:"code"`
						} `json:"error"`
					}
					if err := json.Unmarshal(rec.Body.Bytes(), &envelope); err != nil {
						t.Fatal(err)
					}
					if rec.Code != http.StatusBadGateway || terminal != 0 || envelope.Error.Code != "OVERSIZE" {
						t.Fatalf("typed refusal changed: %d %d %q", rec.Code, terminal, rec.Body.String())
					}
				}
			})
		}
	}
}

func TestToolCallLogFinishReasonClosedVocabulary(t *testing.T) {
	for _, s := range []string{"stop", "length", "tool_calls", "function_call", "content_filter"} {
		if got := toolCallLogFinishReason(s); got != s {
			t.Errorf("%q -> %q", s, got)
		}
	}
	for _, s := range []string{"", "unknown", "payload", "length\nsecret"} {
		if got := toolCallLogFinishReason(s); got != "unknown" {
			t.Errorf("untrusted value logged: %q", got)
		}
	}
}

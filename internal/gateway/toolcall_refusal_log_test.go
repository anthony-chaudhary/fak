package gateway

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
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

// refusalTurnPlanner is a scripted planner whose single turn is a budget-cut
// tool call: nothing lifted, the drop typed OVERSIZE, and the decode ended on
// the output-token budget. It drives the real chat handler end to end so the
// refusal witness runs the production path, not a re-implementation.
type refusalTurnPlanner struct{ comp *agent.Completion }

func (p *refusalTurnPlanner) Complete(_ context.Context, _ []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	return p.comp, nil
}

func (p *refusalTurnPlanner) Model() string { return "test-model" }

// TestToolCallRefusalLogCarriesFinishReason is the #13562 per-turn-log half: a
// refused chat turn still spent a full decode, so it keeps its
// gateway_inference_turn row with the finish reason and token usage — otherwise
// a budget-cut tool call is indistinguishable from a served length stop. The
// refusal response itself (typed 502 / in-band SSE error) is unchanged, and no
// reply text enters the log.
// fak-test:runtime fast est=100ms lane=default
func TestToolCallRefusalLogCarriesFinishReason(t *testing.T) {
	const secretReply = "SECRET_REPLY_13562"
	for _, tc := range []struct {
		name   string
		stream bool
	}{
		{name: "buffered", stream: false},
		{name: "stream", stream: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			srv := newTestServer(t)
			var mu sync.Mutex
			var lines []string
			srv.logf = func(f string, a ...any) {
				mu.Lock()
				lines = append(lines, fmt.Sprintf(f, a...))
				mu.Unlock()
			}
			srv.planner = &refusalTurnPlanner{comp: &agent.Completion{
				ToolCallsDropped:       true,
				ToolCallsDroppedReason: abi.ReasonOversize,
				FinishReason:           "length",
				Usage:                  agent.Usage{PromptTokens: 8, CompletionTokens: 24, TotalTokens: 32},
				Message:                agent.Message{Role: agent.RoleAssistant, Content: secretReply},
			}}
			body := `{"model":"test-model","messages":[{"role":"user","content":"hi"}]}`
			if tc.stream {
				body = `{"model":"test-model","stream":true,"messages":[{"role":"user","content":"hi"}]}`
			}
			rr := postNativeReceipt(t, srv, body)

			// The refusal response stays behavior-unchanged.
			if tc.stream {
				if !strings.Contains(rr.Body.String(), `"code":"OVERSIZE"`) {
					t.Fatalf("stream refusal lost its typed code: %s", rr.Body.String())
				}
			} else {
				if rr.Code != http.StatusBadGateway {
					t.Fatalf("status = %d, want 502; body=%s", rr.Code, rr.Body.String())
				}
				if got := chatErrorCode(t, rr.Body.Bytes()); got != "OVERSIZE" {
					t.Fatalf("error code = %q, want OVERSIZE", got)
				}
			}

			mu.Lock()
			defer mu.Unlock()
			var turns []map[string]any
			for _, l := range lines {
				var ev map[string]any
				if !strings.HasPrefix(l, "{") || json.Unmarshal([]byte(l), &ev) != nil {
					continue
				}
				if ev["event"] == "gateway_inference_turn" {
					turns = append(turns, ev)
				}
			}
			if len(turns) != 1 {
				t.Fatalf("gateway_inference_turn rows = %d, want exactly 1 for the refused turn:\n%s", len(turns), strings.Join(lines, "\n"))
			}
			ev := turns[0]
			if ev["finish_reason"] != "length" {
				t.Errorf("finish_reason = %v, want length", ev["finish_reason"])
			}
			if ev["completion_tokens"] != float64(24) {
				t.Errorf("completion_tokens = %v, want 24", ev["completion_tokens"])
			}
			if ev["prompt_tokens"] != float64(8) {
				t.Errorf("prompt_tokens = %v, want 8", ev["prompt_tokens"])
			}
			if ev["wire"] != "openai_chat_completions" {
				t.Errorf("wire = %v", ev["wire"])
			}
			if ev["stream"] != tc.stream {
				t.Errorf("stream = %v, want %v", ev["stream"], tc.stream)
			}
			for _, l := range lines {
				if strings.Contains(l, secretReply) {
					t.Errorf("reply text leaked into the log: %q", l)
				}
			}
		})
	}
}

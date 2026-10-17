package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// fak-test:runtime fast est=2s
func TestArmFinalFinishClassificationHTTP(t *testing.T) {
	for _, synthesis := range []bool{false, true} {
		for _, stream := range []bool{false, true} {
			for _, finish := range []string{"length", "stop", ""} {
				t.Run(fmt.Sprintf("synthesis=%t/stream=%t/finish=%s", synthesis, stream, finish), func(t *testing.T) {
					const text = "The partial answer remains visible."
					requests := new(atomic.Int32)
					srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
						var req struct {
							Stream bool `json:"stream"`
						}
						if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
							http.Error(w, "invalid request", http.StatusBadRequest)
							return
						}
						msg := Message{Role: RoleAssistant, Content: text}
						reason := finish
						if requests.Add(1) == 1 && synthesis {
							msg.Content = ""
							msg.ToolCalls = []ToolCall{{ID: "final-fixture-call", Type: "function", Function: Func{
								Name: toolGetUser, Arguments: `{"user_id":"mia_li_3668"}`,
							}}}
							reason = "tool_calls"
						}
						choice := map[string]any{"index": 0}
						if reason != "" {
							choice["finish_reason"] = reason
						}
						if req.Stream {
							w.Header().Set("Content-Type", "text/event-stream")
							choice["delta"] = msg
						} else {
							w.Header().Set("Content-Type", "application/json")
							choice["message"] = msg
						}
						body, err := json.Marshal(map[string]any{
							"model": "final-finish-fixture", "choices": []any{choice},
							"usage": map[string]int{"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
						})
						if err != nil {
							http.Error(w, "fixture encoding failed", http.StatusInternalServerError)
							return
						}
						if req.Stream {
							fmt.Fprintf(w, "data: %s\n\ndata: [DONE]\n\n", body)
						} else {
							_, _ = w.Write(body)
						}
					}))
					defer srv.Close()
					p := NewHTTPPlanner(srv.URL, "final-finish-fixture", "")
					p.Client = srv.Client()
					ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
					defer cancel()
					var m ArmMetrics
					var err error
					var visible strings.Builder
					opts := []RunOption{WithGracefulDrain(synthesis), WithInfraRepromptBudget(2)}
					if stream {
						m, err = RunArmStream(ctx, p, "fixture task", false, 1, func(chunk string) error {
							_, _ = visible.WriteString(chunk)
							return nil
						}, nil, opts...)
					} else {
						m, err = RunArm(ctx, p, "fixture task", false, 1, nil, opts...)
					}
					wantCalls := 1
					if synthesis {
						wantCalls++
					}
					if err != nil || int(requests.Load()) != wantCalls || m.Turns != wantCalls || m.InfraReprompts != 0 {
						t.Fatalf("turn behavior changed: err=%v requests=%d metrics=%+v", err, requests.Load(), m)
					}
					if m.FinalAnswer != text || m.PromptTokens != 11*wantCalls || m.CompletionTokens != 7*wantCalls {
						t.Fatalf("partial text or usage changed: %+v", m)
					}
					if m.SynthesizedFinalTurn != synthesis || m.HitTurnCap != synthesis || (stream && visible.String() != text) {
						t.Fatalf("synthesis or streamed text changed: metrics=%+v visible=%q", m, visible.String())
					}
					// Inspect the serialized contract so the identical witness compiles on
					// the parent and fails on the missing classification, not missing fields.
					data, err := json.Marshal(m)
					if err != nil {
						t.Fatal(err)
					}
					var receipt struct {
						FinishReason string `json:"final_finish_reason"`
						Truncated    bool   `json:"final_answer_truncated"`
					}
					if err := json.Unmarshal(data, &receipt); err != nil {
						t.Fatal(err)
					}
					if receipt.FinishReason != finish || receipt.Truncated != (finish == "length") {
						t.Fatalf("terminal classification = %+v, want reason=%q truncated=%t", receipt, finish, finish == "length")
					}
					var fields map[string]json.RawMessage
					if err := json.Unmarshal(data, &fields); err != nil {
						t.Fatal(err)
					}
					_, hasReason := fields["final_finish_reason"]
					_, hasTruncated := fields["final_answer_truncated"]
					if hasReason != (finish != "") || hasTruncated != (finish == "length") {
						t.Fatalf("unexpected JSON field presence: reason=%t truncated=%t for finish=%q", hasReason, hasTruncated, finish)
					}
				})
			}
		}
	}
}

package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

const reasoningOnlyFixtureText = "private-reasoning-marker"

// reasoningHTTPPlanner serves real buffered/SSE completions through the production
// HTTPPlanner. Repeating the last response makes an accidental reprompt observable.
func reasoningHTTPPlanner(t *testing.T, turns ...Message) (*HTTPPlanner, *atomic.Int32) {
	t.Helper()
	requests := new(atomic.Int32)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Stream bool `json:"stream"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "invalid fixture request", http.StatusBadRequest)
			return
		}
		i := int(requests.Add(1)) - 1
		if i >= len(turns) {
			i = len(turns) - 1
		}
		msg := turns[i]
		msg.Role = RoleAssistant
		finish := "stop"
		if len(msg.ToolCalls) != 0 {
			finish = "tool_calls"
		}
		choice := map[string]any{"index": 0, "finish_reason": finish}
		usage := map[string]int{"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}
		if req.Stream {
			w.Header().Set("Content-Type", "text/event-stream")
			choice["delta"] = msg
		} else {
			w.Header().Set("Content-Type", "application/json")
			choice["message"] = msg
		}
		body, err := json.Marshal(map[string]any{"model": "reasoning-fixture", "choices": []any{choice}, "usage": usage})
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
	t.Cleanup(srv.Close)
	p := NewHTTPPlanner(srv.URL, "reasoning-fixture", "")
	p.Client = srv.Client()
	return p, requests
}

func runReasoningFixture(t *testing.T, stream bool, p *HTTPPlanner, maxTurns int, opts ...RunOption) (ArmMetrics, error) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if stream {
		return RunArmStream(ctx, p, "fixture task", false, maxTurns, nil, nil, opts...)
	}
	return RunArm(ctx, p, "fixture task", false, maxTurns, nil, opts...)
}

// fak-test:runtime fast est=2s
func TestReasoningOnlyTerminationHTTP(t *testing.T) {
	for _, stream := range []bool{false, true} {
		for _, budget := range []int{0, 2} {
			t.Run(fmt.Sprintf("stream=%t/budget=%d", stream, budget), func(t *testing.T) {
				p, requests := reasoningHTTPPlanner(t, Message{ReasoningContent: reasoningOnlyFixtureText})
				dir, id := t.TempDir(), "reasoning-terminal"
				if err := SaveSessionCheckpoint(dir, SessionCheckpoint{SessionID: id, Status: "active", Task: "fixture task"}); err != nil {
					t.Fatal(err)
				}
				cpPath := filepath.Join(dir, id+".json")
				before, err := os.ReadFile(cpPath)
				if err != nil {
					t.Fatal(err)
				}
				var debits, output, input, pendingClears int
				var progress []ProgressEvent
				gate := SessionGate{
					Debit: func(_ string, out, in int) { debits++; output += out; input += in },
					Checkpoint: func(_ string, attempt, status int, started int64) {
						if attempt == 0 && status == 0 && started == 0 {
							pendingClears++
						}
					},
				}
				m, err := runReasoningFixture(t, stream, p, 5,
					WithInfraRepromptBudget(budget), WithSessionCheckpoint(id, dir),
					WithSessionGate(gate, "reasoning-http"),
					WithProgressObserver(func(ev ProgressEvent) { progress = append(progress, ev) }),
				)
				if !errors.Is(err, ErrNoFinalAnswer) || err.Error() != ErrNoFinalAnswer.Error() {
					t.Fatalf("terminal error = %v", err)
				}
				if requests.Load() != 1 || m.Turns != 1 || m.InfraReprompts != 0 {
					t.Fatalf("request/retry accounting: requests=%d metrics=%+v", requests.Load(), m)
				}
				if debits != 1 || output != 7 || input != 11 || m.CompletionTokens != output || m.PromptTokens != input {
					t.Fatalf("usage lost: debit=%d output=%d input=%d metrics=%+v", debits, output, input, m)
				}
				if m.FinalAnswer != "" || m.SynthesizedFinalTurn || m.HitTurnCap || m.ToolCalls != 0 {
					t.Fatalf("false success: %+v", m)
				}
				for _, ev := range progress {
					if ev.Kind == ProgressInfraReprompt || ev.Kind == ProgressTurnDone {
						t.Fatalf("success/retry event: %+v", ev)
					}
				}
				after, readErr := os.ReadFile(cpPath)
				if readErr != nil || string(before) != string(after) {
					t.Fatalf("completed checkpoint written: err=%v", readErr)
				}
				// Successful buffered HTTP transport still clears its PendingTurn. This
				// is separate from the disk checkpoint's final-answer status.
				if !stream && pendingClears != 1 {
					t.Fatalf("buffered transport clear = %d", pendingClears)
				}
				if strings.Contains(err.Error(), reasoningOnlyFixtureText) {
					t.Fatal("reasoning leaked into error")
				}
			})
		}
	}
}

func reasoningFixtureToolMessage() Message {
	return Message{ReasoningContent: reasoningOnlyFixtureText, ToolCalls: []ToolCall{{
		ID: "fixture-call", Type: "function",
		Function: Func{Name: toolGetUser, Arguments: `{"user_id":"mia_li_3668"}`},
	}}}
}

// fak-test:runtime fast est=2s
func TestReasoningOnlyTerminationCompatibility(t *testing.T) {
	cases := []struct {
		name string
		msg  Message
		tool bool
	}{
		{"empty", Message{}, false},
		{"answer", Message{Content: "answer", ReasoningContent: reasoningOnlyFixtureText}, false},
		{"whitespace", Message{Content: " ", ReasoningContent: reasoningOnlyFixtureText}, false},
		{"tool_cap", reasoningFixtureToolMessage(), true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p, requests := reasoningHTTPPlanner(t, tc.msg)
			m, err := runReasoningFixture(t, false, p, 1, WithGracefulDrain(false))
			if err != nil || requests.Load() != 1 || m.InfraReprompts != 0 {
				t.Fatalf("compatibility: requests=%d metrics=%+v err=%v", requests.Load(), m, err)
			}
			if tc.tool {
				if m.ToolCalls != len(tc.msg.ToolCalls) || !m.HitTurnCap || m.FinalAnswer != "" {
					t.Fatalf("tool cap changed: %+v", m)
				}
			} else if m.FinalAnswer != tc.msg.Content || m.HitTurnCap {
				t.Fatalf("answer changed: %+v", m)
			}
		})
	}
	t.Run("stop_gate_continuation", func(t *testing.T) {
		p, requests := reasoningHTTPPlanner(t, Message{ReasoningContent: reasoningOnlyFixtureText}, Message{Content: "verified answer"})
		gates := 0
		m, err := runReasoningFixture(t, false, p, 3, WithFinalGate(func() (bool, string) {
			gates++
			return gates > 1, "fixture witness pending"
		}))
		if err != nil || requests.Load() != 2 || gates != 2 || m.FinalAnswer != "verified answer" || m.InfraReprompts != 0 {
			t.Fatalf("continuation changed: requests=%d gates=%d metrics=%+v err=%v", requests.Load(), gates, m, err)
		}
	})
}

// fak-test:runtime fast est=2s
func TestReasoningOnlySynthesisTerminationHTTP(t *testing.T) {
	for _, stream := range []bool{false, true} {
		t.Run(fmt.Sprintf("stream=%t", stream), func(t *testing.T) {
			p, requests := reasoningHTTPPlanner(t, reasoningFixtureToolMessage(), Message{ReasoningContent: reasoningOnlyFixtureText})
			dir, id := t.TempDir(), "reasoning-synthesis"
			var output, input int
			m, err := runReasoningFixture(t, stream, p, 1, WithGracefulDrain(true),
				WithSessionCheckpoint(id, dir), WithSessionGate(SessionGate{Debit: func(_ string, out, in int) { output += out; input += in }}, id))
			if !errors.Is(err, ErrNoFinalAnswer) || requests.Load() != 2 || m.Turns != 2 || m.InfraReprompts != 0 {
				t.Fatalf("synthesis result: requests=%d metrics=%+v err=%v", requests.Load(), m, err)
			}
			if !m.HitTurnCap || !m.GracefulDrained || m.SynthesizedFinalTurn || m.FinalAnswer != "" {
				t.Fatalf("false synthesis success: %+v", m)
			}
			if output != 14 || input != 22 || m.CompletionTokens != output || m.PromptTokens != input {
				t.Fatalf("synthesis usage lost: output=%d input=%d metrics=%+v", output, input, m)
			}
			cp, loadErr := LoadSessionCheckpoint(id, dir)
			if loadErr != nil || cp.Status != "active" || cp.Turn != 1 {
				t.Fatalf("synthesis checkpoint = %+v err=%v", cp, loadErr)
			}
		})
	}
}

// fak-test:runtime fast est=1s
func TestReasoningOnlyTerminationRetiresSpeculation(t *testing.T) {
	for _, synthesis := range []bool{false, true} {
		t.Run(fmt.Sprintf("synthesis=%t", synthesis), func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			fsm := NewStreamingToolFSM()
			if err := fsm.Feed(`{"name":"Read","arguments":{"filePath":"fixture.go"}}`); err != nil {
				t.Fatal(err)
			}
			if err := fsm.SpeculativeDispatch(ctx, func(ctx context.Context, _, _ string) (string, error) { <-ctx.Done(); return "", ctx.Err() }); err != nil {
				t.Fatal(err)
			}
			defer func() {
				cancel()
				select {
				case <-fsm.doneCh:
				case <-time.After(time.Second):
					t.Error("speculative fixture did not join")
				}
			}()
			pending := NewTurn(0)
			pending.Suspend(&abi.ToolCall{Tool: "Read", Spec: abi.SpeculationContext{Speculative: true, Epoch: 1}}, abi.Ref{Kind: abi.RefInline, Inline: []byte("provisional"), Len: int64(len("provisional"))})
			cfg, metrics := runConfig{}, ArmMetrics{SpecIssued: 2}
			r := &armRunner{cfg: &cfg, metrics: &metrics, streamingFSM: fsm,
				speculation:    &specState{pending: pending},
				messages:       []Message{{Role: RoleUser, Content: "fixture task"}},
				stopTerminated: func() bool { return false },
				complete: func(context.Context, []Message, []ToolDef, StreamSink, ...SampleOpt) (*Completion, error) {
					return &Completion{Message: Message{ReasoningContent: reasoningOnlyFixtureText}}, nil
				},
			}
			var err error
			if synthesis {
				err = r.runSynthesisTurn(ctx, 0)
			} else {
				var action armTurnAction
				_, action, err = r.requestModel(ctx, 0, 0)
				if action != armTurnStop {
					t.Fatalf("action = %v", action)
				}
			}
			if !errors.Is(err, ErrNoFinalAnswer) {
				t.Fatalf("terminal error = %v", err)
			}
			if fsm.State() != StateSquashed || fsm.SquashReason() != "turn ended without a final answer" {
				t.Fatalf("stream speculation not squashed: %v %q", fsm.State(), fsm.SquashReason())
			}
			if r.speculation.pending != nil || pending.Suspended() || pending.Sink().PendingEpochs() != 0 || len(pending.Sink().Committed()) != 0 || metrics.SpecSquashed != metrics.SpecIssued || metrics.SpecRollbacks != 1 {
				t.Fatalf("pending speculation survived: %+v", metrics)
			}
		})
	}
}

// fak-test:runtime fast est=100ms
func TestReasoningOnlyTerminationClassification(t *testing.T) {
	wrapped := fmt.Errorf("provider status 401 %s: %w", reasoningOnlyFixtureText, ErrNoFinalAnswer)
	if !errors.Is(wrapped, ErrNoFinalAnswer) {
		t.Fatal("sentinel identity lost")
	}
	got := ClassifyTermination(wrapped)
	if got.Cause != TerminationNoFinalAnswer || got.Evidence != "model ended without a final answer" {
		t.Fatalf("classification = %+v", got)
	}
	if strings.Contains(got.Evidence, reasoningOnlyFixtureText) {
		t.Fatal("reasoning leaked")
	}
	if got := ClassifyTermination(errors.Join(wrapped, context.Canceled)); got.Cause != TerminationCanceled {
		t.Fatalf("cancellation precedence = %+v", got)
	}
	if got := ClassifyTermination(errors.New("provider status 401")); got.Cause != TerminationAuth {
		t.Fatalf("existing classification = %+v", got)
	}
}

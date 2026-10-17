package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
)

const nativeFinalFixtureText = "Measured results: one two three."

type nativeFinalFinishPlanner struct {
	finish string
	calls  atomic.Int32
}

func (p *nativeFinalFinishPlanner) Model() string            { return "native-final-finish" }
func (p *nativeFinalFinishPlanner) StreamingSupported() bool { return true }

func (p *nativeFinalFinishPlanner) Complete(ctx context.Context, messages []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	return p.CompleteStream(ctx, nil, messages, tools, opts...)
}

func (p *nativeFinalFinishPlanner) CompleteStream(_ context.Context, sink agent.StreamSink, _ []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	p.calls.Add(1)
	if sink != nil {
		for _, chunk := range []string{"Measured results: ", "one two three."} {
			if err := sink(chunk); err != nil {
				return nil, err
			}
		}
	}
	return &agent.Completion{
		Message:      agent.Message{Role: agent.RoleAssistant, Content: nativeFinalFixtureText},
		FinishReason: p.finish,
		Usage:        agent.Usage{PromptTokens: 11, CompletionTokens: 7},
	}, nil
}

// fak-test:runtime fast est=2s
func TestNativeServeFinalFinishHTTP(t *testing.T) {
	for _, stream := range []bool{false, true} {
		for _, finish := range []string{"length", "stop", ""} {
			t.Run(fmt.Sprintf("stream=%t/finish=%s", stream, finish), func(t *testing.T) {
				agent.Configure()
				abi.RegisterRegionBackend(inlineBackend{})
				srv, err := New(Config{EngineID: "localtools", Model: "native-final-finish", Native: true, NativeMaxTurns: 3})
				if err != nil {
					t.Fatal(err)
				}
				defer srv.Close()
				planner := &nativeFinalFinishPlanner{finish: finish}
				srv.planner = planner
				ts := httptest.NewServer(srv.Handler())
				defer ts.Close()
				body, err := json.Marshal(map[string]any{
					"model": planner.Model(), "max_tokens": 64, "stream": stream,
					"messages": []map[string]string{{"role": "user", "content": "Report the measured results."}},
				})
				if err != nil {
					t.Fatal(err)
				}
				client := &http.Client{Timeout: 5 * time.Second}
				resp, err := client.Post(ts.URL+"/v1/messages", "application/json", bytes.NewReader(body))
				if err != nil {
					t.Fatal(err)
				}
				raw, err := io.ReadAll(resp.Body)
				resp.Body.Close()
				if err != nil || resp.StatusCode != http.StatusOK {
					t.Fatalf("HTTP status=%d err=%v body=%s", resp.StatusCode, err, raw)
				}
				var text strings.Builder
				var stop string
				var usage anthropicUsage
				var arm *agent.ArmMetrics
				if !stream {
					var got anthropicMessageResponse
					if err := json.Unmarshal(raw, &got); err != nil {
						t.Fatal(err)
					}
					for _, block := range got.Content {
						if block.Type == "text" {
							text.WriteString(block.Text)
						}
					}
					stop, usage = got.StopReason, got.Usage
					if got.Fak != nil {
						arm = got.Fak.NativeArm
					}
				} else {
					frames := readAnthropicSSE(t, bytes.NewReader(raw))
					terminalDeltas, terminalStops := 0, 0
					for _, frame := range frames {
						var got struct {
							Delta struct {
								Text       string `json:"text"`
								StopReason string `json:"stop_reason"`
							} `json:"delta"`
							Usage anthropicUsage `json:"usage"`
							Fak   *FakExt        `json:"fak"`
						}
						if err := json.Unmarshal([]byte(frame.data), &got); err != nil {
							t.Fatal(err)
						}
						switch frame.event {
						case "error":
							t.Fatalf("truncation became a wire error: %s", frame.data)
						case "content_block_delta":
							text.WriteString(got.Delta.Text)
						case "message_delta":
							terminalDeltas++
							stop, usage = got.Delta.StopReason, got.Usage
						case "message_stop":
							terminalStops++
							if got.Fak != nil {
								arm = got.Fak.NativeArm
							}
						}
					}
					if terminalDeltas != 1 || terminalStops != 1 || len(frames) == 0 || frames[len(frames)-1].event != "message_stop" {
						t.Fatalf("terminal SSE framing changed: deltas=%d stops=%d body=%s", terminalDeltas, terminalStops, raw)
					}
				}
				wantStop := "end_turn"
				if finish == "length" {
					wantStop = "max_tokens"
				}
				if stop != wantStop {
					t.Errorf("wire stop_reason=%q, want %q for completion finish=%q", stop, wantStop, finish)
				}
				if text.String() != nativeFinalFixtureText || usage.InputTokens != 11 || usage.OutputTokens != 7 {
					t.Errorf("partial content/accounting changed: text=%q usage=%+v", text.String(), usage)
				}
				if planner.calls.Load() != 1 || arm == nil || arm.Turns != 1 || arm.InfraReprompts != 0 || arm.HitTurnCap || arm.ToolCalls != 0 {
					t.Fatalf("owned-loop behavior changed: calls=%d arm=%+v", planner.calls.Load(), arm)
				}
				if arm.FinalAnswer != nativeFinalFixtureText || arm.FinalFinishReason != finish || arm.FinalAnswerTruncated != (finish == "length") {
					t.Errorf("terminal metrics disagree with preserved content: %+v", arm)
				}
			})
		}
	}
}

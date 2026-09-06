package gateway

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
)

type scriptedMultiTurnPlanner struct {
	turn          int
	turns         []*agent.Completion
	capturedMsgs  [][]agent.Message
	capturedTools [][]agent.ToolDef
}

func (p *scriptedMultiTurnPlanner) Complete(ctx context.Context, m []agent.Message, t []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	p.capturedMsgs = append(p.capturedMsgs, append([]agent.Message(nil), m...))
	p.capturedTools = append(p.capturedTools, append([]agent.ToolDef(nil), t...))
	if p.turn >= len(p.turns) {
		return &agent.Completion{
			FinishReason: "stop",
			Message:      agent.Message{Role: agent.RoleAssistant, Content: "all turns finished"},
		}, nil
	}
	comp := p.turns[p.turn]
	p.turn++
	return comp, nil
}

func (p *scriptedMultiTurnPlanner) Model() string { return "scripted-multi-turn" }

func TestResponsesToolElisionAndCASRestoreDogfoodProbe(t *testing.T) {
	casDir := t.TempDir()
	t.Setenv("FAK_CTXRESTORE_CAS_DIR", casDir)

	srv := newTestServer(t)
	abi.RegisterAdjudicator(1, readAdj{})
	const trace = "t-dogfood-responses-cas-restore"

	largePayload := "LARGE_DOGFOOD_PAYLOAD_START: " + strings.Repeat("W", 40<<10) + " :END"
	sum := sha256.Sum256([]byte(largePayload))
	digest := hex.EncodeToString(sum[:])

	planner := &scriptedMultiTurnPlanner{
		turns: []*agent.Completion{
			// Turn 1: Model requests allow_fetch_data
			{
				FinishReason: "tool_calls",
				Message: agent.Message{
					Role: agent.RoleAssistant,
					ToolCalls: []agent.ToolCall{
						{
							ID:       "c_fetch",
							Type:     "function",
							Function: agent.Func{Name: "allow_fetch_data", Arguments: `{"target":"big_dataset"}`},
						},
					},
				},
			},
			// Turn 2: Model receives large result, proposes allow_step_1
			{
				FinishReason: "tool_calls",
				Message: agent.Message{
					Role: agent.RoleAssistant,
					ToolCalls: []agent.ToolCall{
						{
							ID:       "c_step1",
							Type:     "function",
							Function: agent.Func{Name: "allow_step_1", Arguments: `{}`},
						},
					},
				},
			},
			// Turn 3: Model proposes allow_step_2
			{
				FinishReason: "tool_calls",
				Message: agent.Message{
					Role: agent.RoleAssistant,
					ToolCalls: []agent.ToolCall{
						{
							ID:       "c_step2",
							Type:     "function",
							Function: agent.Func{Name: "allow_step_2", Arguments: `{}`},
						},
					},
				},
			},
			// Turn 4: Model proposes allow_step_3
			{
				FinishReason: "tool_calls",
				Message: agent.Message{
					Role: agent.RoleAssistant,
					ToolCalls: []agent.ToolCall{
						{
							ID:       "c_step3",
							Type:     "function",
							Function: agent.Func{Name: "allow_step_3", Arguments: `{}`},
						},
					},
				},
			},
			// Turn 5: Model proposes allow_step_4 (now 4 recent small results exist)
			{
				FinishReason: "tool_calls",
				Message: agent.Message{
					Role: agent.RoleAssistant,
					ToolCalls: []agent.ToolCall{
						{
							ID:       "c_step4",
							Type:     "function",
							Function: agent.Func{Name: "allow_step_4", Arguments: `{}`},
						},
					},
				},
			},
			// Turn 6: Model detects elision in older turn and proposes context restore
			{
				FinishReason: "tool_calls",
				Message: agent.Message{
					Role: agent.RoleAssistant,
					ToolCalls: []agent.ToolCall{
						{
							ID:   "c_restore",
							Type: "function",
							Function: agent.Func{
								Name:      "mcp__fak__fak_context_restore",
								Arguments: fmt.Sprintf(`{"id":"sha256:%s"}`, digest),
							},
						},
					},
				},
			},
			// Turn 7: Model concludes
			{
				FinishReason: "stop",
				Message: agent.Message{
					Role:    agent.RoleAssistant,
					Content: "dogfood probe completed successfully with restored context",
				},
			},
		},
	}
	srv.planner = planner

	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	sendResponsesTurn := func(input []any) responsesResponse {
		body := map[string]any{
			"model": "test-model",
			"input": input,
			"tools": []any{
				map[string]any{
					"type":        "function",
					"name":        "allow_fetch_data",
					"description": "fetch large data",
					"parameters":  map[string]any{"type": "object"},
				},
				map[string]any{
					"type":        "function",
					"name":        "allow_step_1",
					"description": "step 1 action",
					"parameters":  map[string]any{"type": "object"},
				},
				map[string]any{
					"type":        "function",
					"name":        "allow_step_2",
					"description": "step 2 action",
					"parameters":  map[string]any{"type": "object"},
				},
				map[string]any{
					"type":        "function",
					"name":        "allow_step_3",
					"description": "step 3 action",
					"parameters":  map[string]any{"type": "object"},
				},
				map[string]any{
					"type":        "function",
					"name":        "allow_step_4",
					"description": "step 4 action",
					"parameters":  map[string]any{"type": "object"},
				},
			},
		}
		raw, err := json.Marshal(body)
		if err != nil {
			t.Fatal(err)
		}
		req, err := http.NewRequest(http.MethodPost, ts.URL+"/v1/responses", bytes.NewReader(raw))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("X-Trace-Id", trace)

		httpResp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer httpResp.Body.Close()

		if httpResp.StatusCode != http.StatusOK {
			b, _ := io.ReadAll(httpResp.Body)
			t.Fatalf("POST /v1/responses status %d: %s", httpResp.StatusCode, b)
		}

		var resp responsesResponse
		if err := json.NewDecoder(httpResp.Body).Decode(&resp); err != nil {
			t.Fatal(err)
		}
		return resp
	}

	// 1. Turn 1: User prompt
	conversation := []any{
		map[string]any{"type": "message", "role": "user", "content": "analyze big telemetry dataset"},
	}
	resp1 := sendResponsesTurn(conversation)
	if len(resp1.Output) == 0 || resp1.Output[0].Name != "allow_fetch_data" {
		t.Fatalf("expected tool call allow_fetch_data, got: %+v", resp1.Output)
	}

	// 2. Turn 2: Return 40 KiB tool result
	conversation = append(conversation,
		map[string]any{
			"type":      "function_call",
			"call_id":   "c_fetch",
			"name":      "allow_fetch_data",
			"arguments": `{"target":"big_dataset"}`,
		},
		map[string]any{
			"type":    "function_call_output",
			"call_id": "c_fetch",
			"output":  largePayload,
		},
	)
	resp2 := sendResponsesTurn(conversation)
	if len(resp2.Output) == 0 || resp2.Output[0].Name != "allow_step_1" {
		t.Fatalf("expected tool call allow_step_1, got: %+v", resp2.Output)
	}

	// 3. Turns 3..6: Add 4 recent tool calls and results (allow_step_1 .. allow_step_4)
	steps := []string{"allow_step_1", "allow_step_2", "allow_step_3", "allow_step_4"}
	callIDs := []string{"c_step1", "c_step2", "c_step3", "c_step4"}
	for i, step := range steps {
		conversation = append(conversation,
			map[string]any{
				"type":      "function_call",
				"call_id":   callIDs[i],
				"name":      step,
				"arguments": `{}`,
			},
			map[string]any{
				"type":    "function_call_output",
				"call_id": callIDs[i],
				"output":  fmt.Sprintf("result for %s", step),
			},
		)
		if i < len(steps)-1 {
			resp := sendResponsesTurn(conversation)
			expectedNext := steps[i+1]
			if len(resp.Output) == 0 || resp.Output[0].Name != expectedNext {
				t.Fatalf("expected tool call %s, got: %+v", expectedNext, resp.Output)
			}
		}
	}

	// 4. Turn 7: Large output is now beyond the 4-turn recent window and should be elided!
	respFinal := sendResponsesTurn(conversation)

	// Verify CAS file was persisted to disk under casDir
	casFile := filepath.Join(casDir, digest)
	diskBytes, err := os.ReadFile(casFile)
	if err != nil {
		t.Fatalf("expected CAS file at %s: %v", casFile, err)
	}
	if string(diskBytes) != largePayload {
		t.Fatalf("CAS file content mismatch: got %d bytes, want %d bytes", len(diskBytes), len(largePayload))
	}

	// The model consumes a structured restore result before answering; restored
	// evidence itself must never masquerade as a completed assistant answer.
	if respFinal.OutputText != "dogfood probe completed successfully with restored context" {
		t.Fatalf("expected the model's post-restore answer, got: %q", respFinal.OutputText)
	}
	var restored CtxRestoreResult
	for _, msg := range planner.capturedMsgs[len(planner.capturedMsgs)-1] {
		if msg.Role == agent.RoleTool && msg.ToolCallID == "c_restore" {
			if err := json.Unmarshal([]byte(msg.Content), &restored); err != nil {
				t.Fatal(err)
			}
		}
	}
	if restored.Bytes == "" || !strings.HasPrefix(largePayload, restored.Bytes) || !restored.HasMore || restored.Continuation == nil {
		t.Fatalf("expected bounded restored evidence with continuation, got %d bytes", len(restored.Bytes))
	}

	// Verify telemetry counter incremented
	if srv.AdjudicationSummary().CompactionRestoredTurns == 0 {
		t.Fatal("expected CompactionRestoredTurns > 0 after in-band context restoration")
	}

	// 5. Cold cache restore proof: force stash miss with a fresh trace, resolve directly from disk CAS
	coldRes, err := srv.restoreContext("", ContextRestoreRequest{
		ID:      "sha256:" + digest,
		TraceID: "t-cold-trace-miss",
	})
	if err != nil {
		t.Fatalf("cold trace restore from disk CAS failed: %v", err)
	}
	if coldRes.Bytes != largePayload {
		t.Fatalf("cold trace restored bytes mismatch: got %d bytes, want %d bytes", len(coldRes.Bytes), len(largePayload))
	}
}

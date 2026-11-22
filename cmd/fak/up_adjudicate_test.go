package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/macfit"
)

// turnkeyProposalPlanner proposes a fixed set of tool calls after a line of prose.
// stream selects the live handleChatCompletionsStream path over the buffered one.
type turnkeyProposalPlanner struct {
	stream bool
	calls  []agent.ToolCall
}

func (*turnkeyProposalPlanner) Model() string { return "local" }

func (p *turnkeyProposalPlanner) StreamingSupported() bool { return p.stream }

func (p *turnkeyProposalPlanner) Complete(context.Context, []agent.Message, []agent.ToolDef, ...agent.SampleOpt) (*agent.Completion, error) {
	return p.completion(), nil
}

func (p *turnkeyProposalPlanner) CompleteStream(_ context.Context, sink agent.StreamSink, _ []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	comp := p.completion()
	if err := sink(comp.Message.Content); err != nil {
		return nil, err
	}
	return comp, nil
}

func (p *turnkeyProposalPlanner) completion() *agent.Completion {
	return &agent.Completion{
		Message:      agent.Message{Role: agent.RoleAssistant, Content: "Checking the tree.", ToolCalls: append([]agent.ToolCall(nil), p.calls...)},
		FinishReason: "tool_calls",
		Usage:        agent.Usage{PromptTokens: 8, CompletionTokens: 4, TotalTokens: 12},
	}
}

type turnkeyWireAdjudication struct {
	ToolCallID string `json:"tool_call_id"`
	Admitted   bool   `json:"admitted"`
	Verdict    struct {
		Kind   string `json:"kind"`
		Reason string `json:"reason"`
	} `json:"verdict"`
}

type turnkeyAdjudicatedTurn struct {
	callIDs       []string
	finish        string
	adjudications []turnkeyWireAdjudication
}

func readTurnkeyAdjudicatedTurn(t *testing.T, raw []byte, stream bool) turnkeyAdjudicatedTurn {
	t.Helper()
	var turn turnkeyAdjudicatedTurn
	if !stream {
		var wire struct {
			Choices []struct {
				Message      agent.Message `json:"message"`
				FinishReason string        `json:"finish_reason"`
			} `json:"choices"`
			Fak *struct {
				Adjudications []turnkeyWireAdjudication `json:"adjudications"`
			} `json:"fak"`
		}
		if err := json.Unmarshal(raw, &wire); err != nil || len(wire.Choices) != 1 {
			t.Fatalf("decode buffered response: %v: %s", err, raw)
		}
		for _, call := range wire.Choices[0].Message.ToolCalls {
			turn.callIDs = append(turn.callIDs, call.ID)
		}
		turn.finish = wire.Choices[0].FinishReason
		if wire.Fak != nil {
			turn.adjudications = wire.Fak.Adjudications
		}
		return turn
	}
	for _, line := range strings.Split(string(raw), "\n") {
		data, ok := strings.CutPrefix(strings.TrimSpace(line), "data: ")
		if !ok || data == "[DONE]" {
			continue
		}
		var chunk struct {
			Choices []struct {
				Delta struct {
					ToolCalls []struct {
						ID string `json:"id"`
					} `json:"tool_calls"`
				} `json:"delta"`
				FinishReason *string `json:"finish_reason"`
			} `json:"choices"`
			Fak *struct {
				Adjudications []turnkeyWireAdjudication `json:"adjudications"`
			} `json:"fak"`
		}
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			t.Fatalf("decode SSE data %q: %v", data, err)
		}
		for _, choice := range chunk.Choices {
			for _, call := range choice.Delta.ToolCalls {
				turn.callIDs = append(turn.callIDs, call.ID)
			}
			if choice.FinishReason != nil {
				turn.finish = *choice.FinishReason
			}
		}
		if chunk.Fak != nil {
			turn.adjudications = append(turn.adjudications, chunk.Fak.Adjudications...)
		}
	}
	return turn
}

// fak-test:runtime fast est=300ms lane=default
func TestTurnkeyChatCompletionsAdjudicatesProposedToolCalls(t *testing.T) {
	t.Setenv("FAK_PROFILE", "")
	prior := adjudicator.Default.PolicySnapshot()
	t.Cleanup(func() { adjudicator.Default.SetPolicy(prior) })
	applyFloorWithProfile("", "")

	allowed := agent.ToolCall{ID: "call_read", Type: "function", Function: agent.Func{Name: "read_file", Arguments: `{"path":"main.go"}`}}
	denied := agent.ToolCall{ID: "call_rm", Type: "function", Function: agent.Func{Name: "Bash", Arguments: `{"command":"rm -rf /"}`}}
	tools := []map[string]any{
		{"type": "function", "function": map[string]any{"name": "read_file", "parameters": map[string]any{"type": "object"}}},
		{"type": "function", "function": map[string]any{"name": "Bash", "parameters": map[string]any{"type": "object"}}},
	}
	for _, mode := range []struct {
		name          string
		live, request bool
	}{
		{name: "buffered"},
		{name: "buffered_stream", request: true},
		{name: "live_stream", live: true, request: true},
	} {
		for _, proposal := range []struct {
			name       string
			calls      []agent.ToolCall
			wantKept   []string
			wantFinish string
		}{
			{name: "allowed_and_denied", calls: []agent.ToolCall{allowed, denied}, wantKept: []string{"call_read"}, wantFinish: "tool_calls"},
			{name: "denied_only", calls: []agent.ToolCall{denied}, wantFinish: "stop"},
		} {
			t.Run(mode.name+"/"+proposal.name, func(t *testing.T) {
				planner := &turnkeyProposalPlanner{stream: mode.live, calls: proposal.calls}
				turnkey := &turnkeyServer{planner: planner, engineID: "inkernel", plan: macfit.TurnkeyProfile{ContextBudgetTokens: 8192, Tier: macfit.ModelTier{ModelID: "local"}}}
				ts := httptest.NewServer(http.HandlerFunc(turnkey.handleChatCompletions))
				defer ts.Close()
				body, _ := json.Marshal(map[string]any{
					"model": "local", "stream": mode.request, "tools": tools,
					"messages": []map[string]any{{"role": "user", "content": "inspect the tree"}},
				})
				resp, err := http.Post(ts.URL, "application/json", bytes.NewReader(body))
				if err != nil {
					t.Fatal(err)
				}
				defer resp.Body.Close()
				raw, _ := io.ReadAll(resp.Body)
				if resp.StatusCode != http.StatusOK {
					t.Fatalf("status = %d: %s", resp.StatusCode, raw)
				}
				turn := readTurnkeyAdjudicatedTurn(t, raw, mode.request)
				if strings.Join(turn.callIDs, ",") != strings.Join(proposal.wantKept, ",") {
					t.Fatalf("tool calls on the wire = %v, want %v: %s", turn.callIDs, proposal.wantKept, raw)
				}
				if turn.finish != proposal.wantFinish {
					t.Fatalf("finish_reason = %q, want %q: %s", turn.finish, proposal.wantFinish, raw)
				}
				if len(turn.adjudications) != len(proposal.calls) {
					t.Fatalf("adjudications = %+v, want one per proposed call: %s", turn.adjudications, raw)
				}
				for _, adj := range turn.adjudications {
					wantAdmitted := adj.ToolCallID == allowed.ID
					if adj.Admitted != wantAdmitted {
						t.Fatalf("adjudication %+v admitted=%v, want %v", adj, adj.Admitted, wantAdmitted)
					}
					if !wantAdmitted && (adj.Verdict.Kind != "DENY" || adj.Verdict.Reason != abi.ReasonName(abi.ReasonPolicyBlock)) {
						t.Fatalf("denied call verdict = %+v, want DENY/POLICY_BLOCK", adj.Verdict)
					}
				}
			})
		}
	}
}

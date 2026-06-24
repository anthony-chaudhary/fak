package main

import (
	"context"
	"encoding/json"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/swebench"
)

// swebench_fleet.go is the integrator glue that lets the swebench "fleet" runner
// drive a model WITHOUT swebench (a foundation tier) depending on the chat client.
// It adapts agent.HTTPPlanner — the single outbound /v1/chat/completions client —
// to swebench.CodePlanner. cmd/fak is the only place that knows both, which keeps
// the layered-DAG (internal/architest) intact.

// httpCodePlanner adapts *agent.HTTPPlanner to swebench.CodePlanner.
type httpCodePlanner struct{ p *agent.HTTPPlanner }

// newFleetPlanner builds the fleet runner's planner from a gateway address and a
// model id (FAK_API_KEY supplies the key for an authenticated gateway).
func newFleetPlanner(gateway, model string) httpCodePlanner {
	return httpCodePlanner{p: agent.NewHTTPPlanner(gatewayBaseURL(gateway), model, os.Getenv("FAK_API_KEY"))}
}

// Model returns the model id the underlying HTTP planner posts to the gateway.
func (h httpCodePlanner) Model() string { return h.p.Model() }

func (h httpCodePlanner) Complete(ctx context.Context, msgs []swebench.ChatMessage, tools []swebench.ChatTool) (swebench.ChatTurn, error) {
	comp, err := h.p.Complete(ctx, toAgentMessages(msgs), toAgentTools(tools))
	if err != nil {
		return swebench.ChatTurn{}, err
	}
	return swebench.ChatTurn{Message: fromAgentMessage(comp.Message)}, nil
}

func toAgentMessages(msgs []swebench.ChatMessage) []agent.Message {
	out := make([]agent.Message, len(msgs))
	for i, m := range msgs {
		am := agent.Message{Role: m.Role, Content: m.Content, ToolCallID: m.ToolCallID, Name: m.Name}
		for _, tc := range m.ToolCalls {
			am.ToolCalls = append(am.ToolCalls, agent.ToolCall{
				ID: tc.ID, Type: "function", Function: agent.Func{Name: tc.Name, Arguments: tc.Args},
			})
		}
		out[i] = am
	}
	return out
}

func toAgentTools(tools []swebench.ChatTool) []agent.ToolDef {
	out := make([]agent.ToolDef, len(tools))
	for i, td := range tools {
		out[i] = agent.ToolDef{
			Type:     "function",
			Function: agent.ToolDefFunction{Name: td.Name, Description: td.Description, Parameters: json.RawMessage(td.Parameters)},
		}
	}
	return out
}

func fromAgentMessage(m agent.Message) swebench.ChatMessage {
	out := swebench.ChatMessage{Role: m.Role, Content: m.Content, ToolCallID: m.ToolCallID, Name: m.Name}
	for _, tc := range m.ToolCalls {
		out.ToolCalls = append(out.ToolCalls, swebench.ChatToolCall{ID: tc.ID, Name: tc.Function.Name, Args: tc.Function.Arguments})
	}
	return out
}

// gatewayBaseURL normalizes a gateway address into an OpenAI-compatible base URL.
// "localhost:8080" -> "http://localhost:8080/v1"; a full URL is preserved (and a
// missing /v1 segment appended). The OpenAI adapter posts to <base>/chat/completions.
func gatewayBaseURL(addr string) string {
	addr = strings.TrimSpace(addr)
	if addr == "" {
		return ""
	}
	if !strings.Contains(addr, "://") {
		addr = "http://" + addr
	}
	addr = strings.TrimRight(addr, "/")
	if !strings.Contains(addr, "/v1") {
		addr += "/v1"
	}
	return addr
}

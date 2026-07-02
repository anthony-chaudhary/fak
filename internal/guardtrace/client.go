package guardtrace

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// InboundRoute is the gateway HTTP route a wire posts its turns to.
func InboundRoute(provider string) string {
	switch provider {
	case "anthropic":
		return "/v1/messages"
	default:
		return "/v1/chat/completions"
	}
}

// BuildInboundRequest renders the per-turn request the CLIENT sends INTO the gateway, in
// the wire shape the route expects. Fixtures can provide a full client-side history in
// Turn.Messages; older fixtures get a minimal but well-formed default request with a
// system turn, one user turn, and the tool declarations for every tool the upcoming
// response will call. The gateway forwards this to the fake upstream, whose scripted
// response for this turn supplies the tool calls + usage.
func BuildInboundRequest(provider, model string, t Turn) ([]byte, error) {
	switch provider {
	case "anthropic":
		return buildAnthropicInbound(model, t)
	default:
		return buildOpenAIInbound(model, t)
	}
}

func toolDecls(t Turn) []string {
	seen := map[string]bool{}
	var names []string
	for _, c := range t.Calls {
		if !seen[c.Tool] {
			seen[c.Tool] = true
			names = append(names, c.Tool)
		}
	}
	return names
}

func buildAnthropicInbound(model string, t Turn) ([]byte, error) {
	tools := make([]map[string]any, 0)
	for _, name := range toolDecls(t) {
		tools = append(tools, map[string]any{
			"name":         name,
			"description":  "fixture tool",
			"input_schema": map[string]any{"type": "object"},
		})
	}
	system, messages := anthropicRequestMessages(t)
	req := map[string]any{
		"model":      model,
		"max_tokens": 1024,
		"system":     system,
		"messages":   messages,
		"tools":      tools,
	}
	return json.Marshal(req)
}

func buildOpenAIInbound(model string, t Turn) ([]byte, error) {
	tools := make([]map[string]any, 0)
	for _, name := range toolDecls(t) {
		tools = append(tools, map[string]any{
			"type": "function",
			"function": map[string]any{
				"name":        name,
				"description": "fixture tool",
				"parameters":  map[string]any{"type": "object"},
			},
		})
	}
	req := map[string]any{
		"model":    model,
		"messages": openAIRequestMessages(t),
		"tools":    tools,
	}
	return json.Marshal(req)
}

func anthropicRequestMessages(t Turn) (string, []map[string]any) {
	const defaultSystem = "You are a coding agent under fak guard."
	if len(t.Messages) == 0 {
		return defaultSystem, []map[string]any{
			{"role": "user", "content": "proceed with the next step"},
		}
	}
	system := ""
	messages := make([]map[string]any, 0, len(t.Messages))
	for _, m := range t.Messages {
		role := strings.ToLower(strings.TrimSpace(m.Role))
		content := m.Content
		if role == "system" {
			if system == "" {
				system = content
			} else {
				system += "\n\n" + content
			}
			continue
		}
		messages = append(messages, map[string]any{"role": role, "content": content})
	}
	if system == "" {
		system = defaultSystem
	}
	if len(messages) == 0 {
		messages = append(messages, map[string]any{"role": "user", "content": "proceed with the next step"})
	}
	return system, messages
}

func openAIRequestMessages(t Turn) []map[string]any {
	if len(t.Messages) == 0 {
		return []map[string]any{
			{"role": "system", "content": "You are a coding agent under fak guard."},
			{"role": "user", "content": "proceed with the next step"},
		}
	}
	messages := make([]map[string]any, 0, len(t.Messages))
	for _, m := range t.Messages {
		messages = append(messages, map[string]any{
			"role":    strings.ToLower(strings.TrimSpace(m.Role)),
			"content": m.Content,
		})
	}
	return messages
}

// PostTurn posts one turn's inbound request to the gateway at gatewayURL and returns the
// raw response body. traceHeader/trace, when non-empty, pin the session trace id so every
// turn shares one journal/session — the same X-Trace-Id a real wrapped agent would carry.
func PostTurn(client *http.Client, gatewayURL, traceHeader, trace, provider, model string, t Turn) ([]byte, int, error) {
	body, err := BuildInboundRequest(provider, model, t)
	if err != nil {
		return nil, 0, err
	}
	req, err := http.NewRequest("POST", gatewayURL+InboundRoute(provider), bytes.NewReader(body))
	if err != nil {
		return nil, 0, err
	}
	req.Header.Set("Content-Type", "application/json")
	if traceHeader != "" && trace != "" {
		req.Header.Set(traceHeader, trace)
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, 0, err
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, resp.StatusCode, err
	}
	if resp.StatusCode != http.StatusOK {
		return raw, resp.StatusCode, fmt.Errorf("guardtrace: gateway returned %d: %s", resp.StatusCode, truncate(string(raw), 200))
	}
	return raw, resp.StatusCode, nil
}

// ResponseAdjudication is the per-call verdict the gateway returned in its `fak`
// extension, decoded from EITHER wire's response into one shape so the test and the CLI
// read verdicts uniformly.
type ResponseAdjudication struct {
	Tool     string
	Admitted bool
	Kind     string
	Reason   string
}

// DecodeAdjudications extracts the gateway's per-call verdicts from a turn response on
// either wire. Both the Anthropic and the OpenAI response carry the same `fak` extension
// shape, so one decoder serves both.
func DecodeAdjudications(raw []byte) ([]ResponseAdjudication, error) {
	var env struct {
		Fak *struct {
			Adjudications []struct {
				Tool     string `json:"tool"`
				Admitted bool   `json:"admitted"`
				Verdict  struct {
					Kind   string `json:"kind"`
					Reason string `json:"reason"`
				} `json:"verdict"`
			} `json:"adjudications"`
		} `json:"fak"`
	}
	if err := json.Unmarshal(raw, &env); err != nil {
		return nil, fmt.Errorf("guardtrace: decode fak extension: %w", err)
	}
	if env.Fak == nil {
		return nil, nil
	}
	out := make([]ResponseAdjudication, 0, len(env.Fak.Adjudications))
	for _, a := range env.Fak.Adjudications {
		out = append(out, ResponseAdjudication{
			Tool:     a.Tool,
			Admitted: a.Admitted,
			Kind:     a.Verdict.Kind,
			Reason:   a.Verdict.Reason,
		})
	}
	return out, nil
}

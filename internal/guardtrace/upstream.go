package guardtrace

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
)

// FakeUpstream is an httptest.Server that stands in for a real model provider. It
// replays a fixture turn-by-turn: each inbound completion request pops the next turn
// and answers with a provider-correct response carrying that turn's tool_use /
// tool_calls blocks AND its usage block. Pointed at by the gateway's Config.BaseURL,
// it drives the REAL proxy planner + parse path, so the gateway adjudicates genuine
// upstream-shaped tool calls and accounts genuine provider-reported token usage —
// "a trace that leads to token work", with no API key and no GPU.
type FakeUpstream struct {
	*httptest.Server
	provider string // "anthropic" | "openai"
	model    string

	mu   sync.Mutex
	turn int // index of the next turn to serve
	hits int // total requests served (a turn may be retried; hits >= served turns)
}

// NewFakeUpstream starts a fake provider upstream for the given wire ("anthropic" or
// "openai") that replays f's turns. The caller closes it via Close().
func NewFakeUpstream(provider, model string, f *Fixture) *FakeUpstream {
	u := &FakeUpstream{provider: strings.ToLower(provider), model: model}
	u.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		u.serve(w, r, f)
	}))
	return u
}

// Hits is the number of completion requests served so far (for diagnostics).
func (u *FakeUpstream) Hits() int {
	u.mu.Lock()
	defer u.mu.Unlock()
	return u.hits
}

func (u *FakeUpstream) serve(w http.ResponseWriter, r *http.Request, f *Fixture) {
	// healthz / models preflight some callers do — answer benignly.
	switch {
	case strings.HasSuffix(r.URL.Path, "/healthz"):
		w.WriteHeader(http.StatusOK)
		return
	case strings.HasSuffix(r.URL.Path, "/v1/models") || strings.HasSuffix(r.URL.Path, "/models"):
		_ = json.NewEncoder(w).Encode(map[string]any{"data": []map[string]string{{"id": u.model}}})
		return
	}

	u.mu.Lock()
	idx := u.turn
	if idx >= len(f.Turns) {
		idx = len(f.Turns) - 1 // clamp: re-serve the last turn if asked for more
	}
	u.turn = idx + 1
	u.hits++
	u.mu.Unlock()

	turn := f.Turns[idx]
	body, err := u.renderResponse(turn)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write(body)
}

// renderResponse marshals one turn as the provider's completion response shape. The
// shapes are the EXACT JSON the proxy planner's per-provider adapter parses
// (internal/agent/adapters.go: anthropicResponse / openAIResponse), so the fake
// exercises real parsing, not a shortcut.
func (u *FakeUpstream) renderResponse(t Turn) ([]byte, error) {
	switch u.provider {
	case "anthropic":
		return u.renderAnthropic(t)
	case "openai", "xai", "":
		return u.renderOpenAI(t)
	default:
		return nil, fmt.Errorf("guardtrace: unknown upstream provider %q", u.provider)
	}
}

func (u *FakeUpstream) renderAnthropic(t Turn) ([]byte, error) {
	type block struct {
		Type  string          `json:"type"`
		Text  string          `json:"text,omitempty"`
		ID    string          `json:"id,omitempty"`
		Name  string          `json:"name,omitempty"`
		Input json.RawMessage `json:"input,omitempty"`
	}
	blocks := make([]block, 0, len(t.Calls)+1)
	blocks = append(blocks, block{Type: "text", Text: "Working on it."})
	for _, c := range t.Calls {
		blocks = append(blocks, block{
			Type:  "tool_use",
			ID:    c.ID,
			Name:  c.Tool,
			Input: json.RawMessage(c.ArgString()),
		})
	}
	resp := map[string]any{
		"id":          "msg_guardtrace",
		"type":        "message",
		"role":        "assistant",
		"model":       u.model,
		"content":     blocks,
		"stop_reason": "tool_use",
		"usage": map[string]int{
			"input_tokens":                t.Usage.InputTokens,
			"output_tokens":               t.Usage.OutputTokens,
			"cache_read_input_tokens":     t.Usage.CacheReadInputTokens,
			"cache_creation_input_tokens": t.Usage.CacheCreationInputTokens,
		},
	}
	return json.Marshal(resp)
}

func (u *FakeUpstream) renderOpenAI(t Turn) ([]byte, error) {
	type fn struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"` // a JSON-encoded string of the inner object
	}
	type toolCall struct {
		ID       string `json:"id"`
		Type     string `json:"type"`
		Function fn     `json:"function"`
	}
	calls := make([]toolCall, 0, len(t.Calls))
	for _, c := range t.Calls {
		calls = append(calls, toolCall{
			ID:       c.ID,
			Type:     "function",
			Function: fn{Name: c.Tool, Arguments: c.ArgString()},
		})
	}
	resp := map[string]any{
		"id":     "chatcmpl_guardtrace",
		"object": "chat.completion",
		"model":  u.model,
		"choices": []map[string]any{{
			"index":         0,
			"finish_reason": "tool_calls",
			"message": map[string]any{
				"role":       "assistant",
				"content":    "",
				"tool_calls": calls,
			},
		}},
		"usage": map[string]any{
			"prompt_tokens":           t.Usage.InputTokens + t.Usage.CacheReadInputTokens + t.Usage.CacheCreationInputTokens,
			"completion_tokens":       t.Usage.OutputTokens,
			"total_tokens":            t.Usage.InputTokens + t.Usage.CacheReadInputTokens + t.Usage.CacheCreationInputTokens + t.Usage.OutputTokens,
			"cache_read_input_tokens": t.Usage.CacheReadInputTokens,
			"prompt_tokens_details":   map[string]int{"cached_tokens": t.Usage.CacheReadInputTokens},
		},
	}
	return json.Marshal(resp)
}

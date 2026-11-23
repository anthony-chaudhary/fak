package gateway

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

type providerStubPlanner struct {
	stubPlanner
	provider agent.Provider
}

func (p providerStubPlanner) Provider() agent.Provider { return p.provider }

func postResponsesRaw(t *testing.T, base string, body map[string]any) (int, []byte) {
	t.Helper()
	raw, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := http.Post(base+"/v1/responses", "application/json", bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	out, _ := io.ReadAll(resp.Body)
	return resp.StatusCode, out
}

func TestResponsesHostedToolForcedOnChatUpstreamRefused(t *testing.T) {
	cases := []struct {
		name    string
		planner agent.Planner
		choice  any
		want    string
	}{
		{name: "unknown provider", planner: stubPlanner{comp: hostedToolTextCompletion("x")}, choice: map[string]any{"type": "web_search_preview"}, want: "web_search_preview"},
		{name: "chat provider", planner: providerStubPlanner{stubPlanner{comp: hostedToolTextCompletion("x")}, agent.ProviderOpenAI}, choice: map[string]any{"type": "file_search"}, want: "file_search"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			srv := newTestServer(t)
			srv.planner = c.planner
			ts := httptest.NewServer(srv.Handler())
			defer ts.Close()
			code, body := postResponsesRaw(t, ts.URL, map[string]any{
				"model":       "m",
				"input":       "search for it",
				"tools":       []map[string]any{{"type": c.want}},
				"tool_choice": c.choice,
			})
			if code != http.StatusBadRequest {
				t.Fatalf("status = %d, want 400: %s", code, body)
			}
			var env struct {
				Error struct {
					Code    string `json:"code"`
					Message string `json:"message"`
				} `json:"error"`
			}
			if err := json.Unmarshal(body, &env); err != nil {
				t.Fatalf("decode error envelope: %v: %s", err, body)
			}
			if env.Error.Code != "hosted_tool_unsupported" || !strings.Contains(env.Error.Message, c.want) {
				t.Fatalf("error = %+v, want code hosted_tool_unsupported naming %q", env.Error, c.want)
			}
		})
	}
}

func TestResponsesHostedToolNotForcedStillServed(t *testing.T) {
	cases := []struct {
		name    string
		planner agent.Planner
		tools   []map[string]any
		choice  any
	}{
		{name: "forced function tool", planner: stubPlanner{comp: hostedToolTextCompletion("ok")}, tools: []map[string]any{{"type": "function", "name": "lookup"}}, choice: map[string]any{"type": "function", "name": "lookup"}},
		{name: "optional hosted tool", planner: stubPlanner{comp: hostedToolTextCompletion("ok")}, tools: []map[string]any{{"type": "web_search"}}, choice: "auto"},
		{name: "required hosted tool", planner: stubPlanner{comp: hostedToolTextCompletion("ok")}, tools: []map[string]any{{"type": "web_search"}}, choice: "required"},
		{name: "forced hosted on responses upstream", planner: providerStubPlanner{stubPlanner{comp: hostedToolTextCompletion("ok")}, agent.ProviderOpenAIResponses}, tools: []map[string]any{{"type": "web_search"}}, choice: map[string]any{"type": "web_search"}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			srv := newTestServer(t)
			srv.planner = c.planner
			ts := httptest.NewServer(srv.Handler())
			defer ts.Close()
			code, body := postResponsesRaw(t, ts.URL, map[string]any{
				"model":       "m",
				"input":       "hello",
				"tools":       c.tools,
				"tool_choice": c.choice,
			})
			if code != http.StatusOK {
				t.Fatalf("status = %d, want 200: %s", code, body)
			}
		})
	}
}

func hostedToolTextCompletion(text string) *agent.Completion {
	return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: text}, FinishReason: "stop"}
}

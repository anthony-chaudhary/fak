package trajctl

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

// TestGatewayJudgeClientCannedResponse is the witness the issue names — "a
// gateway test with a canned model response". A stub OpenAI-compatible endpoint
// stands in for the gateway upstream; the test asserts the client sends the
// pinned-schema, forced-tool-choice request shape with the budget cap as
// max_tokens, and folds the canned tool-call reply into a JudgeVerdict + usage.
func TestGatewayJudgeClientCannedResponse(t *testing.T) {
	var gotBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("unexpected path %q", r.URL.Path)
		}
		if auth := r.Header.Get("Authorization"); auth != "Bearer test-key" {
			t.Errorf("Authorization = %q, want Bearer test-key", auth)
		}
		raw, _ := io.ReadAll(r.Body)
		if err := json.Unmarshal(raw, &gotBody); err != nil {
			t.Fatalf("request body: %v", err)
		}
		// Canned model response: the forced tool call carrying the verdict JSON.
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{
		  "choices": [{"message": {"tool_calls": [
		    {"function": {"name": "emit_verdict", "arguments": "{\"progress\":0.6,\"met\":false,\"rationale\":\"three of five docs migrated\"}"}}
		  ]}}],
		  "usage": {"total_tokens": 143}
		}`)
	}))
	defer srv.Close()

	client := &GatewayJudgeClient{BaseURL: srv.URL + "/v1", APIKey: "test-key", Model: "judge-model"}
	verdict, usage, err := client.Judge(JudgeRequest{
		Objective: "Migrate the observability docs.",
		State:     "prior_scores=1 best_progress=0.20 sessions=0",
		MaxTokens: 256,
	})
	if err != nil {
		t.Fatalf("Judge: %v", err)
	}
	if verdict.Progress != 0.6 || verdict.Met {
		t.Errorf("verdict = %+v, want progress 0.6 not-met", verdict)
	}
	if verdict.Rationale != "three of five docs migrated" {
		t.Errorf("rationale = %q", verdict.Rationale)
	}
	if usage.Tokens != 143 {
		t.Errorf("usage tokens = %d, want 143", usage.Tokens)
	}

	// Request-shape assertions: pinned schema, forced tool choice, budget cap.
	if got := gotBody["max_tokens"]; got != float64(256) {
		t.Errorf("max_tokens = %v, want 256 (budget cap forwarded)", got)
	}
	choice, ok := gotBody["tool_choice"].(map[string]any)
	if !ok {
		t.Fatalf("tool_choice not an object: %v", gotBody["tool_choice"])
	}
	fn, _ := choice["function"].(map[string]any)
	if choice["type"] != "function" || fn["name"] != judgeToolName {
		t.Errorf("tool_choice = %v, want forced function %q", choice, judgeToolName)
	}
	tools, ok := gotBody["tools"].([]any)
	if !ok || len(tools) != 1 {
		t.Fatalf("tools = %v, want exactly one advertised tool", gotBody["tools"])
	}
	tool := tools[0].(map[string]any)["function"].(map[string]any)
	if tool["name"] != judgeToolName {
		t.Errorf("advertised tool name = %v, want %q", tool["name"], judgeToolName)
	}
	params, ok := tool["parameters"].(map[string]any)
	if !ok {
		t.Fatalf("tool parameters not an object: %v", tool["parameters"])
	}
	props, _ := params["properties"].(map[string]any)
	if _, has := props["progress"]; !has {
		t.Errorf("pinned schema missing 'progress' property: %v", params)
	}
}

// TestGatewayJudgeClientErrors covers the fail-closed shapes the scorer relies
// on: a non-200 status and a reply with no tool call both surface as errors.
func TestGatewayJudgeClientErrors(t *testing.T) {
	t.Run("non-200", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			http.Error(w, "over quota", http.StatusTooManyRequests)
		}))
		defer srv.Close()
		client := &GatewayJudgeClient{BaseURL: srv.URL, Model: "m"}
		if _, _, err := client.Judge(JudgeRequest{Objective: "o", MaxTokens: 64}); err == nil {
			t.Fatal("want error on 429, got nil")
		}
	})
	t.Run("no tool call", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			io.WriteString(w, `{"choices":[{"message":{"content":"I refuse"}}],"usage":{"total_tokens":5}}`)
		}))
		defer srv.Close()
		client := &GatewayJudgeClient{BaseURL: srv.URL, Model: "m"}
		if _, _, err := client.Judge(JudgeRequest{Objective: "o", MaxTokens: 64}); err == nil {
			t.Fatal("want error when no tool call, got nil")
		}
	})
	t.Run("empty base URL", func(t *testing.T) {
		client := &GatewayJudgeClient{Model: "m"}
		if _, _, err := client.Judge(JudgeRequest{Objective: "o", MaxTokens: 64}); err == nil {
			t.Fatal("want error on empty base URL, got nil")
		}
	})
}

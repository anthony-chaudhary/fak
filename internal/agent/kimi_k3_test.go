package agent

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
)

func TestKimiK3MoonshotWireAndReasoningReplay(t *testing.T) {
	var mu sync.Mutex
	var requests []map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Errorf("path = %q", r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer moonshot-test-key" {
			t.Errorf("authorization = %q", got)
		}
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatal(err)
		}
		var decoded map[string]any
		if err := json.Unmarshal(body, &decoded); err != nil {
			t.Fatalf("request JSON: %v", err)
		}
		mu.Lock()
		requests = append(requests, decoded)
		n := len(requests)
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		if n == 1 {
			_, _ = io.WriteString(w, `{"model":"kimi-k3","choices":[{"message":{"role":"assistant","reasoning_content":"checked the constraints","content":"first"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}`)
			return
		}
		_, _ = io.WriteString(w, `{"model":"kimi-k3","choices":[{"message":{"role":"assistant","content":"second"},"finish_reason":"stop"}]}`)
	}))
	defer server.Close()

	planner := NewHTTPPlanner(server.URL, "kimi-k3", "moonshot-test-key")
	topP := 0.4
	first, err := planner.Complete(context.Background(), []Message{{Role: RoleUser, Content: "one"}}, nil,
		WithTemperature(func() *float64 { v := 0.2; return &v }()), WithTopP(&topP))
	if err != nil {
		t.Fatalf("first Complete: %v", err)
	}
	if first.Message.ReasoningContent != "checked the constraints" {
		t.Fatalf("reasoning_content = %q", first.Message.ReasoningContent)
	}
	_, err = planner.Complete(context.Background(), []Message{
		{Role: RoleUser, Content: "one"},
		first.Message,
		{Role: RoleUser, Content: "two"},
	}, nil)
	if err != nil {
		t.Fatalf("second Complete: %v", err)
	}

	mu.Lock()
	defer mu.Unlock()
	if len(requests) != 2 {
		t.Fatalf("requests = %d, want 2", len(requests))
	}
	for i, req := range requests {
		if req["model"] != "kimi-k3" || req["reasoning_effort"] != "max" {
			t.Errorf("request %d model/effort = %v/%v", i, req["model"], req["reasoning_effort"])
		}
		for _, forbidden := range []string{"temperature", "top_p", "thinking"} {
			if _, ok := req[forbidden]; ok {
				t.Errorf("request %d unexpectedly contains %s: %#v", i, forbidden, req[forbidden])
			}
		}
	}
	messages, ok := requests[1]["messages"].([]any)
	if !ok || len(messages) != 3 {
		t.Fatalf("second messages = %#v", requests[1]["messages"])
	}
	assistant := messages[1].(map[string]any)
	if assistant["reasoning_content"] != "checked the constraints" || assistant["content"] != "first" {
		t.Fatalf("assistant replay = %#v", assistant)
	}
}

func TestKimiK3RejectsUnsupportedReasoningEffort(t *testing.T) {
	planner := NewHTTPPlanner("http://127.0.0.1:1", "moonshotai/kimi-k3", "")
	if err := planner.SetExtraBodyJSON(`{"reasoning_effort":"high"}`); err != nil {
		t.Fatal(err)
	}
	_, err := planner.Complete(context.Background(), []Message{{Role: RoleUser, Content: "x"}}, nil)
	if err == nil || err.Error() != `kimi k3 reasoning_effort must be "max", got high` {
		t.Fatalf("error = %v", err)
	}
}

func TestNonKimiOpenAIKeepsSamplingFields(t *testing.T) {
	adapter, err := NewTranscriptAdapter(ProviderOpenAI)
	if err != nil {
		t.Fatal(err)
	}
	topP := 0.8
	body, err := adapter.MarshalRequest(adapterRequest{Model: "gpt-test", Temperature: 0.3, TopP: &topP})
	if err != nil {
		t.Fatal(err)
	}
	var got map[string]any
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatal(err)
	}
	if got["temperature"] != 0.3 || got["top_p"] != 0.8 {
		t.Fatalf("sampling fields = %#v", got)
	}
}

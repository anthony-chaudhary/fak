package agent

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
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

func TestKimiK3StreamingWireAndReasoning(t *testing.T) {
	var request map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(body, &request); err != nil {
			t.Fatalf("request JSON: %v", err)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: {\"model\":\"kimi-k3\",\"choices\":[{\"delta\":{\"role\":\"assistant\",\"reasoning_content\":\"inspect \"}}]}\n\n")
		_, _ = io.WriteString(w, "data: {\"model\":\"kimi-k3\",\"choices\":[{\"delta\":{\"reasoning_content\":\"constraints\",\"content\":\"pong\"},\"finish_reason\":\"stop\"}]}\n\n")
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	defer server.Close()

	planner := NewHTTPPlanner(server.URL, "kimi-k3", "moonshot-test-key")
	var streamed strings.Builder
	completion, err := planner.CompleteStream(context.Background(), func(delta string) error {
		streamed.WriteString(delta)
		return nil
	}, []Message{{Role: RoleUser, Content: "reply pong"}}, nil)
	if err != nil {
		t.Fatalf("CompleteStream: %v", err)
	}
	if streamed.String() != "pong" || completion.Message.Content != "pong" {
		t.Fatalf("stream/content = %q/%q", streamed.String(), completion.Message.Content)
	}
	if completion.Message.ReasoningContent != "inspect constraints" {
		t.Fatalf("reasoning_content = %q", completion.Message.ReasoningContent)
	}
	if request["reasoning_effort"] != "max" || request["stream"] != true {
		t.Fatalf("K3 stream controls = %#v", request)
	}
	for _, forbidden := range []string{"temperature", "top_p", "thinking"} {
		if _, ok := request[forbidden]; ok {
			t.Errorf("stream request unexpectedly contains %s", forbidden)
		}
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

func TestKimiK31ModelRecognition(t *testing.T) {
	recognized := []string{
		"kimi-k3.1-pro",
		"kimi-k3.1-pro-highspeed",
		"k3.1-pro",
		"k3.1-pro-highspeed",
		"moonshotai/kimi-k3.1-pro",
		"MOONSHOTAI/Kimi-K3.1-Pro-HighSpeed",
		"  k3.1-pro  ",
	}
	for _, id := range recognized {
		if !isKimiK31Model(id) {
			t.Errorf("isKimiK31Model(%q) = false, want true", id)
		}
		if isKimiK3Model(id) {
			t.Errorf("isKimiK3Model(%q) = true, want false (K3.1 is not K3)", id)
		}
	}
	notRecognized := []string{
		"kimi-k3", "k3",
		"kimi-k3.1", "k3.1", // bare family names are not documented API IDs
		"kimi-k3.1-flash", // hypothetical sibling variant
		"kimi-k3.2-pro",   // next-generation spelling
		"gpt-test", "",
	}
	for _, id := range notRecognized {
		if isKimiK31Model(id) {
			t.Errorf("isKimiK31Model(%q) = true, want false", id)
		}
	}
	// K3 recognition stays an exact closed set with no prefix bleed onto K3.1.
	for _, id := range []string{"kimi-k3", "k3", "moonshotai/kimi-k3"} {
		if !isKimiK3Model(id) {
			t.Errorf("isKimiK3Model(%q) = false, want true", id)
		}
	}
	if isKimiK3Model("kimi-k3.1-pro") || isKimiK3Model("kimi-k3.1") {
		t.Error("isKimiK3Model must not match K3.1 spellings")
	}
}

func TestKimiK31ProAndHighspeedEffortContract(t *testing.T) {
	cases := []struct {
		model      string
		extraBody  string
		wantErr    string
		wantEffort string
	}{
		{model: "kimi-k3.1-pro", extraBody: `{"reasoning_effort":"low"}`, wantEffort: "low"},
		{model: "kimi-k3.1-pro", extraBody: `{"reasoning_effort":"high"}`, wantEffort: "high"},
		{model: "kimi-k3.1-pro", extraBody: `{"reasoning_effort":"max"}`, wantEffort: "max"},
		{model: "kimi-k3.1-pro", extraBody: `{}`, wantEffort: "max"}, // absent defaults to max
		{model: "kimi-k3.1-pro", extraBody: `{"reasoning_effort":"medium"}`, wantErr: `kimi k3.1 reasoning_effort must be "low", "high" or "max", got medium`},
		// Highspeed is a serving mode of the same weights: identical effort mapping.
		{model: "kimi-k3.1-pro-highspeed", extraBody: `{"reasoning_effort":"high"}`, wantEffort: "high"},
		{model: "kimi-k3.1-pro-highspeed", extraBody: `{}`, wantEffort: "max"},
	}
	for _, tc := range cases {
		t.Run(tc.model+" "+tc.extraBody, func(t *testing.T) {
			var got map[string]any
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, _ := io.ReadAll(r.Body)
				_ = json.Unmarshal(body, &got)
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, `{"model":"`+tc.model+`","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
			}))
			defer server.Close()

			planner := NewHTTPPlanner(server.URL, tc.model, "moonshot-test-key")
			if err := planner.SetExtraBodyJSON(tc.extraBody); err != nil {
				t.Fatal(err)
			}
			_, err := planner.Complete(context.Background(), []Message{{Role: RoleUser, Content: "x"}}, nil)
			if tc.wantErr != "" {
				if err == nil || err.Error() != tc.wantErr {
					t.Fatalf("error = %v, want %q", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("Complete: %v", err)
			}
			if got["model"] != tc.model || got["reasoning_effort"] != tc.wantEffort {
				t.Fatalf("model/effort = %v/%v, want %q/%q", got["model"], got["reasoning_effort"], tc.model, tc.wantEffort)
			}
			for _, forbidden := range []string{"temperature", "top_p", "thinking"} {
				if _, ok := got[forbidden]; ok {
					t.Errorf("request unexpectedly contains %s: %#v", forbidden, got[forbidden])
				}
			}
		})
	}
}

func TestKimiUnknownVariantPassesThroughUnchanged(t *testing.T) {
	var got map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &got)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"model":"kimi-k3.1","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
	}))
	defer server.Close()

	planner := NewHTTPPlanner(server.URL, "kimi-k3.1", "moonshot-test-key")
	topP := 0.4
	_, err := planner.Complete(context.Background(), []Message{{Role: RoleUser, Content: "x"}}, nil,
		WithTemperature(func() *float64 { v := 0.2; return &v }()), WithTopP(&topP))
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	// Bare "kimi-k3.1" is not a documented API ID: the K3/K3.1 contract must not
	// apply, so sampling fields survive and no effort field is injected.
	if got["model"] != "kimi-k3.1" {
		t.Fatalf("model = %v", got["model"])
	}
	if got["temperature"] != 0.2 || got["top_p"] != 0.4 {
		t.Fatalf("sampling fields = %#v", got)
	}
	if _, ok := got["reasoning_effort"]; ok {
		t.Fatalf("reasoning_effort unexpectedly injected: %#v", got["reasoning_effort"])
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

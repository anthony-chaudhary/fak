package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

func TestOpenAIEndpointCapturesStreamingUsageAndSharedBase(t *testing.T) {
	var calls atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("path=%s", r.URL.Path)
			http.NotFound(w, r)
			return
		}
		var request chatRequest
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Errorf("decode: %v", err)
			http.Error(w, err.Error(), 400)
			return
		}
		if len(request.Messages) != 2 || request.Messages[0].Role != "system" || !strings.Contains(request.Messages[0].Content, "bounded micro-context") {
			t.Errorf("messages=%+v", request.Messages)
		}
		calls.Add(1)
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":20,\"completion_tokens\":1,\"prompt_tokens_details\":{\"cached_tokens\":10}}}\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer server.Close()
	r, err := run(context.Background(), config{Contexts: 4, Workers: 2, Endpoint: server.URL, Model: "test-model", Provider: "test", Hardware: "cpu"})
	if err != nil {
		t.Fatal(err)
	}
	if r.Completed != 4 || r.TurnCount != 4 || calls.Load() != 4 {
		t.Fatalf("report=%+v calls=%d", r, calls.Load())
	}
	if r.PromptTokens != 80 || r.CompletionTokens != 4 || r.CachedPromptTokens != 40 || r.UsageResponses != 4 {
		t.Fatalf("usage=%+v", r)
	}
	if r.Mode != "openai-compatible" || r.Model != "test-model" || r.TTFTP95MS < 0 {
		t.Fatalf("provenance=%+v", r)
	}
}

package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
)

func TestLiveModelAdapterUsesSharedProviderGateway(t *testing.T) {
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Errorf("path = %q", r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Errorf("authorization = %q", got)
		}
		calls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []any{map[string]any{"message": map[string]any{"role": "assistant", "content": "bounded evidence accepted"}, "finish_reason": "stop"}},
			"usage":   map[string]any{"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14},
		})
	}))
	defer server.Close()

	planner, err := newLivePlanner("openai", server.URL, "fixture-live", "test-key")
	if err != nil {
		t.Fatal(err)
	}
	got, err := runWithPlanner(context.Background(), planner, "live", "Plan the next repository harness task.")
	if err != nil {
		t.Fatal(err)
	}
	if calls.Load() != 6 || got.Execution.Mode != "live" || got.Execution.Model != "fixture-live" || got.Execution.Completions != 6 || got.Execution.NonemptyResponses != 6 {
		t.Fatalf("execution = %#v, calls=%d", got.Execution, calls.Load())
	}
	if got.Execution.PromptTokens != 66 || got.Execution.CompletionTokens != 18 {
		t.Fatalf("usage = %#v", got.Execution)
	}
	if err := check(got); err != nil {
		t.Fatal(err)
	}
}

func TestLiveModelAdapterRefusesIncompleteConfiguration(t *testing.T) {
	if _, err := newLivePlanner("openai", "", "model", "key"); err == nil {
		t.Fatal("empty base URL accepted")
	}
	if _, err := newLivePlanner("openai", "http://example.invalid/v1", "", "key"); err == nil {
		t.Fatal("empty model accepted")
	}
}

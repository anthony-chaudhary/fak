package agent

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

// fak-test:justify why=regression when=changed:internal/agent/**
// fak-test:runtime fast est=1s
func TestOpsProviderConformanceEffortWire(t *testing.T) {
	captured := make(chan map[string]any, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
			http.Error(w, "invalid JSON", http.StatusBadRequest)
			return
		}
		captured <- body
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"fixture","object":"chat.completion","model":"fixture","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
	}))
	defer srv.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	p := NewHTTPPlanner(srv.URL, "fixture", "")
	if _, err := p.Complete(ctx, []Message{{Role: RoleUser, Content: "hi"}}, nil, WithReasoningEffort("none")); err != nil {
		t.Fatalf("complete fixture: %v", err)
	}
	select {
	case got := <-captured:
		if got["reasoning_effort"] != "none" {
			t.Fatalf("reasoning_effort = %v, want none", got["reasoning_effort"])
		}
	default:
		t.Fatal("fixture received no request")
	}
}

// fak-test:justify why=regression when=changed:internal/agent/**
// fak-test:runtime fast est=1s
func TestOpsProviderConformanceDefaultsAndCaps(t *testing.T) {
	p := NewHTTPPlanner("http://127.0.0.1:1/v1", "fixture", "")
	messages := []Message{{Role: RoleUser, Content: "hi"}}
	read := func() map[string]json.RawMessage {
		t.Helper()
		req, release, err := p.PrepareRequest(context.Background(), messages, nil, true, WithMaxTokens(512))
		if err != nil {
			t.Fatal(err)
		}
		defer release()
		defer req.Body.Close()
		replay, err := req.GetBody()
		if err != nil {
			t.Fatal(err)
		}
		defer replay.Close()
		actual, _ := io.ReadAll(req.Body)
		copy, _ := io.ReadAll(replay)
		if string(actual) != string(copy) || req.Header.Get("Accept") != "text/event-stream" {
			t.Fatal("prepared request is not replayable SSE")
		}
		release()
		var fields map[string]json.RawMessage
		if json.Unmarshal(actual, &fields) != nil {
			t.Fatal("invalid request")
		}
		return fields
	}
	legacy := read()
	if string(legacy["max_tokens"]) != "512" || legacy["max_completion_tokens"] != nil || legacy["reasoning_effort"] != nil || legacy["tool_choice"] != nil {
		t.Fatal("legacy omitted defaults changed")
	}
	p.MaxTokensCap = 128
	p.CompletionTokenField = "max_completion_tokens"
	modern := read()
	if string(modern["max_completion_tokens"]) != "128" || modern["max_tokens"] != nil {
		t.Fatal("selected cap field lost the planner ceiling")
	}
	for _, extra := range []string{`{"max_tokens":9}`, `{"max_completion_tokens":9}`} {
		p.ExtraBody = json.RawMessage(extra)
		if _, _, err := p.PrepareRequest(context.Background(), messages, nil, true); err == nil {
			t.Fatal("conflicting cap was accepted")
		}
	}
	p.ExtraBody = nil
	p.CompletionTokenField = "other"
	if _, _, err := p.PrepareRequest(context.Background(), messages, nil, true); err == nil {
		t.Fatal("unknown cap field was accepted")
	}
}

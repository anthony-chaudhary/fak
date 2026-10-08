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

// qwenTemplateServer mimics llama.cpp --jinja serving a Qwen3.x chat template:
// a system message anywhere but the leading run is a 500 (strix3, 2026-10-07).
func qwenTemplateServer(t *testing.T, seen *[][]Message, mu *sync.Mutex) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req struct {
			Messages []Message `json:"messages"`
		}
		if err := json.Unmarshal(body, &req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		mu.Lock()
		*seen = append(*seen, req.Messages)
		mu.Unlock()
		if !systemOnlyLeading(req.Messages) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = w.Write([]byte(`{"error":{"code":500,"message":"Error: Jinja Exception: System message must be at the beginning.","type":"server_error"}}`))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`))
	}))
	t.Cleanup(srv.Close)
	return srv
}

func systemOnlyLeading(messages []Message) bool {
	inHead := true
	for _, m := range messages {
		system := m.Role == "system" || m.Role == "developer"
		if !system {
			inHead = false
			continue
		}
		if !inHead {
			return false
		}
	}
	return true
}

func wireMessages(t *testing.T, messages []Message) []Message {
	t.Helper()
	adapter, err := NewTranscriptAdapter(ProviderOpenAI)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := adapter.MarshalRequest(adapterRequest{Model: "m", Messages: messages})
	if err != nil {
		t.Fatal(err)
	}
	var req struct {
		Messages []Message `json:"messages"`
	}
	if err := json.Unmarshal(raw, &req); err != nil {
		t.Fatal(err)
	}
	return req.Messages
}

// fak-test:runtime fast est=50ms lane=default
func TestOpenAIWire_MidConversationSystemReachesQwenTemplate(t *testing.T) {
	t.Setenv("FAK_PLANNER_MAX_ATTEMPTS", "1")
	var (
		mu   sync.Mutex
		seen [][]Message
	)
	srv := qwenTemplateServer(t, &seen, &mu)
	p := NewHTTPPlanner(srv.URL, "Qwen3.8-27B-UD-Q2_K_XL", "")
	conversation := []Message{
		{Role: "system", Content: "base prompt"},
		{Role: "user", Content: "fix the bug"},
		{Role: "assistant", Content: "looking"},
		{Role: "system", Content: "objective changed: cap the delay at 1000"},
	}
	comp, err := p.Complete(context.Background(), conversation, nil)
	if err != nil {
		t.Fatalf("Complete with a mid-conversation system message: %v", err)
	}
	if comp.Message.Content != "ok" {
		t.Fatalf("content = %q, want ok", comp.Message.Content)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(seen) != 1 {
		t.Fatalf("upstream requests = %d, want 1", len(seen))
	}
	last := seen[0][len(seen[0])-1]
	if last.Role != "user" || !strings.Contains(last.Content, "objective changed: cap the delay at 1000") {
		t.Fatalf("late directive not carried in a trailing user turn: %+v", last)
	}
	if conversation[3].Role != "system" {
		t.Fatalf("caller conversation mutated: %+v", conversation[3])
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestOpenAIWire_LateSystemKeepsToolAdjacencyAndOrder(t *testing.T) {
	got := wireMessages(t, []Message{
		{Role: "system", Content: "a"},
		{Role: "developer", Content: "b"},
		{Role: "user", Content: "u1"},
		{Role: "assistant", ToolCalls: []ToolCall{{ID: "c1", Type: "function", Function: Func{Name: "read", Arguments: "{}"}}}},
		{Role: "system", Content: "s1"},
		{Role: "tool", ToolCallID: "c1", Content: "r1"},
		{Role: "system", Content: "s2"},
		{Role: "user", Content: "u2"},
	})
	if !systemOnlyLeading(got) {
		t.Fatalf("system message outside the leading run: %+v", got)
	}
	roles := make([]string, len(got))
	for i, m := range got {
		roles[i] = m.Role
	}
	want := []string{"system", "developer", "user", "assistant", "tool", "user"}
	if strings.Join(roles, ",") != strings.Join(want, ",") {
		t.Fatalf("roles = %v, want %v", roles, want)
	}
	tail := got[len(got)-1].Content
	i1, i2, iu := strings.Index(tail, "s1"), strings.Index(tail, "s2"), strings.Index(tail, "u2")
	if i1 < 0 || i2 < 0 || iu < 0 || !(i1 < i2 && i2 < iu) {
		t.Fatalf("folded user turn lost order or content: %q", tail)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestOpenAIWire_LeadingSystemOnlyIsUnchanged(t *testing.T) {
	in := []Message{
		{Role: "system", Content: "a"},
		{Role: "system", Content: "b"},
		{Role: "user", Content: "u"},
		{Role: "assistant", Content: "x"},
	}
	got := wireMessages(t, in)
	if len(got) != len(in) {
		t.Fatalf("len = %d, want %d", len(got), len(in))
	}
	for i := range in {
		if got[i].Role != in[i].Role || got[i].Content != in[i].Content {
			t.Fatalf("message %d = %+v, want %+v", i, got[i], in[i])
		}
	}
}

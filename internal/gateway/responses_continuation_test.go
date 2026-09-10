package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// continuationPlanner is deliberately model-independent. It captures the messages at
// the real planner seam and returns a fixed completion for each HTTP turn.
type continuationPlanner struct {
	mu          sync.Mutex
	completions []*agent.Completion
	messages    [][]agent.Message
}

func (p *continuationPlanner) Complete(_ context.Context, messages []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.messages = append(p.messages, append([]agent.Message(nil), messages...))
	i := len(p.messages) - 1
	if i >= len(p.completions) {
		i = len(p.completions) - 1
	}
	return p.completions[i], nil
}

func (*continuationPlanner) Model() string { return "continuation-fixture" }

func (p *continuationPlanner) captured(turn int) []agent.Message {
	p.mu.Lock()
	defer p.mu.Unlock()
	return append([]agent.Message(nil), p.messages[turn]...)
}

func newContinuationServer(t *testing.T, planner *continuationPlanner, keys map[string]string) *Server {
	t.Helper()
	srv := newTestServer(t)
	srv.planner = planner
	srv.keyset = newKeyset(keys)
	return srv
}

func postAuthenticatedResponse(t *testing.T, base, key string, body map[string]any) (int, []byte) {
	t.Helper()
	raw, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	req, err := http.NewRequest(http.MethodPost, base+"/v1/responses", bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+key)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	bodyRaw, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	return resp.StatusCode, bodyRaw
}

func responseID(t *testing.T, raw []byte) string {
	t.Helper()
	var resp responsesResponse
	if err := json.Unmarshal(raw, &resp); err != nil {
		t.Fatalf("decode response: %v: %s", err, raw)
	}
	if resp.ID == "" {
		t.Fatalf("response ID is empty: %s", raw)
	}
	return resp.ID
}

func completedStreamResponseID(t *testing.T, raw []byte) string {
	t.Helper()
	for _, line := range strings.Split(string(raw), "\n") {
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		var event struct {
			Type     string            `json:"type"`
			Response responsesResponse `json:"response"`
		}
		if json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &event) == nil && event.Type == "response.completed" {
			return event.Response.ID
		}
	}
	t.Fatalf("stream has no response.completed event: %s", raw)
	return ""
}

func TestResponsesContinuationReconstructsToolHistory(t *testing.T) {
	const callID = "call_continue_1"
	planner := &continuationPlanner{completions: []*agent.Completion{
		{
			Message: agent.Message{Role: agent.RoleAssistant, ToolCalls: []agent.ToolCall{{
				ID: callID, Type: "function", Function: agent.Func{Name: "allow_lookup", Arguments: `{"q":"alpha"}`},
			}}},
			FinishReason: "tool_calls",
		},
		{Message: agent.Message{Role: agent.RoleAssistant, Content: "done"}, FinishReason: "stop"},
	}}
	srv := newContinuationServer(t, planner, map[string]string{"alice-key": "alice"})
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	code, firstRaw := postAuthenticatedResponse(t, ts.URL, "alice-key", map[string]any{
		"model": "fixture", "instructions": "old rules", "input": "look up alpha",
		"tools": []map[string]any{{"type": "function", "name": "allow_lookup"}},
	})
	if code != http.StatusOK {
		t.Fatalf("first response status = %d: %s", code, firstRaw)
	}
	id := responseID(t, firstRaw)

	code, secondRaw := postAuthenticatedResponse(t, ts.URL, "alice-key", map[string]any{
		"model": "fixture", "instructions": "old rules", "previous_response_id": id,
		"input": []map[string]any{{"type": "function_call_output", "call_id": callID, "output": `{"value":42}`}},
		"tools": []map[string]any{{"type": "function", "name": "allow_lookup"}},
	})
	if code != http.StatusOK {
		t.Fatalf("continuation status = %d: %s", code, secondRaw)
	}
	var second responsesResponse
	if err := json.Unmarshal(secondRaw, &second); err != nil {
		t.Fatalf("decode continuation response: %v: %s", err, secondRaw)
	}
	if second.Fak == nil || len(second.Fak.ResultAdmissions) != 1 {
		t.Fatalf("continuation omitted result admission: %+v", second.Fak)
	}
	adm := second.Fak.ResultAdmissions[0]
	if adm.ToolCallID != callID || adm.Tool != "allow_lookup" {
		t.Fatalf("result admission lost originating call identity: %+v", adm)
	}

	got := planner.captured(1)
	if len(got) != 4 {
		t.Fatalf("continuation messages = %#v, want system + user + assistant call + tool output", got)
	}
	if got[0].Role != agent.RoleSystem || got[0].Content != "old rules" ||
		got[1].Role != agent.RoleUser || got[1].Content != "look up alpha" {
		t.Fatalf("prior prompt was not reconstructed: %#v", got)
	}
	if got[2].Role != agent.RoleAssistant || len(got[2].ToolCalls) != 1 {
		t.Fatalf("assistant tool call missing from reconstructed history: %#v", got[2])
	}
	call := got[2].ToolCalls[0]
	if call.ID != callID || call.Function.Name != "allow_lookup" || call.Function.Arguments != `{"q":"alpha"}` {
		t.Fatalf("assistant tool call changed on round trip: %#v", call)
	}
	if got[3].Role != agent.RoleTool || got[3].ToolCallID != callID || got[3].Content != `{"value":42}` {
		t.Fatalf("tool output did not bind to prior call: %#v", got[3])
	}
}

func TestResponsesContinuationFailsClosedAndHonorsStoreFalse(t *testing.T) {
	planner := &continuationPlanner{completions: []*agent.Completion{{
		Message: agent.Message{Role: agent.RoleAssistant, Content: "ok"}, FinishReason: "stop",
	}}}
	srv := newContinuationServer(t, planner, map[string]string{"alice-key": "alice", "bob-key": "bob"})
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	code, raw := postAuthenticatedResponse(t, ts.URL, "alice-key", map[string]any{
		"model": "fixture", "input": "private turn",
	})
	if code != http.StatusOK {
		t.Fatalf("stored response status = %d: %s", code, raw)
	}
	storedID := responseID(t, raw)

	for _, tc := range []struct {
		name string
		key  string
		id   string
	}{
		{name: "unknown", key: "alice-key", id: "resp_fak_unknown"},
		{name: "cross principal", key: "bob-key", id: storedID},
	} {
		t.Run(tc.name, func(t *testing.T) {
			before := len(planner.messages)
			gotCode, gotRaw := postAuthenticatedResponse(t, ts.URL, tc.key, map[string]any{
				"model": "fixture", "previous_response_id": tc.id, "input": "continue",
			})
			if gotCode != http.StatusBadRequest {
				t.Fatalf("status = %d, want 400: %s", gotCode, gotRaw)
			}
			if len(planner.messages) != before {
				t.Fatalf("planner was called for inaccessible previous_response_id")
			}
		})
	}

	code, raw = postAuthenticatedResponse(t, ts.URL, "alice-key", map[string]any{
		"model": "fixture", "input": "ephemeral turn", "store": false,
	})
	if code != http.StatusOK {
		t.Fatalf("store:false response status = %d: %s", code, raw)
	}
	ephemeralID := responseID(t, raw)
	before := len(planner.messages)
	code, raw = postAuthenticatedResponse(t, ts.URL, "alice-key", map[string]any{
		"model": "fixture", "previous_response_id": ephemeralID, "input": "continue",
	})
	if code != http.StatusBadRequest {
		t.Fatalf("store:false continuation status = %d, want 400: %s", code, raw)
	}
	if len(planner.messages) != before {
		t.Fatal("planner was called for a store:false response")
	}
}

func TestResponsesContinuationCurrentInstructionsReplacePrior(t *testing.T) {
	planner := &continuationPlanner{completions: []*agent.Completion{
		{Message: agent.Message{Role: agent.RoleAssistant, Content: "first"}, FinishReason: "stop"},
		{Message: agent.Message{Role: agent.RoleAssistant, Content: "second"}, FinishReason: "stop"},
	}}
	srv := newContinuationServer(t, planner, map[string]string{"alice-key": "alice"})
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	code, raw := postAuthenticatedResponse(t, ts.URL, "alice-key", map[string]any{
		"model": "fixture", "instructions": "old rules", "input": "first",
	})
	if code != http.StatusOK {
		t.Fatalf("first status = %d: %s", code, raw)
	}
	code, raw = postAuthenticatedResponse(t, ts.URL, "alice-key", map[string]any{
		"model": "fixture", "instructions": "new rules", "previous_response_id": responseID(t, raw), "input": "second",
	})
	if code != http.StatusOK {
		t.Fatalf("continuation status = %d: %s", code, raw)
	}
	got := planner.captured(1)
	var systems []string
	for _, m := range got {
		if m.Role == agent.RoleSystem {
			systems = append(systems, m.Content)
		}
	}
	if len(systems) != 1 || systems[0] != "new rules" {
		t.Fatalf("system instructions = %q, want only current instructions", systems)
	}
	code, raw = postAuthenticatedResponse(t, ts.URL, "alice-key", map[string]any{
		"model": "fixture", "instructions": "instructions-only remains valid",
	})
	if code != http.StatusOK {
		t.Fatalf("instructions-only request status = %d, want 200: %s", code, raw)
	}
}

func TestResponsesContinuationStreamingCompletionIsImmediatelyAddressable(t *testing.T) {
	planner := &continuationPlanner{completions: []*agent.Completion{
		{Message: agent.Message{Role: agent.RoleAssistant, Content: "streamed"}, FinishReason: "stop"},
		{Message: agent.Message{Role: agent.RoleAssistant, Content: "continued"}, FinishReason: "stop"},
	}}
	srv := newContinuationServer(t, planner, map[string]string{"alice-key": "alice"})
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	code, raw := postAuthenticatedResponse(t, ts.URL, "alice-key", map[string]any{
		"model": "fixture", "input": "stream this", "stream": true,
	})
	if code != http.StatusOK {
		t.Fatalf("stream status = %d: %s", code, raw)
	}
	id := completedStreamResponseID(t, raw)
	code, raw = postAuthenticatedResponse(t, ts.URL, "alice-key", map[string]any{
		"model": "fixture", "previous_response_id": id, "input": "continue now",
	})
	if code != http.StatusOK {
		t.Fatalf("immediate continuation after response.completed = %d: %s", code, raw)
	}
}

func TestResponsesContinuationPersistsAdmittedResult(t *testing.T) {
	const callID = "call_shape_1"
	const secret = "consumer-secret"
	planner := &continuationPlanner{completions: []*agent.Completion{
		{Message: agent.Message{Role: agent.RoleAssistant, ToolCalls: []agent.ToolCall{{
			ID: callID, Type: "function", Function: agent.Func{Name: "allow_lookup", Arguments: `{}`},
		}}}, FinishReason: "tool_calls"},
		{Message: agent.Message{Role: agent.RoleAssistant, Content: "observed"}, FinishReason: "stop"},
		{Message: agent.Message{Role: agent.RoleAssistant, Content: "continued"}, FinishReason: "stop"},
	}}
	srv := newContinuationServer(t, planner, map[string]string{"alice-key": "alice"})
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	tool := map[string]any{
		"type": "function", "name": "allow_lookup",
		"parameters": map[string]any{
			"type": "object",
			"x-fak-result-contract": map[string]any{
				"schema":  "fak-result-contract/1",
				"outputs": []map[string]any{{"name": "count", "type": "integer"}, {"name": "summary", "type": "string"}},
			},
		},
	}

	code, raw := postAuthenticatedResponse(t, ts.URL, "alice-key", map[string]any{
		"model": "fixture", "input": "lookup", "tools": []map[string]any{tool},
	})
	if code != http.StatusOK {
		t.Fatalf("first status = %d: %s", code, raw)
	}
	firstID := responseID(t, raw)
	code, raw = postAuthenticatedResponse(t, ts.URL, "alice-key", map[string]any{
		"model": "fixture", "previous_response_id": firstID, "tools": []map[string]any{tool},
		"input": []map[string]any{{"type": "function_call_output", "call_id": callID, "output": `{"count":2,"summary":"ok","debug":"` + secret + `"}`}},
	})
	if code != http.StatusOK {
		t.Fatalf("result turn status = %d: %s", code, raw)
	}
	var second responsesResponse
	if err := json.Unmarshal(raw, &second); err != nil {
		t.Fatalf("decode result turn: %v: %s", err, raw)
	}
	if second.Fak == nil || len(second.Fak.ResultAdmissions) != 1 || second.Fak.ResultAdmissions[0].ToolCallID != callID || second.Fak.ResultAdmissions[0].Tool != "allow_lookup" || second.Fak.ResultAdmissions[0].Verdict.Kind != "QUARANTINE" {
		t.Fatalf("result was not admitted against its originating call: %+v", second.Fak)
	}

	code, raw = postAuthenticatedResponse(t, ts.URL, "alice-key", map[string]any{
		"model": "fixture", "previous_response_id": second.ID, "tools": []map[string]any{tool}, "input": "third turn",
	})
	if code != http.StatusOK {
		t.Fatalf("third turn status = %d: %s", code, raw)
	}
	third := planner.captured(2)
	for _, m := range third {
		if m.Role != agent.RoleTool {
			continue
		}
		if strings.Contains(m.Content, secret) || !strings.Contains(m.Content, `"_quarantined":true`) {
			t.Fatalf("third turn recovered raw pre-admission result: %q", m.Content)
		}
		return
	}
	t.Fatalf("third turn omitted historical admitted result: %#v", third)
}

func TestResponsesContinuationStoreEvictsOldestWithinBounds(t *testing.T) {
	store := newResponsesContinuationStore(responsesContinuationStoreConfig{MaxEntries: 2, MaxBytes: 1 << 20})
	now := time.Now()
	message := func(content string) []agent.Message {
		return []agent.Message{{Role: agent.RoleUser, Content: content}}
	}
	if !store.Put("one", "alice", message("one"), now) ||
		!store.Put("two", "alice", message("two"), now.Add(time.Millisecond)) ||
		!store.Put("three", "alice", message("three"), now.Add(2*time.Millisecond)) {
		t.Fatal("store rejected bounded fixtures")
	}
	if got := store.Len(); got != 2 {
		t.Fatalf("store len = %d, want 2", got)
	}
	if _, ok := store.Get("one", "alice", now.Add(3*time.Millisecond)); ok {
		t.Fatal("oldest response survived capacity eviction")
	}
	if got, ok := store.Get("three", "alice", now.Add(3*time.Millisecond)); !ok || len(got) != 1 || got[0].Content != "three" {
		t.Fatalf("newest response missing after eviction: %#v, %v", got, ok)
	}

	original := []agent.Message{{
		Role: agent.RoleTool, Content: "  byte-exact tool output  ", ToolCallID: "call_clone",
		Witness: "sha256:new", RefutesWitness: "sha256:old",
		ToolCalls: []agent.ToolCall{{ID: "nested", Function: agent.Func{Name: "lookup", Arguments: `{"x":1}`}}},
	}}
	if !store.Put("clone", "alice", original, now.Add(4*time.Millisecond)) {
		t.Fatal("store rejected clone fixture")
	}
	original[0].Content = "mutated source"
	original[0].ToolCalls[0].Function.Arguments = "mutated source"
	got, ok := store.Get("clone", "alice", now.Add(5*time.Millisecond))
	if !ok || len(got) != 1 {
		t.Fatalf("clone fixture missing: %#v, %v", got, ok)
	}
	if got[0].Content != "  byte-exact tool output  " || got[0].Witness != "sha256:new" || got[0].RefutesWitness != "sha256:old" {
		t.Fatalf("clone lost exact tool metadata: %#v", got[0])
	}
	got[0].ToolCalls[0].Function.Arguments = "mutated read"
	again, ok := store.Get("clone", "alice", now.Add(6*time.Millisecond))
	if !ok || again[0].ToolCalls[0].Function.Arguments != `{"x":1}` {
		t.Fatalf("Get exposed nested store memory: %#v, %v", again, ok)
	}

	ttlStore := newResponsesContinuationStore(responsesContinuationStoreConfig{MaxEntries: 4, MaxBytes: 1 << 20, TTL: time.Minute})
	if !ttlStore.Put("fresh-first", "alice", message("fresh"), now) ||
		!ttlStore.Put("stale-second", "alice", message("stale"), now.Add(-2*time.Minute)) {
		t.Fatal("store rejected out-of-order TTL fixtures")
	}
	if _, ok := ttlStore.Get("stale-second", "alice", now); ok {
		t.Fatal("expired response survived because a fresh entry preceded it in insertion order")
	}
}

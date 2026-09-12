package gateway

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/engine"
)

func TestOpenCodeReadProposalPreservesClientArguments(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("mock", engine.MockEngine)
	monitor := adjudicator.New(adjudicator.DevAgentPolicy())
	abi.RegisterAdjudicator(100, monitor)
	srv, err := New(Config{EngineID: "mock", Model: "fixture"})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	calls := []agent.ToolCall{{ID: "read-1", Type: "function", ThoughtSignature: "opaque-signed-context", Function: agent.Func{Name: "read", Arguments: `{"filePath":"witness.txt","offset":1,"limit":4}`}}}
	kept, adjs, dropped := srv.adjudicateProposed(context.Background(), calls, "opencode-read")
	if dropped != 0 || len(kept) != 1 || !adjs[0].Admitted {
		t.Fatalf("proposal lost: kept=%v adjs=%v dropped=%d", kept, adjs, dropped)
	}
	if kept[0].ThoughtSignature != "opaque-signed-context" {
		t.Fatal("adjudication lost native provider signature")
	}
	var got map[string]any
	if err := json.Unmarshal([]byte(kept[0].Function.Arguments), &got); err != nil {
		t.Fatal(err)
	}
	if kept[0].Function.Name != "read" || got["filePath"] != "witness.txt" || got["offset"] != float64(1) || got["limit"] != float64(4) || got["file_path"] != nil {
		t.Fatalf("client schema changed: %v", kept)
	}
	// A policy denial remains a denial; restoring client spelling grants nothing.
	monitor.SetPolicy(adjudicator.Policy{Deny: map[string]abi.ReasonCode{"read": abi.ReasonPolicyBlock}})
	kept, _, dropped = srv.adjudicateProposed(context.Background(), calls, "opencode-read-deny")
	if dropped != 1 || len(kept) != 0 {
		t.Fatalf("denied read survived: %v", kept)
	}
}

func TestOpenCodeReadStreamUsesAdvertisedClientPathDialect(t *testing.T) {
	responseArgs := `{"file_path":"witness.txt","offset":1,"limit":4}`
	advertisedKey := "filePath"
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request ChatRequest
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Errorf("decode upstream request: %v", err)
		}
		if len(request.Tools) != 1 || request.Tools[0].Function.Name != "read" ||
			!strings.Contains(string(request.Tools[0].Function.Parameters), `"`+advertisedKey+`"`) {
			t.Errorf("upstream read schema changed: %+v", request.Tools)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, strings.Join([]string{
			`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"read-1","type":"function","function":{"name":"read","arguments":` + strconv.Quote(responseArgs) + `}}]}}]}`,
			`data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}`,
			"data: [DONE]", "",
		}, "\n\n"))
	}))
	defer upstream.Close()

	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	monitor := adjudicator.New(adjudicator.DevAgentPolicy())
	abi.RegisterAdjudicator(100, monitor)
	srv, err := New(Config{EngineID: "test", Model: "x:model", BaseURL: upstream.URL, Provider: "openai-compatible"})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	gateway := httptest.NewServer(srv.Handler())
	defer gateway.Close()

	request := openCodeReadStreamRequest("filePath")
	code, raw := postStream(t, gateway.URL, request)
	if code != http.StatusOK {
		t.Fatalf("status=%d body=%s", code, raw)
	}
	chunks, done := parseSSEChunks(t, raw)
	if !done {
		t.Fatalf("missing DONE: %s", raw)
	}
	var calls []ChatDeltaToolCall
	for _, chunk := range chunks {
		calls = append(calls, chunk.Choices[0].Delta.ToolCalls...)
	}
	if len(calls) != 1 || calls[0].Function.Name != "read" {
		t.Fatalf("calls=%+v", calls)
	}
	var got map[string]any
	if err := json.Unmarshal([]byte(calls[0].Function.Arguments), &got); err != nil {
		t.Fatal(err)
	}
	if got["filePath"] != "witness.txt" || got["offset"] != float64(1) || got["limit"] != float64(4) {
		t.Fatalf("client read arguments=%v", got)
	}
	if _, exists := got["file_path"]; exists {
		t.Fatalf("upstream spelling leaked into client call: %v", got)
	}

	advertisedKey = "file_path"
	responseArgs = `{"file_path":"snake-client.txt","offset":3,"limit":7}`
	code, raw = postStream(t, gateway.URL, openCodeReadStreamRequest("file_path"))
	if code != http.StatusOK {
		t.Fatalf("snake-schema status=%d body=%s", code, raw)
	}
	chunks, _ = parseSSEChunks(t, raw)
	calls = nil
	for _, chunk := range chunks {
		calls = append(calls, chunk.Choices[0].Delta.ToolCalls...)
	}
	got = nil
	if len(calls) != 1 || calls[0].Function.Name != "read" || json.Unmarshal([]byte(calls[0].Function.Arguments), &got) != nil {
		t.Fatalf("snake-schema calls=%+v", calls)
	}
	if got["file_path"] != "snake-client.txt" || got["filePath"] != nil || got["offset"] != float64(3) || got["limit"] != float64(7) {
		t.Fatalf("snake-schema client arguments=%v", got)
	}

	// Differing aliases are ambiguous. The gateway must deny the call instead
	// of choosing or silently discarding either path.
	advertisedKey = "filePath"
	responseArgs = `{"file_path":"canonical.txt","filePath":"conflict.txt","offset":2}`
	code, raw = postStream(t, gateway.URL, request)
	if code != http.StatusOK {
		t.Fatalf("conflict status=%d body=%s", code, raw)
	}
	chunks, _ = parseSSEChunks(t, raw)
	calls = nil
	for _, chunk := range chunks {
		calls = append(calls, chunk.Choices[0].Delta.ToolCalls...)
	}
	if len(calls) != 0 || strings.Contains(string(raw), "canonical.txt") || strings.Contains(string(raw), "conflict.txt") {
		t.Fatalf("conflicting aliases survived instead of being denied: calls=%+v raw=%s", calls, raw)
	}

	monitor.SetPolicy(adjudicator.Policy{Deny: map[string]abi.ReasonCode{"read": abi.ReasonPolicyBlock}})
	responseArgs = `{"file_path":"denied-secret.txt","offset":2}`
	code, raw = postStream(t, gateway.URL, request)
	if code != http.StatusOK {
		t.Fatalf("deny status=%d body=%s", code, raw)
	}
	if strings.Contains(string(raw), "denied-secret.txt") {
		t.Fatalf("denied read arguments leaked: %s", raw)
	}
	chunks, _ = parseSSEChunks(t, raw)
	for _, chunk := range chunks {
		if len(chunk.Choices[0].Delta.ToolCalls) != 0 {
			t.Fatalf("denied read survived: %s", raw)
		}
	}
}

func openCodeReadStreamRequest(pathKey string) map[string]any {
	return map[string]any{
		"model": "x:model", "stream": true,
		"messages": []map[string]string{{"role": "user", "content": "read it"}},
		"tools": []map[string]any{{"type": "function", "function": map[string]any{
			"name": "read", "parameters": map[string]any{
				"type": "object", "properties": map[string]any{
					pathKey:  map[string]any{"type": "string"},
					"offset": map[string]any{"type": "number"},
					"limit":  map[string]any{"type": "number"},
				}, "required": []string{pathKey},
			},
		}}},
	}
}

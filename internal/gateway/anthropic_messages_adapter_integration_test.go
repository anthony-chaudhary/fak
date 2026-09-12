package gateway

import (
	"bufio"
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func TestAnthropicMessagesAdapterLivePlannerToolLoop(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})

	contentSent := make(chan struct{})
	releaseTools := make(chan struct{})
	var calls atomic.Int32
	var bodiesMu sync.Mutex
	var upstreamBodies [][]byte
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		bodiesMu.Lock()
		upstreamBodies = append(upstreamBodies, append([]byte(nil), raw...))
		bodiesMu.Unlock()

		if calls.Add(1) == 2 {
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"model":"served-openai","choices":[{"message":{"role":"assistant","content":"The weather is 68 degrees."},"finish_reason":"stop"}],"usage":{"prompt_tokens":12,"completion_tokens":6,"total_tokens":18}}`)
			return
		}

		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher := w.(http.Flusher)
		_, _ = io.WriteString(w,
			`data: {"model":"served-openai","choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}`+"\n\n"+
				`data: {"choices":[{"delta":{"content":"checking"}}]}`+"\n\n")
		flusher.Flush()
		close(contentSent)
		select {
		case <-releaseTools:
		case <-r.Context().Done():
		case <-time.After(3 * time.Second):
			return
		}
		_, _ = io.WriteString(w,
			`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"a1","type":"function","function":{"name":"allow_a","arguments":"{\"city\":\"SF\"}"}}]}}]}`+"\n\n"+
				`data: {"choices":[{"delta":{"tool_calls":[{"index":1,"id":"d1","type":"function","function":{"name":"deny_b","arguments":"{\"secret\":\"nope\"}"}}]}}]}`+"\n\n"+
				`data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":8,"completion_tokens":5,"total_tokens":13}}`+"\n\n"+
				"data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer upstream.Close()

	srv, err := New(Config{EngineID: "test", Model: "x:model", BaseURL: upstream.URL + "/compat", Provider: "openai-compatible", VDSO: true})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	ts := httptest.NewServer(NewAnthropicMessagesAdapter(srv).Handler())
	defer ts.Close()

	first := []byte(`{"model":"claude-client","max_tokens":256,"stream":true,"tools":[{"name":"allow_a","input_schema":{"type":"object"}},{"name":"deny_b","input_schema":{"type":"object"}}],"messages":[{"role":"user","content":"call tools"}]}`)
	resp, err := http.Post(ts.URL+"/v1/messages", "application/json", bytes.NewReader(first))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(resp.Body)
		t.Fatalf("first turn status = %d: %s", resp.StatusCode, raw)
	}

	lines := make(chan string, 128)
	go func() {
		scanner := bufio.NewScanner(resp.Body)
		for scanner.Scan() {
			lines <- scanner.Text()
		}
		close(lines)
	}()
	select {
	case <-contentSent:
	case <-time.After(2 * time.Second):
		t.Fatal("upstream did not emit its text delta")
	}
	var all []string
	for {
		select {
		case line, ok := <-lines:
			if !ok {
				t.Fatalf("stream ended before downstream text progress: %q", strings.Join(all, "\n"))
			}
			all = append(all, line)
			if strings.Contains(line, `"text_delta"`) && strings.Contains(line, "checking") {
				close(releaseTools)
				goto released
			}
		case <-time.After(2 * time.Second):
			t.Fatalf("downstream did not receive text before tool completion: %q", strings.Join(all, "\n"))
		}
	}

released:
	for line := range lines {
		all = append(all, line)
	}
	streamBody := strings.Join(all, "\n")
	for _, want := range []string{`event: message_start`, `"name":"allow_a"`, `"stop_reason":"tool_use"`, `event: message_stop`} {
		if !strings.Contains(streamBody, want) {
			t.Errorf("stream missing %q:\n%s", want, streamBody)
		}
	}
	for _, leak := range []string{"<tool_call>", `"name":"deny_b"`, `"secret"`, "nope"} {
		if strings.Contains(streamBody, leak) {
			t.Errorf("stream leaked %q:\n%s", leak, streamBody)
		}
	}

	toolID := ""
	for _, event := range all {
		if !strings.HasPrefix(event, "data: ") || !strings.Contains(event, `"tool_use"`) {
			continue
		}
		var payload struct {
			ContentBlock struct {
				ID string `json:"id"`
			} `json:"content_block"`
		}
		_ = json.Unmarshal([]byte(strings.TrimPrefix(event, "data: ")), &payload)
		if payload.ContentBlock.ID != "" {
			toolID = payload.ContentBlock.ID
			break
		}
	}
	if toolID == "" {
		t.Fatalf("allowed tool_use id missing:\n%s", streamBody)
	}

	second := map[string]any{
		"model":      "claude-client",
		"max_tokens": 128,
		"messages": []map[string]any{
			{"role": "user", "content": "call tools"},
			{
				"role": "assistant",
				"content": []map[string]any{
					{"type": "tool_use", "id": toolID, "name": "allow_a", "input": map[string]any{"city": "SF"}},
				},
			},
			{
				"role": "user",
				"content": []map[string]any{
					{"type": "tool_result", "tool_use_id": toolID, "content": "68"},
				},
			},
		},
	}
	rawSecond, _ := json.Marshal(second)
	resp2, err := http.Post(ts.URL+"/v1/messages", "application/json", bytes.NewReader(rawSecond))
	if err != nil {
		t.Fatal(err)
	}
	defer resp2.Body.Close()
	var got anthropicMessageResponse
	if err := json.NewDecoder(resp2.Body).Decode(&got); err != nil {
		t.Fatalf("decode second turn: %v", err)
	}
	if resp2.StatusCode != http.StatusOK || got.StopReason != "end_turn" || len(got.Content) == 0 || got.Content[0].Text != "The weather is 68 degrees." {
		t.Fatalf("second turn response = status %d, %+v", resp2.StatusCode, got)
	}
	bodiesMu.Lock()
	bodies := append([][]byte(nil), upstreamBodies...)
	bodiesMu.Unlock()
	if len(bodies) != 2 || !bytes.Contains(bodies[1], []byte(`"role":"tool"`)) || !bytes.Contains(bodies[1], []byte("68")) {
		t.Fatalf("tool_result was not preserved in second upstream turn: %s", bodies)
	}
}

func TestAnthropicMessagesAdapterPublicHandlersAuthenticate(t *testing.T) {
	srv := newTestServer(t)
	srv.requireKey = "adapter-secret"
	adapter := NewAnthropicMessagesAdapter(srv)
	registered := http.NewServeMux()
	srv.RegisterAnthropicMessagesRoutes(registered)

	handlers := []struct {
		name string
		h    http.Handler
	}{
		{"adapter_handler", adapter.Handler()},
		{"server_handler", srv.AnthropicMessagesHandler()},
		{"registered_routes", registered},
		{"direct_handler", http.HandlerFunc(srv.HandleAnthropicCountTokens)},
	}
	body := []byte(`{"model":"qwen","messages":[{"role":"user","content":"count these tokens"}]}`)
	for _, tc := range handlers {
		t.Run(tc.name, func(t *testing.T) {
			unauthorized := httptest.NewRequest(http.MethodPost, "/v1/messages/count_tokens", bytes.NewReader(body))
			unauthorized.Header.Set("Content-Type", "application/json")
			unauthorizedResult := httptest.NewRecorder()
			tc.h.ServeHTTP(unauthorizedResult, unauthorized)
			if unauthorizedResult.Code != http.StatusUnauthorized {
				t.Fatalf("unauthorized status = %d, want 401", unauthorizedResult.Code)
			}

			authorized := httptest.NewRequest(http.MethodPost, "/v1/messages/count_tokens", bytes.NewReader(body))
			authorized.Header.Set("Content-Type", "application/json")
			authorized.Header.Set("x-api-key", "adapter-secret")
			authorizedResult := httptest.NewRecorder()
			tc.h.ServeHTTP(authorizedResult, authorized)
			if authorizedResult.Code != http.StatusOK {
				t.Fatalf("authorized status = %d, want 200: %s", authorizedResult.Code, authorizedResult.Body.String())
			}
			var count map[string]int
			if err := json.Unmarshal(authorizedResult.Body.Bytes(), &count); err != nil || count["input_tokens"] <= 0 {
				t.Fatalf("count response = %q, err=%v", authorizedResult.Body.String(), err)
			}
		})
	}
}

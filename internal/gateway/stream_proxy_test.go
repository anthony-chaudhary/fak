package gateway

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// TestChatProxyStreamsUpstreamContentLive proves the true-streaming fast path: a
// no-tools stream=true request reaches an SSE-capable upstream, the gateway asks the
// upstream to stream, and it relays each content fragment to the client as an OpenAI
// chunk — so the client's first byte tracks the model, not the whole turn.
func TestChatProxyStreamsUpstreamContentLive(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})

	var upstreamStream bool
	var upstreamHits int
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamHits++
		var req struct {
			Stream bool `json:"stream"`
		}
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &req)
		upstreamStream = req.Stream
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, "data: {\"model\":\"served-x\",\"choices\":[{\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n"+
			"data: {\"choices\":[{\"delta\":{\"content\":\"check\"},\"finish_reason\":null}]}\n\n"+
			"data: {\"choices\":[{\"delta\":{\"content\":\"ing\"},\"finish_reason\":null}]}\n\n"+
			"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":2,\"total_tokens\":6}}\n\n"+
			"data: [DONE]\n\n")
	}))
	defer upstream.Close()

	srv, err := New(Config{EngineID: "test", Model: "x:model", BaseURL: upstream.URL + "/compat", Provider: "openai-compatible"})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	reqBody, _ := json.Marshal(map[string]any{
		"model":    "x:model",
		"messages": []map[string]string{{"role": "user", "content": "are you there"}},
		"stream":   true,
	})
	httpResp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", bytes.NewReader(reqBody))
	if err != nil {
		t.Fatal(err)
	}
	defer httpResp.Body.Close()
	respRaw, _ := io.ReadAll(httpResp.Body)
	if httpResp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200: %s", httpResp.StatusCode, respRaw)
	}
	if !upstreamStream {
		t.Fatalf("upstream was not asked to stream for a no-tools stream request")
	}
	if upstreamHits != 1 {
		t.Fatalf("upstream hits = %d, want 1", upstreamHits)
	}
	if ct := httpResp.Header.Get("Content-Type"); !strings.Contains(ct, "text/event-stream") {
		t.Fatalf("content-type = %q, want text/event-stream", ct)
	}

	chunks, sawDone := parseSSEChunks(t, respRaw)
	if !sawDone {
		t.Fatalf("stream missing [DONE]: %s", respRaw)
	}
	if len(chunks) < 3 {
		t.Fatalf("chunks = %d, want >=3 (role + content + terminal): %s", len(chunks), respRaw)
	}
	if role := chunks[0].Choices[0].Delta.Role; role != "assistant" {
		t.Fatalf("first chunk role = %q, want assistant", role)
	}
	var content strings.Builder
	var finish string
	var usage bool
	for _, c := range chunks {
		content.WriteString(c.Choices[0].Delta.Content)
		if c.Choices[0].FinishReason != nil {
			finish = *c.Choices[0].FinishReason
		}
		if c.Usage != nil && c.Usage.PromptTokens == 4 {
			usage = true
		}
		if len(c.Choices[0].Delta.ToolCalls) != 0 {
			t.Fatalf("unexpected tool call in a no-tools stream: %+v", c.Choices[0].Delta.ToolCalls)
		}
	}
	if got := content.String(); got != "checking" {
		t.Fatalf("reassembled streamed content = %q, want checking", got)
	}
	if finish != "stop" {
		t.Fatalf("finish = %q, want stop", finish)
	}
	if !usage {
		t.Fatalf("terminal chunk missing upstream usage (prompt_tokens 4): %s", respRaw)
	}
}

// TestChatProxyStreamFallsBackWhenPlannerCannotStream proves the gate's false branch:
// a planner that cannot stream (the offline mock) still serves a stream=true request
// via the buffered+synthesized path — nothing was written before the fall-through.
func TestChatProxyStreamFallsBackWhenPlannerCannotStream(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})

	srv, err := New(Config{EngineID: "test", Model: "mock:model"}) // no BaseURL => MockPlanner
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	reqBody, _ := json.Marshal(map[string]any{
		"model":    "mock:model",
		"messages": []map[string]string{{"role": "user", "content": "hello"}},
		"stream":   true,
	})
	httpResp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", bytes.NewReader(reqBody))
	if err != nil {
		t.Fatal(err)
	}
	defer httpResp.Body.Close()
	respRaw, _ := io.ReadAll(httpResp.Body)
	if httpResp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200: %s", httpResp.StatusCode, respRaw)
	}
	if ct := httpResp.Header.Get("Content-Type"); !strings.Contains(ct, "text/event-stream") {
		t.Fatalf("content-type = %q, want text/event-stream", ct)
	}
	if _, sawDone := parseSSEChunks(t, respRaw); !sawDone {
		t.Fatalf("buffered fallback stream missing [DONE]: %s", respRaw)
	}
}

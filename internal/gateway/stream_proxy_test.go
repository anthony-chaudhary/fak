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

func TestChatProxyStreamUsesReplicaRouter(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})

	mkUpstream := func(name string, hits *int) *httptest.Server {
		return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			(*hits)++
			var req struct {
				Stream bool `json:"stream"`
			}
			raw, _ := io.ReadAll(r.Body)
			_ = json.Unmarshal(raw, &req)
			if !req.Stream {
				t.Errorf("%s was not asked to stream: %s", name, raw)
			}
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			_, _ = io.WriteString(w,
				`data: {"model":"`+name+`","choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}`+"\n\n"+
					`data: {"choices":[{"delta":{"content":"`+name+`"},"finish_reason":null}]}`+"\n\n"+
					`data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`+"\n\n"+
					"data: [DONE]\n\n")
		}))
	}
	var aHits, bHits int
	a := mkUpstream("stream-a", &aHits)
	defer a.Close()
	b := mkUpstream("stream-b", &bHits)
	defer b.Close()

	srv, err := New(Config{
		EngineID:        "test",
		Model:           "stream-fleet",
		BaseURL:         a.URL + "/compat",
		ReplicaBaseURLs: []string{b.URL + "/compat"},
		Provider:        "openai-compatible",
		VDSO:            true,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	var got []string
	for i := 0; i < 2; i++ {
		reqBody, _ := json.Marshal(map[string]any{
			"model":    "stream-fleet",
			"messages": []map[string]string{{"role": "user", "content": "hello"}},
			"stream":   true,
		})
		httpResp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", bytes.NewReader(reqBody))
		if err != nil {
			t.Fatal(err)
		}
		respRaw, _ := io.ReadAll(httpResp.Body)
		_ = httpResp.Body.Close()
		if httpResp.StatusCode != http.StatusOK {
			t.Fatalf("request %d status = %d, want 200: %s", i, httpResp.StatusCode, respRaw)
		}
		chunks, sawDone := parseSSEChunks(t, respRaw)
		if !sawDone {
			t.Fatalf("request %d stream missing [DONE]: %s", i, respRaw)
		}
		var content strings.Builder
		for _, c := range chunks {
			content.WriteString(c.Choices[0].Delta.Content)
		}
		got = append(got, content.String())
	}
	if want := []string{"stream-a", "stream-b"}; strings.Join(got, ",") != strings.Join(want, ",") {
		t.Fatalf("stream replica sequence = %v, want %v", got, want)
	}
	if aHits != 1 || bHits != 1 {
		t.Fatalf("stream upstream hits = a:%d b:%d, want one each", aHits, bHits)
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

// TestStreamProxyMidStreamSteeringInjection verifies registering an active stream and
// injecting mid-stream steering frames via StreamRegistry (#11513).
func TestStreamProxyMidStreamSteeringInjection(t *testing.T) {
	registry := newStreamRegistry()

	ch := make(chan SteeringFrame, 10)
	w := httptest.NewRecorder()

	reg := StreamRegistration{
		StreamID:  "stream-test-42",
		SessionID: "sess-test-99",
		Writer:    w,
		Channel:   ch,
	}

	as, unreg := registry.Register(reg)
	if as == nil {
		t.Fatal("expected non-nil ActiveStream")
	}
	defer unreg()

	// 1. Inject by StreamID
	frame := SteeringFrame{
		StreamID:  "stream-test-42",
		Directive: "redirect",
		Text:      "Focus on auth module only",
	}

	count, err := registry.Inject(frame)
	if err != nil {
		t.Fatalf("Inject by StreamID failed: %v", err)
	}
	if count != 1 {
		t.Fatalf("injected count = %d, want 1", count)
	}

	// Verify received on channel
	select {
	case received := <-ch:
		if received.Directive != "redirect" || received.Text != "Focus on auth module only" {
			t.Errorf("unexpected frame on channel: %+v", received)
		}
		if received.Op != "steer" {
			t.Errorf("op = %q, want 'steer'", received.Op)
		}
	default:
		t.Fatal("no frame received on channel")
	}

	// Verify SSE written to writer
	body := w.Body.String()
	if !strings.Contains(body, "event: steering\n") {
		t.Errorf("expected SSE event: steering in writer, got: %q", body)
	}
	if !strings.Contains(body, "Focus on auth module only") {
		t.Errorf("expected steering text in writer body, got: %q", body)
	}

	// 2. Inject by SessionID
	frameSess := SteeringFrame{
		SessionID: "sess-test-99",
		Directive: "note",
		Text:      "Reminder: tests must pass",
	}
	countSess, err := registry.Inject(frameSess)
	if err != nil {
		t.Fatalf("Inject by SessionID failed: %v", err)
	}
	if countSess != 1 {
		t.Fatalf("injected count = %d, want 1", countSess)
	}

	select {
	case received := <-ch:
		if received.Directive != "note" || received.Text != "Reminder: tests must pass" {
			t.Errorf("unexpected frame on channel: %+v", received)
		}
	default:
		t.Fatal("no frame received on channel for session inject")
	}

	// 3. Inject to unknown target returns error
	frameUnknown := SteeringFrame{
		StreamID: "nonexistent-stream",
	}
	_, err = registry.Inject(frameUnknown)
	if err == nil {
		t.Fatal("expected error injecting to nonexistent stream")
	}

	// 4. Teardown and verify stream unregisters cleanly
	unreg()
	_, err = registry.Inject(frame)
	if err == nil {
		t.Fatal("expected error injecting to unregistered stream")
	}
}

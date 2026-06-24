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

// sseUpstream builds an httptest server that answers /compat/chat/completions with the
// given event-stream body, after asserting the gateway asked it to stream. It records
// the hit count so a test can prove a single upstream generation.
func sseUpstream(t *testing.T, hits *int, sse string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		*hits++
		var req struct {
			Stream *bool `json:"stream"`
		}
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &req)
		if req.Stream == nil || !*req.Stream {
			t.Errorf("gateway must ask upstream to stream: %s", raw)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, sse)
	}))
}

func liveStreamServer(t *testing.T, upstreamURL string) *httptest.Server {
	t.Helper()
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})
	srv, err := New(Config{EngineID: "test", Model: "x:model", BaseURL: upstreamURL + "/compat", Provider: "openai-compatible"})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	return httptest.NewServer(srv.Handler())
}

func postStream(t *testing.T, url string, body map[string]any) (int, []byte) {
	t.Helper()
	raw, _ := json.Marshal(body)
	resp, err := http.Post(url+"/v1/chat/completions", "application/json", bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	out, _ := io.ReadAll(resp.Body)
	return resp.StatusCode, out
}

// TestToolBearingStreamHoldsNativeCallsAndStreamsContentLive proves the rung: a
// tool-bearing stream=true request now takes the LIVE path. Content streams as the
// model emits it, the native delta.tool_calls are HELD and adjudicated (denied dropped,
// allow kept), and only the survivor reaches a dedicated post-content tool-call delta.
func TestToolBearingStreamHoldsNativeCallsAndStreamsContentLive(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"model":"served-x","choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}`,
		`data: {"choices":[{"delta":{"content":"Let me "}}]}`,
		`data: {"choices":[{"delta":{"content":"check that."}}]}`,
		`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"allow_x","arguments":""}}]}}]}`,
		`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"a\":1}"}}]}}]}`,
		`data: {"choices":[{"delta":{"tool_calls":[{"index":1,"id":"c2","type":"function","function":{"name":"deny_y","arguments":"{\"k\":\"v\"}"}}]}}]}`,
		`data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":9,"completion_tokens":4,"total_tokens":13}}`,
		"data: [DONE]",
		"",
	}, "\n\n")

	hits := 0
	up := sseUpstream(t, &hits, sse)
	defer up.Close()
	ts := liveStreamServer(t, up.URL)
	defer ts.Close()

	code, raw := postStream(t, ts.URL, map[string]any{
		"model":    "x:model",
		"messages": []map[string]string{{"role": "user", "content": "do it"}},
		"tools":    []map[string]any{{"type": "function", "function": map[string]any{"name": "allow_x"}}},
		"stream":   true,
	})
	if code != http.StatusOK {
		t.Fatalf("status = %d: %s", code, raw)
	}
	if hits != 1 {
		t.Fatalf("upstream hits = %d, want 1", hits)
	}
	chunks, sawDone := parseSSEChunks(t, raw)
	if !sawDone {
		t.Fatalf("missing [DONE]: %s", raw)
	}

	var content strings.Builder
	var tools []ChatDeltaToolCall
	var finish string
	for _, c := range chunks {
		content.WriteString(c.Choices[0].Delta.Content)
		tools = append(tools, c.Choices[0].Delta.ToolCalls...)
		if c.Choices[0].FinishReason != nil {
			finish = *c.Choices[0].FinishReason
		}
	}
	if got := content.String(); got != "Let me check that." {
		t.Fatalf("streamed content = %q, want %q", got, "Let me check that.")
	}
	if len(tools) != 1 || tools[0].Function.Name != "allow_x" || tools[0].Function.Arguments != `{"a":1}` {
		t.Fatalf("kept tool calls = %+v, want one allow_x{a:1}", tools)
	}
	if finish != "tool_calls" {
		t.Fatalf("finish = %q, want tool_calls", finish)
	}
	// The denied call's ARGUMENTS never reach the client (the `fak` audit record names
	// the denied tool by design — that is the transparency contract, not a leak — but
	// it never carries the arguments).
	if strings.Contains(string(raw), `"k":"v"`) {
		t.Fatalf("denied call arguments leaked into the stream: %s", raw)
	}
}

// TestToolBearingStreamNeverLeaksTextFormCallBuriedInContent is the leak-prevention
// proof at the HTTP layer: a model streams a DENIED tool call as CONTENT TEXT (the
// Hermes dialect). The gateway must lift it, adjudicate it (denied), drop it, and —
// crucially — never let the raw dialect or the denied call's arguments touch the wire,
// exactly as the buffered path guarantees.
func TestToolBearingStreamNeverLeaksTextFormCallBuriedInContent(t *testing.T) {
	// The call is delivered as content fragments, split mid-dialect to stress the guard.
	buried := `<tool_call>{"name":"deny_y","arguments":{"secret":"top-secret-xyz"}}</tool_call>`
	frags := []string{`{"choices":[{"delta":{"content":"On it. "}}]}`}
	for i := 0; i < len(buried); i += 5 {
		end := i + 5
		if end > len(buried) {
			end = len(buried)
		}
		piece, _ := json.Marshal(buried[i:end])
		frags = append(frags, `{"choices":[{"delta":{"content":`+string(piece)+`}}]}`)
	}
	lines := []string{`data: {"model":"served-x","choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}`}
	for _, f := range frags {
		lines = append(lines, "data: "+f)
	}
	lines = append(lines,
		`data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":9,"total_tokens":14}}`,
		"data: [DONE]", "")
	sse := strings.Join(lines, "\n\n")

	hits := 0
	up := sseUpstream(t, &hits, sse)
	defer up.Close()
	ts := liveStreamServer(t, up.URL)
	defer ts.Close()

	code, raw := postStream(t, ts.URL, map[string]any{
		"model":    "x:model",
		"messages": []map[string]string{{"role": "user", "content": "do it"}},
		"tools":    []map[string]any{{"type": "function", "function": map[string]any{"name": "deny_y"}}},
		"stream":   true,
	})
	if code != http.StatusOK {
		t.Fatalf("status = %d: %s", code, raw)
	}
	// The raw dialect and the denied call's secret ARGUMENT must appear NOWHERE on the
	// wire — not as content, not as a structured call. (The `fak` audit record names the
	// denied tool by design; it never carries the arguments.)
	for _, needle := range []string{"<tool_call>", "top-secret-xyz", `"secret"`} {
		if strings.Contains(string(raw), needle) {
			t.Fatalf("buried denied call leaked %q into the stream:\n%s", needle, raw)
		}
	}
	chunks, sawDone := parseSSEChunks(t, raw)
	if !sawDone {
		t.Fatalf("missing [DONE]: %s", raw)
	}
	var content strings.Builder
	var finish string
	for _, c := range chunks {
		content.WriteString(c.Choices[0].Delta.Content)
		if len(c.Choices[0].Delta.ToolCalls) != 0 {
			t.Fatalf("a denied buried call reached a tool-call delta: %+v", c.Choices[0].Delta.ToolCalls)
		}
		if c.Choices[0].FinishReason != nil {
			finish = *c.Choices[0].FinishReason
		}
	}
	// The leading prose still streamed; the call text was stripped.
	if got := strings.TrimSpace(content.String()); got != "On it." {
		t.Fatalf("client content = %q, want %q (only the prose, call stripped)", got, "On it.")
	}
	if finish != "stop" {
		t.Fatalf("finish = %q, want stop (every proposed call was denied)", finish)
	}
}

// TestToolBearingStreamLiftsAllowedTextFormCallWithoutLeakingRawText proves the
// allow-side companion: a permitted text-form call is promoted to a STRUCTURED call
// (the client gets the call it needs) while the raw dialect text still never streams as
// content.
func TestToolBearingStreamLiftsAllowedTextFormCallWithoutLeakingRawText(t *testing.T) {
	buried := `<tool_call>{"name":"allow_x","arguments":{"a":1}}</tool_call>`
	piece, _ := json.Marshal(buried)
	sse := strings.Join([]string{
		`data: {"model":"served-x","choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}`,
		`data: {"choices":[{"delta":{"content":` + string(piece) + `}}]}`,
		`data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":7,"total_tokens":12}}`,
		"data: [DONE]", "",
	}, "\n\n")

	hits := 0
	up := sseUpstream(t, &hits, sse)
	defer up.Close()
	ts := liveStreamServer(t, up.URL)
	defer ts.Close()

	code, raw := postStream(t, ts.URL, map[string]any{
		"model":    "x:model",
		"messages": []map[string]string{{"role": "user", "content": "do it"}},
		"tools":    []map[string]any{{"type": "function", "function": map[string]any{"name": "allow_x"}}},
		"stream":   true,
	})
	if code != http.StatusOK {
		t.Fatalf("status = %d: %s", code, raw)
	}
	if strings.Contains(string(raw), "<tool_call>") {
		t.Fatalf("raw dialect text leaked as content: %s", raw)
	}
	chunks, _ := parseSSEChunks(t, raw)
	var tools []ChatDeltaToolCall
	var content strings.Builder
	for _, c := range chunks {
		tools = append(tools, c.Choices[0].Delta.ToolCalls...)
		content.WriteString(c.Choices[0].Delta.Content)
	}
	if strings.TrimSpace(content.String()) != "" {
		t.Fatalf("content should be empty (the whole turn was the call), got %q", content.String())
	}
	if len(tools) != 1 || tools[0].Function.Name != "allow_x" || tools[0].Function.Arguments != `{"a":1}` {
		t.Fatalf("lifted+adjudicated call = %+v, want one allow_x{a:1}", tools)
	}
}

// TestToolBearingStreamConformanceFailClosedNoContent proves the fail-closed floor: an
// upstream that announces tool_calls but emits none parseable, with nothing streamed
// yet, gets a clean 502 — never a benign empty stop that would skip adjudication.
func TestToolBearingStreamConformanceFailClosedNoContent(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"model":"served-x","choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}`,
		`data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":3,"completion_tokens":0,"total_tokens":3}}`,
		"data: [DONE]", "",
	}, "\n\n")

	hits := 0
	up := sseUpstream(t, &hits, sse)
	defer up.Close()
	ts := liveStreamServer(t, up.URL)
	defer ts.Close()

	code, raw := postStream(t, ts.URL, map[string]any{
		"model":    "x:model",
		"messages": []map[string]string{{"role": "user", "content": "do it"}},
		"tools":    []map[string]any{{"type": "function", "function": map[string]any{"name": "allow_x"}}},
		"stream":   true,
	})
	if code != http.StatusBadGateway {
		t.Fatalf("status = %d, want 502 (conformance fail-closed): %s", code, raw)
	}
	if !strings.Contains(string(raw), "tool-call format not recognized") {
		t.Fatalf("missing fail-closed message: %s", raw)
	}
}

// TestToolBearingStreamConformanceFailClosedMidStream proves the mid-stream half: once
// content is on the wire the status can't change, so the gateway emits a terminal error
// frame rather than a benign finish that would mask a skipped call.
func TestToolBearingStreamConformanceFailClosedMidStream(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"model":"served-x","choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}`,
		`data: {"choices":[{"delta":{"content":"Thinking about it now."}}]}`,
		`data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":3,"completion_tokens":5,"total_tokens":8}}`,
		"data: [DONE]", "",
	}, "\n\n")

	hits := 0
	up := sseUpstream(t, &hits, sse)
	defer up.Close()
	ts := liveStreamServer(t, up.URL)
	defer ts.Close()

	code, raw := postStream(t, ts.URL, map[string]any{
		"model":    "x:model",
		"messages": []map[string]string{{"role": "user", "content": "do it"}},
		"tools":    []map[string]any{{"type": "function", "function": map[string]any{"name": "allow_x"}}},
		"stream":   true,
	})
	if code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (headers already sent before the failure): %s", code, raw)
	}
	if !strings.Contains(string(raw), "tool-call format not recognized") {
		t.Fatalf("missing terminal error frame: %s", raw)
	}
	// It must NOT pretend the turn ended cleanly on a skipped call.
	if strings.Contains(string(raw), `"finish_reason":"stop"`) || strings.Contains(string(raw), `"finish_reason":"tool_calls"`) {
		t.Fatalf("conformance failure was masked as a normal finish: %s", raw)
	}
}

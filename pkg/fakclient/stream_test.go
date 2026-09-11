package fakclient_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/pkg/fakclient"
)

// ---------------------------------------------------------------------------
// SSE fixture helpers — one fake gateway upstream per test, serving the exact
// wire shape internal/gateway serves on POST /v1/chat/completions stream:true:
// one `data: {...}\n\n` event per chat.completion.chunk, a role-first opening
// chunk, and a `data: [DONE]` terminator.
// ---------------------------------------------------------------------------

// sseEvent wraps one raw SSE event payload in the `data: ...\n\n` frame the
// gateway writes per chunk (writeSSEData).
func sseEvent(payload string) string {
	return "data: " + payload + "\n\n"
}

func sseChunkJSON(id, model string, delta map[string]any, finish any, usage map[string]any) string {
	return sseChunkJSONTiming(1700000000, id, model, delta, finish, usage)
}

// sseOpenChunk is the role-first opening chunk: delta carries only the assistant
// role — an announcement the gateway emits before any content is known.
func sseOpenChunk(id, model string) string {
	return sseChunkJSON(id, model, map[string]any{"role": "assistant"}, nil, nil)
}

func sseChunkJSONTiming(created int64, id, model string, delta map[string]any, finish any, usage map[string]any) string {
	chunk := map[string]any{
		"id":      id,
		"object":  "chat.completion.chunk",
		"created": created,
		"model":   model,
		"choices": []any{
			map[string]any{
				"index":         0,
				"delta":         delta,
				"finish_reason": finish,
			},
		},
	}
	if usage != nil {
		chunk["usage"] = usage
	}
	b, err := json.Marshal(chunk)
	if err != nil {
		panic(fmt.Sprintf("sseChunkJSON: %v", err))
	}
	return string(b)
}

// recordFragments returns an onDelta that appends each fragment to sink.
// Fragment strings are immutable Go strings, so appending a copy of the header
// is safe; there is no aliasing of caller-owned buffers.
func recordFragments(sink *[]string) func(string) {
	return func(frag string) {
		*sink = append(*sink, frag)
	}
}

// mustStream runs StreamChatCompletions and fails the test on any error.
func mustStream(t *testing.T, c *fakclient.Client, onDelta func(string)) *fakclient.ChatResult {
	t.Helper()
	res, err := c.StreamChatCompletions(context.Background(), fakclient.StreamChatRequest{
		Model:    "test-model",
		Messages: []fakclient.StreamMessage{{Role: "user", Content: "hi"}},
	}, onDelta)
	if err != nil {
		t.Fatalf("StreamChatCompletions: %v", err)
	}
	return res
}

// newStreamServer serves one SSE conversation from frames on every request,
// flushing after each frame so the client observes the stream incrementally.
func newStreamServer(t *testing.T, frames []string) *httptest.Server {
	t.Helper()
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.WriteHeader(http.StatusOK)
		fl, _ := w.(http.Flusher)
		for _, f := range frames {
			_, _ = io.WriteString(w, f)
			if fl != nil {
				fl.Flush()
			}
		}
	}))
	t.Cleanup(ts.Close)
	return ts
}

// ---------------------------------------------------------------------------
// T1 — delta ordering, content assembly, usage/finish/identity propagation,
// and a nil onDelta discarding fragments without panicking.
// ---------------------------------------------------------------------------

func TestStreamChatCompletionsDeltaOrderingAndAssembly(t *testing.T) {
	const id, model = "chatcmpl-fak-t1", "test-model"
	frames := []string{
		sseEvent(sseChunkJSON(id, model, map[string]any{"role": "assistant"}, nil, nil)),
		sseEvent(sseChunkJSON(id, model, map[string]any{"content": "Hel"}, nil, nil)),
		sseEvent(sseChunkJSON(id, model, map[string]any{"content": "lo"}, nil, nil)),
		sseEvent(sseChunkJSON(id, model, map[string]any{"content": ", "}, nil, nil)),
		sseEvent(sseChunkJSON(id, model, map[string]any{"content": "wor"}, nil, nil)),
		sseEvent(sseChunkJSON(id, model, map[string]any{"content": "ld!"}, nil, nil)),
		sseEvent(sseChunkJSON(id, model, map[string]any{}, "stop", map[string]any{
			"prompt_tokens": 12, "completion_tokens": 34, "total_tokens": 46,
		})),
		sseEvent(fakclient.StreamDoneToken),
	}
	ts := newStreamServer(t, frames)
	c := fakclient.New(ts.URL)

	var got []string
	res := mustStream(t, c, recordFragments(&got))

	// (a) onDelta saw exactly the five content fragments in arrival order — the
	// role-first opening chunk is an announcement and must not surface.
	want := []string{"Hel", "lo", ", ", "wor", "ld!"}
	if len(got) != len(want) {
		t.Fatalf("onDelta fragments = %q, want %q", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("fragment[%d] = %q, want %q", i, got[i], want[i])
		}
	}
	// (b) assembled content and (f-adjacent) the fragments join to the same text.
	if res.Content != "Hello, world!" {
		t.Fatalf("Content = %q, want %q", res.Content, "Hello, world!")
	}
	if joined := strings.Join(got, ""); joined != res.Content {
		t.Fatalf("join(fragments)=%q != Content=%q", joined, res.Content)
	}
	// (c) finish reason.
	if res.FinishReason != "stop" {
		t.Fatalf("FinishReason = %q, want %q", res.FinishReason, "stop")
	}
	// (d) usage from the terminal chunk.
	if res.Usage == nil {
		t.Fatal("Usage not populated from the terminal chunk")
	}
	if res.Usage.PromptTokens != 12 || res.Usage.CompletionTokens != 34 || res.Usage.TotalTokens != 46 {
		t.Fatalf("Usage = %+v, want 12/34/46", *res.Usage)
	}
	// (e) stream identity copied from the chunks.
	if res.ID != id || res.Model != model {
		t.Fatalf("ID/Model = %q/%q, want %q/%q", res.ID, res.Model, id, model)
	}

	// (f) a nil onDelta must not panic; assembly is unchanged.
	res2 := mustStream(t, c, nil)
	if res2.Content != "Hello, world!" || res2.FinishReason != "stop" {
		t.Fatalf("nil-onDelta call assembled %+v", res2)
	}
}

// ---------------------------------------------------------------------------
// T2 — tool-call deltas are buffered by index and never surfaced as content,
// including argument fragmentation across chunks.
// ---------------------------------------------------------------------------

func TestStreamChatCompletionsBuffersToolCalls(t *testing.T) {
	const id, model = "chatcmpl-fak-t2", "test-model"
	frames := []string{
		sseEvent(sseChunkJSON(id, model, map[string]any{"role": "assistant"}, nil, nil)),
		sseEvent(sseChunkJSON(id, model, map[string]any{"content": "let me check"}, nil, nil)),
		// First delta carries the call identity with empty arguments.
		sseEvent(sseChunkJSON(id, model, map[string]any{
			"tool_calls": []any{map[string]any{
				"index": 0, "id": "call_1", "type": "function",
				"function": map[string]any{"name": "get_weather", "arguments": ""},
			}},
		}, nil, nil)),
		// Follow-up deltas carry only argument fragments, addressed by index.
		sseEvent(sseChunkJSON(id, model, map[string]any{
			"tool_calls": []any{map[string]any{
				"index":    0,
				"function": map[string]any{"arguments": `{"city":`},
			}},
		}, nil, nil)),
		sseEvent(sseChunkJSON(id, model, map[string]any{
			"tool_calls": []any{map[string]any{
				"index":    0,
				"function": map[string]any{"arguments": `"SF"}`},
			}},
		}, nil, nil)),
		sseEvent(sseChunkJSON(id, model, map[string]any{}, "tool_calls", nil)),
		sseEvent(fakclient.StreamDoneToken),
	}
	ts := newStreamServer(t, frames)
	c := fakclient.New(ts.URL)

	var got []string
	res := mustStream(t, c, recordFragments(&got))

	if len(got) != 1 || got[0] != "let me check" {
		t.Fatalf("onDelta fragments = %q, want only [let me check] — tool-call fragments must never surface as content", got)
	}
	if res.Content != "let me check" {
		t.Fatalf("Content = %q, want %q", res.Content, "let me check")
	}
	if len(res.ToolCalls) != 1 {
		t.Fatalf("ToolCalls = %+v, want exactly one assembled call", res.ToolCalls)
	}
	tc := res.ToolCalls[0]
	if tc.Name != "get_weather" || tc.ID != "call_1" || tc.Index != 0 {
		t.Fatalf("tool call identity = %+v, want get_weather/call_1/index 0", tc)
	}
	if tc.Arguments != `{"city":"SF"}` {
		t.Fatalf("Arguments = %q, want cross-chunk fragments joined to %q", tc.Arguments, `{"city":"SF"}`)
	}
	if res.FinishReason != "tool_calls" {
		t.Fatalf("FinishReason = %q, want %q", res.FinishReason, "tool_calls")
	}
}

// ---------------------------------------------------------------------------
// T3 — a streaming call must NOT inherit the SDK client's total timeout: the
// per-call http.Client copy zeroes Timeout so the context is the only bound;
// a buffered call under the same client still honors it.
// ---------------------------------------------------------------------------

func TestStreamChatCompletionsNoDefaultDeadline(t *testing.T) {
	const id, model = "chatcmpl-fak-t3", "test-model"
	// First delta is on the wire immediately; the rest arrives well after the
	// injected client's 100ms total timeout has already expired.
	late := []string{
		sseEvent(sseChunkJSON(id, model, map[string]any{"content": "wor"}, nil, nil)),
		sseEvent(sseChunkJSON(id, model, map[string]any{"content": "ld!"}, nil, nil)),
		sseEvent(sseChunkJSON(id, model, map[string]any{}, "stop", nil)),
		sseEvent(fakclient.StreamDoneToken),
	}
	frames := append([]string{
		sseEvent(sseOpenChunk(id, model)),
		sseEvent(sseChunkJSON(id, model, map[string]any{"content": "Hel"}, nil, nil)),
		sseEvent(sseChunkJSON(id, model, map[string]any{"content": "lo"}, nil, nil)),
	}, late...)
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		fl, _ := w.(http.Flusher)
		for _, f := range frames[:3] {
			_, _ = io.WriteString(w, f)
			if fl != nil {
				fl.Flush()
			}
		}
		time.Sleep(400 * time.Millisecond)
		for _, f := range frames[3:] {
			_, _ = io.WriteString(w, f)
			if fl != nil {
				fl.Flush()
			}
		}
	}))
	t.Cleanup(ts.Close)

	c := fakclient.New(ts.URL, fakclient.WithHTTPClient(&http.Client{Timeout: 100 * time.Millisecond}))

	var got []string
	start := time.Now()
	res, err := c.StreamChatCompletions(context.Background(), fakclient.StreamChatRequest{
		Model:    "test-model",
		Messages: []fakclient.StreamMessage{{Role: "user", Content: "hi"}},
	}, recordFragments(&got))
	if err != nil {
		t.Fatalf("streamed call failed under a 100ms client timeout: %v (elapsed %s)", err, time.Since(start))
	}
	// All four content fragments must arrive in order despite the mid-stream
	// pause, proving the pause outlived the injected client Timeout.
	want := []string{"Hel", "lo", "wor", "ld!"}
	if len(got) != len(want) {
		t.Fatalf("onDelta fragments = %q, want %q", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("fragment[%d] = %q, want %q", i, got[i], want[i])
		}
	}
	if res.Content != "Helloworld!" || res.FinishReason != "stop" {
		t.Fatalf("assembled result = %+v", res)
	}

	// The buffered (non-streaming) surface must NOT be exempt from the injected
	// client timeout: the same handler sleeping 400ms exceeds 100ms on read.
	ts2 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(400 * time.Millisecond)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"verdict":{"kind":"ALLOW"},"trace_id":"t"}`)
	}))
	t.Cleanup(ts2.Close)
	c2 := fakclient.New(ts2.URL, fakclient.WithHTTPClient(&http.Client{Timeout: 100 * time.Millisecond}))
	if _, err := c2.Adjudicate(context.Background(), fakclient.SyscallRequest{Tool: "t"}); err == nil {
		t.Fatal("buffered Adjudicate succeeded under a 100ms client timeout and a 400ms handler — timeout was bypassed")
	}
}

// ---------------------------------------------------------------------------
// T4 — a mid-stream disconnect, and a clean end without [DONE], both surface as
// *StreamInterruptedError carrying the fragments already delivered.
// ---------------------------------------------------------------------------

func TestStreamChatCompletionsMidStreamDisconnect(t *testing.T) {
	const id, model = "chatcmpl-fak-t4", "test-model"

	// Variant A: the handler writes the opening + two content deltas, then
	// aborts the connection (http.ErrAbortHandler) so the bytes already flushed
	// reach the client and the socket dies mid-stream.
	tsA := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		fl, _ := w.(http.Flusher)
		_, _ = io.WriteString(w, sseEvent(sseOpenChunk(id, model)))
		_, _ = io.WriteString(w, sseEvent(sseChunkJSON(id, model, map[string]any{"content": "Hel"}, nil, nil)))
		_, _ = io.WriteString(w, sseEvent(sseChunkJSON(id, model, map[string]any{"content": "lo "}, nil, nil)))
		if fl != nil {
			fl.Flush()
		}
		panic(http.ErrAbortHandler)
	}))
	t.Cleanup(tsA.Close)

	cA := fakclient.New(tsA.URL)
	var gotA []string
	_, err := cA.StreamChatCompletions(context.Background(), fakclient.StreamChatRequest{
		Model:    "test-model",
		Messages: []fakclient.StreamMessage{{Role: "user", Content: "hi"}},
	}, recordFragments(&gotA))
	if err == nil {
		t.Fatal("mid-stream disconnect returned nil error — a truncated stream must never read as success")
	}
	if !errors.Is(err, fakclient.ErrStreamInterrupted) {
		t.Fatalf("err = %v, want errors.Is(err, ErrStreamInterrupted)", err)
	}
	var sieA *fakclient.StreamInterruptedError
	if !errors.As(err, &sieA) {
		t.Fatalf("err = %T, want *StreamInterruptedError", err)
	}
	if len(sieA.FragmentsDelivered) != 2 || sieA.FragmentsDelivered[0] != "Hel" || sieA.FragmentsDelivered[1] != "lo " {
		t.Fatalf("FragmentsDelivered = %q, want [Hel lo ]", sieA.FragmentsDelivered)
	}
	if !strings.Contains(err.Error(), "2 fragment(s)") {
		t.Fatalf("Error() = %q, want it to mention the delivered fragment count", err.Error())
	}

	// Variant B: the server writes deltas then returns WITHOUT the [DONE]
	// terminator — a clean transport close is still an interruption, not a
	// silently completed answer.
	framesB := []string{
		sseEvent(sseOpenChunk(id, model)),
		sseEvent(sseChunkJSON(id, model, map[string]any{"content": "Hel"}, nil, nil)),
		sseEvent(sseChunkJSON(id, model, map[string]any{"content": "lo "}, nil, nil)),
	}
	tsB := newStreamServer(t, framesB)
	cB := fakclient.New(tsB.URL)
	_, err = cB.StreamChatCompletions(context.Background(), fakclient.StreamChatRequest{
		Model:    "test-model",
		Messages: []fakclient.StreamMessage{{Role: "user", Content: "hi"}},
	}, recordFragments(new([]string)))
	if err == nil {
		t.Fatal("stream ending without [DONE] returned nil error")
	}
	if !errors.Is(err, fakclient.ErrStreamInterrupted) {
		t.Fatalf("err = %v, want errors.Is(err, ErrStreamInterrupted)", err)
	}
	var sieB *fakclient.StreamInterruptedError
	if !errors.As(err, &sieB) {
		t.Fatalf("err = %T, want *StreamInterruptedError", err)
	}
	if len(sieB.FragmentsDelivered) != 2 {
		t.Fatalf("FragmentsDelivered = %q, want 2 entries", sieB.FragmentsDelivered)
	}
}

// ---------------------------------------------------------------------------
// T5 — a non-2xx status maps to *APIError before any delta can fire.
// ---------------------------------------------------------------------------

func TestStreamChatCompletionsHTTPError(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = io.WriteString(w, `{"error":{"message":"bad or missing api key","type":"authentication_error","code":null,"param":null}}`)
	}))
	t.Cleanup(ts.Close)
	c := fakclient.New(ts.URL)

	var fired int
	_, err := c.StreamChatCompletions(context.Background(), fakclient.StreamChatRequest{
		Model:    "test-model",
		Messages: []fakclient.StreamMessage{{Role: "user", Content: "hi"}},
	}, func(string) { fired++ })
	if err == nil {
		t.Fatal("expected an error on a 401")
	}
	apiErr, ok := err.(*fakclient.APIError)
	if !ok {
		t.Fatalf("error is not *APIError: %T %v", err, err)
	}
	if apiErr.StatusCode != http.StatusUnauthorized {
		t.Fatalf("StatusCode = %d, want 401", apiErr.StatusCode)
	}
	if apiErr.Type != "authentication_error" || !strings.Contains(apiErr.Message, "bad or missing api key") {
		t.Fatalf("error body not parsed: %+v", apiErr)
	}
	if fired != 0 {
		t.Fatalf("onDelta fired %d time(s) on a 401; an auth failure must never deliver fragments", fired)
	}
}

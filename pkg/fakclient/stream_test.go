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
	"sync"
	"sync/atomic"
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
	if !errors.Is(err, io.EOF) || errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("bare EOF misclassified: %v", err)
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

// ---------------------------------------------------------------------------
// T6 - an in-band `data: {"error":{...}}` frame followed by `data: [DONE]` is a
// mid-stream upstream failure. It must fail closed with a typed error carrying
// the server message, never be honored as a silently completed 0-token answer
// (issue #12776).
// ---------------------------------------------------------------------------

func TestStreamChatCompletionsInBandErrorFrameFailsClosed(t *testing.T) {
	const id, model = "chatcmpl-fak-t6", "test-model"
	frames := []string{
		sseEvent(sseOpenChunk(id, model)),
		sseEvent(sseChunkJSON(id, model, map[string]any{"content": "Hel"}, nil, nil)),
		sseEvent(`{"error":{"message":"upstream exploded mid-stream","type":"upstream_error","code":null,"param":null}}`),
		sseEvent(fakclient.StreamDoneToken),
	}
	ts := newStreamServer(t, frames)
	c := fakclient.New(ts.URL)

	res, err := c.StreamChatCompletions(context.Background(), fakclient.StreamChatRequest{
		Model:    "test-model",
		Messages: []fakclient.StreamMessage{{Role: "user", Content: "hi"}},
	}, func(string) {})
	if err == nil {
		t.Fatalf("in-band error frame returned nil error with result %+v - a mid-stream failure must never read as success", res)
	}
	if res != nil {
		t.Fatalf("result = %+v, want nil on a mid-stream failure", res)
	}
	apiErr, ok := err.(*fakclient.APIError)
	if !ok {
		t.Fatalf("error is not *APIError: %T %v", err, err)
	}
	if apiErr.Type != "upstream_error" || !strings.Contains(apiErr.Message, "upstream exploded mid-stream") {
		t.Fatalf("in-band error not surfaced: %+v", apiErr)
	}
}

// Estimate only; this test has not been timed.
// fak-test:runtime fast est=10ms lane=default
func TestStreamInterruptedErrorMatchesCauseAndSentinel(t *testing.T) {
	unrelated := errors.New("unrelated transport")
	for _, cause := range []error{context.Canceled, context.DeadlineExceeded, io.EOF, unrelated, nil} {
		wrapped := &fakclient.StreamInterruptedError{Err: cause}
		if errors.Unwrap(wrapped) != fakclient.ErrStreamInterrupted || !errors.Is(wrapped, fakclient.ErrStreamInterrupted) {
			t.Fatal("historical sentinel contract changed")
		}
		for _, target := range []error{context.Canceled, context.DeadlineExceeded, io.EOF, unrelated} {
			if got, want := errors.Is(wrapped, target), errors.Is(cause, target); got != want {
				t.Fatalf("cause=%v target=%v got=%t want=%t", cause, target, got, want)
			}
		}
	}
	for _, cause := range []error{context.Canceled, context.DeadlineExceeded} {
		wrappedCause := &fakclient.StreamInterruptedError{Err: fmt.Errorf("transport: %w", cause)}
		if !errors.Is(wrappedCause, cause) {
			t.Fatal("nested cause lost")
		}
	}
	self := &fakclient.StreamInterruptedError{}
	self.Err = self
	if self.Is(context.Canceled) || !errors.Is(self, fakclient.ErrStreamInterrupted) {
		t.Fatal("direct self-reference changed match semantics")
	}
	var absent *fakclient.StreamInterruptedError
	if absent.Is(context.Canceled) {
		t.Fatal("nil receiver matched cause")
	}
}

// Estimate only; this test has not been timed.
// fak-test:runtime medium est=1s lane=default
func TestStreamChatCompletionsPartialCancellationKeepsCause(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, sseEvent(sseOpenChunk("partial-cancel", "test-model")))
		_, _ = io.WriteString(w, sseEvent(sseChunkJSON("partial-cancel", "test-model", map[string]any{"content": "kept"}, nil, nil)))
		w.(http.Flusher).Flush()
		<-r.Context().Done()
	}))
	t.Cleanup(ts.Close)
	var fragments []string
	result, err := fakclient.New(ts.URL).StreamChatCompletions(ctx, fakclient.StreamChatRequest{Model: "test-model", Messages: []fakclient.StreamMessage{{Role: "user", Content: "hi"}}}, func(fragment string) { fragments = append(fragments, fragment); cancel() })
	if result != nil || !errors.Is(err, fakclient.ErrStreamInterrupted) || !errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("partial cancel result=%v err=%v", result, err)
	}
	var interrupted *fakclient.StreamInterruptedError
	if !errors.As(err, &interrupted) || len(fragments) != 1 || fragments[0] != "kept" || len(interrupted.FragmentsDelivered) != 1 || interrupted.FragmentsDelivered[0] != "kept" {
		t.Fatalf("partial evidence changed: fragments=%q interruption=%#v", fragments, interrupted)
	}
}

// Estimate only; this test has not been timed.
// fak-test:runtime medium est=1s lane=default
func TestStreamChatCompletionsFakExtensionAssembly(t *testing.T) {
	// Both gateway emitters attach fak to the finish chunk; an opted-in usage
	// frame follows with empty choices and no fak field.
	const extension = `{"adjudications":[{"tool":"read","decision":"allow"}],"future":{"nested":[1,{"opaque":"kept"}]}}`
	frames := []string{
		sseEvent(sseOpenChunk("extension", "test-model")),
		sseEvent(`{"choices":[{"delta":{"content":"hello"}}]}`),
		sseEvent(`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call-1","type":"function","function":{"name":"read","arguments":"{\"path\":"}}]}}]}`),
		sseEvent(`{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"x\"}"}}]}}]}`),
		sseEvent(`{"choices":[{"delta":{},"finish_reason":"tool_calls"}],"fak":` + extension + `}`),
		sseEvent(`{"choices":[],"usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5}}`),
		sseEvent(fakclient.StreamDoneToken),
	}
	ts := newStreamServer(t, frames)
	var fragments []string
	res := mustStream(t, fakclient.New(ts.URL), recordFragments(&fragments))
	if string(res.Fak) != extension || res.Content != "hello" || len(fragments) != 1 || fragments[0] != "hello" || res.FinishReason != "tool_calls" || res.ID != "extension" || res.Model != "test-model" {
		t.Fatalf("extension/content assembly changed: result=%+v fragments=%q", res, fragments)
	}
	if len(res.ToolCalls) != 1 || res.ToolCalls[0].ID != "call-1" || res.ToolCalls[0].Name != "read" || res.ToolCalls[0].Arguments != `{"path":"x"}` {
		t.Fatalf("tool assembly changed: %+v", res.ToolCalls)
	}
	if res.Usage == nil || res.Usage.PromptTokens != 2 || res.Usage.CompletionTokens != 3 || res.Usage.TotalTokens != 5 {
		t.Fatalf("usage assembly changed: %+v", res.Usage)
	}
}

// Estimate only; this test has not been timed.
// fak-test:runtime medium est=1s lane=default
func TestStreamChatCompletionsFakExtensionPresence(t *testing.T) {
	for _, tc := range []struct {
		name   string
		frames []string
		want   string
	}{
		{"absent", []string{`{"choices":[]}`}, ""},
		{"present_without_choices", []string{`{"fak":{"unknown":true}}`, `{"choices":[]}`}, `{"unknown":true}`},
		{"replacement", []string{`{"fak":{"old":1}}`, `{"fak":{"new":2}}`, `{"choices":[]}`}, `{"new":2}`},
		{"explicit_null", []string{`{"fak":{"old":1}}`, `{"fak":null}`, `{"choices":[]}`}, `null`},
		{"after_null", []string{`{"fak":null}`, `{"fak":{"new":2}}`}, `{"new":2}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			frames := []string{sseEvent(sseOpenChunk("presence", "test-model"))}
			for _, frame := range tc.frames {
				frames = append(frames, sseEvent(frame))
			}
			frames = append(frames, sseEvent(fakclient.StreamDoneToken))
			res := mustStream(t, fakclient.New(newStreamServer(t, frames).URL), nil)
			if string(res.Fak) != tc.want {
				t.Fatalf("fak=%s want=%s", res.Fak, tc.want)
			}
			encoded, err := json.Marshal(res)
			if err != nil {
				t.Fatal(err)
			}
			var fields map[string]json.RawMessage
			if err := json.Unmarshal(encoded, &fields); err != nil {
				t.Fatal(err)
			}
			got, present := fields["fak"]
			if present != (tc.want != "") || string(got) != tc.want {
				t.Fatalf("serialized fak=%s present=%v want=%s", got, present, tc.want)
			}
		})
	}
}

// Estimate only; this test has not been timed.
// fak-test:runtime medium est=1s lane=default
func TestStreamChatCompletionsFakExtensionDoesNotMaskFailure(t *testing.T) {
	for _, tc := range []struct {
		name        string
		tail        []string
		interrupted bool
	}{
		{"bare_eof", nil, true},
		{"in_band_error", []string{sseEvent(`{"fak":{"later":true},"error":{"message":"upstream failed","type":"upstream_error"}}`), sseEvent(fakclient.StreamDoneToken)}, false},
		{"invalid_json", []string{sseEvent(`{"fak":`), sseEvent(fakclient.StreamDoneToken)}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			frames := append([]string{sseEvent(`{"choices":[],"fak":{"adjudications":[]}}`)}, tc.tail...)
			res, err := fakclient.New(newStreamServer(t, frames).URL).StreamChatCompletions(context.Background(), fakclient.StreamChatRequest{Model: "test-model"}, nil)
			if res != nil || err == nil {
				t.Fatalf("failed stream exposed success metadata: result=%+v err=%v", res, err)
			}
			if tc.interrupted {
				if !errors.Is(err, fakclient.ErrStreamInterrupted) || !errors.Is(err, io.EOF) {
					t.Fatalf("EOF semantics changed: %v", err)
				}
			} else {
				var apiErr *fakclient.APIError
				if !errors.As(err, &apiErr) {
					t.Fatalf("error semantics changed: %T %v", err, err)
				}
				if tc.name == "in_band_error" && (apiErr.Message != "upstream failed" || apiErr.Type != "upstream_error") {
					t.Fatalf("upstream refusal changed: %+v", apiErr)
				}
				if tc.name == "invalid_json" && apiErr.Type != "stream_decode_error" {
					t.Fatalf("decode refusal changed: %+v", apiErr)
				}
			}
		})
	}
}

type errorBodyRoundTripper func(*http.Request) (*http.Response, error)

func (f errorBodyRoundTripper) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

type partialErrorResponseBody struct {
	data          []byte
	cause         error
	reads, closes int
}

func (b *partialErrorResponseBody) Read(p []byte) (int, error) {
	b.reads++
	n := copy(p, b.data)
	b.data = b.data[n:]
	return n, b.cause
}
func (b *partialErrorResponseBody) Close() error { b.closes++; return nil }

// fak-test:runtime fast est=10ms lane=default
// Unmeasured estimate; injected HTTP response body, no upstream model.
func TestStreamHTTPErrorBodyPreservesStatusAndReadCause(t *testing.T) {
	for _, cause := range []error{context.Canceled, context.DeadlineExceeded, io.ErrUnexpectedEOF, errors.New("transport body failure")} {
		t.Run(cause.Error(), func(t *testing.T) {
			body := &partialErrorResponseBody{data: []byte(`{"error":{"message":"partial body retained","type":"upstream_error","code":"broken"}}`), cause: cause}
			requests, callbacks := 0, 0
			c := fakclient.New("http://fixture.invalid", fakclient.WithHTTPClient(&http.Client{Transport: errorBodyRoundTripper(func(r *http.Request) (*http.Response, error) {
				requests++
				return &http.Response{StatusCode: http.StatusBadGateway, Header: make(http.Header), Body: body, Request: r}, nil
			})}))
			result, err := c.StreamChatCompletions(context.Background(), fakclient.StreamChatRequest{}, func(string) { callbacks++ })
			var api *fakclient.APIError
			var interrupted *fakclient.StreamInterruptedError
			if result != nil || !errors.Is(err, cause) || !errors.As(err, &api) || api.StatusCode != http.StatusBadGateway || api.Type != "upstream_error" || api.Message != "partial body retained" || api.Code != "broken" {
				t.Fatalf("status/body/cause lost: result=%v err=%v api=%+v", result, err, api)
			}
			if errors.As(err, &interrupted) || errors.Is(err, fakclient.ErrStreamInterrupted) {
				t.Fatal("HTTP error body misclassified as an SSE interruption")
			}
			if requests != 1 || callbacks != 0 || body.reads != 1 || body.closes != 1 {
				t.Fatalf("request/ownership changed: requests=%d callbacks=%d reads=%d closes=%d", requests, callbacks, body.reads, body.closes)
			}
		})
	}
}

type signaledErrorResponseBody struct {
	io.ReadCloser
	started chan struct{}
	once    sync.Once
	closes  atomic.Int32
}

func (b *signaledErrorResponseBody) Read(p []byte) (int, error) {
	b.once.Do(func() { close(b.started) })
	return b.ReadCloser.Read(p)
}
func (b *signaledErrorResponseBody) Close() error { b.closes.Add(1); return b.ReadCloser.Close() }

// fak-test:runtime medium est=1s lane=default
// Unmeasured estimate. Real loopback HTTP headers and cancellation during body read.
func TestStreamHTTPErrorBodyCancellationAfterHeaders(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	release := make(chan struct{})
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusServiceUnavailable)
		w.(http.Flusher).Flush()
		select {
		case <-r.Context().Done():
		case <-release:
		}
	}))
	defer ts.Close()
	defer close(release)
	transport := http.DefaultTransport.(*http.Transport).Clone()
	defer transport.CloseIdleConnections()
	started := make(chan struct{})
	var body *signaledErrorResponseBody
	c := fakclient.New(ts.URL, fakclient.WithHTTPClient(&http.Client{Transport: errorBodyRoundTripper(func(r *http.Request) (*http.Response, error) {
		resp, err := transport.RoundTrip(r)
		if err == nil {
			body = &signaledErrorResponseBody{ReadCloser: resp.Body, started: started}
			resp.Body = body
		}
		return resp, err
	})}))
	type completion struct {
		result *fakclient.ChatResult
		err    error
	}
	done := make(chan completion, 1)
	var callbacks atomic.Int32
	go func() {
		result, err := c.StreamChatCompletions(ctx, fakclient.StreamChatRequest{}, func(string) { callbacks.Add(1) })
		done <- completion{result, err}
	}()
	select {
	case <-started:
		cancel()
	case early := <-done:
		t.Fatalf("request ended before error-body read: %v", early.err)
	case <-ctx.Done():
		t.Fatal("error-body read did not start")
	}
	select {
	case got := <-done:
		var api *fakclient.APIError
		if got.result != nil || !errors.Is(got.err, context.Canceled) || !errors.As(got.err, &api) || api.StatusCode != http.StatusServiceUnavailable {
			t.Fatalf("post-header cancellation lost: result=%v err=%v api=%+v", got.result, got.err, api)
		}
		if errors.Is(got.err, fakclient.ErrStreamInterrupted) || callbacks.Load() != 0 || body == nil || body.closes.Load() != 1 {
			t.Fatal("post-header cancellation violated callback/body ownership contract")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("canceled error-body request did not return")
	}
}

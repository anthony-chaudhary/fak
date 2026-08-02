package gateway

// messages_stream_inband_test.go — the gateway half of #5491, driven end to end through the
// real handler exactly as the reported failure arrives: a client asks the flagship
// `fak guard -- claude` passthrough to stream, and the upstream answers HTTP 200 +
// text/event-stream and then refuses IN-BAND with an SSE `error` frame before any
// message_start. Nothing has reached the client at that point, so the response is still fak's
// to choose — and what it chooses must be the upstream's own signal, not a hardcoded 502 that
// also cut off every second chance.

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// inBandRefusalFrames is the reported wire shape: a well-formed SSE `error` frame as the FIRST
// event of an accepted (200) stream.
func inBandRefusalFrames(errType string) string {
	return "event: error\n" +
		`data: {"type":"error","error":{"type":"` + errType + `","message":"Overloaded"}}` + "\n\n"
}

// writeInBandRefusal answers 200 + text/event-stream and immediately refuses. Retry-After: 0 is
// a test-speed lever only (the retry loop honors it exactly as it does on a 429/503), so the
// pinned attempt budget is spent without sleeping.
func writeInBandRefusal(w http.ResponseWriter, errType string) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Retry-After", "0")
	w.WriteHeader(http.StatusOK)
	_, _ = io.WriteString(w, inBandRefusalFrames(errType))
	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}
}

func inBandStreamInbound() []byte {
	return []byte(`{"model":"claude-test","max_tokens":4096,"stream":true,` +
		`"messages":[{"role":"user","content":"say hello"}]}`)
}

// The reported bug, end to end. A transient in-band refusal must be RETRIED across the pinned
// budget (three upstream hits, not one), and once the budget is genuinely spent the client must
// see the upstream's OWN condition — HTTP 529 / overloaded_error / upstream_overloaded — so a
// wrapped client that backs off on an overload has something to branch on. Before the fix this
// was one upstream hit and a hardcoded 502 whose errType(502) told the client `server_error`:
// the gateway crashed, as far as anyone downstream could tell.
func TestMessagesStreamInBandRefusalRetriesAndSurfacesUpstreamSignal(t *testing.T) {
	t.Setenv("FAK_PLANNER_MAX_ATTEMPTS", "3")
	t.Setenv("FAK_PLANNER_RETRY_BUDGET", "0")
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})

	var hits int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&hits, 1)
		writeInBandRefusal(w, "overloaded_error")
	}))
	defer upstream.Close()

	srv, err := New(Config{EngineID: "test", Model: "claude-test", BaseURL: upstream.URL, Provider: "anthropic", APIKey: "configured-key"})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body, status := postStreamCollect(t, ts.URL, inBandStreamInbound())

	if got := atomic.LoadInt32(&hits); got != 3 {
		t.Fatalf("upstream hit %d times, want 3 (FAK_PLANNER_MAX_ATTEMPTS): an in-band transient must ride the same retry budget as a 529 on the wire", got)
	}
	if status != 529 {
		t.Fatalf("status = %d, want 529 (the upstream's own overload status, not a flattened 502): %s", status, body)
	}
	if !strings.Contains(body, `"type":"overloaded_error"`) {
		t.Fatalf(`error type must be the upstream's own overloaded_error, not server_error: %s`, body)
	}
	if !strings.Contains(body, `"code":"upstream_overloaded"`) {
		t.Fatalf("error code must be the machine-branchable upstream_overloaded: %s", body)
	}
}

// The negative fence at the gateway: an in-band REQUEST error is not transient, so it surfaces
// on the first try with its real status — no retry burst, and a 400 that reads as a 400.
func TestMessagesStreamInBandRequestErrorSurfacesOnce(t *testing.T) {
	t.Setenv("FAK_PLANNER_MAX_ATTEMPTS", "3")
	t.Setenv("FAK_PLANNER_RETRY_BUDGET", "0")
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})

	var hits int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&hits, 1)
		writeInBandRefusal(w, "invalid_request_error")
	}))
	defer upstream.Close()

	srv, err := New(Config{EngineID: "test", Model: "claude-test", BaseURL: upstream.URL, Provider: "anthropic", APIKey: "configured-key"})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body, status := postStreamCollect(t, ts.URL, inBandStreamInbound())
	if got := atomic.LoadInt32(&hits); got != 1 {
		t.Fatalf("upstream hit %d times, want 1 (a request error must not be retried)", got)
	}
	if status != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400 (the upstream's own request-error status): %s", status, body)
	}
	if !strings.Contains(body, `"code":"upstream_invalid_request"`) {
		t.Fatalf("error code must name the request error: %s", body)
	}
}

// The fallback half of #5491. An `error` frame this build cannot classify carries no status to
// surface, so it must take the SAME door every other pre-start failure takes: the turn falls
// back to the buffered path and the client still gets its turn — instead of being cut off by
// the `wroteError` flag, which made the hardcoded 502 the one and only outcome.
func TestMessagesStreamUnclassifiedInBandRefusalFallsBackToBuffered(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})

	var streamHits, bufferedHits int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		if strings.Contains(string(raw), `"stream":true`) {
			atomic.AddInt32(&streamHits, 1)
			writeInBandRefusal(w, "some_future_error") // a type this build does not know
			return
		}
		atomic.AddInt32(&bufferedHits, 1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, `{"id":"msg_b","type":"message","role":"assistant","model":"claude-test",`+
			`"content":[{"type":"text","text":"buffered rescue"}],"stop_reason":"end_turn",`+
			`"usage":{"input_tokens":3,"output_tokens":2}}`)
	}))
	defer upstream.Close()

	srv, err := New(Config{EngineID: "test", Model: "claude-test", BaseURL: upstream.URL, Provider: "anthropic", APIKey: "configured-key"})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body, status := postStreamCollect(t, ts.URL, inBandStreamInbound())
	if status != http.StatusOK {
		t.Fatalf("status = %d, want 200 (the buffered fallback served the turn): %s", status, body)
	}
	if got := atomic.LoadInt32(&bufferedHits); got != 1 {
		t.Fatalf("buffered upstream hits = %d, want 1 — the unclassified in-band refusal must not cut off the fallback", got)
	}
	if !strings.Contains(body, "buffered rescue") {
		t.Fatalf("client did not receive the buffered turn:\n%s", body)
	}
	if got := atomic.LoadInt32(&streamHits); got != 1 {
		t.Fatalf("streaming upstream hits = %d, want 1", got)
	}
}

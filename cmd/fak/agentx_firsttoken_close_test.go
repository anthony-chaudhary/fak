package main

import (
	"bufio"
	"context"
	"io"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// sseStatusBody is a minimal http.RoundTripper body source for the first-token
// close regression tests. It serves a fixed SSE payload and counts Close calls.
type sseStatusBody struct {
	io.Reader
	closes int64
}

func (b *sseStatusBody) Close() error {
	atomic.AddInt64(&b.closes, 1)
	return nil
}

// sseTransport is a RoundTripper that always returns 200 OK with the given SSE
// payload, without any network I/O.
type sseTransport struct {
	payload string
}

func (t *sseTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	return &http.Response{
		StatusCode: http.StatusOK,
		Header:     make(http.Header),
		Body:       &sseStatusBody{Reader: strings.NewReader(t.payload)},
		Request:    req,
	}, nil
}

// consumeUntilDone mimics the real AgentX runner: wrap the body, read SSE lines
// until the trimmed "data: [DONE]" line, then break and Close WITHOUT draining
// to EOF.
func consumeUntilDone(t *testing.T, body io.ReadCloser) {
	t.Helper()
	br := bufio.NewReader(body)
	for {
		line, err := br.ReadString('\n')
		if err != nil {
			break
		}
		if strings.TrimSpace(line) == "data: [DONE]" {
			break
		}
	}
	if err := body.Close(); err != nil {
		t.Fatalf("body.Close() returned error: %v", err)
	}
}

// holdWithin asserts a slot can be acquired quickly; a leaked slot makes the
// short-timeout hold fail fast instead of hanging the test.
func holdWithin(t *testing.T, gate *prefillGate, d time.Duration) *prefillHold {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), d)
	defer cancel()
	h, err := gate.hold(ctx)
	if err != nil {
		t.Fatalf("prefill slot leaked: gate.hold did not succeed within %s: %v", d, err)
	}
	return h
}

// TestFirstTokenClose_DoneWithoutContentReleasesSlot is the critical defect
// probe: a content-free stream consumed like the real runner (break at
// "data: [DONE]", then Close, no EOF drain) MUST release its prefill slot.
// fak-test:runtime fast est=5ms lane=default
func TestFirstTokenClose_DoneWithoutContentReleasesSlot(t *testing.T) {
	const payload = "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n" +
		"data: [DONE]\n\n"

	gate := newPrefillGate(1)
	tr := &gatedTransport{base: &sseTransport{payload: payload}, gate: gate}
	client := &http.Client{Transport: tr}

	req, err := http.NewRequest(http.MethodPost, "http://agentx.invalid/v1/chat/completions", nil)
	if err != nil {
		t.Fatalf("NewRequest: %v", err)
	}
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("client.Do: %v", err)
	}

	consumeUntilDone(t, resp.Body)

	// The slot must now be free. A short timeout fails fast on a leak.
	h := holdWithin(t, gate, 500*time.Millisecond)
	h.release()
}

// TestFirstTokenClose_ContentFrameAcksAndReleases checks the happy path: the
// first non-empty content delta fires the callback once and frees the slot.
// fak-test:runtime fast est=5ms lane=default
func TestFirstTokenClose_ContentFrameAcksAndReleases(t *testing.T) {
	const payload = "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n" +
		"data: [DONE]\n\n"

	gate := newPrefillGate(1)
	var calls int64
	holder := mustHold(t, gate)
	body, _ := newTestFirstTokenBody(holder, payload, func(ttftNS int64, firstFrame string) bool {
		atomic.AddInt64(&calls, 1)
		return true
	})

	consumeUntilDone(t, body)

	if got := atomic.LoadInt64(&calls); got != 1 {
		t.Fatalf("callback fired %d times, want exactly 1", got)
	}
	h := holdWithin(t, gate, 500*time.Millisecond)
	h.release()
}

// TestFirstTokenClose_DoubleCloseIsSafeAndFiresAtMostOnce checks that calling
// Close twice is a no-op, the callback does not double-fire, and the wrapped
// body's Close is still delegated to exactly as a normal io.ReadCloser expects.
// fak-test:runtime fast est=5ms lane=default
func TestFirstTokenClose_DoubleCloseIsSafeAndFiresAtMostOnce(t *testing.T) {
	const payload = "data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n" +
		"data: [DONE]\n\n"

	gate := newPrefillGate(1)
	var calls int64
	holder := mustHold(t, gate)
	body, rc := newTestFirstTokenBody(holder, payload, func(ttftNS int64, firstFrame string) bool {
		atomic.AddInt64(&calls, 1)
		return true
	})

	// Drain the content so the Read path acks.
	if _, err := io.Copy(io.Discard, body); err != nil {
		t.Fatalf("drain: %v", err)
	}
	if err := body.Close(); err != nil {
		t.Fatalf("first Close: %v", err)
	}
	if err := body.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}

	if got := atomic.LoadInt64(&calls); got != 1 {
		t.Fatalf("callback fired %d times after double Close, want exactly 1", got)
	}
	// Close must still delegate to the wrapped body: a firstTokenBody that
	// swallowed Close would strand callers that rely on the underlying
	// connection/body cleanup.
	if got := atomic.LoadInt64(&rc.closes); got < 1 {
		t.Fatalf("wrapped body Close invoked %d times, want >= 1", got)
	}
	h := holdWithin(t, gate, 500*time.Millisecond)
	h.release()
}

// TestFirstTokenClose_ConcurrentReadCloseNoDoubleFire runs Read and Close
// concurrently; the callback must fire exactly once and the slot must free.
// Intended to be exercised under -race.
// fak-test:runtime fast est=5ms lane=default
func TestFirstTokenClose_ConcurrentReadCloseNoDoubleFire(t *testing.T) {
	const payload = "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{\"content\":\"hello world\"}}]}\n\n" +
		"data: [DONE]\n\n"

	gate := newPrefillGate(1)
	var calls int64
	holder := mustHold(t, gate)
	body, _ := newTestFirstTokenBody(holder, payload, func(ttftNS int64, firstFrame string) bool {
		atomic.AddInt64(&calls, 1)
		return true
	})

	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		_, _ = io.Copy(io.Discard, body)
	}()
	go func() {
		defer wg.Done()
		_ = body.Close()
	}()
	wg.Wait()

	if got := atomic.LoadInt64(&calls); got != 1 {
		t.Fatalf("callback fired %d times under concurrent Read+Close, want exactly 1", got)
	}
	h := holdWithin(t, gate, 500*time.Millisecond)
	h.release()
}

// TestFirstTokenClose_DoneWithoutContentFiresReturnCallback observes the return
// path directly: a content-free stream consumed to "data: [DONE]" then Closed
// must fire the callback exactly once with ttftNS==0 and an empty frame.
// fak-test:runtime fast est=5ms lane=default
func TestFirstTokenClose_DoneWithoutContentFiresReturnCallback(t *testing.T) {
	const payload = "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n" +
		"data: [DONE]\n\n"

	gate := newPrefillGate(1)
	var calls int64
	var gotTTFT int64 = -1
	var gotFrame = "unset"
	holder := mustHold(t, gate)
	body, _ := newTestFirstTokenBody(holder, payload, func(ttftNS int64, firstFrame string) bool {
		atomic.AddInt64(&calls, 1)
		atomic.StoreInt64(&gotTTFT, ttftNS)
		gotFrame = firstFrame
		return true
	})

	consumeUntilDone(t, body)

	if got := atomic.LoadInt64(&calls); got != 1 {
		t.Fatalf("return-path callback fired %d times, want exactly 1", got)
	}
	if ttft := atomic.LoadInt64(&gotTTFT); ttft != 0 {
		t.Fatalf("return-path callback ttftNS = %d, want 0", ttft)
	}
	if gotFrame != "" {
		t.Fatalf("return-path callback firstFrame = %q, want empty", gotFrame)
	}
	h := holdWithin(t, gate, 500*time.Millisecond)
	h.release()
}

// noReturnPathBody is the pre-fix shape of firstTokenBody: it serves the same
// SSE bytes, but its Close does NOT fire the return path. A caller that stops
// reading at "data: [DONE]" and closes therefore leaks the prefill slot. It
// exists only so the negative control below can prove the leak assertion is
// load-bearing.
type noReturnPathBody struct{ r io.Reader }

func (b *noReturnPathBody) Read(p []byte) (int, error) { return b.r.Read(p) }
func (b *noReturnPathBody) Close() error               { return nil }

// TestFirstTokenClose_LeakNegativeControl models the pre-fix bug: a body whose
// Close omits the return-path release, driven exactly like the real runner
// (consume to "data: [DONE]" then Close, no EOF drain), must LEAVE the slot
// held. That a short-timeout hold fails here is what proves the main regression
// test (TestFirstTokenClose_DoneWithoutContentReleasesSlot) actually catches the
// defect rather than passing on a gate that never held anything.
// fak-test:runtime fast est=150ms lane=default
func TestFirstTokenClose_LeakNegativeControl(t *testing.T) {
	const payload = "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n" +
		"data: [DONE]\n\n"

	gate := newPrefillGate(1)
	// The in-flight request already owns the only slot; the pre-fix body will
	// fail to give it back.
	holder := mustHold(t, gate)

	body := &noReturnPathBody{r: strings.NewReader(payload)}
	consumeUntilDone(t, body)

	// A genuinely-leaking body must keep the gate full: a short-timeout hold
	// has to fail. If the gate hands out a second slot the control did not
	// model the defect, so the main probe could not vouch for it.
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	if h, err := gate.hold(ctx); err == nil {
		h.release()
		t.Fatalf("negative control did not model a leak: gate dispatched a second slot after Close")
	}

	holder.release()
}

// mustHold acquires a slot without a real deadline; used to build a directly
// constructed firstTokenBody.
func mustHold(t *testing.T, gate *prefillGate) *prefillHold {
	t.Helper()
	h, err := gate.hold(context.Background())
	if err != nil {
		t.Fatalf("hold: %v", err)
	}
	return h
}

// newTestFirstTokenBody constructs a firstTokenBody directly so the test owns
// the callback and can count its invocations. It mirrors the transport's wiring:
// the callback releases the hold and reports true regardless of the wrapped
// counter's return value. The wrapped sseStatusBody is returned so callers can
// assert the Close delegation actually happened.
func newTestFirstTokenBody(hold *prefillHold, payload string, cb func(int64, string) bool) (*firstTokenBody, *sseStatusBody) {
	rc := &sseStatusBody{Reader: strings.NewReader(payload)}
	return &firstTokenBody{
		rc:    rc,
		r:     bufio.NewReader(rc),
		hold:  hold,
		start: time.Now(),
		cb: func(ttftNS int64, firstFrame string) bool {
			cb(ttftNS, firstFrame)
			hold.release()
			return true
		},
	}, rc
}

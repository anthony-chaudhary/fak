package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agentxbench"
)

func TestAgentXSelfCheck(t *testing.T) {
	var stdout, stderr bytes.Buffer
	args := []string{"--selfcheck", "--agents=2", "--turns=2"}

	code := runAgentX(&stdout, &stderr, args)
	if code != 0 {
		t.Fatalf("expected code 0, got %d. stderr: %s", code, stderr.String())
	}

	out := stdout.String()
	if !strings.Contains(out, "AgentX Benchmark Result:") {
		t.Fatalf("expected benchmark result header, got: %s", out)
	}
	if !strings.Contains(out, "Status         : SIMULATED") {
		t.Fatalf("expected SIMULATED, got: %s", out)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestAgentXLiveRunNeedsEngineAndHardware(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runAgentX(&stdout, &stderr, []string{"--endpoint=http://127.0.0.1:1/v1", "--agents=1", "--turns=1"})
	if code == 0 {
		t.Fatalf("live run without --engine/--hardware succeeded; stdout: %s", stdout.String())
	}
}

func TestAgentXSelfCheckJSONAndValidate(t *testing.T) {
	tmpDir := t.TempDir()
	receiptPath := filepath.Join(tmpDir, "agentx-receipt.json")

	var stdout, stderr bytes.Buffer
	args := []string{
		"--selfcheck",
		"--agents=2",
		"--turns=2",
		"--json",
		"--out=" + receiptPath,
	}

	code := runAgentX(&stdout, &stderr, args)
	if code != 0 {
		t.Fatalf("expected code 0, got %d. stderr: %s", code, stderr.String())
	}

	var receipt agentxbench.AgentXReceipt
	if err := json.Unmarshal(stdout.Bytes(), &receipt); err != nil {
		t.Fatalf("failed to unmarshal JSON output: %v", err)
	}
	if receipt.Schema != agentxbench.SchemaIdentifier {
		t.Fatalf("unexpected schema: %s", receipt.Schema)
	}
	if receipt.ValidationStatus != agentxbench.StatusSimulated {
		t.Fatalf("expected SIMULATED, got: %s", receipt.ValidationStatus)
	}

	// Verify file was written
	data, err := os.ReadFile(receiptPath)
	if err != nil {
		t.Fatalf("failed to read written receipt: %v", err)
	}
	if len(data) == 0 {
		t.Fatalf("written receipt is empty")
	}

	// A self-check receipt is never admitted as evidence.
	stdout.Reset()
	stderr.Reset()
	valCode := runAgentX(&stdout, &stderr, []string{"--validate=" + receiptPath})
	if valCode == 0 {
		t.Fatalf("validate admitted a mock receipt: %s", stdout.String())
	}
	if !strings.Contains(stderr.String(), "SIMULATED") {
		t.Fatalf("expected SIMULATED refusal, got: %s", stderr.String())
	}
}

type roundTripperFunc func(*http.Request) (*http.Response, error)

func (f roundTripperFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

// fak-test:runtime fast est=500ms lane=default
func TestFirstTokenRelease(t *testing.T) {
	const (
		roleOnly   = `{"choices":[{"delta":{"role":"assistant","content":""}}]}`
		contentTok = `{"choices":[{"delta":{"content":"tok"}}]}`
		contentTwo = `{"choices":[{"delta":{"content":"tok2"}}]}`
		reasonOnly = `{"choices":[{"delta":{"reasoning_content":"think"}}]}`
		usageOnly  = `{"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1}}`
	)

	sseBody := func(frames ...string) io.ReadCloser {
		var b strings.Builder
		for _, f := range frames {
			b.WriteString("data: ")
			b.WriteString(f)
			b.WriteString("\n\n")
		}
		return io.NopCloser(strings.NewReader(b.String()))
	}

	newReq := func() *http.Request {
		req, err := http.NewRequest(http.MethodPost, "http://example.test/v1/chat/completions", nil)
		if err != nil {
			t.Fatalf("new request: %v", err)
		}
		return req
	}

	t.Run("inactive gate keeps the default client", func(t *testing.T) {
		for _, n := range []int{0, -1} {
			if c := prefillClient(n); c != nil {
				t.Fatalf("prefillClient(%d) = %#v, want nil so NewRunner keeps its default client", n, c)
			}
		}
	})

	t.Run("slot cap blocks and cancellation surfaces the sentinel", func(t *testing.T) {
		gate := newPrefillGate(1)
		hold, err := gate.hold(context.Background())
		if err != nil {
			t.Fatalf("first hold: %v", err)
		}
		ctx, cancel := context.WithTimeout(context.Background(), 25*time.Millisecond)
		defer cancel()
		if _, err := gate.hold(ctx); !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("second hold while capped = %v, want context.DeadlineExceeded", err)
		}
		hold.release()
		hold.release() // double release must be a no-op
		if h2, err := gate.hold(context.Background()); err != nil {
			t.Fatalf("hold after release: %v", err)
		} else {
			h2.release()
		}
	})

	t.Run("cancelled acquire returns context.Canceled", func(t *testing.T) {
		gate := newPrefillGate(1)
		hold, err := gate.hold(context.Background())
		if err != nil {
			t.Fatalf("hold: %v", err)
		}
		defer hold.release()
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		if _, err := gate.hold(ctx); !errors.Is(err, context.Canceled) {
			t.Fatalf("canceled hold = %v, want context.Canceled", err)
		}
	})

	t.Run("first content frame fires the callback once with recorded TTFT", func(t *testing.T) {
		gate := newPrefillGate(1)
		hold, err := gate.hold(context.Background())
		if err != nil {
			t.Fatalf("hold: %v", err)
		}
		src := sseBody(roleOnly, contentTok, usageOnly)
		var calls int
		var gotTTFT int64
		var gotFrame string
		body := &firstTokenBody{
			rc:    src,
			r:     bufio.NewReader(src),
			hold:  hold,
			start: time.Now().Add(-50 * time.Millisecond),
			cb: func(ttftNS int64, firstFrame string) bool {
				calls++
				gotTTFT = ttftNS
				gotFrame = firstFrame
				return true
			},
		}
		var out bytes.Buffer
		if _, err := io.Copy(&out, body); err != nil {
			t.Fatalf("read body: %v", err)
		}
		want := "data: " + roleOnly + "\n\ndata: " + contentTok + "\n\ndata: " + usageOnly + "\n\n"
		if out.String() != want {
			t.Fatalf("body bytes altered by the wrapper:\n got %q\nwant %q", out.String(), want)
		}
		if calls != 1 {
			t.Fatalf("callback fired %d times, want exactly 1 (stop looking after true)", calls)
		}
		if gotTTFT <= 0 {
			t.Fatalf("ttftNS = %d, want > 0 (start was 50ms in the past)", gotTTFT)
		}
		if !strings.Contains(gotFrame, `"content":"tok"`) {
			t.Fatalf("first frame = %q, want the first content frame", gotFrame)
		}
	})

	t.Run("in-flight request holds the slot until its first content frame", func(t *testing.T) {
		gate := newPrefillGate(1)
		client := &http.Client{Transport: &gatedTransport{
			base: roundTripperFunc(func(r *http.Request) (*http.Response, error) {
				return &http.Response{StatusCode: http.StatusOK, Body: sseBody(roleOnly, contentTok)}, nil
			}),
			gate: gate,
		}}
		resp, err := client.Do(newReq())
		if err != nil {
			t.Fatalf("do: %v", err)
		}
		// Headers received, body unread: the request still holds its prefill slot.
		ctx, cancel := context.WithTimeout(context.Background(), 25*time.Millisecond)
		defer cancel()
		if _, err := gate.hold(ctx); !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("second hold while first in flight = %v, want context.DeadlineExceeded", err)
		}
		// Reading through the first content frame releases the slot.
		if _, err := io.ReadAll(resp.Body); err != nil {
			t.Fatalf("read body: %v", err)
		}
		if h, err := gate.hold(context.Background()); err != nil {
			t.Fatalf("slot not released after first content frame: %v", err)
		} else {
			h.release()
		}
	})

	t.Run("stream ending without content releases on EOF", func(t *testing.T) {
		gate := newPrefillGate(1)
		client := &http.Client{Transport: &gatedTransport{
			base: roundTripperFunc(func(r *http.Request) (*http.Response, error) {
				return &http.Response{StatusCode: http.StatusOK, Body: sseBody(roleOnly, "[DONE]")}, nil
			}),
			gate: gate,
		}}
		resp, err := client.Do(newReq())
		if err != nil {
			t.Fatalf("do: %v", err)
		}
		if _, err := io.ReadAll(resp.Body); err != nil {
			t.Fatalf("read body: %v", err)
		}
		// Return path: the slot must be free even though no token ever arrived.
		if h, err := gate.hold(context.Background()); err != nil {
			t.Fatalf("slot not released after content-free stream: %v", err)
		} else {
			h.release()
		}
	})

	t.Run("content-free stream fires the release callback exactly once", func(t *testing.T) {
		gate := newPrefillGate(1)
		hold, err := gate.hold(context.Background())
		if err != nil {
			t.Fatalf("hold: %v", err)
		}
		src := sseBody(roleOnly, "[DONE]")
		var calls int
		var gotTTFT int64
		var gotFrame string
		body := &firstTokenBody{
			rc:    src,
			r:     bufio.NewReader(src),
			hold:  hold,
			start: time.Now(),
			cb: func(ttftNS int64, firstFrame string) bool {
				calls++
				gotTTFT = ttftNS
				gotFrame = firstFrame
				return true
			},
		}
		var out bytes.Buffer
		if _, err := io.Copy(&out, body); err != nil {
			t.Fatalf("read body: %v", err)
		}
		want := "data: " + roleOnly + "\n\ndata: [DONE]\n\n"
		if out.String() != want {
			t.Fatalf("body bytes altered by the wrapper:\n got %q\nwant %q", out.String(), want)
		}
		// Repeated reads past EOF must not re-fire the callback.
		buf := make([]byte, 64)
		for i := 0; i < 3; i++ {
			n, err := body.Read(buf)
			if n != 0 || !errors.Is(err, io.EOF) {
				t.Fatalf("post-EOF read %d = (%d, %v), want (0, io.EOF)", i+1, n, err)
			}
		}
		if calls != 1 {
			t.Fatalf("callback fired %d times, want exactly 1 on the return path", calls)
		}
		if gotTTFT != 0 {
			t.Fatalf("return-path ttftNS = %d, want 0 (no token ever arrived)", gotTTFT)
		}
		if gotFrame != "" {
			t.Fatalf("return-path frame = %q, want empty", gotFrame)
		}
	})

	t.Run("callback returning false is re-offered the next content frame", func(t *testing.T) {
		gate := newPrefillGate(1)
		hold, err := gate.hold(context.Background())
		if err != nil {
			t.Fatalf("hold: %v", err)
		}
		src := sseBody(roleOnly, contentTok, contentTwo)
		var calls int
		var frames []string
		body := &firstTokenBody{
			rc:    src,
			r:     bufio.NewReader(src),
			hold:  hold,
			start: time.Now().Add(-10 * time.Millisecond),
			cb: func(ttftNS int64, firstFrame string) bool {
				calls++
				frames = append(frames, firstFrame)
				hold.release()    // transport-shaped cb: release per offer, Once makes it one
				return calls == 2 // reject the first content frame, accept the second
			},
		}
		if _, err := io.Copy(io.Discard, body); err != nil {
			t.Fatalf("read body: %v", err)
		}
		if calls != 2 {
			t.Fatalf("callback offered %d times, want exactly 2 (false re-offers the next content frame)", calls)
		}
		if !strings.Contains(frames[0], `"content":"tok"`) || !strings.Contains(frames[1], `"content":"tok2"`) {
			t.Fatalf("offered frames = %q, want the first then the second content frame", frames)
		}
		// Two offers, one release: the slot must be free again.
		if h, err := gate.hold(context.Background()); err != nil {
			t.Fatalf("slot not released after re-offer: %v", err)
		} else {
			h.release()
		}
	})

	t.Run("slot cap bounds concurrency and cascades release", func(t *testing.T) {
		gate := newPrefillGate(2)
		var inFlight, maxInFlight, baseCalls int32
		block := make(chan struct{})
		client := &http.Client{Transport: &gatedTransport{
			base: roundTripperFunc(func(r *http.Request) (*http.Response, error) {
				atomic.AddInt32(&baseCalls, 1)
				cur := atomic.AddInt32(&inFlight, 1)
				defer atomic.AddInt32(&inFlight, -1) // inFlight = concurrently inside base
				for {
					m := atomic.LoadInt32(&maxInFlight)
					if cur <= m || atomic.CompareAndSwapInt32(&maxInFlight, m, cur) {
						break
					}
				}
				<-block // hold the admitted wave in flight until the test releases it
				return &http.Response{StatusCode: http.StatusOK, Body: sseBody(roleOnly, contentTok)}, nil
			}),
			gate: gate,
		}}
		const total = 4
		done := make(chan error, total)
		for i := 0; i < total; i++ {
			go func() {
				resp, err := client.Do(newReq())
				if err != nil {
					done <- err
					return
				}
				_, err = io.ReadAll(resp.Body)
				resp.Body.Close()
				done <- err
			}()
		}
		// Generous deadline: admission is instant when the gate works; the
		// margin only guards against pathological scheduler starvation.
		deadline := time.Now().Add(10 * time.Second)
		for time.Now().Before(deadline) && atomic.LoadInt32(&inFlight) < 2 {
			time.Sleep(time.Millisecond)
		}
		if got := atomic.LoadInt32(&inFlight); got != 2 {
			t.Fatalf("dispatched through a 2-slot gate = %d, want exactly 2", got)
		}
		if got := atomic.LoadInt32(&baseCalls); got != 2 {
			t.Fatalf("base transport reached %d times while the gate was full, want 2", got)
		}
		close(block)
		for i := 0; i < total; i++ {
			select {
			case err := <-done:
				if err != nil {
					t.Fatalf("request %d failed: %v", i, err)
				}
			case <-time.After(5 * time.Second):
				t.Fatalf("request %d did not complete; release did not cascade", i)
			}
		}
		if m := atomic.LoadInt32(&maxInFlight); m > 2 {
			t.Fatalf("max concurrent dispatch = %d, want <= 2", m)
		}
	})

	t.Run("gate wait does not inflate reported TTFT", func(t *testing.T) {
		gate := newPrefillGate(1)
		blocker, err := gate.hold(context.Background())
		if err != nil {
			t.Fatalf("hold: %v", err)
		}
		go func() {
			time.Sleep(200 * time.Millisecond)
			blocker.release()
		}()
		client := &http.Client{Transport: &gatedTransport{
			base: roundTripperFunc(func(r *http.Request) (*http.Response, error) {
				return &http.Response{StatusCode: http.StatusOK, Body: sseBody(contentTok)}, nil
			}),
			gate: gate,
		}}
		start := time.Now()
		resp, err := client.Do(newReq())
		if err != nil {
			t.Fatalf("do: %v", err)
		}
		if waited := time.Since(start); waited < 150*time.Millisecond {
			t.Fatalf("request did not wait in the gate (waited %v); test setup broken", waited)
		}
		ftb, ok := resp.Body.(*firstTokenBody)
		if !ok {
			t.Fatalf("resp.Body is %T, want *firstTokenBody so the callback can be observed", resp.Body)
		}
		var gotTTFT int64
		var gotFrame string
		ftb.cb = func(ttftNS int64, firstFrame string) bool {
			gotTTFT = ttftNS
			gotFrame = firstFrame
			return true
		}
		if _, err := io.ReadAll(resp.Body); err != nil {
			t.Fatalf("read body: %v", err)
		}
		if !strings.Contains(gotFrame, `"content":"tok"`) {
			t.Fatalf("callback frame = %q (ttftNS=%d), want the content frame on the dispatch-measured path", gotFrame, gotTTFT)
		}
		// Upper bound is the pinned contract: the ~200ms gate wait must not
		// appear in the reported TTFT. The lower bound is deliberately not
		// asserted here — on hosts with a coarse monotonic clock two reads
		// microseconds apart can report 0 ns (witnessed on this build host);
		// the positive-TTFT property is pinned by the direct-body subtests,
		// which start 10-50ms in the past.
		if gotTTFT >= 150*time.Millisecond.Nanoseconds() {
			t.Fatalf("ttftNS = %d (~%dms) includes the ~200ms gate wait; want dispatch-measured (<150ms)",
				gotTTFT, gotTTFT/int64(time.Millisecond))
		}
	})

	t.Run("malformed JSON data frame neither fires the callback nor crashes", func(t *testing.T) {
		gate := newPrefillGate(1)
		hold, err := gate.hold(context.Background())
		if err != nil {
			t.Fatalf("hold: %v", err)
		}
		src := sseBody(`{not json`, contentTok, `{"choices":`)
		var calls int
		var gotFrame string
		body := &firstTokenBody{
			rc:    src,
			r:     bufio.NewReader(src),
			hold:  hold,
			start: time.Now().Add(-10 * time.Millisecond),
			cb: func(ttftNS int64, firstFrame string) bool {
				calls++
				gotFrame = firstFrame
				return true
			},
		}
		var out bytes.Buffer
		if _, err := io.Copy(&out, body); err != nil {
			t.Fatalf("read body: %v", err)
		}
		if calls != 1 {
			t.Fatalf("callback fired %d times, want exactly 1 (malformed frames skipped, valid frame fires)", calls)
		}
		if !strings.Contains(gotFrame, `"content":"tok"`) {
			t.Fatalf("callback frame = %q, want the valid content frame", gotFrame)
		}
		if !strings.Contains(out.String(), "{not json") {
			t.Fatalf("malformed frame dropped from the passthrough: %q", out.String())
		}
	})

	t.Run("reasoning_content-only frame counts as content", func(t *testing.T) {
		gate := newPrefillGate(1)
		hold, err := gate.hold(context.Background())
		if err != nil {
			t.Fatalf("hold: %v", err)
		}
		src := sseBody(roleOnly, reasonOnly, contentTok)
		var calls int
		var gotFrame string
		var gotTTFT int64
		body := &firstTokenBody{
			rc:    src,
			r:     bufio.NewReader(src),
			hold:  hold,
			start: time.Now().Add(-10 * time.Millisecond),
			cb: func(ttftNS int64, firstFrame string) bool {
				calls++
				gotFrame = firstFrame
				gotTTFT = ttftNS
				return true
			},
		}
		if _, err := io.Copy(io.Discard, body); err != nil {
			t.Fatalf("read body: %v", err)
		}
		if calls != 1 {
			t.Fatalf("callback fired %d times, want exactly 1", calls)
		}
		if !strings.Contains(gotFrame, `"reasoning_content":"think"`) {
			t.Fatalf("callback frame = %q, want the reasoning_content-only frame", gotFrame)
		}
		if gotTTFT <= 0 {
			t.Fatalf("ttftNS = %d, want > 0", gotTTFT)
		}
	})

	t.Run("data frame without the data-prefix passes through untouched", func(t *testing.T) {
		gate := newPrefillGate(1)
		hold, err := gate.hold(context.Background())
		if err != nil {
			t.Fatalf("hold: %v", err)
		}
		noSpace := "data:" + contentTok + "\n\n" // no space after the colon
		var b strings.Builder
		b.WriteString(noSpace)
		b.WriteString("data: " + contentTwo + "\n\n")
		src := io.NopCloser(strings.NewReader(b.String()))
		var calls int
		var gotFrame string
		body := &firstTokenBody{
			rc:    src,
			r:     bufio.NewReader(src),
			hold:  hold,
			start: time.Now().Add(-10 * time.Millisecond),
			cb: func(ttftNS int64, firstFrame string) bool {
				calls++
				gotFrame = firstFrame
				return true
			},
		}
		var out bytes.Buffer
		if _, err := io.Copy(&out, body); err != nil {
			t.Fatalf("read body: %v", err)
		}
		if out.String() != b.String() {
			t.Fatalf("body bytes altered:\n got %q\nwant %q", out.String(), b.String())
		}
		if calls != 1 {
			t.Fatalf("callback fired %d times, want exactly 1 (no-prefix frame must not fire)", calls)
		}
		if !strings.Contains(gotFrame, `"content":"tok2"`) {
			t.Fatalf("callback frame = %q, want the properly prefixed frame", gotFrame)
		}
	})

	t.Run("canceled waiter blocked in the gate returns Canceled and consumes no slot", func(t *testing.T) {
		gate := newPrefillGate(1)
		blocker, err := gate.hold(context.Background())
		if err != nil {
			t.Fatalf("hold: %v", err)
		}
		var reachedBase atomic.Bool
		client := &http.Client{Transport: &gatedTransport{
			base: roundTripperFunc(func(r *http.Request) (*http.Response, error) {
				reachedBase.Store(true)
				return nil, errors.New("unreachable")
			}),
			gate: gate,
		}}
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		errCh := make(chan error, 1)
		go func() {
			req, err := http.NewRequestWithContext(ctx, http.MethodPost, "http://example.test/v1/chat/completions", nil)
			if err != nil {
				errCh <- err
				return
			}
			_, err = client.Do(req)
			errCh <- err
		}()
		time.Sleep(50 * time.Millisecond) // let the waiter block in hold
		cancel()
		select {
		case err := <-errCh:
			if !errors.Is(err, context.Canceled) {
				t.Fatalf("blocked request err = %v, want context.Canceled", err)
			}
		case <-time.After(5 * time.Second):
			t.Fatal("canceled request did not return while blocked in the gate")
		}
		if reachedBase.Load() {
			t.Fatal("base transport reached despite a full gate")
		}
		// The canceled waiter must not have consumed a slot: after the blocker
		// releases, the gate must be at full capacity again.
		blocker.release()
		if h, err := gate.hold(context.Background()); err != nil {
			t.Fatalf("slot lost: a canceled waiter consumed a prefill slot: %v", err)
		} else {
			h.release()
		}
	})

	t.Run("nil response body releases the slot without panicking", func(t *testing.T) {
		gate := newPrefillGate(1)
		tr := &gatedTransport{
			base: roundTripperFunc(func(r *http.Request) (*http.Response, error) {
				return &http.Response{StatusCode: http.StatusOK, Body: nil}, nil
			}),
			gate: gate,
		}
		resp, err := tr.RoundTrip(newReq())
		if err != nil {
			t.Fatalf("roundtrip: %v", err)
		}
		if resp.Body != nil {
			t.Fatalf("resp.Body = %#v, want nil passthrough", resp.Body)
		}
		if h, err := gate.hold(context.Background()); err != nil {
			t.Fatalf("slot not released for a nil body: %v", err)
		} else {
			h.release()
		}
	})

	t.Run("partial-line reads preserve bytes and fire exactly once", func(t *testing.T) {
		gate := newPrefillGate(1)
		hold, err := gate.hold(context.Background())
		if err != nil {
			t.Fatalf("hold: %v", err)
		}
		src := sseBody(roleOnly, contentTok)
		var calls int
		body := &firstTokenBody{
			rc:    src,
			r:     bufio.NewReader(src),
			hold:  hold,
			start: time.Now().Add(-10 * time.Millisecond),
			cb: func(ttftNS int64, firstFrame string) bool {
				calls++
				return true
			},
		}
		var out bytes.Buffer
		buf := make([]byte, 7) // smaller than any SSE line: exercises the pend path
		for {
			n, err := body.Read(buf)
			out.Write(buf[:n])
			if err == io.EOF {
				break
			}
			if err != nil {
				t.Fatalf("read: %v", err)
			}
		}
		want := "data: " + roleOnly + "\n\ndata: " + contentTok + "\n\n"
		if out.String() != want {
			t.Fatalf("reassembled body altered:\n got %q\nwant %q", out.String(), want)
		}
		if calls != 1 {
			t.Fatalf("callback fired %d times, want exactly 1", calls)
		}
	})

	t.Run("stream whose first frame is content fires immediately and once", func(t *testing.T) {
		gate := newPrefillGate(1)
		hold, err := gate.hold(context.Background())
		if err != nil {
			t.Fatalf("hold: %v", err)
		}
		src := sseBody(contentTok, contentTwo, "[DONE]")
		var calls int
		var gotFrame string
		var gotTTFT int64
		body := &firstTokenBody{
			rc:    src,
			r:     bufio.NewReader(src),
			hold:  hold,
			start: time.Now().Add(-10 * time.Millisecond),
			cb: func(ttftNS int64, firstFrame string) bool {
				calls++
				gotFrame = firstFrame
				gotTTFT = ttftNS
				return true
			},
		}
		if _, err := io.Copy(io.Discard, body); err != nil {
			t.Fatalf("read body: %v", err)
		}
		if calls != 1 {
			t.Fatalf("callback fired %d times, want exactly 1", calls)
		}
		if !strings.Contains(gotFrame, `"content":"tok"`) {
			t.Fatalf("callback frame = %q, want the FIRST content frame", gotFrame)
		}
		if gotTTFT <= 0 {
			t.Fatalf("ttftNS = %d, want > 0", gotTTFT)
		}
	})

	t.Run("transport error releases the slot", func(t *testing.T) {
		gate := newPrefillGate(1)
		client := &http.Client{Transport: &gatedTransport{
			base: roundTripperFunc(func(r *http.Request) (*http.Response, error) {
				return nil, errors.New("boom")
			}),
			gate: gate,
		}}
		if _, err := client.Do(newReq()); err == nil {
			t.Fatal("expected the transport error to surface")
		}
		if h, err := gate.hold(context.Background()); err != nil {
			t.Fatalf("slot not released after transport error: %v", err)
		} else {
			h.release()
		}
	})

	t.Run("runAgentX accepts --prefill-slots", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		code := runAgentX(&stdout, &stderr, []string{"--selfcheck", "--agents=1", "--turns=1", "--prefill-slots=2"})
		if code != 0 {
			t.Fatalf("expected code 0, got %d. stderr: %s", code, stderr.String())
		}
		if !strings.Contains(stdout.String(), "Prefill Gate   : 2 slots") {
			t.Fatalf("expected prefill gate summary line, got: %s", stdout.String())
		}
		if !strings.Contains(stdout.String(), "Status         : SIMULATED") {
			t.Fatalf("--prefill-slots changed the mock arm status, got: %s", stdout.String())
		}
		stdout.Reset()
		stderr.Reset()
		if code := runAgentX(&stdout, &stderr, []string{"--selfcheck", "--prefill-slots=-1"}); code != 2 {
			t.Fatalf("negative --prefill-slots: code = %d, want 2; stderr: %s", code, stderr.String())
		}
	})
}

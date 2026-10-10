package gateway

// responses_stream_live_test.go witnesses the #12765 live passthrough on
// POST /v1/responses: with a planner that can stream the wire live, a stream:true
// client receives the Responses SSE events INCREMENTALLY — content deltas arrive
// while the upstream planner is still running, before CompleteStream returns — and
// a planner the W1 gate refuses falls back to the untouched buffered synthesis
// (writeResponsesStream) with a byte-identical event stream for the same request.

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// scriptedLiveResponsesPlanner is a StreamingPlanner whose CompleteStream
// fragments arrive on a scripted schedule, so the test can prove deltas reach the
// client BEFORE the planner returns. It also carries W1's gate so the fallback
// witness can refuse the live arm explicitly.
type scriptedLiveResponsesPlanner struct {
	fragments []string      // content fragments sinked in order
	fragGap   time.Duration // sleep BETWEEN fragments (after every one but the last)
	comp      *agent.Completion

	supports   bool // StreamingSupported()
	gateAllows bool // ResponsesWireStreamsLive verdict

	returnedAt time.Time // captured when CompleteStream is about to return

	sawFragments                 []string
	optsMu                       sync.Mutex
	streamParams, bufferedParams []agent.SampleParams
}

func (p *scriptedLiveResponsesPlanner) Complete(_ context.Context, _ []agent.Message, _ []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	p.captureParams(false, opts)
	return p.comp, nil
}

func (*scriptedLiveResponsesPlanner) Model() string { return "scripted-responses-live" }

// Provider lets plannerWireProvider resolve the wire family for the W1 gate;
// the live arm is only reachable for providers the gate recognizes (ProviderOpenAI).
func (*scriptedLiveResponsesPlanner) Provider() agent.Provider { return agent.ProviderOpenAI }

func (p *scriptedLiveResponsesPlanner) StreamingSupported() bool { return p.supports }

func (p *scriptedLiveResponsesPlanner) ResponsesWireStreamsLive(agent.Provider, bool) bool {
	return p.gateAllows
}

func (p *scriptedLiveResponsesPlanner) CompleteStream(_ context.Context, sink agent.StreamSink, _ []agent.Message, _ []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	p.captureParams(true, opts)
	defer func() { p.returnedAt = time.Now() }()
	for i, frag := range p.fragments {
		p.sawFragments = append(p.sawFragments, frag)
		if err := sink(frag); err != nil {
			return nil, err
		}
		if p.fragGap > 0 && i < len(p.fragments)-1 {
			time.Sleep(p.fragGap)
		}
	}
	return p.comp, nil
}

func (p *scriptedLiveResponsesPlanner) captureParams(stream bool, opts []agent.SampleOpt) {
	var params agent.SampleParams
	for _, opt := range opts {
		opt(&params)
	}
	p.optsMu.Lock()
	defer p.optsMu.Unlock()
	if stream {
		p.streamParams = append(p.streamParams, params)
	} else {
		p.bufferedParams = append(p.bufferedParams, params)
	}
}

func (p *scriptedLiveResponsesPlanner) assertPerTokenRoute(t *testing.T, stream bool) {
	t.Helper()
	p.optsMu.Lock()
	defer p.optsMu.Unlock()
	if stream {
		if len(p.streamParams) != 1 || len(p.bufferedParams) != 0 || p.streamParams[0].PerTokenStream == nil || !*p.streamParams[0].PerTokenStream {
			t.Fatalf("live sampling missing per-token override: stream=%+v buffered=%+v", p.streamParams, p.bufferedParams)
		}
	} else {
		if len(p.streamParams) != 0 || len(p.bufferedParams) != 1 || p.bufferedParams[0].PerTokenStream != nil {
			t.Fatalf("buffered sampling changed: stream=%+v buffered=%+v", p.streamParams, p.bufferedParams)
		}
	}
}

// timedSSEEvent is one parsed Responses SSE frame with its client-side arrival time.
type timedSSEEvent struct {
	name string
	data string
	at   time.Time
}

// readTimedResponsesStream POSTs a stream:true /v1/responses request to base and
// parses the response body into timed SSE events (arrival order = wire order).
func readTimedResponsesStream(t *testing.T, base, body string) ([]timedSSEEvent, *http.Response) {
	t.Helper()
	resp, err := http.Post(base+"/v1/responses", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		t.Fatalf("status = %d, want 200: %s", resp.StatusCode, raw)
	}
	if ct := resp.Header.Get("Content-Type"); !strings.Contains(ct, "text/event-stream") {
		t.Fatalf("content-type = %q, want text/event-stream", ct)
	}
	var events []timedSSEEvent
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		line := sc.Text()
		switch {
		case strings.HasPrefix(line, "event:"):
			events = append(events, timedSSEEvent{name: strings.TrimSpace(strings.TrimPrefix(line, "event:")), at: time.Now()})
		case strings.HasPrefix(line, "data:"):
			if len(events) > 0 {
				events[len(events)-1].data += strings.TrimSpace(strings.TrimPrefix(line, "data:"))
			}
		}
	}
	resp.Body.Close()
	return events, resp
}

// Estimate only; this test has not been timed.
// fak-test:runtime medium est=1s lane=default
func TestResponsesLiveStreamsDeltasBeforeUpstreamEOF(t *testing.T) {
	srv := newTestServer(t)
	planner := &scriptedLiveResponsesPlanner{
		fragments:  []string{"Hel", "lo"},
		fragGap:    150 * time.Millisecond,
		supports:   true,
		gateAllows: true,
		comp: &agent.Completion{
			Message:      agent.Message{Role: agent.RoleAssistant, Content: "Hello"},
			FinishReason: "stop",
		},
	}
	srv.planner = planner
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	events, _ := readTimedResponsesStream(t, ts.URL, `{"model":"test-model","input":"hi","stream":true}`)
	planner.assertPerTokenRoute(t, true)

	var names []string
	for _, ev := range events {
		names = append(names, ev.name)
	}
	want := []string{
		"response.created",
		"response.output_item.added",
		"response.output_text.delta", // "Hel"
		"response.output_text.delta", // "lo"
		"response.output_text.done",
		"response.output_item.done",
		"response.completed",
	}
	if len(names) != len(want) {
		t.Fatalf("event sequence = %v, want %v", names, want)
	}
	for i := range want {
		if names[i] != want[i] {
			t.Fatalf("event sequence = %v, want %v (mismatch at %d)", names, want, i)
		}
	}

	// LIVE-wALITY witness: the first content delta reached the client's parser while
	// the planner was still streaming — strictly before CompleteStream returned. The
	// 150ms inter-fragment sleep makes the margin far larger than any scheduling
	// jitter; a buffered-only path emits its single fat delta only after the turn.
	returnedAt := planner.returnedAt
	if returnedAt.IsZero() {
		t.Fatal("planner return time not captured — CompleteStream never ran")
	}
	if !events[2].at.Before(returnedAt) {
		t.Fatalf("first delta arrived at %v, want strictly before CompleteStream returned at %v (the live passthrough streamed late)", events[2].at, returnedAt)
	}
	if len(planner.sawFragments) != 2 || planner.sawFragments[0] != "Hel" || planner.sawFragments[1] != "lo" {
		t.Fatalf("sink fragments = %v, want [Hel lo]", planner.sawFragments)
	}

	// Delta payloads: the two fragments streamed verbatim, in order.
	var deltas []string
	for _, ev := range events {
		if ev.name != "response.output_text.delta" {
			continue
		}
		var payload struct {
			Delta string `json:"delta"`
		}
		if err := json.Unmarshal([]byte(ev.data), &payload); err != nil {
			t.Fatalf("decode delta %s: %v", ev.data, err)
		}
		deltas = append(deltas, payload.Delta)
	}
	if len(deltas) != 2 || deltas[0] != "Hel" || deltas[1] != "lo" {
		t.Fatalf("deltas = %q, want [Hel lo]", deltas)
	}

	// response.created: in_progress, empty output/text, zero usage, output_index 0 item.
	created := decodeResponsesLiveEvent(t, events[0].data)
	if created.Status != "in_progress" || created.OutputText != "" || created.ID == "" {
		t.Fatalf("response.created = %+v, want in_progress + non-empty id + empty output_text", created)
	}
	if len(created.Output) != 0 || created.Usage.TotalTokens != 0 {
		t.Fatalf("response.created output/usage = %+v/%+v, want empty/zero", created.Output, created.Usage)
	}
	var addedEnvelope struct {
		OutputIndex int                 `json:"output_index"`
		Item        responsesOutputItem `json:"item"`
	}
	if err := json.Unmarshal([]byte(events[1].data), &addedEnvelope); err != nil {
		t.Fatalf("decode added: %v", err)
	}
	if addedEnvelope.OutputIndex != 0 || addedEnvelope.Item.Type != "message" || addedEnvelope.Item.ID != "msg_fak_live_0" {
		t.Fatalf("output_item.added = %+v, want output_index 0 message msg_fak_live_0", addedEnvelope)
	}

	// response.completed: full text, zero-OK usage, terminal event, model echoed.
	series := events[len(events)-1]
	completed := decodeResponsesLiveEvent(t, series.data)
	if completed.Status != "completed" {
		t.Fatalf("response.completed status = %q, want completed", completed.Status)
	}
	if completed.OutputText != "Hello" {
		t.Fatalf("response.completed output_text = %q, want Hello", completed.OutputText)
	}
	if completed.Usage.TotalTokens != 0 || completed.Usage.InputTokens != 0 || completed.Usage.OutputTokens != 0 {
		t.Fatalf("response.completed usage = %+v, want zero-OK", completed.Usage)
	}
	if completed.Model != "test-model" {
		t.Fatalf("response.completed model = %q, want test-model", completed.Model)
	}
	msgItem, ok := findResponsesLiveItem(completed.Output, "message")
	if !ok || len(msgItem.Content) != 1 || msgItem.Content[0].Text != "Hello" {
		t.Fatalf("response.completed message item = %+v, want one output_text part Hello", msgItem)
	}

	// Sequence discipline: 0,1,2,... one per event, in arrival order.
	for i, ev := range events {
		var envelope struct {
			SequenceNumber int `json:"sequence_number"`
		}
		if err := json.Unmarshal([]byte(ev.data), &envelope); err != nil {
			t.Fatalf("sequence_number at event %d (%s): %v: %s", i, ev.name, err, ev.data)
		}
		if envelope.SequenceNumber != i {
			t.Fatalf("sequence_number at event %d = %d, want %d", i, envelope.SequenceNumber, i)
		}
	}
}

// Estimate only; this test has not been timed.
// fak-test:runtime medium est=1s lane=default
func TestResponsesLiveFallsBackWhenUpstreamCannotStream(t *testing.T) {
	comp := &agent.Completion{
		Message:      agent.Message{Role: agent.RoleAssistant, Content: "Hello"},
		FinishReason: "stop",
	}

	// Fallback server: the planner CAN stream (StreamingSupported) but W1's gate
	// REFUSES the Responses live arm — streamResponsesLive must return false having
	// written nothing, and the request must fall through to writeResponsesStream.
	fallback := newTestServer(t)
	fallbackPlanner := &scriptedLiveResponsesPlanner{supports: true, gateAllows: false, comp: comp}
	fallback.planner = fallbackPlanner
	fallbackTS := httptest.NewServer(fallback.Handler())
	defer fallbackTS.Close()

	// Control server: a plain Complete-only planner (never a StreamingPlanner) on the
	// identical request — the buffered route this wire has always taken.
	control := newTestServer(t)
	control.planner = stubPlanner{comp: comp}
	controlTS := httptest.NewServer(control.Handler())
	defer controlTS.Close()

	const body = `{"model":"test-model","input":"hi","stream":true}`
	fallbackEvents, _ := readTimedResponsesStream(t, fallbackTS.URL, body)
	fallbackPlanner.assertPerTokenRoute(t, false)
	controlEvents, _ := readTimedResponsesStream(t, controlTS.URL, body)

	render := func(t *testing.T, events []timedSSEEvent) string {
		t.Helper()
		var b strings.Builder
		for _, ev := range events {
			b.WriteString("event: " + ev.name + "\n")
			b.WriteString("data: " + ev.data + "\n\n")
		}
		return b.String()
	}

	// Byte-identity witness. The ONLY normalization is the two values that are
	// inherently nondeterministic between two separate requests — the random
	// response id and the wall-clock created_at. Everything else (event names,
	// ordering, envelope field order, deltas, usage) must be byte-identical.
	var idRe = regexp.MustCompile(`"id":"resp_fak_[0-9a-f]+"`)
	var createdRe = regexp.MustCompile(`"created_at":[0-9]+`)
	mask := func(t *testing.T, raw string) string {
		t.Helper()
		masked := idRe.ReplaceAllString(raw, `"id":"resp_fak_NORMALIZED"`)
		return createdRe.ReplaceAllString(masked, `"created_at":0`)
	}

	fallbackBody := mask(t, render(t, fallbackEvents))
	controlBody := mask(t, render(t, controlEvents))
	if fallbackBody != controlBody {
		t.Fatalf("fallback SSE differs from the buffered synthesis:\n--fallback--\n%s\n--buffered--\n%s", fallbackBody, controlBody)
	}

	// Sanity: the fallback stream still carries the full synthesized shape.
	var names []string
	for _, ev := range fallbackEvents {
		names = append(names, ev.name)
	}
	want := []string{
		"response.created",
		"response.output_item.added",
		"response.output_text.delta",
		"response.output_text.done",
		"response.output_item.done",
		"response.completed",
	}
	if strings.Join(names, ",") != strings.Join(want, ",") {
		t.Fatalf("fallback event sequence = %v, want %v", names, want)
	}
	last := decodeResponsesLiveEvent(t, fallbackEvents[len(fallbackEvents)-1].data)
	if last.Status != "completed" || last.OutputText != "Hello" {
		t.Fatalf("fallback response.completed = %+v, want completed/Hello", last)
	}
	if strings.Contains(fallbackBody, "[DONE]") {
		t.Fatal("fallback stream carried a [DONE] sentinel, want none")
	}
}

// decodeResponsesLiveEvent decodes one SSE `data:` payload's embedded response
// object (every Responses event wraps the full response in `response`).
func decodeResponsesLiveEvent(t *testing.T, data string) responsesResponse {
	t.Helper()
	var envelope struct {
		Response responsesResponse `json:"response"`
	}
	if err := json.Unmarshal([]byte(data), &envelope); err != nil {
		t.Fatalf("decode event data: %v: %s", err, data)
	}
	return envelope.Response
}

// findResponsesLiveItem returns the first output item of the given type.
func findResponsesLiveItem(items []responsesOutputItem, typ string) (responsesOutputItem, bool) {
	for _, it := range items {
		if it.Type == typ {
			return it, true
		}
	}
	return responsesOutputItem{}, false
}

// Estimate only; this test has not been timed.
// fak-test:runtime medium est=1s lane=default
func TestResponsesBufferedLeavesPerTokenOverrideAbsent(t *testing.T) {
	srv := newTestServer(t)
	planner := &scriptedLiveResponsesPlanner{
		supports: true, gateAllows: true,
		comp: &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: "Hello"}, FinishReason: "stop"},
	}
	srv.planner = planner
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	resp, err := http.Post(ts.URL+"/v1/responses", "application/json", strings.NewReader(`{"model":"test-model","input":"hi","stream":false}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var result responsesResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusOK || result.Status != "completed" || result.OutputText != "Hello" {
		t.Fatalf("buffered response status=%d result=%+v", resp.StatusCode, result)
	}
	planner.assertPerTokenRoute(t, false)
}

// No embedded ResponseRecorder: promoted WriteString would bypass Write faults.
type responsesFailWriter struct {
	header                          http.Header
	status, writes, flushes, failAt int
	short                           bool
	cause                           error
	body                            strings.Builder
}

func (w *responsesFailWriter) Header() http.Header {
	if w.header == nil {
		w.header = make(http.Header)
	}
	return w.header
}
func (w *responsesFailWriter) WriteHeader(status int) { w.status = status }
func (w *responsesFailWriter) Write(p []byte) (int, error) {
	w.writes++
	if w.writes == w.failAt {
		if w.short {
			return len(p) - 1, nil
		}
		return 0, w.cause
	}
	return w.body.Write(p)
}
func (w *responsesFailWriter) Flush() { w.flushes++ }

type responsesWriteFailurePlanner struct {
	upstreamError                         error
	fragments                             []string
	content                               string
	swallow                               bool
	streamCalls, bufferedCalls, sinkCalls int
	sinkError                             error
	contextError                          error
}

func (*responsesWriteFailurePlanner) Model() string            { return "test-model" }
func (*responsesWriteFailurePlanner) StreamingSupported() bool { return true }
func (p *responsesWriteFailurePlanner) Complete(context.Context, []agent.Message, []agent.ToolDef, ...agent.SampleOpt) (*agent.Completion, error) {
	p.bufferedCalls++
	return nil, errors.New("unexpected buffered fallback")
}
func (p *responsesWriteFailurePlanner) CompleteStream(ctx context.Context, sink agent.StreamSink, _ []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	p.streamCalls++
	for _, fragment := range p.fragments {
		p.sinkCalls++
		if err := sink(fragment); err != nil {
			if p.sinkError == nil {
				p.sinkError, p.contextError = err, ctx.Err()
			}
			if !p.swallow {
				return nil, err
			}
		}
	}
	if p.upstreamError != nil {
		return nil, p.upstreamError
	}
	return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: p.content}, FinishReason: "stop", Usage: agent.Usage{CompletionTokens: 2}}, nil
}

// fak-test:runtime fast est=20ms lane=default
// Unmeasured estimate. Direct helper controls; no socket or native planner.
func TestResponsesSSEHelperPropagatesWriteFailures(t *testing.T) {
	cause := errors.New("writer failed")
	for _, short := range []bool{false, true} {
		for _, at := range []int{1, 2} {
			w := &responsesFailWriter{cause: cause, failAt: at, short: short}
			err := writeSSEEvent(w, "sample", map[string]int{"n": 1})
			want := cause
			if short {
				want = io.ErrShortWrite
			}
			if err != want || w.writes != at || w.flushes != 0 {
				t.Fatalf("write error lost/retried/flushed: err=%v writes=%d flushes=%d", err, w.writes, w.flushes)
			}
		}
	}
	w := &responsesFailWriter{}
	if err := writeSSEEvent(w, "sample", map[string]int{"n": 1}); err != nil || w.body.String() != "event: sample\ndata: {\"n\":1}\n\n" || w.writes != 2 || w.flushes != 1 {
		t.Fatalf("clean wire changed: %q err=%v", w.body.String(), err)
	}
	w = &responsesFailWriter{}
	if err := writeSSEEvent(w, "sample", make(chan int)); err == nil || w.writes != 0 || w.flushes != 0 {
		t.Fatal("marshal failure attempted output")
	}
}

// fak-test:runtime medium est=2s lane=default
// Unmeasured estimate. Real public handler and a live request context, scripted
// planner, failing writer. No physical generation or latency claim.
func TestResponsesLiveWriterFailureStopsSelectedTurn(t *testing.T) {
	t.Setenv("FAK_EP_FANOUT_ADDRS", "")
	for _, short := range []bool{false, true} {
		for failAt := 0; failAt <= 14; failAt++ {
			t.Run(fmt.Sprintf("short=%t/write=%d", short, failAt), func(t *testing.T) {
				srv := newTestServer(t)
				p := &responsesWriteFailurePlanner{fragments: []string{"Hel", "lo"}, content: "Hello"}
				srv.planner = p
				cause := errors.New("client write failed")
				w := &responsesFailWriter{cause: cause, failAt: failAt, short: short}
				ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
				defer cancel()
				req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{"model":"test-model","input":"hi","stream":true}`)).WithContext(ctx)
				req.Header.Set("Content-Type", "application/json")
				srv.Handler().ServeHTTP(w, req)
				if ctx.Err() != nil || p.streamCalls != 1 || p.bufferedCalls != 0 {
					t.Fatalf("wrong selection or canceled request: stream=%d buffered=%d ctx=%v", p.streamCalls, p.bufferedCalls, ctx.Err())
				}
				wantWrites := failAt
				if failAt == 0 {
					wantWrites = 14
				}
				if w.writes != wantWrites {
					t.Fatalf("writer retried or frame count changed: got=%d want=%d body=%s", w.writes, wantWrites, w.body.String())
				}
				if failAt > 0 && failAt <= 8 {
					want := cause
					if short {
						want = io.ErrShortWrite
					}
					wantCalls := 1
					if failAt > 6 {
						wantCalls = 2
					}
					if p.sinkError != want || p.contextError != nil || p.sinkCalls != wantCalls {
						t.Fatalf("sink cause/early abort lost: err=%v ctx=%v calls=%d", p.sinkError, p.contextError, p.sinkCalls)
					}
				} else if p.sinkError != nil {
					t.Fatalf("tail failure misreported as planner sink failure: %v", p.sinkError)
				}
				store := srv.responsesContinuationState()
				store.mu.Lock()
				saved := len(store.entries)
				store.mu.Unlock()
				wantSaved := 0
				if failAt == 0 || failAt >= 13 {
					wantSaved = 1
				}
				if saved != wantSaved {
					t.Fatalf("persist-before-completed contract: saved=%d want=%d", saved, wantSaved)
				}
				if failAt == 0 && (!strings.Contains(w.body.String(), "response.completed") || strings.Contains(w.body.String(), "response.failed")) {
					t.Fatal("clean success changed")
				}
			})
		}
	}
}

// fak-test:runtime medium est=1s lane=default
// Unmeasured estimate. Broken planner and post-generation content controls.
func TestResponsesWriterFailureCannotBecomeSuccess(t *testing.T) {
	t.Setenv("FAK_EP_FANOUT_ADDRS", "")
	for _, tc := range []struct {
		name      string
		fragments []string
		swallow   bool
		failAt    int
	}{
		{"planner-swallows", []string{"Hel", "lo"}, true, 1},
		{"final-remainder", []string{"Hel"}, false, 7},
		{"empty-callback-final-start", nil, false, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			srv := newTestServer(t)
			p := &responsesWriteFailurePlanner{fragments: tc.fragments, content: "Hello", swallow: tc.swallow}
			srv.planner = p
			cause := errors.New("client write failed")
			w := &responsesFailWriter{cause: cause, failAt: tc.failAt}
			req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{"model":"test-model","input":"hi","stream":true}`))
			req.Header.Set("Content-Type", "application/json")
			srv.Handler().ServeHTTP(w, req)
			store := srv.responsesContinuationState()
			store.mu.Lock()
			saved := len(store.entries)
			store.mu.Unlock()
			if w.writes != tc.failAt || p.streamCalls != 1 || p.bufferedCalls != 0 || saved != 0 || strings.Contains(w.body.String(), "response.completed") {
				t.Fatalf("failure became success/retry: writes=%d calls=%d/%d saved=%d", w.writes, p.streamCalls, p.bufferedCalls, saved)
			}
			if tc.swallow && p.sinkError != cause {
				t.Fatal("swallowed sink error identity lost")
			}
		})
	}
}

// fak-test:runtime medium est=1s lane=default
// Unmeasured estimate. Upstream failure keeps its existing best-effort error tail.
func TestResponsesUpstreamFailureTailDoesNotRetryWriter(t *testing.T) {
	t.Setenv("FAK_EP_FANOUT_ADDRS", "")
	for _, failAt := range []int{0, 7, 8} {
		t.Run(fmt.Sprint(failAt), func(t *testing.T) {
			srv := newTestServer(t)
			upstream := errors.New("original upstream failure")
			p := &responsesWriteFailurePlanner{fragments: []string{"Hel"}, upstreamError: upstream}
			srv.planner = p
			w := &responsesFailWriter{cause: errors.New("terminal writer failure"), failAt: failAt}
			req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{"model":"test-model","input":"hi","stream":true}`))
			req.Header.Set("Content-Type", "application/json")
			srv.Handler().ServeHTTP(w, req)
			want := failAt
			if want == 0 {
				want = 8
			}
			store := srv.responsesContinuationState()
			store.mu.Lock()
			saved := len(store.entries)
			store.mu.Unlock()
			if w.writes != want || p.streamCalls != 1 || p.bufferedCalls != 0 || p.sinkError != nil || saved != 0 || strings.Contains(w.body.String(), "response.completed") {
				t.Fatalf("terminal failure retried or completed: writes=%d sink=%v saved=%d", w.writes, p.sinkError, saved)
			}
			if failAt == 0 && !strings.Contains(w.body.String(), "response.failed") {
				t.Fatal("clean upstream-error terminal missing")
			}
		})
	}
}

// responsesStartWriter snapshots headers at the first physical write, not from
// the mutable Header map after the handler returns. It also injects start faults.
type responsesStartWriter struct {
	recorder       *httptest.ResponseRecorder
	firstHeader    http.Header
	writes, failAt int
	cause          error
}

func (w *responsesStartWriter) Header() http.Header    { return w.recorder.Header() }
func (w *responsesStartWriter) WriteHeader(status int) { w.recorder.WriteHeader(status) }
func (w *responsesStartWriter) Write(b []byte) (int, error) {
	if w.writes == 0 {
		w.firstHeader = w.Header().Clone()
	}
	w.writes++
	if w.writes == w.failAt {
		return 0, w.cause
	}
	return w.recorder.Write(b)
}
func (w *responsesStartWriter) Flush() { w.recorder.Flush() }

// fak-test:runtime medium est=1s lane=default
// Unmeasured estimate. Actual handler with admitted tools; no native planner.
func TestResponsesLiveCreatedPrecedesToolOnlyItems(t *testing.T) {
	t.Setenv("FAK_EP_FANOUT_ADDRS", "")
	for _, tc := range []struct {
		name, content string
		tools         bool
	}{
		{"tool-only", "", true}, {"mixed", "Hello", true}, {"prose", "Hello", false}, {"empty", "", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			srv := newTestServer(t)
			comp := &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: tc.content}, FinishReason: "stop", Usage: agent.Usage{PromptTokens: 2, CompletionTokens: 3, TotalTokens: 5}}
			if tc.tools {
				comp.Message.ToolCalls = []agent.ToolCall{
					{ID: "call_a", Type: "function", Function: agent.Func{Name: "allow_a", Arguments: `{"x":1}`}},
					{ID: "call_b", Type: "function", Function: agent.Func{Name: "allow_b", Arguments: `{"x":2}`}},
				}
				comp.FinishReason = "tool_calls"
			}
			p := &scriptedLiveResponsesPlanner{supports: true, gateAllows: true, comp: comp}
			if tc.content != "" {
				p.fragments = []string{tc.content}
			}
			srv.planner = p
			w := &responsesStartWriter{recorder: httptest.NewRecorder()}
			req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{"model":"test-model","input":"hi","stream":true,"tools":[{"type":"function","name":"allow_a"},{"type":"function","name":"allow_b"}]}`))
			req.Header.Set("Content-Type", "application/json")
			srv.Handler().ServeHTTP(w, req)
			p.assertPerTokenRoute(t, true)
			if w.firstHeader.Get("Content-Type") != "text/event-stream" || w.firstHeader.Get("Cache-Control") != "no-cache" || w.firstHeader.Get("X-Accel-Buffering") != "no" {
				t.Fatalf("first write headers=%v", w.firstHeader)
			}
			if w.recorder.Result().Header.Get("Content-Type") != "text/event-stream" {
				t.Fatal("committed response lost SSE headers")
			}
			var names []string
			var final responsesResponse
			toolAdded, toolDone, argsDone := 0, 0, 0
			for _, frame := range strings.Split(w.recorder.Body.String(), "\n\n") {
				if strings.TrimSpace(frame) == "" {
					continue
				}
				lines := strings.Split(frame, "\n")
				if len(lines) != 2 || !strings.HasPrefix(lines[0], "event: ") || !strings.HasPrefix(lines[1], "data: ") {
					t.Fatalf("malformed frame=%q", frame)
				}
				name := strings.TrimPrefix(lines[0], "event: ")
				var ev struct {
					Type      string              `json:"type"`
					Sequence  int                 `json:"sequence_number"`
					Index     int                 `json:"output_index"`
					Item      responsesOutputItem `json:"item"`
					Response  responsesResponse   `json:"response"`
					CallID    string              `json:"call_id"`
					Arguments string              `json:"arguments"`
				}
				if err := json.Unmarshal([]byte(strings.TrimPrefix(lines[1], "data: ")), &ev); err != nil {
					t.Fatal(err)
				}
				if ev.Type != name || ev.Sequence != len(names) {
					t.Fatalf("event=%s type=%s seq=%d want=%d", name, ev.Type, ev.Sequence, len(names))
				}
				names = append(names, name)
				if name == "response.created" && (len(names) != 1 || ev.Response.Status != "in_progress" || len(ev.Response.Output) != 0) {
					t.Fatal("created was late or already populated")
				}
				if ev.Item.Type == "function_call" {
					base := 0
					if tc.content != "" {
						base = 1
					}
					ordinal := toolAdded
					if name == "response.output_item.done" {
						ordinal = toolDone
					}
					if ev.Index != base+ordinal || ev.Item.CallID != fmt.Sprintf("call_%c", 'a'+ordinal) || ev.Item.Arguments != fmt.Sprintf(`{"x":%d}`, ordinal+1) {
						t.Fatalf("tool item/index changed: %+v", ev)
					}
					if name == "response.output_item.added" {
						toolAdded++
					}
					if name == "response.output_item.done" {
						toolDone++
					}
				}
				if name == "response.function_call_arguments.done" {
					if ev.CallID != fmt.Sprintf("call_%c", 'a'+argsDone) || ev.Arguments != fmt.Sprintf(`{"x":%d}`, argsDone+1) {
						t.Fatalf("arguments changed: %+v", ev)
					}
					argsDone++
				}
				if name == "response.completed" {
					final = ev.Response
				}
			}
			if len(names) < 2 || names[0] != "response.created" || names[len(names)-1] != "response.completed" {
				t.Fatalf("event order=%v", names)
			}
			wantTools := 0
			if tc.tools {
				wantTools = 2
			}
			wantItems := wantTools
			if tc.content != "" {
				wantItems++
			}
			if toolAdded != wantTools || toolDone != wantTools || argsDone != wantTools || len(final.Output) != wantItems || final.OutputText != tc.content || final.Usage.TotalTokens != 5 {
				t.Fatalf("tools=%d/%d/%d final=%+v", toolAdded, toolDone, argsDone, final)
			}
			store := srv.responsesContinuationState()
			store.mu.Lock()
			saved := len(store.entries)
			store.mu.Unlock()
			if saved != 1 {
				t.Fatalf("saved=%d", saved)
			}
		})
	}
}

// fak-test:runtime medium est=1s lane=default
// Unmeasured estimate. Failure of either created-frame write cannot reveal tools.
func TestResponsesToolOnlyCreatedFailureDoesNotPublish(t *testing.T) {
	t.Setenv("FAK_EP_FANOUT_ADDRS", "")
	for _, failAt := range []int{1, 2} {
		t.Run(fmt.Sprint(failAt), func(t *testing.T) {
			srv := newTestServer(t)
			p := &scriptedLiveResponsesPlanner{supports: true, gateAllows: true, comp: &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, ToolCalls: []agent.ToolCall{{ID: "call_a", Type: "function", Function: agent.Func{Name: "allow_a", Arguments: `{"x":1}`}}}}, FinishReason: "tool_calls"}}
			srv.planner = p
			w := &responsesStartWriter{recorder: httptest.NewRecorder(), failAt: failAt, cause: errors.New("created write failed")}
			req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{"model":"test-model","input":"hi","stream":true,"tools":[{"type":"function","name":"allow_a"}]}`))
			req.Header.Set("Content-Type", "application/json")
			srv.Handler().ServeHTTP(w, req)
			p.assertPerTokenRoute(t, true)
			store := srv.responsesContinuationState()
			store.mu.Lock()
			saved := len(store.entries)
			store.mu.Unlock()
			if w.writes != failAt || saved != 0 || strings.Contains(w.recorder.Body.String(), "response.output_item") || strings.Contains(w.recorder.Body.String(), "response.completed") || w.firstHeader.Get("Content-Type") != "text/event-stream" {
				t.Fatalf("writes=%d saved=%d body=%s", w.writes, saved, w.recorder.Body.String())
			}
		})
	}
}

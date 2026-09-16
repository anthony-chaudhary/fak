package gateway

// native_admission_1738_test.go — the reproduction witness for #1738.
//
// The defect: a client that opens concurrent streams at session start (OpenCode issues
// `small=true` title + `small=false` build within ~1s) handed the single-resident native
// server a second request that had NO admission path at all. The native owned-loop branch
// (messages.go branched to serveNativeMessages/serveNativeMessagesStream) ran the loop
// directly, so the second request blocked inside the resident model lock and produced no
// output — the client reached its provider header timeout. See the ticket's repro: two
// concurrent curls against :8080 both hit 600s with no files produced.
//
// The fix routes that branch through the same scheduler admission every proxied wire
// crosses, with a bounded queue wait. These tests pin the two outcomes the ticket's
// acceptance gate names: each concurrent request is either ADMITTED (and completes) or
// REFUSED with a typed reason — never a silent hang.

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
)

// nativeAdmissionServer stands up a native Server whose admission gate has exactly one
// running slot and one waiting slot, so a second concurrent stream must QUEUE and a third
// must SHED. The planner blocks until released, which is what keeps the first request
// holding the single slot. The queue-wait ceiling is shrunk so a still-queued request is
// answered with a typed shed inside the test's own timeout rather than the 120s default.
func nativeAdmissionServer(t *testing.T, planner agent.Planner) (*Server, *AdmissionController, *httptest.Server) {
	t.Helper()
	agent.Configure()
	abi.RegisterRegionBackend(inlineBackend{})
	srv, err := New(Config{
		EngineID:       "localtools",
		Model:          "test-model",
		VDSO:           true,
		Native:         true,
		NativeMaxTurns: 4,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(srv.Close)
	// Shrink the declared queue-wait ceiling so a still-queued follower is answered with a
	// typed shed inside the test's own timeout, and restore it afterward.
	SetNativeAdmissionCeiling(2 * time.Second)
	t.Cleanup(func() { SetNativeAdmissionCeiling(0) })
	srv.planner = planner
	ctl := NewAdmissionController(AdmissionPolicy{MaxNumSeqs: 1, MaxWaiting: 1, AgingRounds: 1})
	srv.SetAdmissionController(ctl)
	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(ts.Close)
	return srv, ctl, ts
}

func postConcurrentNativeMessages(t *testing.T, client *http.Client, base, trace string) *http.Response {
	t.Helper()
	body, err := json.Marshal(map[string]any{
		"model":      "test-model",
		"max_tokens": 16,
		"messages":   []map[string]string{{"role": "user", "content": "Reply with exactly: A"}},
	})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	req, err := http.NewRequest(http.MethodPost, base+"/v1/messages", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("NewRequest: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set(traceHeader, trace)
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("POST /v1/messages: %v", err)
	}
	return resp
}

// TestNativeConcurrentStreamsAdmitOrRefuse is the #1738 acceptance witness at the HTTP
// level. A first native request is admitted and parked inside the planner holding the one
// running slot; a concurrent second request must then be answered within the predeclared
// ceiling with a TYPED outcome (429 shed, or 200 once promoted) — never a silent hang.
// Before the fix the follower bypassed admission entirely, entered the planner, and the
// turn never completed (the ticket's two-curl-600s repro).
//
// The test asserts the invariant, not a fixed race outcome: whichever of shed/admitted the
// follower lands on, it must carry a real status and the typed reason, and the admitted
// path must still complete normally.
func TestNativeConcurrentStreamsAdmitOrRefuse(t *testing.T) {
	planner := newBlockingAdmissionPlanner()
	_, _, ts := nativeAdmissionServer(t, planner)

	// First request: admitted, then parked inside the planner holding the single slot.
	firstDone := make(chan int, 1)
	go func() {
		resp := postConcurrentNativeMessages(t, &http.Client{}, ts.URL, "native-first")
		resp.Body.Close()
		firstDone <- resp.StatusCode
	}()
	select {
	case <-planner.entered:
	case <-time.After(10 * time.Second):
		t.Fatal("first native request never reached the planner (admission admit path broken)")
	}

	// Concurrent follower: the gate has no free slot. It must return within the declared
	// ceiling with a typed status, NOT block until the client gives up.
	type result struct {
		code int
		body string
	}
	follower := make(chan result, 1)
	go func() {
		resp := postConcurrentNativeMessages(t, &http.Client{}, ts.URL, "native-follower")
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		follower <- result{code: resp.StatusCode, body: string(body)}
	}()

	select {
	case got := <-follower:
		// The only typed non-200 outcome the gate produces for a busy single slot.
		if got.code != http.StatusTooManyRequests {
			t.Fatalf("concurrent native follower: status = %d, want 429 within the ceiling; body=%s", got.code, got.body)
		}
		if !strings.Contains(got.body, "scheduler_overloaded") {
			t.Fatalf("concurrent native follower body lacks the typed reason scheduler_overloaded: %s", got.body)
		}
	case <-time.After(3 * nativeAdmissionCeiling):
		t.Fatal("concurrent native follower never returned a typed status — the silent hang #1738 reports")
	}

	// The first (admitted) request must still complete once the planner releases.
	planner.Release()
	select {
	case code := <-firstDone:
		if code != http.StatusOK {
			t.Fatalf("first (admitted) native request: status = %d, want 200", code)
		}
	case <-time.After(20 * time.Second):
		t.Fatal("first native request did not complete after the planner released")
	}
}

// TestNativeAdmissionRefusalIsTypedAndNeverHangs pins the second half of the acceptance
// gate: a native request the gate can never admit is refused with a typed 400 BEFORE the
// owned loop runs, so the client sees a real error status rather than a request that
// silently never produces output.
//
// The reachable route is the impossible-budget refusal: a request whose footprint exceeds
// the whole token budget can never be scheduled, so VerdictRefused -> 400
// context_length_exceeded (admission.go admissionErrorStatus). Before the fix this branch
// never consulted the gate at all and the request went straight into the loop.
func TestNativeAdmissionRefusalIsTypedAndNeverHangs(t *testing.T) {
	planner := newBlockingAdmissionPlanner()
	srv, _, ts := nativeAdmissionServer(t, planner)
	// Shrink the budget so the request's own footprint is un-schedulable: impossible ->
	// VerdictRefused (400), not queued and not shed.
	srv.SetAdmissionController(NewAdmissionController(AdmissionPolicy{TokenBudget: 1, MaxWaiting: 1, AgingRounds: 1}))

	resp := postConcurrentNativeMessages(t, http.DefaultClient, ts.URL, "native-impossible")
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("impossible native request: status = %d, want 400; body=%s", resp.StatusCode, body)
	}
	if !strings.Contains(string(body), "context_length_exceeded") {
		t.Fatalf("impossible native request body lacks the typed reason: %s", body)
	}
	if planner.Calls() != 0 {
		t.Fatalf("a refused native request reached the planner (%d calls); admission did not precede the loop", planner.Calls())
	}
}

// blockingStreamingPlanner is blockingAdmissionPlanner plus the StreamingPlanner surface,
// so a native request takes the SSE branch (serveNativeMessagesStream) rather than falling
// back to the buffered handler — the branch whose admission placement the pre-header test
// must actually exercise.
type blockingStreamingPlanner struct{ *blockingAdmissionPlanner }

func (*blockingStreamingPlanner) StreamingSupported() bool { return true }

func (p *blockingStreamingPlanner) CompleteStream(ctx context.Context, sink agent.StreamSink, messages []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	comp, err := p.Complete(ctx, messages, tools, opts...)
	if err != nil {
		return nil, err
	}
	if sink != nil {
		if err := sink(comp.Message.Content); err != nil {
			return nil, err
		}
	}
	return comp, nil
}

// TestNativeStreamAdmissionPrecedesHeaders proves the streamed branch admits BEFORE the
// SSE 200: a refused streamed request must be a real 4xx status, not an error frame inside
// a 200. A client cannot act on the latter — it is exactly the "turn never completes"
// shape #1738 reports.
func TestNativeStreamAdmissionPrecedesHeaders(t *testing.T) {
	planner := &blockingStreamingPlanner{newBlockingAdmissionPlanner()}
	srv, _, ts := nativeAdmissionServer(t, planner)
	srv.SetAdmissionController(NewAdmissionController(AdmissionPolicy{TokenBudget: 1, MaxWaiting: 1, AgingRounds: 1}))

	body := `{"model":"test-model","max_tokens":16,"stream":true,"messages":[{"role":"user","content":"hi"}]}`
	req, err := http.NewRequest(http.MethodPost, ts.URL+"/v1/messages", strings.NewReader(body))
	if err != nil {
		t.Fatalf("NewRequest: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set(traceHeader, "native-stream-refused")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("POST streamed /v1/messages: %v", err)
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(resp.Body)
	if resp.StatusCode == http.StatusOK {
		t.Fatalf("refused STREAMED native request wrote a 200; admission must precede the SSE headers. body=%s", raw)
	}
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("streamed refused request: status = %d, want 400; body=%s", resp.StatusCode, raw)
	}
	if ct := resp.Header.Get("Content-Type"); strings.Contains(ct, "text/event-stream") {
		t.Fatalf("refused streamed request still advertises text/event-stream (%q)", ct)
	}
	if planner.Calls() != 0 {
		t.Fatalf("a refused streamed request reached the planner (%d calls)", planner.Calls())
	}
}

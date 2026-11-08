package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/modelengine"
)

// pdSSEChunk is the closed SSE frame shape the front door emits; tests parse
// it instead of asserting on prose.
type pdSSEChunk struct {
	Choices []struct {
		Delta struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"delta"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
	Usage *struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
	Error *struct {
		Code    string `json:"code"`
		Message string `json:"message"`
	} `json:"error"`
}

// parsePDSSEReplay turns a recorded SSE body into its data frames in order.
// The [DONE] terminator is reported separately; malformed frames are kept as
// raw strings so a failure shows the offending line.
func parsePDSSEReplay(t *testing.T, body string) (chunks []pdSSEChunk, done bool, raws []string) {
	t.Helper()
	for _, line := range strings.Split(body, "\n") {
		line = strings.TrimSpace(line)
		data, ok := strings.CutPrefix(line, "data:")
		if !ok {
			continue
		}
		data = strings.TrimSpace(data)
		if data == "" {
			continue
		}
		if data == "[DONE]" {
			done = true
			return
		}
		var ch pdSSEChunk
		if err := json.Unmarshal([]byte(data), &ch); err != nil {
			raws = append(raws, data)
			continue
		}
		chunks = append(chunks, ch)
	}
	return
}

// fak-test:runtime medium est=5s lane=default
func TestServePDStreamEndToEndOverWire(t *testing.T) {
	// The production P/D call path with stream:true: prefill role's front door
	// ships the paged-KV frame to the decode role over loopback and streams
	// every generated token back as an SSE delta.
	m := fakmodel.NewSynthetic(modelengine.SyntheticConfig())
	prefillCluster := modelengine.NewNativePDCluster(m, 1)
	defer prefillCluster.Close()
	decodeCluster := modelengine.NewNativePDCluster(m, 2)
	defer decodeCluster.Close()

	wire := modelengine.NewPDWireServer(decodeCluster)
	if err := wire.Listen("127.0.0.1:0"); err != nil {
		t.Fatalf("wire listen: %v", err)
	}
	go func() { _ = wire.Serve() }()
	defer wire.Close()

	router := modelengine.NewPDPeerRouter(prefillCluster, []modelengine.PDDecodePeer{
		{WireAddr: wire.Addr().String()},
	}, nil)
	h := &servePDChatHandler{
		cluster:          prefillCluster,
		ship:             router,
		encode:           pdByteEncode,
		decode:           pdByteDecode,
		newStreamDecoder: func() pdStreamDecoder { return &pdByteStreamDecoder{} },
		modelLabel:       "pd-e2e-stream",
	}

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"messages":[{"role":"user","content":"ping"}],"stream":true}`))
	req = req.WithContext(ctx)
	rec := httptest.NewRecorder()
	h.handleChat(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("front door = %d, want 200 (%s)", rec.Code, rec.Body.String())
	}
	if ct := rec.Header().Get("Content-Type"); ct != "text/event-stream" {
		t.Fatalf("Content-Type = %q, want text/event-stream", ct)
	}

	chunks, done, raws := parsePDSSEReplay(t, rec.Body.String())
	if len(raws) > 0 {
		t.Fatalf("unparseable SSE frames: %v", raws)
	}
	if !done {
		t.Fatalf("SSE body does not end with data: [DONE]:\n%s", rec.Body.String())
	}

	// Every generated token arrives as its own content delta, in order, before
	// the final finish/usage chunk.
	var deltas []string
	for i, ch := range chunks {
		if ch.Error != nil {
			t.Fatalf("chunk %d carries error %+v", i, ch.Error)
		}
		if ch.Usage != nil || (len(ch.Choices) > 0 && ch.Choices[0].FinishReason != nil) {
			if i != len(chunks)-1 {
				t.Fatalf("chunk %d carries usage/finish before the last chunk", i)
			}
			break
		}
		if len(ch.Choices) != 1 || ch.Choices[0].Delta.Content == "" {
			t.Fatalf("chunk %d has no non-empty content delta: %+v", i, ch)
		}
		deltas = append(deltas, ch.Choices[0].Delta.Content)
	}
	if len(deltas) == 0 {
		t.Fatalf("no content deltas streamed:\n%s", rec.Body.String())
	}
	if first := chunks[0].Choices[0].Delta.Role; first != "assistant" {
		t.Fatalf("first chunk delta.role = %q, want assistant", first)
	}
	wantTokens := modelengine.PDGenTokens // synthetic checkpoint never early-stops
	if len(deltas) != wantTokens {
		t.Fatalf("streamed %d content deltas, want %d (one per generated token)", len(deltas), wantTokens)
	}

	last := chunks[len(chunks)-1]
	if last.Usage == nil {
		t.Fatalf("final chunk carries no usage: %+v", last)
	}
	if last.Usage.CompletionTokens != wantTokens || last.Usage.PromptTokens != 4 || last.Usage.TotalTokens != 4+wantTokens {
		t.Fatalf("usage = %+v, want prompt 4 / completion %d", last.Usage, wantTokens)
	}
	if len(last.Choices) != 1 || last.Choices[0].FinishReason == nil || *last.Choices[0].FinishReason != "stop" {
		t.Fatalf("final choice = %+v, want finish_reason stop", last.Choices)
	}

	// The decode role admitted the import without re-prefilling, exactly as the
	// non-streaming e2e asserts.
	worker, ok := decodeCluster.DecodeWorker("decode-0")
	if !ok {
		t.Fatal("decode-0 not found on the decode cluster")
	}
	if stats := worker.Stats(); stats.RePrefillOnDecode != 0 {
		t.Fatalf("decode stats = %+v, want no decode-side prefill", stats)
	}
}

// fak-test:runtime fast est=1s lane=default
func TestServePDStreamMaxTokensLengthFinish(t *testing.T) {
	ids := []int{int('a'), int('b'), int('c'), int('d'), int('e'), int('f')}
	ship := &pdStubShip{t: t, tokens: ids}
	h, done := newPDTestHandler(t, ship)
	defer done()
	rec := postPDChat(t, h, `{"messages":[{"role":"user","content":"ping"}],"stream":true,"max_tokens":4}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("stream truncation = %d, want 200 (%s)", rec.Code, rec.Body.String())
	}
	chunks, doneFlag, raws := parsePDSSEReplay(t, rec.Body.String())
	if len(raws) > 0 || !doneFlag {
		t.Fatalf("SSE replay malformed=%v done=%v:\n%s", raws, doneFlag, rec.Body.String())
	}
	var deltas []string
	for _, ch := range chunks[:len(chunks)-1] {
		if len(ch.Choices) != 1 || ch.Choices[0].Delta.Content == "" {
			t.Fatalf("content chunk malformed: %+v", ch)
		}
		deltas = append(deltas, ch.Choices[0].Delta.Content)
	}
	if got := strings.Join(deltas, ""); got != "abcd" {
		t.Fatalf("streamed %q, want abcd truncated at max_tokens 4", got)
	}
	last := chunks[len(chunks)-1]
	if last.Usage == nil || last.Usage.CompletionTokens != 4 {
		t.Fatalf("final usage = %+v, want completion_tokens 4", last.Usage)
	}
	if len(last.Choices) != 1 || last.Choices[0].FinishReason == nil || *last.Choices[0].FinishReason != "length" {
		t.Fatalf("final choice = %+v, want finish_reason length", last.Choices)
	}
	// The request-level cap rides the wire as the lane retirement budget.
	if ship.lastReq.GenBudget != 4 {
		t.Fatalf("transport saw GenBudget %d, want 4", ship.lastReq.GenBudget)
	}
}

// fak-test:runtime fast est=1s lane=default
func TestServePDStreamHandoffFailureBeforeFirstEventIsTyped502(t *testing.T) {
	ship := &pdStubShip{t: t, tokens: []int{1}, shipErr: errors.New("peer down")}
	h, done := newPDTestHandler(t, ship)
	defer done()
	rec := postPDChat(t, h, `{"messages":[{"role":"user","content":"hi"}],"stream":true}`)
	if rec.Code != http.StatusBadGateway || pdChatErrorCode(t, rec) != pdChatCodeHandoffFailed {
		t.Fatalf("pre-stream handoff failure = %d/%s, want 502/%s", rec.Code, pdChatErrorCode(t, rec), pdChatCodeHandoffFailed)
	}
	if ct := rec.Header().Get("Content-Type"); ct != "application/json" {
		t.Fatalf("Content-Type = %q, want application/json (status line was never committed)", ct)
	}
}

// fak-test:runtime fast est=1s lane=default
func TestServePDStreamMidStreamFailureEmitsErrorEventWithoutDone(t *testing.T) {
	ship := &pdStubShip{t: t, tokens: []int{int('x'), int('y'), int('z')}, stopAfter: 2, midStreamErr: errors.New("peer dropped")}
	h, done := newPDTestHandler(t, ship)
	defer done()
	rec := postPDChat(t, h, `{"messages":[{"role":"user","content":"hi"}],"stream":true}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("mid-stream failure = %d, want 200 (SSE already started)", rec.Code)
	}
	body := rec.Body.String()
	if !strings.Contains(body, `"content":"x"`) || !strings.Contains(body, `"content":"y"`) {
		t.Fatalf("missing delivered deltas:\n%s", body)
	}
	if strings.Contains(body, `"content":"z"`) {
		t.Fatalf("undelivered token leaked into the stream:\n%s", body)
	}
	if !strings.Contains(body, pdChatCodeHandoffFailed) {
		t.Fatalf("missing typed handoff_failed error event:\n%s", body)
	}
	if strings.Contains(body, "[DONE]") {
		t.Fatalf("failed stream must not terminate with [DONE]:\n%s", body)
	}
}

// fak-test:runtime fast est=1s lane=default
func TestServePDStreamConsumerStoppedWritesNothingMore(t *testing.T) {
	// A response writer that dies mid-stream surfaces
	// ErrPDStreamConsumerStopped; the handler must stop cleanly — no error
	// event, no [DONE], no panic.
	ids := []int{int('a'), int('b'), int('c')}
	ship := &pdStubShip{t: t, tokens: ids}
	h, cleanup := newPDTestHandler(t, ship)
	defer cleanup()

	// A recorder whose writes fail after the second event models the dead
	// client: wrap the handler's sink path by pointing ShipStream at a sink
	// that fails on the third token.
	failing := &pdFailingSinkShip{inner: ship, failFrom: 2}
	h.ship = failing
	rec := postPDChat(t, h, `{"messages":[{"role":"user","content":"hi"}],"stream":true}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("consumer-stopped stream = %d, want 200", rec.Code)
	}
	if n := strings.Count(rec.Body.String(), `"content":"`); n != 2 {
		t.Fatalf("delivered %d content deltas before the writer died, want 2:\n%s", n, rec.Body.String())
	}
	if strings.Contains(rec.Body.String(), "[DONE]") || strings.Contains(rec.Body.String(), pdChatCodeHandoffFailed) {
		t.Fatalf("consumer-stopped stream must write nothing more:\n%s", rec.Body.String())
	}
}

// pdFailingSinkShip wraps a stub transport whose handoff dies from token
// failFrom onward, reporting ErrPDStreamConsumerStopped exactly like the wire
// client does when the front-door writer goes away.
type pdFailingSinkShip struct {
	inner    *pdStubShip
	failFrom int
}

func (f *pdFailingSinkShip) Ship(ctx context.Context, req modelengine.PDShipRequest) (modelengine.PDShipResult, error) {
	return f.inner.Ship(ctx, req)
}

func (f *pdFailingSinkShip) ShipStream(_ context.Context, req modelengine.PDShipRequest, sink modelengine.PDTokenSink) (modelengine.PDShipResult, error) {
	f.inner.lastReq = req
	for i, tok := range f.inner.tokens {
		if i >= f.failFrom {
			return modelengine.PDShipResult{}, fmt.Errorf("%w: simulated dead writer", modelengine.ErrPDStreamConsumerStopped)
		}
		if err := sink(tok); err != nil {
			return modelengine.PDShipResult{}, err
		}
	}
	return modelengine.PDShipResult{Tokens: append([]int(nil), f.inner.tokens...)}, nil
}

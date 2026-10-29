package gateway

// Reproduction + regression tests for #14110: the Anthropic RAW streaming
// passthrough used to stamp its durable perf row with the CLIENT-REQUESTED model
// id, so a router alias serving a different checkpoint left rows that could not
// answer "which model actually ran"; and a turn whose upstream never reported a
// token count read as a free 0/0 turn instead of "unobserved".
//
// Every assertion here drives the bug through APIs the pre-fix tree already has
// (the /v1/messages relay, the durable ledger file, the perf Report shape), so
// the red-then-green symptom witness can overlay this file onto the parent and
// watch it fail there.

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/perfledger"
)

// newAnthropicPassthroughServer stands up a gateway fronting an Anthropic-wire
// upstream that replays sse verbatim, with a durable perf ledger under dir.
func newAnthropicPassthroughServer(t *testing.T, sse string, dir string) (*Server, *perfledger.Writer) {
	t.Helper()
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, sse)
	}))
	t.Cleanup(upstream.Close)

	srv, err := New(Config{EngineID: "test", Model: "boot-default-model", BaseURL: upstream.URL, Provider: "anthropic", APIKey: "configured-key"})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)

	path := filepath.Join(dir, "gateway-perf.jsonl")
	w := perfledger.OpenWriter(path, perfledger.DefaultMaxBytes)
	srv.SetPerfLedger(w, nil, false)
	return srv, w
}

// postStreamedMessages posts one streamed /v1/messages turn for reqModel.
func postStreamedMessages(t *testing.T, h http.Handler, reqModel string) {
	t.Helper()
	inbound := []byte(`{"model":"` + reqModel + `","max_tokens":64,"stream":true,` +
		`"messages":[{"role":"user","content":"hi"}]}`)
	req, _ := http.NewRequest("POST", "/v1/messages", bytes.NewReader(inbound))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", "caller-key")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200: %s", rec.Code, rec.Body.String())
	}
}

// lastPerfRow closes the ledger and returns its final row plus the raw JSON of
// that row (for fields the typed Record may not name yet).
func lastPerfRow(t *testing.T, w *perfledger.Writer, path string) (perfledger.Record, map[string]any) {
	t.Helper()
	if err := w.Close(); err != nil {
		t.Fatalf("writer close: %v", err)
	}
	recs, _ := perfledger.ReadTail(path)
	if len(recs) != 1 {
		t.Fatalf("ledger rows = %d, want exactly 1: %+v", len(recs), recs)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read ledger: %v", err)
	}
	var row map[string]any
	if err := json.Unmarshal(bytes.TrimSpace(raw), &row); err != nil {
		t.Fatalf("decode ledger row: %v", err)
	}
	return recs[0], row
}

// fak-test:runtime fast est=500ms lane=default
// TestPassthroughPerfRowCarriesServedModelAndUsage is the #14110 reproduction:
// the client asks for "test-model" (a router alias), the upstream reports it
// actually served RadixArk/Qwen3.8-27B-NVFP4 with real usage — the durable row
// must carry the SERVED name and the nonzero counts, not the requested alias.
func TestPassthroughPerfRowCarriesServedModelAndUsage(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})

	const served = "RadixArk/Qwen3.8-27B-NVFP4"
	sse := strings.Join([]string{
		`event: message_start`,
		`data: {"type":"message_start","message":{"id":"m1","type":"message","role":"assistant","model":"` + served + `","content":[],"stop_reason":null,"usage":{"input_tokens":57,"cache_read_input_tokens":0,"cache_creation_input_tokens":0,"output_tokens":0}}}`,
		``,
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"PONG"}}`,
		``,
		`event: content_block_stop`,
		`data: {"type":"content_block_stop","index":0}`,
		``,
		`event: message_delta`,
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":8}}`,
		``,
		`event: message_stop`,
		`data: {"type":"message_stop"}`,
		``,
	}, "\n")

	dir := t.TempDir()
	srv, w := newAnthropicPassthroughServer(t, sse, dir)
	postStreamedMessages(t, srv.Handler(), "test-model")
	row, raw := lastPerfRow(t, w, filepath.Join(dir, "gateway-perf.jsonl"))

	if row.Model != served {
		t.Errorf("row model = %q, want the upstream-reported served model %q (not the requested alias or the boot default)", row.Model, served)
	}
	if row.PromptTokens != 57 || row.CompletionTokens != 8 {
		t.Errorf("row tokens = %d/%d, want 57/8 from the upstream usage blocks", row.PromptTokens, row.CompletionTokens)
	}
	if v, present := raw["usage_unreported"]; present && v != false {
		t.Errorf("row marked usage_unreported=%v although the upstream reported both counts", v)
	}
	if row.FinishReason != "end_turn" {
		t.Errorf("row finish_reason = %q, want end_turn", row.FinishReason)
	}
}

// fak-test:runtime fast est=500ms lane=default
// TestPassthroughPerfRowMarksUnreportedUsage proves the honest-marker half of
// #14110: an upstream that relays content but never reports a token count leaves
// a row whose zeros mean "unobserved", stamped usage_unreported instead of
// masquerading as a free turn.
func TestPassthroughPerfRowMarksUnreportedUsage(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})

	sse := strings.Join([]string{
		`event: message_start`,
		`data: {"type":"message_start","message":{"id":"m2","type":"message","role":"assistant","model":"served-x","content":[],"stop_reason":null}}`,
		``,
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}}`,
		``,
		`event: content_block_stop`,
		`data: {"type":"content_block_stop","index":0}`,
		``,
		`event: message_delta`,
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null}}`,
		``,
		`event: message_stop`,
		`data: {"type":"message_stop"}`,
		``,
	}, "\n")

	dir := t.TempDir()
	srv, w := newAnthropicPassthroughServer(t, sse, dir)
	postStreamedMessages(t, srv.Handler(), "alias-y")
	row, raw := lastPerfRow(t, w, filepath.Join(dir, "gateway-perf.jsonl"))

	if row.PromptTokens != 0 || row.CompletionTokens != 0 {
		t.Fatalf("row tokens = %d/%d, want 0/0 (the upstream reported none)", row.PromptTokens, row.CompletionTokens)
	}
	if v, present := raw["usage_unreported"]; !present || v != true {
		t.Errorf("row must be marked usage_unreported=true: content flowed with no upstream usage (raw=%v)", v)
	}
	// The served model is still captured even when usage is absent.
	if row.Model != "served-x" {
		t.Errorf("row model = %q, want served-x", row.Model)
	}
}

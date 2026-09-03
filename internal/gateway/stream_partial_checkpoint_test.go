package gateway

// stream_partial_checkpoint_test.go — #10672: partial-output durability on the Anthropic
// live passthrough. When a stream that already relayed assistant text dies mid-stream, the
// accumulated text (exactly what the client already saw) is durably checkpointed at the
// failure boundary, so a later resume continues from the last good state instead of
// restarting the turn. The checkpoint is a JSON document under .fak/streamcheckpoints
// (env-overridable), written best-effort and atomically — durability must never become a
// new fault path on the request.

import (
	"bufio"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// anthropicDeathUpstream is the minutes-long-stream fixture from #10672, compressed: it
// opens a healthy Anthropic Messages SSE turn, relays two text deltas ("tok1 ", "tok2 "),
// closes the content block, and then goes SILENT forever — the stall that kills the turn
// after N tokens of delivered output. The stall reader (FAK_STREAM_STALL_TIMEOUT_S) is
// what turns the silence into the terminal failure.
func anthropicDeathUpstream(t *testing.T) *httptest.Server {
	t.Helper()
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		f, _ := w.(http.Flusher)
		const prefix = `event: message_start
data: {"type":"message_start","message":{"id":"msg_ckpt","type":"message","role":"assistant","model":"claude-test","content":[],"stop_reason":null,"usage":{"input_tokens":9,"cache_read_input_tokens":0,"cache_creation_input_tokens":0,"output_tokens":0}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"tok1 "}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"tok2 "}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

`
		_, _ = io.WriteString(w, prefix)
		if f != nil {
			f.Flush()
		}
		<-release // go silent mid-stream: the turn dies here, output already delivered
	}))
	t.Cleanup(func() {
		close(release)
		srv.Close()
	})
	return srv
}

// postAnthropicStreamDrain sends a streaming /v1/messages request and returns the drained
// SSE body (bounded by the caller's expectations below).
func postAnthropicStreamDrain(t *testing.T, gatewayURL string, inbound []byte) string {
	t.Helper()
	req, _ := http.NewRequest("POST", gatewayURL+"/v1/messages", strings.NewReader(string(inbound)))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", "caller-key")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	defer resp.Body.Close()
	var sb strings.Builder
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 0, 8*1024), 1024*1024)
	for sc.Scan() {
		sb.WriteString(sc.Text())
		sb.WriteByte('\n')
	}
	return sb.String()
}

// TestAnthropicPassthroughMidStreamDeathPersistsCheckpoint is the #10672 partial-output
// witness: a stream killed after emitting N tokens leaves a durable checkpoint holding
// exactly those N tokens, plus the phase/bound metadata a resume needs — and the client's
// terminal error frame still arrives cleanly.
func TestAnthropicPassthroughMidStreamDeathPersistsCheckpoint(t *testing.T) {
	ckDir, incDir := t.TempDir(), t.TempDir()
	t.Setenv("FAK_STREAM_CHECKPOINT_DIR", ckDir)
	t.Setenv("FAK_STREAM_INCIDENT_DIR", incDir)
	t.Setenv("FAK_STREAM_STALL_TIMEOUT_S", "5")

	upstream := anthropicDeathUpstream(t)

	srv, err := New(Config{
		EngineID: "mock", Model: "claude-test", BaseURL: upstream.URL, Provider: "anthropic",
		APIKey: "configured-key",
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	const traceID = "stream-checkpoint-witness"
	srv.SetDefaultTraceID(traceID)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	body := postAnthropicStreamDrain(t, ts.URL, []byte(`{"model":"claude-test","max_tokens":64,"stream":true,"messages":[{"role":"user","content":"probe"}]}`))
	// The client still gets a clean terminal: a typed upstream_stalled error event.
	if !strings.Contains(body, "upstream_stalled") {
		t.Fatalf("client stream did not end with the typed stall error; body:\n%s", body)
	}

	// The checkpoint: exactly one file, holding the exact delivered prefix.
	ents, err := os.ReadDir(ckDir)
	if err != nil || len(ents) == 0 {
		t.Fatalf("no durable partial-output checkpoint was written to %s — a stream that died after delivering output lost all of it (#10672): %v", ckDir, err)
	}
	if len(ents) != 1 {
		t.Fatalf("checkpoint dir has %d entries, want exactly one: %v", len(ents), ents)
	}
	raw, err := os.ReadFile(filepath.Join(ckDir, ents[0].Name()))
	if err != nil {
		t.Fatalf("read checkpoint: %v", err)
	}
	var ck struct {
		Schema          string `json:"schema"`
		Wire            string `json:"wire"`
		TraceID         string `json:"trace_id"`
		Model           string `json:"model"`
		Phase           string `json:"phase"`
		ElapsedMS       int64  `json:"elapsed_ms"`
		Text            string `json:"text"`
		EstimatedTokens int    `json:"estimated_tokens"`
		BoundPolicy     string `json:"bound_policy"`
		Reason          string `json:"reason"`
	}
	if err := json.Unmarshal(raw, &ck); err != nil {
		t.Fatalf("decode checkpoint %q: %v", raw, err)
	}
	if ck.Schema != "fak-stream-checkpoint/1" {
		t.Fatalf("schema = %q, want fak-stream-checkpoint/1", ck.Schema)
	}
	if ck.Wire != "anthropic_messages" {
		t.Fatalf("wire = %q, want anthropic_messages", ck.Wire)
	}
	if ck.TraceID != traceID {
		t.Fatalf("trace_id = %q, want %q", ck.TraceID, traceID)
	}
	if ck.Model != "claude-test" {
		t.Fatalf("model = %q, want the requested model", ck.Model)
	}
	if ck.Phase != "mid_stream" {
		t.Fatalf("phase = %q, want mid_stream — the death landed after a first token", ck.Phase)
	}
	// THE witness: the checkpoint holds the exact text the client already received.
	if ck.Text != "tok1 tok2 " {
		t.Fatalf("checkpoint text = %q, want the exact delivered prefix %q", ck.Text, "tok1 tok2 ")
	}
	// The checkpoint holds >= N tokens (2 delivered deltas), by the same estimator the
	// warm-continue resume path uses to decrement the budget.
	if ck.EstimatedTokens < 2 {
		t.Fatalf("estimated_tokens = %d, want >= 2 for the delivered prefix", ck.EstimatedTokens)
	}
	if ck.ElapsedMS <= 0 {
		t.Fatalf("elapsed_ms = %d, want > 0", ck.ElapsedMS)
	}
	if !strings.Contains(ck.BoundPolicy, "max-duration=off") {
		t.Fatalf("bound_policy = %q, want the max-duration policy state recorded", ck.BoundPolicy)
	}
	if ck.Reason != "stream-death" {
		t.Fatalf("reason = %q, want stream-death", ck.Reason)
	}

	// The correlated incident packet for the same death, on the Anthropic wire.
	incs := readIncidents(t, incDir)
	var pkt map[string]any
	for _, m := range incs {
		if m["wire"] == "anthropic_messages" {
			pkt = m
			break
		}
	}
	if pkt == nil {
		t.Fatalf("no anthropic_messages incident in %v", incs)
	}
	if pkt["phase"] != "mid_stream" {
		t.Fatalf("incident phase = %v, want mid_stream", pkt["phase"])
	}
	if v, ok := pkt["bytes_emitted"].(float64); !ok || v < float64(len("tok1 tok2 ")) {
		t.Fatalf("incident bytes_emitted = %v, want >= %d", pkt["bytes_emitted"], len("tok1 tok2 "))
	}
	if pkt["upstream_status_class"] != "stall" {
		t.Fatalf("incident upstream_status_class = %v, want stall", pkt["upstream_status_class"])
	}
}

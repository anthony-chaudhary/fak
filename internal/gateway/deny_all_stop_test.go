package gateway

import (
	"bytes"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// TestRecordAdjudicationOutcomeCountsAndResets pins the pure accumulator: a deny-all turn
// bumps both the cumulative count and the consecutive run; any non-deny-all turn resets the
// consecutive run to 0 while leaving the cumulative count intact.
func TestRecordAdjudicationOutcomeCountsAndResets(t *testing.T) {
	m := newGatewayMetrics(time.Unix(0, 0))

	m.recordAdjudicationOutcome(true)
	m.recordAdjudicationOutcome(true)
	if stops, consec := m.denyAllSnapshot(); stops != 2 || consec != 2 {
		t.Fatalf("after two deny-all turns: stops=%d consec=%d, want 2/2", stops, consec)
	}

	// A non-deny-all turn (a survivor, or a pure-text turn) resets the consecutive run but not
	// the cumulative total.
	m.recordAdjudicationOutcome(false)
	if stops, consec := m.denyAllSnapshot(); stops != 2 || consec != 0 {
		t.Fatalf("after reset turn: stops=%d consec=%d, want 2/0", stops, consec)
	}

	// A fresh deny-all run starts the consecutive count over from 1.
	m.recordAdjudicationOutcome(true)
	if stops, consec := m.denyAllSnapshot(); stops != 3 || consec != 1 {
		t.Fatalf("after new deny-all: stops=%d consec=%d, want 3/1", stops, consec)
	}

	// The render surfaces both series, and the summary carries the cumulative count.
	var b strings.Builder
	m.writeDenyAllMetrics(&b)
	out := b.String()
	if !strings.Contains(out, "fak_guard_deny_all_stops_total 3") {
		t.Fatalf("metrics missing stops_total: %s", out)
	}
	if !strings.Contains(out, "fak_guard_deny_all_consecutive 1") {
		t.Fatalf("metrics missing consecutive gauge: %s", out)
	}
	if got := m.adjudicationSummary().DenyAllStops; got != 3 {
		t.Fatalf("summary DenyAllStops = %d, want 3", got)
	}
}

// TestNilMetricsRecordAdjudicationOutcomeNoPanic guards the nil-receiver contract the other
// observe methods hold: a Server built without metrics must not panic on the hot path.
func TestNilMetricsRecordAdjudicationOutcomeNoPanic(t *testing.T) {
	var m *gatewayMetrics
	m.recordAdjudicationOutcome(true) // must be a no-op, not a nil deref
	if stops, consec := m.denyAllSnapshot(); stops != 0 || consec != 0 {
		t.Fatalf("nil snapshot = %d/%d, want 0/0", stops, consec)
	}
}

// TestAnthropicStreamDenyAllCountsDenyAllStop is the end-to-end witness: an all-denied
// STREAMED turn (the flagship passthrough path) both rewrites stop_reason to end_turn AND
// records exactly one deny-all stop in the adjudication summary — so the otherwise-invisible
// "fak ended the turn" is counted, which is what the guard Stop-hook resumes the agent past.
func TestAnthropicStreamDenyAllCountsDenyAllStop(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})

	inbound := []byte(`{"model":"claude-test","max_tokens":64,"stream":true,` +
		`"tools":[{"name":"deny_b","input_schema":{"type":"object"}}],` +
		`"messages":[{"role":"user","content":"go"}]}`)

	upstreamSSE := strings.Join([]string{
		`event: message_start`,
		`data: {"type":"message_start","message":{"id":"m","type":"message","role":"assistant","model":"claude-test","content":[],"stop_reason":null,"usage":{"input_tokens":2,"output_tokens":0}}}`,
		``,
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"d1","name":"deny_b","input":{}}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{}"}}`,
		``,
		`event: content_block_stop`,
		`data: {"type":"content_block_stop","index":0}`,
		``,
		`event: message_delta`,
		`data: {"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":2}}`,
		``,
		`event: message_stop`,
		`data: {"type":"message_stop"}`,
		``,
	}, "\n")

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, upstreamSSE)
	}))
	defer upstream.Close()

	srv, err := New(Config{EngineID: "test", Model: "claude-test", BaseURL: upstream.URL, Provider: "anthropic", APIKey: "k", VDSO: true})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	req, _ := http.NewRequest("POST", ts.URL+"/v1/messages", bytes.NewReader(inbound))
	req.Header.Set("Content-Type", "application/json")
	httpResp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	// Drain the stream so the turn fully completes (the deny-all fold happens at message_delta).
	frames := readAnthropicSSE(t, httpResp.Body)
	_ = httpResp.Body.Close()
	if len(frames) == 0 {
		t.Fatalf("no SSE frames")
	}

	if got := srv.AdjudicationSummary().DenyAllStops; got != 1 {
		t.Fatalf("DenyAllStops = %d, want 1 (the all-denied streamed turn must count exactly one deny-all stop)", got)
	}
	// And the gauge the Stop-hook polls reads 1 consecutive after the single deny-all turn.
	if !strings.Contains(srv.renderMetrics(), "fak_guard_deny_all_consecutive 1") {
		t.Fatalf("metrics did not show one consecutive deny-all turn")
	}
}

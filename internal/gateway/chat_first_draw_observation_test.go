package gateway

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// fak-test:runtime slow est=2s
func TestChatFirstDrawObservationDefaultBoundedReader(t *testing.T) {
	srv := nativeReceiptServer(t)
	// Ordinary inference: no fak trace, receipt, or observation opt-in.
	sentinel := "FIRST_DRAW_PRIVATE_PROMPT_7f49"
	rr := postNativeReceipt(t, srv, `{"model":"synthetic-live","messages":[{"role":"user","content":"`+sentinel+`"}],"max_tokens":1,"temperature":0}`)
	if rr.Code != http.StatusOK {
		t.Fatalf("inference status=%d", rr.Code)
	}
	var response ChatResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &response); err != nil {
		t.Fatal(err)
	}
	if response.Usage.CompletionTokens != 1 {
		t.Fatalf("synthetic non-stop fixture completion=%d want1", response.Usage.CompletionTokens)
	}
	if response.Fak != nil && (response.Fak.DecodeTrace != nil || response.Fak.NativeInferenceReceipt != nil) {
		t.Fatal("ordinary request unexpectedly enables full tracing")
	}
	// An ordinary standalone HTTP request must be readable without any harness
	// session. Bind to the actual response identity, not a fabricated live owner.
	trace := rr.Header().Get("X-Trace-Id")
	if trace == "" {
		t.Fatal("ordinary inference response missing actual X-Trace-Id")
	}
	reader := httptest.NewRecorder()
	srv.Handler().ServeHTTP(reader, httptest.NewRequest(http.MethodGet, "/debug/vars", nil))
	if reader.Code != http.StatusOK {
		t.Fatalf("reader status=%d", reader.Code)
	}
	var doc map[string]json.RawMessage
	if err := json.Unmarshal(reader.Body.Bytes(), &doc); err != nil {
		t.Fatal(err)
	}
	var admission struct {
		TraceID     string          `json:"trace_id"`
		NativePhase json.RawMessage `json:"native_phase"`
	}
	if err := json.Unmarshal(doc["request_admission"], &admission); err != nil {
		t.Fatalf("actual debug join missing: %v", err)
	}
	if admission.TraceID != trace {
		t.Fatalf("default reader trace=%q want actual response trace=%q", admission.TraceID, trace)
	}
	var phase map[string]json.RawMessage
	if err := json.Unmarshal(admission.NativePhase, &phase); err != nil {
		t.Fatalf("actual native phase missing: %v", err)
	}
	raw, ok := phase["first_draw"]
	if !ok {
		t.Fatalf("default reader missing actual first_draw: %s", admission.NativePhase)
	}
	var draw map[string]any
	if err := json.Unmarshal(raw, &draw); err != nil {
		t.Fatal(err)
	}
	if len(draw) != 4 || draw["resolved_output_ceiling"] != float64(1) || draw["observed"] != true || draw["classification"] != "non_stop_token" {
		t.Fatalf("default scalar diagnostic=%s", raw)
	}
	if _, ok := draw["token_id"].(float64); !ok {
		t.Fatalf("observed token not numeric: %s", raw)
	}
	if strings.Contains(string(admission.NativePhase), sentinel) || len(raw) > 256 {
		t.Fatalf("reader leaked prompt or exceeded bounded scalar payload: %s", raw)
	}
	// Reading again cannot mutate the per-request observation.
	again := httptest.NewRecorder()
	srv.Handler().ServeHTTP(again, httptest.NewRequest(http.MethodGet, "/debug/vars", nil))
	var repeated map[string]json.RawMessage
	if err := json.Unmarshal(again.Body.Bytes(), &repeated); err != nil {
		t.Fatal(err)
	}
	var joined struct {
		NativePhase json.RawMessage `json:"native_phase"`
	}
	if err := json.Unmarshal(repeated["request_admission"], &joined); err != nil {
		t.Fatal(err)
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(joined.NativePhase, &fields); err != nil {
		t.Fatal(err)
	}
	if string(fields["first_draw"]) != string(raw) {
		t.Fatal("read mutated bounded observation")
	}
}

// fak-test:runtime slow est=2s
func TestChatFirstDrawObservationWithUnrelatedLiveRegistryTrace(t *testing.T) {
	// Controlled stale-registry software reproduction. This adapter exposes an
	// unrelated restored descriptor, not an invented owner for this HTTP turn.
	// It does not identify which descriptor was selected in the physical run.
	srv := nativeReceiptServer(t)
	rr := postNativeReceipt(t, srv, `{"model":"synthetic-live","messages":[{"role":"user","content":"controlled stale registry"}],"max_tokens":1,"temperature":0}`)
	if rr.Code != http.StatusOK {
		t.Fatalf("ordinary native inference status=%d", rr.Code)
	}
	var response ChatResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &response); err != nil {
		t.Fatal(err)
	}
	if response.Usage.CompletionTokens != 1 {
		t.Fatalf("actual native completion=%d want1", response.Usage.CompletionTokens)
	}
	trace := rr.Header().Get("X-Trace-Id")
	if trace == "" {
		t.Fatal("ordinary native response missing actual trace")
	}
	const unrelated = "controlled-restored-unrelated-live"
	if unrelated == trace {
		t.Fatal("fixture unrelated descriptor collided with actual HTTP trace")
	}
	srv.listSessions = liveSessions(SessionState{TraceID: unrelated, Run: "running", Rev: 1})
	reader := httptest.NewRecorder()
	srv.Handler().ServeHTTP(reader, httptest.NewRequest(http.MethodGet, "/debug/vars", nil))
	if reader.Code != http.StatusOK {
		t.Fatalf("default reader status=%d", reader.Code)
	}
	var doc map[string]json.RawMessage
	if err := json.Unmarshal(reader.Body.Bytes(), &doc); err != nil {
		t.Fatal(err)
	}
	var admission struct {
		TraceID     string          `json:"trace_id"`
		NativePhase json.RawMessage `json:"native_phase"`
	}
	if err := json.Unmarshal(doc["request_admission"], &admission); err != nil {
		t.Fatalf("native diagnostic hidden by unrelated live descriptor: %v", err)
	}
	if admission.TraceID != trace {
		t.Fatalf("native reader trace=%q want actual ordinary HTTP trace=%q; unrelated descriptor=%q", admission.TraceID, trace, unrelated)
	}
	var phase map[string]json.RawMessage
	if err := json.Unmarshal(admission.NativePhase, &phase); err != nil {
		t.Fatalf("actual native phase hidden by unrelated descriptor: %v", err)
	}
	var draw map[string]any
	if err := json.Unmarshal(phase["first_draw"], &draw); err != nil {
		t.Fatalf("actual first sample hidden by unrelated descriptor: %v", err)
	}
	if draw["resolved_output_ceiling"] != float64(1) || draw["observed"] != true || draw["classification"] != "non_stop_token" {
		t.Fatalf("actual native diagnostic changed by unrelated descriptor: %v", draw)
	}
	if _, ok := draw["token_id"].(float64); !ok {
		t.Fatalf("actual observed sample missing numeric token: %v", draw)
	}
	t.Log("SW-VERIFIED controlled unrelated registry adapter only; physical selected descriptor remains unknown")
}

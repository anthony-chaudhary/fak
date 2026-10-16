package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
)

// fak-test:runtime fast est=1s
func TestFanoutLiveStreamCompletionCLI(t *testing.T) {
	const partial = "data: {\"choices\":[{\"index\":0,\"text\":\"partial\",\"delta\":{\"content\":\"partial\"}}]}\n\n"
	const finish = "data: {\"choices\":[{\"index\":0,\"finish_reason\":\"stop\"}]}"
	const done = "data: [DONE]"
	const failure = "data: {\"error\":{\"message\":\"fixture failure\"}}\n\n"
	tests := []struct {
		name      string
		body      string
		truncated bool
		wantError string
	}{
		{name: "partial_eof", body: partial, wantError: "before a terminal frame"},
		{name: "empty_eof", wantError: "before a terminal frame"},
		{name: "heartbeat_eof", body: ": keepalive\n\n", wantError: "before a terminal frame"},
		{name: "truncated_http", body: partial, truncated: true, wantError: "unexpected EOF"},
		{name: "finish_then_truncated_http", body: partial + finish + "\n\n", truncated: true, wantError: "unexpected EOF"},
		{name: "error_before_done", body: partial + failure + done + "\n\n", wantError: "reported an error"},
		{name: "finish_then_error", body: partial + finish + "\n\n" + failure + done + "\n\n", wantError: "reported an error"},
		{name: "error_with_invalid_choices", body: partial + finish + "\n\n" + "data: {\"error\":{},\"choices\":false}\n\n" + done + "\n\n", wantError: "reported an error"},
		{name: "done_prefix_is_not_terminal", body: partial + done + "junk\n\n", wantError: "before a terminal frame"},
		{name: "other_choice_finish", body: "data: {\"choices\":[{\"index\":1,\"finish_reason\":\"stop\"}]}\n\n", wantError: "before a terminal frame"},
		{name: "empty_finish", body: "data: {\"choices\":[{\"index\":0,\"finish_reason\":\"\"}]}\n\n", wantError: "before a terminal frame"},
		{name: "malformed_finish", body: "data: {\"choices\":[{\"index\":0,\"finish_reason\":true}]}\n\n", wantError: "before a terminal frame"},
		{name: "done", body: partial + done + "\n\n"},
		{name: "finish_eof", body: partial + finish + "\n\n"},
		{name: "done_without_newline", body: partial + done},
		{name: "finish_without_newline", body: partial + finish},
		{name: "null_error_finish", body: "data: {\"error\":null,\"choices\":[{\"finish_reason\":\"stop\"}]}\n\n"},
	}
	for _, shape := range []string{RequestShapeCompletion, RequestShapeChatSystem} {
		t.Run(shape, func(t *testing.T) {
			for _, tt := range tests {
				t.Run(tt.name, func(t *testing.T) {
					var requests atomic.Int32
					srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
						switch r.URL.Path {
						case "/props":
							fmt.Fprint(w, `{"total_slots":1,"default_generation_settings":{"n_ctx":4096}}`)
						case "/metrics":
							fmt.Fprint(w, "fak_gateway_kv_prefix_reused_tokens_total 0\n")
						case "/v1/completions", "/v1/chat/completions":
							requests.Add(1)
							wantPath := "/v1/completions"
							if shape == RequestShapeChatSystem {
								wantPath = "/v1/chat/completions"
							}
							var request map[string]any
							if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
								t.Errorf("decode request: %v", err)
							}
							if r.URL.Path != wantPath || request["stream"] != true || request["n"] != nil {
								t.Errorf("unexpected route or single-choice stream request: path=%q stream=%v n=%v", r.URL.Path, request["stream"], request["n"])
							}
							w.Header().Set("Content-Type", "text/event-stream")
							if tt.truncated {
								w.Header().Set("Content-Length", strconv.Itoa(len(tt.body)+1))
							}
							fmt.Fprint(w, tt.body)
						default:
							http.NotFound(w, r)
						}
					}))
					defer srv.Close()
					var stdout, stderr bytes.Buffer
					code := runBenchSubagentFanout(&stdout, &stderr, []string{
						"-live", "-json", "-verify-contract=false", "-fanout=1", "-arms=no_reuse",
						"-no-reuse-url=" + srv.URL, "-prefix-tokens=64", "-suffix-tokens=16",
						"-decode-tokens=4", "-trials=1", "-request-shape=" + shape,
					})
					var receipt SubagentFanoutReceipt
					if err := json.Unmarshal(stdout.Bytes(), &receipt); err != nil {
						t.Fatalf("decode receipt: %v; exit=%d stderr=%q", err, code, stderr.String())
					}
					if requests.Load() != 1 || len(receipt.Results) != 1 {
						t.Fatalf("requests=%d results=%d, want one actual cell", requests.Load(), len(receipt.Results))
					}
					result := receipt.Results[0]
					if tt.wantError != "" {
						if code != 1 || !strings.Contains(result.Error, tt.wantError) || receipt.UnifiedThroughput != nil {
							t.Fatalf("incomplete stream accepted: exit=%d error=%q throughput=%+v", code, result.Error, receipt.UnifiedThroughput)
						}
						return
					}
					if code != 0 || result.Error != "" || result.RequestsIssued != 1 || result.ServeCompleteness == nil || !result.ServeCompleteness.ControlArmServed {
						t.Fatalf("terminal stream refused: exit=%d result=%+v stderr=%q", code, result, stderr.String())
					}
					if result.OutputEquivalence != nil || result.OutputHash != "" {
						t.Fatal("stream completion must not manufacture output equivalence")
					}
				})
			}
		})
	}
}

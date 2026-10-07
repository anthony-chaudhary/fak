package gateway

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
	"github.com/anthony-chaudhary/fak/internal/perfledger"
)

var errNativePerfTextWrite = errors.New("native perf text write refused")

type nativePerfFailedTextWriter struct {
	*httptest.ResponseRecorder
	err error
}

func (w *nativePerfFailedTextWriter) Write(p []byte) (int, error) {
	if bytes.Contains(p, []byte(`"type":"text_delta"`)) {
		w.err = errNativePerfTextWrite
		return 0, w.err
	}
	return w.ResponseRecorder.Write(p)
}

// fak-test:runtime fast est=1s lane=default
func TestNativeMessagesStreamPerfCoverage(t *testing.T) {
	const contentDelay = 20 * time.Millisecond
	const timingFloorMS = 15.0
	for _, tc := range []struct {
		name      string
		bound     bool
		stream    bool
		text      string
		stops     []string
		wantTTFT  bool
		wantSplit bool
		failWrite bool
	}{
		{name: "passthrough", stream: true, text: "visible answer", wantTTFT: true, wantSplit: true},
		{name: "roster_bound", bound: true, stream: true, text: "visible answer", wantTTFT: true, wantSplit: true},
		{name: "held_text", bound: true, stream: true, text: "STOP", stops: []string{"STOP"}},
		{name: "held_tail", bound: true, stream: true, text: "S", stops: []string{"STOP"}, wantTTFT: true},
		{name: "first_text_write_failure", bound: true, stream: true, text: "visible answer", failWrite: true},
		{name: "buffered", bound: true, text: "visible answer"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var req struct {
					Stream bool `json:"stream"`
				}
				if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
					t.Errorf("decode upstream request: %v", err)
					w.WriteHeader(http.StatusBadRequest)
					return
				}
				if !req.Stream {
					w.Header().Set("Content-Type", "application/json")
					_ = json.NewEncoder(w).Encode(map[string]any{
						"choices": []any{map[string]any{"message": map[string]string{"role": "assistant", "content": tc.text}, "finish_reason": "stop"}},
						"usage":   map[string]int{"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
					})
					return
				}
				w.Header().Set("Content-Type", "text/event-stream")
				_, _ = io.WriteString(w, "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n")
				w.(http.Flusher).Flush()
				time.Sleep(contentDelay)
				chunk, _ := json.Marshal(map[string]any{"choices": []any{map[string]any{"delta": map[string]string{"content": tc.text}}}})
				_, _ = io.WriteString(w, "data: "+string(chunk)+"\n\n")
				w.(http.Flusher).Flush()
				time.Sleep(contentDelay)
				_, _ = io.WriteString(w, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2,\"total_tokens\":5}}\n\ndata: [DONE]\n\n")
			}))
			t.Cleanup(upstream.Close)
			srv := newTestServer(t)
			srv.native = true
			srv.nativeMaxTurns = 2
			srv.planner = agent.NewHTTPPlanner(upstream.URL, "boot-model", "")
			if tc.bound {
				t.Setenv("FAK_NATIVE_PERF_TEST_KEY", "test-key")
				srv.roster = &modelroute.Roster{
					Version:  modelroute.RosterVersion,
					Accounts: []modelroute.Account{{ID: "account", Kind: modelroute.KindOpenAI, BaseURL: upstream.URL, CredEnv: "FAK_NATIVE_PERF_TEST_KEY"}},
					Bindings: []modelroute.Binding{{Model: "test-model", Account: "account", UpstreamModel: "served-model"}},
				}
			}
			ts := httptest.NewServer(srv.Handler())
			t.Cleanup(ts.Close)
			body := map[string]any{"model": "test-model", "max_tokens": 64, "stream": tc.stream,
				"messages": []map[string]string{{"role": "user", "content": "hello"}}, "stop_sequences": tc.stops}
			raw, err := json.Marshal(body)
			if err != nil {
				t.Fatal(err)
			}
			if tc.failWrite {
				writer := &nativePerfFailedTextWriter{ResponseRecorder: httptest.NewRecorder()}
				req := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewReader(raw))
				req.Header.Set("Content-Type", "application/json")
				srv.Handler().ServeHTTP(writer, req)
				if !errors.Is(writer.err, errNativePerfTextWrite) {
					t.Fatal("first text write failure was not exercised")
				}
				report := decodePerfReport(t, getPerfRecent(t, srv, ""))
				if report.Summary.TTFTMeasured != 0 {
					t.Errorf("failed text write measured TTFT count = %d, want 0", report.Summary.TTFTMeasured)
				}
				if len(report.Records) != 1 {
					t.Fatalf("failed text write rows = %d, want 1", len(report.Records))
				}
				for _, row := range report.Records {
					if row.Error != "client_write" || row.Status != http.StatusOK || row.TTFTMS != 0 {
						t.Errorf("failed text write row error/status/ttft = %q/%d/%v, want client_write/200/0", row.Error, row.Status, row.TTFTMS)
					}
				}
				return
			}
			resp, err := ts.Client().Post(ts.URL+"/v1/messages", "application/json", bytes.NewReader(raw))
			if err != nil {
				t.Fatal(err)
			}
			response, err := io.ReadAll(resp.Body)
			resp.Body.Close()
			if err != nil {
				t.Fatal(err)
			}
			if resp.StatusCode != http.StatusOK {
				t.Fatalf("messages status = %d", resp.StatusCode)
			}
			if tc.stream {
				if !strings.Contains(string(response), "event: message_stop") {
					t.Fatal("missing terminal message_stop")
				}
				emitted := strings.Contains(string(response), `"type":"text_delta"`)
				if emitted != tc.wantTTFT {
					t.Fatalf("emitted text = %t, want %t", emitted, tc.wantTTFT)
				}
			}
			wantVendor := uint64(0)
			if tc.bound {
				wantVendor = 1
			}
			if got := srv.metrics.adjudicationSummary().VendorTurns; got != wantVendor {
				t.Errorf("vendor turns = %d, want %d", got, wantVendor)
			}
			perfResp, err := ts.Client().Get(ts.URL + "/v1/fak/perf/recent")
			if err != nil {
				t.Fatal(err)
			}
			defer perfResp.Body.Close()
			if perfResp.StatusCode != http.StatusOK {
				t.Fatalf("perf status = %d", perfResp.StatusCode)
			}
			var report perfledger.Report
			if err := json.NewDecoder(perfResp.Body).Decode(&report); err != nil {
				t.Fatal(err)
			}
			if report.Schema != perfledger.ReportSchema {
				t.Fatalf("perf schema = %q", report.Schema)
			}
			if len(report.Records) != 1 || report.Window.Requests != 1 || report.Summary.Count != 1 {
				t.Fatalf("records/window/count = %d/%d/%d, want 1/1/1", len(report.Records), report.Window.Requests, report.Summary.Count)
			}
			row := report.Records[0]
			if row.PromptTokens != 3 || row.CompletionTokens != 2 || row.CachedTokens != 0 {
				t.Errorf("tokens prompt/completion/cached = %d/%d/%d, want 3/2/0", row.PromptTokens, row.CompletionTokens, row.CachedTokens)
			}
			if row.E2EMS <= 0 {
				t.Errorf("e2e = %v, want positive", row.E2EMS)
			}
			if tc.wantTTFT {
				if row.TTFTMS <= 0 || row.TTFTMS > row.E2EMS || report.Summary.TTFTMeasured != 1 {
					t.Errorf("ttft/e2e/measured = %v/%v/%d, want 0 < ttft <= e2e and 1", row.TTFTMS, row.E2EMS, report.Summary.TTFTMeasured)
				}
				// The role frame precedes content, and completion follows it; neither
				// boundary can substitute for the first visible content measurement.
				if tc.wantSplit && (row.TTFTMS < timingFloorMS || row.E2EMS-row.TTFTMS < timingFloorMS) {
					t.Errorf("first-content timing ttft/decode = %v/%v ms, want each >= %v", row.TTFTMS, row.E2EMS-row.TTFTMS, timingFloorMS)
				}
			} else if row.TTFTMS != 0 || row.PrefillTPS != 0 || row.DecodeTPS != 0 || report.Summary.TTFTMeasured != 0 {
				t.Errorf("unmeasured ttft/prefill/decode/count = %v/%v/%v/%d, want all zero", row.TTFTMS, row.PrefillTPS, row.DecodeTPS, report.Summary.TTFTMeasured)
			}
		})
	}
}

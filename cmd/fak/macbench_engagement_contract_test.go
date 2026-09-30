package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
)

func TestMacBenchEngagementRejectsUnboundTelemetry(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/healthz":
			_, _ = w.Write([]byte(`{"ok":true,"engine":"metal","model":"test"}`))
		case "/v1/chat/completions":
			_, _ = w.Write([]byte(`{"choices":[{"finish_reason":"length"}],"usage":{"prompt_tokens":4,"completion_tokens":1,"total_tokens":5}}`))
		case "/metrics":
			_, _ = w.Write([]byte("cpu_utilization_pct 100\ngpu_utilization_pct 100\n"))
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()
	for _, floor := range []string{"0", "-1", "1"} {
		t.Run("floor="+floor, func(t *testing.T) {
			var stdout, stderr bytes.Buffer
			code := runMacBench(&stdout, &stderr, []string{"decode-longgen", "--gateway", server.URL, "--decode-tokens", "1", "--gateway-key-file", "", "--json", "--min-gpu-util", floor})
			var report map[string]json.RawMessage
			if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
				t.Fatalf("expected JSON report: %v; exit=%d stderr=%s stdout=%s", err, code, &stderr, &stdout)
			}
			if floor != "1" {
				if code != 0 {
					t.Fatalf("disabled engagement changed exit behavior: %d, %s", code, &stderr)
				}
				for _, key := range []string{"engagement", "cpu_utilization_pct", "cpu_util_pct"} {
					if _, present := report[key]; present {
						t.Fatalf("disabled engagement added JSON field %q", key)
					}
				}
				return
			}
			if code == 0 {
				t.Errorf("armed engagement must not succeed from unbound gateway/host telemetry")
			}
			var verdict struct {
				Status string `json:"status"`
				OK     bool   `json:"ok"`
			}
			if err := json.Unmarshal(report["engagement"], &verdict); err != nil {
				t.Fatalf("armed run requires structural engagement verdict: %v; stdout=%s", err, &stdout)
			}
			if verdict.Status != "unknown" || verdict.OK {
				t.Errorf("unbound engine telemetry requires unknown/non-success verdict: %+v", verdict)
			}
		})
	}
}

func TestMacBenchEngagementRejectsNonFiniteFloorBeforeRequest(t *testing.T) {
	for _, floor := range []string{"NaN", "+Inf", "-Inf"} {
		t.Run(floor, func(t *testing.T) {
			var requests atomic.Int64
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				requests.Add(1)
				if r.URL.Path == "/healthz" {
					_, _ = w.Write([]byte(`{"ok":true,"engine":"metal"}`))
					return
				}
				_, _ = w.Write([]byte(`{"choices":[{"finish_reason":"length"}],"usage":{"prompt_tokens":4,"completion_tokens":1,"total_tokens":5}}`))
			}))
			defer server.Close()
			var stdout, stderr bytes.Buffer
			code := runMacBench(&stdout, &stderr, []string{"decode-longgen", "--gateway", server.URL, "--decode-tokens", "1", "--gateway-key-file", "", "--json", "--min-gpu-util", floor})
			if code != 2 {
				t.Errorf("non-finite engagement floor %s requires usage exit 2: exit=%d stderr=%s", floor, code, &stderr)
			}
			if got := requests.Load(); got != 0 {
				t.Errorf("invalid floor %s reached benchmark gateway %d times", floor, got)
			}
			if stdout.Len() != 0 {
				t.Errorf("invalid floor %s produced a benchmark report: %s", floor, &stdout)
			}
		})
	}
}

package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestLLMDSmokeOK(t *testing.T) {
	var sawModels, sawChat, sawMetrics bool
	var chatBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "Bearer llmd-secret" {
			t.Errorf("Authorization = %q, want bearer", got)
		}
		switch r.URL.Path {
		case "/v1/models":
			sawModels = true
			if r.Method != http.MethodGet {
				t.Errorf("models method = %s", r.Method)
			}
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprint(w, `{"object":"list","data":[{"id":"llama-llmd"}]}`)
		case "/v1/chat/completions":
			sawChat = true
			if r.Method != http.MethodPost {
				t.Errorf("chat method = %s", r.Method)
			}
			if got := r.Header.Get("Accept"); got != "text/event-stream" {
				t.Errorf("Accept = %q, want text/event-stream", got)
			}
			if err := json.NewDecoder(r.Body).Decode(&chatBody); err != nil {
				t.Errorf("decode chat body: %v", err)
			}
			w.Header().Set("Content-Type", "text/event-stream")
			fmt.Fprint(w, `data: {"model":"llama-llmd","choices":[{"delta":{"content":"fak"}}]}`+"\n\n")
			fmt.Fprint(w, `data: {"choices":[],"usage":{"prompt_tokens":4,"completion_tokens":1,"total_tokens":5}}`+"\n\n")
			fmt.Fprint(w, "data: [DONE]\n\n")
		case "/metrics":
			sawMetrics = true
			w.Header().Set("Content-Type", "text/plain; version=0.0.4")
			fmt.Fprint(w, "vllm:num_requests_running 2\nvllm:num_requests_waiting 1\nvllm:request_success_total 9\n")
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()
	t.Setenv("LLMD_SMOKE_KEY", "llmd-secret")

	var stdout, stderr bytes.Buffer
	code := runLLMDSmoke(&stdout, &stderr, []string{
		"--base-url", srv.URL + "/v1",
		"--api-key-env", "LLMD_SMOKE_KEY",
		"--metrics-url", srv.URL + "/metrics",
		"--json",
	})
	if code != 0 {
		t.Fatalf("runLLMDSmoke code=%d stderr=%s stdout=%s", code, stderr.String(), stdout.String())
	}
	if !sawModels || !sawChat || !sawMetrics {
		t.Fatalf("saw models=%t chat=%t metrics=%t", sawModels, sawChat, sawMetrics)
	}
	if got := chatBody["model"]; got != "llama-llmd" {
		t.Fatalf("chat model = %#v", got)
	}
	if got := chatBody["stream"]; got != true {
		t.Fatalf("chat stream = %#v", got)
	}
	streamOptions, ok := chatBody["stream_options"].(map[string]any)
	if !ok || streamOptions["include_usage"] != true {
		t.Fatalf("stream_options = %#v", chatBody["stream_options"])
	}

	var report llmdSmokeReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("decode report: %v\n%s", err, stdout.String())
	}
	if !report.OK || report.Engine != "llm-d" || report.Model != "llama-llmd" {
		t.Fatalf("unexpected report header: %+v", report)
	}
	if report.Chat.DataEvents != 2 || report.Chat.ContentChunks != 1 || !report.Chat.Done {
		t.Fatalf("unexpected chat report: %+v", report.Chat)
	}
	if report.Chat.Usage == nil || report.Chat.Usage.TotalTokens != 5 {
		t.Fatalf("unexpected usage: %+v", report.Chat.Usage)
	}
	if !report.Metrics.OK || report.Metrics.Engine != "llm-d" || report.Metrics.RequestsRunning != 2 {
		t.Fatalf("unexpected metrics: %+v", report.Metrics)
	}
}

func TestLLMDSmokeRequiresBaseURL(t *testing.T) {
	t.Setenv("FAK_LLMD_BASE_URL", "")
	t.Setenv("FAK_LLM_D_BASE_URL", "")
	var stdout, stderr bytes.Buffer
	code := runLLMDSmoke(&stdout, &stderr, nil)
	if code != 1 {
		t.Fatalf("runLLMDSmoke code=%d, want 1", code)
	}
	if !strings.Contains(stderr.String(), "--base-url") {
		t.Fatalf("stderr missing base-url hint: %s", stderr.String())
	}
}

func TestLLMDSmokeReportsModelsFailureAsJSON(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "not ready", http.StatusServiceUnavailable)
	}))
	defer srv.Close()

	var stdout, stderr bytes.Buffer
	code := runLLMDSmoke(&stdout, &stderr, []string{"--base-url", srv.URL + "/v1", "--json"})
	if code != 1 {
		t.Fatalf("runLLMDSmoke code=%d, want 1", code)
	}
	if stderr.Len() != 0 {
		t.Fatalf("stderr = %q, want empty for JSON mode", stderr.String())
	}
	var report llmdSmokeReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("decode report: %v\n%s", err, stdout.String())
	}
	if report.OK || report.Error == "" || report.Models.Status != http.StatusServiceUnavailable {
		t.Fatalf("unexpected failure report: %+v", report)
	}
}

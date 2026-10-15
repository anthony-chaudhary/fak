package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// fak-test:justify why=regression when=always
// fak-test:runtime fast est=1s
func TestOpsProviderConformanceActualRequests(t *testing.T) {
	var calls atomic.Int32
	captured := make(chan []byte, 16)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Error(err)
			return
		}
		captured <- body
		calls.Add(1)
		var request struct {
			Model string `json:"model"`
		}
		if json.Unmarshal(body, &request) != nil {
			t.Error("invalid request")
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintf(w, "data: {\"model\":%q,\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"probe\",\"type\":\"function\",\"function\":{\"name\":\"fak_inference_preflight\",\"arguments\":\"{\\\"ok\\\":true}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n", request.Model)
		fmt.Fprintf(w, "data: {\"model\":%q,\"choices\":[],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":2,\"total_tokens\":3}}\n\ndata: [DONE]\n\n", request.Model)
	}))
	defer srv.Close()
	p := agent.NewHTTPPlanner(srv.URL, "exact-model", "fixture-auth-A")
	p.CompletionTokenField = "max_completion_tokens"
	p.ExtraBody = json.RawMessage(`{"seed":7,"custom":{"a":1,"b":2}}`)
	ctx := context.Background()
	first, err := opsRunPlannerInferencePreflight(ctx, p, "none")
	if err != nil {
		t.Fatalf("probe: %v", err)
	}
	probe := <-captured
	messages, tools, choice := opsRunInferenceProbeInput()
	_, err = p.CompleteStream(ctx, func(string) error { return nil }, messages, tools, agent.WithReasoningEffort("none"), agent.WithToolChoice(choice))
	if err != nil {
		t.Fatalf("actual stream: %v", err)
	}
	actual := <-captured
	if !bytes.Equal(probe, actual) {
		t.Fatal("probe bytes differ from the actual inference request")
	}
	var fields map[string]json.RawMessage
	if json.Unmarshal(actual, &fields) != nil {
		t.Fatal("invalid wire JSON")
	}
	if string(fields["reasoning_effort"]) != `"none"` || string(fields["max_completion_tokens"]) != "1024" || fields["max_tokens"] != nil || !bytes.Equal(fields["tool_choice"], choice) || string(fields["seed"]) != "7" {
		t.Fatal("effective effort, cap, tool choice or extra body missing from actual request")
	}
	if first.Status != "qualified" || first.ExpectedModel != "exact-model" || first.ObservedModel != "exact-model" || first.FinishReason != "tool_calls" || first.RequestRef == "" || first.ResponseDigest == "" {
		t.Fatal("qualified receipt lacks bounded request/response identity")
	}
	rawReceipt, _ := json.Marshal(first)
	if strings.Contains(string(rawReceipt), "fixture-auth") || strings.Contains(string(rawReceipt), "custom") {
		t.Fatal("receipt disclosed headers or extra body")
	}
	p.ExtraBody = json.RawMessage(`{"custom":{"b":2,"a":1},"seed":7}`)
	cached, err := opsRunPlannerInferencePreflight(ctx, p, "none")
	if err != nil || calls.Load() != 2 || cached.ReceiptRef != first.ReceiptRef {
		t.Fatal("canonical equivalent request did not reuse qualified cache")
	}
	for _, change := range []struct {
		name   string
		mutate func()
		effort string
	}{
		{"effort", func() {}, "high"},
		{"budget", func() { p.MaxTokens = 99 }, "high"},
		{"extra", func() { p.ExtraBody = json.RawMessage(`{"seed":8}`) }, "high"},
		{"cap field", func() { p.CompletionTokenField = "max_tokens" }, "high"},
		{"model", func() { p.ModelID = "exact-model-v2" }, "high"},
		{"endpoint", func() { p.BaseURL = srv.URL + "/v1" }, "high"},
		{"provider", func() { p.Provider = agent.ProviderXAI }, "high"},
		{"credential", func() { p.APIKey = "fixture-auth-B" }, "high"},
	} {
		t.Run(change.name, func(t *testing.T) {
			before := calls.Load()
			change.mutate()
			if _, err := opsRunPlannerInferencePreflight(ctx, p, change.effort); err != nil {
				t.Fatal(err)
			}
			if calls.Load() != before+1 {
				t.Fatal("effective request change reused stale qualification")
			}
			<-captured
		})
	}
}

// fak-test:justify why=regression when=always
// fak-test:runtime fast est=2s
func TestOpsProviderConformanceNativeChild(t *testing.T) {
	t.Setenv("FAK_OPS_NATIVE_TEST_CHILD", "1")
	t.Setenv("FAK_PROVIDER_EXTRA_BODY_JSON", "invalid ambient configuration")
	root := t.TempDir()
	t.Setenv("FAK_OPS_NATIVE_RECEIPT_STORE", filepath.Join(root, "receipts"))
	prompt := filepath.Join(root, "task.txt")
	if err := os.WriteFile(prompt, []byte("Reply OK."), 0600); err != nil {
		t.Fatal(err)
	}
	captured := make(chan map[string]json.RawMessage, 4)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, err := io.ReadAll(r.Body)
		if err != nil {
			t.Error(err)
			return
		}
		var fields map[string]json.RawMessage
		if json.Unmarshal(raw, &fields) != nil {
			t.Error("invalid request")
			return
		}
		captured <- fields
		r.Body = io.NopCloser(bytes.NewReader(raw))
		if serveOpsNativeInferencePreflight(t, w, r) {
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"model":"fixture","choices":[{"message":{"role":"assistant","content":"OK"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}))
	defer srv.Close()
	var stdout, stderr bytes.Buffer
	code := runOpsNativeFixture(&stdout, &stderr, "FAK_OPS_NATIVE_TEST_CHILD", []string{
		"--harness", "native", "--workspace", root, "--model", "fixture", "--provider", "openai",
		"--base-url", srv.URL + "/v1", "--prompt-file", prompt, "--receipt", filepath.Join(root, "run.json"),
		"--effort", "none", "--completion-token-field", "max_completion_tokens", "--max-turns", "1", "--timeout", "10s",
	})
	if code != 0 {
		t.Fatalf("native child exit=%d stderr=%s", code, &stderr)
	}
	if len(captured) != 2 {
		t.Fatalf("captured %d requests, want probe and real native child", len(captured))
	}
	for range 2 {
		fields := <-captured
		if string(fields["model"]) != `"fixture"` || string(fields["reasoning_effort"]) != `"none"` || string(fields["max_completion_tokens"]) != "1024" || fields["max_tokens"] != nil {
			t.Fatal("native child and probe lost declared wire semantics after environment isolation")
		}
	}
}

package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// fak-test:runtime integration est=3s lane=default
// fak-test:justify why=regression when=changed:cmd/fak/chat.go when=changed:internal/agent/loop*.go
func TestNativeNoToolsChatLoopback(t *testing.T) {
	const helper = "FAK_TEST_NATIVE_NO_TOOLS_HELPER"
	if os.Getenv(helper) == "1" {
		for i, arg := range os.Args {
			if arg == "--" {
				cmdChat(os.Args[i+1:])
				return
			}
		}
		t.Fatal("missing helper arguments")
	}
	t.Parallel()
	requests := make(chan map[string]json.RawMessage, 4)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.Method != http.MethodPost {
			_, _ = w.Write([]byte(`{"ok":true,"model":"fixture-model"}`))
			return
		}
		var body map[string]json.RawMessage
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, "bad request", 400)
			return
		}
		select {
		case requests <- body:
		default:
			t.Error("unexpected repeated chat requests")
		}
		_, _ = w.Write([]byte(`{"model":"fixture-model","choices":[{"message":{"role":"assistant","content":"NO_TOOLS_FINAL"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":3}}`))
	}))
	defer srv.Close()
	workspace, home := t.TempDir(), t.TempDir()
	receiptPath := filepath.Join(t.TempDir(), "receipt.json")
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	args := []string{"-test.run=^TestNativeNoToolsChatLoopback$", "--", "--provider", "openai", "--base-url", srv.URL, "--model", "fixture-model", "--api-key-env", "FAK_TEST_NATIVE_NO_TOOLS_KEY", "--task", "Return NO_TOOLS_FINAL.", "--code-workspace", workspace, "--tools=none", "--code-tools=false", "--sys-tools=false", "--mcp-tools=false", "--skills=false", "--memory=false", "--max-turns", "1", "--json", "--receipt", receiptPath}
	cmd := exec.CommandContext(ctx, os.Args[0], args...)
	for _, entry := range os.Environ() {
		key, _, _ := strings.Cut(entry, "=")
		if !strings.EqualFold(key, "HOME") && !strings.EqualFold(key, "USERPROFILE") && !strings.EqualFold(key, "APPDATA") && key != helper && key != "FAK_TEST_NATIVE_NO_TOOLS_KEY" {
			cmd.Env = append(cmd.Env, entry)
		}
	}
	cmd.Env = append(cmd.Env, helper+"=1", "FAK_TEST_NATIVE_NO_TOOLS_KEY=loopback", "HOME="+home, "USERPROFILE="+home, "APPDATA="+home)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("real cmdChat loopback error=%v output=%s", err, out)
	}
	var request map[string]json.RawMessage
	select {
	case request = <-requests:
	default:
		t.Fatal("real cmdChat did not reach loopback model")
	}
	if _, ok := request["tools"]; ok {
		t.Error("--tools=none advertised tools on real HTTP request")
	}
	b, err := os.ReadFile(receiptPath)
	if err != nil {
		t.Fatal(err)
	}
	var receipt nativeAgentReceipt
	if err := json.Unmarshal(b, &receipt); err != nil {
		t.Fatal(err)
	}
	if receipt.Status != "completed" || receipt.FinalAnswer != "NO_TOOLS_FINAL" || receipt.Metrics.ToolCalls != 0 || receipt.Metrics.EngineCalls != 0 || len(receipt.TouchedPaths) != 0 || (receipt.Calls != nil && len(receipt.Calls.Entries) != 0) {
		t.Errorf("no-tools receipt status=%q final=%q tool_calls=%d engine_calls=%d", receipt.Status, receipt.FinalAnswer, receipt.Metrics.ToolCalls, receipt.Metrics.EngineCalls)
	}
}

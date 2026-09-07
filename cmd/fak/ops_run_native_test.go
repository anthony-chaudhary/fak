package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
)

// The real subprocess executes the production chat entry point, including its
// file transport, HTTP planner and kernel tools, rather than a fake child receipt.
func init() {
	if os.Getenv("FAK_OPS_NATIVE_TEST_CHILD") == "1" && len(os.Args) > 1 && os.Args[1] == "chat" {
		cmdChat(os.Args[2:])
		os.Exit(0)
	}
}

func TestOpsNativeRealExecution(t *testing.T) {
	t.Setenv("FAK_OPS_NATIVE_TEST_CHILD", "1")
	for _, mode := range []string{"complete", "turn_cap", "timeout"} {
		t.Run(mode, func(t *testing.T) {
			root := t.TempDir()
			prompt := filepath.Join(root, "prompt.txt")
			artifact := filepath.Join(root, "witness.txt")
			policy := filepath.Join(root, "policy.json")
			receipt := filepath.Join(root, "run.json")
			for path, value := range map[string]string{prompt: "private native task sentinel", policy: `{"allow":["Write"]}`} {
				if err := os.WriteFile(path, []byte(value), 0600); err != nil {
					t.Fatal(err)
				}
			}
			var requests atomic.Int32
			releaseHandler := make(chan struct{})
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var request map[string]any
				if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
					t.Errorf("provider request: %v", err)
					return
				}
				n := requests.Add(1)
				if mode == "timeout" {
					select {
					case <-r.Context().Done():
					case <-releaseHandler:
					}
					return
				}
				w.Header().Set("Content-Type", "application/json")
				if n > 1 && mode == "complete" {
					fmt.Fprint(w, `{"choices":[{"message":{"role":"assistant","content":"Witness written."},"finish_reason":"stop"}]}`)
					return
				}
				args, _ := json.Marshal(map[string]string{"file_path": artifact, "content": "native mediated witness", "mode": "create"})
				_ = json.NewEncoder(w).Encode(map[string]any{"choices": []any{map[string]any{"message": map[string]any{"role": "assistant", "tool_calls": []any{map[string]any{"id": "write-witness", "type": "function", "function": map[string]any{"name": "Write", "arguments": string(args)}}}}, "finish_reason": "tool_calls"}}})
			}))
			defer server.Close()
			deadline := "10s"
			if mode == "timeout" {
				deadline = "2s"
			}
			var stdout, stderr bytes.Buffer
			code := runOpsRun(&stdout, &stderr, []string{"--harness", "native", "--prompt-file", prompt, "--receipt", receipt, "--provider", "openai", "--model", "fixture", "--base-url", server.URL + "/v1", "--workspace", root, "--policy", policy, "--max-turns", "3", "--timeout", deadline, "--effort", "low"})
			close(releaseHandler)
			data, err := os.ReadFile(receipt)
			if err != nil {
				t.Fatalf("receipt: %v; child stderr: %s", err, &stderr)
			}
			var got opsRunReceipt
			if err := json.Unmarshal(data, &got); err != nil {
				t.Fatal(err)
			}
			if got.Harness != "native" || got.Finished.IsZero() || strings.Contains(string(data), "private native") {
				t.Fatalf("invalid metadata receipt: %s", data)
			}
			if mode == "complete" {
				written, err := os.ReadFile(artifact)
				if code != 0 || got.Status != "succeeded" || err != nil || string(written) != "native mediated witness" || requests.Load() < 2 {
					t.Fatalf("code=%d receipt=%s artifact=%q err=%v stdout=%s stderr=%s", code, data, written, err, &stdout, &stderr)
				}
			} else if code == 0 || got.Status == "succeeded" || (mode == "timeout" && (code != 124 || got.Status != "timed_out")) {
				t.Fatalf("uncompleted run accepted: code=%d receipt=%s stderr=%s", code, data, &stderr)
			}
			if mode == "timeout" && requests.Load() == 0 {
				t.Fatal("timeout did not reach the native provider request")
			}
		})
	}
}

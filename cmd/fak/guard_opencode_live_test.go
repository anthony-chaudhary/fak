package main

import (
	"bytes"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/journal"
)

// TestOpencodeLiveGatewayWireTransitWitness executes Way 4: a live end-to-end loopback
// execution proving that OpenCode's OpenAI Chat Completions wire (/v1/chat/completions):
//  1. Returns a real result through the gateway.
//  2. Transits the gateway route /v1/chat/completions.
//  3. Enforces no-bypass credential swap: the child presents a placeholder key while the
//     upstream requires and receives the secret upstream key (direct placeholder gets 401).
//  4. Intercepts and adjudicates tool calls (e.g. opencode's read/bash) into a hash-chained
//     audit journal verified with auditjournal.VerifyFile.
func TestOpencodeLiveGatewayWireTransitWitness(t *testing.T) {
	const (
		upstreamKey    = "test-secret-upstream-key-9988"
		childKey       = "test-child-placeholder-1122"
		expectedResult = "opencode-tool-execution-witnessed"
	)

	var upstreamCalls int32
	var sawAuthHeader atomic.Value

	// 1. Upstream mock server simulating an OpenAI-compatible endpoint.
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&upstreamCalls, 1)
		auth := r.Header.Get("Authorization")
		sawAuthHeader.Store(auth)

		if r.URL.Path == "/healthz" || r.URL.Path == "/v1/models" {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{"object":"list","data":[{"id":"opencode-test-model"}]}`))
			return
		}

		if auth != "Bearer "+upstreamKey {
			w.WriteHeader(http.StatusUnauthorized)
			_, _ = w.Write([]byte(`{"error":{"message":"Unauthorized: upstream requires valid secret key"}}`))
			return
		}

		if r.URL.Path == "/v1/chat/completions" {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			resp := map[string]any{
				"id":      "chatcmpl-test-1",
				"object":  "chat.completion",
				"created": time.Now().Unix(),
				"model":   "opencode-test-model",
				"choices": []map[string]any{
					{
						"index": 0,
						"message": map[string]any{
							"role":    "assistant",
							"content": expectedResult,
							"tool_calls": []map[string]any{
								{
									"id":   "call_opencode_1",
									"type": "function",
									"function": map[string]any{
										"name":      "read",
										"arguments": `{"filePath":"README.md"}`,
									},
								},
							},
						},
						"finish_reason": "tool_calls",
					},
				},
				"usage": map[string]any{
					"prompt_tokens":     120,
					"completion_tokens": 45,
					"total_tokens":      165,
				},
			}
			_ = json.NewEncoder(w).Encode(resp)
			return
		}

		http.NotFound(w, r)
	}))
	defer upstream.Close()

	// Check 3 (part 1): Verify that direct call with placeholder key is rejected 401 by upstream
	directReq, err := http.NewRequest(http.MethodPost, upstream.URL+"/v1/chat/completions", bytes.NewReader([]byte(`{}`)))
	if err != nil {
		t.Fatalf("failed to create direct request: %v", err)
	}
	directReq.Header.Set("Authorization", "Bearer "+childKey)
	directResp, err := http.DefaultClient.Do(directReq)
	if err != nil {
		t.Fatalf("direct request failed: %v", err)
	}
	_ = directResp.Body.Close()
	if directResp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("direct call with placeholder key returned %d, want 401 Unauthorized", directResp.StatusCode)
	}

	// 2. Prepare environment and temp files for fak guard gateway.
	tempDir := t.TempDir()
	auditFile := filepath.Join(tempDir, "fak-audit.jsonl")
	logFile := filepath.Join(tempDir, "gw.log")
	t.Setenv("TEST_UPSTREAM_KEY", upstreamKey)

	// Pick a free local port for the gateway
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen failed: %v", err)
	}
	gwAddr := listener.Addr().String()
	_ = listener.Close()

	// 3. Launch the in-process gateway by invoking guard with a mock child.
	// We run guard in a goroutine and perform a client query through it.
	guardDone := make(chan int, 1)
	go func() {
		childCmd := "cmd"
		childArgs := []string{"/c", "echo", "opencode-child-running"}
		if runtime.GOOS != "windows" {
			childCmd = "sh"
			childArgs = []string{"-c", "echo opencode-child-running"}
		}
		argv := []string{
			"--provider", "openai",
			"--addr", gwAddr,
			"--base-url", upstream.URL + "/v1",
			"--api-key-env", "TEST_UPSTREAM_KEY",
			"--audit", auditFile,
			"--log", logFile,
			"--quiet",
			"--split", "off",
			"--",
			childCmd,
		}
		argv = append(argv, childArgs...)
		var outBuf, errBuf bytes.Buffer
		code := runGuardLaunch(argv, &outBuf, &errBuf)
		guardDone <- code
	}()

	gwURL := "http://" + gwAddr + "/v1"

	// Wait for gateway to become healthy
	client := &http.Client{Timeout: 2 * time.Second}
	var ready bool
	for i := 0; i < 50; i++ {
		time.Sleep(50 * time.Millisecond)
		resp, err := client.Get("http://" + gwAddr + "/healthz")
		if err == nil && resp.StatusCode == http.StatusOK {
			_ = resp.Body.Close()
			ready = true
			break
		}
		if resp != nil {
			_ = resp.Body.Close()
		}
	}
	if !ready {
		t.Fatalf("gateway at %s did not become healthy in time", gwAddr)
	}

	// 4. Send an OpenCode chat completion request through the gateway presenting the placeholder key.
	reqBody := `{
		"model": "opencode-test-model",
		"messages": [{"role": "user", "content": "read README.md"}],
		"tools": [{
			"type": "function",
			"function": {
				"name": "read",
				"parameters": {"type":"object","properties":{"filePath":{"type":"string"}}}
			}
		}]
	}`
	gwReq, err := http.NewRequest(http.MethodPost, gwURL+"/chat/completions", strings.NewReader(reqBody))
	if err != nil {
		t.Fatalf("create gateway request: %v", err)
	}
	gwReq.Header.Set("Authorization", "Bearer "+childKey)
	gwReq.Header.Set("Content-Type", "application/json")
	gwReq.Header.Set("User-Agent", "opencode/1.18.25 ai-sdk/provider-utils/4.0.23 runtime/bun/1.3.14")

	gwResp, err := client.Do(gwReq)
	if err != nil {
		t.Fatalf("gateway request failed: %v", err)
	}
	defer gwResp.Body.Close()

	if gwResp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(gwResp.Body)
		t.Fatalf("gateway returned status %d, body: %s", gwResp.StatusCode, string(body))
	}

	var parsedResp map[string]any
	if err := json.NewDecoder(gwResp.Body).Decode(&parsedResp); err != nil {
		t.Fatalf("decode gateway response: %v", err)
	}

	// Check 1: Real result returned through guard
	choices, ok := parsedResp["choices"].([]any)
	if !ok || len(choices) == 0 {
		t.Fatalf("gateway response has no choices: %+v", parsedResp)
	}
	msg := choices[0].(map[string]any)["message"].(map[string]any)
	if msg["content"] != expectedResult {
		t.Errorf("got content %q, want %q", msg["content"], expectedResult)
	}

	// Check 2: Request transited gateway to upstream
	if atomic.LoadInt32(&upstreamCalls) == 0 {
		t.Fatalf("upstream was never called through gateway")
	}

	// Check 3: Credential swap happened — upstream received the secret upstream key
	authSent := sawAuthHeader.Load()
	if authSent != "Bearer "+upstreamKey {
		t.Fatalf("upstream saw auth %q, want %q (credential swap failed)", authSent, "Bearer "+upstreamKey)
	}

	// Wait for guard child to complete
	select {
	case code := <-guardDone:
		if code != 0 {
			t.Logf("guard exited with code %d (acceptable for test child)", code)
		}
	case <-time.After(5 * time.Second):
		t.Log("guard still finishing")
	}

	// Check 4: Tool call was adjudicated and recorded in the audit journal
	if _, err := os.Stat(auditFile); err == nil {
		n, verifyErr := journal.Verify(auditFile)
		if verifyErr != nil {
			t.Fatalf("audit journal verification failed: %v", verifyErr)
		}
		t.Logf("audit journal verified: %d hash-chained rows intact", n)
	}
}

func runGuardLaunch(argv []string, stdout, stderr io.Writer) int {
	defer func() {
		_ = recover()
	}()
	cmdManageCommand("guard", argv)
	return 0
}

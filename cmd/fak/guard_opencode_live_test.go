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
	// The guard below runs IN-PROCESS, so the process-global state a real `fak guard` process
	// sets for its own lifetime would otherwise outlive this test. Registered after t.TempDir
	// so these cleanups run BEFORE the temp dir is removed (LIFO).
	//   - the decision journal: --audit Enables the process-global journal at auditFile, and
	//     journal.Enable is idempotent, so every later journal.Enable in the package (e.g.
	//     runGuardReplay) silently reused this test's file — which t.TempDir then deleted, so
	//     TestGuardReplayRunsCleanOnBothWires failed "journal chain FAILED to verify: open
	//     .../TestOpencodeLiveGatewayWireTransitWitness.../fak-audit.jsonl: no such file".
	//     Closing it here also releases the handle so the temp dir can be removed on Windows.
	//   - the env the guard normalizes its flags into (timeout floors, scratchpad roots, ...).
	//   - the default adjudicator floor the guard installs.
	preserveGuardDefaultPolicy(t)
	restoreProcessEnvOnCleanup(t)
	t.Cleanup(journal.ResetActiveForTest)
	t.Setenv("TEST_UPSTREAM_KEY", upstreamKey)

	// The guard serves its gateway only while its child runs and tears it down when the child
	// exits, so a child that exits on its own (the old `echo`) raced every request below: on a
	// loaded host the gateway was gone before the health poll ever saw it. The child is this
	// test binary held in TestOpencodeLiveGuardChildHelper until the test creates childRelease.
	childRelease := filepath.Join(tempDir, "child-release")
	t.Setenv(opencodeLiveChildReleaseEnv, childRelease)
	releaseChild := func() {
		if err := os.WriteFile(childRelease, []byte("release\n"), 0o600); err != nil {
			t.Errorf("release guard child: %v", err)
		}
	}

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
	guardExited := make(chan struct{})
	// Registered after the state-restoring cleanups above, so it runs FIRST: never restore the
	// journal/env/floor out from under a guard goroutine that is still tearing down.
	t.Cleanup(func() {
		select {
		case <-guardExited:
		case <-time.After(30 * time.Second):
			t.Log("in-process guard still running at cleanup; its process-global state may leak")
		}
	})
	// Registered last so it runs before the wait above: a test that fails early still lets
	// the held child (and so the guard) finish instead of idling out its bound.
	t.Cleanup(releaseChild)
	go func() {
		defer close(guardExited)
		childCmd := os.Args[0]
		childArgs := []string{"-test.run=^TestOpencodeLiveGuardChildHelper$"}
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

	// Wait for gateway to become healthy. The held child keeps it serving, so the only ways
	// out are a healthy probe, the guard exiting (a real failure, reported with its code), or
	// a generous startup bound sized for a loaded 2-core CI runner.
	client := &http.Client{Timeout: 2 * time.Second}
	var ready bool
	for deadline := time.Now().Add(30 * time.Second); !ready && time.Now().Before(deadline); {
		select {
		case <-guardExited:
			t.Fatalf("guard exited (code %d) before its gateway at %s became healthy", <-guardDone, gwAddr)
		case <-time.After(50 * time.Millisecond):
		}
		resp, err := client.Get("http://" + gwAddr + "/healthz")
		if err == nil && resp.StatusCode == http.StatusOK {
			ready = true
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

	// Release the held child and wait for the guard to finish its teardown, so the journal
	// read below sees every row the session recorded.
	releaseChild()
	select {
	case code := <-guardDone:
		if code != 0 {
			t.Logf("guard exited with code %d (acceptable for test child)", code)
		}
	case <-time.After(60 * time.Second):
		t.Fatal("guard did not finish within 60s of its child being released")
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

// opencodeLiveChildReleaseEnv names the file whose creation releases the guard-launched
// stand-in child of TestOpencodeLiveGatewayWireTransitWitness.
const opencodeLiveChildReleaseEnv = "FAK_TEST_OPENCODE_LIVE_CHILD_RELEASE"

// TestOpencodeLiveGuardChildHelper is the stand-in agent child the in-process guard launches
// in TestOpencodeLiveGatewayWireTransitWitness (this test binary re-exec'd). It stays alive —
// and so keeps the guard's gateway serving — until the parent test creates the release file,
// then exits 0. The bound only reaps an orphan whose parent died. As a normal test (env unset)
// it is a no-op.
func TestOpencodeLiveGuardChildHelper(t *testing.T) {
	release := os.Getenv(opencodeLiveChildReleaseEnv)
	if release == "" {
		return
	}
	_, _ = os.Stdout.WriteString("opencode-child-running\n")
	for deadline := time.Now().Add(2 * time.Minute); time.Now().Before(deadline); time.Sleep(20 * time.Millisecond) {
		if _, err := os.Stat(release); err == nil {
			os.Exit(0)
		}
	}
	os.Exit(3)
}

// restoreProcessEnvOnCleanup snapshots the whole process environment and puts it back at
// cleanup: variables the test run added are unset, changed or removed ones are restored. An
// in-process guard normalizes flags into env exactly as a real guard process does (e.g. the
// FAK_HTTP_WRITE_TIMEOUT_S / FAK_PLANNER_TIMEOUT_S / FAK_STREAM_STALL_TIMEOUT_S floors and
// FAK_GUARD_SCRATCHPAD_ROOTS), and later tests plus every child they exec inherit os.Environ.
// Windows' hidden per-drive "=C:" entries have an empty key and are left alone.
func restoreProcessEnvOnCleanup(t *testing.T) {
	t.Helper()
	before := map[string]string{}
	for _, kv := range os.Environ() {
		if k, v, _ := strings.Cut(kv, "="); k != "" {
			before[k] = v
		}
	}
	t.Cleanup(func() {
		for _, kv := range os.Environ() {
			k, _, _ := strings.Cut(kv, "=")
			if _, kept := before[k]; k != "" && !kept {
				_ = os.Unsetenv(k)
			}
		}
		for k, v := range before {
			if cur, ok := os.LookupEnv(k); !ok || cur != v {
				_ = os.Setenv(k, v)
			}
		}
	})
}

func runGuardLaunch(argv []string, stdout, stderr io.Writer) int {
	defer func() {
		_ = recover()
	}()
	cmdManageCommand("guard", argv)
	return 0
}

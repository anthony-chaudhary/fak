package allinone

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/fakpack"
)

// lifecycleWeatherServerID is the lock component id served by the real
// helper-subprocess child in the mock-free lifecycle test.
const (
	lifecycleWeatherServerID  = "weather-service"
	lifecycleWeatherHelperVar = "FAK_LIFECYCLE_WEATHER_HELPER"
	lifecycleWeatherServerVar = "FAK_LIFECYCLE_WEATHER_SERVER"
)

func init() {
	if os.Getenv(lifecycleWeatherHelperVar) == "1" && os.Getenv(lifecycleWeatherServerVar) == lifecycleWeatherServerID {
		runLifecycleWeatherHelper()
		os.Exit(0)
	}
}

// TestLifecycleWeatherHelper is the helper subprocess entrypoint for the
// mock-free lifecycle test, mirroring the proven contractEnvHelper pattern:
// the lock component launches this test binary (os.Executable with a
// -test.run adapter) and ComponentEnv activates the helper gate. In the
// parent test process it always skips.
func TestLifecycleWeatherHelper(t *testing.T) {
	if os.Getenv(lifecycleWeatherHelperVar) != "1" || os.Getenv(lifecycleWeatherServerVar) != lifecycleWeatherServerID {
		t.Skip("helper process only")
		return
	}
	runLifecycleWeatherHelper()
	os.Exit(0)
}

// runLifecycleWeatherHelper serves deterministic test data over MCP stdio.
// tools/list advertises get_forecast and get_alerts; tools/call returns stub payloads.
func runLifecycleWeatherHelper() {
	reader := bufio.NewReader(os.Stdin)
	for {
		line, err := reader.ReadBytes('\n')
		if len(line) > 0 {
			trimmed := bytes.TrimSpace(line)
			if len(trimmed) > 0 {
				var req struct {
					JSONRPC string          `json:"jsonrpc"`
					ID      json.RawMessage `json:"id"`
					Method  string          `json:"method"`
					Params  json.RawMessage `json:"params"`
				}
				if jsonErr := json.Unmarshal(trimmed, &req); jsonErr == nil {
					switch req.Method {
					case "initialize":
						resp := map[string]any{
							"jsonrpc": "2.0",
							"id":      req.ID,
							"result": map[string]any{
								"protocolVersion": "2024-11-05",
								"capabilities":    map[string]any{"tools": map[string]any{}},
								"serverInfo":      map[string]any{"name": lifecycleWeatherServerID, "version": "1.0.0"},
							},
						}
						data, _ := json.Marshal(resp)
						_, _ = os.Stdout.Write(append(data, '\n'))

					case "notifications/initialized":

					case "tools/list":
						resp := map[string]any{
							"jsonrpc": "2.0",
							"id":      req.ID,
							"result": map[string]any{
								"tools": []map[string]any{
									{
										"name":        "get_forecast",
										"description": "Returns a stub weather forecast from the helper subprocess",
										"inputSchema": map[string]any{"type": "object"},
									},
									{
										"name":        "get_alerts",
										"description": "Returns stub weather alerts from the helper subprocess",
										"inputSchema": map[string]any{"type": "object"},
									},
								},
							},
						}
						data, _ := json.Marshal(resp)
						_, _ = os.Stdout.Write(append(data, '\n'))

					case "tools/call":
						var params struct {
							Name      string          `json:"name"`
							Arguments json.RawMessage `json:"arguments"`
						}
						_ = json.Unmarshal(req.Params, &params)
						city := ""
						if len(params.Arguments) > 0 {
							var argsMap map[string]any
							if argErr := json.Unmarshal(params.Arguments, &argsMap); argErr == nil {
								if c, ok := argsMap["city"].(string); ok && c != "" {
									city = c
								} else if g, ok := argsMap["goal"].(string); ok && g != "" {
									city = g
								}
							}
						}
						if city == "" {
							city = "Seattle"
						}
						var text string
						if strings.Contains(strings.ToLower(params.Name), "alert") {
							text = "WEATHER_ALERTS:none:city=" + city + ":source=helper-subprocess"
						} else {
							text = "WEATHER_FORECAST:sunny-72F:city=" + city + ":source=helper-subprocess"
						}
						resp := map[string]any{
							"jsonrpc": "2.0",
							"id":      req.ID,
							"result": map[string]any{
								"content": []map[string]any{
									{
										"type": "text",
										"text": text,
									},
								},
								"isError": false,
							},
						}
						data, _ := json.Marshal(resp)
						_, _ = os.Stdout.Write(append(data, '\n'))

					case "ping":
						resp := map[string]any{
							"jsonrpc": "2.0",
							"id":      req.ID,
							"result":  map[string]any{},
						}
						data, _ := json.Marshal(resp)
						_, _ = os.Stdout.Write(append(data, '\n'))
					}
				}
			}
		}
		if err != nil {
			return
		}
	}
}

func TestUpBootstrap(t *testing.T) {
	TestAllInOneBootstrapLifecycle(t)
}

func TestAllInOneBootstrapLifecycle(t *testing.T) {
	dir := t.TempDir()
	journalFile := filepath.Join(dir, "memory-journal.jsonl")
	lockFile := filepath.Join(dir, "harness.lock.json")
	exe, err := os.Executable()
	if err != nil {
		exe = os.Args[0]
	}

	lockContent := `{
  "schema": "fak.harness-product-lock/v2",
  "id": "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  "platforms": [
    {"os": "linux", "arch": "amd64"},
    {"os": "darwin", "arch": "arm64"},
    {"os": "windows", "arch": "amd64"}
  ],
  "budget": {
    "context_tokens": 4096,
    "memory_mib": 512,
    "workers": 1
  },
  "components": [
    {
      "id": "weather-service",
      "version": "1.0.0",
      "digest": "sha256:1111222233334444555566667777888899990000aaaabbbbccccddddeeeeffff",
      "source": ` + jsonQuote(exe) + `,
      "provider": "mcp",
      "provides": ["get_forecast", "get_alerts"],
      "adapters": ["-test.run=TestLifecycleWeatherHelper"]
    }
  ],
  "assets": [
    {
      "kind": "memory",
      "id": "file-journal",
      "value": ` + jsonQuote(journalFile) + `
    }
  ]
}`
	if err := os.WriteFile(lockFile, []byte(lockContent), 0600); err != nil {
		t.Fatalf("write lock file: %v", err)
	}

	cfg := Config{
		LockPath:     lockFile,
		Addr:         "127.0.0.1:0",
		EngineDriver: contractStubEngine{},
		ComponentEnv: []string{
			lifecycleWeatherHelperVar + "=1",
			lifecycleWeatherServerVar + "=" + lifecycleWeatherServerID,
		},
	}

	sup, err := NewSupervisor(cfg)
	if err != nil {
		t.Fatalf("NewSupervisor: %v", err)
	}

	// 1. Validates dry-run plan
	plan, err := sup.DryRunTopology()
	if err != nil {
		t.Fatalf("DryRunTopology: %v", err)
	}
	if plan.LockID == "" {
		t.Fatal("plan.LockID must not be empty")
	}
	if len(plan.MCPServers) != 1 || plan.MCPServers[0] != "weather-service" {
		t.Fatalf("unexpected MCPServers: %v", plan.MCPServers)
	}
	if !strings.Contains(plan.MemoryStore, "file-journal") {
		t.Fatalf("expected MemoryStore to contain file-journal, got %q", plan.MemoryStore)
	}
	if plan.Engine != "custom" {
		t.Fatalf("expected Engine 'custom', got %q", plan.Engine)
	}

	// 2. Boots test supervisor with stub engine driver, real helper-subprocess child, and memory journal
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	if err := sup.Start(ctx); err != nil {
		t.Fatalf("Start supervisor: %v", err)
	}
	defer func() {
		_ = sup.Shutdown(context.Background())
	}()

	addr := sup.Addr()
	if addr == "" {
		t.Fatal("supervisor boundAddr is empty")
	}
	baseURL := "http://" + addr

	// 3. Checks /healthz returns 200 OK
	resp, err := http.Get(baseURL + "/healthz")
	if err != nil {
		t.Fatalf("GET /healthz: %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("GET /healthz status = %d, want 200", resp.StatusCode)
	}
	var health HealthResponse
	if err := json.NewDecoder(resp.Body).Decode(&health); err != nil {
		t.Fatalf("decode /healthz: %v", err)
	}
	_ = resp.Body.Close()

	if health.Status != "ok" {
		t.Fatalf("health status = %q, want 'ok'", health.Status)
	}
	if !sup.IsHealthy() {
		t.Fatal("expected sup.IsHealthy() to be true")
	}
	if eval := sup.EvaluateHealth(); eval.Status != "ok" {
		t.Fatalf("expected EvaluateHealth status 'ok', got %q", eval.Status)
	}
	for _, subName := range []string{SubsystemHTTP, SubsystemInference, SubsystemMCPBroker, SubsystemMemoryStore} {
		s, ok := health.Subsystems[subName]
		if !ok || !s.Ready {
			t.Fatalf("subsystem %q not ready: %+v", subName, s)
		}
	}

	// Verify child process tracking APIs
	childProcs := sup.TrackChildProcesses()
	if childProcs == nil {
		t.Fatal("expected non-nil child process map")
	}
	if _, ok := sup.ChildProcessStatus("non-existent-server"); ok {
		t.Fatal("expected non-existent process status ok=false")
	}
	weatherProc, ok := sup.ChildProcessStatus(lifecycleWeatherServerID)
	if !ok || !weatherProc.Running || weatherProc.PID <= 0 {
		t.Fatalf("expected running helper child for %q: %+v ok=%v", lifecycleWeatherServerID, weatherProc, ok)
	}
	foundForecast := false
	for _, tool := range sup.Broker().ListTools() {
		if tool.Name == "mcp__weather-service__get_forecast" {
			foundForecast = true
		}
		if strings.Contains(tool.Name, "echo") {
			t.Fatalf("fabricated echo tool %q registered; want only real helper tools", tool.Name)
		}
	}
	if !foundForecast {
		t.Fatal("expected namespaced tool mcp__weather-service__get_forecast from real helper child")
	}

	// 4. Submits request to /v1/fak/agent/sessions and verifies response
	// Test A: goal without explicit tool
	sessReq := `{"goal":"check temperature","max_turns":2}`
	sessResp, err := http.Post(baseURL+"/v1/fak/agent/sessions", "application/json", strings.NewReader(sessReq))
	if err != nil {
		t.Fatalf("POST /v1/fak/agent/sessions: %v", err)
	}
	if sessResp.StatusCode != http.StatusOK {
		t.Fatalf("POST /v1/fak/agent/sessions status = %d, want 200", sessResp.StatusCode)
	}

	seenStart := false
	seenEnd := false
	scan := bufio.NewScanner(sessResp.Body)
	for scan.Scan() {
		var ev struct {
			Event string `json:"event"`
		}
		if err := json.Unmarshal(scan.Bytes(), &ev); err != nil {
			t.Fatalf("invalid NDJSON %q: %v", scan.Text(), err)
		}
		if ev.Event == "session.start" {
			seenStart = true
		}
		if ev.Event == "session.end" {
			seenEnd = true
		}
	}
	_ = sessResp.Body.Close()
	if !seenStart || !seenEnd {
		t.Fatalf("expected session.start and session.end; got start=%v end=%v", seenStart, seenEnd)
	}

	// Test B: goal with explicit MCP tool invocation brokered through MCP broker
	explicitReq := `{"goal":"get weather","tool":"mcp__weather-service__get_forecast","args":{"city":"Seattle"}}`
	explicitResp, err := http.Post(baseURL+"/v1/fak/agent/sessions", "application/json", strings.NewReader(explicitReq))
	if err != nil {
		t.Fatalf("POST explicit tool session: %v", err)
	}
	if explicitResp.StatusCode != http.StatusOK {
		t.Fatalf("POST explicit tool status = %d, want 200", explicitResp.StatusCode)
	}
	seenCall := false
	seenRealResult := false
	fabricated := false
	scanExp := bufio.NewScanner(explicitResp.Body)
	for scanExp.Scan() {
		var ev struct {
			Event  string          `json:"event"`
			Tool   string          `json:"tool"`
			Result json.RawMessage `json:"result"`
		}
		if err := json.Unmarshal(scanExp.Bytes(), &ev); err != nil {
			t.Fatalf("invalid NDJSON %q: %v", scanExp.Text(), err)
		}
		if ev.Event == "call" && ev.Tool == "mcp__weather-service__get_forecast" {
			seenCall = true
			resStr := string(ev.Result)
			if strings.Contains(resStr, "WEATHER_FORECAST") && strings.Contains(resStr, "Seattle") {
				seenRealResult = true
			}
			if strings.Contains(resStr, `"echo"`) {
				fabricated = true
			}
		}
	}
	_ = explicitResp.Body.Close()
	if !seenCall {
		t.Fatal("expected brokered call event for mcp__weather-service__get_forecast")
	}
	if !seenRealResult {
		t.Fatal("expected helper-subprocess stub result (WEATHER_FORECAST for Seattle) in call event; got none")
	}
	if fabricated {
		t.Fatal("call result carries fabricated echo payload; want real helper-subprocess result")
	}

	// 5. Injects subsystem failure and verifies /healthz returns 503 Service Unavailable with failing component
	sup.SetSubsystemHealth(SubsystemMCPBroker, false, "broker socket terminated")
	if sup.IsHealthy() {
		t.Fatal("expected sup.IsHealthy() to be false after failure injection")
	}
	if eval := sup.EvaluateHealth(); eval.Status != "unavailable" {
		t.Fatalf("expected EvaluateHealth status 'unavailable', got %q", eval.Status)
	}

	failResp, err := http.Get(baseURL + "/healthz")
	if err != nil {
		t.Fatalf("GET /healthz after failure injection: %v", err)
	}
	if failResp.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503 Service Unavailable", failResp.StatusCode)
	}
	var failHealth HealthResponse
	if err := json.NewDecoder(failResp.Body).Decode(&failHealth); err != nil {
		t.Fatalf("decode fail healthz: %v", err)
	}
	_ = failResp.Body.Close()

	if failHealth.Status != "unavailable" {
		t.Fatalf("health status = %q, want 'unavailable'", failHealth.Status)
	}
	brokerStatus, ok := failHealth.Subsystems[SubsystemMCPBroker]
	if !ok || brokerStatus.Ready || brokerStatus.Error != "broker socket terminated" {
		t.Fatalf("unexpected broker subsystem health: %+v", brokerStatus)
	}

	// 6. Clean shutdown drains sessions and stops child processes
	if err := sup.Shutdown(context.Background()); err != nil {
		t.Fatalf("Shutdown failed: %v", err)
	}

	// Verify memory journal was flushed and persisted to disk
	journalData, err := os.ReadFile(journalFile)
	if err != nil {
		t.Fatalf("read memory journal file: %v", err)
	}
	if len(journalData) == 0 {
		t.Fatal("expected memory journal file to contain flushed entries")
	}
	lines := strings.Split(strings.TrimSpace(string(journalData)), "\n")
	if len(lines) < 2 {
		t.Fatalf("expected multiple journal entries, got %d lines", len(lines))
	}
}

// TestUpLockModeAdvertisedRoutes pins the HTTP contract of the lock-mode
// all-in-one supervisor: every endpoint the `fak up` ready banner advertises
// must be registered, and the mock chat route must satisfy the OpenAI
// completion envelope (issue #12616). It boots the lock-mode supervisor
// directly (v2 product-lock fixture, mock engine, loopback ephemeral port) —
// the same machinery that runAllInOneUp drives — and never touches the
// turnkey server helper.
func TestUpLockModeAdvertisedRoutes(t *testing.T) {
	dir := t.TempDir()
	journalFile := filepath.Join(dir, "memory-journal.jsonl")
	lockFile := filepath.Join(dir, "harness.lock.json")

	lockContent := `{
  "schema": "fak.harness-product-lock/v2",
  "id": "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  "platforms": [
    {"os": "linux", "arch": "amd64"},
    {"os": "windows", "arch": "amd64"}
  ],
  "budget": {
    "context_tokens": 2048,
    "memory_mib": 256,
    "workers": 1
  },
  "components": [
    {
      "id": "weather-service",
      "version": "1.0.0",
      "digest": "sha256:1111222233334444555566667777888899990000aaaabbbbccccddddeeeeffff",
      "source": "mcp/weather",
      "provider": "mcp",
      "provides": ["get_forecast"]
    }
  ],
  "assets": [
    {
      "kind": "memory",
      "id": "file-journal",
      "value": ` + jsonQuote(journalFile) + `
    }
  ]
}`
	if err := os.WriteFile(lockFile, []byte(lockContent), 0600); err != nil {
		t.Fatalf("write lock file: %v", err)
	}

	cfg := Config{
		LockPath: lockFile,
		Addr:     "127.0.0.1:0",
		Engine:   "mock",
		Mock:     true,
	}

	sup, err := NewSupervisor(cfg)
	if err != nil {
		t.Fatalf("NewSupervisor: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if err := sup.Start(ctx); err != nil {
		t.Fatalf("Start supervisor: %v", err)
	}
	defer func() {
		_ = sup.Shutdown(context.Background())
	}()

	addr := sup.Addr()
	if addr == "" {
		t.Fatal("supervisor boundAddr is empty")
	}
	baseURL := "http://" + addr

	// Bounded readiness polling: /healthz must become 200 with a subsystems
	// payload; never sleep unboundedly.
	deadline := time.Now().Add(5 * time.Second)
	var health HealthResponse
	var lastStatus int
	var lastErr error
	for {
		resp, err := http.Get(baseURL + "/healthz")
		if err == nil {
			lastStatus = resp.StatusCode
			if resp.StatusCode == http.StatusOK {
				if err := json.NewDecoder(resp.Body).Decode(&health); err != nil {
					resp.Body.Close()
					t.Fatalf("decode /healthz: %v", err)
				}
				resp.Body.Close()
				break
			}
			resp.Body.Close()
		} else {
			lastErr = err
		}
		if time.Now().After(deadline) {
			t.Fatalf("/healthz not ready within deadline (last status=%d err=%v)", lastStatus, lastErr)
		}
		time.Sleep(25 * time.Millisecond)
	}
	if health.Status != "ok" {
		t.Fatalf("health status = %q, want 'ok'", health.Status)
	}
	if len(health.Subsystems) == 0 {
		t.Fatal("expected /healthz subsystems payload, got empty map")
	}

	// POST /v1/chat/completions returns the OpenAI-compatible mock envelope.
	chatBody := `{"model":"audit","messages":[{"role":"user","content":"hello"}]}`
	chatResp, err := http.Post(baseURL+"/v1/chat/completions", "application/json", strings.NewReader(chatBody))
	if err != nil {
		t.Fatalf("POST /v1/chat/completions: %v", err)
	}
	if chatResp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(chatResp.Body)
		chatResp.Body.Close()
		t.Fatalf("POST /v1/chat/completions status = %d, want 200; body: %s", chatResp.StatusCode, raw)
	}
	var chat map[string]any
	if err := json.NewDecoder(chatResp.Body).Decode(&chat); err != nil {
		t.Fatalf("decode chat completion body as JSON object: %v", err)
	}
	_ = chatResp.Body.Close()
	if chat["object"] != "chat.completion" {
		t.Fatalf("chat object = %v, want chat.completion", chat["object"])
	}
	if id, _ := chat["id"].(string); !strings.HasPrefix(id, "chatcmpl-") {
		t.Fatalf("chat id = %v, want chatcmpl- prefix", chat["id"])
	}
	choices, ok := chat["choices"].([]any)
	if !ok || len(choices) != 1 {
		t.Fatalf("chat choices = %#v, want exactly one choice", chat["choices"])
	}
	choice, _ := choices[0].(map[string]any)
	if choice == nil {
		t.Fatalf("chat choice = %#v, want object", choices[0])
	}
	msg, _ := choice["message"].(map[string]any)
	if msg == nil {
		t.Fatalf("choice message = %#v, want object", choice["message"])
	}
	if msg["role"] != "assistant" {
		t.Fatalf("message role = %v, want assistant", msg["role"])
	}
	content, _ := msg["content"].(string)
	if strings.TrimSpace(content) == "" {
		t.Fatalf("message content = %q, want non-empty", content)
	}
	if choice["finish_reason"] != "stop" {
		t.Fatalf("finish_reason = %v, want stop", choice["finish_reason"])
	}
	usage, _ := chat["usage"].(map[string]any)
	if usage == nil {
		t.Fatalf("usage block missing from envelope: %v", chat["usage"])
	}
	if usage["prompt_tokens"] == nil || usage["completion_tokens"] == nil || usage["total_tokens"] == nil {
		t.Fatalf("usage token counts missing: %#v", usage)
	}

	// GET /v1/chat/completions is registered but POST-only: expect 405.
	getResp, err := http.Get(baseURL + "/v1/chat/completions")
	if err != nil {
		t.Fatalf("GET /v1/chat/completions: %v", err)
	}
	_, _ = io.Copy(io.Discard, getResp.Body)
	_ = getResp.Body.Close()
	if getResp.StatusCode != http.StatusMethodNotAllowed {
		t.Fatalf("GET /v1/chat/completions status = %d, want 405", getResp.StatusCode)
	}

	// POST with an invalid JSON body: expect 400.
	badResp, err := http.Post(baseURL+"/v1/chat/completions", "application/json", strings.NewReader("{not json"))
	if err != nil {
		t.Fatalf("POST /v1/chat/completions invalid JSON: %v", err)
	}
	_, _ = io.Copy(io.Discard, badResp.Body)
	_ = badResp.Body.Close()
	if badResp.StatusCode != http.StatusBadRequest {
		t.Fatalf("POST invalid JSON status = %d, want 400", badResp.StatusCode)
	}

	// /v1/fak/agent/sessions must stay registered (non-404 proves the mux
	// entry; a bare GET answers 405).
	sessResp, err := http.Get(baseURL + "/v1/fak/agent/sessions")
	if err != nil {
		t.Fatalf("GET /v1/fak/agent/sessions: %v", err)
	}
	_, _ = io.Copy(io.Discard, sessResp.Body)
	_ = sessResp.Body.Close()
	if sessResp.StatusCode == http.StatusNotFound {
		t.Fatal("/v1/fak/agent/sessions not registered (got 404)")
	}
	if sessResp.StatusCode != http.StatusMethodNotAllowed {
		t.Fatalf("GET /v1/fak/agent/sessions status = %d, want 405", sessResp.StatusCode)
	}
}

func jsonQuote(s string) string {
	b, _ := json.Marshal(s)
	return string(b)
}

func TestBundledRelativeMemorySurvivesShutdown(t *testing.T) {
	dir := t.TempDir()

	lockPath := filepath.Join(dir, "harness.lock.json")
	lockContent := `{
  "schema": "fak.harness-product-lock/v2",
  "id": "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  "platforms": [
    {"os": "linux", "arch": "amd64"},
    {"os": "darwin", "arch": "arm64"},
    {"os": "windows", "arch": "amd64"}
  ],
  "budget": {
    "context_tokens": 4096,
    "memory_mib": 512,
    "workers": 1
  },
  "components": [
    {
      "id": "echo-service",
      "version": "1.0.0",
      "digest": "sha256:1111222233334444555566667777888899990000aaaabbbbccccddddeeeeffff",
      "source": "bin/echo-service",
      "provider": "mcp",
      "provides": ["echo"]
    }
  ],
  "assets": [
    {
      "kind": "memory",
      "id": "rel-journal",
      "value": "relative-session-journal.jsonl"
    }
  ]
}`
	if err := os.WriteFile(lockPath, []byte(lockContent), 0600); err != nil {
		t.Fatalf("write lock file: %v", err)
	}

	policyPath := filepath.Join(dir, "policy.json")
	if err := os.WriteFile(policyPath, []byte(`{"version":"v1","allow":["*"]}`), 0600); err != nil {
		t.Fatalf("write policy: %v", err)
	}
	assetsDir := filepath.Join(dir, "assets")
	_ = os.MkdirAll(assetsDir, 0755)
	binDir := filepath.Join(dir, "bin")
	_ = os.MkdirAll(binDir, 0755)
	_ = os.WriteFile(filepath.Join(binDir, "echo-service"), []byte("#!/bin/sh\necho echo-service\n"), 0755)
	modelPath := filepath.Join(dir, "model.bin")
	_ = os.WriteFile(modelPath, []byte("fake-model-weights"), 0600)

	bundlePath := filepath.Join(dir, "test.fakpack")
	createRes, err := fakpack.Create(fakpack.CreateOptions{
		LockPath:   lockPath,
		PolicyPath: policyPath,
		AssetsDir:  assetsDir,
		BinDir:     binDir,
		ModelPath:  modelPath,
		OutPath:    bundlePath,
	})
	if err != nil {
		t.Fatalf("fakpack.Create: %v", err)
	}
	if createRes == nil || createRes.BundlePath != bundlePath {
		t.Fatalf("unexpected bundle creation: %+v", createRes)
	}

	cfg := Config{
		BundlePath: bundlePath,
		Addr:       "127.0.0.1:0",
		Engine:     "mock",
		Mock:       true,
	}

	// Session 1: start supervisor, execute session request, then shutdown.
	sup1, err := NewSupervisor(cfg)
	if err != nil {
		t.Fatalf("NewSupervisor (1): %v", err)
	}
	ctx1, cancel1 := context.WithCancel(context.Background())
	defer cancel1()
	if err := sup1.Start(ctx1); err != nil {
		t.Fatalf("Start (1): %v", err)
	}

	sessReq1 := `{"goal":"session 1 task","max_turns":1}`
	resp1, err := http.Post("http://"+sup1.Addr()+"/v1/fak/agent/sessions", "application/json", strings.NewReader(sessReq1))
	if err != nil {
		t.Fatalf("POST session (1): %v", err)
	}
	_, _ = io.Copy(io.Discard, resp1.Body)
	_ = resp1.Body.Close()
	if resp1.StatusCode != http.StatusOK {
		t.Fatalf("session (1) status = %d, want 200", resp1.StatusCode)
	}

	// Verify session 1 recorded entries
	entries1 := sup1.Memory().Entries()
	if len(entries1) < 2 {
		t.Fatalf("expected >=2 memory entries in session 1, got %d", len(entries1))
	}

	// Clean shutdown removes unpackDir. Relative memory MUST survive!
	if err := sup1.Shutdown(context.Background()); err != nil {
		t.Fatalf("Shutdown (1): %v", err)
	}

	expectedJournalFile := filepath.Join(dir, "relative-session-journal.jsonl")
	dataAfterShutdown, err := os.ReadFile(expectedJournalFile)
	if err != nil {
		t.Fatalf("relative memory journal did not survive shutdown at %s: %v", expectedJournalFile, err)
	}
	if len(dataAfterShutdown) == 0 {
		t.Fatalf("expected non-empty journal file after shutdown")
	}

	// Session 2: restart supervisor with same bundle. Verify session 1 entries are recalled.
	sup2, err := NewSupervisor(cfg)
	if err != nil {
		t.Fatalf("NewSupervisor (2): %v", err)
	}
	ctx2, cancel2 := context.WithCancel(context.Background())
	defer cancel2()
	if err := sup2.Start(ctx2); err != nil {
		t.Fatalf("Start (2): %v", err)
	}

	// Memory journal should already have session 1 entries
	recalledEntries := sup2.Memory().Entries()
	if len(recalledEntries) < len(entries1) {
		t.Fatalf("expected recalled entries >= %d, got %d", len(entries1), len(recalledEntries))
	}

	// Execute session 2
	sessReq2 := `{"goal":"session 2 task","max_turns":1}`
	resp2, err := http.Post("http://"+sup2.Addr()+"/v1/fak/agent/sessions", "application/json", strings.NewReader(sessReq2))
	if err != nil {
		t.Fatalf("POST session (2): %v", err)
	}
	_, _ = io.Copy(io.Discard, resp2.Body)
	_ = resp2.Body.Close()
	if resp2.StatusCode != http.StatusOK {
		t.Fatalf("session (2) status = %d, want 200", resp2.StatusCode)
	}

	if err := sup2.Shutdown(context.Background()); err != nil {
		t.Fatalf("Shutdown (2): %v", err)
	}

	finalData, err := os.ReadFile(expectedJournalFile)
	if err != nil {
		t.Fatalf("read final journal: %v", err)
	}
	finalLines := strings.Split(strings.TrimSpace(string(finalData)), "\n")
	if len(finalLines) < len(entries1)+2 {
		t.Fatalf("expected combined journal lines >= %d, got %d", len(entries1)+2, len(finalLines))
	}
}

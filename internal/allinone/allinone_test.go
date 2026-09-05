package allinone

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/fakpack"
)

func TestUpBootstrap(t *testing.T) {
	TestAllInOneBootstrapLifecycle(t)
}

func TestAllInOneBootstrapLifecycle(t *testing.T) {
	dir := t.TempDir()
	journalFile := filepath.Join(dir, "memory-journal.jsonl")
	lockFile := filepath.Join(dir, "harness.lock.json")

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
      "source": "mcp/weather",
      "provider": "mcp",
      "provides": ["get_forecast", "get_alerts"]
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
	if plan.Engine != "mock" {
		t.Fatalf("expected Engine 'mock', got %q", plan.Engine)
	}

	// 2. Boots test supervisor with mock model, MCP broker, and memory journal
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
	for _, subName := range []string{SubsystemHTTP, SubsystemInference, SubsystemMCPBroker, SubsystemMemoryStore} {
		s, ok := health.Subsystems[subName]
		if !ok || !s.Ready {
			t.Fatalf("subsystem %q not ready: %+v", subName, s)
		}
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
	scanExp := bufio.NewScanner(explicitResp.Body)
	for scanExp.Scan() {
		var ev struct {
			Event string          `json:"event"`
			Tool  string          `json:"tool"`
			Echo  json.RawMessage `json:"result"`
		}
		if err := json.Unmarshal(scanExp.Bytes(), &ev); err != nil {
			t.Fatalf("invalid NDJSON %q: %v", scanExp.Text(), err)
		}
		if ev.Event == "call" && ev.Tool == "mcp__weather-service__get_forecast" {
			seenCall = true
		}
	}
	_ = explicitResp.Body.Close()
	if !seenCall {
		t.Fatal("expected brokered call event for mcp__weather-service__get_forecast")
	}

	// 5. Injects subsystem failure and verifies /healthz returns 503 Service Unavailable with failing component
	sup.SetSubsystemHealth(SubsystemMCPBroker, false, "broker socket terminated")

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

package airgaptest

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/allinone"
	"github.com/anthony-chaudhary/fak/internal/fakpack"
)

func init() {
	if os.Getenv("GO_WANT_HELPER_PROCESS") == "1" {
		runHelperProcess()
		os.Exit(0)
	}
}

// TestHelperProcess provides the helper process entry point when executed via go test flag.
func TestHelperProcess(t *testing.T) {
	if os.Getenv("GO_WANT_HELPER_PROCESS") != "1" {
		return
	}
	runHelperProcess()
	os.Exit(0)
}

func runHelperProcess() {
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
								"serverInfo":      map[string]any{"name": "local-mcp-server", "version": "1.0.0"},
							},
						}
						data, _ := json.Marshal(resp)
						_, _ = os.Stdout.Write(append(data, '\n'))

					case "notifications/initialized":
						// Notification carries no response

					case "tools/list":
						resp := map[string]any{
							"jsonrpc": "2.0",
							"id":      req.ID,
							"result": map[string]any{
								"tools": []map[string]any{
									{
										"name":        "echo",
										"description": "Echo tool for airgap test",
										"inputSchema": map[string]any{"type": "object"},
									},
								},
							},
						}
						data, _ := json.Marshal(resp)
						_, _ = os.Stdout.Write(append(data, '\n'))

					case "tools/call":
						var callParams struct {
							Name      string          `json:"name"`
							Arguments json.RawMessage `json:"arguments"`
						}
						_ = json.Unmarshal(req.Params, &callParams)
						resp := map[string]any{
							"jsonrpc": "2.0",
							"id":      req.ID,
							"result": map[string]any{
								"content": []map[string]any{
									{
										"type": "text",
										"text": fmt.Sprintf("dispatched to mcp child process: tool=%s args=%s", callParams.Name, string(callParams.Arguments)),
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

// EgressTracer intercepts network dial attempts and strictly rejects non-loopback egress.
type EgressTracer struct {
	mu               sync.Mutex
	loopbackDials    int64
	egressViolations int64
	violationAddrs   []string
}

func (et *EgressTracer) TrackDial(ctx context.Context, network, addr string) (net.Conn, error) {
	host, _, err := net.SplitHostPort(addr)
	if err != nil {
		host = addr
	}

	isLoop := false
	if host == "localhost" || host == "127.0.0.1" || host == "::1" {
		isLoop = true
	} else if ip := net.ParseIP(host); ip != nil && ip.IsLoopback() {
		isLoop = true
	}

	if !isLoop {
		et.mu.Lock()
		et.egressViolations++
		et.violationAddrs = append(et.violationAddrs, addr)
		et.mu.Unlock()
		return nil, fmt.Errorf("AIRGAP_EGRESS_VIOLATION: dial blocked to non-loopback destination %s", addr)
	}

	atomic.AddInt64(&et.loopbackDials, 1)
	var dialer net.Dialer
	return dialer.DialContext(ctx, network, addr)
}

func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()

	out, err := os.OpenFile(dst, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o755)
	if err != nil {
		return err
	}
	defer out.Close()

	if _, err := io.Copy(out, in); err != nil {
		return err
	}
	return out.Close()
}

// TestAirGapBootstrap_ZeroEgress asserts that an air-gapped harness bundle can be verified,
// bootstrapped, and executed with full tool dispatch and durable memory recording while
// strictly enforcing zero outbound network egress outside loopback.
func TestAirGapBootstrap_ZeroEgress(t *testing.T) {
	// 1. Hermetic environment isolation: override proxy environment variables
	t.Setenv("HTTP_PROXY", "http://127.0.0.1:1")
	t.Setenv("HTTPS_PROXY", "http://127.0.0.1:1")
	t.Setenv("ALL_PROXY", "http://127.0.0.1:1")
	t.Setenv("GOPROXY", "off")

	// 2. Setup socket tracer / egress trap
	tracer := &EgressTracer{}
	origTransport := http.DefaultTransport
	customTransport := http.DefaultTransport.(*http.Transport).Clone()
	customTransport.DialContext = tracer.TrackDial
	http.DefaultTransport = customTransport
	t.Cleanup(func() {
		http.DefaultTransport = origTransport
	})

	// Verify egress trap triggers AIRGAP_EGRESS_VIOLATION on non-loopback dial
	_, testDialErr := tracer.TrackDial(context.Background(), "tcp", "192.0.2.1:80")
	if testDialErr == nil || !strings.Contains(testDialErr.Error(), "AIRGAP_EGRESS_VIOLATION") {
		t.Fatalf("expected egress trap to return AIRGAP_EGRESS_VIOLATION, got: %v", testDialErr)
	}
	// Reset counter from the sanity probe
	tracer.mu.Lock()
	tracer.egressViolations = 0
	tracer.violationAddrs = nil
	tracer.mu.Unlock()

	// 3. Build synthetic self-contained .fakpack bundle
	dir := t.TempDir()
	journalFile := filepath.Join(dir, "memory-journal.jsonl")
	lockFile := filepath.Join(dir, "harness.lock.json")
	policyFile := filepath.Join(dir, "policy.json")
	modelFile := filepath.Join(dir, "model.bin")
	binDir := filepath.Join(dir, "bin")
	assetsDir := filepath.Join(dir, "assets")

	if err := os.MkdirAll(binDir, 0o755); err != nil {
		t.Fatalf("mkdir bin: %v", err)
	}
	if err := os.MkdirAll(assetsDir, 0o755); err != nil {
		t.Fatalf("mkdir assets: %v", err)
	}

	exe, err := os.Executable()
	if err != nil {
		exe = os.Args[0]
	}
	mcpBinaryName := "mcp-server"
	if runtime.GOOS == "windows" {
		mcpBinaryName += ".exe"
	}
	targetMcpBinary := filepath.Join(binDir, mcpBinaryName)
	if err := copyFile(exe, targetMcpBinary); err != nil {
		t.Fatalf("copy executable to bin: %v", err)
	}

	lockJSON := fmt.Sprintf(`{
  "schema": "fak.harness-product-lock/v2",
  "id": "sha256:airgap0000000000000000000000000000000000000000000000000000000001",
  "platforms": [
    {"os": %q, "arch": %q}
  ],
  "budget": {
    "context_tokens": 4096,
    "memory_mib": 512,
    "workers": 1
  },
  "components": [
    {
      "id": "local-mcp-server",
      "version": "1.0.0",
      "digest": "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      "source": %q,
      "provider": "mcp",
      "provides": ["echo"],
      "adapters": ["-test.run=TestHelperProcess", "--"]
    }
  ],
  "assets": [
    {
      "kind": "memory",
      "id": "local-journal",
      "value": %q,
      "source": "local"
    }
  ]
}`, runtime.GOOS, runtime.GOARCH, "bin/"+mcpBinaryName, journalFile)

	if err := os.WriteFile(lockFile, []byte(lockJSON), 0o644); err != nil {
		t.Fatalf("write lock file: %v", err)
	}
	if err := os.WriteFile(policyFile, []byte(`{"version":"v1","allow":["tool:*"]}`), 0o644); err != nil {
		t.Fatalf("write policy: %v", err)
	}
	if err := os.WriteFile(modelFile, []byte("in-kernel-mock-weights-data"), 0o644); err != nil {
		t.Fatalf("write model weights: %v", err)
	}

	bundlePath := filepath.Join(dir, "airgap-test-bundle.fakpack")
	createRes, err := fakpack.Create(fakpack.CreateOptions{
		LockPath:   lockFile,
		AssetsDir:  assetsDir,
		BinDir:     binDir,
		ModelPath:  modelFile,
		PolicyPath: policyFile,
		OutPath:    bundlePath,
	})
	if err != nil {
		t.Fatalf("fakpack.Create failed: %v", err)
	}
	if createRes.BundlePath != bundlePath {
		t.Fatalf("expected bundle path %s, got %s", bundlePath, createRes.BundlePath)
	}

	// 4. Verify bundle offline with fakpack.Verify
	verifyRes, err := fakpack.Verify(fakpack.VerifyOptions{
		BundlePath:       bundlePath,
		ExpectedLockPath: lockFile,
	})
	if err != nil {
		t.Fatalf("fakpack.Verify failed: %v", err)
	}
	if !verifyRes.AirGapVerified {
		t.Fatal("expected verifyRes.AirGapVerified to be true")
	}
	if !verifyRes.LockMatches {
		t.Fatal("expected verifyRes.LockMatches to be true")
	}

	// 5. Boot using allinone.NewSupervisor with bundle path
	cfg := allinone.Config{
		BundlePath: bundlePath,
		Addr:       "127.0.0.1:0",
		Engine:     "mock",
		Mock:       true,
	}
	sup, err := allinone.NewSupervisor(cfg)
	if err != nil {
		t.Fatalf("allinone.NewSupervisor: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	if err := sup.Start(ctx); err != nil {
		t.Fatalf("sup.Start failed: %v", err)
	}
	defer func() {
		_ = sup.Shutdown(context.Background())
	}()

	addr := sup.Addr()
	if addr == "" {
		t.Fatal("supervisor boundAddr is empty")
	}
	baseURL := "http://" + addr

	// 6. Wait for /healthz to report 200 OK
	var health allinone.HealthResponse
	healthDeadline := time.Now().Add(5 * time.Second)
	healthOK := false
	for time.Now().Before(healthDeadline) {
		hResp, err := http.Get(baseURL + "/healthz")
		if err == nil {
			if hResp.StatusCode == http.StatusOK {
				if decErr := json.NewDecoder(hResp.Body).Decode(&health); decErr == nil && health.Status == "ok" {
					_ = hResp.Body.Close()
					healthOK = true
					break
				}
			}
			_ = hResp.Body.Close()
		}
		time.Sleep(50 * time.Millisecond)
	}
	if !healthOK {
		t.Fatalf("healthz did not report 200 OK within deadline; status=%s", health.Status)
	}

	// 7. Execute session turn via /v1/fak/agent/sessions
	sessReq := `{"goal":"execute airgap task","tool":"mcp__local-mcp-server__echo","args":{"msg":"airgap-witness-payload"}}`
	sessResp, err := http.Post(baseURL+"/v1/fak/agent/sessions", "application/json", strings.NewReader(sessReq))
	if err != nil {
		t.Fatalf("POST /v1/fak/agent/sessions: %v", err)
	}
	if sessResp.StatusCode != http.StatusOK {
		t.Fatalf("POST /v1/fak/agent/sessions status = %d, want 200", sessResp.StatusCode)
	}

	seenStart := false
	seenCall := false
	seenEnd := false
	var callResult string

	scan := bufio.NewScanner(sessResp.Body)
	for scan.Scan() {
		var ev struct {
			Event  string          `json:"event"`
			Tool   string          `json:"tool"`
			Result json.RawMessage `json:"result"`
			Status string          `json:"status"`
		}
		if err := json.Unmarshal(scan.Bytes(), &ev); err != nil {
			t.Fatalf("invalid NDJSON: %v", err)
		}
		if ev.Event == "session.start" {
			seenStart = true
		}
		if ev.Event == "call" && ev.Tool == "mcp__local-mcp-server__echo" {
			seenCall = true
			callResult = string(ev.Result)
		}
		if ev.Event == "session.end" && ev.Status == "ok" {
			seenEnd = true
		}
	}
	_ = sessResp.Body.Close()

	if !seenStart {
		t.Fatal("session.start event not observed")
	}
	if !seenCall {
		t.Fatal("call event for mcp__local-mcp-server__echo not observed")
	}
	// Verify tool dispatch to the MCP child process
	if !strings.Contains(callResult, "dispatched to mcp child process") {
		t.Fatalf("expected tool dispatch to witness MCP child process execution, got: %s", callResult)
	}
	// Verify agent response completion
	if !seenEnd {
		t.Fatal("session.end event with status ok not observed")
	}

	// Gracefully shut down supervisor so all flushed logs persist
	if err := sup.Shutdown(context.Background()); err != nil {
		t.Fatalf("sup.Shutdown failed: %v", err)
	}

	// Verify memory journal record written to disk
	journalBytes, err := os.ReadFile(journalFile)
	if err != nil {
		t.Fatalf("read memory journal file %s: %v", journalFile, err)
	}
	if len(journalBytes) == 0 {
		t.Fatal("expected memory journal file to contain flushed entries")
	}
	journalStr := string(journalBytes)
	if !strings.Contains(journalStr, "session.start") {
		t.Fatalf("expected journal to contain session.start, got:\n%s", journalStr)
	}
	if !strings.Contains(journalStr, "call") || !strings.Contains(journalStr, "mcp__local-mcp-server__echo") {
		t.Fatalf("expected journal to contain tool call record, got:\n%s", journalStr)
	}
	if !strings.Contains(journalStr, "session.end") {
		t.Fatalf("expected journal to contain session.end, got:\n%s", journalStr)
	}

	// Assert zero outbound network connection attempts were made outside loopback
	tracer.mu.Lock()
	violations := tracer.egressViolations
	violationAddrs := append([]string(nil), tracer.violationAddrs...)
	tracer.mu.Unlock()
	if violations > 0 {
		t.Fatalf("AIRGAP_EGRESS_VIOLATION: %d outbound connections attempted outside loopback: %v", violations, violationAddrs)
	}

	// 8. Negative control test: lockfile referencing remote MCP server URL
	t.Run("NegativeControl_RemoteDependencyRefusal", func(t *testing.T) {
		badDir := t.TempDir()
		badLockFile := filepath.Join(badDir, "remote.lock.json")
		badLockJSON := `{
  "schema": "fak.harness-product-lock/v2",
  "id": "sha256:badremote0000000000000000000000000000000000000000000000000000000001",
  "components": [
    {
      "id": "remote-mcp-service",
      "version": "1.0.0",
      "digest": "sha256:1111222233334444555566667777888899990000aaaabbbbccccddddeeeeffff",
      "source": "http://external.service/mcp",
      "provider": "mcp",
      "provides": ["fetch"]
    }
  ]
}`
		if err := os.WriteFile(badLockFile, []byte(badLockJSON), 0o644); err != nil {
			t.Fatalf("write bad lock: %v", err)
		}

		badBundlePath := filepath.Join(badDir, "bad.fakpack")
		_, createErr := fakpack.Create(fakpack.CreateOptions{
			LockPath: badLockFile,
			OutPath:  badBundlePath,
		})
		if createErr == nil {
			t.Fatal("expected fakpack.Create to fail closed on remote dependency")
		}
		if !strings.Contains(createErr.Error(), "AIRGAP_UNRESOLVED_REMOTE_DEPENDENCY") {
			t.Fatalf("expected error containing AIRGAP_UNRESOLVED_REMOTE_DEPENDENCY, got: %v", createErr)
		}

		// Also assert bootstrap preflight fails closed with AIRGAP_UNRESOLVED_REMOTE_DEPENDENCY
		badSup, err := allinone.NewSupervisor(allinone.Config{
			LockPath: badLockFile,
			Addr:     "127.0.0.1:0",
		})
		if err == nil {
			startErr := badSup.Start(context.Background())
			if startErr == nil {
				_ = badSup.Shutdown(context.Background())
				t.Fatal("expected bootstrap preflight to fail closed on remote dependency")
			}
			if !strings.Contains(startErr.Error(), "AIRGAP_UNRESOLVED_REMOTE_DEPENDENCY") {
				t.Fatalf("expected start error containing AIRGAP_UNRESOLVED_REMOTE_DEPENDENCY, got: %v", startErr)
			}
		}
	})
}

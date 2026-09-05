package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/allinone"
)

func TestUpHelpUsesServeSurface(t *testing.T) {
	fs, _ := newServeFlagSet()
	for _, name := range []string{"addr", "gguf", "base-url", "policy", "require-key-env", "metrics-snapshot", "session-state"} {
		if fs.Lookup(name) == nil {
			t.Fatalf("serve/up flag %q is missing", name)
		}
	}
}

func TestUpBootsUnifiedAgentRuntime(t *testing.T) {
	if testing.Short() {
		t.Skip("process witness")
	}
	root := filepath.Clean(filepath.Join("..", ".."))
	cacheRoot := t.TempDir()
	bin := filepath.Join(cacheRoot, "fak")
	if runtime.GOOS == "windows" {
		bin += ".exe"
	}
	goCache := filepath.Join(cacheRoot, "gocache")
	goTmp := filepath.Join(cacheRoot, "gotmp")
	if err := os.MkdirAll(goTmp, 0o755); err != nil {
		t.Fatal(err)
	}
	build := exec.Command("go", "build", "-o", bin, "./cmd/fak")
	build.Dir = root
	build.Env = append(os.Environ(), "GOCACHE="+goCache, "GOTMPDIR="+goTmp)
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build fak: %v\n%s", err, out)
	}

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	addr := ln.Addr().String()
	_ = ln.Close()

	// A parent harness may legitimately use FAK_SESSION_REGISTRY for the
	// child-registration lineage. Keep this process witness hermetic by naming
	// the descriptor registry explicitly, as a real co-located operator must.
	childRegistry := filepath.Join(cacheRoot, "child-registrations.jsonl")
	childRegistryBody := []byte("{\"schema\":\"fak-child-registration/1\"}\n")
	if err := os.WriteFile(childRegistry, childRegistryBody, 0o600); err != nil {
		t.Fatal(err)
	}
	descriptorRegistry := filepath.Join(cacheRoot, "session-registry.json")
	cmd := exec.Command(bin, "up", "--addr", addr, "--engine", "mock", "--native", "--session-registry", descriptorRegistry)
	cmd.Dir = root
	cmd.Env = append(os.Environ(), "FAK_SESSION_REGISTRY="+childRegistry)
	var output bytes.Buffer
	cmd.Stdout, cmd.Stderr = &output, &output
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	waited := false
	defer func() {
		if !waited && (cmd.ProcessState == nil || !cmd.ProcessState.Exited()) {
			_ = cmd.Process.Kill()
			_ = cmd.Wait()
		}
	}()

	base := "http://" + addr
	deadline := time.Now().Add(20 * time.Second)
	for {
		resp, getErr := http.Get(base + "/readyz")
		if getErr == nil {
			_, _ = io.Copy(io.Discard, resp.Body)
			_ = resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				break
			}
		}
		if time.Now().After(deadline) {
			t.Fatalf("ready timeout: %v\n%s", getErr, output.String())
		}
		time.Sleep(40 * time.Millisecond)
	}

	body := strings.NewReader(`{"goal":"book the task","max_turns":4}`)
	resp, err := http.Post(base+"/v1/fak/agent/sessions", "application/json", body)
	if err != nil {
		t.Fatalf("post session: %v\n%s", err, output.String())
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(resp.Body)
		t.Fatalf("session status=%d body=%s", resp.StatusCode, raw)
	}
	seenEnd := false
	scan := bufio.NewScanner(resp.Body)
	for scan.Scan() {
		var event struct {
			Event string `json:"event"`
		}
		if err := json.Unmarshal(scan.Bytes(), &event); err != nil {
			t.Fatalf("invalid NDJSON %q: %v", scan.Text(), err)
		}
		if event.Event == "session.end" {
			seenEnd = true
		}
	}
	if err := scan.Err(); err != nil {
		t.Fatal(err)
	}
	if !seenEnd {
		t.Fatalf("session.end not observed; process output:\n%s", output.String())
	}
	if got, err := os.ReadFile(childRegistry); err != nil || !bytes.Equal(got, childRegistryBody) {
		t.Fatalf("child-registration lineage changed: got=%q err=%v", got, err)
	}

	if err := cmd.Process.Signal(os.Interrupt); err != nil {
		if runtime.GOOS == "windows" {
			_ = cmd.Process.Kill()
		} else {
			t.Fatalf("interrupt: %v", err)
		}
	}
	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()
	select {
	case err := <-done:
		waited = true
		if err != nil && runtime.GOOS != "windows" {
			t.Fatalf("up did not terminate cleanly: %v\n%s", err, output.String())
		}
	case <-time.After(8 * time.Second):
		t.Fatalf("up did not stop after interrupt\n%s", output.String())
	}
}

func TestLocalNativeLauncherLifetimeOwnershipConformance(t *testing.T) {
	type launcher struct {
		name           string
		file           string
		metalAvailable bool
		owner          string
		marker         string
	}
	const sharedOwner = "loadLocalLauncherModelWithMetalLease"
	launchers := []launcher{
		{name: "serve", file: "serve.go", metalAvailable: true, owner: sharedOwner, marker: "loadLocalLauncherModelWithMetalLease("},
		{name: "up", file: "up.go", metalAvailable: true, owner: sharedOwner, marker: "cmdServe(argv)"},
		{name: "guard", file: "guard_local.go", metalAvailable: false, marker: "guardDetectLocalBackend"},
		// model-canary has a distinct lifetime owner because it replaces an
		// incumbent process, runs a candidate, restores the incumbent, and only
		// then releases. Its shared OS lease is acquired by the Darwin adapter.
		{name: "model-canary", file: "model_canary_run_darwin.go", metalAvailable: true, owner: "modelCanaryLease", marker: "gpulease.Acquire(gpulease.Options{Path: cfg.Path"},
		{name: "run", file: "run_model.go", metalAvailable: false, marker: "metal=false"},
		{name: "scout", file: "scout_native.go", metalAvailable: false, marker: "metal=false"},
	}

	for _, tc := range launchers {
		t.Run(tc.name, func(t *testing.T) {
			raw, err := os.ReadFile(tc.file)
			if err != nil {
				t.Fatal(err)
			}
			source := string(raw)
			if !strings.Contains(source, tc.marker) {
				t.Fatalf("%s no longer carries conformance marker %q", tc.file, tc.marker)
			}
			if tc.metalAvailable && tc.owner == "" {
				t.Fatalf("Metal launcher %s has no lifetime owner", tc.name)
			}
			if !tc.metalAvailable && strings.Contains(source, sharedOwner) {
				t.Fatalf("Metal-unavailable launcher %s acquired local launcher residency", tc.name)
			}
		})
	}

	serveSource, err := os.ReadFile("serve.go")
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.Count(string(serveSource), "loadLocalLauncherModelWithMetalLease("); got != 1 {
		t.Fatalf("serve ownership acquisitions=%d, want 1", got)
	}
	upSource, err := os.ReadFile("up.go")
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.Count(string(upSource), "cmdServe(argv)"); got != 1 {
		t.Fatalf("up direct serve delegations=%d, want 1", got)
	}
	if strings.Contains(string(upSource), sharedOwner) {
		t.Fatal("up must reuse serve's owner, not acquire a second launcher lease")
	}
}

func TestUpBootstrap(t *testing.T) {
	// 1. Tests fak up --help outputs help
	var helpBuf bytes.Buffer
	printUpHelp(&helpBuf)
	helpText := helpBuf.String()
	for _, expected := range []string{"--lock", "--bundle", "--mock", "--dry-run"} {
		if !strings.Contains(helpText, expected) {
			t.Fatalf("up help output missing %q:\n%s", expected, helpText)
		}
	}

	// 2. Tests fak up --lock ... --dry-run prints plan and exits 0
	dir := t.TempDir()
	lockPath := filepath.Join(dir, "harness.lock.json")
	lockJSON := `{
  "schema": "fak.harness-product-lock/v2",
  "id": "bootstrap-test-lock-id",
  "platforms": [
    {"os": "linux", "arch": "amd64"},
    {"os": "darwin", "arch": "arm64"},
    {"os": "windows", "arch": "amd64"}
  ],
  "budget": {
    "context_tokens": 2048,
    "memory_mib": 256,
    "workers": 1
  },
  "components": [
    {
      "id": "mcp-service",
      "version": "1.0.0",
      "digest": "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      "source": "pkg/mcp",
      "provider": "mcp",
      "provides": ["ping", "echo"]
    }
  ]
}`
	if err := os.WriteFile(lockPath, []byte(lockJSON), 0600); err != nil {
		t.Fatalf("write lock file: %v", err)
	}

	origStdout := os.Stdout
	rPipe, wPipe, err := os.Pipe()
	if err != nil {
		t.Fatalf("create pipe: %v", err)
	}
	os.Stdout = wPipe

	cmdUp([]string{"--lock", lockPath, "--dry-run"})

	_ = wPipe.Close()
	os.Stdout = origStdout

	var dryRunOutput bytes.Buffer
	_, _ = io.Copy(&dryRunOutput, rPipe)
	_ = rPipe.Close()

	var plan allinone.TopologySpec
	if err := json.Unmarshal(dryRunOutput.Bytes(), &plan); err != nil {
		t.Fatalf("failed to decode dry-run output as JSON: %v\nOutput: %s", err, dryRunOutput.String())
	}
	if plan.LockID != "bootstrap-test-lock-id" {
		t.Fatalf("plan.LockID = %q, want bootstrap-test-lock-id", plan.LockID)
	}
	if len(plan.MCPServers) != 1 || plan.MCPServers[0] != "mcp-service" {
		t.Fatalf("unexpected plan.MCPServers: %v", plan.MCPServers)
	}

	// 3. Tests all-in-one lifecycle
	liveCfg := allinone.Config{
		LockPath: lockPath,
		Addr:     "127.0.0.1:0",
		Engine:   "mock",
		Mock:     true,
	}
	sup, err := allinone.NewSupervisor(liveCfg)
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
		t.Fatal("empty supervisor address")
	}
	base := "http://" + addr

	// Verify /healthz returns 200 OK
	hResp, err := http.Get(base + "/healthz")
	if err != nil {
		t.Fatalf("GET /healthz: %v", err)
	}
	if hResp.StatusCode != http.StatusOK {
		t.Fatalf("healthz status = %d, want 200", hResp.StatusCode)
	}
	var health allinone.HealthResponse
	if err := json.NewDecoder(hResp.Body).Decode(&health); err != nil {
		t.Fatalf("decode healthz: %v", err)
	}
	_ = hResp.Body.Close()
	if health.Status != "ok" {
		t.Fatalf("health status = %q, want 'ok'", health.Status)
	}

	// Submits request to /v1/fak/agent/sessions and verifies response
	body := strings.NewReader(`{"goal":"bootstrap verify goal","tool":"mcp__mcp-service__ping","args":{"token":"xyz"}}`)
	sessResp, err := http.Post(base+"/v1/fak/agent/sessions", "application/json", body)
	if err != nil {
		t.Fatalf("POST /v1/fak/agent/sessions: %v", err)
	}
	if sessResp.StatusCode != http.StatusOK {
		t.Fatalf("session response status = %d, want 200", sessResp.StatusCode)
	}

	seenEnd := false
	scan := bufio.NewScanner(sessResp.Body)
	for scan.Scan() {
		var event struct {
			Event string `json:"event"`
		}
		if err := json.Unmarshal(scan.Bytes(), &event); err != nil {
			t.Fatalf("invalid NDJSON %q: %v", scan.Text(), err)
		}
		if event.Event == "session.end" {
			seenEnd = true
		}
	}
	_ = sessResp.Body.Close()
	if err := scan.Err(); err != nil {
		t.Fatalf("scanner error: %v", err)
	}
	if !seenEnd {
		t.Fatal("session.end not observed in response")
	}

	// Injects subsystem failure and verifies /healthz returns 503
	sup.SetSubsystemHealth(allinone.SubsystemInference, false, "engine GPU timeout")
	failResp, err := http.Get(base + "/healthz")
	if err != nil {
		t.Fatalf("GET /healthz: %v", err)
	}
	if failResp.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("healthz status = %d, want 503", failResp.StatusCode)
	}
	var failHealth allinone.HealthResponse
	if err := json.NewDecoder(failResp.Body).Decode(&failHealth); err != nil {
		t.Fatalf("decode failed healthz: %v", err)
	}
	_ = failResp.Body.Close()
	if failHealth.Status != "unavailable" {
		t.Fatalf("expected health status 'unavailable', got %q", failHealth.Status)
	}

	// Clean shutdown drains sessions and stops child processes
	if err := sup.Shutdown(context.Background()); err != nil {
		t.Fatalf("Shutdown: %v", err)
	}
}

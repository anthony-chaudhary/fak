package main

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
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/allinone"
	"github.com/anthony-chaudhary/fak/internal/macfit"
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
	goTmp := filepath.Join(cacheRoot, "gotmp")
	if err := os.MkdirAll(goTmp, 0o755); err != nil {
		t.Fatal(err)
	}
	buildCtx, cancelBuild := context.WithTimeout(t.Context(), 30*time.Second)
	defer cancelBuild()
	build := exec.CommandContext(buildCtx, "go", "build", "-o", bin, "./cmd/fak")
	build.Dir = root
	build.Env = append(os.Environ(), "GOTMPDIR="+goTmp)
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

func TestFakUpTurnkeyBootstrap(t *testing.T) {
	// Scoped Acceptance Criterion 1: fak up correctly inspects Apple Silicon unified memory and chooses compatible quant tier.
	t.Run("MemoryInspectionAndTierSelection", func(t *testing.T) {
		t.Setenv("FAK_UP_MEMORY_BYTES", fmt.Sprint(16*macfit.GiB))
		mem16, err := macfit.DetectUnifiedMemory()
		if err != nil {
			t.Fatalf("DetectUnifiedMemory: %v", err)
		}
		plan16, err := macfit.ConfigureTurnkey(mem16)
		if err != nil {
			t.Fatalf("ConfigureTurnkey(16GB): %v", err)
		}
		if plan16.Tier.Name != "7B" || plan16.Tier.QuantTier != "Q4_K_M" {
			t.Fatalf("16GB tier = %s (%s), want 7B (Q4_K_M)", plan16.Tier.Name, plan16.Tier.QuantTier)
		}

		t.Setenv("FAK_UP_MEMORY_BYTES", fmt.Sprint(36*macfit.GiB))
		mem36, err := macfit.DetectUnifiedMemory()
		if err != nil {
			t.Fatalf("DetectUnifiedMemory: %v", err)
		}
		plan36, err := macfit.ConfigureTurnkey(mem36)
		if err != nil {
			t.Fatalf("ConfigureTurnkey(36GB): %v", err)
		}
		if plan36.Tier.Name != "27B" || plan36.Tier.QuantTier != "Q4_K_M" {
			t.Fatalf("36GB tier = %s (%s), want 27B (Q4_K_M)", plan36.Tier.Name, plan36.Tier.QuantTier)
		}

		t.Setenv("FAK_UP_MEMORY_BYTES", fmt.Sprint(64*macfit.GiB))
		mem64, err := macfit.DetectUnifiedMemory()
		if err != nil {
			t.Fatalf("DetectUnifiedMemory: %v", err)
		}
		plan64, err := macfit.ConfigureTurnkey(mem64)
		if err != nil {
			t.Fatalf("ConfigureTurnkey(64GB): %v", err)
		}
		if plan64.Tier.Name != "70B" || plan64.Tier.QuantTier != "Q4_K_M" {
			t.Fatalf("64GB tier = %s (%s), want 70B (Q4_K_M)", plan64.Tier.Name, plan64.Tier.QuantTier)
		}
	})

	// Scoped Acceptance Criterion 2: Auto-selected context budget guarantees >= 20% memory headroom to prevent swap.
	t.Run("ContextBudgetHeadroomGuarantee", func(t *testing.T) {
		for _, gib := range []uint64{16, 24, 36, 48, 64, 128} {
			memBytes := gib * macfit.GiB
			plan, err := macfit.ConfigureTurnkey(memBytes)
			if err != nil {
				t.Fatalf("ConfigureTurnkey(%d GiB): %v", gib, err)
			}
			if plan.HeadroomRatio < 0.20 {
				t.Fatalf("memory %d GiB: headroom ratio = %.3f, want >= 0.20", gib, plan.HeadroomRatio)
			}
			if plan.ContextBudgetTokens == 0 {
				t.Fatalf("memory %d GiB: context budget tokens = 0", gib)
			}
			allocated := plan.Tier.WeightBytes + (plan.ContextBudgetTokens * plan.KVBytesPerToken)
			maxAllocated := (memBytes * 80) / 100
			if allocated > maxAllocated {
				t.Fatalf("memory %d GiB: allocated bytes %d exceeds 80%% limit %d", gib, allocated, maxAllocated)
			}
		}
	})

	// CLI Dry Run validation
	t.Run("DryRunPlanOutput", func(t *testing.T) {
		var out bytes.Buffer
		var errOut bytes.Buffer
		runTurnkeyUp(nil, &out, &errOut, []string{"--dry-run", "--memory-gib", "36", "--json"})
		var plan macfit.TurnkeyProfile
		if err := json.Unmarshal(out.Bytes(), &plan); err != nil {
			t.Fatalf("unmarshal dry-run json: %v\noutput: %s", err, out.String())
		}
		if plan.Tier.Name != "27B" {
			t.Fatalf("dry-run plan tier = %q, want 27B", plan.Tier.Name)
		}
		if plan.HeadroomRatio < 0.20 {
			t.Fatalf("dry-run plan headroom = %.3f, want >= 0.20", plan.HeadroomRatio)
		}
	})

	// Scoped Acceptance Criterion 3: OpenAI-compatible completion endpoint answers successfully on loopback.
	t.Run("OpenAICompatibleLoopbackCompletions", func(t *testing.T) {
		plan, err := macfit.ConfigureTurnkey(36 * macfit.GiB)
		if err != nil {
			t.Fatalf("ConfigureTurnkey: %v", err)
		}

		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		server, err := startTurnkeyServer(ctx, plan, "127.0.0.1:0", true)
		if err != nil {
			t.Fatalf("startTurnkeyServer: %v", err)
		}
		defer func() {
			_ = server.Shutdown(context.Background())
		}()

		base := "http://" + server.Addr()

		// 1. Check /healthz
		hResp, err := http.Get(base + "/healthz")
		if err != nil {
			t.Fatalf("GET /healthz: %v", err)
		}
		if hResp.StatusCode != http.StatusOK {
			t.Fatalf("healthz status = %d, want 200", hResp.StatusCode)
		}
		var health map[string]any
		if err := json.NewDecoder(hResp.Body).Decode(&health); err != nil {
			t.Fatalf("decode healthz: %v", err)
		}
		_ = hResp.Body.Close()
		if health["status"] != "ok" || health["tier"] != "27B" {
			t.Fatalf("healthz payload mismatch: %+v", health)
		}

		// 2. Check /readyz
		rResp, err := http.Get(base + "/readyz")
		if err != nil {
			t.Fatalf("GET /readyz: %v", err)
		}
		if rResp.StatusCode != http.StatusOK {
			t.Fatalf("readyz status = %d, want 200", rResp.StatusCode)
		}
		_ = rResp.Body.Close()

		// 3. Check /v1/models
		mResp, err := http.Get(base + "/v1/models")
		if err != nil {
			t.Fatalf("GET /v1/models: %v", err)
		}
		if mResp.StatusCode != http.StatusOK {
			t.Fatalf("models status = %d, want 200", mResp.StatusCode)
		}
		var modelsResp struct {
			Data []struct {
				ID string `json:"id"`
			} `json:"data"`
		}
		if err := json.NewDecoder(mResp.Body).Decode(&modelsResp); err != nil {
			t.Fatalf("decode models: %v", err)
		}
		_ = mResp.Body.Close()
		if len(modelsResp.Data) == 0 || modelsResp.Data[0].ID != plan.Tier.ModelID {
			t.Fatalf("models mismatch: %+v", modelsResp)
		}

		// 4. Non-streaming completion
		bodyJSON := `{"model":"qwen3.8-27b-q4_k_m","messages":[{"role":"user","content":"Explain turnkey Apple Silicon inference."}],"stream":false}`
		cResp, err := http.Post(base+"/v1/chat/completions", "application/json", strings.NewReader(bodyJSON))
		if err != nil {
			t.Fatalf("POST /v1/chat/completions: %v", err)
		}
		if cResp.StatusCode != http.StatusOK {
			raw, _ := io.ReadAll(cResp.Body)
			t.Fatalf("chat completion status = %d, body = %s", cResp.StatusCode, raw)
		}
		var compResp chatCompletionResponse
		if err := json.NewDecoder(cResp.Body).Decode(&compResp); err != nil {
			t.Fatalf("decode chat completion: %v", err)
		}
		_ = cResp.Body.Close()

		if len(compResp.Choices) == 0 {
			t.Fatal("chat completion returned no choices")
		}
		if compResp.Choices[0].Message.Role != "assistant" {
			t.Fatalf("choice role = %q, want 'assistant'", compResp.Choices[0].Message.Role)
		}
		if compResp.Choices[0].Message.Content == "" {
			t.Fatal("choice content is empty")
		}
		if compResp.Usage.CompletionTokens == 0 {
			t.Fatal("completion tokens is 0")
		}

		// 5. Streaming completion
		streamJSON := `{"model":"qwen3.8-27b-q4_k_m","messages":[{"role":"user","content":"Stream tokens."}],"stream":true}`
		sResp, err := http.Post(base+"/v1/chat/completions", "application/json", strings.NewReader(streamJSON))
		if err != nil {
			t.Fatalf("POST /v1/chat/completions (stream): %v", err)
		}
		if sResp.StatusCode != http.StatusOK {
			t.Fatalf("stream status = %d, want 200", sResp.StatusCode)
		}
		scanner := bufio.NewScanner(sResp.Body)
		seenChunk := false
		seenDone := false
		for scanner.Scan() {
			line := scanner.Text()
			if strings.HasPrefix(line, "data: ") {
				data := strings.TrimPrefix(line, "data: ")
				if data == "[DONE]" {
					seenDone = true
					break
				}
				seenChunk = true
			}
		}
		_ = sResp.Body.Close()
		if !seenChunk || !seenDone {
			t.Fatalf("streaming SSE incomplete: seenChunk=%v, seenDone=%v", seenChunk, seenDone)
		}

		// 6. Interactive REPL with token telemetry
		replIn := strings.NewReader("Hello turnkey model!\n/quit\n")
		var replOut bytes.Buffer
		if err := runTurnkeyREPL(ctx, replIn, &replOut, base, plan); err != nil {
			t.Fatalf("runTurnkeyREPL: %v", err)
		}
		replText := replOut.String()
		if !strings.Contains(replText, "you> ") {
			t.Fatalf("REPL output missing 'you> ' prompt:\n%s", replText)
		}
		if !strings.Contains(replText, "fak> ") {
			t.Fatalf("REPL output missing 'fak> ' response:\n%s", replText)
		}
		if !strings.Contains(replText, "telemetry:") || !strings.Contains(replText, "tok/s") {
			t.Fatalf("REPL output missing token/s telemetry:\n%s", replText)
		}
	})
}

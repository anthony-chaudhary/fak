package allinone

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/mcpbroker"
)

// contractStubEngine is a minimal EngineDriver standing in for an explicitly
// configured real engine. It isolates capability-contract tests from model
// inference: Start must treat a non-nil driver as configured ("custom").
type contractStubEngine struct{}

func (contractStubEngine) Complete(_ context.Context, c *abi.ToolCall) (*abi.Result, error) {
	body := `{"status":"admitted","tool":"` + c.Tool + `"}`
	return &abi.Result{
		Call:    c,
		Payload: abi.Ref{Kind: abi.RefInline, Inline: []byte(body), Len: int64(len(body))},
		Status:  abi.StatusOK,
	}, nil
}

func (contractStubEngine) Caps() []abi.Capability { return nil }

// writeContractLock writes a minimal v2 product lock with the given component
// and asset fragments and returns its path.
func writeContractLock(t *testing.T, components string, assets string) string {
	t.Helper()
	if assets == "" {
		assets = `[]`
	}
	lockContent := `{
  "schema": "fak.harness-product-lock/v2",
  "id": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "platforms": [
    {"os": ` + jsonQuote(runtime.GOOS) + `, "arch": ` + jsonQuote(runtime.GOARCH) + `}
  ],
  "budget": {
    "context_tokens": 1024,
    "memory_mib": 128,
    "workers": 1
  },
  "components": ` + components + `,
  "assets": ` + assets + `
}`
	lockFile := filepath.Join(t.TempDir(), "harness.lock.json")
	if err := os.WriteFile(lockFile, []byte(lockContent), 0600); err != nil {
		t.Fatalf("write lock file: %v", err)
	}
	return lockFile
}

// ghostComponentLock returns a lock with a single MCP component whose binary
// does not exist on disk. Real mode must degrade, never fabricate, for it.
func ghostComponentLock(t *testing.T) string {
	t.Helper()
	return writeContractLock(t, `[
    {
      "id": "ghost-mcp-server",
      "version": "1.0.0",
      "digest": "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
      "source": "definitely-missing-binary-xyz123",
      "provider": "mcp",
      "provides": ["echo"]
    }
  ]`, `[]`)
}

// TestContractEngineUnconfiguredFailFast pins the fail-fast contract: real mode
// with an empty Engine and no EngineDriver returns ErrEngineUnconfigured and
// binds no listener.
func TestContractEngineUnconfiguredFailFast(t *testing.T) {
	lockFile := writeContractLock(t, `[]`, `[]`)
	sup, err := NewSupervisor(Config{
		LockPath: lockFile,
		Addr:     "127.0.0.1:0",
	})
	if err != nil {
		t.Fatalf("NewSupervisor: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if err := sup.Start(ctx); !errors.Is(err, ErrEngineUnconfigured) {
		t.Fatalf("Start err = %v, want errors.Is(ErrEngineUnconfigured)", err)
	}
	if addr := sup.Addr(); addr != "" {
		t.Fatalf("supervisor listens on %q after ErrEngineUnconfigured; want nothing bound", addr)
	}
	_ = sup.Shutdown(context.Background())
}

// TestDryRunEngineLabels pins the DryRunTopology engine-label truth table.
func TestDryRunEngineLabels(t *testing.T) {
	lockFile := writeContractLock(t, `[]`, `[]`)
	cases := []struct {
		name string
		cfg  Config
		want string
	}{
		{"driver_set_reports_custom", Config{LockPath: lockFile, EngineDriver: contractStubEngine{}}, "custom"},
		{"mock_flag_reports_mock", Config{LockPath: lockFile, Mock: true}, "mock"},
		{"mock_engine_reports_mock", Config{LockPath: lockFile, Engine: "mock"}, "mock"},
		{"explicit_mock_reports_mock", Config{LockPath: lockFile, Engine: "mock", Mock: true}, "mock"},
		{"inkernel_engine_preserved", Config{LockPath: lockFile, Engine: "inkernel"}, "inkernel"},
		{"empty_real_mode_reports_unconfigured", Config{LockPath: lockFile}, "unconfigured"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			sup, err := NewSupervisor(tc.cfg)
			if err != nil {
				t.Fatalf("NewSupervisor: %v", err)
			}
			spec, err := sup.DryRunTopology()
			if err != nil {
				t.Fatalf("DryRunTopology: %v", err)
			}
			if spec.Engine != tc.want {
				t.Fatalf("Engine = %q, want %q", spec.Engine, tc.want)
			}
		})
	}
}

// TestContractDegradedBrokerNoFabrication pins the real-mode degradation
// contract: a lock component whose binary is missing keeps Start green, but the
// broker subsystem reports not-ready naming the component, Memory() records a
// Type:"component.degraded" entry, and no fabricated echo tool is registered.
func TestContractDegradedBrokerNoFabrication(t *testing.T) {
	const ghostID = "ghost-mcp-server"
	lockFile := ghostComponentLock(t)
	sup, err := NewSupervisor(Config{
		LockPath:     lockFile,
		Addr:         "127.0.0.1:0",
		EngineDriver: contractStubEngine{},
	})
	if err != nil {
		t.Fatalf("NewSupervisor: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if err := sup.Start(ctx); err != nil {
		t.Fatalf("Start with missing component binary: %v (want success with degraded broker)", err)
	}
	defer func() {
		_ = sup.Shutdown(context.Background())
	}()

	brokerStatus, ok := sup.EvaluateHealth().Subsystems[SubsystemMCPBroker]
	if !ok {
		t.Fatal("mcp_broker subsystem missing from health snapshot")
	}
	if brokerStatus.Ready {
		t.Fatal("mcp_broker ready=true with missing component binary; want not-ready")
	}
	if !strings.Contains(brokerStatus.Error, ghostID) {
		t.Fatalf("mcp_broker error %q does not name component %q", brokerStatus.Error, ghostID)
	}

	foundDegraded := false
	for _, e := range sup.Memory().Entries() {
		if e.Type != "component.degraded" {
			continue
		}
		if strings.Contains(string(e.Data), ghostID) {
			foundDegraded = true
		}
	}
	if !foundDegraded {
		t.Fatalf("Memory() has no Type:component.degraded entry naming %q", ghostID)
	}

	for _, tool := range sup.Broker().ListTools() {
		if tool.ServerID == ghostID || strings.Contains(tool.Name, ghostID) {
			t.Fatalf("fabricated tool %q registered for degraded server %q; want none", tool.Name, ghostID)
		}
	}
}

// TestContractMockEchoOptIn documents the intentional opt-in: explicit mock
// mode still registers the echo stand-in (guards #12616 behavior).
func TestContractMockEchoOptIn(t *testing.T) {
	const ghostID = "ghost-mcp-server"
	lockFile := ghostComponentLock(t)
	sup, err := NewSupervisor(Config{
		LockPath: lockFile,
		Addr:     "127.0.0.1:0",
		Engine:   "mock",
		Mock:     true,
	})
	if err != nil {
		t.Fatalf("NewSupervisor: %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if err := sup.Start(ctx); err != nil {
		t.Fatalf("Start in explicit mock mode: %v", err)
	}
	defer func() {
		_ = sup.Shutdown(context.Background())
	}()

	want := mcpbroker.NamespaceTool(ghostID, "echo")
	names := []string{}
	found := false
	for _, tool := range sup.Broker().ListTools() {
		names = append(names, tool.Name)
		if tool.Name == want {
			found = true
		}
	}
	if !found {
		t.Fatalf("explicit mock mode did not register echo stand-in %q; tools=%v", want, names)
	}
}

// TestDisclosureInMemoryVsDurable pins the lifecycle disclosure contract:
// in-memory stores record a Type:"lifecycle.disclosure" entry naming the
// in-memory store, while durable journal paths record none.
func TestDisclosureInMemoryVsDurable(t *testing.T) {
	t.Run("in_memory_emits_disclosure", func(t *testing.T) {
		lockFile := writeContractLock(t, `[]`, `[]`)
		sup, err := NewSupervisor(Config{
			LockPath: lockFile,
			Addr:     "127.0.0.1:0",
			Engine:   "mock",
			Mock:     true,
		})
		if err != nil {
			t.Fatalf("NewSupervisor: %v", err)
		}
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		if err := sup.Start(ctx); err != nil {
			t.Fatalf("Start: %v", err)
		}
		defer func() {
			_ = sup.Shutdown(context.Background())
		}()

		found := false
		for _, e := range sup.Memory().Entries() {
			if e.Type != "lifecycle.disclosure" {
				continue
			}
			var payload map[string]string
			if err := json.Unmarshal(e.Data, &payload); err != nil {
				t.Fatalf("unmarshal disclosure data: %v", err)
			}
			if payload["memory_store"] == "in-memory" {
				found = true
			}
		}
		if !found {
			t.Fatal("in-memory store produced no lifecycle.disclosure entry with memory_store in-memory")
		}
	})

	t.Run("durable_journal_emits_no_disclosure", func(t *testing.T) {
		journalFile := filepath.Join(t.TempDir(), "memory-journal.jsonl")
		assets := `[{"kind":"memory","id":"file-journal","value":` + jsonQuote(journalFile) + `}]`
		lockFile := writeContractLock(t, `[]`, assets)
		sup, err := NewSupervisor(Config{
			LockPath: lockFile,
			Addr:     "127.0.0.1:0",
			Engine:   "mock",
			Mock:     true,
		})
		if err != nil {
			t.Fatalf("NewSupervisor: %v", err)
		}
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		if err := sup.Start(ctx); err != nil {
			t.Fatalf("Start: %v", err)
		}
		defer func() {
			_ = sup.Shutdown(context.Background())
		}()

		for _, e := range sup.Memory().Entries() {
			if e.Type == "lifecycle.disclosure" {
				t.Fatalf("durable journal path produced unexpected lifecycle.disclosure entry: %s", string(e.Data))
			}
		}
	})
}

const (
	contractEnvServerID  = "contract-env-server"
	contractEnvHelperVar = "FAK_CONTRACT_ENV_HELPER"
	contractEnvServerVar = "FAK_CONTRACT_ENV_SERVER"
	contractEnvKey       = "FAK_CONTRACT_ENV_VALUE"
	contractParentMarker = "FAK_CONTRACT_PARENT_MARKER"
)

func init() {
	if os.Getenv(contractEnvHelperVar) == "1" && os.Getenv(contractEnvServerVar) == contractEnvServerID {
		runContractEnvHelper()
		os.Exit(0)
	}
}

// TestContractEnvHelper is the helper subprocess entrypoint for production
// child-env tests. In the parent test process it always skips.
func TestContractEnvHelper(t *testing.T) {
	if os.Getenv(contractEnvHelperVar) != "1" || os.Getenv(contractEnvServerVar) != contractEnvServerID {
		t.Skip("helper process only")
		return
	}
	runContractEnvHelper()
	os.Exit(0)
}

// runContractEnvHelper serves the minimal MCP stdio protocol and reports the
// child process environment back through the probe_env tool.
func runContractEnvHelper() {
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
								"serverInfo":      map[string]any{"name": contractEnvServerID, "version": "1.0.0"},
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
										"name":        "probe_env",
										"description": "Reports child process environment for contract tests",
										"inputSchema": map[string]any{"type": "object"},
									},
								},
							},
						}
						data, _ := json.Marshal(resp)
						_, _ = os.Stdout.Write(append(data, '\n'))

					case "tools/call":
						text := "ENV:" + os.Getenv(contractEnvKey) +
							";PARENT:" + os.Getenv(contractParentMarker) +
							";PATH:" + os.Getenv("PATH")
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

// writeContractEnvLock returns a lock whose MCP component launches this test
// binary as the helper child, mirroring the TestStructuredMCPHelper pattern.
func writeContractEnvLock(t *testing.T) string {
	t.Helper()
	exe, err := os.Executable()
	if err != nil {
		exe = os.Args[0]
	}
	return writeContractLock(t, `[
    {
      "id": "`+contractEnvServerID+`",
      "version": "1.0.0",
      "digest": "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
      "source": `+jsonQuote(exe)+`,
      "provider": "mcp",
      "provides": ["probe_env"],
      "adapters": ["-test.run=TestContractEnvHelper"]
    }
  ]`, `[]`)
}

func contractProbeChild(t *testing.T, sup *Supervisor) string {
	t.Helper()
	toolName := mcpbroker.NamespaceTool(contractEnvServerID, "probe_env")
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	resp, err := sup.Broker().RouteCall(ctx, mcpbroker.CallRequest{
		Tool:      toolName,
		Arguments: json.RawMessage(`{}`),
	})
	if err != nil {
		t.Fatalf("RouteCall %q: %v", toolName, err)
	}
	return extractTextBlock(t, resp.Content)
}

// TestComponentEnvReachesChild pins production child env plumbing: ComponentEnv
// entries reach the launched helper child, and os.Environ inheritance still
// works when ComponentEnv is nil.
func TestComponentEnvReachesChild(t *testing.T) {
	if os.Getenv(contractEnvHelperVar) == "1" && os.Getenv(contractEnvServerVar) == contractEnvServerID {
		runContractEnvHelper()
		os.Exit(0)
	}

	t.Run("component_env_reaches_child", func(t *testing.T) {
		const sentinel = "sentinel-9f27-component-env"
		if _, ok := os.LookupEnv(contractEnvKey); ok {
			t.Skip("parent environment already defines " + contractEnvKey)
		}
		lockFile := writeContractEnvLock(t)
		sup, err := NewSupervisor(Config{
			LockPath:     lockFile,
			Addr:         "127.0.0.1:0",
			EngineDriver: contractStubEngine{},
			ComponentEnv: []string{
				contractEnvHelperVar + "=1",
				contractEnvServerVar + "=" + contractEnvServerID,
				contractEnvKey + "=" + sentinel,
			},
		})
		if err != nil {
			t.Fatalf("NewSupervisor: %v", err)
		}
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		if err := sup.Start(ctx); err != nil {
			t.Fatalf("Start: %v", err)
		}
		defer func() {
			_ = sup.Shutdown(context.Background())
		}()

		if text := contractProbeChild(t, sup); !strings.Contains(text, "ENV:"+sentinel) {
			t.Fatalf("child did not observe ComponentEnv %s=%q; probe=%q", contractEnvKey, sentinel, text)
		}
	})

	t.Run("environ_inheritance_without_component_env", func(t *testing.T) {
		const parentValue = "parent-marker-4b81-inheritance"
		t.Setenv(contractEnvHelperVar, "1")
		t.Setenv(contractEnvServerVar, contractEnvServerID)
		t.Setenv(contractParentMarker, parentValue)
		parentPath := os.Getenv("PATH")
		if parentPath == "" {
			t.Fatal("parent PATH is empty; cannot verify inheritance")
		}
		lockFile := writeContractEnvLock(t)
		sup, err := NewSupervisor(Config{
			LockPath:     lockFile,
			Addr:         "127.0.0.1:0",
			EngineDriver: contractStubEngine{},
		})
		if err != nil {
			t.Fatalf("NewSupervisor: %v", err)
		}
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		if err := sup.Start(ctx); err != nil {
			t.Fatalf("Start: %v", err)
		}
		defer func() {
			_ = sup.Shutdown(context.Background())
		}()

		text := contractProbeChild(t, sup)
		if !strings.Contains(text, "PARENT:"+parentValue) {
			t.Fatalf("child did not inherit parent env marker; probe=%q", text)
		}
		if !strings.Contains(text, "PATH:"+parentPath) {
			t.Fatalf("child did not inherit PATH with nil ComponentEnv; probe len=%d", len(text))
		}
	})
}

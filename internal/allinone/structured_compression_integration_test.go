package allinone

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/refutil"
)

const (
	testHelperEnv = "GO_WANT_HELPER_PROCESS"
	testServerID  = "structured-mcp-server"
)

func init() {
	if os.Getenv(testHelperEnv) == "1" && os.Getenv("MOCK_SERVER_ID") == testServerID {
		runStructuredMCPHelper()
		os.Exit(0)
	}
}

// TestStructuredMCPHelper acts as the test target when the subprocess is invoked with -test.run.
func TestStructuredMCPHelper(t *testing.T) {
	if os.Getenv(testHelperEnv) != "1" || os.Getenv("MOCK_SERVER_ID") != testServerID {
		t.Skip("helper process only")
		return
	}
	runStructuredMCPHelper()
	os.Exit(0)
}

func runStructuredMCPHelper() {
	pretty := "{\n" + strings.Repeat(" ", 100) + `"n":9007199254740993,"n":1e+09,"s":"\u0061  b"` + "\n}"
	compact := `{"n":9007199254740993,"n":1e+09,"s":"\u0061  b"}`

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
								"serverInfo":      map[string]any{"name": testServerID, "version": "1.0.0"},
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
										"name":        "query_structured",
										"description": "Returns structured test data",
										"inputSchema": map[string]any{"type": "object"},
									},
								},
							},
						}
						data, _ := json.Marshal(resp)
						_, _ = os.Stdout.Write(append(data, '\n'))

					case "tools/call":
						resp := map[string]any{
							"jsonrpc": "2.0",
							"id":      req.ID,
							"result": map[string]any{
								"content": []map[string]any{
									{
										"type": "text",
										"text": pretty,
									},
								},
								"structuredContent": json.RawMessage(compact),
								"isError":           false,
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

type recordingEngine struct {
	mu    sync.Mutex
	calls []*abi.ToolCall
}

func (r *recordingEngine) Complete(ctx context.Context, c *abi.ToolCall) (*abi.Result, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	callCopy := *c
	if c.Args.Kind == abi.RefInline {
		inlineCopy := make([]byte, len(c.Args.Inline))
		copy(inlineCopy, c.Args.Inline)
		callCopy.Args.Inline = inlineCopy
	}
	r.calls = append(r.calls, &callCopy)

	body := `{"status":"admitted","tool":"` + c.Tool + `"}`
	return &abi.Result{
		Call:    c,
		Payload: abi.Ref{Kind: abi.RefInline, Inline: []byte(body), Len: int64(len(body))},
		Status:  abi.StatusOK,
	}, nil
}

func (r *recordingEngine) Caps() []abi.Capability { return nil }

func (r *recordingEngine) LastCall() *abi.ToolCall {
	r.mu.Lock()
	defer r.mu.Unlock()
	if len(r.calls) == 0 {
		return nil
	}
	return r.calls[len(r.calls)-1]
}

func TestStructuredMCPResultReachesProvider(t *testing.T) {
	// Subprocess entrypoint guard: if spawned as helper, never run test body
	if os.Getenv(testHelperEnv) == "1" && os.Getenv("MOCK_SERVER_ID") == testServerID {
		runStructuredMCPHelper()
		os.Exit(0)
	}

	exe, err := os.Executable()
	if err != nil {
		exe = os.Args[0]
	}

	setupSupervisorWithEngine := func(t *testing.T, eng abi.EngineDriver) (*Supervisor, string, func()) {
		t.Helper()
		dir := t.TempDir()
		journalFile := filepath.Join(dir, "memory-journal.jsonl")
		lockFile := filepath.Join(dir, "harness.lock.json")

		lockContent := `{
  "schema": "fak.harness-product-lock/v2",
  "id": "sha256:1111111111111111111111111111111111111111111111111111111111111111",
  "platforms": [
    {"os": ` + jsonQuote(runtime.GOOS) + `, "arch": ` + jsonQuote(runtime.GOARCH) + `}
  ],
  "budget": {
    "context_tokens": 4096,
    "memory_mib": 512,
    "workers": 1
  },
  "components": [
    {
      "id": ` + jsonQuote(testServerID) + `,
      "version": "1.0.0",
      "digest": "sha256:2222222222222222222222222222222222222222222222222222222222222222",
      "source": ` + jsonQuote(exe) + `,
      "provider": "mcp",
      "provides": ["query_structured"],
      "adapters": ["-test.run=TestStructuredMCPHelper"]
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
			EngineDriver: eng,
		}

		sup, err := NewSupervisor(cfg)
		if err != nil {
			t.Fatalf("NewSupervisor: %v", err)
		}

		ctx, cancel := context.WithCancel(context.Background())
		if err := sup.Start(ctx); err != nil {
			cancel()
			t.Fatalf("Start supervisor: %v", err)
		}

		cleanup := func() {
			cancel()
			_ = sup.Shutdown(context.Background())
		}
		return sup, "http://" + sup.Addr(), cleanup
	}

	callAgentSession := func(t *testing.T, baseURL string) ([]map[string]any, int) {
		t.Helper()
		reqBody := fmt.Sprintf(`{"goal":"query structured stats","tool":%q,"args":{}}`, "mcp__"+testServerID+"__query_structured")
		resp, err := http.Post(baseURL+"/v1/fak/agent/sessions", "application/json", strings.NewReader(reqBody))
		if err != nil {
			t.Fatalf("POST agent session: %v", err)
		}
		defer resp.Body.Close()

		var events []map[string]any
		scan := bufio.NewScanner(resp.Body)
		for scan.Scan() {
			var ev map[string]any
			if err := json.Unmarshal(scan.Bytes(), &ev); err == nil {
				events = append(events, ev)
			}
		}
		return events, resp.StatusCode
	}

	var defaultPayload []byte
	var noopPayload []byte

	// 1. Ordinary default configuration: structured MCP content must be demonstrably compacted.
	t.Run("default_compacted", func(t *testing.T) {
		t.Setenv("FAK_COMPRESSOR", "")

		recEng := &recordingEngine{}
		sup, baseURL, cleanup := setupSupervisorWithEngine(t, recEng)
		defer cleanup()

		// Verify child process tracking for active supervised subprocess
		tracked := sup.TrackChildProcesses()
		procInfo, ok := tracked[testServerID]
		if !ok || !procInfo.Running || procInfo.PID <= 0 {
			t.Fatalf("expected running child process tracked for %q: %+v", testServerID, procInfo)
		}
		if statusInfo, ok := sup.ChildProcessStatus(testServerID); !ok || statusInfo.PID != procInfo.PID {
			t.Fatalf("ChildProcessStatus mismatch: got %+v, want PID %d", statusInfo, procInfo.PID)
		}

		events, status := callAgentSession(t, baseURL)
		if status != http.StatusOK {
			t.Fatalf("unexpected HTTP status: %d", status)
		}

		lastCall := recEng.LastCall()
		if lastCall == nil {
			t.Fatal("provider received no ToolCall from the supervisor loop")
		}

		payload := refutil.Bytes(context.Background(), lastCall.Args)
		if len(payload) == 0 {
			t.Fatal("provider request carried empty args payload")
		}
		defaultPayload = payload

		// Verify compaction: the 100 whitespace spaces must be stripped
		if bytes.Contains(payload, bytes.Repeat([]byte(" "), 100)) {
			t.Fatalf("expected compacted content to strip insignificant whitespace, got %s", string(payload))
		}
		// Must retain JSON token fidelity in decoded block
		text := extractTextBlock(t, payload)
		if !strings.Contains(text, `"n":9007199254740993`) {
			t.Fatalf("compacted payload missing required JSON token: %s", text)
		}

		// Verify event stream delivered the compacted content to caller
		var seenCall, seenStep bool
		for _, ev := range events {
			if ev["event"] == "call" {
				seenCall = true
				resStr, _ := json.Marshal(ev["result"])
				if !bytes.Equal(bytes.TrimSpace(resStr), bytes.TrimSpace(payload)) {
					t.Fatalf("call event result mismatch with provider payload:\ngot: %s\nwant: %s", string(resStr), string(payload))
				}
			}
			if ev["event"] == "step" {
				seenStep = true
			}
		}
		if !seenCall {
			t.Fatal("expected call event in session NDJSON stream")
		}
		if !seenStep {
			t.Fatal("expected step event from provider completion in session NDJSON stream")
		}
	})

	// 2. Noop control: FAK_COMPRESSOR=noop preserves uncompressed raw bytes.
	t.Run("noop_control", func(t *testing.T) {
		t.Setenv("FAK_COMPRESSOR", "noop")

		recEng := &recordingEngine{}
		_, baseURL, cleanup := setupSupervisorWithEngine(t, recEng)
		defer cleanup()

		_, status := callAgentSession(t, baseURL)
		if status != http.StatusOK {
			t.Fatalf("unexpected HTTP status: %d", status)
		}

		lastCall := recEng.LastCall()
		if lastCall == nil {
			t.Fatal("provider received no ToolCall under noop control")
		}

		payload := refutil.Bytes(context.Background(), lastCall.Args)
		if len(payload) == 0 {
			t.Fatal("provider request carried empty args payload under noop")
		}
		noopPayload = payload

		// Under noop, the 100 whitespace spaces MUST be preserved
		if !bytes.Contains(payload, bytes.Repeat([]byte(" "), 100)) {
			t.Fatalf("FAK_COMPRESSOR=noop did not preserve raw uncompressed bytes: %s", string(payload))
		}

		if len(defaultPayload) >= len(noopPayload) {
			t.Fatalf("expected default compacted payload (%d bytes) < noop uncompressed payload (%d bytes)", len(defaultPayload), len(noopPayload))
		}
	})

	// 3. Unchanged payload semantics: JSON values must be identical between default and noop.
	t.Run("unchanged_payload_semantics", func(t *testing.T) {
		if len(defaultPayload) == 0 || len(noopPayload) == 0 {
			t.Fatal("missing default or noop payloads for semantic comparison")
		}

		textDefault := extractTextBlock(t, defaultPayload)
		textNoop := extractTextBlock(t, noopPayload)

		var objDefault, objNoop any
		if err := json.Unmarshal([]byte(textDefault), &objDefault); err != nil {
			t.Fatalf("unmarshal default JSON: %v (text: %s)", err, textDefault)
		}
		if err := json.Unmarshal([]byte(textNoop), &objNoop); err != nil {
			t.Fatalf("unmarshal noop JSON: %v (text: %s)", err, textNoop)
		}

		if !reflect.DeepEqual(objDefault, objNoop) {
			t.Fatalf("payload semantics changed between compressed and uncompressed:\ndefault: %+v\nnoop:    %+v", objDefault, objNoop)
		}
	})
}

func extractTextBlock(t *testing.T, raw []byte) string {
	t.Helper()
	var blocks []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(raw, &blocks); err != nil {
		t.Fatalf("unmarshal blocks: %v in %s", err, string(raw))
	}
	if len(blocks) != 1 || blocks[0].Type != "text" {
		t.Fatalf("unexpected content blocks: %+v", blocks)
	}
	return blocks[0].Text
}

package mcpbroker

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"
)

// TestHelperProcess implements the mock MCP server subprocess pattern for stdio testing.
// It is invoked when the test binary is executed with -test.run=TestHelperProcess and
// the environment variable GO_WANT_HELPER_PROCESS=1 is set.
func TestHelperProcess(t *testing.T) {
	if os.Getenv("GO_WANT_HELPER_PROCESS") != "1" {
		return
	}
	defer os.Exit(0)

	// If heavy stderr requested, emit >64KB to verify pipe drain does not block.
	if os.Getenv("FAKE_MCP_HEAVY_STDERR") == "1" {
		chunk := bytes.Repeat([]byte("HEAVY-STDERR-TELEMETRY-LINE-DATA-PADDING\n"), 3000) // ~120KB > 64KB
		_, _ = os.Stderr.Write(chunk)
	}

	useContentLength := os.Getenv("FAKE_MCP_FRAMING") == "content-length"

	reader := bufio.NewReader(os.Stdin)
	for {
		// Skip leading whitespace / blank lines
		for {
			b, err := reader.Peek(1)
			if err != nil {
				return
			}
			if b[0] == '\r' || b[0] == '\n' || b[0] == ' ' || b[0] == '\t' {
				_, _ = reader.ReadByte()
				continue
			}
			break
		}

		b, err := reader.Peek(1)
		if err != nil {
			return
		}

		var reqBytes []byte
		if b[0] == '{' || b[0] == '[' {
			line, err := reader.ReadBytes('\n')
			if len(line) > 0 {
				reqBytes = bytes.TrimSpace(line)
			}
			if err != nil && len(reqBytes) == 0 {
				return
			}
		} else {
			var contentLength int = -1
			for {
				line, err := reader.ReadString('\n')
				if err != nil {
					return
				}
				trimmed := strings.TrimRight(line, "\r\n")
				if trimmed == "" {
					break
				}
				parts := strings.SplitN(trimmed, ":", 2)
				if len(parts) == 2 && strings.EqualFold(strings.TrimSpace(parts[0]), "content-length") {
					val, err := strconv.Atoi(strings.TrimSpace(parts[1]))
					if err == nil && val >= 0 {
						contentLength = val
					}
				}
			}
			if contentLength > 0 {
				payload := make([]byte, contentLength)
				_, err := io.ReadFull(reader, payload)
				if err != nil {
					return
				}
				reqBytes = bytes.TrimSpace(payload)
			}
		}

		if len(reqBytes) == 0 {
			continue
		}

		var req struct {
			JSONRPC string          `json:"jsonrpc"`
			ID      json.RawMessage `json:"id"`
			Method  string          `json:"method"`
			Params  json.RawMessage `json:"params"`
		}
		if err := json.Unmarshal(reqBytes, &req); err != nil {
			continue
		}

		switch req.Method {
		case "initialize":
			resp := map[string]interface{}{
				"jsonrpc": "2.0",
				"id":      req.ID,
				"result": map[string]interface{}{
					"protocolVersion": "2024-11-05",
					"capabilities": map[string]interface{}{
						"tools": map[string]interface{}{},
					},
					"serverInfo": map[string]interface{}{
						"name":    "mock-mcp-server",
						"version": "1.0.0",
					},
				},
			}
			writeHelperResponse(resp, useContentLength)

		case "notifications/initialized":
			// Notification has no response

		case "tools/list":
			toolsList := []map[string]interface{}{
				{
					"name":        "read_file",
					"description": "Reads file contents",
					"inputSchema": map[string]interface{}{"type": "object"},
				},
				{
					"name":        "echo",
					"description": "Echoes input arguments",
					"inputSchema": map[string]interface{}{"type": "object"},
				},
				{
					"name":        "greet",
					"description": "Returns a greeting",
					"inputSchema": map[string]interface{}{"type": "object"},
				},
			}
			if customTools := os.Getenv("FAKE_MCP_TOOLS"); customTools != "" {
				toolsList = nil
				for _, name := range strings.Split(customTools, ",") {
					name = strings.TrimSpace(name)
					if name != "" {
						toolsList = append(toolsList, map[string]interface{}{
							"name":        name,
							"description": "Tool " + name,
							"inputSchema": map[string]interface{}{"type": "object"},
						})
					}
				}
			}

			resp := map[string]interface{}{
				"jsonrpc": "2.0",
				"id":      req.ID,
				"result": map[string]interface{}{
					"tools": toolsList,
				},
			}
			writeHelperResponse(resp, useContentLength)

		case "tools/call":
			if os.Getenv("FAKE_MCP_CRASH_ON_CALL") == "1" {
				os.Exit(2)
			}

			var callParams struct {
				Name      string          `json:"name"`
				Arguments json.RawMessage `json:"arguments"`
			}
			_ = json.Unmarshal(req.Params, &callParams)

			serverID := os.Getenv("MOCK_SERVER_ID")
			if serverID == "" {
				serverID = "mock"
			}

			resp := map[string]interface{}{
				"jsonrpc": "2.0",
				"id":      req.ID,
				"result": map[string]interface{}{
					"content": []map[string]interface{}{
						{
							"type": "text",
							"text": fmt.Sprintf("executed %s on %s with %s", callParams.Name, serverID, string(callParams.Arguments)),
						},
					},
					"isError": false,
				},
			}
			writeHelperResponse(resp, useContentLength)

		case "ping":
			resp := map[string]interface{}{
				"jsonrpc": "2.0",
				"id":      req.ID,
				"result":  map[string]interface{}{},
			}
			writeHelperResponse(resp, useContentLength)
		}
	}
}

func writeHelperResponse(resp interface{}, useContentLength bool) {
	data, err := json.Marshal(resp)
	if err != nil {
		return
	}

	if useContentLength {
		header := fmt.Sprintf("Content-Length: %d\r\n\r\n", len(data))
		_, _ = os.Stdout.Write([]byte(header))
		_, _ = os.Stdout.Write(data)
	} else {
		data = append(data, '\n')
		_, _ = os.Stdout.Write(data)
	}
}

// helperServerConfig builds a ServerConfig that runs TestHelperProcess as a child MCP subprocess.
func helperServerConfig(t *testing.T, id string, env ...string) ServerConfig {
	t.Helper()
	exe, err := os.Executable()
	if err != nil {
		exe = os.Args[0]
	}

	mergedEnv := append([]string{
		"GO_WANT_HELPER_PROCESS=1",
		"MOCK_SERVER_ID=" + id,
	}, env...)

	return ServerConfig{
		ID:      id,
		Name:    "Server " + id,
		Command: exe,
		Args:    []string{"-test.run=TestHelperProcess", "--"},
		Env:     mergedEnv,
		Timeout: 5 * time.Second,
	}
}

// TestMCPBrokerStdioSupervision tests process launch, JSON-RPC 2.0 handshake,
// tool discovery, namespaced registration, execution, and graceful termination.
func TestMCPBrokerStdioSupervision(t *testing.T) {
	b := NewBroker()
	defer b.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	cfg := helperServerConfig(t, "srv1")

	sup, err := b.LaunchSupervisor(ctx, cfg)
	if err != nil {
		t.Fatalf("failed to launch supervisor: %v", err)
	}
	if !sup.IsRunning() {
		t.Fatalf("expected supervisor to be running")
	}

	// Verify tools discovered and registered with namespacing
	tools := b.ListTools()
	expectedTools := map[string]bool{
		"mcp__srv1__read_file": false,
		"mcp__srv1__echo":      false,
		"mcp__srv1__greet":     false,
	}
	for _, tool := range tools {
		if _, ok := expectedTools[tool.Name]; ok {
			expectedTools[tool.Name] = true
		}
	}
	for name, found := range expectedTools {
		if !found {
			t.Errorf("expected namespaced tool %q not found in broker", name)
		}
	}

	// Verify tool execution
	callResp, err := b.RouteCall(ctx, CallRequest{
		Tool:      "mcp__srv1__echo",
		Arguments: json.RawMessage(`{"text":"hello world"}`),
	})
	if err != nil {
		t.Fatalf("unexpected call error: %v", err)
	}
	if callResp.IsError || callResp.Filtered {
		t.Fatalf("expected clean response, got isError=%v filtered=%v", callResp.IsError, callResp.Filtered)
	}
	if !strings.Contains(string(callResp.Content), "hello world") {
		t.Fatalf("expected content to contain 'hello world', got: %s", string(callResp.Content))
	}
	if callResp.ServerID != "srv1" {
		t.Fatalf("expected ServerID 'srv1', got %q", callResp.ServerID)
	}

	// Test Ping
	if err := sup.Ping(ctx); err != nil {
		t.Fatalf("expected ping to succeed, got: %v", err)
	}

	// Graceful shutdown via StopSupervisor
	if err := b.StopSupervisor("srv1"); err != nil {
		t.Fatalf("failed to stop supervisor: %v", err)
	}
	if sup.IsRunning() {
		t.Fatalf("expected supervisor to not be running after stop")
	}

	// Verify tools were unregistered on stop
	postTools := b.ListTools()
	for _, tool := range postTools {
		if tool.ServerID == "srv1" {
			t.Errorf("found lingering tool %q after server stopped", tool.Name)
		}
	}
}

// TestMCPBrokerStdio exercises all stdio supervision capabilities via structured subtests.
func TestMCPBrokerStdio(t *testing.T) {
	t.Run("HandshakeAndDiscovery", testStdioHandshakeAndDiscovery)
	t.Run("NamespacedToolCollisionFree", testStdioNamespacedToolCollisionFree)
	t.Run("RouteCallExecution", testStdioRouteCallExecution)
	t.Run("LargeStderrNoDeadlock", testStdioLargeStderrNoDeadlock)
	t.Run("DuplicateRegistrationError", testStdioDuplicateRegistrationError)
	t.Run("ChaosProcessCrash", testStdioChaosProcessCrash)
	t.Run("ContextCancellationAndStop", testStdioContextCancellationAndStop)
	t.Run("FramingContentLength", testStdioFramingContentLength)
	t.Run("ServerPolicies", testStdioServerPolicies)
	t.Run("RestartBackoff", testStdioRestartBackoff)
}

func testStdioHandshakeAndDiscovery(t *testing.T) {
	b := NewBroker()
	defer b.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	cfg := helperServerConfig(t, "srv_handshake")

	sup, err := b.LaunchSupervisor(ctx, cfg)
	if err != nil {
		t.Fatalf("handshake and discovery failed: %v", err)
	}
	defer sup.Stop()

	if !sup.IsRunning() {
		t.Fatalf("expected supervisor to be running")
	}

	tools := sup.DiscoveredTools()
	if len(tools) < 2 {
		t.Fatalf("expected at least 2 discovered tools, got %d", len(tools))
	}

	foundReadFile := false
	for _, tool := range tools {
		if tool.Name == "read_file" {
			foundReadFile = true
			break
		}
	}
	if !foundReadFile {
		t.Fatalf("expected tool 'read_file' in discovered tools: %+v", tools)
	}
}

func testStdioNamespacedToolCollisionFree(t *testing.T) {
	b := NewBroker()
	defer b.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	cfgA := helperServerConfig(t, "srvA")
	cfgB := helperServerConfig(t, "srvB")

	_, err := b.LaunchSupervisor(ctx, cfgA)
	if err != nil {
		t.Fatalf("failed to launch server A: %v", err)
	}
	_, err = b.LaunchSupervisor(ctx, cfgB)
	if err != nil {
		t.Fatalf("failed to launch server B: %v", err)
	}

	// Both servers expose "read_file", registered as mcp__srvA__read_file and mcp__srvB__read_file
	respA, err := b.RouteCall(ctx, CallRequest{
		Tool:      "mcp__srvA__read_file",
		Arguments: json.RawMessage(`{"path":"/a.txt"}`),
	})
	if err != nil || respA.IsError {
		t.Fatalf("call to srvA failed: %v", err)
	}
	if respA.ServerID != "srvA" {
		t.Fatalf("expected ServerID srvA, got %s", respA.ServerID)
	}
	if !strings.Contains(string(respA.Content), "srvA") {
		t.Fatalf("expected content from srvA, got: %s", string(respA.Content))
	}

	respB, err := b.RouteCall(ctx, CallRequest{
		Tool:      "mcp__srvB__read_file",
		Arguments: json.RawMessage(`{"path":"/b.txt"}`),
	})
	if err != nil || respB.IsError {
		t.Fatalf("call to srvB failed: %v", err)
	}
	if respB.ServerID != "srvB" {
		t.Fatalf("expected ServerID srvB, got %s", respB.ServerID)
	}
	if !strings.Contains(string(respB.Content), "srvB") {
		t.Fatalf("expected content from srvB, got: %s", string(respB.Content))
	}
}

func testStdioRouteCallExecution(t *testing.T) {
	b := NewBroker()
	defer b.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	cfg := helperServerConfig(t, "srv_exec")
	sup, err := b.LaunchSupervisor(ctx, cfg)
	if err != nil {
		t.Fatalf("failed to launch supervisor: %v", err)
	}
	defer sup.Stop()

	resp, err := b.RouteCall(ctx, CallRequest{
		Tool:      "mcp__srv_exec__read_file",
		Arguments: json.RawMessage(`{"path":"/var/log/app.log"}`),
	})
	if err != nil {
		t.Fatalf("RouteCall failed: %v", err)
	}
	if resp == nil || resp.IsError {
		t.Fatalf("expected successful response, got %+v", resp)
	}
	if resp.Tool != "mcp__srv_exec__read_file" {
		t.Fatalf("expected namespaced tool name %q, got %q", "mcp__srv_exec__read_file", resp.Tool)
	}
	if resp.ServerID != "srv_exec" {
		t.Fatalf("expected serverID %q, got %q", "srv_exec", resp.ServerID)
	}

	stats := b.Stats()
	if stats.AllowedCalls < 1 {
		t.Fatalf("expected allowed calls >= 1, got %d", stats.AllowedCalls)
	}
}

func testStdioLargeStderrNoDeadlock(t *testing.T) {
	b := NewBroker()
	defer b.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	cfg := helperServerConfig(t, "srv_stderr", "FAKE_MCP_HEAVY_STDERR=1")

	sup, err := b.LaunchSupervisor(ctx, cfg)
	if err != nil {
		t.Fatalf("failed to launch supervisor with heavy stderr (>64KB): %v", err)
	}
	defer sup.Stop()

	// Tool call to ensure pipe is open and server is interactive
	resp, err := b.RouteCall(ctx, CallRequest{
		Tool:      "mcp__srv_stderr__echo",
		Arguments: json.RawMessage(`{"check":"draining"}`),
	})
	if err != nil || resp.IsError {
		t.Fatalf("tool call failed: %v, resp=%+v", err, resp)
	}

	// Verify stderr buffer captured data >64KB
	stderrBytes := sup.Stderr()
	if len(stderrBytes) < 65536 {
		t.Fatalf("expected captured stderr bytes >= 65536 (>64KB), got %d", len(stderrBytes))
	}
	if !bytes.Contains(stderrBytes, []byte("HEAVY-STDERR-TELEMETRY")) {
		t.Fatalf("expected stderr telemetry in captured buffer")
	}
}

func testStdioDuplicateRegistrationError(t *testing.T) {
	b := NewBroker()
	defer b.Close()

	tool := ToolRegistration{
		Name:        "duplicate_target_tool",
		Description: "Standalone tool for duplicate check",
	}

	if err := b.RegisterTool(tool); err != nil {
		t.Fatalf("first registration failed: %v", err)
	}

	err := b.RegisterTool(tool)
	if !errors.Is(err, ErrToolAlreadyRegistered) {
		t.Fatalf("expected ErrToolAlreadyRegistered on duplicate tool, got: %v", err)
	}
}

func testStdioChaosProcessCrash(t *testing.T) {
	b := NewBroker()
	defer b.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	cfg := helperServerConfig(t, "srv_crash", "FAKE_MCP_CRASH_ON_CALL=1")
	zeroRestarts := 0
	cfg.MaxRestarts = &zeroRestarts

	sup, err := b.LaunchSupervisor(ctx, cfg)
	if err != nil {
		t.Fatalf("failed to launch supervisor: %v", err)
	}

	// Trigger tool call which causes subprocess to exit(2)
	resp, err := b.RouteCall(ctx, CallRequest{
		Tool:      "mcp__srv_crash__echo",
		Arguments: json.RawMessage(`{"trigger":"crash"}`),
	})

	if err == nil && (resp == nil || !resp.IsError) {
		t.Fatalf("expected call to fail due to process exit, got resp: %+v", resp)
	}

	if !IsProcessCrash(err) && (resp == nil || !strings.Contains(resp.ErrorMessage, MCPProcessCrashCode)) {
		t.Fatalf("expected MCP_PROCESS_CRASH error or response, got err=%v resp=%+v", err, resp)
	}

	// Subsequent ping must return ErrMCPProcessCrash
	pingErr := sup.Ping(ctx)
	if !IsProcessCrash(pingErr) {
		t.Fatalf("expected ping after crash to return MCP_PROCESS_CRASH, got: %v", pingErr)
	}

	// Host broker must remain fully alive and responsive
	survivorTool := ToolRegistration{
		Name: "survivor_tool",
	}
	if err := b.RegisterTool(survivorTool); err != nil {
		t.Fatalf("expected host broker to accept registrations after child crash, got %v", err)
	}
	survResp, err := b.RouteCall(ctx, CallRequest{Tool: "survivor_tool"})
	if err != nil || survResp.IsError {
		t.Fatalf("expected host broker to route calls after child crash, got err=%v resp=%+v", err, survResp)
	}

	stats := b.Stats()
	if stats.ErrorCalls < 1 {
		t.Fatalf("expected stats to record error calls, got %d", stats.ErrorCalls)
	}

	_ = sup.Stop()
}

func testStdioContextCancellationAndStop(t *testing.T) {
	// 1. Verify Stop() cleanly reaps child process
	t.Run("StopReapsProcess", func(t *testing.T) {
		b := NewBroker()
		defer b.Close()

		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()

		cfg := helperServerConfig(t, "srv_stop")
		sup, err := b.LaunchSupervisor(ctx, cfg)
		if err != nil {
			t.Fatalf("failed to launch supervisor: %v", err)
		}

		pid := sup.PID()
		if pid <= 0 {
			t.Fatalf("expected valid PID > 0, got %d", pid)
		}

		if err := sup.Stop(); err != nil {
			t.Fatalf("failed to stop supervisor: %v", err)
		}

		if sup.IsRunning() {
			t.Fatalf("expected supervisor to not be running after Stop()")
		}
	})

	// 2. Verify Context Cancellation terminates child process
	t.Run("ContextCancellationTerminatesProcess", func(t *testing.T) {
		b := NewBroker()
		defer b.Close()

		cancelCtx, cancelFn := context.WithCancel(context.Background())
		cfg := helperServerConfig(t, "srv_cancel")

		sup, err := b.LaunchSupervisor(cancelCtx, cfg)
		if err != nil {
			cancelFn()
			t.Fatalf("failed to launch supervisor: %v", err)
		}

		pid := sup.PID()
		if pid <= 0 {
			cancelFn()
			t.Fatalf("expected valid PID > 0, got %d", pid)
		}

		// Cancel context to initiate graceful termination
		cancelFn()

		// Poll up to 2 seconds for supervisor to register termination
		deadline := time.Now().Add(2 * time.Second)
		stopped := false
		for time.Now().Before(deadline) {
			if !sup.IsRunning() {
				stopped = true
				break
			}
			time.Sleep(50 * time.Millisecond)
		}

		if !stopped {
			t.Fatalf("expected supervisor to terminate on context cancellation within 2s")
		}
	})
}

func testStdioFramingContentLength(t *testing.T) {
	b := NewBroker()
	defer b.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	cfg := helperServerConfig(t, "srv_framed", "FAKE_MCP_FRAMING=content-length")

	sup, err := b.LaunchSupervisor(ctx, cfg)
	if err != nil {
		t.Fatalf("failed to launch supervisor with content-length framing: %v", err)
	}
	defer sup.Stop()

	resp, err := b.RouteCall(ctx, CallRequest{
		Tool:      "mcp__srv_framed__greet",
		Arguments: json.RawMessage(`{"user":"Alice"}`),
	})
	if err != nil || resp.IsError {
		t.Fatalf("call with content-length framing failed: %v, resp=%+v", err, resp)
	}
	if !strings.Contains(string(resp.Content), "Alice") {
		t.Fatalf("expected response to contain 'Alice', got: %s", string(resp.Content))
	}
}

func testStdioServerPolicies(t *testing.T) {
	b := NewBroker()
	defer b.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	cfg := helperServerConfig(t, "srv_policy")
	cfg.AllowedTools = []string{"echo"}
	cfg.DeniedTools = []string{"greet"}

	sup, err := b.LaunchSupervisor(ctx, cfg)
	if err != nil {
		t.Fatalf("failed to launch supervisor: %v", err)
	}
	defer sup.Stop()

	// "echo" should be registered; "greet" must not be registered
	tools := b.ListTools()
	foundEcho := false
	foundGreet := false
	for _, tool := range tools {
		if tool.Name == "mcp__srv_policy__echo" {
			foundEcho = true
		}
		if tool.Name == "mcp__srv_policy__greet" {
			foundGreet = true
		}
	}

	if !foundEcho {
		t.Errorf("expected allowed tool mcp__srv_policy__echo to be registered")
	}
	if foundGreet {
		t.Errorf("expected denied tool mcp__srv_policy__greet to NOT be registered")
	}
}

func testStdioRestartBackoff(t *testing.T) {
	b := NewBroker()
	defer b.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	cfg := helperServerConfig(t, "srv_restart")

	sup, err := b.LaunchSupervisor(ctx, cfg)
	if err != nil {
		t.Fatalf("failed to launch supervisor: %v", err)
	}
	defer sup.Stop()

	// Kill the child process directly
	pid := sup.PID()
	if pid > 0 {
		p, err := os.FindProcess(pid)
		if err == nil {
			_ = p.Kill()
		}
	}

	// Wait up to 2 seconds for supervisor to restart the process
	restarted := false
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if sup.Restarts() > 0 && sup.IsRunning() {
			restarted = true
			break
		}
		time.Sleep(50 * time.Millisecond)
	}

	if !restarted {
		t.Fatalf("expected supervisor to restart crashed process within 2s, restarts=%d isRunning=%v", sup.Restarts(), sup.IsRunning())
	}

	// Tool call after restart should succeed
	resp, err := b.RouteCall(ctx, CallRequest{
		Tool:      "mcp__srv_restart__echo",
		Arguments: json.RawMessage(`{"post":"restart"}`),
	})
	if err != nil || resp.IsError {
		t.Fatalf("expected call to succeed after restart, got err=%v resp=%+v", err, resp)
	}
}

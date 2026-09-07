package mcpbroker

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestRegisterServer(t *testing.T) {
	b := NewBroker()

	// Empty server ID should fail
	if err := b.RegisterServer(ServerConfig{}); err == nil {
		t.Fatalf("expected error registering server with empty ID, got nil")
	}

	// Valid server registration
	cfg := ServerConfig{
		ID:           "test-server",
		Name:         "Test Server",
		AllowedTools: []string{"read_file", "list_dir"},
		DeniedTools:  []string{"delete_all"},
		ReadOnly:     true,
	}
	if err := b.RegisterServer(cfg); err != nil {
		t.Fatalf("unexpected error registering server: %v", err)
	}

	stats := b.Stats()
	if stats.RegisteredServers != 1 {
		t.Fatalf("expected 1 registered server, got %d", stats.RegisteredServers)
	}
}

func TestRegisterTool(t *testing.T) {
	b := NewBroker()

	// 1. Invalid tool name (empty)
	if err := b.RegisterTool(ToolRegistration{}); !errors.Is(err, ErrInvalidToolName) {
		t.Fatalf("expected ErrInvalidToolName, got %v", err)
	}

	// 2. Register valid standalone tool
	validTool := ToolRegistration{
		Name:        "get_weather",
		Description: "Fetches current weather",
		ReadOnly:    true,
	}
	if err := b.RegisterTool(validTool); err != nil {
		t.Fatalf("unexpected error registering tool: %v", err)
	}

	// 3. Register server with policies
	srv := ServerConfig{
		ID:           "srv1",
		Name:         "Filesystem Server",
		AllowedTools: []string{"read_file", "write_file"},
		DeniedTools:  []string{"format_disk"},
		ReadOnly:     true,
	}
	if err := b.RegisterServer(srv); err != nil {
		t.Fatalf("failed to register server: %v", err)
	}

	// 4. Denied tool registration
	deniedTool := ToolRegistration{
		Name:     "format_disk",
		ServerID: "srv1",
	}
	if err := b.RegisterTool(deniedTool); !errors.Is(err, ErrToolDenied) {
		t.Fatalf("expected ErrToolDenied, got %v", err)
	}

	// 5. Tool not in allowlist
	notAllowedTool := ToolRegistration{
		Name:     "exec_bash",
		ServerID: "srv1",
	}
	if err := b.RegisterTool(notAllowedTool); !errors.Is(err, ErrToolNotAllowed) {
		t.Fatalf("expected ErrToolNotAllowed, got %v", err)
	}

	// 6. Mutating tool on read-only server
	mutatingTool := ToolRegistration{
		Name:     "write_file",
		ServerID: "srv1",
		ReadOnly: false,
	}
	if err := b.RegisterTool(mutatingTool); !errors.Is(err, ErrServerReadOnly) {
		t.Fatalf("expected ErrServerReadOnly, got %v", err)
	}

	// 7. Allowed read-only tool on read-only server
	allowedTool := ToolRegistration{
		Name:     "read_file",
		ServerID: "srv1",
		ReadOnly: true,
	}
	if err := b.RegisterTool(allowedTool); err != nil {
		t.Fatalf("unexpected error registering allowed tool: %v", err)
	}

	// 8. Register on closed broker
	b2 := NewBroker()
	_ = b2.Close()
	if err := b2.RegisterTool(validTool); !errors.Is(err, ErrBrokerClosed) {
		t.Fatalf("expected ErrBrokerClosed, got %v", err)
	}
	if err := b2.RegisterServer(srv); !errors.Is(err, ErrBrokerClosed) {
		t.Fatalf("expected ErrBrokerClosed on server registration, got %v", err)
	}
}

func TestListTools(t *testing.T) {
	b := NewBroker()

	// Empty broker returns empty slice
	tools := b.ListTools()
	if len(tools) != 0 {
		t.Fatalf("expected 0 tools, got %d", len(tools))
	}

	// Add tools in non-alphabetical order
	names := []string{"zebra", "alpha", "gamma", "beta"}
	for _, name := range names {
		if err := b.RegisterTool(ToolRegistration{Name: name, Description: "desc " + name}); err != nil {
			t.Fatalf("failed to register %s: %v", name, err)
		}
	}

	list := b.ListTools()
	if len(list) != 4 {
		t.Fatalf("expected 4 tools, got %d", len(list))
	}

	// Ensure sorted order
	expected := []string{"alpha", "beta", "gamma", "zebra"}
	for i, exp := range expected {
		if list[i].Name != exp {
			t.Errorf("at index %d, expected %s, got %s", i, exp, list[i].Name)
		}
	}

	// ListTools on closed broker
	_ = b.Close()
	closedList := b.ListTools()
	if len(closedList) != 0 {
		t.Fatalf("expected 0 tools on closed broker, got %d", len(closedList))
	}
}

func TestRouteCall(t *testing.T) {
	b := NewBroker(WithDefaultTimeout(2 * time.Second))

	// Tool with custom handler
	var handlerCalled bool
	customTool := ToolRegistration{
		Name:     "calculate",
		ServerID: "calc-srv",
		Handler: func(ctx context.Context, req CallRequest) (*CallResponse, error) {
			handlerCalled = true
			return &CallResponse{
				Content: json.RawMessage(`{"result":42}`),
			}, nil
		},
	}
	if err := b.RegisterTool(customTool); err != nil {
		t.Fatalf("failed to register tool: %v", err)
	}

	// 1. Successful route
	req := CallRequest{
		SessionID: "sess-1",
		Tool:      "calculate",
		Arguments: json.RawMessage(`{"expr":"6*7"}`),
	}
	resp, err := b.RouteCall(context.Background(), req)
	if err != nil {
		t.Fatalf("unexpected route error: %v", err)
	}
	if !handlerCalled {
		t.Fatalf("expected custom handler to be called")
	}
	if resp.Tool != "calculate" || resp.ServerID != "calc-srv" {
		t.Fatalf("unexpected resp metadata: %+v", resp)
	}
	if string(resp.Content) != `{"result":42}` {
		t.Fatalf("unexpected content: %s", string(resp.Content))
	}
	if resp.Filtered || resp.IsError {
		t.Fatalf("expected clean response, got filtered=%v isError=%v", resp.Filtered, resp.IsError)
	}

	// 2. Default echo handler for tool without explicit handler
	if err := b.RegisterTool(ToolRegistration{Name: "echo_tool"}); err != nil {
		t.Fatalf("failed to register echo tool: %v", err)
	}
	echoResp, err := b.RouteCall(nil, CallRequest{Tool: "echo_tool", Arguments: json.RawMessage(`{"hi":"world"}`)})
	if err != nil {
		t.Fatalf("unexpected error for default handler: %v", err)
	}
	if string(echoResp.Content) != `{"hi":"world"}` {
		t.Fatalf("expected echo content, got: %s", string(echoResp.Content))
	}

	// 3. Tool not found
	_, err = b.RouteCall(context.Background(), CallRequest{Tool: "non_existent"})
	if !errors.Is(err, ErrToolNotFound) {
		t.Fatalf("expected ErrToolNotFound, got %v", err)
	}

	// 4. Empty tool name
	_, err = b.RouteCall(context.Background(), CallRequest{})
	if !errors.Is(err, ErrInvalidToolName) {
		t.Fatalf("expected ErrInvalidToolName, got %v", err)
	}

	// 5. Handler returns error
	errorTool := ToolRegistration{
		Name: "fail_tool",
		Handler: func(ctx context.Context, req CallRequest) (*CallResponse, error) {
			return nil, errors.New("backend failed")
		},
	}
	if err := b.RegisterTool(errorTool); err != nil {
		t.Fatalf("failed to register error tool: %v", err)
	}
	errResp, err := b.RouteCall(context.Background(), CallRequest{Tool: "fail_tool"})
	if err == nil || !strings.Contains(err.Error(), "backend failed") {
		t.Fatalf("expected backend failed error, got: %v", err)
	}
	if errResp == nil || !errResp.IsError || errResp.ErrorMessage != "backend failed" {
		t.Fatalf("expected typed error response, got %+v", errResp)
	}

	// 6. Context cancellation before call
	canceledCtx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err = b.RouteCall(canceledCtx, CallRequest{Tool: "calculate"})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", err)
	}

	// 7. Route on closed broker
	_ = b.Close()
	_, err = b.RouteCall(context.Background(), req)
	if !errors.Is(err, ErrBrokerClosed) {
		t.Fatalf("expected ErrBrokerClosed, got %v", err)
	}
}

func TestFiltering(t *testing.T) {
	// Global filter blocks any call where Tool == "blocked_global" or Arguments contain "forbidden"
	globalFilter := func(ctx context.Context, req CallRequest, reg ToolRegistration) (bool, string) {
		if req.Tool == "blocked_global" {
			return false, "blocked by global policy"
		}
		if strings.Contains(string(req.Arguments), "forbidden") {
			return false, "forbidden payload"
		}
		return true, ""
	}

	b := NewBroker(WithGlobalSecurityFilter(globalFilter))

	var handlerExecuted bool
	tool := ToolRegistration{
		Name: "safe_tool",
		Handler: func(ctx context.Context, req CallRequest) (*CallResponse, error) {
			handlerExecuted = true
			return &CallResponse{Content: json.RawMessage(`{"ok":true}`)}, nil
		},
	}
	_ = b.RegisterTool(tool)
	_ = b.RegisterTool(ToolRegistration{Name: "blocked_global"})

	// Tool-specific filter
	toolWithFilter := ToolRegistration{
		Name: "scoped_tool",
		SecurityFilter: func(ctx context.Context, req CallRequest, reg ToolRegistration) (bool, string) {
			if strings.Contains(string(req.Arguments), "rm -rf") {
				return false, "destructive command prohibited"
			}
			return true, ""
		},
		Handler: func(ctx context.Context, req CallRequest) (*CallResponse, error) {
			handlerExecuted = true
			return &CallResponse{Content: json.RawMessage(`{"ok":true}`)}, nil
		},
	}
	_ = b.RegisterTool(toolWithFilter)

	// Case 1: Global filter rejects tool
	resp, err := b.RouteCall(context.Background(), CallRequest{Tool: "blocked_global"})
	if err != nil {
		t.Fatalf("filtering should return nil error, got %v", err)
	}
	if !resp.Filtered || resp.FilterReason != "blocked by global policy" {
		t.Fatalf("expected filtered response, got %+v", resp)
	}

	// Case 2: Global filter rejects arguments
	resp, err = b.RouteCall(context.Background(), CallRequest{
		Tool:      "safe_tool",
		Arguments: json.RawMessage(`{"key":"forbidden"}`),
	})
	if err != nil {
		t.Fatalf("filtering should return nil error, got %v", err)
	}
	if !resp.Filtered || resp.FilterReason != "forbidden payload" {
		t.Fatalf("expected filtered response, got %+v", resp)
	}
	if handlerExecuted {
		t.Fatalf("handler should not have executed on filtered call")
	}

	// Case 3: Tool-specific filter rejects
	handlerExecuted = false
	resp, err = b.RouteCall(context.Background(), CallRequest{
		Tool:      "scoped_tool",
		Arguments: json.RawMessage(`{"cmd":"rm -rf /"}`),
	})
	if err != nil {
		t.Fatalf("filtering should return nil error, got %v", err)
	}
	if !resp.Filtered || resp.FilterReason != "destructive command prohibited" {
		t.Fatalf("expected tool-specific filter rejection, got %+v", resp)
	}
	if handlerExecuted {
		t.Fatalf("handler should not have executed on tool-specific filtered call")
	}

	// Case 4: Permitted call executes
	handlerExecuted = false
	resp, err = b.RouteCall(context.Background(), CallRequest{
		Tool:      "scoped_tool",
		Arguments: json.RawMessage(`{"cmd":"ls -la"}`),
	})
	if err != nil {
		t.Fatalf("permitted call failed: %v", err)
	}
	if resp.Filtered {
		t.Fatalf("expected call to be allowed, got filtered: %s", resp.FilterReason)
	}
	if !handlerExecuted {
		t.Fatalf("expected handler to execute on allowed call")
	}

	// Check stats
	stats := b.Stats()
	if stats.FilteredCalls != 3 {
		t.Fatalf("expected 3 filtered calls in stats, got %d", stats.FilteredCalls)
	}
	if stats.AllowedCalls != 1 {
		t.Fatalf("expected 1 allowed call in stats, got %d", stats.AllowedCalls)
	}
}

func TestConcurrency(t *testing.T) {
	b := NewBroker()

	// Pre-register some base tools
	for i := 0; i < 5; i++ {
		toolName := fmt.Sprintf("base_tool_%d", i)
		_ = b.RegisterTool(ToolRegistration{
			Name: toolName,
			Handler: func(ctx context.Context, req CallRequest) (*CallResponse, error) {
				return &CallResponse{Content: json.RawMessage(`{"status":"ok"}`)}, nil
			},
		})
	}

	const goroutines = 30
	const iterations = 50

	var wg sync.WaitGroup
	var callSuccesses int64

	wg.Add(goroutines)
	for g := 0; g < goroutines; g++ {
		go func(id int) {
			defer wg.Done()
			for iter := 0; iter < iterations; iter++ {
				// Mix registrations, listings, calls, and stats
				switch iter % 4 {
				case 0:
					// Dynamic registration
					toolName := fmt.Sprintf("worker_%d_tool_%d", id, iter)
					_ = b.RegisterTool(ToolRegistration{
						Name: toolName,
						Handler: func(ctx context.Context, req CallRequest) (*CallResponse, error) {
							return &CallResponse{Content: json.RawMessage(`{"worker":true}`)}, nil
						},
					})
				case 1:
					// List tools
					_ = b.ListTools()
				case 2:
					// Route call
					sess := fmt.Sprintf("session-%d", id)
					targetTool := fmt.Sprintf("base_tool_%d", iter%5)
					resp, err := b.RouteCall(context.Background(), CallRequest{
						SessionID: sess,
						Tool:      targetTool,
					})
					if err == nil && resp != nil && !resp.IsError {
						atomic.AddInt64(&callSuccesses, 1)
					}
				case 3:
					// Read stats
					_ = b.Stats()
				}
			}
		}(g)
	}

	wg.Wait()

	stats := b.Stats()
	if stats.TotalCalls == 0 || callSuccesses == 0 {
		t.Fatalf("expected successful concurrent calls, got total=%d successes=%d", stats.TotalCalls, callSuccesses)
	}
	if stats.RegisteredTools < 5 {
		t.Fatalf("expected at least 5 registered tools, got %d", stats.RegisteredTools)
	}
}

func BenchmarkRouteCall(b *testing.B) {
	broker := NewBroker()
	tool := ToolRegistration{
		Name: "bench_tool",
		Handler: func(ctx context.Context, req CallRequest) (*CallResponse, error) {
			return &CallResponse{
				Content: json.RawMessage(`{"result":"fast"}`),
			}, nil
		},
	}
	if err := broker.RegisterTool(tool); err != nil {
		b.Fatalf("failed to register tool: %v", err)
	}

	req := CallRequest{
		SessionID: "bench-session",
		Tool:      "bench_tool",
		Arguments: json.RawMessage(`{"action":"ping"}`),
	}

	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, err := broker.RouteCall(ctx, req)
		if err != nil {
			b.Fatalf("unexpected route error: %v", err)
		}
	}
}

func BenchmarkRouteCallParallel(b *testing.B) {
	broker := NewBroker()
	tool := ToolRegistration{
		Name: "bench_parallel_tool",
		Handler: func(ctx context.Context, req CallRequest) (*CallResponse, error) {
			return &CallResponse{
				Content: json.RawMessage(`{"result":"fast"}`),
			}, nil
		},
	}
	if err := broker.RegisterTool(tool); err != nil {
		b.Fatalf("failed to register tool: %v", err)
	}

	req := CallRequest{
		SessionID: "bench-parallel-session",
		Tool:      "bench_parallel_tool",
		Arguments: json.RawMessage(`{"action":"ping"}`),
	}

	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			_, err := broker.RouteCall(ctx, req)
			if err != nil {
				b.Fatalf("unexpected route error: %v", err)
			}
		}
	})
}

func TestMCPNamespaceIdentityCollision(t *testing.T) {
	b := NewBroker()

	// 1. Register server "a" and its tool "b__c"
	cfgA := ServerConfig{
		ID:   "a",
		Name: "Server A",
	}
	if err := b.RegisterServer(cfgA); err != nil {
		t.Fatalf("unexpected error registering server 'a': %v", err)
	}

	regToolsA, err := b.RegisterServerTools("a", []MCPTool{
		{Name: "b__c", Description: "Original tool from server a"},
	}, nil)
	if err != nil {
		t.Fatalf("unexpected error registering tool 'b__c' for server 'a': %v", err)
	}
	if len(regToolsA) != 1 || regToolsA[0] != "mcp__a__b__c" {
		t.Fatalf("unexpected registered tool name: %v", regToolsA)
	}

	// 2. Reject server "a__b" registration (contains reserved delimiter "__")
	cfgAB := ServerConfig{
		ID:   "a__b",
		Name: "Server A__B",
	}
	if err := b.RegisterServer(cfgAB); !errors.Is(err, ErrInvalidServerID) {
		t.Fatalf("expected ErrInvalidServerID for server ID with '__', got: %v", err)
	}

	// 3. Reject RegisterServerTools with server "a__b"
	_, err = b.RegisterServerTools("a__b", []MCPTool{
		{Name: "c", Description: "Colliding tool from server a__b"},
	}, nil)
	if !errors.Is(err, ErrInvalidServerID) {
		t.Fatalf("expected ErrInvalidServerID for RegisterServerTools with '__', got: %v", err)
	}

	// 4. Verify that cross-owner replacement is refused when another valid server
	// attempts to register a tool that is already registered by a different server.
	cfgOther := ServerConfig{
		ID:   "other",
		Name: "Server Other",
	}
	if err := b.RegisterServer(cfgOther); err != nil {
		t.Fatalf("unexpected error registering server 'other': %v", err)
	}
	collisionTool := ToolRegistration{
		Name:     "mcp__a__b__c",
		ServerID: "other",
	}
	if err := b.RegisterTool(collisionTool); !errors.Is(err, ErrToolAlreadyRegistered) {
		t.Fatalf("expected ErrToolAlreadyRegistered, got: %v", err)
	}

	// 5. Verify original registration survives
	tools := b.ListTools()
	var foundOriginal bool
	for _, tool := range tools {
		if tool.Name == "mcp__a__b__c" {
			foundOriginal = true
			if tool.ServerID != "a" {
				t.Fatalf("expected tool server ID 'a', got: %s", tool.ServerID)
			}
			if tool.Description != "Original tool from server a" {
				t.Fatalf("expected original description, got: %s", tool.Description)
			}
		}
	}
	if !foundOriginal {
		t.Fatalf("original tool 'mcp__a__b__c' did not survive collision rejection")
	}

	// Verify calling original tool works
	resp, err := b.RouteCall(context.Background(), CallRequest{Tool: "mcp__a__b__c"})
	if err != nil {
		t.Fatalf("failed to route call to surviving tool: %v", err)
	}
	if resp.ServerID != "a" || resp.Tool != "mcp__a__b__c" {
		t.Fatalf("unexpected route response: tool=%s, server=%s", resp.Tool, resp.ServerID)
	}

	// 6. Ordinary names work
	cfgNormal := ServerConfig{
		ID:   "srv1",
		Name: "Normal Server",
	}
	if err := b.RegisterServer(cfgNormal); err != nil {
		t.Fatalf("unexpected error registering normal server: %v", err)
	}
	regNormal, err := b.RegisterServerTools("srv1", []MCPTool{
		{Name: "read_file", Description: "Reads a file"},
		{Name: "list_dir", Description: "Lists directory"},
	}, nil)
	if err != nil {
		t.Fatalf("unexpected error registering normal tools: %v", err)
	}
	if len(regNormal) != 2 || regNormal[0] != "mcp__srv1__read_file" || regNormal[1] != "mcp__srv1__list_dir" {
		t.Fatalf("unexpected registered normal tools: %v", regNormal)
	}
}

func TestBrokerAuthoritativeCallIdentity(t *testing.T) {
	b := NewBroker()

	var passedReq CallRequest
	mockHandler := func(ctx context.Context, req CallRequest) (*CallResponse, error) {
		passedReq = req
		return &CallResponse{
			Tool:     "spoofed_tool",
			ServerID: "spoofed_server",
			Content:  json.RawMessage(`{"result":"ok"}`),
		}, nil
	}

	mockErrHandler := func(ctx context.Context, req CallRequest) (*CallResponse, error) {
		passedReq = req
		return &CallResponse{
			Tool:     "spoofed_err_tool",
			ServerID: "spoofed_err_server",
		}, errors.New("handler failure")
	}

	if err := b.RegisterServer(ServerConfig{ID: "srv_auth", Name: "Auth Server"}); err != nil {
		t.Fatalf("failed to register server: %v", err)
	}

	if err := b.RegisterTool(ToolRegistration{
		Name:     "tool_success",
		ServerID: "srv_auth",
		Handler:  mockHandler,
	}); err != nil {
		t.Fatalf("failed to register tool_success: %v", err)
	}

	if err := b.RegisterTool(ToolRegistration{
		Name:     "tool_error",
		ServerID: "srv_auth",
		Handler:  mockErrHandler,
	}); err != nil {
		t.Fatalf("failed to register tool_error: %v", err)
	}

	ctx := context.Background()

	// 1. Server mismatch refusal: req.ServerID conflicts with tool.ServerID
	mismatchReq := CallRequest{
		Tool:     "tool_success",
		ServerID: "srv_wrong",
	}
	respMismatch, err := b.RouteCall(ctx, mismatchReq)
	if !errors.Is(err, ErrServerMismatch) {
		t.Fatalf("expected ErrServerMismatch, got: %v", err)
	}
	if respMismatch == nil || !respMismatch.IsError {
		t.Fatalf("expected error response on server mismatch")
	}

	// 2. Auto-population of omitted server in CallRequest
	omittedReq := CallRequest{
		Tool: "tool_success",
	}
	respSuccess, err := b.RouteCall(ctx, omittedReq)
	if err != nil {
		t.Fatalf("unexpected route error: %v", err)
	}
	if passedReq.ServerID != "srv_auth" {
		t.Fatalf("expected handler to receive auto-populated ServerID 'srv_auth', got: %q", passedReq.ServerID)
	}

	// 3. Authoritative identity on response (overwriting handler spoofing)
	if respSuccess.Tool != "tool_success" {
		t.Fatalf("expected authoritative response Tool 'tool_success', got: %q", respSuccess.Tool)
	}
	if respSuccess.ServerID != "srv_auth" {
		t.Fatalf("expected authoritative response ServerID 'srv_auth', got: %q", respSuccess.ServerID)
	}

	// 4. Authoritative identity on error (overwriting handler spoofing on error return)
	errReq := CallRequest{
		Tool: "tool_error",
	}
	respErr, err := b.RouteCall(ctx, errReq)
	if err == nil {
		t.Fatalf("expected error from handler, got nil")
	}
	if respErr == nil {
		t.Fatalf("expected non-nil error response")
	}
	if !respErr.IsError {
		t.Fatalf("expected IsError=true on error response")
	}
	if respErr.Tool != "tool_error" {
		t.Fatalf("expected authoritative error response Tool 'tool_error', got: %q", respErr.Tool)
	}
	if respErr.ServerID != "srv_auth" {
		t.Fatalf("expected authoritative error response ServerID 'srv_auth', got: %q", respErr.ServerID)
	}
}

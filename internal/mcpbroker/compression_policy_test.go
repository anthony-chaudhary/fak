package mcpbroker

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"sync"
	"testing"
	"time"
)

// setupMockTransport creates an in-memory StdioTransport connected via io.Pipe()
// to a mock MCP server responding to "tools/call" with configurable structured results.
func setupMockTransport(t *testing.T, pretty, compact string) (*StdioTransport, func()) {
	t.Helper()
	quote := func(s string) string { b, _ := json.Marshal(s); return string(b) }
	block := `[{ "type" : "text", "annotations": { "x":1e+09, "x":2 }, "text" : ` + quote(pretty) + `, "_meta" : { "a" : true } }]`
	result := `{"content":` + block + `,"structuredContent":` + compact + `}`

	requestReader, requestWriter := io.Pipe()
	responseReader, responseWriter := io.Pipe()

	transport := &StdioTransport{
		stdin:   requestWriter,
		stdout:  responseReader,
		pending: make(map[int64]chan *rpcResponse),
		doneCh:  make(chan struct{}),
	}
	go transport.pumpReader()

	serverDone := make(chan struct{})
	go func() {
		defer close(serverDone)
		scanner := bufio.NewScanner(requestReader)
		var writeMu sync.Mutex
		for scanner.Scan() {
			line := scanner.Bytes()
			if len(line) == 0 {
				continue
			}
			var req struct {
				JSONRPC string          `json:"jsonrpc"`
				ID      *int64          `json:"id"`
				Method  string          `json:"method"`
				Params  json.RawMessage `json:"params"`
			}
			if err := json.Unmarshal(line, &req); err != nil {
				return
			}
			if req.Method == "tools/call" && req.ID != nil {
				resp := fmt.Sprintf(`{"jsonrpc":"2.0","id":%d,"result":%s}`+"\n", *req.ID, result)
				writeMu.Lock()
				_, err := responseWriter.Write([]byte(resp))
				writeMu.Unlock()
				if err != nil {
					return
				}
			}
		}
	}()

	cleanup := func() {
		_ = requestReader.Close()
		_ = requestWriter.Close()
		_ = responseReader.Close()
		_ = responseWriter.Close()
		_ = transport.Close()
		<-serverDone
	}

	return transport, cleanup
}

func testPrettyAndCompact() (pretty, compact, wantOriginal, wantCompact string) {
	pretty = "{\n" + strings.Repeat(" ", 100) + `"n":9007199254740993,"n":1e+09,"s":"\u0061  b"` + "\n}"
	compact = `{"n":9007199254740993,"n":1e+09,"s":"\u0061  b"}`
	quote := func(s string) string { b, _ := json.Marshal(s); return string(b) }
	wantOriginal = `[{ "type" : "text", "annotations": { "x":1e+09, "x":2 }, "text" : ` + quote(pretty) + `, "_meta" : { "a" : true } }]`
	wantCompact = `[{ "type" : "text", "annotations": { "x":1e+09, "x":2 }, "text" : ` + quote(compact) + `, "_meta" : { "a" : true } }]`
	return
}

// TestCompressionPolicy_MetadataVariations tests all supported metadata keys and values
// for opting out of structured compression.
func TestCompressionPolicy_MetadataVariations(t *testing.T) {
	pretty, compact, wantOriginal, wantCompact := testPrettyAndCompact()
	transport, cleanup := setupMockTransport(t, pretty, compact)
	defer cleanup()

	broker := NewBroker()
	defer broker.Close()

	_ = broker.RegisterTool(ToolRegistration{
		Name: "get_data",
		Handler: func(ctx context.Context, req CallRequest) (*CallResponse, error) {
			return transport.CallTool(ctx, "get_data", req.Arguments)
		},
	})

	testCases := []struct {
		name        string
		metadata    map[string]string
		wantContent string
	}{
		{name: "compression:identity", metadata: map[string]string{"compression": "identity"}, wantContent: wantOriginal},
		{name: "compression:off", metadata: map[string]string{"compression": "off"}, wantContent: wantOriginal},
		{name: "compression:opt_out", metadata: map[string]string{"compression": "opt_out"}, wantContent: wantOriginal},
		{name: "compression:opt-out", metadata: map[string]string{"compression": "opt-out"}, wantContent: wantOriginal},
		{name: "compression:none", metadata: map[string]string{"compression": "none"}, wantContent: wantOriginal},
		{name: "compression:noop", metadata: map[string]string{"compression": "noop"}, wantContent: wantOriginal},
		{name: "compression:disabled", metadata: map[string]string{"compression": "disabled"}, wantContent: wantOriginal},
		{name: "compression:false", metadata: map[string]string{"compression": "false"}, wantContent: wantOriginal},
		{name: "compression:0", metadata: map[string]string{"compression": "0"}, wantContent: wantOriginal},
		{name: "mcp_compression:opt_out", metadata: map[string]string{"mcp_compression": "opt_out"}, wantContent: wantOriginal},
		{name: "mcp_compression:identity", metadata: map[string]string{"mcp_compression": "identity"}, wantContent: wantOriginal},
		{name: "mcp_compression:off", metadata: map[string]string{"mcp_compression": "off"}, wantContent: wantOriginal},
		{name: "mcp-compression:opt_out", metadata: map[string]string{"mcp-compression": "opt_out"}, wantContent: wantOriginal},
		{name: "structured_compression:off", metadata: map[string]string{"structured_compression": "off"}, wantContent: wantOriginal},
		{name: "structured-compression:disabled", metadata: map[string]string{"structured-compression": "disabled"}, wantContent: wantOriginal},
		{name: "case-insensitive key and value", metadata: map[string]string{"COMPRESSION": " IDENTITY "}, wantContent: wantOriginal},
		{name: "default auto", metadata: nil, wantContent: wantCompact},
		{name: "explicit auto", metadata: map[string]string{"compression": "auto"}, wantContent: wantCompact},
		{name: "explicit on", metadata: map[string]string{"compression": "on"}, wantContent: wantCompact},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer cancel()

			req := CallRequest{
				Tool:     "get_data",
				Metadata: tc.metadata,
			}
			resp, err := broker.RouteCall(ctx, req)
			if err != nil {
				t.Fatalf("RouteCall failed: %v", err)
			}
			if string(resp.Content) != tc.wantContent {
				t.Fatalf("unexpected content:\ngot:  %s\nwant: %s", string(resp.Content), tc.wantContent)
			}
		})
	}
}

// TestCompressionPolicy_TypedField tests the typed Compression field on CallRequest.
func TestCompressionPolicy_TypedField(t *testing.T) {
	pretty, compact, wantOriginal, wantCompact := testPrettyAndCompact()
	transport, cleanup := setupMockTransport(t, pretty, compact)
	defer cleanup()

	broker := NewBroker()
	defer broker.Close()

	_ = broker.RegisterTool(ToolRegistration{
		Name: "get_data",
		Handler: func(ctx context.Context, req CallRequest) (*CallResponse, error) {
			return transport.CallTool(ctx, "get_data", req.Arguments)
		},
	})

	for _, tc := range []struct {
		name        string
		policy      CompressionPolicy
		wantContent string
	}{
		{name: "CompressionIdentity", policy: CompressionIdentity, wantContent: wantOriginal},
		{name: "CompressionOff", policy: CompressionOff, wantContent: wantOriginal},
		{name: "CompressionOptOut", policy: CompressionOptOut, wantContent: wantOriginal},
		{name: "CompressionAuto", policy: CompressionAuto, wantContent: wantCompact},
		{name: "Empty (Default)", policy: "", wantContent: wantCompact},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer cancel()

			req := CallRequest{
				Tool:        "get_data",
				Compression: tc.policy,
			}
			resp, err := broker.RouteCall(ctx, req)
			if err != nil {
				t.Fatalf("RouteCall failed: %v", err)
			}
			if string(resp.Content) != tc.wantContent {
				t.Fatalf("content mismatch:\ngot:  %s\nwant: %s", string(resp.Content), tc.wantContent)
			}
		})
	}
}

// TestCompressionPolicy_ContextOptOut tests context-based compression policy configuration.
func TestCompressionPolicy_ContextOptOut(t *testing.T) {
	pretty, compact, wantOriginal, wantCompact := testPrettyAndCompact()
	transport, cleanup := setupMockTransport(t, pretty, compact)
	defer cleanup()

	broker := NewBroker()
	defer broker.Close()

	_ = broker.RegisterTool(ToolRegistration{
		Name: "get_data",
		Handler: func(ctx context.Context, req CallRequest) (*CallResponse, error) {
			return transport.CallTool(ctx, "get_data", req.Arguments)
		},
	})

	t.Run("WithCompressionOptOut", func(t *testing.T) {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		ctx = WithCompressionOptOut(ctx)

		resp, err := broker.RouteCall(ctx, CallRequest{Tool: "get_data"})
		if err != nil {
			t.Fatalf("RouteCall failed: %v", err)
		}
		if string(resp.Content) != wantOriginal {
			t.Fatalf("expected identity content, got: %s", string(resp.Content))
		}
	})

	t.Run("WithCompressionPolicy Identity", func(t *testing.T) {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		ctx = WithCompressionPolicy(ctx, CompressionIdentity)

		resp, err := broker.RouteCall(ctx, CallRequest{Tool: "get_data"})
		if err != nil {
			t.Fatalf("RouteCall failed: %v", err)
		}
		if string(resp.Content) != wantOriginal {
			t.Fatalf("expected identity content, got: %s", string(resp.Content))
		}
	})

	t.Run("WithCompressionPolicy Auto", func(t *testing.T) {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		ctx = WithCompressionPolicy(ctx, CompressionAuto)

		resp, err := broker.RouteCall(ctx, CallRequest{Tool: "get_data"})
		if err != nil {
			t.Fatalf("RouteCall failed: %v", err)
		}
		if string(resp.Content) != wantCompact {
			t.Fatalf("expected compact content, got: %s", string(resp.Content))
		}
	})

	t.Run("Direct Transport CallTool with Context", func(t *testing.T) {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		ctx = WithCompressionOptOut(ctx)

		resp, err := transport.CallTool(ctx, "get_data", nil)
		if err != nil {
			t.Fatalf("CallTool failed: %v", err)
		}
		if string(resp.Content) != wantOriginal {
			t.Fatalf("expected identity content, got: %s", string(resp.Content))
		}
	})
}

// TestCompressionPolicy_SessionConfiguration tests session-level compression policies.
func TestCompressionPolicy_SessionConfiguration(t *testing.T) {
	pretty, compact, wantOriginal, wantCompact := testPrettyAndCompact()
	transport, cleanup := setupMockTransport(t, pretty, compact)
	defer cleanup()

	broker := NewBroker()
	defer broker.Close()

	_ = broker.RegisterTool(ToolRegistration{
		Name: "get_data",
		Handler: func(ctx context.Context, req CallRequest) (*CallResponse, error) {
			return transport.CallTool(ctx, "get_data", req.Arguments)
		},
	})

	// Configure session A for opt-out
	broker.SetSessionCompression("session-optout", CompressionIdentity)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// Session A gets uncompressed original bytes
	respA, err := broker.RouteCall(ctx, CallRequest{SessionID: "session-optout", Tool: "get_data"})
	if err != nil {
		t.Fatalf("RouteCall session A failed: %v", err)
	}
	if string(respA.Content) != wantOriginal {
		t.Fatalf("session A expected original bytes, got: %s", string(respA.Content))
	}

	// Session B (default) gets compressed bytes
	respB, err := broker.RouteCall(ctx, CallRequest{SessionID: "session-default", Tool: "get_data"})
	if err != nil {
		t.Fatalf("RouteCall session B failed: %v", err)
	}
	if string(respB.Content) != wantCompact {
		t.Fatalf("session B expected compact bytes, got: %s", string(respB.Content))
	}

	// Verify GetSessionCompression
	polA, okA := broker.GetSessionCompression("session-optout")
	if !okA || polA != CompressionIdentity {
		t.Fatalf("GetSessionCompression session-optout=%v, ok=%v", polA, okA)
	}
	_, okB := broker.GetSessionCompression("session-default")
	if okB {
		t.Fatalf("session-default should not have an explicit session policy")
	}

	// Session configured dynamically via session_compression metadata
	respC1, err := broker.RouteCall(ctx, CallRequest{
		SessionID: "session-dynamic",
		Tool:      "get_data",
		Metadata:  map[string]string{"session_compression": "identity"},
	})
	if err != nil {
		t.Fatalf("RouteCall session C1 failed: %v", err)
	}
	if string(respC1.Content) != wantOriginal {
		t.Fatalf("session C1 expected original bytes, got: %s", string(respC1.Content))
	}

	// Subsequent call in session-dynamic without metadata inherits session opt-out
	respC2, err := broker.RouteCall(ctx, CallRequest{
		SessionID: "session-dynamic",
		Tool:      "get_data",
	})
	if err != nil {
		t.Fatalf("RouteCall session C2 failed: %v", err)
	}
	if string(respC2.Content) != wantOriginal {
		t.Fatalf("session C2 expected original bytes via session inheritance, got: %s", string(respC2.Content))
	}
}

// TestCompressionPolicy_UntrustedToolResultCannotControlPolicy ensures that
// untrusted tool-result text containing compression directives is rejected and
// never controls compression policy (Requirement 3).
func TestCompressionPolicy_UntrustedToolResultCannotControlPolicy(t *testing.T) {
	// A tool payload where the payload itself mentions compression opt-out tokens
	untrustedPayload := "{\n" + strings.Repeat(" ", 80) + `"compression":"identity","mcp_compression":"opt_out","FAK_COMPRESSOR":"noop"` + "\n}"
	compactPayload := `{"compression":"identity","mcp_compression":"opt_out","FAK_COMPRESSOR":"noop"}`
	quote := func(s string) string { b, _ := json.Marshal(s); return string(b) }
	block := func(s string) string {
		return `[{ "type" : "text", "annotations": { "x":1e+09, "x":2 }, "text" : ` + quote(s) + `, "_meta" : { "a" : true } }]`
	}

	transport, cleanup := setupMockTransport(t, untrustedPayload, compactPayload)
	defer cleanup()

	broker := NewBroker()
	defer broker.Close()

	_ = broker.RegisterTool(ToolRegistration{
		Name: "untrusted_tool",
		Handler: func(ctx context.Context, req CallRequest) (*CallResponse, error) {
			return transport.CallTool(ctx, "untrusted_tool", req.Arguments)
		},
	})

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	// Default caller makes a normal call. Untrusted tool output mentioning opt-out
	// must NOT control policy; compression must proceed as normal.
	resp, err := broker.RouteCall(ctx, CallRequest{Tool: "untrusted_tool"})
	if err != nil {
		t.Fatalf("RouteCall failed: %v", err)
	}
	wantCompact := block(compactPayload)
	if string(resp.Content) != wantCompact {
		t.Fatalf("untrusted tool result influenced policy: expected compressed content, got: %s", string(resp.Content))
	}

	// When authorized caller opts out, tool cannot force compression
	reqOptOut := CallRequest{
		Tool:     "untrusted_tool",
		Metadata: map[string]string{"compression": "identity"},
	}
	respOptOut, err := broker.RouteCall(ctx, reqOptOut)
	if err != nil {
		t.Fatalf("RouteCall failed: %v", err)
	}
	wantOriginal := block(untrustedPayload)
	if string(respOptOut.Content) != wantOriginal {
		t.Fatalf("authorized opt-out violated: expected original content, got: %s", string(respOptOut.Content))
	}
}

// TestCompressionPolicy_OperatorPrecedenceDominates verifies the precedence rule:
// operator-forced identity (FAK_COMPRESSOR=noop or none) dominates everything.
// Caller cannot widen behavior (cannot force compression if operator disabled it) (Requirement 2).
func TestCompressionPolicy_OperatorPrecedenceDominates(t *testing.T) {
	pretty, compact, wantOriginal, _ := testPrettyAndCompact()
	transport, cleanup := setupMockTransport(t, pretty, compact)
	defer cleanup()

	broker := NewBroker()
	defer broker.Close()

	_ = broker.RegisterTool(ToolRegistration{
		Name: "get_data",
		Handler: func(ctx context.Context, req CallRequest) (*CallResponse, error) {
			return transport.CallTool(ctx, "get_data", req.Arguments)
		},
	})

	for _, opEnv := range []string{"noop", "none", "NOOP", "NONE"} {
		t.Run("Operator="+opEnv, func(t *testing.T) {
			t.Setenv("FAK_COMPRESSOR", opEnv)

			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer cancel()

			// 1. Caller without opt-out: gets original bytes because operator disabled compression
			resp1, err := broker.RouteCall(ctx, CallRequest{Tool: "get_data"})
			if err != nil {
				t.Fatalf("RouteCall failed: %v", err)
			}
			if string(resp1.Content) != wantOriginal {
				t.Fatalf("operator %s must force identity, got: %s", opEnv, string(resp1.Content))
			}

			// 2. Caller attempting to explicitly widen/force compression via metadata: MUST be ignored
			resp2, err := broker.RouteCall(ctx, CallRequest{
				Tool:     "get_data",
				Metadata: map[string]string{"compression": "auto"},
			})
			if err != nil {
				t.Fatalf("RouteCall failed: %v", err)
			}
			if string(resp2.Content) != wantOriginal {
				t.Fatalf("caller cannot widen when operator=%s, got: %s", opEnv, string(resp2.Content))
			}

			// 3. Caller attempting to explicitly widen via typed field: MUST be ignored
			resp3, err := broker.RouteCall(ctx, CallRequest{
				Tool:        "get_data",
				Compression: CompressionAuto,
			})
			if err != nil {
				t.Fatalf("RouteCall failed: %v", err)
			}
			if string(resp3.Content) != wantOriginal {
				t.Fatalf("caller cannot widen via typed field when operator=%s, got: %s", opEnv, string(resp3.Content))
			}
		})
	}
}

// TestCompressionPolicy_ConcurrentTwoSessionsIntegration executes a concurrent integration test
// between two sessions calling the broker (Requirement 4 & Requirement 5):
// Session A gets exact original bytes (opt-out), while Session B keeps automatic default (compacted).
// Then operator noop/none dominates both.
func TestCompressionPolicy_ConcurrentTwoSessionsIntegration(t *testing.T) {
	pretty, compact, wantOriginal, wantCompact := testPrettyAndCompact()
	transport, cleanup := setupMockTransport(t, pretty, compact)
	defer cleanup()

	broker := NewBroker()
	defer broker.Close()

	_ = broker.RegisterTool(ToolRegistration{
		Name: "get_data",
		Handler: func(ctx context.Context, req CallRequest) (*CallResponse, error) {
			return transport.CallTool(ctx, "get_data", req.Arguments)
		},
	})

	// Configure session A for opt-out
	broker.SetSessionCompression("session-A", CompressionIdentity)

	const concurrency = 20
	var wg sync.WaitGroup
	errs := make(chan error, concurrency*2)

	// Phase 1: Concurrent execution with default operator env
	for i := 0; i < concurrency; i++ {
		wg.Add(2)

		// Session A goroutine: expects exact original bytes
		go func(idx int) {
			defer wg.Done()
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()

			req := CallRequest{
				SessionID: "session-A",
				Tool:      "get_data",
				Metadata: map[string]string{
					"request_id": fmt.Sprintf("req-a-%d", idx),
				},
			}
			resp, err := broker.RouteCall(ctx, req)
			if err != nil {
				errs <- fmt.Errorf("session A call %d failed: %w", idx, err)
				return
			}
			if string(resp.Content) != wantOriginal {
				errs <- fmt.Errorf("session A call %d got compressed bytes, want exact original", idx)
				return
			}
		}(i)

		// Session B goroutine: expects automatic default (compacted)
		go func(idx int) {
			defer wg.Done()
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()

			req := CallRequest{
				SessionID: "session-B",
				Tool:      "get_data",
				Metadata: map[string]string{
					"request_id": fmt.Sprintf("req-b-%d", idx),
				},
			}
			resp, err := broker.RouteCall(ctx, req)
			if err != nil {
				errs <- fmt.Errorf("session B call %d failed: %w", idx, err)
				return
			}
			if string(resp.Content) != wantCompact {
				errs <- fmt.Errorf("session B call %d got uncompressed bytes, want compact default", idx)
				return
			}
		}(i)
	}

	wg.Wait()
	close(errs)

	for err := range errs {
		t.Fatalf("phase 1 concurrency failure: %v", err)
	}

	// Phase 2: Operator forced identity (FAK_COMPRESSOR=noop) dominates both sessions
	t.Setenv("FAK_COMPRESSOR", "noop")

	errs2 := make(chan error, concurrency*2)
	for i := 0; i < concurrency; i++ {
		wg.Add(2)

		go func(idx int) {
			defer wg.Done()
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()

			resp, err := broker.RouteCall(ctx, CallRequest{SessionID: "session-A", Tool: "get_data"})
			if err != nil {
				errs2 <- fmt.Errorf("operator noop session A call %d failed: %w", idx, err)
				return
			}
			if string(resp.Content) != wantOriginal {
				errs2 <- fmt.Errorf("operator noop session A call %d did not get original bytes", idx)
				return
			}
		}(i)

		go func(idx int) {
			defer wg.Done()
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()

			resp, err := broker.RouteCall(ctx, CallRequest{SessionID: "session-B", Tool: "get_data"})
			if err != nil {
				errs2 <- fmt.Errorf("operator noop session B call %d failed: %w", idx, err)
				return
			}
			if string(resp.Content) != wantOriginal {
				errs2 <- fmt.Errorf("operator noop session B call %d did not get original bytes", idx)
				return
			}
		}(i)
	}

	wg.Wait()
	close(errs2)

	for err := range errs2 {
		t.Fatalf("phase 2 operator dominance failure: %v", err)
	}
}

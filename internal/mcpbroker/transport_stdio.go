package mcpbroker

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os/exec"
	"strconv"
	"strings"
	"sync"
)

// rpcRequest represents a JSON-RPC 2.0 request or notification.
type rpcRequest struct {
	JSONRPC string      `json:"jsonrpc"`
	ID      *int64      `json:"id,omitempty"`
	Method  string      `json:"method"`
	Params  interface{} `json:"params,omitempty"`
}

// rpcResponse represents a JSON-RPC 2.0 response.
type rpcResponse struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id,omitempty"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *rpcError       `json:"error,omitempty"`
}

// rpcError represents a JSON-RPC 2.0 error object.
type rpcError struct {
	Code    int             `json:"code"`
	Message string          `json:"message"`
	Data    json.RawMessage `json:"data,omitempty"`
}

func (e *rpcError) Error() string {
	return fmt.Sprintf("rpc error (code %d): %s", e.Code, e.Message)
}

// StdioTransport manages JSON-RPC 2.0 communication over standard I/O streams of an OS subprocess.
// It supports both newline-delimited JSON and Content-Length framed messages, non-blocking I/O,
// and asynchronous draining of stderr to prevent buffer deadlocks.
type StdioTransport struct {
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout io.ReadCloser
	stderr io.ReadCloser

	writeMu sync.Mutex
	reqSeq  int64

	pendingMu sync.Mutex
	pending   map[int64]chan *rpcResponse
	closed    bool
	doneCh    chan struct{}

	stderrMu  sync.Mutex
	stderrBuf bytes.Buffer
	maxStderr int
}

// NewStdioTransport initializes a StdioTransport for the provided command.
// Pipes for stdin, stdout, and stderr are attached and the process group is configured.
func NewStdioTransport(cmd *exec.Cmd) (*StdioTransport, error) {
	if cmd == nil {
		return nil, errors.New("mcpbroker: nil exec.Cmd")
	}

	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("mcpbroker: stdin pipe error: %w", err)
	}

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		_ = stdin.Close()
		return nil, fmt.Errorf("mcpbroker: stdout pipe error: %w", err)
	}

	stderr, err := cmd.StderrPipe()
	if err != nil {
		_ = stdin.Close()
		_ = stdout.Close()
		return nil, fmt.Errorf("mcpbroker: stderr pipe error: %w", err)
	}

	setProcessGroup(cmd)

	t := &StdioTransport{
		cmd:       cmd,
		stdin:     stdin,
		stdout:    stdout,
		stderr:    stderr,
		pending:   make(map[int64]chan *rpcResponse),
		doneCh:    make(chan struct{}),
		maxStderr: 1024 * 1024, // 1MB buffer cap
	}

	return t, nil
}

// Start launches the subprocess and initiates asynchronous stderr draining and stdout reading.
func (t *StdioTransport) Start() error {
	if err := t.cmd.Start(); err != nil {
		_ = t.stdin.Close()
		_ = t.stdout.Close()
		_ = t.stderr.Close()
		return fmt.Errorf("mcpbroker: failed to start process: %w", err)
	}

	// Drain stderr asynchronously to prevent pipe buffer deadlocks.
	go t.drainStderr()

	// Read and dispatch incoming JSON-RPC 2.0 messages from stdout.
	go t.pumpReader()

	return nil
}

// Stderr returns a copy of the captured stderr buffer for diagnostics and telemetry.
func (t *StdioTransport) Stderr() []byte {
	t.stderrMu.Lock()
	defer t.stderrMu.Unlock()
	cp := make([]byte, t.stderrBuf.Len())
	copy(cp, t.stderrBuf.Bytes())
	return cp
}

// drainStderr reads stderr until EOF, capturing up to maxStderr bytes in a rolling buffer.
func (t *StdioTransport) drainStderr() {
	defer t.stderr.Close()
	buf := make([]byte, 8192)
	for {
		n, err := t.stderr.Read(buf)
		if n > 0 {
			t.stderrMu.Lock()
			if t.stderrBuf.Len()+n > t.maxStderr {
				overflow := (t.stderrBuf.Len() + n) - t.maxStderr
				if overflow < t.stderrBuf.Len() {
					t.stderrBuf.Next(overflow)
				} else {
					t.stderrBuf.Reset()
				}
			}
			t.stderrBuf.Write(buf[:n])
			t.stderrMu.Unlock()
		}
		if err != nil {
			break
		}
	}
}

// pumpReader decodes incoming JSON-RPC 2.0 messages from stdout, supporting both
// newline-delimited JSON and Content-Length framed HTTP-style streams.
func (t *StdioTransport) pumpReader() {
	defer func() {
		t.pendingMu.Lock()
		t.closed = true
		t.pendingMu.Unlock()
		t.closePending(ErrMCPProcessCrash)
		close(t.doneCh)
	}()

	reader := bufio.NewReader(t.stdout)
	for {
		// Consume leading whitespace, carriage returns, or blank lines
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

		var msgBytes []byte

		if b[0] == '{' || b[0] == '[' {
			// Newline-delimited JSON
			line, err := reader.ReadBytes('\n')
			if len(line) > 0 {
				msgBytes = bytes.TrimSpace(line)
			}
			if err != nil && len(msgBytes) == 0 {
				return
			}
		} else {
			// Content-Length / LSP header framing
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
				msgBytes = bytes.TrimSpace(payload)
			}
		}

		if len(msgBytes) > 0 {
			t.dispatchMessage(msgBytes)
		}
	}
}

// dispatchMessage parses an incoming JSON-RPC message and routes responses to waiting callers.
func (t *StdioTransport) dispatchMessage(data []byte) {
	var resp rpcResponse
	if err := json.Unmarshal(data, &resp); err != nil {
		return
	}

	id, ok := parseID(resp.ID)
	if !ok {
		// Server notification or malformed ID
		return
	}

	t.pendingMu.Lock()
	ch, found := t.pending[id]
	if found {
		delete(t.pending, id)
	}
	t.pendingMu.Unlock()

	if found && ch != nil {
		select {
		case ch <- &resp:
		default:
		}
	}
}

// parseID parses an ID from a raw JSON payload, handling numeric and string representations.
func parseID(raw json.RawMessage) (int64, bool) {
	if len(raw) == 0 {
		return 0, false
	}
	var num int64
	if err := json.Unmarshal(raw, &num); err == nil {
		return num, true
	}
	var str string
	if err := json.Unmarshal(raw, &str); err == nil {
		if n, err := strconv.ParseInt(str, 10, 64); err == nil {
			return n, true
		}
	}
	return 0, false
}

// sendRequest serializes and dispatches a JSON-RPC 2.0 request, awaiting the matching response.
func (t *StdioTransport) sendRequest(ctx context.Context, method string, params interface{}) (*rpcResponse, error) {
	t.writeMu.Lock()
	t.reqSeq++
	id := t.reqSeq
	t.writeMu.Unlock()

	ch := make(chan *rpcResponse, 1)

	t.pendingMu.Lock()
	if t.closed {
		t.pendingMu.Unlock()
		return nil, ErrMCPProcessCrash
	}
	t.pending[id] = ch
	t.pendingMu.Unlock()

	defer func() {
		t.pendingMu.Lock()
		delete(t.pending, id)
		t.pendingMu.Unlock()
	}()

	req := rpcRequest{
		JSONRPC: "2.0",
		ID:      &id,
		Method:  method,
		Params:  params,
	}

	data, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("mcpbroker: marshal request error: %w", err)
	}
	data = append(data, '\n')

	t.writeMu.Lock()
	_, err = t.stdin.Write(data)
	t.writeMu.Unlock()

	if err != nil {
		return nil, fmt.Errorf("%w: failed to write to stdin: %v", ErrMCPProcessCrash, err)
	}

	select {
	case <-t.doneCh:
		return nil, ErrMCPProcessCrash
	default:
	}

	select {
	case <-ctx.Done():
		select {
		case <-t.doneCh:
			return nil, ErrMCPProcessCrash
		default:
			return nil, ctx.Err()
		}
	case <-t.doneCh:
		return nil, ErrMCPProcessCrash
	case resp, ok := <-ch:
		if !ok || resp == nil {
			return nil, ErrMCPProcessCrash
		}
		return resp, nil
	}
}

// sendNotification serializes and writes a JSON-RPC 2.0 notification (no response expected).
func (t *StdioTransport) sendNotification(method string, params interface{}) error {
	req := rpcRequest{
		JSONRPC: "2.0",
		Method:  method,
		Params:  params,
	}

	data, err := json.Marshal(req)
	if err != nil {
		return fmt.Errorf("mcpbroker: marshal notification error: %w", err)
	}
	data = append(data, '\n')

	t.writeMu.Lock()
	_, err = t.stdin.Write(data)
	t.writeMu.Unlock()

	if err != nil {
		return fmt.Errorf("%w: failed to write notification: %v", ErrMCPProcessCrash, err)
	}
	return nil
}

// Handshake executes protocol initialization: sends initialize, receives response,
// and sends notifications/initialized.
func (t *StdioTransport) Handshake(ctx context.Context) error {
	initParams := map[string]interface{}{
		"protocolVersion": "2024-11-05",
		"capabilities":    map[string]interface{}{},
		"clientInfo": map[string]interface{}{
			"name":    "fak-mcpbroker",
			"version": "1.0.0",
		},
	}

	resp, err := t.sendRequest(ctx, "initialize", initParams)
	if err != nil {
		return fmt.Errorf("mcpbroker: initialize handshake failed: %w", err)
	}
	if resp.Error != nil {
		return fmt.Errorf("mcpbroker: initialize error: %s", resp.Error.Message)
	}

	if err := t.sendNotification("notifications/initialized", map[string]interface{}{}); err != nil {
		return fmt.Errorf("mcpbroker: initialized notification failed: %w", err)
	}

	return nil
}

// ListTools sends a tools/list request and returns the slice of discovered tools.
func (t *StdioTransport) ListTools(ctx context.Context) ([]MCPTool, error) {
	resp, err := t.sendRequest(ctx, "tools/list", map[string]interface{}{})
	if err != nil {
		return nil, fmt.Errorf("mcpbroker: tools/list failed: %w", err)
	}
	if resp.Error != nil {
		return nil, fmt.Errorf("mcpbroker: tools/list error: %s", resp.Error.Message)
	}

	var result struct {
		Tools []MCPTool `json:"tools"`
	}
	if len(resp.Result) > 0 {
		if err := json.Unmarshal(resp.Result, &result); err != nil {
			return nil, fmt.Errorf("mcpbroker: parse tools/list result error: %w", err)
		}
	}
	return result.Tools, nil
}

// CallTool invokes tools/call on the MCP server with the specified name and arguments.
func (t *StdioTransport) CallTool(ctx context.Context, toolName string, args json.RawMessage) (*CallResponse, error) {
	params := map[string]interface{}{
		"name": toolName,
	}
	if len(args) > 0 && string(args) != "null" {
		params["arguments"] = json.RawMessage(args)
	} else {
		params["arguments"] = map[string]interface{}{}
	}

	resp, err := t.sendRequest(ctx, "tools/call", params)
	if err != nil {
		return nil, err
	}

	callResp := &CallResponse{
		Tool: toolName,
	}

	if resp.Error != nil {
		callResp.IsError = true
		callResp.ErrorMessage = resp.Error.Message
		return callResp, resp.Error
	}

	// Parse MCP ToolCall result format
	var mcpResult struct {
		Content json.RawMessage `json:"content"`
		IsError bool            `json:"isError"`
	}

	if len(resp.Result) > 0 {
		if err := json.Unmarshal(resp.Result, &mcpResult); err == nil && len(mcpResult.Content) > 0 {
			callResp.Content = mcpResult.Content
			callResp.IsError = mcpResult.IsError
		} else {
			callResp.Content = resp.Result
		}
	}

	return callResp, nil
}

// Ping sends a ping request and waits for an empty result to verify process liveness.
func (t *StdioTransport) Ping(ctx context.Context) error {
	resp, err := t.sendRequest(ctx, "ping", map[string]interface{}{})
	if err != nil {
		return fmt.Errorf("%w: ping failed: %v", ErrMCPProcessCrash, err)
	}
	if resp.Error != nil {
		return fmt.Errorf("mcpbroker: ping error: %s", resp.Error.Message)
	}
	return nil
}

// closePending unblocks all pending callers with the provided error.
func (t *StdioTransport) closePending(err error) {
	t.pendingMu.Lock()
	defer t.pendingMu.Unlock()
	for id, ch := range t.pending {
		close(ch)
		delete(t.pending, id)
	}
}

// Close closes the transport and underlying pipes.
func (t *StdioTransport) Close() error {
	t.pendingMu.Lock()
	wasClosed := t.closed
	t.closed = true
	t.pendingMu.Unlock()

	_ = t.stdin.Close()
	_ = t.stdout.Close()
	_ = t.stderr.Close()

	if wasClosed {
		return nil
	}
	return nil
}

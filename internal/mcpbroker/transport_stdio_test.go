package mcpbroker

import (
	"bytes"
	"context"
	"io"
	"testing"
	"time"
)

func TestStdioResponseEnvelopeBinding(t *testing.T) {
	stdinReader, stdinWriter := io.Pipe()
	stdoutReader, stdoutWriter := io.Pipe()
	stderrReader, stderrWriter := io.Pipe()

	defer func() {
		_ = stdinWriter.Close()
		_ = stdinReader.Close()
		_ = stdoutWriter.Close()
		_ = stdoutReader.Close()
		_ = stderrWriter.Close()
		_ = stderrReader.Close()
	}()

	// Drain stdin in background so sendRequest writes do not block.
	go func() {
		_, _ = io.Copy(io.Discard, stdinReader)
	}()

	transport := &StdioTransport{
		stdin:   stdinWriter,
		stdout:  stdoutReader,
		stderr:  stderrReader,
		pending: make(map[int64]chan *rpcResponse),
		doneCh:  make(chan struct{}),
	}
	go transport.pumpReader()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	type callResult struct {
		resp *rpcResponse
		err  error
	}
	resCh1 := make(chan callResult, 1)
	go func() {
		resp, err := transport.sendRequest(ctx, "testMethod", map[string]string{"k": "v"})
		resCh1 <- callResult{resp: resp, err: err}
	}()

	// Wait for pending entry 1 to be registered.
	waitForPending := func(id int64) {
		deadline := time.Now().Add(2 * time.Second)
		for {
			transport.pendingMu.Lock()
			_, found := transport.pending[id]
			transport.pendingMu.Unlock()
			if found {
				return
			}
			if time.Now().After(deadline) {
				t.Fatalf("timed out waiting for pending call %d to register", id)
			}
			time.Sleep(5 * time.Millisecond)
		}
	}
	waitForPending(1)

	// 1. Emit a server request using the pending numeric ID (id: 1, has method).
	serverReq := `{"jsonrpc":"2.0","id":1,"method":"roots/list","params":{}}` + "\n"
	if _, err := stdoutWriter.Write([]byte(serverReq)); err != nil {
		t.Fatalf("failed to write server request: %v", err)
	}

	// 2. Emit a string-ID response (id: "1", numeric string).
	stringIDResp := `{"jsonrpc":"2.0","id":"1","result":{"status":"string-id"}}` + "\n"
	if _, err := stdoutWriter.Write([]byte(stringIDResp)); err != nil {
		t.Fatalf("failed to write string-id response: %v", err)
	}

	// 3. Emit malformed responses (e.g. missing result/error, or missing jsonrpc).
	malformedResp := `{"jsonrpc":"2.0","id":1}` + "\n"
	if _, err := stdoutWriter.Write([]byte(malformedResp)); err != nil {
		t.Fatalf("failed to write malformed response: %v", err)
	}

	// Allow pumpReader to process incoming rejected messages.
	time.Sleep(100 * time.Millisecond)

	// Assert that pending call has NOT returned prematurely.
	select {
	case res := <-resCh1:
		t.Fatalf("pending call 1 was prematurely completed by rejected messages: resp=%+v err=%v", res.resp, res.err)
	default:
	}

	// Assert pending state survives the rejected messages.
	transport.pendingMu.Lock()
	_, pendingStillExists := transport.pending[1]
	transport.pendingMu.Unlock()
	if !pendingStillExists {
		t.Fatalf("expected pending call 1 to survive rejected messages, but it was removed from pending map")
	}

	// 4. Emit valid numeric-ID response.
	validResp := `{"jsonrpc":"2.0","id":1,"result":{"status":"valid-success"}}` + "\n"
	if _, err := stdoutWriter.Write([]byte(validResp)); err != nil {
		t.Fatalf("failed to write valid response: %v", err)
	}

	// Assert only the valid response completes the client call.
	select {
	case res := <-resCh1:
		if res.err != nil {
			t.Fatalf("unexpected sendRequest error: %v", res.err)
		}
		if res.resp == nil {
			t.Fatalf("expected non-nil response")
		}
		if res.resp.Error != nil {
			t.Fatalf("expected no error in response, got: %v", res.resp.Error)
		}
		if !bytes.Contains(res.resp.Result, []byte("valid-success")) {
			t.Fatalf("expected result to contain 'valid-success', got: %s", string(res.resp.Result))
		}
	case <-time.After(2 * time.Second):
		t.Fatalf("timed out waiting for client call 1 to complete with valid response")
	}

	// 5. Valid error-response control.
	resCh2 := make(chan callResult, 1)
	go func() {
		resp, err := transport.sendRequest(ctx, "failingMethod", nil)
		resCh2 <- callResult{resp: resp, err: err}
	}()
	waitForPending(2)

	validErrResp := `{"jsonrpc":"2.0","id":2,"error":{"code":-32601,"message":"Method not found"}}` + "\n"
	if _, err := stdoutWriter.Write([]byte(validErrResp)); err != nil {
		t.Fatalf("failed to write valid error response: %v", err)
	}

	select {
	case res := <-resCh2:
		if res.err != nil {
			t.Fatalf("unexpected sendRequest error: %v", res.err)
		}
		if res.resp == nil {
			t.Fatalf("expected non-nil response")
		}
		if res.resp.Error == nil {
			t.Fatalf("expected error object in control response, got nil")
		}
		if res.resp.Error.Code != -32601 {
			t.Fatalf("expected error code -32601, got: %d", res.resp.Error.Code)
		}
		if res.resp.Error.Message != "Method not found" {
			t.Fatalf("expected error message 'Method not found', got: %q", res.resp.Error.Message)
		}
	case <-time.After(2 * time.Second):
		t.Fatalf("timed out waiting for control client call 2 to complete with valid error response")
	}
}

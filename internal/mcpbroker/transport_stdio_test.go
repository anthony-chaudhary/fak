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

	// 2. Emit malformed responses (e.g. missing result/error, or missing jsonrpc).
	malformedResp := `{"jsonrpc":"2.0","id":1}` + "\n"
	if _, err := stdoutWriter.Write([]byte(malformedResp)); err != nil {
		t.Fatalf("failed to write malformed response: %v", err)
	}

	// Allow pumpReader to process incoming rejected messages.
	time.Sleep(50 * time.Millisecond)

	// Assert that pending call has NOT returned prematurely from server request or malformed envelope.
	select {
	case res := <-resCh1:
		t.Fatalf("pending call 1 was prematurely completed by server request/malformed response: resp=%+v err=%v", res.resp, res.err)
	default:
	}

	// Assert pending state survives server request and malformed envelope.
	transport.pendingMu.Lock()
	_, pendingStillExists := transport.pending[1]
	transport.pendingMu.Unlock()
	if !pendingStillExists {
		t.Fatalf("expected pending call 1 to survive rejected messages, but it was removed from pending map")
	}

	// 3. Emit a string-ID response (id: "1", non-int64 string).
	// Must fail fast with an immediate error response instead of hanging until timeout.
	stringIDResp := `{"jsonrpc":"2.0","id":"1","result":{"status":"string-id"}}` + "\n"
	if _, err := stdoutWriter.Write([]byte(stringIDResp)); err != nil {
		t.Fatalf("failed to write string-id response: %v", err)
	}

	select {
	case res := <-resCh1:
		if res.err != nil {
			t.Fatalf("unexpected sendRequest transport error: %v", res.err)
		}
		if res.resp == nil {
			t.Fatalf("expected non-nil response for non-int64 string ID")
		}
		if res.resp.Error == nil {
			t.Fatalf("expected error response for non-int64 string ID, got result: %s", string(res.resp.Result))
		}
		if res.resp.Error.Code != -32600 {
			t.Fatalf("expected error code -32600 for non-int64 response ID, got: %d", res.resp.Error.Code)
		}
	case <-time.After(2 * time.Second):
		t.Fatalf("timed out waiting for client call 1 to fail fast on non-int64 string ID")
	}

	// 4. Test float-ID response (id: 2.0). Must also fail fast with immediate error response.
	resCh2 := make(chan callResult, 1)
	go func() {
		resp, err := transport.sendRequest(ctx, "testFloatMethod", nil)
		resCh2 <- callResult{resp: resp, err: err}
	}()
	waitForPending(2)

	floatIDResp := `{"jsonrpc":"2.0","id":2.0,"result":{"status":"float-id"}}` + "\n"
	if _, err := stdoutWriter.Write([]byte(floatIDResp)); err != nil {
		t.Fatalf("failed to write float-id response: %v", err)
	}

	select {
	case res := <-resCh2:
		if res.err != nil {
			t.Fatalf("unexpected sendRequest transport error: %v", res.err)
		}
		if res.resp == nil {
			t.Fatalf("expected non-nil response for float ID")
		}
		if res.resp.Error == nil {
			t.Fatalf("expected error response for float ID, got result: %s", string(res.resp.Result))
		}
		if res.resp.Error.Code != -32600 {
			t.Fatalf("expected error code -32600 for float ID, got: %d", res.resp.Error.Code)
		}
	case <-time.After(2 * time.Second):
		t.Fatalf("timed out waiting for client call 2 to fail fast on float ID")
	}

	// 5. Test non-numeric string ID (id: "unparseable_string"). Must fail fast on single pending caller.
	resCh3 := make(chan callResult, 1)
	go func() {
		resp, err := transport.sendRequest(ctx, "testUnparseableMethod", nil)
		resCh3 <- callResult{resp: resp, err: err}
	}()
	waitForPending(3)

	unparseableIDResp := `{"jsonrpc":"2.0","id":"unparseable_string","result":{"status":"unparseable"}}` + "\n"
	if _, err := stdoutWriter.Write([]byte(unparseableIDResp)); err != nil {
		t.Fatalf("failed to write unparseable-id response: %v", err)
	}

	select {
	case res := <-resCh3:
		if res.err != nil {
			t.Fatalf("unexpected sendRequest transport error: %v", res.err)
		}
		if res.resp == nil {
			t.Fatalf("expected non-nil response for unparseable ID")
		}
		if res.resp.Error == nil {
			t.Fatalf("expected error response for unparseable ID, got result: %s", string(res.resp.Result))
		}
		if res.resp.Error.Code != -32600 {
			t.Fatalf("expected error code -32600 for unparseable ID, got: %d", res.resp.Error.Code)
		}
	case <-time.After(2 * time.Second):
		t.Fatalf("timed out waiting for client call 3 to fail fast on unparseable string ID")
	}

	// 6. Emit valid numeric-ID response (id: 4).
	resCh4 := make(chan callResult, 1)
	go func() {
		resp, err := transport.sendRequest(ctx, "testValidMethod", nil)
		resCh4 <- callResult{resp: resp, err: err}
	}()
	waitForPending(4)

	validResp := `{"jsonrpc":"2.0","id":4,"result":{"status":"valid-success"}}` + "\n"
	if _, err := stdoutWriter.Write([]byte(validResp)); err != nil {
		t.Fatalf("failed to write valid response: %v", err)
	}

	// Assert only the valid response completes the client call.
	select {
	case res := <-resCh4:
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
		t.Fatalf("timed out waiting for client call 4 to complete with valid response")
	}

	// 7. Valid error-response control (id: 5).
	resCh5 := make(chan callResult, 1)
	go func() {
		resp, err := transport.sendRequest(ctx, "failingMethod", nil)
		resCh5 <- callResult{resp: resp, err: err}
	}()
	waitForPending(5)

	validErrResp := `{"jsonrpc":"2.0","id":5,"error":{"code":-32601,"message":"Method not found"}}` + "\n"
	if _, err := stdoutWriter.Write([]byte(validErrResp)); err != nil {
		t.Fatalf("failed to write valid error response: %v", err)
	}

	select {
	case res := <-resCh5:
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
		t.Fatalf("timed out waiting for control client call 5 to complete with valid error response")
	}
}

func TestParseResponseIDs(t *testing.T) {
	tests := []struct {
		name     string
		raw      string
		exactNum int64
		exactOk  bool
		anyNum   int64
		anyOk    bool
	}{
		{"exact integer", `1`, 1, true, 1, true},
		{"exact integer 42", `42`, 42, true, 42, true},
		{"string integer", `"1"`, 0, false, 1, true},
		{"string integer 42", `"42"`, 0, false, 42, true},
		{"float integer", `1.0`, 0, false, 1, true},
		{"string float integer", `"1.0"`, 0, false, 1, true},
		{"float fraction", `1.5`, 0, false, 0, false},
		{"string text", `"abc"`, 0, false, 0, false},
		{"null", `null`, 0, false, 0, false},
		{"empty", ``, 0, false, 0, false},
		{"boolean", `true`, 0, false, 0, false},
		{"object", `{}`, 0, false, 0, false},
		{"array", `[1]`, 0, false, 0, false},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			num, ok := parseNumericID([]byte(tc.raw))
			if ok != tc.exactOk || num != tc.exactNum {
				t.Errorf("parseNumericID(%q) = (%d, %v), want (%d, %v)", tc.raw, num, ok, tc.exactNum, tc.exactOk)
			}
			anyNum, anyOk := parseAnyNumericID([]byte(tc.raw))
			if anyOk != tc.anyOk || anyNum != tc.anyNum {
				t.Errorf("parseAnyNumericID(%q) = (%d, %v), want (%d, %v)", tc.raw, anyNum, anyOk, tc.anyNum, tc.anyOk)
			}
		})
	}
}

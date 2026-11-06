package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/ctxplan"
)

// fak-test:runtime integration est=200ms lane=default
// Estimate is unmeasured. Each actual MCP HTTP request has a five-second bound.
func TestMCPRestoreBoundsCannotOverflow(t *testing.T) {
	srv := newTestServer(t)
	const trace, owner, body = "restore-overflow-trace", "restore-owner", "restore-boundary-payload"
	id := ctxplan.Digest([]byte(body))
	srv.bindTraceOwner(trace, owner)
	srv.stashRestore(trace, id, "small page", []byte(body))
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	maxInt := int(^uint(0) >> 1)
	post := func(t *testing.T, principal string, args ContextRestoreRequest) rpcResponse {
		t.Helper()
		payload, err := json.Marshal(map[string]any{
			"jsonrpc": "2.0", "id": 1, "method": "tools/call",
			"params": map[string]any{"name": "fak_context_restore", "arguments": args},
		})
		if err != nil {
			t.Fatal(err)
		}
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, ts.URL+"/mcp", bytes.NewReader(payload))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("X-Fak-Principal", principal)
		resp, err := ts.Client().Do(req)
		if err != nil {
			t.Fatalf("bounded restore request failed: %v", err)
		}
		raw, readErr := io.ReadAll(resp.Body)
		resp.Body.Close()
		if readErr != nil || resp.StatusCode != http.StatusOK {
			t.Fatalf("MCP status=%d body=%s err=%v", resp.StatusCode, raw, readErr)
		}
		var rpc rpcResponse
		if err := json.Unmarshal(raw, &rpc); err != nil {
			t.Fatalf("MCP decode: %v (%s)", err, raw)
		}
		return rpc
	}
	restore := func(t *testing.T, args ContextRestoreRequest) CtxRestoreResult {
		t.Helper()
		args.ID, args.TraceID = id, trace
		rpc := post(t, owner, args)
		if rpc.Error != nil {
			t.Fatalf("restore RPC refusal: %+v", rpc.Error)
		}
		var result CtxRestoreResult
		decodeMCPResult(t, rpc.Result, &result)
		return result
	}

	for _, tc := range []struct {
		name   string
		args   ContextRestoreRequest
		offset int
	}{
		{"positive_offset", ContextRestoreRequest{Offset: 1, Limit: maxInt}, 1},
		{"at_eof", ContextRestoreRequest{Offset: len(body), Limit: maxInt}, len(body)},
		{"past_eof", ContextRestoreRequest{Offset: maxInt, Limit: maxInt}, len(body)},
		{"max_bytes", ContextRestoreRequest{Offset: 1, MaxBytes: maxInt}, 1},
		{"continuation_offset", ContextRestoreRequest{ContinuationToken: "offset:1", Limit: maxInt}, 1},
		{"inclusive_range", ContextRestoreRequest{Range: "bytes=1-" + strconv.Itoa(maxInt)}, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := restore(t, tc.args)
			if got.Bytes != body[tc.offset:] || got.Offset != tc.offset || got.Limit != maxInt || got.TotalBytes != len(body) || got.HasMore || got.NextOffset != len(body) || got.Continuation != nil || got.ContinuationToken != "" {
				t.Fatalf("overflow-safe terminal slice: %+v", got)
			}
			if got.ID != id || got.TraceID != trace || got.Schema != ctxRestoreSchema {
				t.Fatalf("restore identity changed: %+v", got)
			}
		})
	}

	// A real cap still wins over a huge requested limit, and its continuation
	// remains actionable. The reported Limit keeps its existing effective-limit
	// meaning rather than being changed to the number of bytes returned.
	first := restore(t, ContextRestoreRequest{Offset: 1, Limit: maxInt, MaxBytes: 3})
	if first.Bytes != body[1:4] || first.Offset != 1 || first.Limit != 3 || !first.HasMore || first.NextOffset != 4 || first.ContinuationToken != "4" || first.Continuation == nil {
		t.Fatalf("bounded first page: %+v", first)
	}
	if first.Continuation.Arguments.Offset != 4 || first.Continuation.Arguments.Limit != 3 || first.Continuation.Range != "4-7" {
		t.Fatalf("continuation metadata changed: %+v", first.Continuation)
	}
	next := restore(t, first.Continuation.Arguments)
	if next.Bytes != body[4:7] || next.Offset != 4 || next.Limit != 3 || next.NextOffset != 7 || !next.HasMore {
		t.Fatalf("advertised next page: %+v", next)
	}
	whole := restore(t, ContextRestoreRequest{})
	if whole.Bytes != body || whole.Offset != 0 || whole.Limit != len(body) || whole.HasMore {
		t.Fatalf("unbounded compatibility changed: %+v", whole)
	}

	// Scope and trust refusals still happen before any arithmetic or disclosure.
	dangerousBounds := ContextRestoreRequest{ID: id, TraceID: trace, Offset: 1, Limit: maxInt}
	if rpc := post(t, "other-principal", dangerousBounds); rpc.Error == nil || rpc.Result != nil {
		t.Fatalf("cross-principal restore disclosed content: %+v", rpc)
	}
	// Set only this in-memory fixture's seal, avoiding unrelated durable CAS I/O.
	srv.ctxRestoreMu.Lock()
	for i := range srv.ctxRestore[trace].entries {
		if srv.ctxRestore[trace].entries[i].id == id {
			srv.ctxRestore[trace].entries[i].sealed = true
		}
	}
	srv.ctxRestoreMu.Unlock()
	if rpc := post(t, owner, dangerousBounds); rpc.Error == nil || rpc.Result != nil {
		t.Fatalf("sealed restore disclosed content: %+v", rpc)
	}
}

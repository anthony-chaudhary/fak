package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// fak-test:justify why=regression when=changed:internal/gateway/**
// fak-test:runtime fast est=1s lane=default
func TestFakReadBatchPaginationOverStdioAndCache(t *testing.T) {
	t.Setenv("FAK_TOON_WIRE", "")
	dir := t.TempDir()
	paths := []string{filepath.Join(dir, "b.txt"), filepath.Join(dir, "a.txt")}
	prefixes := []string{"b", "a"}
	for i, path := range paths {
		lines := make([]string, 8)
		for line := range lines {
			lines[line] = fmt.Sprintf("%s%d", prefixes[i], line+1)
		}
		if err := os.WriteFile(path, []byte(strings.Join(lines, "\n")), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	adj := &batchReadAdj{}
	srv := newBatchReadServer(t, dir, adj)
	cases := []struct {
		name        string
		offset      int
		limit       int
		lineNumbers bool
		wantLines   []int
		cacheHit    bool
	}{
		{name: "numbered-cold", offset: 2, limit: 3, lineNumbers: true, wantLines: []int{2, 3, 4}},
		{name: "numbered-reuse", offset: 2, limit: 3, lineNumbers: true, wantLines: []int{2, 3, 4}, cacheHit: true},
		{name: "unnumbered-window", offset: 2, limit: 3, wantLines: []int{2, 3, 4}},
		{name: "different-window", offset: 5, limit: 2, lineNumbers: true, wantLines: []int{5, 6}},
		{name: "default-full", wantLines: []int{1, 2, 3, 4, 5, 6, 7, 8}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			before := srv.k.Counters()
			seenBefore := len(adj.seen())
			args := map[string]any{"file_paths": paths, "trace_id": "batch-window-trace"}
			if tc.offset > 0 {
				args["offset"] = tc.offset
			}
			if tc.limit > 0 {
				args["limit"] = tc.limit
			}
			if tc.lineNumbers {
				args["line_numbers"] = true
			}
			resp := batchWindowStdioResponse(t, srv, args)
			after := srv.k.Counters()
			if resp.TraceID != "batch-window-trace" || resp.ItemCount != len(paths) || len(resp.Results) != len(paths) {
				t.Fatalf("batch trace=%q item_count=%d results=%d", resp.TraceID, resp.ItemCount, len(resp.Results))
			}
			for i, item := range resp.Results {
				if item.FilePath != paths[i] || item.Verdict.Kind != "ALLOW" || item.Result == nil || item.Result.Status != "OK" || item.Error != "" {
					t.Fatalf("item %d did not preserve its requested path and successful outcome", i)
				}
				body := decodeFakReadPayload(t, item.Result)
				wantLines := make([]string, len(tc.wantLines))
				for j, line := range tc.wantLines {
					wantLines[j] = fmt.Sprintf("%s%d", prefixes[i], line)
					if tc.lineNumbers {
						wantLines[j] = fmt.Sprintf("%d: %s", line, wantLines[j])
					}
				}
				want := strings.Join(wantLines, "\n")
				if body["content"] != want {
					t.Fatalf("item %d content=%q, want %q", i, body["content"], want)
				}
				if body["file_path"] != paths[i] {
					t.Fatalf("item %d payload path=%v, want %q", i, body["file_path"], paths[i])
				}
				if tc.limit > 0 && (body["offset"] != float64(tc.offset) || body["limit"] != float64(tc.limit) || body["total_lines"] != float64(8) || body["truncated"] != true) {
					t.Fatalf("item %d pagination offset=%v limit=%v total_lines=%v truncated=%v", i, body["offset"], body["limit"], body["total_lines"], body["truncated"])
				}
				receipt := receiptOf(t, body)
				outcome, witness := "executed_cold_read", "filesystem_read"
				if tc.cacheHit {
					outcome, witness = "verified_fresh_reuse", "vdso"
				}
				if receipt["schema"] != "fak-read-receipt/1" || receipt["outcome"] != outcome || receipt["witness"] != witness || receipt["freshness_verified"] != true || receipt["bytes"] != float64(len([]byte(want))) {
					t.Fatalf("item %d receipt=%v", i, receipt)
				}
			}
			wantEngineCalls, wantHits := len(paths), 0
			if tc.cacheHit {
				wantEngineCalls, wantHits = 0, len(paths)
			}
			if got := int(after.EngineCalls - before.EngineCalls); got != wantEngineCalls {
				t.Fatalf("engine calls=%d, want %d", got, wantEngineCalls)
			}
			if got := int(after.VDSOHits - before.VDSOHits); got != wantHits {
				t.Fatalf("fresh cache hits=%d, want %d", got, wantHits)
			}
			if !tc.cacheHit {
				seen := adj.seen()[seenBefore:]
				next := 0
				for _, path := range seen {
					if next < len(paths) && path == paths[next] {
						next++
					}
				}
				if next != len(paths) {
					t.Fatalf("per-item adjudications=%v, missing requested order %v", seen, paths)
				}
			}
		})
	}
}

func batchWindowStdioResponse(t *testing.T, srv *Server, args map[string]any) FakReadBatchResponse {
	t.Helper()
	var in, out bytes.Buffer
	if err := json.NewEncoder(&in).Encode(map[string]any{
		"jsonrpc": "2.0", "id": 1, "method": "tools/call",
		"params": map[string]any{"name": "fak_read", "arguments": args},
	}); err != nil {
		t.Fatal(err)
	}
	if err := srv.ServeStdio(context.Background(), &in, &out); err != nil {
		t.Fatalf("ServeStdio: %v", err)
	}
	for _, frame := range decodeRPCLines(t, out.Bytes()) {
		if frame.ID != float64(1) {
			continue
		}
		if frame.Error != nil {
			t.Fatalf("tools/call RPC error code=%d", frame.Error.Code)
		}
		result, ok := frame.Result.(map[string]any)
		if !ok {
			t.Fatal("tools/call result was not an object")
		}
		blocks, ok := result["content"].([]any)
		if !ok || len(blocks) != 1 {
			t.Fatal("tools/call requires one content block")
		}
		block, ok := blocks[0].(map[string]any)
		if !ok {
			t.Fatal("tools/call content block was not an object")
		}
		result["content"] = []map[string]any{block}
		var resp FakReadBatchResponse
		if err := json.Unmarshal([]byte(mcpResultText(t, result)), &resp); err != nil {
			t.Fatalf("decode batch response: %v", err)
		}
		return resp
	}
	t.Fatal("missing tools/call response")
	return FakReadBatchResponse{}
}

package gateway

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/vdso"
)

// batchReadAdj records every Read adjudication and refuses one named path. The
// record is the safety witness: a batch cannot be treated as one aggregate call.
type batchReadAdj struct {
	mu    sync.Mutex
	paths []string
}

func (*batchReadAdj) Caps() []abi.Capability { return nil }

func (a *batchReadAdj) Adjudicate(ctx context.Context, c *abi.ToolCall) abi.Verdict {
	if c.Tool != "Read" && c.Tool != "fak_read" {
		return abi.Verdict{Kind: abi.VerdictAllow, By: "batch-read-test"}
	}
	var args struct {
		FilePath string `json:"file_path"`
	}
	_ = json.Unmarshal(resolveBytes(ctx, c.Args), &args)
	a.mu.Lock()
	a.paths = append(a.paths, args.FilePath)
	a.mu.Unlock()
	if filepath.Base(args.FilePath) == "refused.go" {
		return abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonPolicyBlock, By: "batch-read-test"}
	}
	return abi.Verdict{Kind: abi.VerdictAllow, By: "batch-read-test"}
}

func (a *batchReadAdj) seen() []string {
	a.mu.Lock()
	defer a.mu.Unlock()
	return append([]string(nil), a.paths...)
}

func newBatchReadServer(t *testing.T, dir string, adj abi.Adjudicator) *Server {
	t.Helper()
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterAdjudicator(0, adj)
	agent.RegisterReadEngine(dir)
	v := vdso.New(vdso.DefaultCacheSize)
	v.SetGranularity(vdso.Resource)
	abi.RegisterFastPath(1, v)
	abi.RegisterEmitter(v)
	srv, err := New(Config{EngineID: "fakread", Model: "m", VDSO: true})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(srv.Close)
	return srv
}

func callFakReadBatch(t *testing.T, srv *Server, paths []string) FakReadBatchResponse {
	t.Helper()
	params, err := json.Marshal(map[string]any{
		"name": "fak_read",
		"arguments": map[string]any{
			"file_paths": paths,
			"trace_id":   "batch-trace",
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	got, rpcErr := srv.callTool(context.Background(), params)
	if rpcErr != nil {
		t.Fatalf("callTool: %+v", rpcErr)
	}
	var resp FakReadBatchResponse
	if err := json.Unmarshal([]byte(mcpResultText(t, got.(map[string]any))), &resp); err != nil {
		t.Fatalf("decode batch response: %v", err)
	}
	return resp
}

func TestFakReadBatchAdjudicatesEveryItemAndKeepsErrorsLocal(t *testing.T) {
	dir := t.TempDir()
	paths := []string{
		filepath.Join(dir, "first.go"),
		filepath.Join(dir, "refused.go"),
		filepath.Join(dir, "missing.go"),
		filepath.Join(dir, "last.go"),
	}
	for i, path := range []string{paths[0], paths[1], paths[3]} {
		if err := os.WriteFile(path, []byte(strings.Repeat("x", i+1)), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	adj := &batchReadAdj{}
	srv := newBatchReadServer(t, dir, adj)

	before := srv.k.Counters()
	resp := callFakReadBatch(t, srv, paths)
	after := srv.k.Counters()
	if resp.ItemCount != len(paths) || len(resp.Results) != len(paths) {
		t.Fatalf("batch shape=%+v", resp)
	}
	for i, path := range paths {
		if resp.Results[i].FilePath != path {
			t.Fatalf("result[%d].file_path=%q, want %q", i, resp.Results[i].FilePath, path)
		}
	}
	if resp.Results[0].Verdict.Kind != "ALLOW" || resp.Results[0].Result == nil {
		t.Fatalf("first allowed item=%+v", resp.Results[0])
	}
	if resp.Results[1].Verdict.Kind != "DENY" {
		t.Fatalf("refused item=%+v", resp.Results[1])
	}
	if resp.Results[2].Verdict.Kind != "ALLOW" || resp.Results[2].Result == nil || resp.Results[2].Result.Status != "ERROR" || resp.Results[2].Error == "" {
		t.Fatalf("missing-file error was not isolated to its row: %+v", resp.Results[2])
	}
	if resp.Results[3].Verdict.Kind != "ALLOW" || resp.Results[3].Result == nil {
		t.Fatalf("last item was dropped after peer failures: %+v", resp.Results[3])
	}
	if got := after.EngineCalls - before.EngineCalls; got != 3 {
		t.Fatalf("engine calls=%d, want only the three allowed items (before=%+v after=%+v)", got, before, after)
	}
	seen := adj.seen()
	previous := -1
	for _, path := range paths {
		found := -1
		for i := previous + 1; i < len(seen); i++ {
			if seen[i] == path {
				found = i
				break
			}
		}
		if found < 0 {
			t.Fatalf("adjudications=%v, missing independent item %q after index %d", seen, path, previous)
		}
		previous = found
	}
}

func TestFakReadBatchCacheSemanticsArePerItem(t *testing.T) {
	dir := t.TempDir()
	paths := make([]string, 5)
	for i := range paths {
		paths[i] = filepath.Join(dir, "file"+string(rune('a'+i))+".go")
		if err := os.WriteFile(paths[i], []byte("package p"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	adj := &batchReadAdj{}
	srv := newBatchReadServer(t, dir, adj)
	for _, path := range paths[:3] {
		if _, env, err := srv.fakRead(context.Background(), path, "prime-trace", ""); err != nil || env == nil {
			t.Fatalf("prime %q: env=%+v err=%v", path, env, err)
		}
	}
	before := srv.k.Counters()
	resp := callFakReadBatch(t, srv, paths)
	after := srv.k.Counters()
	if len(resp.Results) != 5 {
		t.Fatalf("results=%+v", resp.Results)
	}
	if got := after.VDSOHits - before.VDSOHits; got != 3 {
		t.Fatalf("verified-fresh hits=%d, want 3 (before=%+v after=%+v)", got, before, after)
	}
	if got := after.EngineCalls - before.EngineCalls; got != 2 {
		t.Fatalf("disk-backed engine calls=%d, want 2 (before=%+v after=%+v)", got, before, after)
	}
}

func TestFakReadSingularWireShapeIsUnchanged(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "one.go")
	if err := os.WriteFile(path, []byte("package one"), 0o644); err != nil {
		t.Fatal(err)
	}
	srv := newBatchReadServer(t, dir, &batchReadAdj{})
	params, _ := json.Marshal(map[string]any{
		"name":      "fak_read",
		"arguments": map[string]any{"file_path": path, "trace_id": "singular-trace"},
	})
	got, rpcErr := srv.callTool(context.Background(), params)
	if rpcErr != nil {
		t.Fatalf("callTool: %+v", rpcErr)
	}
	var resp SyscallResponse
	if err := json.Unmarshal([]byte(mcpResultText(t, got.(map[string]any))), &resp); err != nil {
		t.Fatal(err)
	}
	if resp.TraceID != "singular-trace" || resp.Verdict.Kind != "ALLOW" || resp.Result == nil || !strings.Contains(resp.Result.Content, "package one") {
		t.Fatalf("legacy singular response=%+v", resp)
	}
}

func TestFakAdmitBatchContinuesAfterItemError(t *testing.T) {
	dir := t.TempDir()
	srv := newBatchReadServer(t, dir, &batchReadAdj{})
	params, _ := json.Marshal(map[string]any{
		"name": "fak_admit",
		"arguments": map[string]any{
			"trace_id": "admit-batch",
			"items": []map[string]any{
				{"tool": "", "result": map[string]any{"bad": true}},
				{"tool": "Read", "result": map[string]any{"ok": true}},
			},
		},
	})
	got, rpcErr := srv.callTool(context.Background(), params)
	if rpcErr != nil {
		t.Fatalf("one item error aborted the batch: %+v", rpcErr)
	}
	var resp FakAdmitBatchResponse
	if err := json.Unmarshal([]byte(mcpResultText(t, got.(map[string]any))), &resp); err != nil {
		t.Fatal(err)
	}
	if resp.ItemCount != 2 || len(resp.Results) != 2 || resp.Results[0].Error == "" || resp.Results[1].Result == nil {
		t.Fatalf("admit batch did not preserve per-item outcomes: %+v", resp)
	}
}

func TestMCPPerItemSchemasAdvertisePreferredBatchAxes(t *testing.T) {
	want := map[string]string{"fak_read": "file_paths", "fak_admit": "items"}
	for _, desc := range toolDescriptors() {
		name, _ := desc["name"].(string)
		axis, ok := want[name]
		if !ok {
			continue
		}
		description, _ := desc["description"].(string)
		schema, _ := desc["inputSchema"].(json.RawMessage)
		if !strings.Contains(description, "Prefer") || !strings.Contains(string(schema), `"`+axis+`"`) {
			t.Fatalf("%s does not advertise preferred %s batch axis: description=%q schema=%s", name, axis, description, schema)
		}
		delete(want, name)
	}
	if len(want) != 0 {
		t.Fatalf("missing per-item tool descriptors: %v", want)
	}
}

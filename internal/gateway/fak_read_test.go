package gateway

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/vdso"
)

// readAdj allows Read (and the write-shaped Edit used to invalidate) so the fak_read serve
// path reaches the vDSO rather than being denied by the policy floor.
type readAdj struct{}

func (readAdj) Caps() []abi.Capability { return nil }
func (readAdj) Adjudicate(ctx context.Context, c *abi.ToolCall) abi.Verdict {
	return abi.Verdict{Kind: abi.VerdictAllow, By: "test"}
}

type deferReadAdj struct{}

func (deferReadAdj) Caps() []abi.Capability { return nil }
func (deferReadAdj) Adjudicate(context.Context, *abi.ToolCall) abi.Verdict {
	return abi.Verdict{Kind: abi.VerdictDefer, By: "test"}
}

// TestFakRead_ServesFreshHitNotStale is the end-to-end #795 vToolcall witness: a second
// fak_read of an unchanged file is served from the vDSO with NO disk read (VDSOHits++,
// EngineCalls flat), and after a Write/Edit to that path the next fak_read MISSES and reads
// the real file again (the #795 per-path invalidator turns hit -> miss on a write, never
// serving stale). No Claude Code change is involved — this is the kernel-mediated read the
// fak_read MCP tool exposes.
func TestFakRead_ServesFreshHitNotStale(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterAdjudicator(0, readAdj{})

	dir := t.TempDir()
	agent.RegisterReadEngine(dir) // the confined real-read miss path
	file := filepath.Join(dir, "a.go")
	if err := os.WriteFile(file, []byte("package a // v1"), 0o644); err != nil {
		t.Fatal(err)
	}

	// Arm a fresh vDSO as both the serve fast path and the tier-2 fill emitter, at Resource
	// granularity (what binds the per-path tag files:<path>; Global flushes everything,
	// Namespace can't reach the leaf). This mirrors newSharingServer.
	v := vdso.New(vdso.DefaultCacheSize)
	v.SetGranularity(vdso.Resource)
	abi.RegisterFastPath(1, v)
	abi.RegisterEmitter(v)

	srv, err := New(Config{EngineID: "fakread", Model: "m", VDSO: true})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(srv.Close)

	read := func() string {
		t.Helper()
		_, env, err := srv.fakRead(context.Background(), file, "trace-1", "")
		if err != nil {
			t.Fatalf("fakRead: %v", err)
		}
		if env == nil {
			t.Fatal("fakRead: nil result")
		}
		return env.Content
	}

	// Call 1: MISS — the real engine reads the file.
	c0 := srv.k.Counters()
	got1 := read()
	if !strings.Contains(got1, "package a // v1") {
		t.Fatalf("call 1 content = %q, want the file body", got1)
	}
	c1 := srv.k.Counters()
	if c1.EngineCalls <= c0.EngineCalls {
		t.Fatalf("call 1 did not dispatch to the read engine (EngineCalls %d -> %d): the miss path did not run",
			c0.EngineCalls, c1.EngineCalls)
	}

	// Call 2: HIT — served from the vDSO, NO disk read, NO engine dispatch.
	got2 := read()
	c2 := srv.k.Counters()
	var body1, body2 map[string]any
	if err := json.Unmarshal([]byte(got1), &body1); err != nil {
		t.Fatalf("call 1 payload: %v", err)
	}
	if err := json.Unmarshal([]byte(got2), &body2); err != nil {
		t.Fatalf("call 2 payload: %v", err)
	}
	if body2["content"] != body1["content"] {
		t.Fatalf("call 2 content = %q, want identical file bytes %q", body2["content"], body1["content"])
	}
	if c2.VDSOHits <= c1.VDSOHits {
		t.Fatalf("call 2 was NOT served from the vDSO (VDSOHits %d -> %d): the cache hit did not fire",
			c1.VDSOHits, c2.VDSOHits)
	}
	if c2.EngineCalls != c1.EngineCalls {
		t.Fatalf("call 2 dispatched to the engine anyway (EngineCalls %d -> %d): the disk read was NOT avoided",
			c1.EngineCalls, c2.EngineCalls)
	}

	// Now an Edit to the same path bumps files:<path> (the #795 invalidator). Emit a
	// write-shaped completion through the SAME vDSO the kernel serves from.
	editArgs, _ := json.Marshal(map[string]string{"file_path": file, "old_string": "v1", "new_string": "v2"})
	v.Emit(abi.Event{
		Kind: abi.EvComplete,
		Call: &abi.ToolCall{Tool: "Edit", Args: abi.Ref{Kind: abi.RefInline, Inline: editArgs}},
		Result: &abi.Result{
			Status: abi.StatusOK,
			Meta:   map[string]string{},
		},
	})
	// Change the file on disk so a re-read returns the new bytes (proving it actually re-ran).
	if err := os.WriteFile(file, []byte("package a // v2"), 0o644); err != nil {
		t.Fatal(err)
	}

	// Call 3: MISS again — the per-path write stranded the cached entry, so the engine
	// re-reads and returns the NEW bytes.
	got3 := read()
	c3 := srv.k.Counters()
	if c3.EngineCalls <= c2.EngineCalls {
		t.Fatalf("call 3 did NOT re-dispatch after the Edit (EngineCalls %d -> %d): a STALE cached read was served",
			c2.EngineCalls, c3.EngineCalls)
	}
	if !strings.Contains(got3, "package a // v2") {
		t.Fatalf("call 3 content = %q, want the post-edit body (stale serve)", got3)
	}
}

// TestFakRead_ConfinesPath proves the read engine refuses a path escaping its root: a
// model-supplied "../" traversal cannot exfiltrate a file outside the working tree.
func TestFakRead_ConfinesPath(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterAdjudicator(0, readAdj{})

	dir := t.TempDir()
	// A secret one level ABOVE the read root.
	secret := filepath.Join(filepath.Dir(dir), "secret.txt")
	_ = os.WriteFile(secret, []byte("TOP SECRET"), 0o644)
	t.Cleanup(func() { _ = os.Remove(secret) })

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

	_, env, err := srv.fakRead(context.Background(), "../secret.txt", "trace-2", "")
	if err != nil {
		t.Fatalf("fakRead: %v", err)
	}
	if env == nil {
		t.Fatal("nil result")
	}
	if strings.Contains(env.Content, "TOP SECRET") {
		t.Fatalf("path confinement FAILED: a '../' traversal read a file outside the root: %q", env.Content)
	}
	if !strings.Contains(env.Content, "escapes the read root") {
		t.Fatalf("expected a confinement refusal, got %q", env.Content)
	}
}

// TestNewArmsAdvertisedFakReadRoute is the #10296 regression witness: a gateway
// that advertises fak_read must register its execution route before the first
// single or batch call. Before the fix both calls were ALLOWed and then failed
// with "no engine registered for route fakread".
func TestNewArmsAdvertisedFakReadRoute(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterAdjudicator(0, readAdj{})
	abi.RegisterEngine("inkernel", echoEngine{})
	if abi.Engine(agent.FakReadEngineID) != nil {
		t.Fatal("fakread engine unexpectedly registered before gateway startup")
	}

	srv, err := New(Config{EngineID: "inkernel", Model: "guard-mcp-test", VDSO: true})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(srv.Close)
	if abi.Engine(agent.FakReadEngineID) == nil {
		t.Fatal("gateway advertised fak_read without registering its fakread engine")
	}

	files := []string{"fak_read_test.go", "mcp.go"}
	want := make(map[string]string, len(files))
	for _, path := range files {
		b, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("read fixture %s: %v", path, err)
		}
		want[path] = string(b)
	}

	wv, single, err := srv.fakRead(context.Background(), files[0], "guard-single", "")
	if err != nil {
		t.Fatalf("single fak_read: %v", err)
	}
	if wv.Kind != "ALLOW" || single == nil || single.Status != "OK" {
		t.Fatalf("single fak_read result: verdict=%+v result=%+v", wv, single)
	}
	var singlePayload struct {
		FilePath string `json:"file_path"`
		Content  string `json:"content"`
	}
	if err := json.Unmarshal([]byte(single.Content), &singlePayload); err != nil {
		t.Fatalf("decode single payload: %v", err)
	}
	if singlePayload.FilePath != files[0] || singlePayload.Content != want[files[0]] || single.Meta["engine"] != agent.FakReadEngineID {
		t.Fatalf("single fak_read payload=%+v meta=%+v", singlePayload, single.Meta)
	}

	batch := srv.fakReadBatch(context.Background(), files, "guard-batch", "")
	if batch.ItemCount != len(files) || len(batch.Results) != len(files) {
		t.Fatalf("batch shape: %+v", batch)
	}
	for i, item := range batch.Results {
		if item.Error != "" || item.Verdict.Kind != "ALLOW" || item.Result == nil || item.Result.Status != "OK" {
			t.Fatalf("batch result[%d]=%+v", i, item)
		}
		var payload struct {
			FilePath string `json:"file_path"`
			Content  string `json:"content"`
		}
		if err := json.Unmarshal([]byte(item.Result.Content), &payload); err != nil {
			t.Fatalf("decode batch result[%d]: %v", i, err)
		}
		if payload.FilePath != files[i] || payload.Content != want[files[i]] || item.Result.Meta["engine"] != agent.FakReadEngineID {
			t.Fatalf("batch result[%d] payload=%+v meta=%+v", i, payload, item.Result.Meta)
		}
	}
}

func decodeFakReadPayload(t testing.TB, env *ResultEnvelope) map[string]any {
	t.Helper()
	if env == nil {
		t.Fatal("nil fak_read result")
	}
	var body map[string]any
	if err := json.Unmarshal([]byte(env.Content), &body); err != nil {
		t.Fatalf("decode fak_read payload: %v", err)
	}
	return body
}

func receiptOf(t testing.TB, body map[string]any) map[string]any {
	t.Helper()
	r, ok := body["receipt"].(map[string]any)
	if !ok {
		t.Fatalf("missing receipt: %v", body)
	}
	return r
}

func TestFakReadReceiptOutcomesAndNoLeak(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterAdjudicator(0, readAdj{})
	root := t.TempDir()
	agent.RegisterReadEngine(root)
	path := filepath.Join(root, "receipt.txt")
	if err := os.WriteFile(path, []byte("receipt-secret-body"), 0o644); err != nil {
		t.Fatal(err)
	}
	v := vdso.New(vdso.DefaultCacheSize)
	v.SetGranularity(vdso.Resource)
	abi.RegisterFastPath(1, v)
	abi.RegisterEmitter(v)
	srv, err := New(Config{EngineID: "fakread", Model: "m", VDSO: true})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)

	read := func(p string) map[string]any {
		_, env, err := srv.fakRead(context.Background(), p, "receipt-trace", "git:fixture")
		if err != nil {
			t.Fatal(err)
		}
		return decodeFakReadPayload(t, env)
	}
	cold := read(path)
	coldReceipt := receiptOf(t, cold)
	if coldReceipt["outcome"] != "executed_cold_read" || coldReceipt["witness"] != "filesystem_read" {
		t.Fatalf("cold receipt=%v", coldReceipt)
	}
	if coldReceipt["bytes"] != float64(len("receipt-secret-body")) || coldReceipt["duration_ns"].(float64) < 0 {
		t.Fatalf("cold accounting=%v", coldReceipt)
	}
	hit := read(path)
	hitReceipt := receiptOf(t, hit)
	if hitReceipt["outcome"] != "verified_fresh_reuse" || hitReceipt["witness"] != "vdso" {
		t.Fatalf("hit receipt=%v", hitReceipt)
	}
	if hit["content"] != cold["content"] {
		t.Fatalf("reuse changed bytes")
	}

	missing := read(filepath.Join(root, "missing-secret-name.txt"))
	missingReceipt := receiptOf(t, missing)
	errMeta, ok := missingReceipt["error"].(map[string]any)
	if !ok || errMeta["code"] != "not_found" || errMeta["source"] != "filesystem" {
		t.Fatalf("typed error=%v", missingReceipt)
	}
	encoded, _ := json.Marshal(missingReceipt)
	if strings.Contains(string(encoded), "receipt-secret-body") || strings.Contains(string(encoded), "missing-secret-name") || strings.Contains(string(encoded), root) {
		t.Fatalf("receipt leaked content/path: %s", encoded)
	}
}

func TestFakReadNeverStaleAcrossMutationClasses(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterAdjudicator(0, readAdj{})
	root := t.TempDir()
	agent.RegisterReadEngine(root)
	path := filepath.Join(root, "mutate.txt")
	v := vdso.New(vdso.DefaultCacheSize)
	v.SetGranularity(vdso.Resource)
	abi.RegisterFastPath(1, v)
	abi.RegisterEmitter(v)
	srv, err := New(Config{EngineID: "fakread", Model: "m", VDSO: true})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	invalidate := func() {
		args, _ := json.Marshal(map[string]string{"file_path": path, "old_string": "x", "new_string": "y"})
		v.Emit(abi.Event{Kind: abi.EvComplete, Call: &abi.ToolCall{Tool: "Edit", Args: abi.Ref{Kind: abi.RefInline, Inline: args}}, Result: &abi.Result{Status: abi.StatusOK, Meta: map[string]string{}}})
	}
	read := func() map[string]any {
		_, env, err := srv.fakRead(context.Background(), path, "mutate-trace", "")
		if err != nil {
			t.Fatal(err)
		}
		return decodeFakReadPayload(t, env)
	}
	for _, want := range []string{"aaaa", "bbbb", "x", "growth-value"} {
		if err := os.WriteFile(path, []byte(want), 0o644); err != nil {
			t.Fatal(err)
		}
		invalidate()
		body := read()
		if body["content"] != want {
			t.Fatalf("stale after mutation: got %q want %q", body["content"], want)
		}
		if receiptOf(t, body)["outcome"] != "executed_cold_read" {
			t.Fatalf("mutation reused stale entry: %v", body)
		}
	}
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	invalidate()
	if body := read(); receiptOf(t, body)["error"].(map[string]any)["code"] != "not_found" {
		t.Fatalf("delete=%v", body)
	}
	if err := os.WriteFile(path, []byte("recreated"), 0o644); err != nil {
		t.Fatal(err)
	}
	invalidate()
	if body := read(); body["content"] != "recreated" {
		t.Fatalf("recreate=%v", body)
	}
}

func TestFakReadPreservesDefaultDeny(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterAdjudicator(0, deferReadAdj{})
	root := t.TempDir()
	agent.RegisterReadEngine(root)
	path := filepath.Join(root, "must-not-read.txt")
	if err := os.WriteFile(path, []byte("denied-secret"), 0o644); err != nil {
		t.Fatal(err)
	}
	srv, err := New(Config{EngineID: "fakread", Model: "m"})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)

	before := srv.k.Counters()
	verdict, env, err := srv.fakRead(context.Background(), path, "deny-trace", "")
	if err != nil {
		t.Fatal(err)
	}
	after := srv.k.Counters()
	if verdict.Kind != "DENY" || verdict.Reason != "DEFAULT_DENY" {
		t.Fatalf("verdict=%+v", verdict)
	}
	if env == nil || env.Status != "ERROR" || env.Content != "" {
		t.Fatalf("denied read leaked a payload: %+v", env)
	}
	if after.EngineCalls != before.EngineCalls {
		t.Fatalf("denied read reached engine: before=%+v after=%+v", before, after)
	}
}

// TestFakRead_LargeFileWithCtxMMU proves that when ctxmmu is active in the result-admitter chain,
// a normal code file > 4 KiB (e.g. 20 KiB) is NOT paged out to a {"_paged":true} stub and returns
// its full content, satisfying the documented fak_read contract.
func TestFakRead_LargeFileWithCtxMMU(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterAdjudicator(0, readAdj{})

	root := t.TempDir()
	agent.RegisterReadEngine(root)
	path := filepath.Join(root, "large_file.go")
	largeContent := strings.Repeat("// line of code in large file\n", 800) // ~24 KiB, > 4 KiB
	if err := os.WriteFile(path, []byte(largeContent), 0o644); err != nil {
		t.Fatal(err)
	}

	srv, err := New(Config{EngineID: "fakread", Model: "m"})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)

	wv, env, err := srv.fakRead(context.Background(), path, "large-trace", "")
	if err != nil {
		t.Fatalf("fakRead: %v", err)
	}
	if wv.Kind != "ALLOW" || env == nil || env.Status != "OK" {
		t.Fatalf("result: verdict=%+v env=%+v", wv, env)
	}
	if strings.Contains(env.Content, `"_paged":true`) {
		t.Fatalf("fakRead returned paged-out stub for 24 KiB file: %s", env.Content)
	}
	var body map[string]any
	if err := json.Unmarshal([]byte(env.Content), &body); err != nil {
		t.Fatalf("decode payload: %v", err)
	}
	if body["content"] != largeContent {
		t.Fatalf("content mismatch: got %d bytes, want %d bytes", len(body["content"].(string)), len(largeContent))
	}
}

// TestFakContextRestore_ResolvesMMUPagedBlob proves that an MMU _paged.ref pointer can be
// immediately resolved via fak_context_restore under the same session/trace (#10018).
func TestFakContextRestore_ResolvesMMUPagedBlob(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})

	srv, err := New(Config{EngineID: "fakread", Model: "m"})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)

	// Store a test payload into the blob CAS
	originalText := "package main\n// this is a paged-out tool result stored in blob CAS\nfunc Hello() string { return \"world\" }\n"
	b, ok := abi.PageOut("blob")
	if !ok {
		t.Fatal("blob PageOut backend not registered")
	}
	ref, err := b.PageOut(context.Background(), abi.Ref{Kind: abi.RefInline, Inline: []byte(originalText)})
	if err != nil {
		t.Fatalf("PageOut: %v", err)
	}
	digest := ref.Digest

	// Restore using fak_context_restore via srv.restoreContext
	res, err := srv.restoreContext("", ContextRestoreRequest{
		ID:      digest,
		TraceID: "test-trace",
	})
	if err != nil {
		t.Fatalf("restoreContext failed for MMU blob digest %q: %v", digest, err)
	}
	if res.Bytes != originalText {
		t.Fatalf("restoreContext bytes mismatch: got %q, want %q", res.Bytes, originalText)
	}
	if res.ID != digest {
		t.Fatalf("restoreContext ID mismatch: got %q, want %q", res.ID, digest)
	}
}

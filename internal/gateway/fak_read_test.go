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
	if got2 != got1 {
		t.Fatalf("call 2 content = %q, want identical to call 1 %q", got2, got1)
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

package engine_test

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	// Complete() stores payloads through abi.ActiveResolver(); the blob backend's
	// init() registers the RegionBackend so the resolver is non-nil. We never call
	// abi.ResetForTest() (that would unregister it).
	_ "github.com/anthony-chaudhary/fak/internal/blob"
	"github.com/anthony-chaudhary/fak/internal/engine"
)

// inlineCall builds a tool call carrying an inline args payload (no resolver
// needed for the request side).
func inlineCall(tool, args string) *abi.ToolCall {
	return &abi.ToolCall{
		Tool: tool,
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(args)},
		Meta: map[string]string{"readOnlyHint": "true"},
	}
}

// resolvePayload materializes the bytes a Result's Payload Ref points at, via the
// active resolver (handles both inline and blob-backed refs).
func resolvePayload(t *testing.T, ctx context.Context, r abi.Ref) []byte {
	t.Helper()
	if r.Kind == abi.RefInline {
		return r.Inline
	}
	res := abi.ActiveResolver()
	if res == nil {
		t.Fatal("ActiveResolver() is nil; blob backend not registered")
	}
	b, err := res.Resolve(ctx, r)
	if err != nil {
		t.Fatalf("resolve payload: %v", err)
	}
	return b
}

// ---------------------------------------------------------------------------
// Unit 42 — Mock.Complete: StatusOK + integer-parseable token meta.
// ---------------------------------------------------------------------------

func TestMockCompleteStatusAndUsage(t *testing.T) {
	ctx := context.Background()
	m := &engine.Mock{}
	res, err := m.Complete(ctx, inlineCall("search", `{"q":"hello"}`))
	if err != nil {
		t.Fatalf("Mock.Complete error: %v", err)
	}
	if res.Status != abi.StatusOK {
		t.Fatalf("Mock status = %v, want StatusOK", res.Status)
	}
	if res.Meta["engine"] != "mock" {
		t.Fatalf("Mock engine meta = %q, want mock", res.Meta["engine"])
	}
	in, err := strconv.Atoi(res.Meta["input_tokens"])
	if err != nil {
		t.Fatalf("input_tokens %q not an int: %v", res.Meta["input_tokens"], err)
	}
	out, err := strconv.Atoi(res.Meta["output_tokens"])
	if err != nil {
		t.Fatalf("output_tokens %q not an int: %v", res.Meta["output_tokens"], err)
	}
	if in <= 0 || out <= 0 {
		t.Fatalf("expected positive token counts, got in=%d out=%d", in, out)
	}
	// The mock echoes the tool + args into the payload.
	body := resolvePayload(t, ctx, res.Payload)
	if !strings.Contains(string(body), "search") {
		t.Fatalf("mock payload missing tool name: %s", body)
	}
}

// NOTE: the live OpenAI-compatible HTTP client (units 39/40/44/45 — canned
// response, base_url swap, retry/backoff) now lives in internal/agent's
// HTTPPlanner and is exercised by that package's tests. The degenerate
// engine.HTTPEngine that used to be tested here was deleted as the never-wired
// duplicate seam (TICKETS T4); the single-client invariant is guarded by
// architest's TestSingleOpenAIChatClient.

// ---------------------------------------------------------------------------
// Unit 41 — cassette record/replay: a hit returns the recorded response with NO
// network and the recorded usage; a miss yields StatusError.
// ---------------------------------------------------------------------------

func TestCassetteReplayHitAndMiss(t *testing.T) {
	ctx := context.Background()

	tool := "lookup"
	args := []byte(`{"id":42}`)
	key := engine.CallKey(tool, args)
	recorded := `{"choices":[{"message":{"content":"recorded"}}]}`

	// Author a cassette file: one entry keyed by CallKey(tool, args) with usage.
	doc := map[string]any{
		"entries": []map[string]any{
			{
				"key":      key,
				"tool":     tool,
				"response": json.RawMessage(recorded),
				"usage": map[string]int{
					"prompt_tokens":     33,
					"completion_tokens": 11,
					"total_tokens":      44,
				},
			},
		},
	}
	raw, err := json.Marshal(doc)
	if err != nil {
		t.Fatalf("marshal cassette: %v", err)
	}
	path := filepath.Join(t.TempDir(), "cassette.json")
	if err := os.WriteFile(path, raw, 0o644); err != nil {
		t.Fatalf("write cassette: %v", err)
	}

	cas, err := engine.LoadCassette(path)
	if err != nil {
		t.Fatalf("LoadCassette: %v", err)
	}
	ce := engine.NewCassetteEngine(cas)

	// HIT: same (tool, args) => the recorded response + usage, offline.
	hit, err := ce.Complete(ctx, inlineCall(tool, string(args)))
	if err != nil {
		t.Fatalf("cassette hit Complete error: %v", err)
	}
	if hit.Status != abi.StatusOK {
		t.Fatalf("hit status = %v, want StatusOK; meta=%v", hit.Status, hit.Meta)
	}
	if hit.Meta["engine"] != "cassette" {
		t.Fatalf("engine meta = %q, want cassette", hit.Meta["engine"])
	}
	if hit.Meta["input_tokens"] != "33" {
		t.Fatalf("input_tokens = %q, want \"33\"", hit.Meta["input_tokens"])
	}
	if hit.Meta["output_tokens"] != "11" {
		t.Fatalf("output_tokens = %q, want \"11\"", hit.Meta["output_tokens"])
	}
	body := resolvePayload(t, ctx, hit.Payload)
	if !strings.Contains(string(body), "recorded") {
		t.Fatalf("hit payload not the recorded response: %s", body)
	}

	// MISS: different args (and thus a different key) => StatusError.
	miss, err := ce.Complete(ctx, inlineCall(tool, `{"id":999}`))
	if err != nil {
		t.Fatalf("cassette miss Complete error: %v", err)
	}
	if miss.Status != abi.StatusError {
		t.Fatalf("miss status = %v, want StatusError", miss.Status)
	}
	if !strings.Contains(miss.Meta["error"], "cassette miss") {
		t.Fatalf("miss error meta = %q, want a cassette-miss message", miss.Meta["error"])
	}
}

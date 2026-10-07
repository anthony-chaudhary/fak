package gateway

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
)

// ctxPagingBody is a benign ~20 KiB tool result: above the 4 KiB "other"-class page-out
// threshold, far below the 1 MiB read/exec thresholds, and clean of every screen.
var ctxPagingBody = strings.Repeat("func helper() int { return 42 }\n", 640)

type ctxPagingRun struct {
	upstreamTool map[string]string // tool_call_id -> content the upstream model received
	pagingHeader string
	metrics      string
}

func runCtxPagingChat(t *testing.T, tools []string, messages []map[string]any, header string) ctxPagingRun {
	t.Helper()
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterResultAdmitter(10, ctxmmu.New())

	var upstreamRaw []byte
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamRaw, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":11,"completion_tokens":5,"total_tokens":16}}`))
	}))
	defer upstream.Close()

	srv, err := New(Config{EngineID: "test", Model: "paging-host:model", BaseURL: upstream.URL, Provider: "openai-compatible"})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	toolDefs := make([]map[string]any, 0, len(tools))
	for _, name := range tools {
		toolDefs = append(toolDefs, map[string]any{"type": "function", "function": map[string]any{
			"name": name, "description": name, "parameters": map[string]any{"type": "object", "properties": map[string]any{}},
		}})
	}
	raw, err := json.Marshal(map[string]any{"model": "client-model", "messages": messages, "tools": toolDefs})
	if err != nil {
		t.Fatal(err)
	}
	req, err := http.NewRequest(http.MethodPost, ts.URL+"/v1/chat/completions", bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	if header != "" {
		req.Header.Set("X-Fak-Ctx-Paging", header)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	respRaw, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200: %s", resp.StatusCode, respRaw)
	}
	var wire struct {
		Messages []struct {
			Role       string          `json:"role"`
			ToolCallID string          `json:"tool_call_id"`
			Content    json.RawMessage `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(upstreamRaw, &wire); err != nil {
		t.Fatalf("decode upstream body: %v: %s", err, upstreamRaw)
	}
	run := ctxPagingRun{upstreamTool: map[string]string{}, pagingHeader: resp.Header.Get("X-Fak-Ctx-Paging")}
	for _, m := range wire.Messages {
		if m.Role != "tool" {
			continue
		}
		var s string
		if err := json.Unmarshal(m.Content, &s); err != nil {
			s = string(m.Content)
		}
		run.upstreamTool[m.ToolCallID] = s
	}
	mresp, err := http.Get(ts.URL + "/metrics")
	if err == nil {
		mraw, _ := io.ReadAll(mresp.Body)
		mresp.Body.Close()
		run.metrics = string(mraw)
	}
	return run
}

func ctxPagingAssistantCall(id, tool string) map[string]any {
	return map[string]any{"role": "assistant", "content": "", "tool_calls": []map[string]any{
		{"id": id, "type": "function", "function": map[string]any{"name": tool, "arguments": `{}`}},
	}}
}

func ctxPagingToolResult(id, name string) map[string]any {
	m := map[string]any{"role": "tool", "tool_call_id": id, "content": ctxPagingBody}
	if name != "" {
		m["name"] = name
	}
	return m
}

func ctxPagingWholeOrFail(t *testing.T, run ctxPagingRun, id string) {
	t.Helper()
	got, ok := run.upstreamTool[id]
	if !ok {
		t.Fatalf("upstream body carried no tool message %q: %v", id, run.upstreamTool)
	}
	if got != ctxPagingBody {
		t.Fatalf("tool result %q reached upstream rewritten (%d bytes, want %d whole): %.200s", id, len(got), len(ctxPagingBody), got)
	}
}

func ctxPagingStubOrFail(t *testing.T, run ctxPagingRun, id string) {
	t.Helper()
	var stub map[string]any
	if err := json.Unmarshal([]byte(run.upstreamTool[id]), &stub); err != nil {
		t.Fatalf("tool result %q was not paged to a JSON stub: %.200s", id, run.upstreamTool[id])
	}
	if paged, _ := stub["_paged"].(bool); !paged {
		t.Fatalf("tool result %q stub lacks _paged=true: %v", id, stub)
	}
}

// OpenCode Code Mode (read/shell/glob/execute, no restore tool) answering the current
// turn with a nameless 20 KiB result: neither admission pass may stub it.
func TestCtxPagingOpenCodeTrailingNamelessResultForwardedWhole(t *testing.T) {
	run := runCtxPagingChat(t, []string{"read", "shell", "glob", "execute", "skill"}, []map[string]any{
		{"role": "user", "content": "read the file"},
		ctxPagingAssistantCall("call_1", "read"),
		ctxPagingToolResult("call_1", ""),
	}, "")
	ctxPagingWholeOrFail(t, run, "call_1")
	if !strings.HasPrefix(run.pagingHeader, "off;") || !strings.Contains(run.pagingHeader, "harness_cannot_restore") {
		t.Fatalf("X-Fak-Ctx-Paging = %q, want off;reason=harness_cannot_restore", run.pagingHeader)
	}
}

// An OLDER "other"-class result (a later assistant turn follows it) still is not paged
// for a client with no restore tool: the stub would be an unrecoverable hole.
func TestCtxPagingOpenCodeOlderResultNotPagedWithoutRestoreTool(t *testing.T) {
	run := runCtxPagingChat(t, []string{"read", "shell", "glob", "execute", "skill"}, []map[string]any{
		{"role": "user", "content": "run it"},
		ctxPagingAssistantCall("call_1", "execute"),
		ctxPagingToolResult("call_1", ""),
		ctxPagingAssistantCall("call_2", "glob"),
		map[string]any{"role": "tool", "tool_call_id": "call_2", "content": "a.go\nb.go"},
	}, "")
	ctxPagingWholeOrFail(t, run, "call_1")
	if !strings.Contains(run.metrics, `fak_gateway_ctx_paging_suppressed_total{reason="harness_cannot_restore"}`) ||
		strings.Contains(run.metrics, `fak_gateway_ctx_paging_suppressed_total{reason="harness_cannot_restore"} 0`+"\n") {
		t.Fatalf("/metrics did not count a harness_cannot_restore suppression")
	}
}

// A client that declares the restore tool keeps paging for history, but never for the
// result the current turn just asked for.
func TestCtxPagingRestoreCapableClientPagesOnlyOlderResults(t *testing.T) {
	run := runCtxPagingChat(t, []string{"fetch_doc", "fak_context_restore"}, []map[string]any{
		{"role": "user", "content": "fetch both"},
		ctxPagingAssistantCall("call_old", "fetch_doc"),
		ctxPagingToolResult("call_old", "fetch_doc"),
		ctxPagingAssistantCall("call_new", "fetch_doc"),
		ctxPagingToolResult("call_new", "fetch_doc"),
	}, "")
	ctxPagingStubOrFail(t, run, "call_old")
	ctxPagingWholeOrFail(t, run, "call_new")
	if !strings.HasPrefix(run.pagingHeader, "auto-on") {
		t.Fatalf("X-Fak-Ctx-Paging = %q, want auto-on", run.pagingHeader)
	}
}

// OpenCode spells the fak MCP restore tool fak_fak_context_restore; it counts as capable.
func TestCtxPagingOpenCodeRestoreSpellingIsCapable(t *testing.T) {
	run := runCtxPagingChat(t, []string{"read", "fetch_doc", "fak_fak_context_restore"}, []map[string]any{
		{"role": "user", "content": "fetch"},
		ctxPagingAssistantCall("call_old", "fetch_doc"),
		ctxPagingToolResult("call_old", "fetch_doc"),
		ctxPagingAssistantCall("call_new", "read"),
		map[string]any{"role": "tool", "tool_call_id": "call_new", "content": "ok"},
	}, "")
	ctxPagingStubOrFail(t, run, "call_old")
}

// X-Fak-Ctx-Paging: off overrides a restore-capable request.
func TestCtxPagingHeaderOffOverridesCapableClient(t *testing.T) {
	run := runCtxPagingChat(t, []string{"fetch_doc", "fak_context_restore"}, []map[string]any{
		{"role": "user", "content": "fetch both"},
		ctxPagingAssistantCall("call_old", "fetch_doc"),
		ctxPagingToolResult("call_old", "fetch_doc"),
		ctxPagingAssistantCall("call_new", "fetch_doc"),
		ctxPagingToolResult("call_new", "fetch_doc"),
	}, "off")
	ctxPagingWholeOrFail(t, run, "call_old")
	ctxPagingWholeOrFail(t, run, "call_new")
	if !strings.HasPrefix(run.pagingHeader, "off;") {
		t.Fatalf("X-Fak-Ctx-Paging = %q, want off;reason=...", run.pagingHeader)
	}
}

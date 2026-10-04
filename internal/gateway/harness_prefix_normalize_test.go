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
	"github.com/anthony-chaudhary/fak/internal/agent"
)

// harness_prefix_normalize_test.go — parent and subagent requests from a third-party
// harness must render to the SAME system+tools head so radix prefix-KV reuse fires.
// The witnesses read the body the gateway actually forwards to an OpenAI-compatible
// upstream (the strix serving shape: fak proxying to llama-server), for both the
// Anthropic /v1/messages wire and the /v1/chat/completions wire.

type forwardedChat struct {
	Messages []capturedMessage `json:"messages"`
	Tools    json.RawMessage   `json:"tools"`
}

// captureForwardedChats records every request body an OpenAI-compatible mock upstream
// receives and answers each with a trivial no-tool completion.
func captureForwardedChats(t *testing.T, got *[]forwardedChat) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read upstream body: %v", err)
			return
		}
		var fc forwardedChat
		if err := json.Unmarshal(raw, &fc); err != nil {
			t.Errorf("decode upstream request: %v", err)
			return
		}
		*got = append(*got, fc)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":11,"completion_tokens":1,"total_tokens":12}}`))
	}))
}

func newPrefixProxyGateway(t *testing.T, upstreamURL string) *httptest.Server {
	t.Helper()
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})
	srv, err := New(Config{EngineID: "test", Model: "test-model", BaseURL: upstreamURL, Provider: "openai"})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(ts.Close)
	return ts
}

func postPrefixJSON(t *testing.T, url string, body any) {
	t.Helper()
	raw, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	resp, err := http.Post(url, "application/json", bytes.NewReader(raw))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d: %s", resp.StatusCode, b)
	}
}

// renderedHead is the forwarded request's system+tools section as one text: the
// order a chat template lays it out (tools, then the standing system prompt).
func renderedHead(fc forwardedChat) string {
	var sys strings.Builder
	for _, m := range fc.Messages {
		if m.Role != agent.RoleSystem && m.Role != "developer" {
			break
		}
		sys.WriteString(m.Content)
		sys.WriteString("\n")
	}
	return string(fc.Tools) + "\n" + sys.String()
}

// renderedFull is renderedHead followed by the conversation turns.
func renderedFull(fc forwardedChat) string {
	var b strings.Builder
	b.WriteString(renderedHead(fc))
	for _, m := range fc.Messages {
		if m.Role == agent.RoleSystem || m.Role == "developer" {
			continue
		}
		b.WriteString(m.Role + ": " + m.Content + "\n")
	}
	return b.String()
}

// assertSharedHead is the closed contract: the parent's and child's rendered text
// agree through the end of the system+tools section, the heads are non-empty, tools
// arrive sorted by name, and the billing header never reaches the upstream.
func assertSharedHead(t *testing.T, got []forwardedChat, wantTools []string) {
	t.Helper()
	if len(got) != 2 {
		t.Fatalf("upstream saw %d requests, want 2", len(got))
	}
	parent, child := got[0], got[1]
	head := renderedHead(parent)
	if head != renderedHead(child) {
		t.Fatalf("system+tools heads differ:\nparent=%q\nchild =%q", head, renderedHead(child))
	}
	if cp := commonPrefixLen(renderedFull(parent), renderedFull(child)); cp < len(head) {
		t.Fatalf("rendered common prefix = %d bytes, want >= system+tools head %d", cp, len(head))
	}
	var tools []struct {
		Function struct {
			Name string `json:"name"`
		} `json:"function"`
	}
	if err := json.Unmarshal(parent.Tools, &tools); err != nil {
		t.Fatalf("decode forwarded tools: %v", err)
	}
	var names []string
	for _, tl := range tools {
		names = append(names, tl.Function.Name)
	}
	if strings.Join(names, ",") != strings.Join(wantTools, ",") {
		t.Fatalf("forwarded tool order = %v, want %v", names, wantTools)
	}
	for _, fc := range got {
		for _, m := range fc.Messages {
			if strings.Contains(m.Content, agent.AnthropicBillingHeaderPrefix) {
				t.Fatalf("billing header forwarded upstream in %s message", m.Role)
			}
		}
	}
}

// assertVolatileRelocated: the volatile date line left the system prompt but was not
// dropped — it rides the first user turn.
func assertVolatileRelocated(t *testing.T, fc forwardedChat, line string) {
	t.Helper()
	for _, m := range fc.Messages {
		if m.Role == agent.RoleUser {
			if !strings.Contains(m.Content, line) {
				t.Fatalf("first user turn %q lost the relocated %q", m.Content, line)
			}
			return
		}
		if strings.Contains(m.Content, line) {
			t.Fatalf("%q still inside the %s head", line, m.Role)
		}
	}
	t.Fatal("no user turn forwarded")
}

func anthropicTool(name string) map[string]any {
	return map[string]any{"name": name, "description": name + " tool", "input_schema": map[string]any{"type": "object"}}
}

// fak-test:runtime slow est=1s
func TestMessagesForwardedPrefixSharedAcrossParentAndSubagent(t *testing.T) {
	var got []forwardedChat
	upstream := captureForwardedChats(t, &got)
	defer upstream.Close()
	ts := newPrefixProxyGateway(t, upstream.URL)

	request := func(billing, date, user string, tools ...string) map[string]any {
		var defs []map[string]any
		for _, n := range tools {
			defs = append(defs, anthropicTool(n))
		}
		return map[string]any{
			"model": "test-model", "max_tokens": 64,
			"system": []map[string]any{
				{"type": "text", "text": billing},
				{"type": "text", "text": "You are Claude Code, a coding agent.\nToday's date: " + date},
			},
			"messages": []map[string]any{{"role": "user", "content": user}},
			"tools":    defs,
		}
	}
	postPrefixJSON(t, ts.URL+"/v1/messages", request("x-anthropic-billing-header: cc_version=2.1.37.a1b; cc_entrypoint=cli; cch=0f3e2;", "2026-10-04", "refactor the parser", "Read", "Bash", "Edit"))
	postPrefixJSON(t, ts.URL+"/v1/messages", request("x-anthropic-billing-header: cc_version=2.1.37.c3d; cc_entrypoint=cli; cch=91b0a;", "2026-10-05", "find all callers of Parse", "Edit", "Read", "Bash"))

	assertSharedHead(t, got, []string{"Bash", "Edit", "Read"})
	assertVolatileRelocated(t, got[0], "Today's date: 2026-10-04")
}

// fak-test:runtime slow est=1s
func TestChatCompletionsForwardedPrefixSharedAcrossParentAndSubagent(t *testing.T) {
	var got []forwardedChat
	upstream := captureForwardedChats(t, &got)
	defer upstream.Close()
	ts := newPrefixProxyGateway(t, upstream.URL)

	tool := func(name string) map[string]any {
		return map[string]any{"type": "function", "function": map[string]any{"name": name, "description": name + " tool", "parameters": map[string]any{"type": "object"}}}
	}
	request := func(date, user string, tools ...string) map[string]any {
		var defs []map[string]any
		for _, n := range tools {
			defs = append(defs, tool(n))
		}
		return map[string]any{
			"model": "test-model",
			"messages": []map[string]any{
				{"role": "system", "content": "You are OpenCode.\nCurrent date: " + date},
				{"role": "user", "content": user},
			},
			"tools":       defs,
			"tool_choice": map[string]any{"type": "function", "function": map[string]any{"name": "write"}},
		}
	}
	postPrefixJSON(t, ts.URL+"/v1/chat/completions", request("2026-10-04", "parent task", "write", "bash", "read"))
	postPrefixJSON(t, ts.URL+"/v1/chat/completions", request("2026-10-05", "child task", "read", "write", "bash"))

	assertSharedHead(t, got, []string{"bash", "read", "write"})
	assertVolatileRelocated(t, got[1], "Current date: 2026-10-05")
}

// fak-test:runtime fast est=10ms lane=default
func TestNormalizeHarnessPrefixIdentityAndNoMutation(t *testing.T) {
	msgs := []agent.Message{
		{Role: agent.RoleSystem, Content: "x-anthropic-billing-header: cc_version=1; cch=2;\nStable rules.\nToday's date: 2026-10-04"},
		{Role: agent.RoleUser, Content: "hi"},
		{Role: agent.RoleAssistant, Content: "hello"},
		{Role: agent.RoleUser, Content: "next"},
	}
	orig := append([]agent.Message(nil), msgs...)
	out, _ := normalizeHarnessPrefix(msgs, nil)
	if out[0].Content != "Stable rules." {
		t.Fatalf("system = %q, want billing header and volatile line removed", out[0].Content)
	}
	if out[1].Content != "hi\n\nToday's date: 2026-10-04" || out[3].Content != "next" {
		t.Fatalf("volatile must anchor on the FIRST user turn: %q / %q", out[1].Content, out[3].Content)
	}
	for i := range msgs {
		if msgs[i].Content != orig[i].Content || msgs[i].Role != orig[i].Role {
			t.Fatalf("input message %d mutated", i)
		}
	}

	// No user turn to carry it: volatile metadata stays put (never dropped).
	lone := []agent.Message{{Role: agent.RoleSystem, Content: "Rules.\nToday's date: 2026-10-04"}}
	if got, _ := normalizeHarnessPrefix(lone, nil); got[0].Content != lone[0].Content {
		t.Fatalf("system without a user turn rewritten: %q", got[0].Content)
	}
	// Already-canonical input passes through unchanged.
	plain := []agent.Message{{Role: agent.RoleSystem, Content: "Rules."}, {Role: agent.RoleUser, Content: "u"}}
	if got, _ := normalizeHarnessPrefix(plain, nil); got[0].Content != plain[0].Content || got[1].Content != plain[1].Content {
		t.Fatal("canonical transcript changed")
	}
}

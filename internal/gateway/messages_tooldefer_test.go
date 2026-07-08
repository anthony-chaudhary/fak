package gateway

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
)

// messages_tooldefer_test.go is the #3232 witness: the deterministic, cache-safe,
// fail-safe cold-tool deferral transform. It proves cold defs get defer_loading:true,
// the hot core stays eager, one tool_search_tool is injected, the non-tools body bytes
// are byte-identical after the splice AND the body re-decodes, an already-deferred body
// is a no-op, identity holds on ambiguity, and the transform is deterministic.

// a Claude-Code-shaped body: hot built-ins (Read/Bash) + cold custom tools
// (mcp__fak__*, mcp__dos__*), the last carrying the cache_control anchor.
const deferBody = `{"model":"claude-x","system":"SYS-PROMPT","messages":[{"role":"user","content":"hello"}],` +
	`"tools":[` +
	`{"name":"Read","description":"read a file","input_schema":{"type":"object"}},` +
	`{"name":"Bash","description":"run a command","input_schema":{"type":"object"}},` +
	`{"name":"mcp__fak__fak_syscall","description":"adjudicate and execute","input_schema":{"type":"object"}},` +
	`{"name":"mcp__dos__dos_verify","description":"did it ship","input_schema":{"type":"object"},"cache_control":{"type":"ephemeral"}}` +
	`]}`

func anthropicDecode(b []byte) error { _, err := agent.DecodeAnthropicMessagesRequest(b); return err }

func toolsOf(t *testing.T, raw []byte) []map[string]json.RawMessage {
	t.Helper()
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(raw, &obj); err != nil {
		t.Fatalf("unmarshal body: %v", err)
	}
	var elems []map[string]json.RawMessage
	if err := json.Unmarshal(obj["tools"], &elems); err != nil {
		t.Fatalf("unmarshal tools: %v", err)
	}
	return elems
}

func hasDeferLoading(m map[string]json.RawMessage) bool {
	v, ok := m["defer_loading"]
	return ok && string(v) == "true"
}

func TestDeferColdToolsMarksColdEagerHotInjectsSearch(t *testing.T) {
	res := deferColdToolsInBody([]byte(deferBody), defaultHotToolSet, anthropicDecode)
	if !res.Changed {
		t.Fatalf("expected a change, got identity (reason=%q)", res.Reason)
	}
	if res.ColdCount != 2 {
		t.Fatalf("ColdCount=%d, want 2 (the two mcp__ tools)", res.ColdCount)
	}

	elems := toolsOf(t, res.Body)
	if len(elems) != 5 {
		t.Fatalf("tools len=%d, want 5 (4 original + tool_search_tool)", len(elems))
	}

	byName := map[string]map[string]json.RawMessage{}
	for _, m := range elems {
		byName[rawStringField(m, "name")] = m
	}
	// Hot built-ins stay eager (no defer_loading).
	for _, hot := range []string{"Read", "Bash"} {
		if hasDeferLoading(byName[hot]) {
			t.Errorf("hot tool %q was deferred; it must stay eager", hot)
		}
	}
	// Cold custom tools are deferred.
	for _, cold := range []string{"mcp__fak__fak_syscall", "mcp__dos__dos_verify"} {
		if !hasDeferLoading(byName[cold]) {
			t.Errorf("cold tool %q missing defer_loading:true", cold)
		}
	}
	// Exactly one tool_search_tool injected, carrying the cache_control anchor
	// (the client's last tool was cached).
	st, ok := byName[toolSearchToolName]
	if !ok {
		t.Fatalf("no tool_search_tool injected; got %v", keysOf(byName))
	}
	if got := rawStringField(st, "type"); got != toolSearchToolType {
		t.Errorf("tool_search_tool type=%q, want %q", got, toolSearchToolType)
	}
	if _, cc := st["cache_control"]; !cc {
		t.Errorf("tool_search_tool did not carry the cache_control anchor")
	}
}

func TestDeferColdToolsPreservesNonToolsBytesAndDecodes(t *testing.T) {
	raw := []byte(deferBody)
	res := deferColdToolsInBody(raw, defaultHotToolSet, anthropicDecode)
	if !res.Changed {
		t.Fatalf("identity: %q", res.Reason)
	}
	// The system + messages bytes are spliced through verbatim.
	for _, verbatim := range []string{`"system":"SYS-PROMPT"`, `"messages":[{"role":"user","content":"hello"}]`, `"model":"claude-x"`} {
		if !bytes.Contains(res.Body, []byte(verbatim)) {
			t.Errorf("non-tools fragment %q not preserved verbatim in output", verbatim)
		}
	}
	// And it re-decodes as a valid Anthropic request with system intact.
	req, err := agent.DecodeAnthropicMessagesRequest(res.Body)
	if err != nil {
		t.Fatalf("deferred body did not re-decode: %v", err)
	}
	if req.System != "SYS-PROMPT" {
		t.Fatalf("system changed after defer: %q", req.System)
	}
}

func TestDeferColdToolsIsDeterministic(t *testing.T) {
	raw := []byte(deferBody)
	a := deferColdToolsInBody(raw, defaultHotToolSet, anthropicDecode)
	b := deferColdToolsInBody(raw, defaultHotToolSet, anthropicDecode)
	if !a.Changed || !b.Changed {
		t.Fatalf("expected both to change")
	}
	if !bytes.Equal(a.Body, b.Body) {
		t.Fatalf("non-deterministic output — cache prefix would not be stable turn-over-turn")
	}
}

func TestDeferColdToolsIdempotentStandDown(t *testing.T) {
	// A body already carrying a tool_search_tool (the Claude Code ENABLE_TOOL_SEARCH case).
	already := `{"model":"claude-x","messages":[{"role":"user","content":"hi"}],"tools":[` +
		`{"name":"mcp__fak__x","description":"d","input_schema":{"type":"object"}},` +
		`{"type":"` + toolSearchToolType + `","name":"` + toolSearchToolName + `"}]}`
	res := deferColdToolsInBody([]byte(already), defaultHotToolSet, anthropicDecode)
	if res.Changed {
		t.Fatalf("expected no-op on an already-deferred body, got a change")
	}
	if res.Reason != "already_deferred" {
		t.Fatalf("reason=%q, want already_deferred", res.Reason)
	}

	// A body whose def already has defer_loading is also a stand-down.
	withDefer := `{"messages":[],"tools":[{"name":"mcp__fak__x","description":"d","input_schema":{"type":"object"},"defer_loading":true}]}`
	if r := deferColdToolsInBody([]byte(withDefer), defaultHotToolSet, anthropicDecode); r.Changed {
		t.Fatalf("expected no-op when a def already carries defer_loading")
	}
}

func TestDeferColdToolsIdentityOnAmbiguity(t *testing.T) {
	cases := []struct {
		name string
		raw  string
		want string
	}{
		{"not json", `not a json object`, "not_json_object"},
		{"no tools", `{"messages":[{"role":"user","content":"hi"}]}`, "no_tools"},
		{"empty tools", `{"tools":[]}`, "no_tools"},
		{"only hot tools", `{"tools":[{"name":"Read","description":"r","input_schema":{"type":"object"}},{"name":"Bash","description":"b","input_schema":{"type":"object"}}]}`, "no_cold_tools"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			res := deferColdToolsInBody([]byte(tc.raw), defaultHotToolSet, anthropicDecode)
			if res.Changed {
				t.Fatalf("expected identity, got a change")
			}
			if res.Reason != tc.want {
				t.Fatalf("reason=%q, want %q", res.Reason, tc.want)
			}
			if res.Body != nil {
				t.Fatalf("identity must carry no Body")
			}
		})
	}
}

func TestObserveToolDeferMetric(t *testing.T) {
	m := &gatewayMetrics{}
	m.observeToolDefer(3, true)
	m.observeToolDefer(2, true)
	m.observeToolDefer(5, false) // a stand-down records nothing
	m.observeToolDefer(0, true)  // zero cold records nothing
	turns, cold := m.toolDeferSnapshot()
	if turns != 2 || cold != 5 {
		t.Fatalf("defer snapshot turns=%d cold=%d, want 2/5", turns, cold)
	}
}

func TestMaybeDeferColdToolsDefaultOff(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})
	srv, err := New(Config{EngineID: "test", Model: "test-model", VDSO: true}) // DeferColdTools unset
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	req := &agent.AnthropicMessagesRequest{Model: "test-model", Raw: []byte(deferBody)}
	if n := srv.maybeDeferColdTools(req, "trace"); n != 0 {
		t.Fatalf("default-off server deferred %d tools; want 0", n)
	}
	if !bytes.Equal(req.Raw, []byte(deferBody)) {
		t.Fatalf("default-off server mutated req.Raw")
	}
}

func keysOf(m map[string]map[string]json.RawMessage) string {
	var b strings.Builder
	for k := range m {
		b.WriteString(k)
		b.WriteByte(' ')
	}
	return b.String()
}

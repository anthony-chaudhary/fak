package gateway

import (
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// ctxfootprint_test.go is the #3233 witness: it drives a captured Claude-Code
// inbound body (a system prompt + built-in tools + mcp__fak__* and mcp__dos__*
// tools + a folded-system duplicate) through observeCtxFootprint and proves
// (a) Floor == System + Tools, (b) the built-in vs MCP split reconstructs from
// the wire-witnessed mcp__ prefix, and (c) the folded RoleSystem duplicate is
// NOT double-counted.

func fpTool(name, desc, params string) agent.ToolDef {
	return agent.ToolDef{Type: "function", Function: agent.ToolDefFunction{
		Name: name, Description: desc, Parameters: json.RawMessage(params)}}
}

func TestObserveCtxFootprintSplitsAndDeFolds(t *testing.T) {
	srv := newExposeServer(t)

	const sys = "You are a coding agent with a nontrivial system prompt."
	const userMsg = "compact my context please"
	readParams := `{"type":"object"}`
	writeParams := `{"type":"object"}`
	fakParams := `{"type":"object","properties":{"tool":{"type":"string"}}}`
	dosParams := `{"type":"object"}`

	req := &agent.AnthropicMessagesRequest{
		System: sys,
		Tools: []agent.ToolDef{
			fpTool("Read", "read a file from disk", readParams),                              // built-in
			fpTool("Write", "write a file to disk", writeParams),                             // built-in
			fpTool("mcp__fak__fak_syscall", "adjudicate and execute a tool call", fakParams), // MCP (fak)
			fpTool("mcp__dos__dos_verify", "did a plan phase actually ship", dosParams),      // MCP (dos)
		},
		Messages: []agent.Message{
			{Role: agent.RoleSystem, Content: sys}, // the folded duplicate the decoder prepends
			{Role: "user", Content: userMsg},
		},
	}

	// Test premise: the NAIVE footprint double-counts the system (System bucket AND
	// the folded RoleSystem message in History) — the gap this issue closes.
	naive := agent.RequestFootprint(req)
	if naive.History.Bytes != len(sys) {
		t.Fatalf("premise broken: naive footprint History=%d, expected the folded system dup (%d)", naive.History.Bytes, len(sys))
	}

	srv.observeCtxFootprint("trace-fp", req)
	fp := srv.CtxValueReportFor("trace-fp").Footprint
	if fp == nil {
		t.Fatal("fak_context_value report carried no footprint after observeCtxFootprint")
	}

	// (a) Floor == System + Tools.
	if fp.FloorBytes != fp.SystemBytes+fp.ToolsBytes {
		t.Fatalf("Floor %d != System %d + Tools %d", fp.FloorBytes, fp.SystemBytes, fp.ToolsBytes)
	}

	// (b) built-in vs MCP split, exact.
	if fp.ToolsBytes != fp.BuiltinToolBytes+fp.MCPToolBytes {
		t.Fatalf("Tools %d != builtin %d + mcp %d", fp.ToolsBytes, fp.BuiltinToolBytes, fp.MCPToolBytes)
	}
	if fp.BuiltinToolCount != 2 || fp.MCPToolCount != 2 {
		t.Fatalf("tool counts: builtin=%d mcp=%d, want 2/2", fp.BuiltinToolCount, fp.MCPToolCount)
	}
	tb := func(name, desc, params string) int { return len(name) + len(desc) + len(params) }
	wantMCP := tb("mcp__fak__fak_syscall", "adjudicate and execute a tool call", fakParams) +
		tb("mcp__dos__dos_verify", "did a plan phase actually ship", dosParams)
	if fp.MCPToolBytes != wantMCP {
		t.Fatalf("MCP sub-bucket %d != sum of mcp__ tool bytes %d", fp.MCPToolBytes, wantMCP)
	}
	wantBuiltin := tb("Read", "read a file from disk", readParams) + tb("Write", "write a file to disk", writeParams)
	if fp.BuiltinToolBytes != wantBuiltin {
		t.Fatalf("built-in sub-bucket %d != sum of non-mcp tool bytes %d", fp.BuiltinToolBytes, wantBuiltin)
	}

	// (c) No system double-count: System counted once, the folded duplicate is
	// gone from History (only the user turn remains, as Tail).
	if fp.SystemBytes != len(sys) {
		t.Fatalf("System %d != len(system) %d (counted twice?)", fp.SystemBytes, len(sys))
	}
	if fp.HistoryBytes != 0 {
		t.Fatalf("History %d != 0 — folded system duplicate not de-folded (double count)", fp.HistoryBytes)
	}
	if fp.TailBytes != len(userMsg) {
		t.Fatalf("Tail %d != len(user msg) %d", fp.TailBytes, len(userMsg))
	}
	if fp.Provenance != agent.FootprintProvenance {
		t.Fatalf("provenance %q, want %q", fp.Provenance, agent.FootprintProvenance)
	}

	t.Logf("live floor: %d tokens (%d B) = system %d B + built-in tools %d B (%d) + MCP tools %d B (%d); no double-count (naive history would be %d B)",
		fp.FloorTokens, fp.FloorBytes, fp.SystemBytes, fp.BuiltinToolBytes, fp.BuiltinToolCount, fp.MCPToolBytes, fp.MCPToolCount, naive.History.Bytes)
}

// TestObserveCtxFootprintNoSystemFold covers a request with no leading folded
// system message (e.g. System empty): de-fold is a no-op and nothing is dropped.
func TestObserveCtxFootprintNoSystemFold(t *testing.T) {
	srv := newExposeServer(t)
	req := &agent.AnthropicMessagesRequest{
		Tools:    []agent.ToolDef{fpTool("Read", "read", `{"type":"object"}`)},
		Messages: []agent.Message{{Role: "user", Content: "hi"}},
	}
	srv.observeCtxFootprint("trace-nofold", req)
	fp := srv.CtxValueReportFor("trace-nofold").Footprint
	if fp == nil {
		t.Fatal("no footprint surfaced")
	}
	if fp.SystemBytes != 0 {
		t.Fatalf("System %d, want 0 (no system prompt)", fp.SystemBytes)
	}
	if fp.TailBytes != len("hi") {
		t.Fatalf("Tail %d, want %d", fp.TailBytes, len("hi"))
	}
}

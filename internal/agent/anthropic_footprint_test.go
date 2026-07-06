package agent

import (
	"encoding/json"
	"testing"
)

// sampleClaudeCodeRequest builds a request shaped like what Claude Code sends: a
// non-trivial system prompt, several tool defs, a multi-turn history, and a tail user
// turn. Numbers are arbitrary but the SHAPE (system + tools floor, growing history,
// one volatile tail) is the real thing the footprint audits.
func sampleClaudeCodeRequest() *AnthropicMessagesRequest {
	return &AnthropicMessagesRequest{
		Model:  "claude-opus-4-8",
		System: "You are Claude Code, Anthropic's official CLI. Be concise and act directly.",
		Tools: []ToolDef{
			{Type: "function", Function: ToolDefFunction{
				Name: "Read", Description: "Read a file from the local filesystem.",
				Parameters: json.RawMessage(`{"type":"object","properties":{"file_path":{"type":"string"}}}`),
			}},
			{Type: "function", Function: ToolDefFunction{
				Name: "Bash", Description: "Execute a bash command and return its output.",
				Parameters: json.RawMessage(`{"type":"object","properties":{"command":{"type":"string"},"timeout":{"type":"number"}}}`),
			}},
			{Type: "function", Function: ToolDefFunction{
				Name: "Grep", Description: "Search file contents with ripgrep.",
				Parameters: json.RawMessage(`{"type":"object","properties":{"pattern":{"type":"string"}}}`),
			}},
		},
		Messages: []Message{
			{Role: RoleUser, Content: "Please audit the token footprint of this repo."},
			{Role: RoleAssistant, Content: "", ToolCalls: []ToolCall{
				{ID: "t1", Type: "function", Function: Func{Name: "Grep", Arguments: `{"pattern":"token"}`}},
			}},
			{Role: RoleUser, Content: "Here are 40 matching files across the gateway package."},
			{Role: RoleUser, Content: "Now summarize where the tokens go on the latest turn."},
		},
	}
}

func TestRequestFootprintReconcilesWithEstimate(t *testing.T) {
	req := sampleClaudeCodeRequest()
	fp := RequestFootprint(req)

	// The whole point: the footprint is a faithful PARTITION of the scalar estimate.
	if got, want := fp.Total.Tokens, EstimateAnthropicTokens(req); got != want {
		t.Fatalf("Total.Tokens=%d, EstimateAnthropicTokens=%d — footprint is not a faithful partition of the estimate", got, want)
	}

	// Exact byte quantities must sum to the total (Bytes is the additive quantity).
	if sum := fp.System.Bytes + fp.Tools.Bytes + fp.History.Bytes + fp.Tail.Bytes; sum != fp.Total.Bytes {
		t.Fatalf("bucket bytes sum to %d, Total.Bytes=%d", sum, fp.Total.Bytes)
	}

	// Floor is exactly system + tools — the fixed per-call tax / clean minimal baseline.
	if fp.Floor.Bytes != fp.System.Bytes+fp.Tools.Bytes {
		t.Fatalf("Floor.Bytes=%d, want system(%d)+tools(%d)=%d",
			fp.Floor.Bytes, fp.System.Bytes, fp.Tools.Bytes, fp.System.Bytes+fp.Tools.Bytes)
	}
}

func TestRequestFootprintPerToolSumsToToolsBucket(t *testing.T) {
	req := sampleClaudeCodeRequest()
	fp := RequestFootprint(req)

	if fp.ToolCount != len(req.Tools) {
		t.Fatalf("ToolCount=%d, want %d", fp.ToolCount, len(req.Tools))
	}
	if len(fp.PerTool) != len(req.Tools) {
		t.Fatalf("len(PerTool)=%d, want %d", len(fp.PerTool), len(req.Tools))
	}
	sum := 0
	for i, pt := range fp.PerTool {
		if pt.Name != req.Tools[i].Function.Name {
			t.Errorf("PerTool[%d].Name=%q, want %q", i, pt.Name, req.Tools[i].Function.Name)
		}
		if pt.Bytes != toolBytes(req.Tools[i]) {
			t.Errorf("PerTool[%d].Bytes=%d, want %d", i, pt.Bytes, toolBytes(req.Tools[i]))
		}
		sum += pt.Bytes
	}
	// The per-tool footprints must exactly reconstruct the tools bucket (#2924's gate
	// sums them into a tool_schema_footprint; a drift here would double-count).
	if sum != fp.Tools.Bytes {
		t.Fatalf("per-tool bytes sum to %d, Tools.Bytes=%d", sum, fp.Tools.Bytes)
	}
}

func TestRequestFootprintHistoryTailSplit(t *testing.T) {
	req := sampleClaudeCodeRequest()
	fp := RequestFootprint(req)

	// Tail is the LAST message only; history is everything before it.
	n := len(req.Messages)
	if wantTail := messageBytes(req.Messages[n-1]); fp.Tail.Bytes != wantTail {
		t.Fatalf("Tail.Bytes=%d, want last-message bytes %d", fp.Tail.Bytes, wantTail)
	}
	wantHistory := 0
	for i := 0; i < n-1; i++ {
		wantHistory += messageBytes(req.Messages[i])
	}
	if fp.History.Bytes != wantHistory {
		t.Fatalf("History.Bytes=%d, want %d", fp.History.Bytes, wantHistory)
	}
	if fp.MessageCount != n {
		t.Fatalf("MessageCount=%d, want %d", fp.MessageCount, n)
	}
}

func TestRequestFootprintProvenanceAlwaysEstimated(t *testing.T) {
	if fp := RequestFootprint(nil); fp.Provenance != FootprintProvenance {
		t.Fatalf("nil req provenance=%q, want %q", fp.Provenance, FootprintProvenance)
	}
	if fp := RequestFootprint(sampleClaudeCodeRequest()); fp.Provenance != FootprintProvenance {
		t.Fatalf("provenance=%q, want %q", fp.Provenance, FootprintProvenance)
	}
}

func TestRequestFootprintEdgeCases(t *testing.T) {
	// Nil and empty requests must be zero, not a panic — a caller renders them unguarded.
	if fp := RequestFootprint(nil); fp.Total.Bytes != 0 || fp.Total.Tokens != 0 {
		t.Fatalf("nil req not zero: %+v", fp.Total)
	}
	if fp := RequestFootprint(&AnthropicMessagesRequest{}); fp.Total.Bytes != 0 {
		t.Fatalf("empty req not zero: %+v", fp.Total)
	}

	// A single-message request is all Tail, no History (there is no earlier turn).
	single := &AnthropicMessagesRequest{
		System:   "sys",
		Messages: []Message{{Role: RoleUser, Content: "just one turn"}},
	}
	fp := RequestFootprint(single)
	if fp.History.Bytes != 0 {
		t.Fatalf("single-message History.Bytes=%d, want 0", fp.History.Bytes)
	}
	if fp.Tail.Bytes != len("just one turn") {
		t.Fatalf("single-message Tail.Bytes=%d, want %d", fp.Tail.Bytes, len("just one turn"))
	}
	// With no tools the floor is just the system prompt.
	if fp.Floor.Bytes != len("sys") {
		t.Fatalf("no-tools Floor.Bytes=%d, want %d (system only)", fp.Floor.Bytes, len("sys"))
	}
}

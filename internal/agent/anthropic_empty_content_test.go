package agent

import (
	"encoding/json"
	"strings"
	"testing"
)

// TestRepairEmptyToolResultContent is the #3118 passthrough-side repro for the OUTBOUND
// empty-content gate: a tool_result whose content array is empty (`[]`) — the residual shape after
// the tool_reference sanitizer strips every block, or a genuinely empty source result — must be
// backfilled with a non-empty wire-valid text block, never forwarded empty (an empty array is a 400).
func TestRepairEmptyToolResultContent(t *testing.T) {
	body := `{"model":"claude-opus-4-8","messages":[` +
		`{"role":"user","content":[{"type":"tool_result","tool_use_id":"t","content":[]}]}` +
		`]}`
	out, outcome := RepairEmptyToolResultContent([]byte(body))
	t.Logf("reason=%q repaired=%d out=%s", outcome.Reason, outcome.Repaired, out)
	if outcome.Reason != EmptyContentReasonNone || outcome.Repaired != 1 {
		t.Fatalf("empty content not repaired: reason=%q repaired=%d", outcome.Reason, outcome.Repaired)
	}
	if strings.Contains(string(out), `"content":[]`) {
		t.Fatalf("empty content array survived the gate:\n%s", out)
	}
	if !strings.Contains(string(out), emptyToolResultText) {
		t.Fatalf("placeholder text missing from repaired body:\n%s", out)
	}
	// The repaired body must still be valid JSON, and the tool_result content must now be a
	// one-element text array.
	var probe struct {
		Messages []struct {
			Content []struct {
				Type    string          `json:"type"`
				Content json.RawMessage `json:"content"`
			} `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(out, &probe); err != nil {
		t.Fatalf("repaired body is not valid JSON: %v\n%s", err, out)
	}
	inner := probe.Messages[0].Content[0].Content
	var blocks []map[string]any
	if err := json.Unmarshal(inner, &blocks); err != nil || len(blocks) != 1 || blocks[0]["type"] != "text" {
		t.Fatalf("repaired content is not a single text block: %s", inner)
	}
}

// TestRepairEmptyToolResultContent_SanitizerHandoff proves the two correctness transforms compose:
// a ToolSearch tool_result of all tool_reference blocks first converts to text (never goes empty),
// and a SEPARATELY empty tool_result is then backfilled by the gate — the exact pipeline order the
// gateway wires (sanitize → repair).
func TestRepairEmptyToolResultContent_SanitizerHandoff(t *testing.T) {
	body := `{"messages":[` +
		`{"role":"user","content":[{"type":"tool_result","tool_use_id":"a","content":[{"type":"tool_reference","tool_name":"Read"}]}]},` +
		`{"role":"user","content":[{"type":"tool_result","tool_use_id":"b","content":[]}]}` +
		`]}`
	// Stage 1: tool_reference sanitizer.
	afterSanitize, refOut := SanitizeAnthropicToolReferences([]byte(body))
	if refOut.Reason != ToolRefReasonNone || refOut.Converted != 1 {
		t.Fatalf("sanitizer stage: reason=%q converted=%d", refOut.Reason, refOut.Converted)
	}
	// Stage 2: empty-content gate on the already-converted body.
	afterRepair, emptyOut := RepairEmptyToolResultContent(afterSanitize)
	if emptyOut.Reason != EmptyContentReasonNone || emptyOut.Repaired != 1 {
		t.Fatalf("gate stage: reason=%q repaired=%d", emptyOut.Reason, emptyOut.Repaired)
	}
	if strings.Contains(string(afterRepair), `"content":[]`) {
		t.Fatalf("empty content array survived the composed pipeline:\n%s", afterRepair)
	}
	if !strings.Contains(string(afterRepair), "[tool: Read]") || !strings.Contains(string(afterRepair), emptyToolResultText) {
		t.Fatalf("composed pipeline lost a repair:\n%s", afterRepair)
	}
	var probe map[string]json.RawMessage
	if err := json.Unmarshal(afterRepair, &probe); err != nil {
		t.Fatalf("composed body is not valid JSON: %v", err)
	}
}

// TestRepairEmptyToolResultContent_Identity confirms the gate is fail-safe: a body with only
// non-empty tool_result content is returned unchanged with a distinguishing reason (silence must
// not read as success).
func TestRepairEmptyToolResultContent_Identity(t *testing.T) {
	cases := map[string]string{
		"non_empty_array": `{"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"t","content":[{"type":"text","text":"ok"}]}]}]}`,
		"string_content":  `{"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"t","content":"ok"}]}]}`,
		"no_messages_key": `{"model":"m"}`,
		"non_json":        `not json at all`,
	}
	wantReason := map[string]string{
		"non_empty_array": EmptyContentReasonNoEmpty,
		"string_content":  EmptyContentReasonNoEmpty,
		"no_messages_key": EmptyContentReasonNoMsgsKey,
		"non_json":        EmptyContentReasonNonJSON,
	}
	for name, body := range cases {
		t.Run(name, func(t *testing.T) {
			out, outcome := RepairEmptyToolResultContent([]byte(body))
			if outcome.Reason != wantReason[name] {
				t.Fatalf("reason = %q, want %q", outcome.Reason, wantReason[name])
			}
			if outcome.Repaired != 0 {
				t.Fatalf("Repaired = %d, want 0 for identity case", outcome.Repaired)
			}
			if string(out) != body {
				t.Fatalf("identity case mutated the body:\nin:  %s\nout: %s", body, out)
			}
		})
	}
}

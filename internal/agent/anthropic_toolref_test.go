package agent

import (
	"encoding/json"
	"strings"
	"testing"
)

// TestSanitizeToolReferencesRepro reproduces the witnessed defect (session
// b98cf818-14ca-4ea8-8373-026656447933): a ToolSearch tool_result whose content array carries
// Claude-Code-internal `tool_reference` blocks — NOT a valid Anthropic tool_result.content type —
// which the real Messages API 400s as malformed. The sanitizer must rewrite each tool_reference
// into a wire-valid text block naming the tool, leaving every other byte untouched, and the result
// must re-decode through fak's own decoder as a well-formed request.
func TestSanitizeToolReferencesRepro(t *testing.T) {
	body := `{"model":"claude-opus-4-8","messages":[` +
		`{"role":"user","content":[{"type":"text","text":"search for the restore tool"}]},` +
		`{"role":"assistant","content":[{"type":"tool_use","id":"call_1","name":"ToolSearch","input":{"query":"restore"}}]},` +
		`{"role":"user","content":[{"type":"tool_result","tool_use_id":"call_1","content":[` +
		`{"type":"tool_reference","tool_name":"mcp__fak__fak_context_restore"},` +
		`{"type":"tool_reference","tool_name":"mcp__fak__fak_context_save"}` +
		`]}]}` +
		`]}`

	out, outcome := SanitizeAnthropicToolReferences([]byte(body))
	if outcome.Reason != ToolRefReasonNone {
		t.Fatalf("expected FIRED (reason none), got reason=%q", outcome.Reason)
	}
	if outcome.Converted != 2 {
		t.Fatalf("expected 2 tool_reference blocks converted, got %d", outcome.Converted)
	}
	// No tool_reference block may survive on the wire.
	if strings.Contains(string(out), "tool_reference") {
		t.Fatalf("sanitized body still contains a tool_reference block:\n%s", out)
	}
	// The referenced tool names must survive as text so the model still sees the discovery result.
	for _, want := range []string{"mcp__fak__fak_context_restore", "mcp__fak__fak_context_save"} {
		if !strings.Contains(string(out), want) {
			t.Fatalf("sanitized body dropped tool name %q:\n%s", want, out)
		}
	}
	// The rewritten body must re-decode as a valid Messages request whose tool_result now carries
	// two text blocks (non-empty content — an empty content array is itself a 400).
	if err := assertValidToolResultText(out); err != nil {
		t.Fatalf("sanitized body is not a well-formed request: %v\n%s", err, out)
	}
}

// TestSanitizeToolReferencesIdentity confirms the sanitizer returns its input UNCHANGED (identity)
// on every no-op path: an empty body, a non-JSON body, a body with no messages, a body whose
// tool_result content is a normal string, and a body whose tool_result carries only valid text
// blocks. A false fire (rewriting a body that needs no repair) would gratuitously burst the cache.
func TestSanitizeToolReferencesIdentity(t *testing.T) {
	cases := []struct {
		name       string
		body       string
		wantReason string
	}{
		{"empty", "", ToolRefReasonEmptyBody},
		{"non_json", "not json at all", ToolRefReasonNonJSON},
		{"no_messages_key", `{"model":"x"}`, ToolRefReasonNoMsgsKey},
		{
			"string_content_tool_result",
			`{"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"t","content":"plain string output"}]}]}`,
			ToolRefReasonNoToolRef,
		},
		{
			"valid_text_tool_result",
			`{"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"t","content":[{"type":"text","text":"ok"}]}]}]}`,
			ToolRefReasonNoToolRef,
		},
		{
			"plain_user_turn",
			`{"messages":[{"role":"user","content":[{"type":"text","text":"hello"}]}]}`,
			ToolRefReasonNoToolRef,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			in := []byte(tc.body)
			out, outcome := SanitizeAnthropicToolReferences(in)
			if outcome.Reason != tc.wantReason {
				t.Fatalf("reason: got %q, want %q", outcome.Reason, tc.wantReason)
			}
			if string(out) != tc.body {
				t.Fatalf("identity violated: body was rewritten\n got: %s\nwant: %s", out, tc.body)
			}
		})
	}
}

// TestSanitizeToolReferencesMixedBlocks covers a tool_result whose content array MIXES a valid text
// block with a tool_reference block: only the tool_reference is rewritten, the text block's bytes
// are preserved verbatim, and the byte prefix before the edit is byte-identical (cache-safe splice).
func TestSanitizeToolReferencesMixedBlocks(t *testing.T) {
	body := `{"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"t","content":[` +
		`{"type":"text","text":"here are the matches:"},` +
		`{"type":"tool_reference","tool_name":"Read"}` +
		`]}]}]}`

	out, outcome := SanitizeAnthropicToolReferences([]byte(body))
	if outcome.Reason != ToolRefReasonNone || outcome.Converted != 1 {
		t.Fatalf("expected 1 conversion, got reason=%q converted=%d", outcome.Reason, outcome.Converted)
	}
	if !strings.Contains(string(out), `"here are the matches:"`) {
		t.Fatalf("original text block was not preserved verbatim:\n%s", out)
	}
	if strings.Contains(string(out), "tool_reference") {
		t.Fatalf("tool_reference survived:\n%s", out)
	}
	// The bytes up to the tool_result content's first block (the untouched text block) must be
	// byte-identical to the input — proof the splice touched only the tool_reference block.
	prefix := `{"messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"t","content":[{"type":"text","text":"here are the matches:"},`
	if !strings.HasPrefix(string(out), prefix) {
		t.Fatalf("prefix bytes shifted; splice was not surgical:\n%s", out)
	}
	if err := assertValidToolResultText(out); err != nil {
		t.Fatalf("sanitized body is not well-formed: %v\n%s", err, out)
	}
}

// assertValidToolResultText checks that every tool_result in a Messages body has a non-empty
// content array whose blocks are all `text` with a non-empty text value — the shape the real
// Anthropic API requires (an empty content array, or a non-text block, is a 400). It mirrors the
// provider-side contract the sanitizer exists to satisfy.
func assertValidToolResultText(raw []byte) error {
	var body struct {
		Messages []struct {
			Content json.RawMessage `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(raw, &body); err != nil {
		return err
	}
	for _, m := range body.Messages {
		if len(m.Content) == 0 || m.Content[0] != '[' {
			continue
		}
		var blocks []struct {
			Type    string          `json:"type"`
			Content json.RawMessage `json:"content"`
		}
		if err := json.Unmarshal(m.Content, &blocks); err != nil {
			return err
		}
		for _, b := range blocks {
			if b.Type != "tool_result" || len(b.Content) == 0 || b.Content[0] != '[' {
				continue
			}
			var inner []struct {
				Type string `json:"type"`
				Text string `json:"text"`
			}
			if err := json.Unmarshal(b.Content, &inner); err != nil {
				return err
			}
			if len(inner) == 0 {
				return errEmptyToolResult
			}
			for _, ib := range inner {
				if ib.Type != "text" {
					return errNonTextBlock
				}
				if ib.Text == "" {
					return errEmptyText
				}
			}
		}
	}
	return nil
}

var (
	errEmptyToolResult = &toolRefTestErr{"tool_result content array is empty"}
	errNonTextBlock    = &toolRefTestErr{"tool_result content has a non-text block"}
	errEmptyText       = &toolRefTestErr{"tool_result content has an empty text value"}
)

type toolRefTestErr struct{ s string }

func (e *toolRefTestErr) Error() string { return e.s }

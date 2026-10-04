package agent

import (
	"strings"
	"testing"
)

// anthropicSystemBody builds a Claude-Code-shaped /v1/messages body whose system array
// leads with the given billing-header block text.
func anthropicSystemBody(billing, user string) []byte {
	return []byte(`{"model":"m","max_tokens":64,"system":[` +
		`{"type":"text","text":"` + billing + `"},` +
		`{"type":"text","text":"You are Claude Code."},` +
		`{"type":"text","text":"Follow the rules.","cache_control":{"type":"ephemeral"}}],` +
		`"messages":[{"role":"user","content":"` + user + `"}]}`)
}

// TestDecodeAnthropicDropsBillingHeaderBlock: a parent and a subagent request whose
// only system difference is the per-request billing header decode to the SAME system
// prompt, and the header never reaches the rendered transcript. req.Raw is untouched.
// fak-test:runtime fast est=10ms lane=default
func TestDecodeAnthropicDropsBillingHeaderBlock(t *testing.T) {
	parentRaw := anthropicSystemBody("x-anthropic-billing-header: cc_version=2.1.37.a1b; cc_entrypoint=cli; cch=0f3e2;", "parent task")
	childRaw := anthropicSystemBody("x-anthropic-billing-header: cc_version=2.1.37.9zz; cc_entrypoint=sdk-ts; cch=77aa1;", "child task")
	parent, err := DecodeAnthropicMessagesRequest(parentRaw)
	if err != nil {
		t.Fatal(err)
	}
	child, err := DecodeAnthropicMessagesRequest(childRaw)
	if err != nil {
		t.Fatal(err)
	}
	const want = "You are Claude Code.\nFollow the rules."
	if parent.System != want || child.System != want {
		t.Fatalf("system = %q / %q, want %q", parent.System, child.System, want)
	}
	if parent.Messages[0].Role != RoleSystem || parent.Messages[0].Content != child.Messages[0].Content {
		t.Fatalf("leading system message differs: %q vs %q", parent.Messages[0].Content, child.Messages[0].Content)
	}
	if string(parent.Raw) != string(parentRaw) {
		t.Fatal("req.Raw must stay byte-faithful for the Anthropic passthrough")
	}
}

// TestDecodeAnthropicKeepsNonHeaderSystemBlocks pins the tight match: only a
// standalone single-line header block is dropped.
// fak-test:runtime fast est=10ms lane=default
func TestDecodeAnthropicKeepsNonHeaderSystemBlocks(t *testing.T) {
	cases := []struct {
		name, raw, want string
	}{
		{"string system verbatim",
			`{"system":"x-anthropic-billing-header: a=1;","messages":[]}`,
			"x-anthropic-billing-header: a=1;"},
		{"header block with trailing prose kept",
			`{"system":[{"type":"text","text":"x-anthropic-billing-header: a=1;\nreal rule"}],"messages":[]}`,
			"x-anthropic-billing-header: a=1;\nreal rule"},
		{"header mid-text kept",
			`{"system":[{"type":"text","text":"see x-anthropic-billing-header: a=1;"}],"messages":[]}`,
			"see x-anthropic-billing-header: a=1;"},
		{"no header unchanged",
			`{"system":[{"type":"text","text":"A"},{"type":"text","text":"B"}],"messages":[]}`,
			"A\nB"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req, err := DecodeAnthropicMessagesRequest([]byte(tc.raw))
			if err != nil {
				t.Fatal(err)
			}
			if req.System != tc.want {
				t.Fatalf("system = %q, want %q", req.System, tc.want)
			}
		})
	}
	if !IsAnthropicBillingHeaderText("  x-anthropic-billing-header: cc_version=1; \n") {
		t.Fatal("single header line with surrounding whitespace must match")
	}
	if IsAnthropicBillingHeaderText("X-Anthropic-Billing-Header: a") || strings.Contains(AnthropicBillingHeaderPrefix, " ") {
		t.Fatal("match must be exact-case on the lowercase prefix")
	}
}

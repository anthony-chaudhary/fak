package gateway

import (
	"net/http"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// harness_prefix_normalize.go — cross-request prefix stabilization for third-party
// harness wires (/v1/messages from Claude Code, /v1/chat/completions from Pi /
// OpenCode / Codex). A parent agent and the subagents it spawns send the SAME system
// prompt and tool set, but (a) list the tools in different orders, (b) carry
// per-request volatile metadata (date, session/turn ids) inside the system prompt, and
// (c) on Claude Code, a per-request `x-anthropic-billing-header:` line at byte 0 of
// the system prompt. Each of those breaks the rendered token prefix at or before the
// end of the system+tools section, so radix prefix-KV reuse never fires across them.
//
// normalizeHarnessPrefix applies the /v1/responses normalization
// (CanonicalizeResponsesPrefix: deterministic tool order + volatile metadata out of the
// standing instructions) to the decoded transcript the planner renders and forwards
// upstream. Semantics are preserved: tools are reordered, never renamed (tool_choice
// references by name stay valid); volatile lines are moved, never dropped.

// normalizeHarnessPrefix returns messages/tools with a deterministic, request-invariant
// system+tools head:
//   - tools sorted by function name (CanonicalizeToolDefs, stable for duplicates);
//   - a leading `x-anthropic-billing-header:` line removed from each leading
//     system/developer message (agent.IsAnthropicBillingHeaderText — exact, single-line);
//   - volatile metadata (ExtractVolatileMetadata) lifted out of the leading
//     system/developer messages and appended to the FIRST user message. The first user
//     message (not the last, as /v1/responses does) is the anchor so the relocated text
//     stays at a fixed position for the life of the conversation: later turns extend the
//     cached history instead of rewriting it.
//
// When no user message follows the system head, volatile metadata is left in place
// (there is nowhere to move it without dropping it). Inputs are never mutated.
func normalizeHarnessPrefix(messages []agent.Message, tools []agent.ToolDef) ([]agent.Message, []agent.ToolDef) {
	tools = CanonicalizeToolDefs(tools)

	head := 0
	for head < len(messages) && isStandingInstructionRole(messages[head].Role) {
		head++
	}
	if head == 0 {
		return messages, tools
	}
	anchor := -1
	for i := head; i < len(messages); i++ {
		if messages[i].Role == agent.RoleUser {
			anchor = i
			break
		}
	}

	var out []agent.Message
	ensureCopy := func() {
		if out == nil {
			out = append([]agent.Message(nil), messages...)
		}
	}
	var volatile []string
	for i := 0; i < head; i++ {
		content := stripLeadingBillingHeader(messages[i].Content)
		if anchor >= 0 {
			if cleaned, extracted := ExtractVolatileMetadata(content); len(extracted) > 0 {
				content = cleaned
				volatile = append(volatile, extracted...)
			}
		}
		if content != messages[i].Content {
			ensureCopy()
			out[i].Content = content
		}
	}
	if out == nil {
		return messages, tools
	}
	if len(volatile) > 0 {
		text := strings.Join(volatile, "\n")
		if strings.TrimSpace(out[anchor].Content) == "" {
			out[anchor].Content = text
		} else {
			out[anchor].Content = strings.TrimRight(out[anchor].Content, "\r\n") + "\n\n" + text
		}
	}
	return out, tools
}

func isStandingInstructionRole(role string) bool {
	return role == agent.RoleSystem || role == "developer"
}

// stripLeadingBillingHeader drops the first line of s when that line alone is a
// Claude Code billing header. Only the leading line is examined: the header is always
// prepended, and a mid-prompt occurrence is left verbatim.
func stripLeadingBillingHeader(s string) string {
	trimmed := strings.TrimLeft(s, " \t\r\n")
	first, rest, _ := strings.Cut(trimmed, "\n")
	if !agent.IsAnthropicBillingHeaderText(first) {
		return s
	}
	return strings.TrimLeft(rest, "\r\n")
}

// normalizeAnthropicHarnessPrefix applies normalizeHarnessPrefix to a decoded
// /v1/messages request whenever the transcript is rendered by fak (in-kernel, or
// re-marshaled for an OpenAI-compatible upstream such as llama-server). The
// byte-faithful Anthropic passthrough is exempt: it forwards req.Raw, and its raw-body
// transforms match on the decoded messages, which must stay aligned with those bytes.
// Call after prepareChatRoute so the per-request route is known.
func (s *Server) normalizeAnthropicHarnessPrefix(r *http.Request, req *agent.AnthropicMessagesRequest) {
	if chatRouteFromContext(r.Context()) == nil && s.anthropicPassthroughFor(req.Model) {
		return
	}
	req.Messages, req.Tools = normalizeHarnessPrefix(req.Messages, req.Tools)
}

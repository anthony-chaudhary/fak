package agent

import (
	"encoding/json"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

// checkStop reports whether the accumulated decode text ends with any of the
// per-request stop sequences, returning the text with the matched stop suffix
// trimmed. It mirrors the HTTP wires' contract: the stop string ends generation and
// is NOT echoed in the returned content. The LONGEST matching stop wins so the trim
// is maximal, and an empty stop string is ignored (it would otherwise match every
// text and truncate every turn to nothing). An empty stop set never fires, so the
// default in-kernel path is byte-for-byte the pre-seam behavior.
func checkStop(text string, stop []string) (string, bool) {
	best := ""
	for _, s := range stop {
		if s == "" {
			continue
		}
		if strings.HasSuffix(text, s) && len(s) > len(best) {
			best = s
		}
	}
	if best == "" {
		return text, false
	}
	return text[:len(text)-len(best)], true
}

// renderChatML renders the transcript as Qwen/SmolLM2 ChatML, terminating with an
// open assistant turn for generation. System messages fold into one leading system
// block. It is the zero-tools form: renderChatMLTools(messages, nil). The eviction /
// reuse paths use this (and renderTranscript) so their token path is byte-identical to
// the pre-tool-calling behavior — protecting the radix prefix invariant.
func renderChatML(messages []Message) string {
	return renderChatMLTools(messages, nil)
}

// renderChatMLTools is renderChatML with tool support: it advertises the tool JSON
// schemas to the model and renders prior tool-call / tool-result history in Qwen2.5's
// canonical <tool_call>/<tool_response> ChatML. It terminates with an open assistant
// turn for generation. When tools is empty AND no message carries a structured tool
// call or tool result, its output is byte-for-byte identical to the old renderChatML.
func renderChatMLTools(messages []Message, tools []ToolDef) string {
	return renderTranscriptTools(messages, tools) + "<|im_start|>assistant\n"
}

// renderTranscript renders the messages as complete ChatML turns WITHOUT the trailing
// open assistant turn. The zero-tools form: renderTranscriptTools(messages, nil). The
// poison-eviction path uses this so its token path ends exactly on a turn boundary (the
// atomic <|im_end|> special token), keeping it a clean token-prefix of any cached turn
// that began with these messages.
func renderTranscript(messages []Message) string {
	return renderTranscriptTools(messages, nil)
}

// toolSpecBlock renders the canonical Qwen2.5 tool-spec preamble for the folded system
// block: the <tools>…</tools> signatures plus the "emit a <tool_call> json object"
// instruction. It is deterministic (schemas in declaration order) so it is a stable part
// of every token-prefix when folded into the single leading system block — the constraint
// that keeps radix KV reuse valid across a tool-using session.
func toolSpecBlock(tools []ToolDef) string {
	if len(tools) == 0 {
		return ""
	}
	var b strings.Builder
	b.WriteString("\n\n# Tools\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>")
	for _, t := range tools {
		fn := t.Function
		params := fn.Parameters
		if len(params) == 0 {
			params = json.RawMessage("{}")
		}
		// Marshal one OpenAI-style {"type":"function","function":{…}} signature per tool.
		// Build it from a stable field order via json.Marshal of a map alternative would
		// re-sort keys; use an explicit struct so the rendering is deterministic.
		sig := struct {
			Type     string `json:"type"`
			Function struct {
				Name        string          `json:"name"`
				Description string          `json:"description"`
				Parameters  json.RawMessage `json:"parameters"`
			} `json:"function"`
		}{Type: "function"}
		sig.Function.Name = fn.Name
		sig.Function.Description = fn.Description
		sig.Function.Parameters = params
		enc, err := json.Marshal(sig)
		if err != nil {
			// A malformed tool schema must not corrupt the prompt; skip it (the gateway
			// validates schemas upstream, so this is belt-and-suspenders).
			continue
		}
		b.WriteString("\n")
		b.Write(enc)
	}
	b.WriteString("\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>")
	return b.String()
}

// renderTranscriptTools is the single rendering core. When tools is non-empty it folds
// the tool-spec block into the leading system block; assistant tool calls render as
// canonical <tool_call> blocks and role=tool messages render as <tool_response> user
// turns. With nil tools and no structured tool call/result on any message, the output is
// byte-for-byte identical to the historical renderTranscript.
func renderTranscriptTools(messages []Message, tools []ToolDef) string {
	var b strings.Builder
	var sys []string
	for _, m := range messages {
		if m.Role == "system" && strings.TrimSpace(m.Content) != "" {
			sys = append(sys, m.Content)
		}
	}
	spec := toolSpecBlock(tools)
	// Emit a leading system block when there is any system text OR a tool spec to
	// advertise. The spec folds into the SAME block (after the system text) so it is part
	// of every token-prefix.
	if len(sys) > 0 || spec != "" {
		b.WriteString("<|im_start|>system\n")
		b.WriteString(strings.Join(sys, "\n"))
		b.WriteString(spec)
		b.WriteString("<|im_end|>\n")
	}
	for _, m := range messages {
		role, content := m.Role, m.Content
		switch role {
		case "system":
			continue
		case "tool":
			// A tool result reads as user-supplied context to the model. When the result
			// carries a tool name, wrap it in Qwen's canonical <tool_response> grammar so a
			// tool-trained model recognizes the multi-turn tool flow; otherwise keep the
			// historical bare "name: content" form (byte-identical to the pre-tool path).
			role = "user"
			if m.Name != "" {
				content = "<tool_response>\n" + m.Name + ": " + content + "\n</tool_response>"
			}
		case "assistant":
			for _, tc := range m.ToolCalls {
				// Canonical Qwen2.5 <tool_call> block: arguments as a JSON VALUE, not a
				// quoted string, so it round-trips cleanly through LiftTextToolCalls.
				args := strings.TrimSpace(tc.Function.Arguments)
				if args == "" || !json.Valid([]byte(args)) {
					args = "{}"
				}
				content += "\n<tool_call>\n{\"name\": " + strconv.Quote(tc.Function.Name) + ", \"arguments\": " + args + "}\n</tool_call>"
			}
		}
		b.WriteString("<|im_start|>")
		b.WriteString(role)
		b.WriteString("\n")
		b.WriteString(content)
		b.WriteString("<|im_end|>\n")
	}
	return b.String()
}

// inKernelStopIDs mirrors cmd/fakchat.stopIDs: <|im_end|>, <|endoftext|>, and any
// EOS ids the model config declares.
func inKernelStopIDs(tok *tokenizer.Tokenizer, cfg model.Config) map[int]bool {
	stops := map[int]bool{}
	for id, content := range tok.SpecialTokens() {
		if content == "<|im_end|>" || content == "<|endoftext|>" {
			stops[id] = true
		}
	}
	if cfg.EOSTokenID > 0 {
		stops[cfg.EOSTokenID] = true
	}
	for _, e := range cfg.EOSTokenIDs {
		if e > 0 {
			stops[e] = true
		}
	}
	return stops
}

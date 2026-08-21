package agent

import (
	"encoding/json"
	"os"
	"regexp"
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

const qwenNoThinkAssistantSeed = "<think>\n\n</think>\n\n"

// renderInKernelChatMLTools is the live in-kernel prompt renderer. For Qwen3.5/Qwen3.6 hybrid
// reasoning checkpoints it mirrors tokenizer.apply_chat_template(enable_thinking=false) by
// pre-seeding an empty reasoning block after the assistant header. Otherwise short max_tokens turns
// spend their whole budget inside an unclosed <think> block; splitReasoning then correctly returns
// empty visible content. FAK_INKERNEL_ENABLE_THINKING=1 keeps the raw reasoning mode for diagnosis.
func renderInKernelChatMLTools(messages []Message, tools []ToolDef, cfg model.Config) string {
	return renderInKernelChatMLRequest(messages, tools, cfg, nil, nil)
}

// renderInKernelChatMLRequest carries request-level output constraints into the
// same ChatML prompt the in-kernel model sees. Generic provider adapters carry
// response_format upstream on the wire; the in-kernel path has no such second
// channel, so dropping it here silently turns strict JSON into unconstrained prose.
func renderInKernelChatMLRequest(messages []Message, tools []ToolDef, cfg model.Config, responseFormat, toolChoice json.RawMessage) string {
	instruction := inKernelResponseFormatInstruction(responseFormat)
	if forced := inKernelForcedToolInstruction(toolChoice, tools); forced != "" {
		if instruction != "" {
			instruction += "\n"
		}
		instruction += forced
	}
	if instruction != "" {
		messages = append([]Message{{Role: RoleSystem, Content: instruction}}, messages...)
	}
	if inKernelEffectiveToolName(toolChoice, tools) != "" {
		tools = nil
	}
	chat := renderChatMLTools(messages, tools)
	if inKernelSuppressQwenThinking(cfg) {
		chat += qwenNoThinkAssistantSeed
	}
	return chat
}

func inKernelSuppressQwenThinking(cfg model.Config) bool {
	return cfg.IsQwen35Hybrid() && os.Getenv("FAK_INKERNEL_ENABLE_THINKING") != "1"
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
	b.WriteString("\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer normally\n</IMPORTANT>")
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
	toolByID := make(map[string]string)
	for _, m := range messages {
		role, content := m.Role, m.Content
		switch role {
		case "system":
			continue
		case "tool":
			// A tool result reads as user-supplied context to the model. When the result
			// carries a tool name, wrap it in Qwen's canonical <tool_response> grammar so a
			// tool-trained model recognizes the multi-turn tool flow; otherwise keep the
			// historical bare content form (byte-identical to the pre-tool path).
			role = "user"
			name := strings.TrimSpace(m.Name)
			if name == "" && strings.TrimSpace(m.ToolCallID) != "" {
				name = toolByID[strings.TrimSpace(m.ToolCallID)]
			}
			content = qwenToolResponseBlock(name, content)
		case "assistant":
			for _, tc := range m.ToolCalls {
				// Canonical Qwen2.5 <tool_call> block: arguments as a JSON VALUE, not a
				// quoted string, so it round-trips cleanly through LiftTextToolCalls.
				if id, name := strings.TrimSpace(tc.ID), strings.TrimSpace(tc.Function.Name); id != "" && name != "" {
					toolByID[id] = name
				}
				content += qwenToolCallBlock(tc.Function.Name, tc.Function.Arguments)
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

func enforceForcedToolChoice(comp *Completion, choice json.RawMessage, tools []ToolDef, messages []Message) *Completion {
	if comp == nil || len(comp.Message.ToolCalls) > 0 {
		return comp
	}
	if comp.FinishReason == "length" {
		comp.ToolCallsDropped = true
		return comp
	}
	name := inKernelEffectiveToolName(choice, tools)
	if name == "" {
		return comp
	}
	args, ok := forcedToolArgumentsFromMessages(name, tools, messages, comp.Message.Content)
	if !ok {
		comp.ToolCallsDropped = true
		return comp
	}
	comp.Message.Content = ""
	comp.Message.ToolCalls = []ToolCall{{ID: "call_forced_0", Type: "function", Function: Func{Name: name, Arguments: args}}}
	comp.FinishReason = "tool_calls"
	return comp
}
func forcedToolArgumentsFromMessages(name string, tools []ToolDef, messages []Message, assistantContent ...string) (string, bool) {
	var required []string
	for _, tool := range tools {
		if tool.Function.Name != name {
			continue
		}
		var schema struct {
			Required []string `json:"required"`
		}
		if json.Unmarshal(tool.Function.Parameters, &schema) == nil {
			required = schema.Required
		}
		break
	}
	if len(required) == 0 {
		return "", false
	}
	text := ""
	for _, message := range messages {
		if message.Role == RoleUser {
			text += "\n" + message.Content
		}
	}
	args := make(map[string]any, len(required))
	for _, key := range required {
		pattern := regexp.MustCompile(`(?i)\b` + regexp.QuoteMeta(key) + `\b\s+(?:to\s+|=\s*|is\s+)?(?:the\s+boolean\s+)?("[^"]*"|'[^']*'|[^.,;\n]+)`)
		match := pattern.FindStringSubmatch(text)
		if len(match) < 2 {
			if name == "Write" {
				return forcedWriteArguments(text, assistantContent)
			}
			return "", false
		}
		valueText := strings.Trim(strings.TrimSpace(match[1]), `"'`)
		var value any
		if json.Unmarshal([]byte(valueText), &value) != nil {
			value = valueText
		}
		args[key] = value
	}
	encoded, err := json.Marshal(args)
	return string(encoded), err == nil
}

func forcedWriteArguments(userText string, assistantContent []string) (string, bool) {
	pathMatch := regexp.MustCompile(`(?i)(?:^|[\s"''` + "`" + `])([a-z0-9_./\\-]+\.(?:html?|css|js|json|md|txt|go|py|rs|java|c|cc|cpp|h|hpp|yaml|yml|toml))(?:$|[\s"''` + "`" + `,.;:])`).FindStringSubmatch(userText)
	if len(pathMatch) < 2 || len(assistantContent) == 0 {
		return "", false
	}
	content := strings.TrimSpace(assistantContent[len(assistantContent)-1])
	if fenced := regexp.MustCompile("(?s)```(?:[a-zA-Z0-9_+-]+)?\\s*\\n(.*?)```").FindStringSubmatch(content); len(fenced) == 2 {
		content = fenced[1]
	} else if !strings.HasPrefix(strings.ToLower(content), "<!doctype") && !strings.HasPrefix(strings.ToLower(content), "<html") {
		return "", false
	}
	if strings.TrimSpace(content) == "" {
		return "", false
	}
	encoded, err := json.Marshal(map[string]any{"file_path": pathMatch[1], "content": content, "mode": "create"})
	return string(encoded), err == nil
}

func inKernelForcedToolInstruction(raw json.RawMessage, tools []ToolDef) string {
	name := inKernelEffectiveToolName(raw, tools)
	if name == "" {
		return ""
	}
	parameters := json.RawMessage(`{"type":"object"}`)
	for _, tool := range tools {
		if tool.Function.Name == name && len(tool.Function.Parameters) > 0 {
			parameters = tool.Function.Parameters
			break
		}
	}
	schema, _ := json.Marshal(struct {
		Type       string `json:"type"`
		Properties struct {
			Name struct {
				Const string `json:"const"`
			} `json:"name"`
			Arguments json.RawMessage `json:"arguments"`
		} `json:"properties"`
		Required             []string `json:"required"`
		AdditionalProperties bool     `json:"additionalProperties"`
	}{Type: "object", Properties: struct {
		Name struct {
			Const string `json:"const"`
		} `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	}{Name: struct {
		Const string `json:"const"`
	}{Const: name}, Arguments: parameters}, Required: []string{"name", "arguments"}, AdditionalProperties: false})
	return "Return only one valid JSON object matching this schema exactly. Do not use XML, Markdown, reasoning, or prose. JSON schema: " + string(schema)
}

func inKernelEffectiveToolName(raw json.RawMessage, tools []ToolDef) string {
	if name := inKernelForcedToolName(raw); name != "" {
		return name
	}
	if inKernelRequiresTool(raw) && len(tools) == 1 {
		return strings.TrimSpace(tools[0].Function.Name)
	}
	return ""
}
func inKernelRequiresTool(raw json.RawMessage) bool {
	var choice string
	return json.Unmarshal(raw, &choice) == nil && choice == "required"
}
func inKernelForcedToolName(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var choice struct {
		Type     string `json:"type"`
		Function struct {
			Name string `json:"name"`
		} `json:"function"`
	}
	if json.Unmarshal(raw, &choice) != nil || choice.Type != "function" {
		return ""
	}
	return strings.TrimSpace(choice.Function.Name)
}

func inKernelResponseFormatInstruction(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var format struct {
		Type       string `json:"type"`
		JSONSchema struct {
			Schema json.RawMessage `json:"schema"`
		} `json:"json_schema"`
	}
	if json.Unmarshal(raw, &format) != nil {
		return ""
	}
	switch strings.TrimSpace(format.Type) {
	case "json_object":
		return "Return only one valid JSON object. Do not use Markdown fences or explanatory prose."
	case "json_schema":
		schema := strings.TrimSpace(string(format.JSONSchema.Schema))
		if schema == "" || !json.Valid([]byte(schema)) {
			return "Return only one valid JSON object. Do not use Markdown fences or explanatory prose."
		}
		return "Return only one valid JSON object matching this schema exactly. Do not use Markdown fences or explanatory prose. JSON schema: " + schema
	default:
		return ""
	}
}

func qwenToolResponseBlock(name, content string) string {
	if strings.TrimSpace(name) == "" {
		return content
	}
	return "<tool_response>\n" + strings.TrimSpace(name) + ": " + content + "\n</tool_response>"
}

func qwenToolCallBlock(name, args string) string {
	args = qwenToolCallArgs(args)
	return "\n<tool_call>\n{\"name\": " + strconv.Quote(strings.TrimSpace(name)) + ", \"arguments\": " + args + "}\n</tool_call>"
}

func qwenToolCallArgs(args string) string {
	args = strings.TrimSpace(args)
	if args == "" || !json.Valid([]byte(args)) {
		return "{}"
	}
	return args
}

// StopIDs collects the generation stop tokens for a ChatML-family model: the
// <|im_end|> / <|endoftext|> special tokens the tokenizer declares, plus any EOS id
// (singular or list) the model config declares. Ids <= 0 are treated as "unset" and
// never become a stop, so a config that omits the field cannot halt decode at token 0.
//
// Exported because a stop set is a property of the MODEL, not of a caller: this
// in-kernel planner and the cmd/fakchat REPL decode the same checkpoints, and each
// previously carried a byte-identical private copy. A stop token that one of them
// honoured and the other did not would end the SAME turn at two different places.
func StopIDs(tok *tokenizer.Tokenizer, cfg model.Config) map[int]bool {
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

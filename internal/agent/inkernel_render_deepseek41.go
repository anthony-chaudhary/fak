package agent

import (
	"bytes"
	"encoding/json"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

// DeepSeek V4.1 control tokens, as the GGUF vocab spells them.
const (
	deepSeekV41BOS       = "<｜begin▁of▁sentence｜>"
	deepSeekV41EOS       = "<｜end▁of▁sentence｜>"
	deepSeekV41User      = "<｜User｜>"
	deepSeekV41Assistant = "<｜Assistant｜>"
	deepSeekV41ThinkEnd  = "</think>"
)

const deepSeekV41ChatRendererID = "deepseek41-chat-nothink-v2"

// Template text of the GGUF's tokenizer.chat_template; dsml is its tool-call marker.
const (
	deepSeekV41DSML           = "｜DSML｜"
	deepSeekV41ToolsFooter    = "\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n"
	deepSeekV41ResponseFormat = "## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n"
	deepSeekV41ToolsHeader    = "## Tools\n\nYou have access to a set of tools to help answer the user's question. You can invoke tools by writing a \"<" + deepSeekV41DSML + "tool_calls>\" block like the following:\n\n<" + deepSeekV41DSML + "tool_calls>\n<" + deepSeekV41DSML + "invoke name=\"$TOOL_NAME\">\n<" + deepSeekV41DSML + "parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</" + deepSeekV41DSML + "parameter>\n...\n</" + deepSeekV41DSML + "invoke>\n<" + deepSeekV41DSML + "invoke name=\"$TOOL_NAME2\">\n...\n</" + deepSeekV41DSML + "invoke>\n</" + deepSeekV41DSML + "tool_calls>\n\nString parameters should be specified as is and set `string=\"true\"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\nIf thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.\n\nOtherwise, output directly after </think> with tool calls or final response.\n\n### Available Tool Schemas\n\n"
)

// renderDeepSeekV41Chat ports the non-thinking branch of the GGUF's
// tokenizer.chat_template (vcruz305/DeepSeek-V4.1-Flash-GGUF Q2_K shard 1), the
// authority over the minimal layout in vcruz305/llama.cpp
// src/llama-chat.cpp:548-561@5210c7c (MIT). BOS, then the system prompt: system
// messages joined by a blank line, then the tool schemas and the response format.
// Consecutive user, developer and tool turns share one <｜User｜> block (tool
// results wrapped in <tool_result>); each assistant turn is <｜Assistant｜></think>,
// its content, any DSML tool calls and EOS; the open generation turn closes it.
func renderDeepSeekV41Chat(messages []Message, tools []ToolDef, responseFormat json.RawMessage) string {
	return joinPromptSegments(renderDeepSeekV41ChatSegments(messages, tools, responseFormat))
}

// renderDeepSeekV41ChatSegments is renderDeepSeekV41Chat split into template text
// (Trusted) and client text: message content, tool schemas, tool-call names and
// arguments, and the response format.
func renderDeepSeekV41ChatSegments(messages []Message, tools []ToolDef, responseFormat json.RawMessage) []tokenizer.Segment {
	var segs []tokenizer.Segment
	trusted := func(text string) {
		if text != "" {
			segs = append(segs, tokenizer.Segment{Text: text, Trusted: true})
		}
	}
	ordinary := func(text string) {
		if text != "" {
			segs = append(segs, tokenizer.Segment{Text: text})
		}
	}
	trusted(deepSeekV41BOS)
	system := len(segs)
	first := true
	for _, m := range messages {
		if m.Role != RoleSystem {
			continue
		}
		if !first {
			trusted("\n\n")
		}
		ordinary(m.Content)
		first = false
	}
	systemJoin := func() {
		if len(segs) > system {
			trusted("\n\n")
		}
	}
	if schemas := deepSeekV41ToolSchemas(tools); len(schemas) > 0 {
		systemJoin()
		trusted(deepSeekV41ToolsHeader)
		for _, schema := range schemas {
			ordinary(schema)
			trusted("\n")
		}
		trusted(deepSeekV41ToolsFooter)
	}
	if format := bytes.TrimSpace(responseFormat); len(format) > 0 && !bytes.Equal(format, []byte("null")) {
		systemJoin()
		trusted(deepSeekV41ResponseFormat)
		ordinary(spacedJSON(string(format)))
	}

	inUser := false
	openUser := func() {
		if inUser {
			trusted("\n\n")
			return
		}
		trusted(deepSeekV41User)
		inUser = true
	}
	for _, m := range messages {
		switch m.Role {
		case RoleUser, RoleDeveloper:
			openUser()
			ordinary(m.Content)
		case RoleTool:
			openUser()
			trusted("<tool_result>")
			ordinary(m.Content)
			trusted("</tool_result>")
		case RoleAssistant:
			inUser = false
			trusted(deepSeekV41Assistant + deepSeekV41ThinkEnd)
			ordinary(m.Content)
			if len(m.ToolCalls) > 0 {
				trusted("\n\n<" + deepSeekV41DSML + "tool_calls>\n")
				for _, call := range m.ToolCalls {
					deepSeekV41RenderToolCall(trusted, ordinary, call.Function)
				}
				trusted("</" + deepSeekV41DSML + "tool_calls>")
			}
			trusted(deepSeekV41EOS)
		}
	}
	trusted(deepSeekV41Assistant + deepSeekV41ThinkEnd)
	return segs
}

// deepSeekV41ToolSchemas renders `tool['function'] | tojson` per function tool,
// keeping the declared parameter schema's key order; the template puts one per line.
func deepSeekV41ToolSchemas(tools []ToolDef) []string {
	var out []string
	for _, tool := range tools {
		if tool.Type != "" && tool.Type != "function" {
			continue
		}
		fn := struct {
			Name        string          `json:"name"`
			Description string          `json:"description,omitempty"`
			Parameters  json.RawMessage `json:"parameters,omitempty"`
		}{Name: tool.Function.Name, Description: tool.Function.Description, Parameters: tool.Function.Parameters}
		encoded, err := qwenTojsonSpaced(fn)
		if err != nil {
			continue
		}
		out = append(out, encoded)
	}
	return out
}

// deepSeekV41RenderToolCall renders one historical call in the template's DSML
// form: string arguments verbatim, every other value as tojson, in source order.
func deepSeekV41RenderToolCall(trusted, ordinary func(string), fn Func) {
	trusted("<" + deepSeekV41DSML + "invoke name=\"")
	ordinary(fn.Name)
	trusted("\">\n")
	args := deepSeekV41ToolArguments(fn.Arguments)
	for _, arg := range args {
		kind := "false"
		if arg.isString {
			kind = "true"
		}
		trusted("<" + deepSeekV41DSML + "parameter name=\"")
		ordinary(arg.name)
		trusted("\" string=\"" + kind + "\">")
		ordinary(arg.value)
		trusted("</" + deepSeekV41DSML + "parameter>\n")
	}
	if len(args) == 0 {
		trusted("\n")
	}
	trusted("</" + deepSeekV41DSML + "invoke>\n")
}

type deepSeekV41ToolArgument struct {
	name, value string
	isString    bool
}

func deepSeekV41ToolArguments(raw string) []deepSeekV41ToolArgument {
	decoder := json.NewDecoder(strings.NewReader(raw))
	if start, err := decoder.Token(); err != nil || start != json.Delim('{') {
		return nil
	}
	var out []deepSeekV41ToolArgument
	for decoder.More() {
		name, err := decoder.Token()
		if err != nil {
			return nil
		}
		var value json.RawMessage
		if err := decoder.Decode(&value); err != nil {
			return nil
		}
		arg := deepSeekV41ToolArgument{name: name.(string)}
		if value = bytes.TrimSpace(value); len(value) > 0 && value[0] == '"' {
			var text string
			if json.Unmarshal(value, &text) == nil {
				arg.value, arg.isString = text, true
			}
		}
		if !arg.isString {
			arg.value = spacedJSON(string(value))
		}
		out = append(out, arg)
	}
	return out
}

// inKernelRenderPrompt is the text a native planner encodes for one request: the
// raw V4.1 completion text, the V4.1 chat template, or ChatML for every other family.
func inKernelRenderPrompt(messages []Message, tools []ToolDef, cfg model.Config, sp SampleParams) string {
	var toolChoice string
	if json.Unmarshal(sp.ToolChoice, &toolChoice) == nil && toolChoice == "none" {
		tools = nil
	}
	switch {
	case inKernelRawV41Prompt(cfg, sp):
		var b strings.Builder
		for _, m := range messages {
			b.WriteString(m.Content)
		}
		return b.String()
	case cfg.IsDeepSeekV41():
		messages, tools = deepSeekV41ChatRequest(messages, tools, sp.ToolChoice)
		return renderDeepSeekV41Chat(messages, tools, sp.ResponseFormat)
	default:
		return renderInKernelChatMLRequest(messages, tools, cfg, sp.ResponseFormat, sp.ToolChoice, sp)
	}
}

// inKernelRenderDeepSeekV41ChatSegments is the V4.1 chat branch of
// inKernelRenderPrompt as trusted template and ordinary client segments.
func inKernelRenderDeepSeekV41ChatSegments(messages []Message, tools []ToolDef, sp SampleParams) []tokenizer.Segment {
	messages, tools = deepSeekV41ChatRequest(messages, tools, sp.ToolChoice)
	return renderDeepSeekV41ChatSegments(messages, tools, sp.ResponseFormat)
}

// deepSeekV41ChatRequest applies tool_choice to a V4.1 chat request. The template
// carries response_format itself; a forced tool keeps the shared instruction and
// drops the schemas, as the ChatML path does.
func deepSeekV41ChatRequest(messages []Message, tools []ToolDef, toolChoice json.RawMessage) ([]Message, []ToolDef) {
	var choice string
	if json.Unmarshal(toolChoice, &choice) == nil && choice == "none" {
		tools = nil
	}
	if forced := inKernelForcedToolInstruction(toolChoice, tools); forced != "" {
		messages = append([]Message{{Role: RoleSystem, Content: forced}}, messages...)
	}
	if inKernelEffectiveToolName(toolChoice, tools) != "" {
		tools = nil
	}
	return messages, tools
}

package agent

import (
	"bytes"
	"encoding/json"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/model"
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
	var system strings.Builder
	first := true
	for _, m := range messages {
		if m.Role != RoleSystem {
			continue
		}
		if !first {
			system.WriteString("\n\n")
		}
		system.WriteString(m.Content)
		first = false
	}
	if schemas := deepSeekV41ToolSchemas(tools); schemas != "" {
		if system.Len() > 0 {
			system.WriteString("\n\n")
		}
		system.WriteString(deepSeekV41ToolsHeader + schemas + deepSeekV41ToolsFooter)
	}
	if format := bytes.TrimSpace(responseFormat); len(format) > 0 && !bytes.Equal(format, []byte("null")) {
		if system.Len() > 0 {
			system.WriteString("\n\n")
		}
		system.WriteString(deepSeekV41ResponseFormat + spacedJSON(string(format)))
	}

	var b strings.Builder
	b.WriteString(deepSeekV41BOS)
	b.WriteString(system.String())
	inUser := false
	openUser := func() {
		if inUser {
			b.WriteString("\n\n")
			return
		}
		b.WriteString(deepSeekV41User)
		inUser = true
	}
	for _, m := range messages {
		switch m.Role {
		case RoleUser, RoleDeveloper:
			openUser()
			b.WriteString(m.Content)
		case RoleTool:
			openUser()
			b.WriteString("<tool_result>" + m.Content + "</tool_result>")
		case RoleAssistant:
			inUser = false
			b.WriteString(deepSeekV41Assistant + deepSeekV41ThinkEnd + m.Content)
			if len(m.ToolCalls) > 0 {
				b.WriteString("\n\n<" + deepSeekV41DSML + "tool_calls>\n")
				for _, call := range m.ToolCalls {
					deepSeekV41RenderToolCall(&b, call.Function)
				}
				b.WriteString("</" + deepSeekV41DSML + "tool_calls>")
			}
			b.WriteString(deepSeekV41EOS)
		}
	}
	b.WriteString(deepSeekV41Assistant + deepSeekV41ThinkEnd)
	return b.String()
}

// deepSeekV41ToolSchemas renders `tool['function'] | tojson` per function tool,
// one per line, keeping the declared parameter schema's key order.
func deepSeekV41ToolSchemas(tools []ToolDef) string {
	var b strings.Builder
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
		b.WriteString(encoded + "\n")
	}
	return b.String()
}

// deepSeekV41RenderToolCall renders one historical call in the template's DSML
// form: string arguments verbatim, every other value as tojson, in source order.
func deepSeekV41RenderToolCall(b *strings.Builder, fn Func) {
	b.WriteString("<" + deepSeekV41DSML + "invoke name=\"" + fn.Name + "\">\n")
	args := deepSeekV41ToolArguments(fn.Arguments)
	for _, arg := range args {
		kind := "false"
		if arg.isString {
			kind = "true"
		}
		b.WriteString("<" + deepSeekV41DSML + "parameter name=\"" + arg.name + "\" string=\"" + kind + "\">" + arg.value + "</" + deepSeekV41DSML + "parameter>\n")
	}
	if len(args) == 0 {
		b.WriteByte('\n')
	}
	b.WriteString("</" + deepSeekV41DSML + "invoke>\n")
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
	switch {
	case inKernelRawV41Prompt(cfg, sp):
		var b strings.Builder
		for _, m := range messages {
			b.WriteString(m.Content)
		}
		return b.String()
	case cfg.IsDeepSeekV41():
		// The template carries response_format itself; a forced tool keeps the
		// shared instruction and drops the schemas, as the ChatML path does.
		if forced := inKernelForcedToolInstruction(sp.ToolChoice, tools); forced != "" {
			messages = append([]Message{{Role: RoleSystem, Content: forced}}, messages...)
		}
		if inKernelEffectiveToolName(sp.ToolChoice, tools) != "" {
			tools = nil
		}
		return renderDeepSeekV41Chat(messages, tools, sp.ResponseFormat)
	default:
		return renderInKernelChatMLRequest(messages, tools, cfg, sp.ResponseFormat, sp.ToolChoice, sp)
	}
}

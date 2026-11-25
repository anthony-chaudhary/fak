package agent

import (
	"bytes"
	"encoding/json"
	"strings"
)

// Inkling-2 writes one block per tool call (renderInkling2Chat renders the same
// form for history):
//
//	<|message_model|>NAME<|content_invoke_tool_json|>{"name":"NAME","args":{...}}<|end_message|>
//
// The header and trailing <|end_message|> are optional on lift: a split turn may
// already have dropped them, and a stop on the end-of-turn marker can cut the
// final <|end_message|>. The JSON payload must still be complete.
func extractInkling2ToolBlocks(content string) []liftedBlock {
	var out []liftedBlock
	for cursor := 0; cursor < len(content); {
		rel := strings.Index(content[cursor:], inkling2InvokeToolJSON)
		if rel < 0 {
			break
		}
		marker := cursor + rel
		payloadStart := marker + len(inkling2InvokeToolJSON)
		start, headerName := inkling2CallHeader(content[cursor:marker])
		start += cursor

		dec := json.NewDecoder(strings.NewReader(content[payloadStart:]))
		var payload struct {
			Name string          `json:"name"`
			Args json.RawMessage `json:"args"`
		}
		if err := dec.Decode(&payload); err != nil {
			cursor = payloadStart
			continue
		}
		end := payloadStart + int(dec.InputOffset())
		if tail := content[end:]; strings.HasPrefix(strings.TrimLeft(tail, " \t\r\n"), inkling2EndMessage) {
			end += strings.Index(tail, inkling2EndMessage) + len(inkling2EndMessage)
		}
		cursor = end

		name := strings.TrimSpace(payload.Name)
		if name == "" {
			name = headerName
		}
		// No example-prose gate: the invoke marker is a special token, so prose
		// such as "mock" or an "examples/" arg cannot fake or suppress a call.
		if name == "" || (headerName != "" && headerName != name) {
			continue
		}
		args := bytes.TrimSpace(payload.Args)
		if len(args) == 0 || bytes.Equal(args, []byte("null")) {
			args = []byte("{}")
		}
		var stringified string
		if args[0] == '"' && json.Unmarshal(args, &stringified) == nil {
			args = []byte(stringified)
		}
		out = append(out, liftedBlock{start: start, end: end, call: ToolCall{
			Type:     "function",
			Function: Func{Name: name, Arguments: string(args)},
		}})
	}
	return out
}

// inkling2CallHeader finds the <|message_model|>NAME header that directly
// precedes an invoke marker, returning where the block starts and the name. With
// no header the block starts at the marker itself.
func inkling2CallHeader(before string) (int, string) {
	i := strings.LastIndex(before, inkling2RoleModel)
	if i < 0 {
		return len(before), ""
	}
	name := before[i+len(inkling2RoleModel):]
	if strings.Contains(name, "<|") || strings.ContainsAny(name, "\r\n") {
		return len(before), ""
	}
	return i, strings.TrimSpace(name)
}

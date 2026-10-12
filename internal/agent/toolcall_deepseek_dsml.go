package agent

import (
	"bytes"
	"encoding/json"
	"strings"
)

// DeepSeek V4.1's tool-call markup, as its GGUF chat template teaches it
// (renderDeepSeekV41Chat renders the same form for history):
//
//	<｜DSML｜tool_calls>
//	<｜DSML｜invoke name="get_weather">
//	<｜DSML｜parameter name="city" string="true">Paris</｜DSML｜parameter>
//	<｜DSML｜parameter name="days" string="false">3</｜DSML｜parameter>
//	</｜DSML｜invoke>
//	</｜DSML｜tool_calls>
//
// string="true" values are raw text; every other value is JSON.
const (
	deepSeekDSMLCallsOpen  = "<" + deepSeekV41DSML + "tool_calls>"
	deepSeekDSMLCallsClose = "</" + deepSeekV41DSML + "tool_calls>"
	deepSeekDSMLInvokeOpen = "<" + deepSeekV41DSML + "invoke name=\""
	deepSeekDSMLInvokeEnd  = "</" + deepSeekV41DSML + "invoke>"
	deepSeekDSMLParamOpen  = "<" + deepSeekV41DSML + "parameter name=\""
	deepSeekDSMLParamEnd   = "</" + deepSeekV41DSML + "parameter>"
)

// extractDeepSeekDSMLBlocks lifts each invoke of every complete tool_calls block.
// As for a Mistral array, the first call spans the whole wrapper and the rest are
// empty at its end, so the lift decides and strips the group together. A block
// with any malformed invoke, or one introduced as an example, stays in the text.
func extractDeepSeekDSMLBlocks(content string) []liftedBlock {
	var out []liftedBlock
	for cursor := 0; cursor < len(content); {
		startRel := strings.Index(content[cursor:], deepSeekDSMLCallsOpen)
		if startRel < 0 {
			break
		}
		start := cursor + startRel
		bodyStart := start + len(deepSeekDSMLCallsOpen)
		closeRel := strings.Index(content[bodyStart:], deepSeekDSMLCallsClose)
		if closeRel < 0 {
			break
		}
		end := bodyStart + closeRel + len(deepSeekDSMLCallsClose)
		cursor = end
		if isPrecededByExampleProse(content[:start]) {
			continue
		}
		calls, ok := parseDeepSeekDSMLInvokes(content[bodyStart : bodyStart+closeRel])
		if !ok || len(calls) == 0 {
			continue
		}
		for i, call := range calls {
			block := liftedBlock{start: start, end: end, call: call}
			if i > 0 {
				block.start = end
			}
			out = append(out, block)
		}
	}
	return out
}

func parseDeepSeekDSMLInvokes(body string) ([]ToolCall, bool) {
	var calls []ToolCall
	rest := body
	for {
		rest = strings.TrimSpace(rest)
		if rest == "" {
			return calls, true
		}
		if !strings.HasPrefix(rest, deepSeekDSMLInvokeOpen) {
			return nil, false
		}
		rest = rest[len(deepSeekDSMLInvokeOpen):]
		nameEnd := strings.Index(rest, "\">")
		if nameEnd <= 0 {
			return nil, false
		}
		name := rest[:nameEnd]
		rest = rest[nameEnd+2:]
		invokeEnd := strings.Index(rest, deepSeekDSMLInvokeEnd)
		if invokeEnd < 0 {
			return nil, false
		}
		args, ok := parseDeepSeekDSMLParameters(rest[:invokeEnd])
		if !ok {
			return nil, false
		}
		calls = append(calls, ToolCall{Type: "function", Function: Func{Name: name, Arguments: args}})
		rest = rest[invokeEnd+len(deepSeekDSMLInvokeEnd):]
	}
}

// parseDeepSeekDSMLParameters returns the invoke's arguments as a JSON object in
// source order: string="true" values become JSON strings, the rest must be JSON.
func parseDeepSeekDSMLParameters(body string) (string, bool) {
	var b bytes.Buffer
	b.WriteByte('{')
	rest := body
	for n := 0; ; n++ {
		rest = strings.TrimLeft(rest, " \t\r\n")
		if rest == "" {
			break
		}
		if !strings.HasPrefix(rest, deepSeekDSMLParamOpen) {
			return "", false
		}
		rest = rest[len(deepSeekDSMLParamOpen):]
		nameEnd := strings.Index(rest, "\" string=\"")
		if nameEnd <= 0 {
			return "", false
		}
		name := rest[:nameEnd]
		rest = rest[nameEnd+len("\" string=\""):]
		kindEnd := strings.Index(rest, "\">")
		if kindEnd < 0 {
			return "", false
		}
		kind := rest[:kindEnd]
		rest = rest[kindEnd+2:]
		valueEnd := strings.Index(rest, deepSeekDSMLParamEnd)
		if valueEnd < 0 {
			return "", false
		}
		value := rest[:valueEnd]
		rest = rest[valueEnd+len(deepSeekDSMLParamEnd):]

		key, _ := json.Marshal(name)
		var encoded []byte
		switch kind {
		case "true":
			encoded, _ = json.Marshal(value)
		case "false":
			var compact bytes.Buffer
			if err := json.Compact(&compact, []byte(strings.TrimSpace(value))); err != nil {
				return "", false
			}
			encoded = compact.Bytes()
		default:
			return "", false
		}
		if n > 0 {
			b.WriteByte(',')
		}
		b.Write(key)
		b.WriteByte(':')
		b.Write(encoded)
	}
	b.WriteByte('}')
	return b.String(), true
}

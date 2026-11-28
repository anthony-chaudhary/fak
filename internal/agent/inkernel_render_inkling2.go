package agent

import (
	"bytes"
	"encoding/json"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"
)

// Inkling2EndOfTurnMarker ends an Inkling-2 assistant turn. Decode must stop
// here, not on <|end_message|>: that token closes every block, including the
// thinking block and each of several tool calls in one turn.
const Inkling2EndOfTurnMarker = "<|content_model_end_sampling|>"

const (
	inkling2RoleSystem      = "<|message_system|>"
	inkling2RoleUser        = "<|message_user|>"
	inkling2RoleModel       = "<|message_model|>"
	inkling2RoleTool        = "<|message_tool|>"
	inkling2ContentText     = "<|content_text|>"
	inkling2ContentThinking = "<|content_thinking|>"
	inkling2ContentXML      = "<|content_xml|>"
	inkling2InvokeToolJSON  = "<|content_invoke_tool_json|>"
	inkling2EndMessage      = "<|end_message|>"
)

var inkling2RoleToken = map[string]string{
	RoleSystem:    inkling2RoleSystem,
	RoleUser:      inkling2RoleUser,
	RoleAssistant: inkling2RoleModel,
	RoleTool:      inkling2RoleTool,
}

// inkling2EffortLevel maps the template's named levels, plus fak's own tiers,
// onto the numeric effort the template prints. Unknown names fall back to none
// rather than raising as the template does: a serve request must not fail on a
// tier name another family accepts.
var inkling2EffortLevel = map[string]float64{
	"none":              0,
	"minimal":           0.1,
	EffortTierFast:      0.1,
	EffortTierLow:       0.2,
	EffortTierMedium:    0.7,
	EffortTierBalanced:  0.7,
	EffortTierAdaptive:  0.7,
	EffortTierStandard:  0.7,
	EffortTierHigh:      0.9,
	"xhigh":             0.99,
	"max":               0.99,
	EffortTierRigor:     0.99,
	EffortTierUltra:     0.99,
	EffortTierUltracode: 0.99,
}

// inkling2Effort is the number the effort preamble prints. The template defaults
// to 0.9; fak defaults to none so an unset request does not spend its token
// budget thinking.
func inkling2Effort(sp SampleParams) string {
	num := 0.0
	effort := strings.ToLower(strings.TrimSpace(sp.ReasoningEffort))
	switch {
	case effort != "":
		if v, ok := inkling2EffortLevel[effort]; ok {
			num = v
		} else if f, err := strconv.ParseFloat(effort, 64); err == nil && f >= 0 && f <= 0.99 {
			num = f
		}
	case sp.ThinkingBudget != nil && *sp.ThinkingBudget > 0:
		num = 0.9
	}
	if num == 0 {
		return "0"
	}
	return strconv.FormatFloat(num, 'f', -1, 64)
}

// inkling2ChatRequest applies tool_choice and response_format the way the ChatML
// path does: the instruction rides as a leading system message and a forced
// tool drops the declarations.
func inkling2ChatRequest(messages []Message, tools []ToolDef, sp SampleParams) ([]Message, []ToolDef) {
	instruction := inKernelResponseFormatInstruction(sp.ResponseFormat)
	if forced := inKernelForcedToolInstruction(sp.ToolChoice, tools); forced != "" {
		if instruction != "" {
			instruction += "\n"
		}
		instruction += forced
	}
	if instruction != "" {
		messages = append([]Message{{Role: RoleSystem, Content: instruction}}, messages...)
	}
	if inKernelEffectiveToolName(sp.ToolChoice, tools) != "" {
		tools = nil
	}
	return messages, tools
}

// inkling2PrefillsAnswerHeader reports whether renderInkling2Chat should append
// the assistant text-block header directly after the prompt. When a caller
// explicitly asks for reasoning_effort="none" and no tools are offered,
// prefilling <|message_model|><|content_text|> keeps the model from opening a
// hidden <|content_thinking|> block and burning max_tokens before any visible
// answer is emitted. When tools are offered, the header is left open so the
// model can still emit <|message_model|><tool><|content_invoke_tool_json|>.
func inkling2PrefillsAnswerHeader(tools []ToolDef, sp SampleParams) bool {
	return strings.EqualFold(strings.TrimSpace(sp.ReasoningEffort), "none") && len(tools) == 0
}

// The template has no generation prompt: the model writes its own
// <|message_model|> header, so the prompt ends on the last closed block, except
// when effort=none with no tools prefills the visible answer header.
func renderInkling2Chat(messages []Message, tools []ToolDef, sp SampleParams) string {
	messages, tools = inkling2ChatRequest(messages, tools, sp)
	var b strings.Builder
	if len(tools) > 0 {
		b.WriteString(inkling2RoleSystem + "tool_declare" + inkling2ContentXML)
		b.WriteString(inkling2ToolSpecs(tools))
		b.WriteString(inkling2EndMessage)
	}
	effort := inkling2RoleSystem + inkling2ContentText + "Thinking effort level: " + inkling2Effort(sp) + inkling2EndMessage
	effortEmitted := false
	for idx, m := range messages {
		rtok, ok := inkling2RoleToken[m.Role]
		if !ok {
			// The template raises on an unknown role; a user turn keeps the text visible.
			rtok = inkling2RoleUser
		}
		if !effortEmitted && m.Role != RoleSystem {
			b.WriteString(effort)
			effortEmitted = true
		}
		switch {
		case m.Role == RoleTool && m.Name != "":
			rtok += m.Name
		case m.Role == RoleTool && m.ToolCallID != "":
			rtok += inkling2ToolNameFor(messages, idx, m.ToolCallID)
		case m.Role == RoleAssistant && m.Name != "":
			rtok += m.Name
		}
		calls := m.ToolCalls
		if len(calls) == 0 && m.FunctionCall != nil {
			calls = []ToolCall{{Type: "function", Function: *m.FunctionCall}}
		}
		if m.Role == RoleAssistant && m.ReasoningContent != "" {
			b.WriteString(rtok + inkling2ContentThinking + m.ReasoningContent + inkling2EndMessage)
		}
		if m.Content != "" || !(m.Role == RoleAssistant && (len(calls) > 0 || m.ReasoningContent != "")) {
			b.WriteString(rtok + inkling2ContentText + m.Content + inkling2EndMessage)
		}
		if m.Role == RoleAssistant {
			for _, tc := range calls {
				inkling2WriteToolCall(&b, tc.Function)
			}
			b.WriteString(Inkling2EndOfTurnMarker)
		}
	}
	if !effortEmitted {
		b.WriteString(effort)
	}
	if inkling2PrefillsAnswerHeader(tools, sp) {
		b.WriteString(inkling2RoleModel + inkling2ContentText)
	}
	return b.String()
}

// inkling2ToolNameFor resolves a tool result's header name by searching
// backward from idx-1 down to 0 for the nearest preceding assistant tool call
// matching id. Returning on the first match prevents cross-turn concatenation
// when a client or fallback reuses tool-call IDs, keeping earlier turns'
// rendered bytes invariant for KV prefix-cache reuse.
func inkling2ToolNameFor(messages []Message, idx int, id string) string {
	if idx > len(messages) {
		idx = len(messages)
	}
	for i := idx - 1; i >= 0; i-- {
		prev := messages[i]
		if prev.Role != RoleAssistant {
			continue
		}
		for j := len(prev.ToolCalls) - 1; j >= 0; j-- {
			tc := prev.ToolCalls[j]
			if tc.ID == id && tc.Function.Name != "" {
				return tc.Function.Name
			}
		}
	}
	return ""
}

func inkling2WriteToolCall(b *strings.Builder, fn Func) {
	b.WriteString(inkling2RoleModel + fn.Name + inkling2InvokeToolJSON)
	b.WriteString(`{"name":`)
	b.WriteString(inkling2Quote(fn.Name))
	b.WriteString(`,"args":`)
	b.WriteString(inkling2ToolArgs(fn.Arguments))
	b.WriteString("}" + inkling2EndMessage)
}

// inkling2ToolArgs renders arguments as the template does: top-level keys in
// byte order, each value compact with its own key order kept. Arguments that
// are not an object stay verbatim; the template would refuse them, and dropping
// them would hide what the model actually called.
func inkling2ToolArgs(raw string) string {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return "{}"
	}
	node, ok := parseInkling2JSON(raw)
	if !ok || node.kind != inkling2JSONObject {
		return raw
	}
	sorted := node
	sorted.fields = append([]inkling2JSONField(nil), node.fields...)
	sort.SliceStable(sorted.fields, func(i, j int) bool { return sorted.fields[i].key < sorted.fields[j].key })
	var b strings.Builder
	sorted.write(&b, false)
	return b.String()
}

// inkling2ToolSpecs is the tool_declare payload: tojson(sort_keys=true) over
// {description,name,parameters,type}, sorted at every depth.
func inkling2ToolSpecs(tools []ToolDef) string {
	var b strings.Builder
	b.WriteByte('[')
	for i, tool := range tools {
		if i > 0 {
			b.WriteByte(',')
		}
		params := inkling2JSONNode{kind: inkling2JSONObject}
		if node, ok := parseInkling2JSON(string(tool.Function.Parameters)); ok && node.truthy() {
			params = node
		}
		typ := tool.Type
		if typ == "" {
			typ = "function"
		}
		b.WriteString(`{"description":`)
		b.WriteString(inkling2Quote(tool.Function.Description))
		b.WriteString(`,"name":`)
		b.WriteString(inkling2Quote(tool.Function.Name))
		b.WriteString(`,"parameters":`)
		params.write(&b, true)
		b.WriteString(`,"type":`)
		b.WriteString(inkling2Quote(typ))
		b.WriteByte('}')
	}
	b.WriteByte(']')
	return b.String()
}

type inkling2JSONKind int

const (
	inkling2JSONLiteral inkling2JSONKind = iota
	inkling2JSONString
	inkling2JSONObject
	inkling2JSONArray
)

type inkling2JSONField struct {
	key   string
	value inkling2JSONNode
}

// inkling2JSONNode keeps object key order, which Go maps lose; the template
// sorts the top-level args but keeps nested argument objects in source order.
type inkling2JSONNode struct {
	kind   inkling2JSONKind
	text   string // literal text, or the decoded string
	fields []inkling2JSONField
	items  []inkling2JSONNode
}

func parseInkling2JSON(raw string) (inkling2JSONNode, bool) {
	dec := json.NewDecoder(strings.NewReader(raw))
	dec.UseNumber()
	node, ok := decodeInkling2JSON(dec)
	if !ok || dec.More() {
		return inkling2JSONNode{}, false
	}
	if _, err := dec.Token(); err == nil {
		return inkling2JSONNode{}, false
	}
	return node, true
}

func decodeInkling2JSON(dec *json.Decoder) (inkling2JSONNode, bool) {
	tok, err := dec.Token()
	if err != nil {
		return inkling2JSONNode{}, false
	}
	switch v := tok.(type) {
	case json.Delim:
		switch v {
		case '{':
			node := inkling2JSONNode{kind: inkling2JSONObject}
			for dec.More() {
				keyTok, err := dec.Token()
				key, isString := keyTok.(string)
				if err != nil || !isString {
					return inkling2JSONNode{}, false
				}
				value, ok := decodeInkling2JSON(dec)
				if !ok {
					return inkling2JSONNode{}, false
				}
				node.fields = append(node.fields, inkling2JSONField{key: key, value: value})
			}
			_, err := dec.Token()
			return node, err == nil
		case '[':
			node := inkling2JSONNode{kind: inkling2JSONArray}
			for dec.More() {
				item, ok := decodeInkling2JSON(dec)
				if !ok {
					return inkling2JSONNode{}, false
				}
				node.items = append(node.items, item)
			}
			_, err := dec.Token()
			return node, err == nil
		}
		return inkling2JSONNode{}, false
	case string:
		return inkling2JSONNode{kind: inkling2JSONString, text: v}, true
	case json.Number:
		return inkling2JSONNode{kind: inkling2JSONLiteral, text: v.String()}, true
	case bool:
		return inkling2JSONNode{kind: inkling2JSONLiteral, text: strconv.FormatBool(v)}, true
	case nil:
		return inkling2JSONNode{kind: inkling2JSONLiteral, text: "null"}, true
	}
	return inkling2JSONNode{}, false
}

// truthy mirrors Jinja truthiness for the template's `fn.parameters` test.
func (n inkling2JSONNode) truthy() bool {
	switch n.kind {
	case inkling2JSONObject:
		return len(n.fields) > 0
	case inkling2JSONArray:
		return len(n.items) > 0
	case inkling2JSONString:
		return n.text != ""
	case inkling2JSONLiteral:
		return n.text != "null" && n.text != "false" && n.text != "0"
	}
	return false
}

func (n inkling2JSONNode) write(b *strings.Builder, sortKeys bool) {
	switch n.kind {
	case inkling2JSONString:
		b.WriteString(inkling2Quote(n.text))
	case inkling2JSONObject:
		fields := n.fields
		if sortKeys {
			fields = append([]inkling2JSONField(nil), fields...)
			sort.SliceStable(fields, func(i, j int) bool { return fields[i].key < fields[j].key })
		}
		b.WriteByte('{')
		for i, f := range fields {
			if i > 0 {
				b.WriteByte(',')
			}
			b.WriteString(inkling2Quote(f.key))
			b.WriteByte(':')
			f.value.write(b, sortKeys)
		}
		b.WriteByte('}')
	case inkling2JSONArray:
		b.WriteByte('[')
		for i, item := range n.items {
			if i > 0 {
				b.WriteByte(',')
			}
			item.write(b, sortKeys)
		}
		b.WriteByte(']')
	default:
		b.WriteString(n.text)
	}
}

// inkling2Quote quotes s like the template's tojson (json.dumps with
// ensure_ascii=False): non-ASCII and <>& stay literal, unlike encoding/json.
func inkling2Quote(s string) string {
	var b bytes.Buffer
	b.WriteByte('"')
	for i := 0; i < len(s); {
		c := s[i]
		if c >= utf8.RuneSelf {
			r, size := utf8.DecodeRuneInString(s[i:])
			if r == utf8.RuneError && size == 1 {
				b.WriteRune(utf8.RuneError)
			} else {
				b.WriteString(s[i : i+size])
			}
			i += size
			continue
		}
		switch c {
		case '"':
			b.WriteString(`\"`)
		case '\\':
			b.WriteString(`\\`)
		case '\n':
			b.WriteString(`\n`)
		case '\r':
			b.WriteString(`\r`)
		case '\t':
			b.WriteString(`\t`)
		case '\b':
			b.WriteString(`\b`)
		case '\f':
			b.WriteString(`\f`)
		default:
			if c < 0x20 {
				b.WriteString(`\u00`)
				b.WriteByte("0123456789abcdef"[c>>4])
				b.WriteByte("0123456789abcdef"[c&0xf])
			} else {
				b.WriteByte(c)
			}
		}
		i++
	}
	b.WriteByte('"')
	return b.String()
}

// SplitInkling2Output separates one decoded Inkling-2 assistant turn into
// reasoning (thinking blocks) and content (text blocks, with tool-call blocks
// kept verbatim so the inkling2 dialect lifts them). Text with no Inkling
// markers is returned unchanged as content.
func SplitInkling2Output(raw string) (reasoning, content string) {
	if !strings.Contains(raw, inkling2EndMessage) && !strings.Contains(raw, inkling2RoleModel) &&
		!strings.Contains(raw, inkling2ContentThinking) && !strings.Contains(raw, inkling2ContentText) &&
		!strings.Contains(raw, Inkling2EndOfTurnMarker) {
		return "", raw
	}
	rest := raw
	if i := strings.Index(rest, Inkling2EndOfTurnMarker); i >= 0 {
		rest = rest[:i]
	}
	var thoughts, texts []string
	var out strings.Builder
	for _, block := range strings.SplitAfter(rest, inkling2EndMessage) {
		body := strings.TrimPrefix(strings.TrimLeft(block, " \t\r\n"), inkling2RoleModel)
		closed := strings.HasSuffix(body, inkling2EndMessage)
		inner := strings.TrimSuffix(body, inkling2EndMessage)
		switch {
		case strings.HasPrefix(inner, inkling2ContentThinking):
			// An unclosed thinking block is a truncated turn: still reasoning.
			if t := strings.TrimSpace(strings.TrimPrefix(inner, inkling2ContentThinking)); t != "" {
				thoughts = append(thoughts, t)
			}
		case strings.HasPrefix(inner, inkling2ContentText):
			texts = append(texts, strings.TrimPrefix(inner, inkling2ContentText))
		case strings.Contains(inner, inkling2InvokeToolJSON):
			if len(texts) > 0 {
				out.WriteString(strings.Join(texts, "\n"))
				out.WriteByte('\n')
				texts = nil
			}
			out.WriteString(inkling2RoleModel + inner)
			if closed {
				out.WriteString(inkling2EndMessage)
			}
			out.WriteByte('\n')
		case strings.TrimSpace(inner) != "":
			texts = append(texts, inner)
		}
	}
	if len(texts) > 0 {
		out.WriteString(strings.Join(texts, "\n"))
	}
	return strings.TrimSpace(strings.Join(thoughts, "\n")), strings.TrimSpace(out.String())
}

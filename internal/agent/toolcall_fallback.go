package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// hermesToolCallRe matches a Hermes/Qwen-style tool call emitted as plain TEXT
// rather than through the provider's structured tool_calls channel:
//
//	<tool_call>{"name": "Bash", "arguments": {"command": "ls"}}</tool_call>
//
// Small local models (e.g. qwen2.5:1.5b) under a large multi-tool prompt
// intermittently fall back to this template instead of the structured field, and
// ollama's own parser only lifts the well-formed case — the rest lands in the
// content string. The capture group is the inner JSON object.
var hermesToolCallRe = regexp.MustCompile(`(?s)<tool_call>\s*(\{.*?\})\s*</tool_call>`)

// functionCallTagRe matches an XML-ish <function_call>{...}</function_call> block.
// Some shims (and a few fine-tunes) emit the OpenAI "function call" concept as a
// literal tag rather than the Hermes <tool_call> name. The payload is the same
// {"name","arguments"} / {"function":{...}} shape the Hermes extractor parses.
var functionCallTagRe = regexp.MustCompile(`(?s)<function_call>\s*(\{.*?\})\s*</function_call>`)

// llamaPythonTagRe matches Llama-3.1's <|python_tag|> tool-call template:
//
//	<|python_tag|>{"name": "Bash", "arguments": {"command": "ls"}}<|eom_id|>
//
// The JSON object runs to the first Llama control token (<|eom_id|>/<|eot_id|>)
// or to end-of-content if the model stopped without one. The capture group is the
// inner JSON; the trailing terminator (if present) is consumed so it is stripped
// from the content along with the call.
var llamaPythonTagRe = regexp.MustCompile(`(?s)<\|python_tag\|>\s*(\{.*?\})\s*(?:<\|eom_id\|>|<\|eot_id\|>|$)`)

// mistralToolCallsRe matches Mistral/Mixtral's [TOOL_CALLS][ ... ] template. The
// capture group is the JSON ARRAY of call objects (Mistral always emits an array,
// even for a single call), parsed by the array-aware extractor below.
var mistralToolCallsRe = regexp.MustCompile(`(?s)\[TOOL_CALLS\]\s*(\[.*\])`)

// fencedJSONRe matches a ```json … ```, ```tool_call … ```, or bare ``` … ``` fence.
// Group 1 captures the optional fence tag, and group 2 captures the fence body.
var fencedJSONRe = regexp.MustCompile("(?s)```(json|tool_call)?\\s*(\\{.*?\\}|\\[.*?\\])\\s*```")

// exampleProseRe matches prose phrases that introduce a tool call as an example
// rather than an executable instruction (#12042, #12067).
var exampleProseRe = regexp.MustCompile(`(?i)(?:here(?:'s|\s+is)\s+an?\s+example|for\s+example|\b(?:examples?|for\s+instance|illustrative|mock)\b|\b(?:sample\s+(?:tool\s+call|call|payload|request))\b|e\.g\.|eg\s*:)`)

// hermesToolCallPayload is the inner JSON of a text-embedded tool call. arguments
// is intentionally a RawMessage: models emit it as either a JSON object (Hermes)
// or an already-stringified JSON blob, and Func.Arguments wants the raw string.
type hermesToolCallPayload struct {
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
	Function  *Func           `json:"function"`
}

// liftedBlock is one tool call recovered from a span of content text: the
// [start,end) byte range the dialect occupied (stripped on lift) and the call it
// yields. Every dialect extractor returns these so the strip/assemble logic in
// LiftTextToolCalls is shared and identical across formats.
type liftedBlock struct {
	start int
	end   int
	call  ToolCall
}

// dialectExtractor finds every tool-call block of ONE text dialect in content and
// returns them in source order. It must be conservative: skip a malformed or
// nameless block (leave it in the text) rather than fabricate a call. A dialect
// that finds nothing returns nil.
type dialectExtractor struct {
	name    string
	extract func(content string) []liftedBlock
}

// toolCallDialects is the ordered registry of text-form tool-call dialects the
// fallback recognizes. Precedence is by descending delimiter specificity:
// explicit-tag / bracketed / fenced dialects first (unambiguous), then bare JSON
// last (the most ambiguous — only when it is the ENTIRE content). The FIRST
// dialect that yields ≥1 valid block wins; we never mix dialects within one
// message, so an overlapping match (e.g. a Hermes tag whose inner JSON also looks
// like bare JSON) is lifted exactly once, by the more specific dialect.
var toolCallDialects = []dialectExtractor{
	{name: "hermes", extract: extractDelimited(hermesToolCallRe)},
	{name: "function_call_tag", extract: extractDelimited(functionCallTagRe)},
	{name: "llama_python_tag", extract: extractDelimited(llamaPythonTagRe)},
	{name: "mistral_tool_calls", extract: extractArrayDelimited(mistralToolCallsRe)},
	// Bare antl sits ahead of fenced/bare JSON because its inner <parameter> values
	// may themselves contain JSON that the more ambiguous dialects would misread.
	{name: "qwen_function_parameter", extract: extractQwenFunctionBlocks},
	{name: "fenced_json", extract: extractFenced},
	{name: "bare_json", extract: extractBareJSON},
}

// OfferedTools is a request-owned snapshot of names offered before prompt pruning.
// Its zero value permits no text lifts. Structured provider calls are unaffected.
type OfferedTools struct {
	names     map[string]struct{}
	extension func(string) bool
}

func NewOfferedTools(tools []ToolDef) OfferedTools {
	o := OfferedTools{names: make(map[string]struct{}, len(tools))}
	for _, tool := range tools {
		if name := tool.Function.Name; name != "" {
			o.names[name] = struct{}{}
		}
	}
	return o
}

type textToolExtensionKey struct{}

// WithTextToolExtension binds gateway-owned served-name/alias recognition to this
// call. It cannot authorize a lift when the planner received no offered names.
// The caller must capture an immutable request snapshot, not shared mutable data.
func WithTextToolExtension(ctx context.Context, extension func(string) bool) context.Context {
	return context.WithValue(ctx, textToolExtensionKey{}, extension)
}

func offeredToolsFor(ctx context.Context, tools []ToolDef) OfferedTools {
	o := NewOfferedTools(tools)
	o.extension, _ = ctx.Value(textToolExtensionKey{}).(func(string) bool)
	return o
}

func (o OfferedTools) allows(name string) bool {
	if len(o.names) == 0 {
		return false
	}
	if _, ok := o.names[name]; ok {
		return true
	}
	return o.extension != nil && o.extension(name)
}

// LiftTextToolCalls promotes recognized text calls only within the request's
// offered set. Unoffered blocks remain byte-identical, including wrapped Qwen
// blocks and mixed-name arrays. No structured call is filtered here.
func LiftTextToolCalls(m Message, offered OfferedTools) Message {
	m, _ = liftTextToolCalls(m, offered)
	return m
}

func liftTextToolCalls(m Message, offered OfferedTools) (Message, bool) {
	if len(m.ToolCalls) > 0 || m.Content == "" {
		return m, false
	}
	content := normalizeQwenFunctionToolCalls(m.Content, offered)
	var blocks []liftedBlock
	for _, d := range toolCallDialects {
		if blocks = d.extract(content); len(blocks) > 0 {
			break
		}
	}
	// Array extractors assign a full span to their first call and zero-width
	// spans at its end to the rest. Decide the entire group before stripping any
	// bytes, so an unoffered member cannot be erased by an offered sibling.
	var selected []liftedBlock
	rejected := false
	for i := 0; i < len(blocks); {
		end := i + 1
		for end < len(blocks) && blocks[end].start == blocks[i].end && blocks[end].end == blocks[i].end {
			end++
		}
		allowed := true
		for _, block := range blocks[i:end] {
			if !offered.allows(block.call.Function.Name) {
				allowed = false
				rejected = true
			}
		}
		if allowed {
			selected = append(selected, blocks[i:end]...)
		}
		i = end
	}
	if len(selected) == 0 {
		return m, rejected
	}
	var calls []ToolCall
	var stripped strings.Builder
	last := 0
	for _, block := range selected {
		stripped.WriteString(content[last:block.start])
		last = block.end
		block.call.ID = fmt.Sprintf("call_text_%d", len(calls))
		calls = append(calls, block.call)
	}
	stripped.WriteString(content[last:])
	m.Content = strings.TrimSpace(stripped.String())
	m.ToolCalls = calls
	return m, rejected
}

// qwenFunctionParamRe locates Qwen's antl-style <function=name>…</function> block when
// it is emitted WITHOUT the <tool_call> wrapper. Qwen2.5-Coder-3B improvises exactly
// this shape as bare content under a tools-bearing prompt (issue #10600 captured
// `<function=bash>\n<parameter=command>…`); the wrapper-presence gate in
// normalizeQwenFunctionToolCalls never sees it there, so the call stayed text and never
// reached adjudication. parseQwenFunctionToolCall stays the authority on whether a
// located block is well-formed; a malformed one is left in the content untouched.
var qwenFunctionParamRe = regexp.MustCompile(`(?s)<function=[^>]*>.*?</function>`)

// extractQwenFunctionBlocks lifts bare <function=name><parameter=key>value blocks into
// structured calls. The wrapped form never reaches here: normalizeQwenFunctionToolCalls
// has already rewritten it into the hermes JSON form, and both paths share one parser,
// so a block malformed for one is malformed for the other (no double-lift, no husk).
func extractQwenFunctionBlocks(content string) []liftedBlock {
	matches := qwenFunctionParamRe.FindAllStringSubmatchIndex(content, -1)
	if len(matches) == 0 {
		return nil
	}
	var blocks []liftedBlock
	for _, loc := range matches {
		if isPrecededByExampleProse(content[:loc[0]]) {
			continue
		}
		name, args, ok := parseQwenFunctionToolCall(content[loc[0]:loc[1]])
		if !ok {
			continue
		}
		encoded, err := json.Marshal(args)
		if err != nil {
			continue
		}
		blocks = append(blocks, liftedBlock{
			start: loc[0],
			end:   loc[1],
			call:  ToolCall{Type: "function", Function: Func{Name: name, Arguments: string(encoded)}},
		})
	}
	return blocks
}

// liftPayload parses one inner JSON object as a {"name","arguments"} or
// {"function":{...}} tool call, applying the conservative posture shared by every
// dialect: a malformed or nameless payload yields ok=false (the caller leaves the
// block in the text rather than fabricate a call).

func normalizeQwenFunctionToolCalls(content string, offered OfferedTools) string {
	const open, close = "<tool_call>", "</tool_call>"
	var out strings.Builder
	for cursor := 0; cursor < len(content); {
		startRel := strings.Index(content[cursor:], open)
		if startRel < 0 {
			out.WriteString(content[cursor:])
			break
		}
		start := cursor + startRel
		out.WriteString(content[cursor:start])
		endRel := strings.Index(content[start+len(open):], close)
		if endRel < 0 {
			out.WriteString(content[start:])
			break
		}
		end := start + len(open) + endRel + len(close)
		block := content[start:end]
		if isPrecededByExampleProse(content[:start]) {
			out.WriteString(block)
			cursor = end
			continue
		}
		if name, args, ok := parseQwenFunctionToolCall(block); ok && offered.allows(name) {
			encoded, _ := json.Marshal(struct {
				Name      string         `json:"name"`
				Arguments map[string]any `json:"arguments"`
			}{name, args})
			out.WriteString(open + "\n" + string(encoded) + "\n" + close)
		} else {
			out.WriteString(block)
		}
		cursor = end
	}
	return out.String()
}

func parseQwenFunctionToolCall(block string) (string, map[string]any, bool) {
	fnStart := strings.Index(block, "<function=")
	if fnStart < 0 {
		return "", nil, false
	}
	nameEndRel := strings.Index(block[fnStart:], ">")
	if nameEndRel < 0 {
		return "", nil, false
	}
	nameEnd := fnStart + nameEndRel
	name := strings.TrimSpace(block[fnStart+len("<function=") : nameEnd])
	fnCloseRel := strings.Index(block[nameEnd+1:], "</function>")
	if name == "" || fnCloseRel < 0 {
		return "", nil, false
	}
	body := block[nameEnd+1 : nameEnd+1+fnCloseRel]
	args := map[string]any{}
	for {
		trimmed := strings.TrimSpace(body)
		if trimmed == "" {
			break
		}
		key, value, rest, ok := nextQwenParameter(trimmed)
		if !ok {
			// Non-empty body we cannot parse: fail closed instead of fabricating {}.
			return "", nil, false
		}
		args[key] = value
		body = rest
	}
	return name, args, true
}

// nextQwenParameter parses one argument from the front of body. Canonical form is
// <parameter=KEY>VALUE</parameter>; small distills also emit the shorthand
// <KEY>VALUE</parameter> or <KEY>VALUE</KEY>, which is accepted too.
func nextQwenParameter(body string) (key string, value any, rest string, ok bool) {
	if !strings.HasPrefix(body, "<") {
		return "", nil, "", false
	}
	tagEnd := strings.Index(body, ">")
	if tagEnd < 0 {
		return "", nil, "", false
	}
	tag := strings.TrimSpace(body[1:tagEnd])
	canonical := strings.HasPrefix(tag, "parameter=")
	if canonical {
		key = strings.TrimSpace(strings.TrimPrefix(tag, "parameter="))
	} else {
		key = tag
	}
	if key == "" || strings.ContainsAny(key, " /<=") || key == "function" || key == "tool_call" || key == "parameter" {
		return "", nil, "", false
	}
	after := body[tagEnd+1:]
	valueEnd, closeLen := -1, 0
	if canonical {
		// A canonical open closes on </parameter> so a value may itself contain
		// </KEY> (XML/HTML content); </KEY> is only the fallback closer.
		if i := strings.Index(after, "</parameter>"); i >= 0 {
			valueEnd, closeLen = i, len("</parameter>")
		} else if i := strings.Index(after, "</"+key+">"); i >= 0 {
			valueEnd, closeLen = i, len("</"+key+">")
		}
	} else {
		for _, c := range []string{"</parameter>", "</" + key + ">"} {
			if i := strings.Index(after, c); i >= 0 && (valueEnd < 0 || i < valueEnd) {
				valueEnd, closeLen = i, len(c)
			}
		}
	}
	if valueEnd < 0 {
		return "", nil, "", false
	}
	raw := strings.TrimSpace(after[:valueEnd])
	if json.Unmarshal([]byte(raw), &value) != nil {
		value = raw
	}
	return key, value, after[valueEnd+closeLen:], true
}

func liftPayload(inner string) (ToolCall, bool) {
	var p hermesToolCallPayload
	if err := json.Unmarshal([]byte(inner), &p); err != nil {
		return ToolCall{}, false
	}
	name := p.Name
	args := normalizeToolArguments(p.Arguments)
	if name == "" && p.Function != nil {
		name = p.Function.Name
		args = p.Function.Arguments
		if strings.TrimSpace(args) == "" {
			args = "{}"
		}
	}
	if name == "" {
		return ToolCall{}, false
	}
	return ToolCall{Type: "function", Function: Func{Name: name, Arguments: args}}, true
}

// extractDelimited builds an extractor for a dialect whose regex captures ONE
// inner JSON object per match (Hermes, function_call tag, Llama python_tag). The
// stripped span is the whole match (group 0), so the delimiters go with the call.
func extractDelimited(re *regexp.Regexp) func(string) []liftedBlock {
	return func(content string) []liftedBlock {
		matches := re.FindAllStringSubmatchIndex(content, -1)
		if len(matches) == 0 {
			return nil
		}
		var blocks []liftedBlock
		for _, loc := range matches {
			if isPrecededByExampleProse(content[:loc[0]]) {
				continue
			}
			blocks = appendLiftedSpan(blocks, content[loc[2]:loc[3]], loc[0], loc[1])
		}
		return blocks
	}
}

// appendLiftedSpan lifts payload (the inner JSON of one match) into a name-bearing call and,
// on success, appends a liftedBlock covering the whole match span [start,end); a non-liftable
// payload is left untouched. Shared by the delimited and fenced single-object extractors.
func appendLiftedSpan(blocks []liftedBlock, payload string, start, end int) []liftedBlock {
	call, ok := liftPayload(payload)
	if !ok {
		return blocks
	}
	return append(blocks, liftedBlock{start: start, end: end, call: call})
}

// extractArrayDelimited builds an extractor for a dialect whose regex captures a
// JSON ARRAY of call objects in one match (Mistral [TOOL_CALLS][...]). The whole
// match is stripped; every well-formed array element becomes a call, malformed or
// nameless elements are skipped. If NO element is liftable the whole block is left
// untouched (don't strip a [TOOL_CALLS] marker we couldn't actually parse).
func extractArrayDelimited(re *regexp.Regexp) func(string) []liftedBlock {
	return func(content string) []liftedBlock {
		loc := re.FindStringSubmatchIndex(content)
		if loc == nil {
			return nil
		}
		if isPrecededByExampleProse(content[:loc[0]]) {
			return nil
		}
		var raws []json.RawMessage
		if err := json.Unmarshal([]byte(content[loc[2]:loc[3]]), &raws); err != nil {
			return nil
		}
		return arrayLiftedBlocks(raws, loc[0], loc[1])
	}
}

// arrayLiftedBlocks lifts a JSON array of call objects into liftedBlocks that all
// collapse to a single stripped span [start,end): the first liftable element carries
// the full span so the array marker is stripped exactly once, every later element is a
// zero-width block at end so it adds no further strip range. Nameless / malformed
// elements are skipped; nil when none lift. Shared by the [TOOL_CALLS], fenced, and
// bare-JSON array paths.
func arrayLiftedBlocks(raws []json.RawMessage, start, end int) []liftedBlock {
	var blocks []liftedBlock
	for _, raw := range raws {
		call, ok := liftPayload(string(raw))
		if !ok {
			continue
		}
		if len(blocks) == 0 {
			blocks = append(blocks, liftedBlock{start: start, end: end, call: call})
		} else {
			blocks = append(blocks, liftedBlock{start: end, end: end, call: call})
		}
	}
	return blocks
}

var precedingBlockTerminators = []string{
	"```",
	"</tool_call>",
	"</function_call>",
	"</function>",
	"<|eom_id|>",
	"<|eot_id|>",
}

// isPrecededByExampleProse reports whether the preceding text indicates the following
// tool call block is an illustrative example rather than an intended execution (#12042, #12067).
func isPrecededByExampleProse(preceding string) bool {
	lastEnd := -1
	for _, term := range precedingBlockTerminators {
		if idx := strings.LastIndex(preceding, term); idx >= 0 {
			end := idx + len(term)
			if end > lastEnd {
				lastEnd = end
			}
		}
	}
	if lastEnd >= 0 {
		preceding = preceding[lastEnd:]
	}
	return exampleProseRe.MatchString(preceding)
}

// extractFenced lifts a tool call emitted inside a ```json … ```, ```tool_call … ```,
// or bare ``` … ``` fence. A fence body is lifted ONLY when it parses as a name-bearing
// tool-call object (or an array of them). If the fence tag is not "tool_call" and the
// preceding text is example prose, the block is left as content text (#12042).
// The whole fence (group 0) is stripped when lifted.
func extractFenced(content string) []liftedBlock {
	matches := fencedJSONRe.FindAllStringSubmatchIndex(content, -1)
	if len(matches) == 0 {
		return nil
	}
	var blocks []liftedBlock
	for _, loc := range matches {
		tag := ""
		if loc[2] >= 0 && loc[3] >= 0 {
			tag = content[loc[2]:loc[3]]
		}
		if tag != "tool_call" && isPrecededByExampleProse(content[:loc[0]]) {
			continue
		}
		body := strings.TrimSpace(content[loc[4]:loc[5]])
		if strings.HasPrefix(body, "[") {
			var raws []json.RawMessage
			if err := json.Unmarshal([]byte(body), &raws); err != nil {
				continue
			}
			blocks = append(blocks, arrayLiftedBlocks(raws, loc[0], loc[1])...)
			continue
		}
		blocks = appendLiftedSpan(blocks, body, loc[0], loc[1])
	}
	return blocks
}

// extractBareJSON lifts a tool call when the ENTIRE trimmed content is a single
// JSON tool-call object (or array of them) with no delimiter at all. This is the
// most ambiguous dialect, so it is gated hardest: it fires only when the whole
// message is the JSON (a model that emitted a call as its complete answer), never
// on JSON embedded in prose — that would risk lifting an example the model is
// merely discussing. On a lift the whole content is consumed.
func extractBareJSON(content string) []liftedBlock {
	trimmed := strings.TrimSpace(content)
	if trimmed == "" {
		return nil
	}
	start := strings.Index(content, trimmed[:1])
	end := start + len(trimmed)
	if strings.HasPrefix(trimmed, "[") {
		var raws []json.RawMessage
		if err := json.Unmarshal([]byte(trimmed), &raws); err != nil {
			return nil
		}
		return arrayLiftedBlocks(raws, start, end)
	}
	if !strings.HasPrefix(trimmed, "{") {
		return nil
	}
	call, ok := liftPayload(trimmed)
	if !ok {
		return nil
	}
	return []liftedBlock{{start: start, end: end, call: call}}
}

func normalizeCompletionToolCalls(comp *Completion, offered OfferedTools) *Completion {
	if comp == nil {
		return nil
	}
	// The upstream's RAW finish reason, before we possibly rewrite it below. If it
	// announced tool calls but none survive parsing + the text-lift fallback, that
	// is a conformance failure, not an empty turn (see Completion.ToolCallsDropped).
	rawClaimedToolCalls := finishReasonClaimsToolCalls(comp.FinishReason)

	var rejected bool
	comp.Message, rejected = liftTextToolCalls(comp.Message, offered)
	normalizeToolCallFields(&comp.Message)
	if len(comp.Message.ToolCalls) > 0 {
		comp.FinishReason = "tool_calls"
	} else if rawClaimedToolCalls {
		comp.ToolCallsDropped = true
		if rejected {
			comp.ToolCallsDroppedReason = abi.ReasonUnknownTool
		}
	}
	return comp
}

func normalizeCompletionFields(comp *Completion) *Completion {
	if comp == nil {
		return nil
	}
	normalizeToolCallFields(&comp.Message)
	if len(comp.Message.ToolCalls) > 0 {
		comp.FinishReason = "tool_calls"
	}
	return comp
}

// finishReasonClaimsToolCalls reports whether a raw OpenAI-family finish_reason
// indicates the model intended to call a tool. Both the modern "tool_calls" and
// the legacy "function_call" forms count; matching is case/space-insensitive so a
// provider variant ("Tool_Calls", " tool_calls ") is still recognized.
func finishReasonClaimsToolCalls(raw string) bool {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "tool_calls", "function_call":
		return true
	}
	return false
}

func normalizeToolCallFields(m *Message) {
	if m == nil || len(m.ToolCalls) == 0 {
		return
	}
	used := make(map[string]bool, len(m.ToolCalls))
	first := make(map[string]bool, len(m.ToolCalls))
	for _, tc := range m.ToolCalls {
		id := strings.TrimSpace(tc.ID)
		if id != "" {
			used[id] = true
		}
	}
	for i := range m.ToolCalls {
		tc := &m.ToolCalls[i]
		id := strings.TrimSpace(tc.ID)
		switch {
		case id == "":
			tc.ID = nextGeneratedToolCallID(i, used)
		case first[id]:
			tc.ID = nextGeneratedToolCallID(i, used)
		default:
			first[id] = true
		}
		used[tc.ID] = true
		if strings.TrimSpace(tc.Type) == "" {
			tc.Type = "function"
		}
	}
}

func nextGeneratedToolCallID(index int, used map[string]bool) string {
	for i := index; ; i++ {
		id := fmt.Sprintf("call_fak_%d", i)
		if !used[id] {
			return id
		}
	}
}

// normalizeToolArguments renders the inner `arguments` value as the raw JSON string
// Func.Arguments expects. A JSON object stays as compact JSON; an already-quoted
// JSON string is unquoted to its underlying value (so {"arguments":"{\"x\":1}"}
// and {"arguments":{"x":1}} both yield `{"x":1}`). Anything else is passed through
// as-is. An empty/absent value becomes "{}".
func normalizeToolArguments(raw json.RawMessage) string {
	s := strings.TrimSpace(string(raw))
	if s == "" || s == "null" {
		return "{}"
	}
	if strings.HasPrefix(s, "\"") {
		var unquoted string
		if err := json.Unmarshal(raw, &unquoted); err == nil {
			return unquoted
		}
	}
	return s
}

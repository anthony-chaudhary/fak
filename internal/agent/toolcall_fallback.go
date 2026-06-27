package agent

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
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

// fencedJSONRe matches a ```json … ``` fence (or a bare ``` … ``` fence). The
// capture group is the fence body, which the fenced extractor then parses as a
// single call object or an array of them. A fence whose body is not a name-bearing
// tool-call object is left untouched (it is ordinary fenced output, not a call).
var fencedJSONRe = regexp.MustCompile("(?s)```(?:json)?\\s*(\\{.*?\\}|\\[.*?\\])\\s*```")

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
	{name: "fenced_json", extract: extractFenced},
	{name: "bare_json", extract: extractBareJSON},
}

// LiftTextToolCalls promotes tool calls that a model emitted as TEXT — in any of
// the dialects in toolCallDialects (Hermes <tool_call>, XML <function_call>,
// Llama <|python_tag|>, Mistral [TOOL_CALLS], fenced ```json, or a bare JSON
// object) — into structured Message.ToolCalls, stripping the recovered spans from
// the content. It is a no-op when the message already carries structured ToolCalls
// (the provider parsed them) or when no dialect yields a well-formed call.
//
// This matters for more than weak-model ergonomics: the gateway adjudicates only
// STRUCTURED tool calls (s.adjudicateProposed reads Message.ToolCalls), so a call
// left as content text would bypass the kernel boundary entirely — silently
// breaking the "every proposed call is adjudicated" guarantee. Every un-recognized
// dialect is therefore a silent adjudication bypass; lifting it here puts the call
// back in front of the kernel.
func LiftTextToolCalls(m Message) Message {
	// If the provider already gave us structured calls, trust them — don't
	// double-count a model that emitted both (the structured one is authoritative).
	if len(m.ToolCalls) > 0 || m.Content == "" {
		return m
	}

	// First dialect that recovers at least one valid call wins (precedence order).
	var blocks []liftedBlock
	for _, d := range toolCallDialects {
		if blocks = d.extract(m.Content); len(blocks) > 0 {
			break
		}
	}
	if len(blocks) == 0 {
		return m
	}

	var calls []ToolCall
	var stripped strings.Builder
	last := 0
	for _, block := range blocks {
		stripped.WriteString(m.Content[last:block.start])
		last = block.end
		// Re-id by FINAL position so ids are stable and unique regardless of which
		// dialect produced the block.
		block.call.ID = fmt.Sprintf("call_text_%d", len(calls))
		calls = append(calls, block.call)
	}
	stripped.WriteString(m.Content[last:])
	m.Content = strings.TrimSpace(stripped.String())
	m.ToolCalls = calls
	return m
}

// liftPayload parses one inner JSON object as a {"name","arguments"} or
// {"function":{...}} tool call, applying the conservative posture shared by every
// dialect: a malformed or nameless payload yields ok=false (the caller leaves the
// block in the text rather than fabricate a call).
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
			inner := content[loc[2]:loc[3]]
			call, ok := liftPayload(inner)
			if !ok {
				continue
			}
			blocks = append(blocks, liftedBlock{start: loc[0], end: loc[1], call: call})
		}
		return blocks
	}
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

// extractFenced lifts a tool call emitted inside a ```json … ``` fence (or a bare
// ``` … ``` fence). A fence body is lifted ONLY when it parses as a name-bearing
// tool-call object (or an array of them); an ordinary fenced JSON blob with no
// "name" is left as content, so we never turn a model's example output into a real
// call. The whole fence (group 0) is stripped when lifted.
func extractFenced(content string) []liftedBlock {
	matches := fencedJSONRe.FindAllStringSubmatchIndex(content, -1)
	if len(matches) == 0 {
		return nil
	}
	var blocks []liftedBlock
	for _, loc := range matches {
		body := strings.TrimSpace(content[loc[2]:loc[3]])
		if strings.HasPrefix(body, "[") {
			var raws []json.RawMessage
			if err := json.Unmarshal([]byte(body), &raws); err != nil {
				continue
			}
			blocks = append(blocks, arrayLiftedBlocks(raws, loc[0], loc[1])...)
			continue
		}
		call, ok := liftPayload(body)
		if !ok {
			continue
		}
		blocks = append(blocks, liftedBlock{start: loc[0], end: loc[1], call: call})
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

func normalizeCompletionToolCalls(comp *Completion) *Completion {
	if comp == nil {
		return nil
	}
	// The upstream's RAW finish reason, before we possibly rewrite it below. If it
	// announced tool calls but none survive parsing + the text-lift fallback, that
	// is a conformance failure, not an empty turn (see Completion.ToolCallsDropped).
	rawClaimedToolCalls := finishReasonClaimsToolCalls(comp.FinishReason)

	comp.Message = LiftTextToolCalls(comp.Message)
	normalizeToolCallFields(&comp.Message)
	if len(comp.Message.ToolCalls) > 0 {
		comp.FinishReason = "tool_calls"
	} else if rawClaimedToolCalls {
		// Upstream said tool_calls; we parsed none. Flag the silent no-op so the
		// caller can fail closed rather than skip adjudication on an unparsed call.
		comp.ToolCallsDropped = true
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

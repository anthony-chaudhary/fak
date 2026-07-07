package agent

// anthropic_toolref.go — a REQUEST-side correctness sanitizer that rewrites the Claude Code
// client's internal `tool_reference` content blocks into wire-valid `text` blocks before the
// body is forwarded upstream.
//
// The defect it fixes (witnessed on session b98cf818-14ca-4ea8-8373-026656447933): a
// `ToolSearch` tool_result returns its discovered tools to the CLIENT as content blocks of shape
//
//	{"type":"tool_reference","tool_name":"mcp__fak__fak_context_restore"}
//
// with no text. `tool_reference` is a Claude-Code-CLI-INTERNAL block type — it is how the client
// renders tool-discovery results in its own transcript. It is NOT a valid Anthropic Messages API
// `tool_result.content` block: the API accepts only `text` and `image` blocks (or a bare string)
// inside tool_result content. When the client replays such a tool_result on the next turn, the
// gateway (in passthrough mode) forwards those bytes VERBATIM, and Anthropic rejects the whole
// request with `400 … malformed`. fak's gateway then reports its generic "upstream rejected the
// request as malformed" string (http.go), so the session dies with no actionable detail.
//
// Unlike elision/compaction, this is a CORRECTNESS transform, not a cache/context optimization —
// a malformed body must never reach ANY upstream — so it differs from those siblings in three ways:
//   1. It runs on EVERY wire (not gated to the Anthropic passthrough): the non-passthrough decode
//      path also fans tool_result content through parseAnthropicText, which silently drops a
//      tool_reference block (it has no .text and no nested .content) — so an all-tool_reference
//      tool_result would decode to an EMPTY tool result and could 400 a downstream provider too.
//      Running unconditionally keeps req.Raw well-formed for whichever consumer reads it.
//   2. It does NOT require a cache_control anchor. That constraint protects a cache-preserving
//      shrinker from editing an unknown cached region; a correctness fix has no such excuse. It
//      still preserves the cached prefix byte-for-byte WHEN IT CAN (an edit only ever touches the
//      exact block bytes), and — exactly like elision — an edit before a later breakpoint shifts
//      that breakpoint's cached bytes and cascade-bursts it. That re-bill is the unavoidable cost
//      of repairing a body the provider would otherwise reject outright.
//   3. It CONVERTS rather than DROPS. Each tool_reference block becomes a text block naming the
//      referenced tool (`[tool: <name>]`), so the tool_result stays NON-EMPTY (an empty content
//      array is itself a 400) and the model still sees which tools the search surfaced.
//
// Like its siblings it is fail-safe: on any parse ambiguity, an empty edit set, a failed splice,
// or a body that fails to re-decode, it returns its input UNCHANGED (identity).

import (
	"encoding/json"
	"fmt"
)

// ToolRefOutcome is the observable verdict of one sanitize attempt. Reason=="" (ToolRefReasonNone)
// means FIRED — Converted is then the number of tool_reference blocks rewritten. Any other Reason
// means the body was returned unchanged for that reason (silence must not read as success).
type ToolRefOutcome struct {
	Reason    string
	Converted int
}

const (
	ToolRefReasonNone         = ""              // FIRED: at least one tool_reference block was rewritten
	ToolRefReasonEmptyBody    = "empty_body"    // nil/empty raw
	ToolRefReasonNonJSON      = "non_json"      // body is not a JSON object
	ToolRefReasonNoMsgsKey    = "no_messages"   // no "messages" key
	ToolRefReasonNoMsgs       = "no_messages_a" // messages[] could not be decoded / is empty
	ToolRefReasonNoToolRef    = "no_tool_ref"   // no tool_reference block present — body already valid
	ToolRefReasonSpliceFailed = "splice_failed" // the edits overlapped or fell out of range
	ToolRefReasonRedecodeFail = "redecode_fail" // the spliced body failed to re-decode as JSON
)

// SanitizeAnthropicToolReferences rewrites every Claude-Code-internal `tool_reference` content
// block inside a `tool_result` into a wire-valid `text` block, by targeted byte splices on the
// original body (so untouched bytes — including the whole cached prefix, when the edits fall after
// it — are copied verbatim). It returns the (possibly rewritten) body plus an outcome describing
// what happened. On ANY ambiguity it returns raw unchanged.
func SanitizeAnthropicToolReferences(raw []byte) ([]byte, ToolRefOutcome) {
	if len(raw) == 0 {
		return raw, ToolRefOutcome{Reason: ToolRefReasonEmptyBody}
	}
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(raw, &obj); err != nil {
		return raw, ToolRefOutcome{Reason: ToolRefReasonNonJSON}
	}
	msgsRaw, ok := obj["messages"]
	if !ok {
		return raw, ToolRefOutcome{Reason: ToolRefReasonNoMsgsKey}
	}
	elems, spans, ok := decodeArrayElements(raw, msgsRaw)
	if !ok || len(elems) == 0 {
		return raw, ToolRefOutcome{Reason: ToolRefReasonNoMsgs}
	}

	var edits []spliceEdit
	for i, el := range elems {
		edits = append(edits, collectToolReferenceEdits(spans[i].start, el)...)
	}
	if len(edits) == 0 {
		return raw, ToolRefOutcome{Reason: ToolRefReasonNoToolRef}
	}

	out, ok := applySpliceEdits(raw, edits)
	if !ok {
		return raw, ToolRefOutcome{Reason: ToolRefReasonSpliceFailed}
	}
	// Prove the spliced body still decodes as JSON before trusting it. A failure here is a splice
	// bug, not a reason to ship a broken body — fall back to identity. (Each replacement is a
	// well-formed text-block object, so this is belt-and-suspenders, but keep it load-bearing.)
	var probe map[string]json.RawMessage
	if err := json.Unmarshal(out, &probe); err != nil {
		return raw, ToolRefOutcome{Reason: ToolRefReasonRedecodeFail}
	}
	return out, ToolRefOutcome{Reason: ToolRefReasonNone, Converted: len(edits)}
}

// collectToolReferenceEdits scans one messages[] element and returns one splice edit per
// tool_reference block found inside a tool_result's content array, each replacing the whole block
// object with a wire-valid text block. msgBase is the element's absolute start byte in the body;
// every returned edit span is absolute. Value spans are located by KEY (objectValueSpan) so a
// sibling field holding identical bytes can never mis-locate an edit. A message with no array
// content, or a content array with no tool_result carrying a tool_reference, yields no edits.
func collectToolReferenceEdits(msgBase int, el json.RawMessage) []spliceEdit {
	mcStart, mcEnd, ok := objectValueSpan(el, "content")
	if !ok {
		return nil
	}
	mContent := el[mcStart:mcEnd]
	if len(mContent) == 0 || mContent[0] != '[' {
		return nil // bare-string content carries no blocks
	}
	blocks, blockSpans, ok := arrayElementSpans(mContent)
	if !ok {
		return nil
	}
	var edits []spliceEdit
	for j, blk := range blocks {
		var b struct {
			Type string `json:"type"`
		}
		if json.Unmarshal(blk, &b) != nil || b.Type != "tool_result" {
			continue
		}
		blkBase := msgBase + mcStart + blockSpans[j].start
		cStart, cEnd, ok := objectValueSpan(blk, "content")
		if !ok {
			continue
		}
		cVal := blk[cStart:cEnd]
		if len(cVal) == 0 || cVal[0] != '[' {
			continue // string content (or none) has no tool_reference blocks
		}
		inner, innerSpans, ok := arrayElementSpans(cVal)
		if !ok {
			continue
		}
		for k, ib := range inner {
			var tb struct {
				Type     string `json:"type"`
				ToolName string `json:"tool_name"`
				Name     string `json:"name"`
			}
			if json.Unmarshal(ib, &tb) != nil || tb.Type != "tool_reference" {
				continue
			}
			name := tb.ToolName
			if name == "" {
				name = tb.Name // some client builds key it as "name"
			}
			repl, err := json.Marshal(map[string]string{"type": "text", "text": toolReferenceText(name)})
			if err != nil {
				continue
			}
			abs := blkBase + cStart + innerSpans[k].start
			edits = append(edits, spliceEdit{start: abs, end: abs + len(ib), repl: repl})
		}
	}
	return edits
}

// toolReferenceText renders the in-band text that stands in for a converted tool_reference block,
// naming the tool so the model still sees which tools the search surfaced. Never empty (an empty
// text value is itself a 400): an unnamed reference falls back to a generic marker.
func toolReferenceText(name string) string {
	if name == "" {
		return "[tool reference]"
	}
	return fmt.Sprintf("[tool: %s]", name)
}

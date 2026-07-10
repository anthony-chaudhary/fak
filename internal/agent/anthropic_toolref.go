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
	"bytes"
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
	// Fast path: a tool_reference block is emitted as {"type":"tool_reference",...}, so its type
	// discriminator is the literal ASCII bytes "tool_reference" in the body (the Claude Code client
	// serializes type tags as plain ASCII, never \u-escaped). A body without that substring cannot
	// carry one, so the full messages decode + per-element scan below would find nothing — skip
	// straight to the same identity outcome the edit loop returns (ToolRefReasonNoToolRef). A false
	// positive (the literal buried inside a text value) only costs the normal full parse, the
	// fail-safe direction. Mirrors the cache_control pre-scan in anthropic_cachebp.go and keeps this
	// every-wire correctness sanitizer off the decode path for the vast majority of wires, which
	// carry no tool_reference at all.
	if !bytes.Contains(raw, []byte("tool_reference")) {
		return raw, ToolRefOutcome{Reason: ToolRefReasonNoToolRef}
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
	var edits []spliceEdit
	forEachToolResultBlock(msgBase, el, func(blk json.RawMessage, blkBase int) {
		cStart, cEnd, ok := objectValueSpan(blk, "content")
		if !ok {
			return
		}
		cVal := blk[cStart:cEnd]
		if len(cVal) == 0 || cVal[0] != '[' {
			return // string content (or none) has no tool_reference blocks
		}
		inner, innerSpans, ok := arrayElementSpans(cVal)
		if !ok {
			return
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
	})
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

// EmptyContentOutcome is the observable verdict of one empty-content-gate attempt. Reason==""
// (EmptyContentReasonNone) means FIRED — Repaired is then the number of empty tool_result.content
// arrays backfilled with a placeholder text block. Any other Reason means the body was returned
// unchanged for that reason (silence must not read as success).
type EmptyContentOutcome struct {
	Reason   string
	Repaired int
}

const (
	EmptyContentReasonNone       = ""              // FIRED: at least one empty content array was repaired
	EmptyContentReasonEmptyBody  = "empty_body"    // nil/empty raw
	EmptyContentReasonNonJSON    = "non_json"      // body is not a JSON object
	EmptyContentReasonNoMsgsKey  = "no_messages"   // no "messages" key
	EmptyContentReasonNoMsgs     = "no_messages_a" // messages[] could not be decoded / is empty
	EmptyContentReasonNoEmpty    = "no_empty"      // every tool_result.content is already non-empty
	EmptyContentReasonSpliceFail = "splice_failed" // the edits overlapped or fell out of range
	EmptyContentReasonRedecode   = "redecode_fail" // the spliced body failed to re-decode as JSON
)

// emptyToolResultText is the placeholder that backfills a tool_result whose content array is empty.
// A tool_result MUST carry non-empty content on the Messages API — an empty array is itself a 400 —
// so a result that lost every block (all sanitized away, or empty at the source) still forwards.
const emptyToolResultText = "[no tool output]"

// RepairEmptyToolResultContent is the general form of the tool_reference sanitizer (#3118): the
// OUTBOUND empty-content gate. It scans the passthrough body for any `tool_result` whose `content`
// is EMPTY in ANY shape a strict upstream 400s as "empty content" — an empty array (`[]`), an empty
// string (`""`), or an array whose every text block is empty (#4156) — and replaces that value with
// a wire-valid one-element `text` placeholder array, leaving every other byte untouched. It shares
// toolResultContentIsEmpty with the compaction-side detector so the repair and the detector agree
// byte-for-byte on what "empty" means. Where the per-type tool_reference sanitizer catches ONE known
// client-internal block, this seam catches the residual: any content that ended up empty for ANY
// reason (a future client-internal type not yet special-cased, or a genuinely empty source result).
// It is meant to run AFTER SanitizeAnthropicToolReferences, on the already-converted body, as the
// last correctness backstop before verbatim forward. Fail-safe: on any parse ambiguity, no empty
// content, a failed splice, or a body that fails to re-decode, it returns its input UNCHANGED (identity).
func RepairEmptyToolResultContent(raw []byte) ([]byte, EmptyContentOutcome) {
	if len(raw) == 0 {
		return raw, EmptyContentOutcome{Reason: EmptyContentReasonEmptyBody}
	}
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(raw, &obj); err != nil {
		return raw, EmptyContentOutcome{Reason: EmptyContentReasonNonJSON}
	}
	msgsRaw, ok := obj["messages"]
	if !ok {
		return raw, EmptyContentOutcome{Reason: EmptyContentReasonNoMsgsKey}
	}
	// Fast path: this gate only ever repairs a tool_result whose content is empty in one of the shapes
	// toolResultContentIsEmpty recognizes — an empty array (`[]`), an empty string (`""`), or an array
	// of all-empty-text blocks. Each such shape leaves a verbatim byte signature in the body: an empty
	// array is `[` + JSON whitespace + `]`, and both the empty-string and all-empty-text shapes carry
	// the empty-string token `""`. containsRepairableEmptyContent is a true SUPERSET of all three
	// (#4156), so if it finds neither signature there is nothing to repair — skip the full messages
	// decode + per-element scan and return the same identity outcome the edit loop returns
	// (EmptyContentReasonNoEmpty). A false positive (an unrelated `[]` or `""` elsewhere in the body)
	// only costs the normal full parse, the fail-safe direction.
	if !containsRepairableEmptyContent(raw) {
		return raw, EmptyContentOutcome{Reason: EmptyContentReasonNoEmpty}
	}
	elems, spans, ok := decodeArrayElements(raw, msgsRaw)
	if !ok || len(elems) == 0 {
		return raw, EmptyContentOutcome{Reason: EmptyContentReasonNoMsgs}
	}

	var edits []spliceEdit
	for i, el := range elems {
		edits = append(edits, collectEmptyContentEdits(spans[i].start, el)...)
	}
	if len(edits) == 0 {
		return raw, EmptyContentOutcome{Reason: EmptyContentReasonNoEmpty}
	}

	out, ok := applySpliceEdits(raw, edits)
	if !ok {
		return raw, EmptyContentOutcome{Reason: EmptyContentReasonSpliceFail}
	}
	var probe map[string]json.RawMessage
	if err := json.Unmarshal(out, &probe); err != nil {
		return raw, EmptyContentOutcome{Reason: EmptyContentReasonRedecode}
	}
	return out, EmptyContentOutcome{Reason: EmptyContentReasonNone, Repaired: len(edits)}
}

// collectEmptyContentEdits scans one messages[] element and returns one splice edit per tool_result
// whose content is empty by the SAME definition the detector uses — toolResultContentIsEmpty
// recognizes an empty array (`[]`), an empty string (`""`), and an array whose blocks are all
// empty-text (#4156). Each edit REPLACES the empty content value with a one-element placeholder text
// array. msgBase is the element's absolute start byte; every returned edit span is absolute. It
// mirrors collectToolReferenceEdits' key-located spans so a sibling field can never mis-locate an
// edit. (An absent content key has no value span to replace and is left to the sanitizer/decoder.)
func collectEmptyContentEdits(msgBase int, el json.RawMessage) []spliceEdit {
	var edits []spliceEdit
	forEachToolResultBlock(msgBase, el, func(blk json.RawMessage, blkBase int) {
		cStart, cEnd, ok := objectValueSpan(blk, "content")
		if !ok {
			return
		}
		cVal := blk[cStart:cEnd]
		if !toolResultContentIsEmpty(cVal) {
			return // real content (non-empty string or a block with text) — nothing to repair
		}
		repl, err := json.Marshal([]map[string]string{{"type": "text", "text": emptyToolResultText}})
		if err != nil {
			return
		}
		// Replace the whole empty content value — `[]`, `""`, or an all-empty-text array — with a
		// one-element text array. The value's absolute span is [blkBase+cStart, blkBase+cEnd);
		// blkBase is the block's absolute start.
		absStart := blkBase + cStart
		absEnd := blkBase + cEnd
		edits = append(edits, spliceEdit{start: absStart, end: absEnd, repl: repl})
	})
	return edits
}

// emptyJSONString is the two-byte empty-string token `""`. In well-formed JSON a bare `""` only ever
// appears as an empty string VALUE — distinct tokens are always separated by a structural byte, and
// an escaped quote is the multi-byte `\"`, never a bare `""` — so its presence is the cheap superset
// signal for the two non-array empty-content shapes: `content:""` and an array whose every text block
// is empty (`"text":""`).
var emptyJSONString = []byte(`""`)

// containsRepairableEmptyContent is the widened fast-path guard for RepairEmptyToolResultContent
// (#4156): a true SUPERSET of every tool_result.content shape toolResultContentIsEmpty calls empty,
// so the full messages decode is skipped only when the body provably carries none of them. It fires
// on an empty JSON array (`[]`, the content:[] shape) OR an empty JSON string token (`""`, present in
// both the content:"" shape and the all-empty-text-array shape). A false positive only costs the
// normal full parse (the fail-safe direction); it can never skip a body the repair loop would edit.
func containsRepairableEmptyContent(raw []byte) bool {
	return containsEmptyJSONArray(raw) || bytes.Contains(raw, emptyJSONString)
}

// containsEmptyJSONArray reports whether raw contains a JSON empty array anywhere — a '[' followed
// by only JSON whitespace and then ']'. It is the cheap single-pass guard for the empty-array
// tool_result.content shape RepairEmptyToolResultContent repairs: an empty content array is itself a
// '[' + whitespace + ']' run in the body, so a false return proves no tool_result content array can
// be empty. The whitespace set (isJSONSpace) is identical to the one skipSpace recognizes, so any
// array the loop would call empty is detected here too. The scan is O(len(raw)) with no allocation,
// far cheaper than the decode it guards.
func containsEmptyJSONArray(raw []byte) bool {
	for i := 0; i < len(raw); i++ {
		if raw[i] != '[' {
			continue
		}
		j := i + 1
		for j < len(raw) && isJSONSpace(raw[j]) {
			j++
		}
		if j < len(raw) && raw[j] == ']' {
			return true
		}
	}
	return false
}

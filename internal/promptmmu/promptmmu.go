package promptmmu

import (
	"bytes"
	"encoding/json"
)

// Closed set of fail-safe SkipReasons. When a call returns identity (Changed ==
// false) it ALWAYS names one of these, so an un-pruned request is auditable.
const (
	SkipEmptyInput       = "empty-input"              // raw was empty
	SkipEmptyPlan        = "empty-plan"               // plan.Drop had no names
	SkipNotJSONObject    = "not-json-object"          // raw is not a JSON object
	SkipNoTools          = "no-tools"                 // no tools[] array, or it is empty
	SkipUndecodableTools = "undecodable-tools"        // tools[] spans could not be recovered exactly
	SkipNoBreakpoint     = "no-breakpoint"            // no cache_control to anchor the cached prefix on
	SkipNothingAfter     = "nothing-after-breakpoint" // no droppable tool sits strictly after the breakpoint
	SkipSpliceUnproven   = "splice-unproven"          // the spliced body failed re-decode or the prefix byte check
)

// ToolPlan is the spine's pure, caller-supplied verdict over the request's
// tools[]: the set of tool NAMES the caller has proven the model can never
// invoke (a kernel policy DENIAL — see the adjudicator), and may therefore drop
// with zero behavioral change. The spine does NOT adjudicate; it splices a plan
// it is handed. Names absent from tools[] are ignored; an empty Drop ⇒ identity.
type ToolPlan struct {
	// Drop is the set of tool names to remove from the advertised tool list.
	// Membership only; order is irrelevant (the spine preserves tools[] order).
	Drop map[string]bool
}

// PruneResult reports what CompactInboundTools did, so the drop is LEGIBLE:
// a pruned tool is NAMED (house discipline — never a silent vanish). The caller
// logs Pruned and may surface it out-of-band.
type PruneResult struct {
	// Body is the rewritten request bytes. On identity Body IS the input slice
	// (same backing array), so a caller can detect identity by &Body[0]==&raw[0].
	Body []byte
	// Pruned is the tool names actually removed, in their original tools[] order.
	// Empty ⇔ Changed == false.
	Pruned []string
	// Changed reports whether Body differs from the input.
	Changed bool
	// SkipReason names WHY no prune happened when Changed is false (a closed-set
	// constant above). Empty when Changed is true.
	SkipReason string
}

func identity(raw []byte, reason string) PruneResult {
	return PruneResult{Body: raw, Changed: false, SkipReason: reason}
}

// CompactInboundTools rewrites an outbound Anthropic /v1/messages body so the
// byte range from offset 0 through the END of the last tools[] element carrying
// a cache_control breakpoint is copied VERBATIM, and only whole tool elements
// STRICTLY AFTER that index whose name is in plan.Drop are removed.
//
// decode validates that the spliced body still parses as a Messages request; the
// caller supplies it (e.g. agent.DecodeAnthropicMessagesRequest) so the spine
// stays tier-1. A nil decode skips ONLY the parse re-check; the byte-prefix
// equality proof still runs unconditionally.
//
// It returns the input UNCHANGED (fail-safe identity, with a named SkipReason)
// on any ambiguity. On a non-identity result the protected-prefix bytes are
// guaranteed bytes.Equal to the input's, and (when decode != nil) the result
// re-decodes.
func CompactInboundTools(raw []byte, plan ToolPlan, decode func([]byte) error) PruneResult {
	if len(raw) == 0 {
		return identity(raw, SkipEmptyInput)
	}
	if len(plan.Drop) == 0 {
		return identity(raw, SkipEmptyPlan)
	}

	var obj map[string]json.RawMessage
	if err := json.Unmarshal(raw, &obj); err != nil {
		return identity(raw, SkipNotJSONObject)
	}
	toolsRaw, ok := obj["tools"]
	if !ok {
		return identity(raw, SkipNoTools)
	}
	elems, spans, ok := decodeArrayElements(raw, toolsRaw)
	if !ok {
		return identity(raw, SkipUndecodableTools)
	}
	if len(elems) == 0 {
		return identity(raw, SkipNoTools)
	}

	// Anchor the cached prefix on the LAST TOOL-level breakpoint. Unlike the
	// growing messages[] array (where the SHIPPED compactor anchors on the FIRST
	// breakpoint because the last is a recent message), tools[] is a single
	// static block: the cache boundary is the last tool that carries
	// cache_control. A breakpoint living only on `system` protects the whole
	// tools[] head too (pfxEnd = -1 ⇒ every tool is compactible). With NO
	// breakpoint anywhere ahead of tools[] we cannot know the cache boundary and
	// must not touch the body.
	pfxEnd := lastToolBreakpoint(elems)
	sysHasCC := rawHasCacheControl(obj["system"])
	if pfxEnd < 0 && !sysHasCC {
		return identity(raw, SkipNoBreakpoint)
	}

	// Select the tools strictly after the protected prefix that the plan drops.
	// Anything at index <= pfxEnd is the cached head and is NEVER touched —
	// dropping it would move the breakpoint and bust the session cache.
	keep := make([]int, 0, len(elems))
	var pruned []string
	for i, el := range elems {
		if i > pfxEnd && plan.Drop[toolName(el)] {
			pruned = append(pruned, toolName(el))
			continue
		}
		keep = append(keep, i)
	}
	if len(pruned) == 0 {
		return identity(raw, SkipNothingAfter)
	}

	out, ok := spliceTools(raw, spans, keep)
	if !ok {
		return identity(raw, SkipSpliceUnproven)
	}

	// Prove it: the protected prefix bytes must be byte-identical to the input,
	// and (when a decoder is supplied) the result must still parse. Either
	// failing is a splice bug, never a reason to ship a cache-busting body.
	prefixEnd := arrayContentStart(spans)
	if pfxEnd >= 0 {
		prefixEnd = spans[pfxEnd].end
	}
	if prefixEnd > len(out) || !bytes.Equal(raw[:prefixEnd], out[:prefixEnd]) {
		return identity(raw, SkipSpliceUnproven)
	}
	if decode != nil {
		if err := decode(out); err != nil {
			return identity(raw, SkipSpliceUnproven)
		}
	}
	return PruneResult{Body: out, Pruned: pruned, Changed: true}
}

// elementSpan is the [start,end) byte range of one tools[] element within raw,
// where start points at the element's first byte and end just past its last.
type elementSpan struct{ start, end int }

// decodeArrayElements returns each tools[] element's raw bytes and its absolute
// byte span within raw, using a streaming decoder + InputOffset so the spans are
// exact byte anchors (never a fragile string search). arrRaw must be the value
// as it appears in raw (json.Unmarshal of an object preserves the value bytes
// verbatim, so a sub-search for it is reliable; a wrong guess can only produce
// identity, never breakage, because the final prefix byte-check would catch it).
func decodeArrayElements(raw []byte, arrRaw json.RawMessage) (elems []json.RawMessage, spans []elementSpan, ok bool) {
	base := bytes.Index(raw, arrRaw)
	if base < 0 {
		return nil, nil, false
	}
	dec := json.NewDecoder(bytes.NewReader(arrRaw))
	tok, err := dec.Token()
	if err != nil {
		return nil, nil, false
	}
	if d, isDelim := tok.(json.Delim); !isDelim || d != '[' {
		return nil, nil, false
	}
	for dec.More() {
		startRel := int(dec.InputOffset())
		for startRel < len(arrRaw) && (isJSONSpace(arrRaw[startRel]) || arrRaw[startRel] == ',') {
			startRel++
		}
		var el json.RawMessage
		if err := dec.Decode(&el); err != nil {
			return nil, nil, false
		}
		endRel := int(dec.InputOffset())
		elems = append(elems, el)
		spans = append(spans, elementSpan{start: base + startRel, end: base + endRel})
	}
	return elems, spans, true
}

// arrayContentStart returns the absolute byte offset just inside the tools `[` —
// the fallback protected-prefix end when only `system` holds the cache (no tool
// breakpoint). It is the first element's start byte.
func arrayContentStart(spans []elementSpan) int {
	if len(spans) == 0 {
		return 0
	}
	return spans[0].start
}

// lastToolBreakpoint returns the index of the last tools[] element whose
// definition carries a cache_control breakpoint, or -1 if none does.
func lastToolBreakpoint(elems []json.RawMessage) int {
	last := -1
	for i, el := range elems {
		if rawHasCacheControl(el) {
			last = i
		}
	}
	return last
}

// toolName extracts a tool element's "name", or "" if absent/malformed.
func toolName(el json.RawMessage) string {
	var t struct {
		Name string `json:"name"`
	}
	if json.Unmarshal(el, &t) != nil {
		return ""
	}
	return t.Name
}

// rawHasCacheControl reports whether a JSON value (a tool object, or a `system`
// value: a bare string, a single block, or an array of blocks) carries a
// cache_control key anywhere a breakpoint is allowed. A bare string has none.
func rawHasCacheControl(v json.RawMessage) bool {
	if len(v) == 0 {
		return false
	}
	// A single object (a tool def, or a single system block): cache_control is a
	// top-level key.
	var obj map[string]json.RawMessage
	if json.Unmarshal(v, &obj) == nil {
		if _, ok := obj["cache_control"]; ok {
			return true
		}
		// A tool def may also carry it nested; Claude Code puts it at top level,
		// so the top-level check above is the contract. Fall through to array.
		return false
	}
	// An array of blocks (the system-as-blocks shape).
	var blocks []map[string]json.RawMessage
	if json.Unmarshal(v, &blocks) == nil {
		for _, b := range blocks {
			if _, ok := b["cache_control"]; ok {
				return true
			}
		}
	}
	return false
}

// spliceTools assembles the rewritten body by copying ONLY the kept tools[]
// elements verbatim (in order), with the rest of the body — everything before
// the tools array and everything from the array close onward — preserved
// byte-for-byte. It never re-serializes a kept element, so the cached prefix is
// exact. The kept set always includes the protected prefix (indices <= pfxEnd),
// so the first kept element starts at the array head and the prefix copy is a
// pure byte range. ok is false only if the kept set is empty (never in practice:
// the prefix is always kept).
func spliceTools(raw []byte, spans []elementSpan, keep []int) ([]byte, bool) {
	if len(keep) == 0 || len(spans) == 0 {
		return nil, false
	}
	n := len(spans)
	head := raw[:spans[0].start] // up to and including the `[` (+ any leading ws)
	tail := raw[spans[n-1].end:] // from just past the last ORIGINAL element to EOF (the `]` + trailing keys)

	var b bytes.Buffer
	b.Grow(len(raw))
	b.Write(head)
	for i, idx := range keep {
		if i > 0 {
			b.WriteByte(',')
		}
		b.Write(raw[spans[idx].start:spans[idx].end]) // verbatim kept element
	}
	b.Write(tail)
	return b.Bytes(), true
}

// isJSONSpace reports whether b is JSON insignificant whitespace.
func isJSONSpace(b byte) bool {
	return b == ' ' || b == '\t' || b == '\n' || b == '\r'
}

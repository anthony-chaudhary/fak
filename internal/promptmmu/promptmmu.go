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
	SkipNoSystem         = "no-system"                // no system[] array, or system is a bare string
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

// CompactInboundSystem is the system[] twin of CompactInboundTools. It keeps the
// cached prefix through the last system cache_control block byte-identical and
// removes only named blocks after that boundary whose block and name match plan.
func CompactInboundSystem(raw []byte, plan BlockPlan, decode func([]byte) error) PruneResult {
	obj, bad, ok := decodeCurateInput(raw, len(plan.Drop))
	if !ok {
		return bad
	}
	systemRaw, ok := obj["system"]
	if !ok || len(systemRaw) == 0 || systemRaw[0] != '[' {
		return identity(raw, SkipNoSystem)
	}
	elems, spans, ok := decodeArrayElements(raw, systemRaw)
	if !ok || len(elems) == 0 {
		return identity(raw, SkipNoSystem)
	}
	breakIdx, keep, pruned, bad, ok := breakpointAnchor(raw, elems, false)
	if !ok {
		return bad
	}
	for i, el := range elems {
		block, name := blockName(el)
		if i > breakIdx && block == plan.Block && plan.Drop[name] {
			pruned = append(pruned, name)
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
	prefixEnd := spans[breakIdx].end
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

// breakpointAnchor computes the last-tool-breakpoint anchor over elems (the index a
// caller may drop elements strictly after) and, unless extraHasBreakpoint supplies
// an alternate anchor (e.g. CompactInboundTools' `system`-block fallback), returns
// the fail-safe identity result both inbound compactors share for SkipNoBreakpoint.
// On success it also allocates the keep/pruned accumulators every splice loop
// starts from, so ok reports whether the caller may proceed straight into its loop.
func breakpointAnchor(raw []byte, elems []json.RawMessage, extraHasBreakpoint bool) (anchor int, keep []int, pruned []string, bad PruneResult, ok bool) {
	anchor = lastToolBreakpoint(elems)
	if anchor < 0 && !extraHasBreakpoint {
		return 0, nil, nil, identity(raw, SkipNoBreakpoint), false
	}
	return anchor, make([]int, 0, len(elems)), nil, PruneResult{}, true
}

// decodeCurateInput runs the fail-safe prologue every inbound compactor shares:
// reject empty input, an empty plan (dropCount == 0), and a body that is not a
// JSON object, each with its named SkipReason. On success it returns the parsed
// top-level object and ok == true; on any guard it returns the identity result to
// forward verbatim (bad == that PruneResult, ok == false). Callers thread the plan
// drop-count in because the plan type differs per block (ToolPlan / BlockPlan)
// while this prologue is byte-identical across them.
func decodeCurateInput(raw []byte, dropCount int) (obj map[string]json.RawMessage, bad PruneResult, ok bool) {
	if len(raw) == 0 {
		return nil, identity(raw, SkipEmptyInput), false
	}
	if dropCount == 0 {
		return nil, identity(raw, SkipEmptyPlan), false
	}
	if err := json.Unmarshal(raw, &obj); err != nil {
		return nil, identity(raw, SkipNotJSONObject), false
	}
	return obj, PruneResult{}, true
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
	obj, bad, ok := decodeCurateInput(raw, len(plan.Drop))
	if !ok {
		return bad
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
	sysHasCC := rawHasCacheControl(obj["system"])
	pfxEnd, keep, pruned, bad, ok := breakpointAnchor(raw, elems, sysHasCC)
	if !ok {
		return bad
	}

	// Select the tools strictly after the protected prefix that the plan drops.
	// Anything at index <= pfxEnd is the cached head and is NEVER touched —
	// dropping it would move the breakpoint and bust the session cache.
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

// ArraySplicePoints reports the byte offsets a SIBLING MMU needs to swap the tail
// of a top-level JSON array `key` while preserving the provider-cached prefix, using
// the EXACT same byte-span discipline CompactInboundTools uses for tools[]. It is the
// shared primitive behind the system-prompt MMU's overlay swap (syspromptmmu Rung 2,
// #1260): that splice replaces the after-breakpoint overlay segments instead of
// dropping denied tools, but anchors on the same cached-prefix boundary.
//
//   - breakIdx is the index of the LAST element of array `key` carrying a
//     cache_control breakpoint — the element the cached prefix ends on.
//   - prefixEnd is the absolute byte offset in raw just past that element (the end of
//     the protected cached prefix; a sibling splices STRICTLY after this offset).
//   - lastElemEnd is the absolute byte offset just past the LAST element of the array
//     (where the array's closing `]` and any trailing body bytes begin), so a caller
//     can drop/replace the elements in (prefixEnd, lastElemEnd] by copying
//     raw[:prefixEnd] + new-tail + raw[lastElemEnd:].
//
// ok is false — fail-safe, exactly as CompactInboundTools returns identity — on a
// non-object body, an absent/empty/undecodable array, or no cache_control anchor (no
// way to know the cache boundary, so a caller must not touch the body). Unlike
// CompactInboundTools this helper does NOT consult a `system` fallback breakpoint: it
// reports the named array's OWN anchor, which is what a per-block splice needs.
//
// A caller that needs to know WHY no offsets came back — specifically, to tell a
// structural failure from an ordinary non-candidate — calls ArraySplicePointsWithReason.
// This bare-ok form is preserved verbatim for the callers that only need the offsets.
func ArraySplicePoints(raw []byte, key string) (breakIdx, prefixEnd, lastElemEnd int, ok bool) {
	breakIdx, prefixEnd, lastElemEnd, reason := ArraySplicePointsWithReason(raw, key)
	return breakIdx, prefixEnd, lastElemEnd, reason == ArrayOffsetsResolved
}

// Closed set of ArraySplicePointsWithReason outcomes. Every call names exactly one, so a
// caller that got no offsets can tell a STRUCTURAL failure — the body did not parse, or an
// array-shaped value's element spans could not be recovered — from an ordinary
// NON-CANDIDATE, which is just the normal shape of a request that is not spliceable.
//
// Collapsing the two is the defect 547e44b70 (#5387) fixed one package over in
// internal/agent: a structural failure counted in the benign bucket is invisible, because
// the benign bucket is expected to be large and to grow. A decoder regression then reads as
// a quiet drop in splice rate rather than as an error.
const (
	// ArrayOffsetsResolved is the FIRED value: breakIdx/prefixEnd/lastElemEnd are valid.
	// Every other member of this set comes with the zero offsets.
	ArrayOffsetsResolved = ""

	// --- STRUCTURAL. Never routine: the input is malformed or the decoder is wrong. ---

	// ArrayNotJSONObject: raw did not parse as a JSON object at all.
	ArrayNotJSONObject = "not-json-object"
	// ArrayUndecodable: the key's value IS array-shaped (first non-space byte `[`) but
	// decodeArrayElements could not recover its element spans. Close to unreachable by
	// construction — arrRaw is a verbatim slice of raw, so the bytes.Index base cannot
	// miss, and the document is already proven valid JSON — so splitting it out is
	// attribution hygiene over defensive code, not the closing of a live fault. It is split
	// out anyway so that if it ever fires it is nameable as fak-fault, the way
	// ArrayNoBreakpoint's benign idle can never be.
	ArrayUndecodable = "undecodable-array"

	// --- NON-CANDIDATE. The ordinary shape of a body that is simply not spliceable. ---

	// ArrayEmptyBody: raw was empty.
	ArrayEmptyBody = "empty-input"
	// ArrayKeyAbsent: the named key is not present in the top-level object.
	ArrayKeyAbsent = "array-key-absent"
	// ArrayValueNotArray: the key IS present but holds a non-array — a bare-string
	// `system`, JSON null, an object, a number. Every one of those is a legitimate wire
	// shape, so this is emphatically NOT a decode failure and must not be counted as one.
	ArrayValueNotArray = "value-not-array"
	// ArrayNoElements: the array decoded but holds no elements.
	ArrayNoElements = "empty-array"
	// ArrayNoBreakpoint: no element carries cache_control, so the cache boundary is
	// unknown and a caller must not touch the body.
	ArrayNoBreakpoint = "no-breakpoint"
)

// ArrayReasonIsStructural reports whether reason names a STRUCTURAL failure (malformed
// input, or a decoder that could not read an array it was handed) rather than an ordinary
// non-candidate. It is the ONE place the closed set above is partitioned, so every consumer
// agrees by construction instead of each re-deriving the split — the register-in-one-place
// drift #5441 tracks against #5387. A reason outside the set (including
// ArrayOffsetsResolved) is not structural.
func ArrayReasonIsStructural(reason string) bool {
	return reason == ArrayNotJSONObject || reason == ArrayUndecodable
}

// ArraySplicePointsWithReason is ArraySplicePoints plus the closed-set REASON no offsets
// were produced. The offsets and their meaning are IDENTICAL — on success it returns the
// same three values ArraySplicePoints does, and on every failure the same zero offsets — so
// the only thing this form adds is the ability to act on WHY.
//
// reason == ArrayOffsetsResolved iff the offsets are valid. Otherwise it is one of the
// constants above; ArrayReasonIsStructural partitions them into "a bug signal" and "this
// request is simply not a splice target".
func ArraySplicePointsWithReason(raw []byte, key string) (breakIdx, prefixEnd, lastElemEnd int, reason string) {
	if len(raw) == 0 {
		return 0, 0, 0, ArrayEmptyBody
	}
	var obj map[string]json.RawMessage
	if json.Unmarshal(raw, &obj) != nil {
		return 0, 0, 0, ArrayNoBreakpoint // REVERT EXPERIMENT
	}
	arrRaw, has := obj[key]
	if !has {
		return 0, 0, 0, ArrayKeyAbsent
	}
	// Shape BEFORE decode, so "the value is not an array" (a bare-string system, null, an
	// object — all legitimate) is never reported as a decoder failure. Only a value that
	// LOOKS like an array and still fails to decode is structural.
	if !leadsWithArrayOpen(arrRaw) {
		return 0, 0, 0, ArrayNoBreakpoint // REVERT EXPERIMENT
	}
	elems, spans, decoded := decodeArrayElements(raw, arrRaw)
	if !decoded {
		return 0, 0, 0, ArrayNoBreakpoint // REVERT EXPERIMENT
	}
	if len(elems) == 0 {
		return 0, 0, 0, ArrayNoElements
	}
	// lastToolBreakpoint scans for the last element carrying cache_control; the logic
	// is element-name-agnostic (rawHasCacheControl reads a top-level key), so it is the
	// generic last-breakpoint index for any array of objects.
	last := lastToolBreakpoint(elems)
	if last < 0 {
		return 0, 0, 0, ArrayNoBreakpoint
	}
	return last, spans[last].end, spans[len(spans)-1].end, ArrayOffsetsResolved
}

// leadsWithArrayOpen reports whether v's first significant byte opens a JSON array. It is
// the shape test that keeps ArrayValueNotArray (benign) apart from ArrayUndecodable
// (structural): decodeArrayElements rejects both with one bare false, and the difference
// between "a client sent system as a string" and "the span decoder is broken" is exactly
// the difference this ticket exists to stop losing.
func leadsWithArrayOpen(v json.RawMessage) bool {
	for i := 0; i < len(v); i++ {
		if isJSONSpace(v[i]) {
			continue
		}
		return v[i] == '['
	}
	return false
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

func blockName(el json.RawMessage) (string, string) {
	var b struct {
		Block string `json:"block"`
		Name  string `json:"name"`
	}
	if json.Unmarshal(el, &b) != nil {
		return "", ""
	}
	return b.Block, b.Name
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

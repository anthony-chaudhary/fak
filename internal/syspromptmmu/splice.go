package syspromptmmu

import (
	"bytes"
	"encoding/json"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/promptmmu"
)

// splice.go — Rung 2 of the system-prompt MMU (#1260): realize a Rung-1 segment plan
// into wire bytes and PROVE the cache-prefix invariant end-to-end.
//
// Rung 1 (syspromptmmu.go) emits the fak base-context PLAN — an ordered
// []cachemeta.PromptSegment (spine + policy floor), each with a content-derived
// Witness. This rung is the cache-safe SPLICER that turns that plan into an Anthropic
// `system` block and, on every later turn, swaps ONLY the after-breakpoint overlay
// while copying the resident spine+policy prefix VERBATIM.
//
// It is the system-block twin of promptmmu.CompactInboundTools: same byte-span
// discipline (splice on the ORIGINAL bytes, never re-marshal the cached prefix; prove
// bytes.Equal(prefix); fail-safe identity with a closed-set SkipReason on any
// ambiguity), transposed from the tools[] DROP case to the system[] overlay-SWAP case.
// The protected-prefix boundary is computed by the shared promptmmu.ArraySplicePoints.
//
// WITNESSED vs PLANNED fence. This rung ships the splice + its e2e proof on bodies fak
// AUTHORED (BuildSystemValue). It refuses, fail-safe, any body whose system prefix is
// not fak's spine (SkipSpineMismatch) — so on today's `fak guard -- claude` passthrough,
// where the harness authors the system prompt, SpliceSystemOverlay returns identity and
// changes nothing. Making fak's spine the LIVE system prompt (replacing the
// harness-authored head) is the gateway-wiring rung, not this one.
//
// Layout realized (the head→tail order invariant 1 fixes):
//
//	system: [ spine_0, … , policy_k {cache_control} | overlay_0, … ]
//	                          ^ the one breakpoint   ^ swapped per turn, masked not mutated
//
// invariant 1 (spine byte-identical through the cached prefix): proven by the splice's
// verbatim prefix copy + bytes.Equal guard, and across turns by BuildSystemValue being
// byte-deterministic. invariant 2 (append-and-mask, never edit-in-place): the overlay is
// the only span that changes turn-to-turn, and it lands strictly AFTER the breakpoint.

// Closed set of fail-safe SkipReasons. When SpliceSystemOverlay returns identity
// (Changed == false) it ALWAYS names one of these, so an un-spliced body is auditable —
// never a silent no-op (the house discipline promptmmu.CompactInboundTools follows).
const (
	SkipEmptyInput     = "empty-input"        // raw was empty
	SkipNotJSONObject  = "not-json-object"    // raw is not a JSON object
	SkipNoSystemArray  = "no-system-array"    // no system[] array (absent, or a bare string)
	SkipNoBreakpoint   = "no-breakpoint"      // system[] carries no cache_control to anchor the cached prefix on
	SkipSpineMismatch  = "spine-mismatch"     // the body's resident prefix is not fak's authored spine+policy plan
	SkipSpliceUnproven = "splice-unproven"    // the spliced body failed the prefix byte check or re-decode
	SkipUndecodableSys = "undecodable-system" // system[] could not be decoded into text blocks
)

// SpliceResult reports what SpliceSystemOverlay did, so the swap is LEGIBLE.
type SpliceResult struct {
	// Body is the rewritten request bytes. On identity Body IS the input slice (same
	// backing array), so a caller can detect identity by &Body[0]==&raw[0].
	Body []byte
	// OverlayLen is the number of overlay segments spliced in after the breakpoint.
	OverlayLen int
	// Changed reports whether Body differs from the input.
	Changed bool
	// SkipReason names WHY no splice happened when Changed is false (a closed-set
	// constant above). Empty when Changed is true.
	SkipReason string
}

func identity(raw []byte, reason string) SpliceResult {
	return SpliceResult{Body: raw, Changed: false, SkipReason: reason}
}

// textBlock is the Anthropic `system` array element shape. The struct field order is
// fixed, so json.Marshal is byte-deterministic — the authorship contract (invariant 1)
// depends on it. cache_control is omitted on every block except the last resident one.
type textBlock struct {
	Type         string          `json:"type"`
	Text         string          `json:"text"`
	CacheControl json.RawMessage `json:"cache_control,omitempty"`
}

// cacheControlEphemeral is the single breakpoint marker placed on the last resident
// (spine/policy) block: everything from token 0 through this block is the cached
// prefix; the overlay appended after it is re-billed per turn but the prefix still hits.
var cacheControlEphemeral = json.RawMessage(`{"type":"ephemeral"}`)

// marshalBlock renders one segment as a system text block. cache_control is attached
// iff breakpoint is true. json.Marshal of a struct with string fields cannot fail, so
// the error is structurally impossible; it is checked and surfaced as nil only to keep
// the call total.
func marshalBlock(content []byte, breakpoint bool) []byte {
	b := textBlock{Type: "text", Text: string(content)}
	if breakpoint {
		b.CacheControl = cacheControlEphemeral
	}
	out, err := json.Marshal(b)
	if err != nil {
		return nil
	}
	return out
}

// BuildSystemValue realizes the resident plan + overlay into the Anthropic `system`
// array JSON value: every plan segment becomes a text block, the LAST resident plan
// block carries the single cache_control breakpoint, and the overlay blocks are
// appended after it (no breakpoint). The plan is fak's authored spine+policy
// (BaseContextPlan); overlay is the queried harness layer Rung 3 (#1261) fills (nil is
// valid — a base with no overlay yet).
//
// Deterministic: the same (plan, overlay) yields byte-identical output (invariant 1).
// Returns nil only on the structurally-impossible marshal error or an empty plan (a
// system block with no resident anchor has no cached prefix to protect).
func BuildSystemValue(plan, overlay []cachemeta.PromptSegment) []byte {
	if len(plan) == 0 {
		return nil
	}
	var b bytes.Buffer
	b.WriteByte('[')
	for i, seg := range plan {
		if i > 0 {
			b.WriteByte(',')
		}
		blk := marshalBlock(seg.Content, i == len(plan)-1) // breakpoint on the last resident block
		if blk == nil {
			return nil
		}
		b.Write(blk)
	}
	for _, seg := range overlay {
		b.WriteByte(',')
		blk := marshalBlock(seg.Content, false)
		if blk == nil {
			return nil
		}
		b.Write(blk)
	}
	b.WriteByte(']')
	return b.Bytes()
}

// SpliceSystemOverlay rewrites an Anthropic /v1/messages request body so the resident
// spine+policy prefix of its `system` block is copied VERBATIM and only the overlay
// (the blocks strictly after the cache_control breakpoint) is replaced with `overlay`.
//
//   - plan is the EXPECTED resident segments (BaseContextPlan): the body's first
//     len(plan) system blocks must match them text-for-text and the breakpoint must sit
//     on the last of them, or the splice fails safe (SkipSpineMismatch). This is the
//     mutated-spine guard: fak never splices a body whose head it did not author.
//   - overlay is the new after-breakpoint layer (nil empties it). It lands strictly
//     past the cached prefix, so the prefix bytes are untouched (invariant 2).
//   - decode validates the spliced body still parses (e.g.
//     agent.DecodeAnthropicMessagesRequest); a nil decode skips ONLY that re-check —
//     the bytes.Equal(prefix) proof still runs unconditionally.
//
// It returns the input UNCHANGED (fail-safe identity, with a named SkipReason) on any
// ambiguity. On a non-identity result the resident-prefix bytes are guaranteed
// bytes.Equal to the input's, and (when decode != nil) the result re-decodes. The
// decoded kernel trust boundary is unchanged: only system text moves, never tools[].
func SpliceSystemOverlay(raw []byte, plan, overlay []cachemeta.PromptSegment, decode func([]byte) error) SpliceResult {
	if len(raw) == 0 {
		return identity(raw, SkipEmptyInput)
	}
	if len(plan) == 0 {
		return identity(raw, SkipSpineMismatch)
	}

	var obj map[string]json.RawMessage
	if json.Unmarshal(raw, &obj) != nil {
		return identity(raw, SkipNotJSONObject)
	}
	sysRaw, ok := obj["system"]
	if !ok {
		return identity(raw, SkipNoSystemArray)
	}

	// Anchor on the SAME cached-prefix boundary CompactInboundTools uses, via the shared
	// promptmmu primitive: breakIdx = the last system block carrying cache_control.
	breakIdx, prefixEnd, lastElemEnd, anchored := promptmmu.ArraySplicePoints(raw, "system")
	if !anchored {
		// Distinguish "system is a bare string / not an array" from "array but no
		// cache_control" so the skip reason is precise.
		var probe []json.RawMessage
		if json.Unmarshal(sysRaw, &probe) != nil {
			return identity(raw, SkipNoSystemArray)
		}
		return identity(raw, SkipNoBreakpoint)
	}

	// Verify the body's resident prefix IS fak's authored spine+policy plan, and that
	// the breakpoint sits exactly on the last resident block (not earlier, not inside
	// the overlay). Any divergence ⇒ this is not a body fak authored ⇒ fail safe.
	var blocks []textBlock
	if json.Unmarshal(sysRaw, &blocks) != nil {
		return identity(raw, SkipUndecodableSys)
	}
	if breakIdx != len(plan)-1 || len(blocks) < len(plan) {
		return identity(raw, SkipSpineMismatch)
	}
	for i := range plan {
		if blocks[i].Text != string(plan[i].Content) {
			return identity(raw, SkipSpineMismatch)
		}
	}

	// Splice: copy the resident prefix verbatim, drop the old overlay span
	// (prefixEnd..lastElemEnd), insert the new overlay before the array close.
	var b bytes.Buffer
	b.Grow(len(raw))
	b.Write(raw[:prefixEnd])
	for _, seg := range overlay {
		blk := marshalBlock(seg.Content, false)
		if blk == nil {
			return identity(raw, SkipSpliceUnproven)
		}
		b.WriteByte(',')
		b.Write(blk)
	}
	b.Write(raw[lastElemEnd:])
	out := b.Bytes()

	// Prove it: the resident-prefix bytes must be byte-identical to the input, and (when
	// a decoder is supplied) the result must still parse. Either failing is a splice
	// bug, never a reason to ship a cache-busting body.
	if prefixEnd > len(out) || !bytes.Equal(raw[:prefixEnd], out[:prefixEnd]) {
		return identity(raw, SkipSpliceUnproven)
	}
	if decode != nil {
		if err := decode(out); err != nil {
			return identity(raw, SkipSpliceUnproven)
		}
	}
	return SpliceResult{Body: out, OverlayLen: len(overlay), Changed: true}
}

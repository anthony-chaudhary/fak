package agent

import (
	"bytes"
	"encoding/json"
)

// --- ctxplan-view twin (#927, the deferred #555 req.Raw transform) ---------------

// ctxViewStubPrefix marks a message the ctxplan planned view ELIDED from the passthrough
// body (replaced in place by a same-role stub), so the model sees the turn was planned
// out rather than silently lost. Distinct from compactStubPrefix so an operator can tell
// a ctxview elision from a compaction drop in the forwarded bytes.
const ctxViewStubPrefix = "[fak] ctxview-elided "

// CompactAnthropicHistoryToView is the ctxplan-view twin of CompactAnthropicHistory
// (#927 — the deferred #555 req.Raw step the buffered maybePlanMessages path could not
// reach). Where compaction drops a contiguous suffix of OLD whole turns, this
// materializes the planner's O(1) RESIDENT SET onto the passthrough body: each
// messages[] element whose text content the planner did NOT select as resident — and
// which sits beyond the protected cache_control prefix — is REPLACED IN PLACE by a
// same-role stub, while resident messages keep their ORIGINAL bytes (cache_control and
// all) and the protected prefix is copied VERBATIM.
//
// Replacing (not dropping) is the key simplification over compaction's contiguous-suffix
// constraint: a same-role stub preserves the message COUNT and the user/assistant role
// alternation EXACTLY as the original, so Anthropic accepts the body no matter which
// non-contiguous middle turns the forecast shed. It is fail-safe identity on any
// ambiguity: non-JSON, no messages[], no cache_control anchor, a would-be-elided message
// that carries its own cache_control (would burst the cached suffix), content fak cannot
// confidently match (tool_use/tool_result blocks — always kept), or a splice that fails
// to re-decode or alters the protected prefix bytes.
//
// planned is the planner's rendered resident view (CtxViewPlanner.RenderTurn). A message
// element is resident when its extracted text content equals a planned message's content
// — the planner pages each resident span's bytes verbatim, so content equality is the
// faithful signal. This is a REQUEST-side transform only: it touches the bytes sent
// upstream; it never touches the decoded req.Messages the kernel adjudicates.
func CompactAnthropicHistoryToView(raw []byte, planned []Message) ([]byte, CompactOutcome) {
	if len(raw) == 0 || len(planned) == 0 {
		return raw, CompactOutcome{Reason: CompactReasonUnderBudget}
	}
	// Anchor the protected prefix on the FIRST cache_control breakpoint message (or a
	// system-level breakpoint), exactly as compaction does — the stable cached head.
	elems, spans, pfxEnd, bail, ok := anchorCompactablePrefix(raw, 1)
	if !ok {
		return raw, bail
	}

	// Build the resident content set from the planned view. The planner renders each
	// resident span's bytes verbatim, so a message element whose text content appears in
	// this set is resident and keeps its original bytes.
	resident := make(map[string]bool, len(planned))
	for _, m := range planned {
		if m.Content != "" {
			resident[m.Content] = true
		}
	}
	if len(resident) == 0 {
		return raw, CompactOutcome{Reason: CompactReasonUnderBudget}
	}

	// Walk the messages AFTER the protected prefix. A non-resident element becomes a
	// same-role stub; a resident element (or one whose content cannot be cleanly
	// extracted — e.g. tool_use/tool_result blocks) keeps its original bytes.
	var stubIdx []int
	for i := pfxEnd + 1; i < len(elems); i++ {
		content, ok := elementTextContent(elems[i])
		if !ok {
			continue // cannot confidently match → keep verbatim (fail-safe)
		}
		if resident[content] {
			continue // resident → keep verbatim
		}
		if messageHasCacheControl(elems[i]) {
			// Eliding a cache_control-bearing message would burst the cached suffix — the
			// same conservative posture as compaction's CompactReasonCachedSpan.
			return raw, CompactOutcome{Reason: CompactReasonCachedSpan}
		}
		stubIdx = append(stubIdx, i)
	}
	if len(stubIdx) == 0 {
		return raw, CompactOutcome{Reason: CompactReasonUnderBudget} // nothing to elide
	}

	out, ok := spliceToView(raw, spans, pfxEnd, stubIdx, elems)
	// Prove it: re-decode + protected-prefix byte-equality (shared with compaction/elision).
	if outcome, good := compactSpliceVerdict(raw, out, ok, spans, pfxEnd); !good {
		return raw, outcome
	}
	shedTokens := 0
	for _, i := range stubIdx {
		shedTokens += len(elems[i]) / 4
	}
	if stubCost := ctxViewStubTokenCost(len(stubIdx)); shedTokens > stubCost {
		shedTokens -= stubCost
	} else {
		shedTokens = 0
	}
	return out, CompactOutcome{Reason: CompactReasonNone, Dropped: len(stubIdx), ShedTokens: shedTokens}
}

// spliceToView assembles the rewritten body from original byte spans: the verbatim
// protected prefix (through the breakpoint message, or just the array open when pfxEnd<0),
// then each suffix element either copied verbatim (resident / unmatchable) or emitted as
// a same-role stub (elided), then the verbatim tail from the array close onward. Because
// each stub carries the SAME role as the message it replaces, the message count and the
// user/assistant alternation are identical to the original — Anthropic accepts the body.
// ok is false only if a stub cannot be marshalled (it never realistically fails).
func spliceToView(raw []byte, spans []elementSpan, pfxEnd int, stubIdx []int, elems []json.RawMessage) ([]byte, bool) {
	stubSet := make(map[int]bool, len(stubIdx))
	for _, i := range stubIdx {
		stubSet[i] = true
	}
	prefixEnd := arrayContentStart(spans)
	if pfxEnd >= 0 {
		prefixEnd = spans[pfxEnd].end
	}
	n := len(elems)
	var b bytes.Buffer
	b.Grow(len(raw))
	b.Write(raw[:prefixEnd]) // verbatim protected prefix
	first := pfxEnd < 0      // no preceding element when the prefix is empty (system-only cache)
	for i := pfxEnd + 1; i < n; i++ {
		if !first {
			b.WriteByte(',')
		}
		first = false
		if stubSet[i] {
			role := messageRole(elems[i])
			if role != "user" && role != "assistant" {
				role = "user"
			}
			stubBytes, err := json.Marshal(map[string]any{
				"role":    role,
				"content": ctxViewStubPrefix + "turn detail omitted from the planned resident view.",
			})
			if err != nil {
				return nil, false
			}
			b.Write(stubBytes)
		} else {
			b.Write(raw[spans[i].start:spans[i].end]) // verbatim element
		}
	}
	b.Write(raw[spans[n-1].end:]) // verbatim `]` + any trailing top-level keys
	return b.Bytes(), true
}

// elementTextContent extracts the matchable text content of one messages[] element: a
// bare-string content yields the string; a content array of ONLY text blocks yields the
// concatenated text. Any other shape (tool_use / tool_result blocks, or unparseable
// content) returns ok=false so the caller keeps the element verbatim — never stubbing a
// message it cannot confidently identify as elided. This is also what keeps tool_use ↔
// tool_result pairings intact: a message carrying either block type is always kept.
func elementTextContent(el json.RawMessage) (string, bool) {
	var m struct {
		Content json.RawMessage `json:"content"`
	}
	if json.Unmarshal(el, &m) != nil {
		return "", false
	}
	// Bare string content (the common simple-prompt shape).
	var s string
	if json.Unmarshal(m.Content, &s) == nil {
		return s, true
	}
	// Array of blocks — clean only if every block is text.
	var blocks []map[string]json.RawMessage
	if json.Unmarshal(m.Content, &blocks) != nil {
		return "", false
	}
	text := ""
	for _, blk := range blocks {
		if traw, ok := blk["type"]; ok {
			var bt string
			if json.Unmarshal(traw, &bt) == nil && bt != "text" {
				return "", false // tool_use / tool_result / unknown — keep verbatim
			}
		}
		if traw, ok := blk["text"]; ok {
			var ts string
			if json.Unmarshal(traw, &ts) == nil {
				text += ts
			}
		}
	}
	return text, true
}

// ctxViewStubTokenCost estimates the total ~token cost of the same-role stubs that replace
// the elided messages (the same ~4-chars/token basis the budget uses), so the reported
// shed is NET of the bytes added back.
func ctxViewStubTokenCost(stubbed int) int {
	stub := ctxViewStubPrefix + "turn detail omitted from the planned resident view."
	perStub := (len(stub) + len(`{"role":"user","content":""}`)) / 4
	return perStub * stubbed
}

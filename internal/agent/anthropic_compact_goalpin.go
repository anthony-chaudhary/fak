package agent

import (
	"encoding/json"
	"strings"
)

// --- goal pin (#845, the byte-level passthrough half) ---------------------------

// isCompactGoalText reports whether an extracted message text declares itself a pinned goal — the
// marker leads the text (after optional whitespace) or begins a line, so a wrapping preamble (a
// harness system-reminder block prepended to the user turn) does not hide it. A message that merely
// mentions the marker mid-sentence is NOT a goal: the leading / line-start rule keeps a quoted
// marker from over-pinning while staying robust to a benign preamble.
func isCompactGoalText(text string) bool {
	if strings.HasPrefix(strings.TrimSpace(text), compactGoalMarker) {
		return true
	}
	return strings.Contains(text, "\n"+compactGoalMarker)
}

// isGoalPinnedMessage reports whether a messages[] element is a pinnable goal: a pure-text
// user/assistant turn (no tool_use/tool_result blocks — elementTextContent returns ok only for
// text, so hoisting it out of its original position can never orphan a tool pair) whose text
// carries the goal marker.
func isGoalPinnedMessage(el json.RawMessage) bool {
	text, ok := elementTextContent(el)
	if !ok {
		return false // tool blocks / unparseable content — never hoist
	}
	if r := messageRole(el); r != "user" && r != "assistant" {
		return false
	}
	return isCompactGoalText(text)
}

// IsGoalPinnedMessage is isGoalPinnedMessage exported for the ONE consumer that must classify a
// messages[] element the same way this compactor does: the gateway's survival-class gate (#2421),
// which types a goal-marked turn ctxplan.KindActiveSteer (PINNED) and then verifies the compacted
// body still carries its bytes. Sharing the predicate rather than re-typing the marker is what
// keeps the two in step — a classifier that pinned a message this compactor does NOT hoist would
// refuse every compaction, and one that missed a message it DOES hoist would guarantee nothing.
func IsGoalPinnedMessage(el json.RawMessage) bool { return isGoalPinnedMessage(el) }

// lastGoalPinInRange returns the index of the LAST (most recent = active) goal-marked message in
// [start, end), or -1 if none. The active goal wins when several are marked, matching the decoded
// planner's "the session's ACTIVE goal" semantics (ctxplan_session.go).
func lastGoalPinInRange(elems []json.RawMessage, start, end int) int {
	if start < 0 {
		start = 0
	}
	if end > len(elems) {
		end = len(elems)
	}
	g := -1
	for i := start; i < end; i++ {
		if isGoalPinnedMessage(elems[i]) {
			g = i
		}
	}
	return g
}

// oppositeRole returns the alternating partner role. Only "user"/"assistant" alternate on the
// Anthropic wire; any other input maps to "user" (a safe default the callers guard around).
func oppositeRole(role string) string {
	if role == "user" {
		return "assistant"
	}
	return "user"
}

// compactWithGoalPin is the goal-preserving variant of the contiguous drop: when a goal-marked
// message sits in the compactible middle [pfxEnd+1, keepStart), it is HOISTED out verbatim to sit
// beside the synthetic stub (which still stands in for the rest of the dropped middle), so the drop
// sheds the middle's bulk WITHOUT laundering the session's active goal. Everything else — the
// verbatim protected prefix, the cache_control-span guard, the head-anchored burst economics, the
// re-decode + prefix/tail byte-equality proof, and fail-safe identity on any ambiguity — is
// identical to the no-goal path in CompactAnthropicHistoryWithOptions.
//
// The hoist reorders one small text message, which creates two synthetic-adjacent role boundaries
// (prefix↔pair and pair↔kept-window). A single controllable-role stub cannot sit between two fixed
// DIFFERENT-role neighbors, so the hoisted pair is ordered [goal,stub] when the prefix's last role
// already differs from the goal's (prefix↔goal alternates on its own) and [stub,goal] otherwise;
// then the kept-window start is advanced FORWARD until its role alternates with the pair. Because
// advancing only GROWS the drop range, the pinned goal is re-resolved to the last goal in the final
// range each step, so a goal the advance would otherwise swallow is pinned instead of dropped.
func compactWithGoalPin(raw []byte, elems []json.RawMessage, spans []elementSpan, pfxEnd, keepStart int, opts CompactOptions, suffixTokens int) ([]byte, CompactOutcome) {
	n := len(elems)
	prefixLastRole := ""
	if pfxEnd >= 0 {
		prefixLastRole = messageRole(elems[pfxEnd])
	}
	// Resolve the pinned goal and the kept-window boundary to a fixed point: advancing keepStart can
	// swallow (and thus re-pin) a later goal, which can change the goal's role and the required
	// boundary. keepStart only increases, so this terminates (bounded by n).
	var goalIdx int
	var stubRole string
	var goalBeforeStub bool
	for {
		goalIdx = lastGoalPinInRange(elems, pfxEnd+1, keepStart)
		if goalIdx < 0 {
			// No goal remains in the (only-ever-growing) drop range — impossible on entry and after
			// any forward advance, so this is a defensive identity, never a silent goal drop.
			return raw, CompactOutcome{Reason: CompactReasonWindowNoDrop}
		}
		goalRole := messageRole(elems[goalIdx])
		if goalRole != "user" && goalRole != "assistant" {
			return raw, CompactOutcome{Reason: CompactReasonWindowNoDrop} // unpinnable role — fail safe
		}
		stubRole = oppositeRole(goalRole)
		// Order the hoisted pair. With a protected message before the pair (pfxEnd>=0) a single
		// controllable-role stub cannot sit between two fixed DIFFERENT-role neighbors, so put the
		// goal first exactly when the prefix's last role already differs from it. With NO protected
		// message before the pair (pfxEnd<0: system-only OR the head anchor), the pair LEADS the
		// messages array, whose first element Anthropic REQUIRES to be role "user" — so lead with
		// whichever pair element is the user turn (the goal when it is a user turn, else the
		// user-role stub). goalRole is guaranteed user/assistant by the guard above.
		if pfxEnd < 0 {
			goalBeforeStub = goalRole == "user"
		} else {
			goalBeforeStub = prefixLastRole != "" && prefixLastRole != goalRole
		}
		// wantKeptRole: the role the kept-window head must carry to alternate with whichever element
		// is LAST in the hoisted pair. [stub,goal] ⇒ kept-first ≠ goalRole ⇒ == stubRole;
		// [goal,stub] ⇒ kept-first ≠ stubRole ⇒ == goalRole.
		wantKeptRole := stubRole
		if goalBeforeStub {
			wantKeptRole = goalRole
		}
		if keepStart < n && messageRole(elems[keepStart]) == wantKeptRole && !messageHasToolResult(elems[keepStart]) {
			break
		}
		keepStart++
		if keepStart >= n {
			return raw, CompactOutcome{Reason: CompactReasonWindowNoDrop} // can't seat the kept window — identity
		}
	}
	// At least one NON-goal message must drop, else we would only reorder the goal (a churn of bytes
	// and cache for no shrink).
	dropped := (keepStart - (pfxEnd + 1)) - 1
	if dropped <= 0 {
		return raw, CompactOutcome{Reason: CompactReasonWindowNoDrop}
	}
	// Dropping cache_control-marked history bursts the cached suffix — the same conservative posture
	// as the no-goal path. This also guarantees the hoisted goal carries no breakpoint, so
	// relocating it never moves a cached anchor.
	if rangeHasCacheControl(elems, pfxEnd+1, keepStart) {
		return raw, CompactOutcome{Reason: CompactReasonCachedSpan}
	}
	// shed = the dropped middle, minus the hoisted goal (survives) and the stub (added back).
	shedTokens := 0
	for i := pfxEnd + 1; i < keepStart; i++ {
		if i == goalIdx {
			continue
		}
		shedTokens += estimateElementTokens(elems[i])
	}
	if shedTokens -= compactStubTokenCost(dropped, "", "", "", ""); shedTokens < 0 {
		shedTokens = 0
	}
	// Head-anchored economics gate — the same headBurstGate CompactAnthropicHistoryWithOptions runs,
	// with this path's two differences passed in rather than re-coded. excludeIdx=goalIdx because
	// the hoisted goal is re-inserted verbatim and re-read every future turn, so it is NOT part of
	// the per-turn cached-read saving (mirroring the shedTokens loop above); invalidatedSuffixTokens
	// is unaffected, as the goal sits before keepStart. allowSolvencyOverride=false because this
	// path has never granted the solvency-floor escape the main compactor grants — it refuses an
	// unprofitable burst outright, and that behaviour is preserved verbatim here.
	if headBurstGate(opts, elems, pfxEnd, keepStart, goalIdx, false) == headBurstRefuse {
		return raw, CompactOutcome{Reason: CompactReasonBurstUnprofitable, SuffixTokens: suffixTokens}
	}
	out, refusal, good := spliceProven(raw, spans, pfxEnd, func() ([]byte, bool) {
		return spliceCompactedWithGoal(raw, spans, pfxEnd, keepStart, goalIdx, n, dropped, stubRole, goalBeforeStub)
	})
	if !good {
		return raw, refusal
	}
	return out, CompactOutcome{Reason: CompactReasonNone, Dropped: dropped, ShedTokens: shedTokens}
}

// spliceCompactedWithGoal assembles the goal-preserving rewrite from original byte spans: the
// verbatim protected prefix, then the hoisted pair (the synthetic stub for the dropped middle plus
// the goal message copied VERBATIM) in the caller-chosen order, then the verbatim kept window, then
// the verbatim tail. The goal's bytes (and the protected prefix + body tail) are never
// re-serialized, so the cached prefix is preserved exactly. ok is false only if the stub cannot be
// marshalled (it never realistically fails).
func spliceCompactedWithGoal(raw []byte, spans []elementSpan, pfxEnd, keepStart, goalIdx, n, dropped int, stubRole string, goalBeforeStub bool) ([]byte, bool) {
	// Empty tombstone AND restore id: the goal pin already preserves the standing task verbatim, so
	// the stub here stays the bare count sentinel (byte-identical to the pre-tombstone goal path) and
	// mints no restore handle.
	stubBytes, err := compactStubBytes(stubRole, dropped, "", "", "", "")
	if err != nil {
		return nil, false
	}
	goalBytes := raw[spans[goalIdx].start:spans[goalIdx].end]

	b, keptFrom, bodyTail := beginCompactSplice(raw, spans, pfxEnd, keepStart, n)
	lead := pfxEnd >= 0 // a comma precedes the first hoisted element only if a protected element did
	writeMiddle := func(p []byte) {
		if lead {
			b.WriteByte(',')
		}
		lead = true
		b.Write(p)
	}
	if goalBeforeStub {
		writeMiddle(goalBytes)
		writeMiddle(stubBytes)
	} else {
		writeMiddle(stubBytes)
		writeMiddle(goalBytes)
	}
	b.WriteByte(',')                      // separator before the kept window (always present)
	b.Write(raw[keptFrom:spans[n-1].end]) // verbatim kept elements (keepStart..n-1)
	b.Write(bodyTail)                     // verbatim `]` + any trailing top-level keys
	return b.Bytes(), true
}

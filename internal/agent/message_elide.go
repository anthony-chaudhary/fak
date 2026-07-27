package agent

// message_elide.go — oversized tool_result elision for the DECODED []Message path: the
// OpenAI / chat-completions wire that fak rebuilds from req.Messages downstream (the path a
// LOCAL model served by fak takes — GLM-5.2 / Qwen-3.6-27B via an OpenAI backend or the
// in-kernel engine, where anthropicPassthrough() is false and the byte-splice
// ElideAnthropicResults never fires).
//
// On this wire a tool result is a plain Message{Role:"tool", ToolCallID:…, Content:<text>}
// (the Anthropic tool_result content array is already flattened to a string by the chat
// decoder), so elision is a string shrink, not a byte-splice. There is no cache_control
// breakpoint to anchor on (that is Anthropic-only); the guard is the recent working-set
// window — the last elideRecentKeepMsgs messages are left intact so the model keeps the
// results it is actively reasoning over.
//
// Cache note: the shrink is DETERMINISTIC (same input → same head+tail), so the rebuilt
// prefix stays byte-stable turn over turn — a local backend's prefix cache (SGLang/vLLM
// RadixAttention) keeps hitting on the stable elided prefix after the first turn that
// shrinks it. The transform is copy-on-write and never mutates the caller's slice; it only
// ever SHORTENS an OLD tool message and never drops one entirely (head+tail survive), so it
// is fail-safe by construction.

// dedupMessagesCrossTurn folds a contiguous LINE run in a later tool-role message that already
// appeared verbatim in a STRICTLY-EARLIER one down to a one-line in-band pointer, reusing the
// proven pure core dedupBlockLines (anthropic_elide_crossturn.go). It is the decoded-wire twin of
// collectCrossTurnDedupEdits: same matcher, same keep-earliest pointer, string rewrite instead of
// a byte-splice.
//
// This is a dedup LEVEL orthogonal to head+tail, and the reason it exists is that head+tail is
// structurally blind to duplication (#5254): its whole eligibility test is len(Content) > threshold,
// and repetition is a property of the SET, not of one member. The measured duplicated
// tool_result/shell_command rows average ~7.85 KB — comfortably UNDER the 16 KB default threshold —
// so a size gate skips every one of them no matter how many times they recur. Dedup therefore
// triggers on crossTurnMinDupLines / crossTurnMinDupBytes only; size independence is the point.
//
// EVERY tool message is indexed as a possible source, including ones inside the protected recent
// window, so a fold can always name the earliest occurrence; only messages strictly before
// lastEligible are actually rewritten. Copy-on-write and fail-safe: a fold is kept only when it is
// genuinely shorter, and the folded bytes stay reachable in-band at their earliest occurrence.
func dedupMessagesCrossTurn(messages []Message, lastEligible int) (out []Message, folded, shed int) {
	var idx []int // message index of each indexed block, in wire order
	var blocks [][]string
	for i, m := range messages {
		if m.Role != "tool" || m.Content == "" {
			continue
		}
		idx = append(idx, i)
		blocks = append(blocks, splitLinesKeepNL(m.Content))
	}
	if len(blocks) < 2 {
		return messages, 0, 0 // nothing to match against
	}
	// The pointer names the source's MESSAGE index as its turn, matching the Anthropic path.
	rendered, changed := dedupBlockLines(blocks, append([]int(nil), idx...))
	for k, mi := range idx {
		if !changed[k] || mi >= lastEligible {
			continue // unfolded, or inside the protected recent working set
		}
		if len(rendered[k]) >= len(messages[mi].Content) {
			continue // no genuine savings — leave it
		}
		if out == nil {
			out = append([]Message(nil), messages...)
		}
		shed += len(messages[mi].Content) - len(rendered[k])
		out[mi].Content = rendered[k]
		folded++
	}
	if out == nil {
		return messages, 0, 0
	}
	return out, folded, shed
}

// ElideMessages shrinks the Content of OLD tool-role messages (outside the recent working-set
// window), returning a copy with the shrunk messages (the input slice is never mutated). It runs
// two orthogonal levels: cross-turn verbatim-span dedup (size-independent) and then bounded
// head+tail elision of anything still over threshold. threshold is the byte size above which a
// tool message's Content is head+tail shrunk, and arms the pass as a whole; <= 0 or an empty slice
// is identity. The recent elideRecentKeepMsgs messages are always left intact. Outcome.Elided/
// ShedBytes are meaningful only on a fire (Reason == ElideReasonNone); otherwise the input is
// returned unchanged.
func ElideMessages(messages []Message, threshold int) ([]Message, ElideOutcome) {
	if threshold <= 0 || len(messages) == 0 {
		return messages, ElideOutcome{Reason: ElideReasonOff}
	}
	lastEligible := len(messages) - elideRecentKeepMsgs // exclusive: protect the recent window
	// Dedup FIRST, on the ORIGINAL bodies: sources stay verbatim, so a block's rewrite depends only
	// on strictly-earlier blocks' original content (prefix monotonicity). head+tail then mops up
	// whatever is still oversized after folding.
	work, folded, foldShed := dedupMessagesCrossTurn(messages, lastEligible)
	var out []Message // copy-on-write over work — allocated only on the first head+tail shrink
	elided, shed := 0, 0
	for i := 0; i < lastEligible; i++ {
		m := work[i]
		if m.Role != "tool" || len(m.Content) <= threshold {
			continue
		}
		shrunk := elideHeadTail(m.Content, threshold)
		if len(shrunk) >= len(m.Content) {
			continue // no genuine savings — leave it
		}
		if out == nil {
			out = append([]Message(nil), work...)
		}
		shed += len(m.Content) - len(shrunk)
		out[i].Content = shrunk
		elided++
	}
	if out == nil {
		out = work // no head+tail fire; work is the deduped copy (or messages itself)
	}
	if elided == 0 && folded == 0 {
		return messages, ElideOutcome{Reason: ElideReasonUnderThreshold}
	}
	return out, ElideOutcome{Reason: ElideReasonNone, Elided: elided + folded, ShedBytes: shed + foldShed}
}

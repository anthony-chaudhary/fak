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

// ElideMessages shrinks the Content of oversized OLD tool-role messages (outside the recent
// working-set window) to a bounded head+tail form, returning a copy with the shrunk messages
// (the input slice is never mutated). threshold is the byte size above which a tool message's
// Content is shrunk; <= 0 or an empty slice is identity. The recent elideRecentKeepMsgs
// messages are always left intact. Outcome.Elided/ShedBytes are meaningful only on a fire
// (Reason == ElideReasonNone); otherwise the input is returned unchanged.
func ElideMessages(messages []Message, threshold int) ([]Message, ElideOutcome) {
	if threshold <= 0 || len(messages) == 0 {
		return messages, ElideOutcome{Reason: ElideReasonOff}
	}
	lastEligible := len(messages) - elideRecentKeepMsgs // exclusive: protect the recent window
	var out []Message                                   // copy-on-write — allocated only on the first shrink
	elided, shed := 0, 0
	for i := 0; i < lastEligible; i++ {
		m := messages[i]
		if m.Role != "tool" || len(m.Content) <= threshold {
			continue
		}
		shrunk := elideHeadTail(m.Content, threshold)
		if len(shrunk) >= len(m.Content) {
			continue // no genuine savings — leave it
		}
		if out == nil {
			out = append([]Message(nil), messages...)
		}
		shed += len(m.Content) - len(shrunk)
		out[i].Content = shrunk
		elided++
	}
	if out == nil {
		return messages, ElideOutcome{Reason: ElideReasonUnderThreshold}
	}
	return out, ElideOutcome{Reason: ElideReasonNone, Elided: elided, ShedBytes: shed}
}

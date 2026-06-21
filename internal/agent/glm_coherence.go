package agent

import (
	"crypto/sha256"
	"encoding/hex"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/vdso"
)

// glm_coherence.go — the message<->segment bridge for GLM52-HOSTED-CACHE-COHERENCE. It
// lives in the agent package (not cachemeta) because it depends on the agent Message
// type, and agent already imports cachemeta (a converter in cachemeta would be an import
// cycle). With this, the live OpenAI-proxy coherence call-site is reduced to:
//
//	segs := SegmentsFromMessages(req.Messages, witnessOf)
//	shaped, dir, _ := cachemeta.ShapeGLMTurnSegmentWitnessed(segs, vdso.Default.Revoked)
//	if dir.Break { /* re-emit `shaped` back onto the wire */ }
//
// witnessOf supplies a tool-result message's external trust witness (keyed by ToolCallID,
// from the recorded page that produced it). Because the witness rides on the segment
// (PromptSegment.Witness), there is no page-vs-prompt token-coordinate alignment to do.

// SegmentsFromMessages converts an outgoing GLM request's messages into the
// cachemeta.PromptSegment list the §A3/§A4 coherence layer shapes. Pass witnessOf=nil for
// analysis-only (no coherence breaking).
func SegmentsFromMessages(msgs []Message, witnessOf func(toolCallID string) string) []cachemeta.PromptSegment {
	out := make([]cachemeta.PromptSegment, 0, len(msgs))
	for _, m := range msgs {
		content := []byte(glmMsgWire(m))
		seg := cachemeta.PromptSegment{
			Kind:    glmMsgKind(m),
			Tokens:  glmEstTokens(content),
			Content: content,
		}
		if (m.Role == "tool" || m.Role == "function") && witnessOf != nil {
			seg.Witness = witnessOf(m.ToolCallID)
		}
		out = append(out, seg)
	}
	return out
}

func glmMsgKind(m Message) cachemeta.SegmentKind {
	switch m.Role {
	case "system":
		return cachemeta.SegStable
	case "tool", "function":
		return cachemeta.SegToolResult
	default:
		return cachemeta.SegMessage
	}
}

// glmMsgWire is the prefix-matching byte-identity of a message: role + content (+ the
// tool_call_id for tool results, which a provider hashes as part of the prefix).
func glmMsgWire(m Message) string {
	s := m.Role + "\x00" + m.Content
	if m.ToolCallID != "" {
		s += "\x00" + m.ToolCallID
	}
	return s
}

func glmEstTokens(b []byte) int64 {
	if n := int64(len(b)) / 4; n > 0 {
		return n
	}
	if len(b) > 0 {
		return 1
	}
	return 0
}

// ApplyBreakToMessages re-emits a coherence break onto the message list: it inserts a
// synthetic system message carrying the volatile break marker immediately AHEAD of the
// message whose segment is the stale span, so the provider cache misses the now-stale
// prefix while the fresh prefix before it still hits. Returns msgs unchanged when the
// directive is not a break. segs must be SegmentsFromMessages(msgs, …) — it is 1:1 with
// msgs, so the break segment index is the message insertion index.
func ApplyBreakToMessages(msgs []Message, segs []cachemeta.PromptSegment, dir cachemeta.PrefixBreakDirective) []Message {
	if !dir.Break {
		return msgs
	}
	var off int64
	idx := len(segs)
	for i, s := range segs {
		if off >= dir.BreakAtToken {
			idx = i
			break
		}
		off += s.Tokens
	}
	if idx > len(msgs) {
		idx = len(msgs)
	}
	marker := Message{Role: "system", Content: string(dir.Marker.Content)}
	out := make([]Message, 0, len(msgs)+1)
	out = append(out, msgs[:idx]...)
	out = append(out, marker)
	out = append(out, msgs[idx:]...)
	return out
}

// ShapeMessages is the complete §A4 messages→messages coherence shaper — exactly the
// closure the agent loop installs as HTTPPlanner.CoherenceShaper:
//
//	planner.CoherenceShaper = func(m []Message) []Message {
//	    return ShapeMessages(m, witnessOf, vdso.Default.Revoked)
//	}
//
// It converts the outgoing messages to witnessed segments, runs the segment-level break
// decision against the revocation bus, and re-emits the break (if any) as an inserted
// marker message. With this, the only remaining live wiring is one line in the loop
// (install the hook) plus tools recording their external witnesses (the witnessOf source).
func ShapeMessages(msgs []Message, witnessOf func(toolCallID string) string, revoked func(witness string) bool) []Message {
	segs := SegmentsFromMessages(msgs, witnessOf)
	_, dir, _ := cachemeta.ShapeGLMTurnSegmentWitnessed(segs, revoked)
	return ApplyBreakToMessages(msgs, segs, dir)
}

// contentWitness is the externally-refutable witness for a tool result: the content
// identity of the external resource at read time. It is refuted (revoked) when the same
// resource is re-read under DIFFERENT content — NOT a hash of nothing, but of the actual
// external bytes the tool returned, so it carries real freshness signal (refusal rule 1).
func contentWitness(content string) string {
	if content == "" {
		return ""
	}
	sum := sha256.Sum256([]byte(content))
	return "content:" + hex.EncodeToString(sum[:8])
}

// GLMCoherenceShaper returns a SELF-CONTAINED HTTPPlanner.CoherenceShaper closure for the
// §A4 live path: it derives the coherence witnesses from the MESSAGE HISTORY alone — no
// loop-level data flow needed. A tool result's content is the external resource's
// content-identity at read time (a valid witness); the matching assistant tool_call
// (by id) identifies the resource (its name + args). When a resource is re-read in the
// history under different content, the tracker publishes a revocation of the earlier,
// now-stale witness, and the turn is shaped so that stale prefix span is broken. The whole
// remaining live wiring is therefore ONE line in the loop:
//
//	planner.CoherenceShaper = agent.GLMCoherenceShaper(tracker)   // tracker persists across turns
func GLMCoherenceShaper(tracker *vdso.WitnessTracker) func([]Message) []Message {
	return func(msgs []Message) []Message {
		// toolCallID -> resource identity, from the assistant tool_calls.
		resourceOf := make(map[string]string)
		for _, m := range msgs {
			for _, tc := range m.ToolCalls {
				if tc.ID != "" {
					resourceOf[tc.ID] = tc.Function.Name + ":" + tc.Function.Arguments
				}
			}
		}
		// toolCallID -> content witness, from the tool results; publish re-read revocations.
		witnessOf := make(map[string]string)
		for _, m := range msgs {
			if (m.Role != "tool" && m.Role != "function") || m.ToolCallID == "" {
				continue
			}
			w := contentWitness(m.Content)
			if w == "" {
				continue
			}
			witnessOf[m.ToolCallID] = w
			if tracker != nil {
				if res := resourceOf[m.ToolCallID]; res != "" {
					tracker.Observe(res, w) // a later differing read of res revokes this earlier witness
				}
			}
		}
		revoked := func(string) bool { return false }
		if tracker != nil {
			revoked = tracker.Revoked
		}
		return ShapeMessages(msgs, func(id string) string { return witnessOf[id] }, revoked)
	}
}

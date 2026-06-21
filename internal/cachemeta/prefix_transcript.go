package cachemeta

// prefix_transcript.go — the §A3 "front-end" seam: map a recorded conversation (roles +
// the ctxmmu write-time SEAL decision per tool result, as produced by cdb.IngestSession
// over a real Claude Code / GLM transcript) into the PromptSegment turns the A3 linter
// analyzes. Kept as a pure mapping over a typed part list so the classification + sealing
// logic is testable without a JSONL; a cmd/ devtool supplies the parts.
//
// The load-bearing link is sealing: a tool_result the gate quarantined (injection /
// secret / pollution) becomes a SegSealed segment, so A3 refuses to count it in any
// provider-cacheable prefix (refusal rule 3 — never re-serve a fak-sealed span from a
// shared hosted prefix). Role classification drives the layout lint; an explicit
// volatility flag marks the cache-killers.

// ConvPart is one ordered piece of a recorded conversation.
type ConvPart struct {
	Role     string // "system" | "tool_schema" | "user" | "assistant" | "tool_result"
	Content  []byte
	Tokens   int64 // 0 => estimated from Content (~4 bytes/token)
	Sealed   bool  // the ctxmmu write-time gate quarantined this part
	Volatile bool  // caller flags known-volatile metadata (timestamp / request id / nonce)
}

// estTokens is a coarse, deterministic token estimate used only when the caller has no
// exact count (~4 bytes/token, min 1 for non-empty). It feeds the linter's RELATIVE
// cacheable/lost accounting; it is never a billed count.
func estTokens(p ConvPart) int64 {
	if p.Tokens > 0 {
		return p.Tokens
	}
	if n := int64(len(p.Content)) / 4; n > 0 {
		return n
	}
	if len(p.Content) > 0 {
		return 1
	}
	return 0
}

// partKind maps a conversation part to its A3 SegmentKind. Precedence is deliberate:
// SEALED wins over everything (a quarantined span must never be cacheable, whatever its
// role), then an explicit volatile flag, then the role.
func partKind(p ConvPart) SegmentKind {
	switch {
	case p.Sealed:
		return SegSealed
	case p.Volatile:
		return SegVolatile
	}
	switch p.Role {
	case "system":
		return SegStable
	case "tool_schema", "tools":
		return SegToolSchema
	case "tool_result":
		return SegToolResult
	default:
		return SegMessage
	}
}

// SegmentsFromParts maps one turn's ordered parts into the PromptSegment list A3
// analyzes, wiring the ctxmmu seal decision into SegSealed.
func SegmentsFromParts(parts []ConvPart) []PromptSegment {
	out := make([]PromptSegment, len(parts))
	for i, p := range parts {
		out[i] = PromptSegment{Kind: partKind(p), Tokens: estTokens(p), Content: p.Content}
	}
	return out
}

// TurnsFromConversation builds the cumulative per-turn prefixes AnalyzeStability
// consumes. Each "assistant" part is a model request, so the prompt for that request is
// every part strictly before it (the running system + tool-schema + prior messages +
// tool results). The result is one turn per assistant request, in order.
func TurnsFromConversation(parts []ConvPart) [][]PromptSegment {
	segs := SegmentsFromParts(parts)
	var turns [][]PromptSegment
	for i, p := range parts {
		if p.Role == "assistant" && i > 0 {
			turns = append(turns, append([]PromptSegment(nil), segs[:i]...))
		}
	}
	return turns
}

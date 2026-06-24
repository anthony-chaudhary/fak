package gateway

import "strings"

// stream_lift_guard.go keeps a text-form tool-call dialect from ever reaching the
// client on the LIVE token stream. The buffered path runs agent.LiftTextToolCalls,
// which promotes a tool call a model emitted as plain CONTENT TEXT — Hermes
// <tool_call>{…}</tool_call>, Llama <|python_tag|>{…}, Mistral [TOOL_CALLS][…], a
// fenced ```json {…}``` block, or a bare JSON object — into a STRUCTURED tool call,
// stripping the recovered span from the content. The kernel adjudicates only the
// structured call, so the buffered client never sees the raw call text (and a DENIED
// call's arguments never reach it at all).
//
// On a naive live stream that hazard returns: the content sink would forward the raw
// dialect bytes the instant the model emits them, BEFORE LiftTextToolCalls strips
// them — leaking the call text (a denied call included) and diverging from the
// buffered path. The native delta.tool_calls channel has no such hazard — it is a
// separate field from delta.content, exactly like an Anthropic tool_use block — so
// the leak is confined to the text-lift dialects.
//
// liftGuard closes it by GUARANTEEING the bytes streamed live are a prefix of the
// content LiftTextToolCalls leaves behind: it streams prose the instant it is
// provably outside any dialect span and withholds a span the moment a dialect opener
// appears (or could appear, split across a chunk boundary). A fenced block is peeked
// so an ordinary code fence still streams live and only a JSON-bodied fence is held.
// Whatever a guard withheld is reconciled against the buffered content at finalize.

// delimitedToolCallTags are the UNAMBIGUOUS text-form tool-call openers
// LiftTextToolCalls recognizes (see internal/agent/toolcall_fallback.go). They never
// occur in ordinary prose, so holding a stream from the opener onward changes nothing
// a normal turn streams.
var delimitedToolCallTags = []string{
	"<tool_call>",
	"<function_call>",
	"<|python_tag|>",
	"[TOOL_CALLS]",
}

// toolCallFenceMarker opens a fenced block. A fence is ambiguous — it may carry a JSON
// tool call (held) or ordinary code (streamed) — so it is peeked rather than held on
// sight (see fenceDecision).
const toolCallFenceMarker = "```"

// liftGuard wraps a raw content emitter so the bytes it forwards live are always a
// prefix of the content the buffered LiftTextToolCalls would leave — i.e. it never
// emits a span that lift would strip into a separately-adjudicated structured call.
type liftGuard struct {
	emit    func(string) error // raw content emitter; lazily opens the SSE stream
	full    strings.Builder    // every content byte seen, in arrival order
	emitted int                // count of full's bytes already passed to emit
}

func newLiftGuard(emit func(string) error) *liftGuard {
	return &liftGuard{emit: emit}
}

// write feeds one upstream content fragment through the guard, streaming as much as is
// provably free of a text-form tool-call dialect and withholding the rest until a
// later fragment (or finalize) resolves it.
func (g *liftGuard) write(frag string) error {
	if frag == "" {
		return nil
	}
	g.full.WriteString(frag)
	upTo := classifyLiftFlush(g.full.String(), g.emitted)
	if upTo <= g.emitted {
		return nil
	}
	out := g.full.String()[g.emitted:upTo]
	g.emitted = upTo
	return g.emit(out)
}

// streamed returns the exact content bytes already delivered to the client, so the
// caller can reconcile them against the buffered (post-lift) content at finalize.
func (g *liftGuard) streamed() string { return g.full.String()[:g.emitted] }

// classifyLiftFlush returns the byte offset in s up to which content is provably safe
// to stream live, given emitted bytes already went out. It never returns less than
// emitted (nothing un-streams) and never more than len(s). A held dialect pins the
// returned offset at the opener, so successive fragments keep re-deciding from there
// until the dialect either completes (stays held, lifted at finalize) or is refuted.
func classifyLiftFlush(s string, emitted int) int {
	region := s[emitted:]
	if region == "" {
		return emitted
	}

	// Bare JSON: when nothing but whitespace has streamed, the whole turn may be a
	// single JSON value, the shape a buried call uses, so hold from its opener for lift
	// to decide. Gated on "only whitespace streamed" rather than emitted==0 because a
	// whitespace-only prefix can flush before the opener arrives — once real prose is
	// out, the turn is not a bare JSON value.
	if strings.TrimSpace(s[:emitted]) == "" {
		if i := firstNonSpaceByte(region); i >= 0 {
			if c := region[i]; c == '{' || c == '[' {
				return emitted + i
			}
		}
	}

	hold := -1 // earliest region offset we must NOT stream past, or -1 for none

	// Unambiguous tag openers: hold from the earliest one.
	for _, tag := range delimitedToolCallTags {
		if i := strings.Index(region, tag); i >= 0 && (hold < 0 || i < hold) {
			hold = i
		}
	}

	// Fences: scan left to right. A JSON-bodied (or not-yet-decidable) fence becomes the
	// hold; an ordinary code fence is skipped so the block still streams live.
	for search := 0; ; {
		f := strings.Index(region[search:], toolCallFenceMarker)
		if f < 0 {
			break
		}
		f += search
		if hold >= 0 && f >= hold {
			break // a tag/bare hold already precedes this fence
		}
		if fenceDecision(region[f:]) != fenceRelease {
			hold = f
			break
		}
		search = f + len(toolCallFenceMarker) // released code fence — look past it
	}

	if hold >= 0 {
		return emitted + hold
	}
	// No dialect to hold: stream everything except a trailing PARTIAL opener that a
	// later fragment might complete.
	return len(s) - unflushableTail(region)
}

type fenceVerdict int

const (
	fenceHold    fenceVerdict = iota // body is a JSON value — a tool-call fence
	fenceRelease                     // body is code/text — ordinary content
	fencePending                     // body has not arrived — hold provisionally
)

// fenceDecision peeks the body of a fence (afterFence begins with the marker) the way
// the fenced-JSON extractor does: an optional "json" hint, then whitespace, then the
// payload. A payload that opens with { or [ is a JSON value the extractor would lift
// (hold); anything else is ordinary code the extractor leaves alone (release); a body
// that has not arrived — or is still only a PARTIAL "json" hint, ambiguous between the
// hint and a code language ("js", "java") — is not yet decidable (pending), so the
// fence is held provisionally until the next fragment resolves it.
func fenceDecision(afterFence string) fenceVerdict {
	body := afterFence[len(toolCallFenceMarker):]
	switch {
	case strings.HasPrefix(body, "json"):
		body = body[len("json"):] // complete hint — strip it and read the payload
	case body != "" && strings.HasPrefix("json", body):
		return fencePending // a proper prefix of the hint is still streaming
	}
	trimmed := strings.TrimLeft(body, " \t\r\n")
	if trimmed == "" {
		return fencePending
	}
	if trimmed[0] == '{' || trimmed[0] == '[' {
		return fenceHold
	}
	return fenceRelease
}

// unflushableTail returns the length of the longest suffix of region that is a PROPER
// prefix of a tool-call opener (a delimited tag or the fence marker) — a partial
// opener split across a chunk boundary that must be withheld until the next fragment
// completes or refutes it. A COMPLETE opener is found by classifyLiftFlush's Index
// scan, not here. All openers are ASCII, so the cut never lands inside a multibyte
// rune.
func unflushableTail(region string) int {
	best := 0
	consider := func(opener string) {
		limit := len(opener) - 1 // proper prefixes only
		if limit > len(region) {
			limit = len(region)
		}
		for k := limit; k > best; k-- {
			if strings.HasSuffix(region, opener[:k]) {
				best = k
				break
			}
		}
	}
	for _, tag := range delimitedToolCallTags {
		consider(tag)
	}
	consider(toolCallFenceMarker)
	return best
}

func firstNonSpaceByte(s string) int {
	for i := 0; i < len(s); i++ {
		switch s[i] {
		case ' ', '\t', '\r', '\n':
		default:
			return i
		}
	}
	return -1
}

// liftRemainder returns the content the caller must still emit after the live stream:
// the buffered (post-lift) content beyond the prose already streamed. The streamed
// prose is a prefix of cleaned except where lift trimmed leading whitespace, which the
// second branch absorbs. The third branch is a safety net for an unexpected
// divergence (a dialect that reached the wire before lift stripped it): it emits
// cleaned past the common prefix so no prose is dropped, accepting that the
// already-streamed — but still adjudicated — call text cannot be recalled.
func liftRemainder(streamed, cleaned string) string {
	if strings.HasPrefix(cleaned, streamed) {
		return cleaned[len(streamed):]
	}
	if sp := strings.TrimLeft(streamed, " \t\r\n"); strings.HasPrefix(cleaned, sp) {
		return cleaned[len(sp):]
	}
	return cleaned[commonPrefixLen(streamed, cleaned):]
}

func commonPrefixLen(a, b string) int {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	for i := 0; i < n; i++ {
		if a[i] != b[i] {
			return i
		}
	}
	return n
}

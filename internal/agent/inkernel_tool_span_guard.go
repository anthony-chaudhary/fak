package agent

import (
	"strings"
	"unicode"
	"unicode/utf8"
)

// toolSpanGuard sits between the in-kernel per-token decode seam and the client sink.
// The decode seam forwards RAW text, before the post-decode tool-call lift strips
// explicit tool-call markup; forwarding it verbatim would leak tool-call syntax into the
// client's prose stream before adjudication ever sees a ToolCall. The guard therefore
// holds text that could be the start of an explicit tool-call span, and once a span is
// confirmed it DROPS it (the span is recovered post-decode by the tool-call lift and
// returned in the final Completion, never as streamed content). Prose before and after a
// span is forwarded normally, so the concatenated content remains a prefix of the final
// post-lift Content the caller reconciles against.
//
// Legacy span-only users guard explicit tag/delimiter openers. The production
// deferTools mode also delays possible fenced/bare JSON lifts until the final
// offered-set verdict; those bytes are emitted unchanged when they remain content. Partial openers split across token pieces
// are held until they resolve (open) or definitively cannot be an opener (flushed).
type toolSpanGuard struct {
	onReasoningClose    func()
	utf8Tail            string
	deferTools          bool
	deferred            bool
	emit                StreamSink
	held                strings.Builder
	span                strings.Builder
	inSpan              bool
	spanReasoning       bool
	trimReasoningOutput bool
	contentStarted      bool
	trailingSpace       strings.Builder
	closers             []string
	cap                 int
	// reasoning, when set, receives the first <think> span's body (vLLM's streaming
	// reasoning_content), trimmed the way splitReasoning trims it, and only while the
	// bytes seen so far already fix the final split; StreamReasoningRemainder covers
	// the rest.
	reasoning StreamSink
	// reasoningAtClose holds a seeded span until its closer. Complete counts the seed's
	// open tag only under a thinking budget, which the stream knows only when it is
	// explicit, so without one an unclosed seeded span may still split as content.
	reasoningAtClose bool
	reasoningStarted bool
	reasoningDone    bool
	reasoningSpace   strings.Builder
	proseSeen        bool
}

// toolSpanOpeners lists the explicit tool-call span openers the guard suppresses, each
// paired with the closers that end its span. An opener with no closer drops to the end
// of the turn (there is no reliable terminator in that dialect).
var toolSpanOpeners = []struct {
	open      string
	closers   []string
	reasoning bool
}{
	{open: thinkOpen, closers: []string{thinkClose}, reasoning: true},
	{open: "<tool_call>", closers: []string{"</tool_call>"}},
	{open: "<function_call>", closers: []string{"</function_call>"}},
	{open: "<|python_tag|>", closers: []string{"<|eom_id|>", "<|eot_id|>"}},
	{open: "[TOOL_CALLS]", closers: nil},
	{open: deepSeekDSMLCallsOpen, closers: []string{deepSeekDSMLCallsClose}},
}

const toolSpanGuardMaxBytes = 4 << 20

func newToolSpanGuard(emit StreamSink) *toolSpanGuard {
	return newToolSpanGuardState(emit, false)
}

func newToolSpanGuardState(emit StreamSink, startsInReasoning bool) *toolSpanGuard {
	g := &toolSpanGuard{emit: emit, cap: toolSpanGuardMaxBytes}
	if startsInReasoning {
		g.inSpan = true
		g.spanReasoning = true
		g.closers = []string{thinkClose}
	}
	return g
}

// feed consumes one raw decoded piece and forwards only prose.
func (g *toolSpanGuard) feed(piece string) error {
	if g.deferTools {
		piece = g.utf8Tail + piece
		g.utf8Tail = ""
		end := 0
		for end < len(piece) {
			if !utf8.FullRuneInString(piece[end:]) {
				break
			}
			_, size := utf8.DecodeRuneInString(piece[end:])
			end += size
		}
		g.utf8Tail = strings.Clone(piece[end:]) // own at most UTFMax-1 incomplete bytes
		piece = piece[:end]
	}
	return g.feedText(piece)
}

func (g *toolSpanGuard) feedText(piece string) error {
	if g.deferred {
		return nil
	}
	if g.inSpan {
		g.span.WriteString(piece)
		return g.drainSpan()
	}
	g.held.WriteString(piece)
	return g.drainHeld()
}

// flush emits any held non-span text at end of turn. A confirmed span is never flushed.
func (g *toolSpanGuard) flush() error {
	if g.deferred || g.inSpan {
		return nil
	}
	rest := g.held.String()
	g.held.Reset()
	if err := g.emitText(rest); err != nil {
		return err
	}
	// splitReasoning trims final content only after a reasoning span. Retained
	// whitespace is therefore a trailing trim candidate and must not be emitted.
	g.trailingSpace.Reset()
	return nil
}

// drainSpan drops buffered span text through the first closer, then hands any trailing
// text back to the prose scanner. A span that never closes (or an over-cap buffer) is
// dropped to end of turn without ever reaching the sink.
func (g *toolSpanGuard) drainSpan() error {
	buf := g.span.String()
	for _, closer := range g.closers {
		if idx := strings.Index(buf, closer); idx >= 0 {
			tail := buf[idx+len(closer):]
			wasReasoning := g.spanReasoning
			if g.liveReasoning() {
				body := buf[:idx]
				if g.reasoningAtClose && strings.Contains(body, thinkOpen) {
					// Only Complete's budget decides whether the split skips to this
					// inner open, so the remainder carries the whole span.
					body = ""
				}
				if err := g.emitReasoning(body); err != nil {
					return err
				}
				g.reasoningSpace.Reset()
				g.reasoningDone = true
			}
			g.span.Reset()
			g.inSpan = false
			g.spanReasoning = false
			g.closers = nil
			if wasReasoning {
				g.trimReasoningOutput = true
				if g.onReasoningClose != nil {
					g.onReasoningClose()
				}
			}
			g.held.WriteString(tail)
			return g.drainHeld()
		}
	}
	if g.liveReasoning() && !g.reasoningAtClose {
		// Forward the reasoning body as it decodes; only a possible split closer waits.
		keep := partialCloserSuffix(buf, g.closers)
		if err := g.emitReasoning(buf[:len(buf)-keep]); err != nil {
			return err
		}
		g.span.Reset()
		g.span.WriteString(buf[len(buf)-keep:])
		return nil
	}
	if g.closers == nil || len(buf) > g.cap {
		// No terminator will arrive: drop the rest of the turn.
		if g.liveReasoning() {
			g.reasoningDone = true
		}
		g.span.Reset()
	}
	return nil
}

// liveReasoning reports whether the open span is the first reasoning span and a
// reasoning sink wants its bytes. Later spans keep the held behavior because
// splitReasoning reports only the first block.
func (g *toolSpanGuard) liveReasoning() bool {
	return g.reasoning != nil && g.inSpan && g.spanReasoning && !g.reasoningDone
}

// emitReasoning applies splitReasoning's TrimSpace incrementally: leading
// whitespace is dropped and trailing whitespace waits for later reasoning text.
func (g *toolSpanGuard) emitReasoning(text string) error {
	if !g.reasoningStarted {
		text = strings.TrimLeftFunc(text, unicode.IsSpace)
		if text == "" {
			return nil
		}
		g.reasoningStarted = true
	}
	text = g.reasoningSpace.String() + text
	g.reasoningSpace.Reset()
	visible := strings.TrimRightFunc(text, unicode.IsSpace)
	g.reasoningSpace.WriteString(text[len(visible):])
	if visible == "" {
		return nil
	}
	return g.reasoning(visible)
}

// partialCloserSuffix is the length of the longest suffix of s that is a proper
// prefix of one of closers.
func partialCloserSuffix(s string, closers []string) int {
	best := 0
	for _, closer := range closers {
		for n := min(len(closer)-1, len(s)); n > best; n-- {
			if strings.HasPrefix(closer, s[len(s)-n:]) {
				best = n
				break
			}
		}
	}
	return best
}

// drainHeld scans buffered prose for the earliest confirmed opener. Text before it is
// emitted; the opener switches the guard into span mode. If the buffer ends in a partial
// opener prefix, that suffix is held until the next piece resolves it.
func (g *toolSpanGuard) drainHeld() error {
	buf := g.held.String()
	earliest := -1
	var matched []string
	matchedReasoning := false
	for _, o := range toolSpanOpeners {
		if idx := strings.Index(buf, o.open); idx >= 0 && (earliest < 0 || idx < earliest) {
			earliest, matched, matchedReasoning = idx, o.closers, o.reasoning
		}
	}
	if g.deferTools {
		for _, opener := range []string{"<function=", "`", "{", "["} {
			if idx := strings.Index(buf, opener); idx >= 0 && (earliest < 0 || idx < earliest) {
				earliest, matched, matchedReasoning = idx, nil, false
			}
		}
	}
	if earliest >= 0 {
		if err := g.emitText(buf[:earliest]); err != nil {
			return err
		}
		if g.deferTools && !matchedReasoning {
			g.deferred = true
			g.held.Reset()
			return nil
		}
		tail := buf[earliest+lenMatchedOpener(buf[earliest:]):]
		g.held.Reset()
		g.span.Reset()
		g.span.WriteString(tail)
		g.inSpan = true
		g.spanReasoning = matchedReasoning
		g.closers = matched
		if matchedReasoning && g.proseSeen {
			// Earlier prose may hold the </think> the final split reasons up to.
			g.reasoningDone = true
		}
		return g.drainSpan()
	}
	if keep := partialOpenerSuffix(buf); keep > 0 {
		if err := g.emitText(buf[:len(buf)-keep]); err != nil {
			return err
		}
		rest := buf[len(buf)-keep:]
		g.held.Reset()
		g.held.WriteString(rest)
		return nil
	}
	g.held.Reset()
	return g.emitText(buf)
}

func (g *toolSpanGuard) emitText(text string) error {
	if text == "" {
		return nil
	}
	if g.reasoning != nil && !g.proseSeen {
		g.proseSeen = strings.TrimSpace(text) != ""
	}
	if g.deferTools {
		if g.deferred {
			return nil
		}
		if !g.contentStarted && strings.TrimLeftFunc(text, unicode.IsSpace) != text {
			g.deferred = true
			return nil
		}
		g.contentStarted = true
		text = g.trailingSpace.String() + text
		g.trailingSpace.Reset()
		visible := strings.TrimRightFunc(text, unicode.IsSpace)
		if len(text)-len(visible) > g.cap {
			g.deferred = true
		} else if len(visible) < len(text) {
			g.trailingSpace.WriteString(text[len(visible):])
		}
		if visible == "" {
			return nil
		}
		return g.emit(visible)
	}

	if !g.trimReasoningOutput {
		return g.emit(text)
	}
	if !g.contentStarted {
		text = strings.TrimLeftFunc(text, unicode.IsSpace)
		if text == "" {
			return nil
		}
		g.contentStarted = true
	}
	text = g.trailingSpace.String() + text
	g.trailingSpace.Reset()
	visible := strings.TrimRightFunc(text, unicode.IsSpace)
	if len(visible) < len(text) {
		g.trailingSpace.WriteString(text[len(visible):])
	}
	if visible == "" {
		return nil
	}
	return g.emit(visible)
}

// lenMatchedOpener returns the byte length of the opener that prefixes s, so the opener
// itself is consumed into the span buffer (and later dropped) rather than re-scanned.
func lenMatchedOpener(s string) int {
	for _, o := range toolSpanOpeners {
		if strings.HasPrefix(s, o.open) {
			return len(o.open)
		}
	}
	return 0
}

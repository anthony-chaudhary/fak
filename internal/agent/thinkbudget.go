package agent

// thinkbudget.go — a per-turn reasoning-token budget. It runs a counter over the
// tokens a model emits INSIDE its <think>…</think> reasoning span and reports the
// moment that budget is spent, so a caller can force the reasoning-end marker NOW
// instead of leaving the reasoning length to the provider's discretion.
//
// fak today only forwards a provider's coarse reasoning-effort string; it never
// bounds the tokens the model actually spends thinking. This counter turns that
// spend into a bounded, measured quantity: same tokens in, same result out, no
// wall clock, no GPU. It mirrors sglang's reasoner budget, whose _can_think_more
// rule allows a further reasoning token only while (limit < 0) OR (count < limit):
//
//   - limit  < 0  => unlimited. The counter never forces; the model reasons freely.
//   - limit == 0  => the first reasoning token already trips the force. No reasoning
//                    token is ever permitted; the span is closed immediately.
//   - limit  > 0  => exactly `limit` reasoning tokens are permitted. The token that
//                    would be the (limit+1)-th trips the force.
//
// Force is a ONE-WAY latch: Observe returns true EXACTLY ONCE — on the token that
// spends the budget — and reports the span forced-closed thereafter, so it does not
// re-raise on every following token. Forced reads the latched state.
//
// Only tokens INSIDE the reasoning span count. Tokens before the open marker and
// tokens at/after the close marker are ignored, so a leading prompt or a trailing
// answer never eats into the reasoning budget. The open/close markers reuse the
// package's thinkOpen / thinkClose tags; a marker split across a token boundary is
// detected via a small rolling suffix, matching case-insensitively like the sibling
// reasoning strip.

import "strings"

// tailKeep is the number of trailing bytes carried between Observe calls so a marker
// split across two tokens is still detected. It is one less than the longest marker
// (len(thinkClose) == 8), which is all that could begin a marker at a token boundary.
const tailKeep = len(thinkClose) - 1

// ThinkBudget counts reasoning tokens against a per-turn budget and reports when the
// reasoning-end marker must be forced. The zero value is unusable; build one with
// NewThinkBudget. It is not safe for concurrent use — one decode stream per instance.
type ThinkBudget struct {
	limit  int    // reasoning-token budget: <0 unlimited, 0 force at once, >0 that many
	inSpan bool   // currently inside the <think>…</think> reasoning span
	forced bool   // one-way latch: the force signal has already been raised
	count  int    // reasoning tokens counted in the span so far
	tail   string // rolling lower-cased suffix for cross-token marker detection
}

// NewThinkBudget builds a counter for one decode stream. limit is the reasoning-token
// budget (a negative limit means unlimited; zero forbids any reasoning token; a positive
// limit permits exactly that many). Set startInSpan true when the reasoning span is
// already open at the first observed token — e.g. a prompt pre-seeded with an open
// <think> — so the first token counts without waiting for an open marker.
func NewThinkBudget(limit int, startInSpan bool) *ThinkBudget {
	return &ThinkBudget{limit: limit, inSpan: startInSpan}
}

// Observe records that tok was just emitted and returns whether the reasoning-end
// marker must be forced right now. It returns true EXACTLY ONCE — on the token that
// spends the budget — and false on every other call (before the span, after a natural
// close, and after the latch has already fired). Once it forces, the span is treated
// as closed so no later token can re-raise the signal.
func (b *ThinkBudget) Observe(tok string) bool {
	if b.forced {
		return false
	}
	win := b.tail + strings.ToLower(tok)

	if !b.inSpan {
		// Waiting for the reasoning span to open. The token carrying the open marker
		// is the marker itself, not a reasoning token, so it never counts.
		if strings.Contains(win, thinkOpen) {
			b.inSpan = true
			b.tail = ""
			return false
		}
		b.tail = keepTail(win)
		return false
	}

	// Inside the span. A token that closes the span is the marker, not a reasoning
	// token — the model ended thinking on its own, so no force is needed.
	if strings.Contains(win, thinkClose) {
		b.inSpan = false
		b.tail = ""
		return false
	}

	// A genuine reasoning token. Check the budget BEFORE counting it: if no further
	// reasoning token is permitted, this token trips the force and closes the span.
	if !b.canThinkMore() {
		b.forced = true
		b.inSpan = false
		b.tail = ""
		return true
	}
	b.count++
	b.tail = keepTail(win)
	return false
}

// canThinkMore reports whether one more reasoning token is permitted under the budget,
// mirroring sglang's rule: a negative limit is unlimited, otherwise the count of
// reasoning tokens so far must be strictly below the limit.
func (b *ThinkBudget) canThinkMore() bool {
	return b.limit < 0 || b.count < b.limit
}

// Forced reports whether the force signal has been raised. Once true it stays true for
// the life of the counter.
func (b *ThinkBudget) Forced() bool { return b.forced }

// Count reports how many reasoning tokens have been counted inside the span so far.
func (b *ThinkBudget) Count() int { return b.count }

// InSpan reports whether the counter currently considers itself inside the reasoning
// span. It is false before the open marker, after a natural close marker, and after a
// forced exit.
func (b *ThinkBudget) InSpan() bool { return b.inSpan }

// keepTail returns the trailing bytes of win that could still begin a marker at the
// next token boundary — at most tailKeep bytes. This is what lets a marker split across
// two tokens be detected on the following Observe.
func keepTail(win string) string {
	if len(win) > tailKeep {
		return win[len(win)-tailKeep:]
	}
	return win
}

// ForceIndex runs a whole token stream through a fresh counter and returns the index of
// the token at which the reasoning-end marker is forced, or -1 if the budget is never
// spent (unlimited, a natural close, or a stream that stays under budget). startInSpan
// has the same meaning as in NewThinkBudget. It is deterministic and wall-clock-free.
func ForceIndex(tokens []string, limit int, startInSpan bool) int {
	b := NewThinkBudget(limit, startInSpan)
	for i, tok := range tokens {
		if b.Observe(tok) {
			return i
		}
	}
	return -1
}

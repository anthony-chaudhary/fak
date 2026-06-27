package agent

// reasoning_strip.go — strips qwen3-style <think>…</think> reasoning blocks out of in-kernel
// model output before it is handed back into the harness (Claude Code) context. A local model
// served by fak's own kernel (GLM-5.2 / Qwen-3.6-27B / Ornith) emits its chain-of-thought
// inline, wrapped in <think>…</think>, ahead of the user-visible answer. That reasoning is for
// the model's own decode, not for the caller — left in, it leaks into the downstream transcript
// and pollutes every later prefix the harness rebuilds.
//
// The transform is a pure string shrink (strings only, no regexp, no allocation when there is
// nothing to strip), so it is cheap and deterministic: same input → same output, which keeps a
// caller's prefix cache stable. Tags are matched case-insensitively because some checkpoints
// emit <Think>/<THINK>; only the canonical paired form is recognised.
//
// Truncation policy: a streamed block may be cut off mid-thought, so an OPEN <think> with no
// matching </think> is stripped from the open tag to end-of-string. The reasoning is never
// content we want to surface, so dropping an unterminated tail is the fail-safe choice — better
// to lose a partial answer fragment that trailed a runaway think block than to leak the block.

import "strings"

const (
	thinkOpen  = "<think>"
	thinkClose = "</think>"
)

// StripReasoning removes qwen3-style <think>…</think> reasoning blocks from in-kernel model
// output so the model's chain-of-thought does not leak into the downstream (Claude Code)
// context. It handles any number of well-formed blocks; an UNCLOSED <think> (no matching
// </think>, e.g. a truncated stream) is stripped from the open tag to the end of the string.
// Tags are matched case-insensitively. Input with no <think> tag is returned UNCHANGED (identity,
// no allocation). After stripping, surrounding whitespace is tidied so a removed leading block
// does not leave a blank line at the top: the result is left-trimmed and internal runs of three
// or more newlines created by the removal are collapsed to two.
func StripReasoning(s string) string {
	lower := strings.ToLower(s)
	if !strings.Contains(lower, thinkOpen) {
		return s // identity fast-path — nothing to strip, no allocation
	}
	var b strings.Builder
	b.Grow(len(s))
	for {
		open := strings.Index(lower, thinkOpen)
		if open < 0 {
			b.WriteString(s) // no more open tags — keep the rest verbatim
			break
		}
		b.WriteString(s[:open]) // text before the block survives
		rest := lower[open+len(thinkOpen):]
		close := strings.Index(rest, thinkClose)
		if close < 0 {
			// Unclosed <think>: drop from the open tag to end-of-string (truncated-stream policy).
			break
		}
		// Advance past the closing tag in both the original and the lowercased mirror.
		consumed := open + len(thinkOpen) + close + len(thinkClose)
		s = s[consumed:]
		lower = lower[consumed:]
	}
	return tidyAfterStrip(b.String())
}

// tidyAfterStrip removes the whitespace artifacts a strip leaves behind: a leading blank line
// from a removed top-of-string block, a trailing space/newline left when a block (or an unclosed
// truncated block) was the tail of the string, and any run of three-or-more newlines (an empty
// line that used to hold a block) collapsed back to a single blank line.
func tidyAfterStrip(s string) string {
	for strings.Contains(s, "\n\n\n") {
		s = strings.ReplaceAll(s, "\n\n\n", "\n\n")
	}
	return strings.TrimSpace(s)
}

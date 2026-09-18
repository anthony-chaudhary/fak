package agent

import "strings"

// ReasoningParser splits a decoded assistant turn into its reasoning and content
// segments. It is the function signature the named-parser registry resolves to, so
// a caller can select a segmentation policy by the same NAME SGLang/vLLM expose on
// the wire (`--reasoning-parser qwen3`).
type ReasoningParser func(string) (reasoning, content string)

// Named reasoning-parser selector. These are the parity surface for SGLang/vLLM's
// `--reasoning-parser` flag: ReasoningParserQwen3 selects the exact in-kernel
// equivalent of vLLM's `qwen3` parser. An UNSET (or unknown) selector resolves to
// splitReasoning, which reproduces the current segmentation byte-for-byte.
const (
	ReasoningParserQwen3 = "qwen3"
)

// reasoningParsers is the name -> parser registry. ReasoningParserQwen3 reuses
// splitReasoning VERBATIM (same function value; no copied body), so a named
// selection cannot drift from the default behavior.
var reasoningParsers = map[string]ReasoningParser{
	ReasoningParserQwen3: splitReasoning,
}

// LookupReasoningParser resolves a named reasoning parser (e.g. "qwen3") to its
// split function. The name is matched case-insensitively and whitespace-trimmed.
// An unset/blank name resolves to the default parser so the default path is always
// available; an unknown name reports false rather than guessing a dialect.
func LookupReasoningParser(name string) (ReasoningParser, bool) {
	key := strings.ToLower(strings.TrimSpace(name))
	if key == "" {
		return splitReasoning, true
	}
	p, ok := reasoningParsers[key]
	return p, ok
}

// SelectReasoningParser returns the parser for a named selector. An UNSET/blank
// selector — and an unknown one — returns splitReasoning, the default that
// reproduces the current segmentation byte-for-byte. Callers that need to
// distinguish "unknown" from "unset" should use LookupReasoningParser.
func SelectReasoningParser(name string) ReasoningParser {
	if p, ok := LookupReasoningParser(name); ok {
		return p
	}
	return splitReasoning
}

// ValidReasoningParsers returns the supported named reasoning-parser selectors.
// Mirrors ValidReasoningProfiles in effort_budget.go.
func ValidReasoningParsers() []string {
	return []string{ReasoningParserQwen3}
}

// splitReasoning is the preserving sibling of StripReasoning (reasoning_strip.go):
// where StripReasoning DROPS the <think>…</think> block, splitReasoning RETURNS it so
// the planner can stash the reasoning in Message.ReasoningContent instead of losing it.
// It also handles the prompt-pre-seeded "close tag only" case that StripReasoning does
// not (StripReasoning requires an open <think>). The shared tag constants thinkOpen /
// thinkClose live in reasoning_strip.go. For Qwen3.5 these are PLAIN BPE token sequences,
// not atomic special tokens, so the split operates on the DECODED STRING the planner
// already accumulates — never on token ids. This is the in-kernel equivalent of vLLM's
// --reasoning-parser qwen3.

// splitReasoning separates a leading `<think>…</think>` reasoning block from the
// final answer in a decoded assistant turn. It returns the trimmed reasoning and
// the trimmed content that follows the close tag.
//
// It is deterministic and gated so a NON-reasoning model is unaffected: when the
// input carries no close tag, reasoning is "" and content is returned byte-for-byte
// (no trim, no copy) — identical to the pre-seam behavior. The cases it handles:
//
//   - open + close (the normal Ornith emission, since renderChatMLTools does NOT
//     pre-seed the open tag and the model emits both): everything between the first
//     `<think>` and the first following `</think>` is reasoning; everything after
//     that `</think>` is content. Leading whitespace before `<think>` is tolerated.
//   - close only, no open: some templates pre-seed `<think>\n` into the prompt so
//     the model only EMITS the close tag. Then the decoded output starts mid-reasoning
//     and the whole prefix up to the first `</think>` is reasoning; the rest is content.
//   - no think tags at all: return ("", s) with s untouched — the non-reasoning path.
//   - unclosed `<think>` (reasoning still in progress, e.g. a length cutoff inside the
//     block): the whole remainder is reasoning and content is "" — never leak a
//     half-emitted reasoning block as the answer.
func splitReasoning(s string) (reasoning, content string) {
	// Fast path / non-reasoning gate: no close tag means there is no completed
	// reasoning block to strip. Return the input untouched so non-reasoning turns are
	// byte-identical to today. (An unclosed open tag is handled just below this; it
	// also has no close tag, so check it first.)
	closeIdx := strings.Index(s, thinkClose)
	if closeIdx < 0 {
		// No close tag. If an open tag is present, the model is still inside an
		// unclosed reasoning block (e.g. truncated by maxNew); treat the whole thing
		// as in-progress reasoning so it does not leak into the answer.
		if openIdx := strings.Index(s, thinkOpen); openIdx >= 0 {
			return strings.TrimSpace(s[openIdx+len(thinkOpen):]), ""
		}
		return "", s
	}

	// A close tag exists. Find where the reasoning begins: just after an open tag if
	// the model emitted one, otherwise at the start of the string (the pre-seeded
	// "close-only" case). Only honor an open tag that precedes the close tag — a
	// `<think>` appearing AFTER the first `</think>` belongs to the content, not here.
	reasonStart := 0
	if openIdx := strings.Index(s, thinkOpen); openIdx >= 0 && openIdx < closeIdx {
		reasonStart = openIdx + len(thinkOpen)
	}

	reasoning = strings.TrimSpace(s[reasonStart:closeIdx])
	content = strings.TrimSpace(s[closeIdx+len(thinkClose):])
	return reasoning, content
}

// Token-level prompt assembly and the BPE-stable terminator probe.
//
// ADAPT of the prompt-generation seam from ai-dynamo/aiperf
// (PromptGenerator.generate_prompt / _decode_to_exact_len /
// _determine_bpe_stable_terminator), pinned at
// https://github.com/ai-dynamo/aiperf/blob/ab610add4e8876fca223197596c728e17365f381/src/aiperf/dataset/generator/prompt.py
// (commit ab610add4e8876fca223197596c728e17365f381, Apache-2.0). Re-expressed
// in idiomatic Go against a minimal Tokenizer interface; no upstream code is
// copied. Study note: fak-private docs/research/STUDY-aiperf-2026-11-09.md,
// candidate aiperf-data-6.
//
// Why the seam exists: joining decoded prompt segments as strings lets the
// server-side BPE charge for the seam (upstream measured ~0.84 extra tokens
// per request, AIP-1118). Joining at the token-ID level keeps the wire ISL
// equal to the sum of the per-segment token counts, and the bounded
// decode/re-encode correction loop absorbs round-trip drift so the final
// prompt encodes to exactly len(prefix)+len(body) tokens. The terminator
// probe finds a short token sequence that can be appended between segments
// without re-merging across the seam, so multi-segment synthesis stays
// byte-exact under the join formula.
package main

import (
	"errors"
	"fmt"
)

// Tokenizer is the closed encode/decode contract the prompt seam needs. Any
// real tokenizer (HF-style, tiktoken-style) or a deterministic test fake
// satisfies it; loadgen itself carries no third-party tokenizer dependency.
type Tokenizer interface {
	// Encode returns the token IDs for text.
	Encode(text string) []int
	// Decode returns the text for token IDs.
	Decode(tokens []int) string
}

var (
	// ErrEmptyPrompt marks an assembly call with neither prefix nor body
	// tokens; there is nothing to assemble.
	ErrEmptyPrompt = errors.New("loadgen/prompt: no tokens to assemble")
	// ErrLengthMismatch marks a prompt whose re-encoded length could not be
	// corrected to the exact target within the retry bound (or with no top-up
	// source available). Callers match it with errors.Is.
	ErrLengthMismatch = errors.New("loadgen/prompt: cannot reach exact token length")
)

// maxPromptCorrectionRetries bounds the decode/re-encode correction loop,
// mirroring the upstream _max_retries default.
const maxPromptCorrectionRetries = 10

// AssemblePrompt builds one synthetic prompt at the token-ID level:
// prefixTokens and bodyTokens are joined before any decoding, so when the
// round-trip is stable the wire prompt encodes to exactly
// len(prefixTokens)+len(bodyTokens) tokens. String-concatenating the decoded
// segments instead would let BPE charge for the seam.
//
// The joined sequence is then corrected to its exact length by
// CorrectToExactLength: decode, re-encode, trim or top up until the
// re-encoded sequence has exactly the target count. topup supplies extra
// tokens when the re-encoding falls short; it may be nil only when the
// initial join already round-trips exactly.
func AssemblePrompt(tok Tokenizer, prefixTokens, bodyTokens []int, topup func(n int) []int) (string, error) {
	if tok == nil {
		return "", fmt.Errorf("%w: nil tokenizer", ErrEmptyPrompt)
	}
	tokens := make([]int, 0, len(prefixTokens)+len(bodyTokens))
	tokens = append(tokens, prefixTokens...)
	tokens = append(tokens, bodyTokens...)
	if len(tokens) == 0 {
		return "", fmt.Errorf("%w: prefix and body are both empty", ErrEmptyPrompt)
	}
	_, text, err := CorrectToExactLength(tok, tokens, len(tokens), topup)
	return text, err
}

// CorrectToExactLength drives tokens through repeated decode/re-encode cycles
// until the re-encoded sequence has exactly target tokens. Each cycle:
// decode to text, re-encode; if longer than target, trim to target; if
// shorter, extend with topup(target-len) draws. The re-encoded sequence is
// carried into the next cycle (not the pre-correction one), so trimming acts
// on what the tokenizer actually produced. The loop is bounded by
// maxPromptCorrectionRetries cycles; a final verification pass accepts a
// state that converged on the boundary. When the bound exhausts without
// reaching target, or a top-up draw makes no progress, it reports
// ErrLengthMismatch wrapped with the observed lengths. On success it returns
// the final token sequence and the decoded text whose re-encoding has
// exactly target tokens.
func CorrectToExactLength(tok Tokenizer, tokens []int, target int, topup func(n int) []int) ([]int, string, error) {
	cur := append([]int(nil), tokens...)
	stuck := false
	for attempt := 0; attempt < maxPromptCorrectionRetries && !stuck; attempt++ {
		text := tok.Decode(cur)
		reenc := tok.Encode(text)
		switch {
		case len(reenc) == target:
			return reenc, text, nil
		case len(reenc) > target:
			cur = reenc[:target]
		default: // len(reenc) < target: top up the shortfall
			if topup == nil {
				stuck = true
				break
			}
			extra := topup(target - len(reenc))
			if len(extra) == 0 {
				stuck = true // no progress possible
				break
			}
			cur = append(append([]int(nil), reenc...), extra...)
		}
	}
	text := tok.Decode(cur)
	reenc := tok.Encode(text)
	if len(reenc) == target {
		return reenc, text, nil
	}
	return nil, "", fmt.Errorf("%w: target %d tokens, best effort %d after %d corrections",
		ErrLengthMismatch, target, len(reenc), maxPromptCorrectionRetries)
}

// bpeStableTerminatorCandidates is the probe ladder, in priority order:
// double newline first (typically a singleton in modern tokenizers), single
// newline as fallback, bare space last (degenerate).
var bpeStableTerminatorCandidates = []string{"\n\n", "\n", " "}

// ProbeBPETerminator probes tok for a short token sequence that, when
// appended to arbitrary content and followed by " " + more content, does not
// cause BPE re-merging across the seam. Two windows of the token corpus are
// decoded to form the probe texts a (corpus[100:164]) and b (corpus[200:264]);
// a candidate passes the byte-exact join probe when
//
//	len(encode(a+cand+" "+b)) == len(encode(a)) + len(encode(cand)) + len(encode(" "+b))
//
// i.e. the join costs exactly the candidate's own tokens plus the leading
// space, with no merge or split charged at either seam. Candidates are tried
// in bpeStableTerminatorCandidates order and the first that passes wins.
// It returns the winning candidate's token IDs, or nil when no candidate
// passes or the corpus is too small to probe (< 264 tokens); callers then
// fall back to no terminator and accept segment-join drift.
func ProbeBPETerminator(tok Tokenizer, corpus []int) []int {
	if tok == nil || len(corpus) < 264 {
		return nil
	}
	a := tok.Decode(corpus[100:164])
	b := tok.Decode(corpus[200:264])
	aTokens := tok.Encode(a)
	bLeadSpace := tok.Encode(" " + b)
	for _, cand := range bpeStableTerminatorCandidates {
		candTokens := tok.Encode(cand)
		if len(candTokens) == 0 {
			continue
		}
		joined := a + cand + " " + b
		if len(tok.Encode(joined)) == len(aTokens)+len(candTokens)+len(bLeadSpace) {
			return append([]int(nil), candTokens...)
		}
	}
	return nil
}

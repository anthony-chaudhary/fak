// Package coherence is the deterministic output-coherence probe shared by the
// engine's readiness gate and fleet-side deploy verification.
//
// Liveness is not coherence. A serving binary paired with a mismatched compute
// kernel set (for example a fak binary loading SPIR-V modules built from a
// different source revision) can bind its listener, report /healthz ok, decode at
// full speed — and emit garbage. Shape heuristics (empty, punctuation-only, one
// token repeated) catch some of that garbage but not all of it: wrong-kernel
// output is frequently fluent-looking noise.
//
// This package closes the gap with a check whose correct answer is known in
// advance: ask the model, greedily (temperature 0), to count from 1 to N, and
// require the decoded text to contain the contiguous integer run 1..N. A coherent
// instruction-tuned model passes trivially; a model whose arithmetic is corrupted
// does not. The check is a pure function of the text so both the engine (public
// fak gateway readiness) and a fleet verifier (private deploy tooling) run the
// identical policy.
package coherence

import (
	"fmt"
	"strconv"
	"strings"
	"unicode"
)

// DefaultCount is the N used by CountPrompt callers that have no reason to pick
// another. Ten numbers is long enough that a corrupted decode cannot pass by
// accident and short enough to cost well under a second of decode.
const DefaultCount = 10

// MaxCount bounds N so a caller cannot turn the readiness probe into an
// expensive decode.
const MaxCount = 64

// Verdict is the closed result vocabulary of Check. Callers branch on these
// values, never on prose.
type Verdict string

const (
	// Coherent means the decoded text contains the contiguous run 1..N.
	Coherent Verdict = "coherent"
	// Empty means the decode produced no visible text.
	Empty Verdict = "empty"
	// SequenceMissing means the decode produced text, but not the run 1..N.
	SequenceMissing Verdict = "sequence_missing"
	// InvalidCount means the caller asked for N outside [1, MaxCount].
	InvalidCount Verdict = "invalid_count"
)

// Verdicts lists every Verdict value; it is the closed vocabulary contract.
func Verdicts() []Verdict {
	return []Verdict{Coherent, Empty, SequenceMissing, InvalidCount}
}

// CountPrompt returns the fixed, code-authored probe prompt for N. It never
// carries client payload, so echoing a bounded sample of the answer is safe.
func CountPrompt(n int) string {
	return fmt.Sprintf("Count from 1 to %d. Reply with only the numbers, separated by spaces.", n)
}

// MaxTokens is the decode ceiling a caller should request for CountPrompt(n):
// a few tokens per number plus headroom for a chat preamble or a short closed
// reasoning block.
func MaxTokens(n int) int {
	if n < 1 {
		n = 1
	}
	return 64 + 4*n
}

// Check judges the decoded answer to CountPrompt(n). Reasoning blocks
// (<think>...</think>) are ignored; the remaining text must contain the integers
// 1..n as a contiguous, in-order run (any separators).
func Check(text string, n int) Verdict {
	if n < 1 || n > MaxCount {
		return InvalidCount
	}
	answer := strings.TrimSpace(stripReasoning(text))
	if answer == "" {
		return Empty
	}
	nums := integers(answer)
	for start := 0; start+n <= len(nums); start++ {
		if nums[start] != 1 {
			continue
		}
		ok := true
		for i := 1; i < n; i++ {
			if nums[start+i] != i+1 {
				ok = false
				break
			}
		}
		if ok {
			return Coherent
		}
	}
	return SequenceMissing
}

// stripReasoning removes a closed <think>...</think> block. An unclosed block is
// kept: a decode that never left its reasoning has not answered.
func stripReasoning(text string) string {
	for {
		open := strings.Index(text, "<think>")
		if open < 0 {
			return text
		}
		end := strings.Index(text[open:], "</think>")
		if end < 0 {
			return text[:open]
		}
		text = text[:open] + text[open+end+len("</think>"):]
	}
}

// integers extracts maximal ASCII digit runs, in order, as integers.
func integers(text string) []int {
	var out []int
	field := func(r rune) bool { return r > unicode.MaxASCII || !unicode.IsDigit(r) }
	for _, f := range strings.FieldsFunc(text, field) {
		v, err := strconv.Atoi(f)
		if err != nil {
			continue
		}
		out = append(out, v)
	}
	return out
}

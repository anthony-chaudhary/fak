package tokenizer

import (
	"strings"
	"unicode"
)

// isDeepSeek3PreTokenizer recognizes the GGUF hints llama.cpp maps to the DeepSeek-V3
// regex set: "deepseek-v3" (LLAMA_VOCAB_PRE_TYPE_DEEPSEEK3_LLM) and "joyai-llm"
// (LLAMA_VOCAB_PRE_TYPE_JOYAI_LLM, the hint DeepSeek-V4.1-Flash GGUFs carry). Both types
// share one regex_exprs case in llama-vocab.cpp, so they share one splitter here. Exact
// match only: hunyuan-dense/hy_v4 also share that case but are not wired yet.
func isDeepSeek3PreTokenizer(pre string) bool {
	switch strings.ToLower(pre) {
	case "deepseek-v3", "joyai-llm":
		return true
	}
	return false
}

// preTokenizeDeepSeek3 ports llama.cpp's DEEPSEEK3_LLM/JOYAI_LLM pre-tokenizer
// (vcruz305/llama.cpp@5210c7c src/llama-vocab.cpp:318-327). Unlike the single-regex
// families, it is THREE regexes applied in sequence by unicode_regex_split: each stage
// re-splits every chunk the previous stage produced, and text a stage does not match is
// kept as one chunk (not split per rune). The stages are:
//
//  1. \p{N}{1,3}
//  2. [一-龥぀-ゟ゠-ヿ]+
//  3. [!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+
//     | [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+
//     |  ?[\p{P}\p{S}]+[\r\n]*
//     | \s*[\r\n]+
//     | \s+(?!\S)
//     | \s+
//
// Stage 1 isolates digit runs, so " 17" becomes " " and "17" (the trailing space is a
// chunk end, which satisfies stage 3's (?!\S)). Because a chunk end is end-of-input to
// the next stage, the stages must run per chunk, never over the joined text.
func preTokenizeDeepSeek3(text string) []string {
	rs := []rune(text)
	if len(rs) == 0 {
		return nil
	}
	spans := []runeSpan{{0, len(rs)}}
	spans = splitSpans(rs, spans, matchDeepSeek3Digits)
	spans = splitSpans(rs, spans, matchDeepSeek3CJK)
	spans = splitSpans(rs, spans, matchDeepSeek3Main)
	out := make([]string, len(spans))
	for i, s := range spans {
		out[i] = string(rs[s.start:s.end])
	}
	return out
}

type runeSpan struct{ start, end int }

// splitSpans is unicode_regex_split_stl over rune spans: within each span it emits every
// leftmost match of matchAt plus the unmatched gaps between them, each gap as ONE span.
// matchAt sees the span end as end-of-input and returns the match length (0 = no match).
func splitSpans(rs []rune, spans []runeSpan, matchAt func(rs []rune, i, end int) int) []runeSpan {
	out := make([]runeSpan, 0, len(spans))
	for _, s := range spans {
		gap := s.start
		for i := s.start; i < s.end; {
			m := matchAt(rs, i, s.end)
			if m == 0 {
				i++
				continue
			}
			if i > gap {
				out = append(out, runeSpan{gap, i})
			}
			out = append(out, runeSpan{i, i + m})
			i += m
			gap = i
		}
		if gap < s.end {
			out = append(out, runeSpan{gap, s.end})
		}
	}
	return out
}

// The stage-3 classes follow llama.cpp's collapsed-category matching: ASCII is matched
// literally, and a non-ASCII rune counts as whitespace first (unicode_set_whitespace,
// the Unicode White_Space set unicode.IsSpace implements), otherwise by its general
// category. Testing whitespace first keeps U+00A0 or U+3000 out of every \p class.
func ds3Space(r rune) bool { return unicode.IsSpace(r) }

// The pinned llama.cpp tables include Unicode 15.1 additions absent from Go's
// Unicode 15.0 tables: CJK Extension I letters and five ideographic symbols.
// Source: vcruz305/llama.cpp@5210c7c5ed61dddaee6ed476623abf4b63093d16,
// src/unicode-data.cpp (MIT; Copyright (c) 2023-2026 The ggml authors).
// TestDeepSeek3UnicodeVersion requires review when Go's Unicode tables change.
func ds3Letter(r rune) bool {
	return !ds3Space(r) && (unicode.IsLetter(r) || (r >= 0x2EBF0 && r <= 0x2EE5D))
}

func ds3Mark(r rune) bool { return !ds3Space(r) && unicode.Is(unicode.M, r) }

func ds3Number(r rune) bool { return !ds3Space(r) && unicode.IsNumber(r) }

func ds3PunctOrSymbol(r rune) bool {
	return !ds3Space(r) && (unicode.IsPunct(r) || unicode.IsSymbol(r) ||
		(r >= 0x2FFC && r <= 0x2FFF) || r == 0x31EF)
}

func ds3LetterOrMark(r rune) bool { return ds3Letter(r) || ds3Mark(r) }

func isASCIILetter(r rune) bool { return (r >= 'A' && r <= 'Z') || (r >= 'a' && r <= 'z') }

// isASCIIPunctOrSymbol is stage 3's first bracket: every printable ASCII non-alnum.
func isASCIIPunctOrSymbol(r rune) bool {
	return r < 0x80 && r > ' ' && r != 0x7f && !isASCIILetter(r) && (r < '0' || r > '9')
}

// matchDeepSeek3Digits is stage 1, \p{N}{1,3}.
func matchDeepSeek3Digits(rs []rune, i, end int) int {
	j := i
	for j < end && j-i < 3 && ds3Number(rs[j]) {
		j++
	}
	return j - i
}

// matchDeepSeek3CJK is stage 2, [一-龥぀-ゟ゠-ヿ]+: U+4E00-U+9FA5 and U+3040-U+30FF.
func matchDeepSeek3CJK(rs []rune, i, end int) int {
	j := i
	for j < end && ((rs[j] >= 0x4E00 && rs[j] <= 0x9FA5) || (rs[j] >= 0x3040 && rs[j] <= 0x30FF)) {
		j++
	}
	return j - i
}

// matchDeepSeek3Main is stage 3. Alternatives are tried in order and the first that
// matches wins (ECMAScript), which matters for \s+(?!\S) versus \s+.
func matchDeepSeek3Main(rs []rune, i, end int) int {
	r := rs[i]

	// [!"#...~][A-Za-z]+
	if isASCIIPunctOrSymbol(r) {
		j := i + 1
		for j < end && isASCIILetter(rs[j]) {
			j++
		}
		if j > i+1 {
			return j - i
		}
	}

	// [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+
	if r != '\r' && r != '\n' && !ds3Letter(r) && !ds3PunctOrSymbol(r) {
		j := i + 1
		for j < end && ds3LetterOrMark(rs[j]) {
			j++
		}
		if j > i+1 {
			return j - i
		}
	}
	if ds3LetterOrMark(r) {
		j := i
		for j < end && ds3LetterOrMark(rs[j]) {
			j++
		}
		return j - i
	}

	//  ?[\p{P}\p{S}]+[\r\n]*
	{
		j := i
		if r == ' ' {
			j++
		}
		start := j
		for j < end && ds3PunctOrSymbol(rs[j]) {
			j++
		}
		if j > start {
			for j < end && (rs[j] == '\r' || rs[j] == '\n') {
				j++
			}
			return j - i
		}
	}

	if !ds3Space(r) {
		return 0
	}
	j := i
	lastNL := -1
	for j < end && ds3Space(rs[j]) {
		if rs[j] == '\r' || rs[j] == '\n' {
			lastNL = j
		}
		j++
	}
	// \s*[\r\n]+ : backtracking \s* stops at the run's last CR/LF.
	if lastNL >= 0 {
		return lastNL - i + 1
	}
	// \s+(?!\S) keeps the run when it reaches the span end and otherwise leaves its last
	// space for the next token; a lone space before \S falls through to \s+.
	if j == end || j-i == 1 {
		return j - i
	}
	return j - i - 1
}

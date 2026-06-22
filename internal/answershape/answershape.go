// Package answershape is a deterministic, dependency-free guard over the SHAPE of
// a piece of text — how repetitive (degenerate) it is and how long (verbose) it
// is — checked against caller-chosen thresholds.
//
// It is the consumer-facing, GRADED dual of the kernel's write-time admit rung.
// The context-MMU's repeat predicate (internal/ctxmmu) is a conservative BINARY
// gate: it quarantines only the most blatant byte-repeat pollution (a 16-byte
// chunk repeated >50 times) so it never wrongly seals a benign result. This
// package answers the softer, tunable question an operator or agent actually asks
// of a candidate answer — "is this output looping or runaway?" — as a graded
// fraction in [0,1] judged against thresholds the caller picks, off the hot path
// and with no kernel dependency. It is to that admit rung what `fak lint` is to
// the kernel's call-time re-checks: the same concern, surfaced as a consumer
// witness instead of an internal-only verdict.
//
// Three repetition sub-signals are computed. Each is ~0 for natural prose and
// each spikes on a distinct degeneration mode; the headline RepeatFraction is
// their maximum, so a text trips on whichever way it actually degenerated:
//
//   - NGramRepeat: 1 - unique/total word n-grams (rep-n, the standard neural-text
//     degeneration metric). Catches looped phrases and repeated sentences.
//   - LineBlockRepeat: runes covered by the single most-frequent non-blank line
//     that occurs at least twice, over total runes. Catches a block/paragraph
//     emitted over and over.
//   - PeriodRepeat: the runes covered by the largest short-period tiling (a unit
//     up to maxPeriod bytes repeated back-to-back), over total runes. Catches a
//     whitespace-free character runaway ("AAAA…") and a tiled unit ("abcabc…",
//     ".assistant.assistant…") that the word- and line-level signals miss. It
//     generalizes the dominantPeriod/looksDegenerate detector in cmd/simpledemo
//     (issue #91) from "is the WHOLE string periodic" to "what fraction is".
//
// Measure is pure and deterministic: identical bytes and identical Limits always
// yield an identical Report.
package answershape

import (
	"fmt"
	"strings"
	"unicode"
	"unicode/utf8"
)

// Defaults for the two knobs the consumer surface exposes.
const (
	// DefaultNGram is the word n-gram width for NGramRepeat. Trigrams catch looped
	// phrases without the false-positive rate of unigrams/bigrams (natural prose
	// repeats single words and common word-pairs constantly).
	DefaultNGram = 3
	// DefaultMaxRepeat is the default degeneration threshold: a RepeatFraction
	// above this is degenerate. Natural prose sits well below it; a looping model
	// climbs above it fast.
	DefaultMaxRepeat = 0.5
)

// repeatFloorChars is the minimum text length (in runes) at which the repetition
// signals are allowed to call a text degenerate. Below it, repetition fractions
// are reported but never trip the verdict — a handful of characters is too short
// to distinguish a loop from a legitimately terse answer. It matches the floor the
// existing degeneration detector uses (cmd/simpledemo looksDegenerate, issue #91).
const repeatFloorChars = 24

// maxPeriod bounds the tiling-unit length PeriodRepeat scans, in bytes. A unit as
// long as a role header (".assistant\n") still tiles within it; natural language
// is never exactly periodic at any of these lengths, so the gate is safe.
const maxPeriod = 32

// Limits are the caller-chosen thresholds the consumer surface binds. The two
// named knobs mirror `dos answer-shape --max-repeat --max-chars`.
type Limits struct {
	// MaxRepeat is the largest RepeatFraction (0..1) treated as in-shape. A report
	// whose RepeatFraction exceeds it is degenerate. <= 0 disables the repeat check.
	MaxRepeat float64 `json:"max_repeat"`
	// MaxChars is the largest rune count treated as in-shape. A report longer than
	// it is over-verbose. <= 0 disables the length check.
	MaxChars int `json:"max_chars"`
	// NGram is the word n-gram width for NGramRepeat. <= 0 uses DefaultNGram.
	NGram int `json:"ngram"`
}

// Report is the measured shape of a text plus the verdict against the Limits it
// was measured under. Every fraction is always populated (informational), even
// when the corresponding check is disabled, so a caller can log the shape without
// committing to a threshold.
type Report struct {
	Bytes int `json:"bytes"`
	Chars int `json:"chars"`
	Words int `json:"words"`
	NGram int `json:"ngram"`

	NGramRepeat     float64 `json:"ngram_repeat"`
	LineBlockRepeat float64 `json:"line_block_repeat"`
	PeriodRepeat    float64 `json:"period_repeat"`
	// RepeatFraction is the headline = max(NGramRepeat, LineBlockRepeat,
	// PeriodRepeat). The verdict's repeat check compares this to Limits.MaxRepeat.
	RepeatFraction float64 `json:"repeat_fraction"`

	// TopNGram is the most-frequent word n-gram (the looped phrase), surfaced for a
	// human-readable reason. Empty when no n-gram repeats.
	TopNGram      string `json:"top_ngram,omitempty"`
	TopNGramCount int    `json:"top_ngram_count,omitempty"`
	// Period and PeriodUnit describe the dominant short-period tiling, when any.
	Period     int    `json:"period,omitempty"`
	PeriodUnit string `json:"period_unit,omitempty"`

	Degenerate bool     `json:"degenerate"`
	Reasons    []string `json:"reasons,omitempty"`
	Limits     Limits   `json:"limits"`
}

// Measure computes the shape Report for text under lim. It is pure: no I/O, no
// globals, deterministic in (text, lim).
func Measure(text []byte, lim Limits) Report {
	n := lim.NGram
	if n <= 0 {
		n = DefaultNGram
	}
	// Work over valid UTF-8 so rune counting and splitting are well-defined.
	s := strings.ToValidUTF8(string(text), "")

	rep := Report{
		Bytes:  len(text),
		Chars:  utf8.RuneCountInString(s),
		NGram:  n,
		Limits: lim,
	}

	words := normalizedWords(s)
	rep.Words = len(words)

	rep.NGramRepeat, rep.TopNGram, rep.TopNGramCount = ngramRepeat(words, n)
	rep.LineBlockRepeat = lineBlockRepeat(s, rep.Chars)
	rep.PeriodRepeat, rep.Period, rep.PeriodUnit = periodRepeat(s, rep.Chars)
	rep.RepeatFraction = max3(rep.NGramRepeat, rep.LineBlockRepeat, rep.PeriodRepeat)

	// Verdict. Repetition only judges text at or above the floor; length judges any
	// length. A disabled check (<= 0) never trips.
	if lim.MaxRepeat > 0 && rep.Chars >= repeatFloorChars && rep.RepeatFraction > lim.MaxRepeat {
		rep.Degenerate = true
		rep.Reasons = append(rep.Reasons, fmt.Sprintf(
			"repetitive: %s %.2f > max-repeat %.2f", rep.dominantSignal(), rep.RepeatFraction, lim.MaxRepeat))
	}
	if lim.MaxChars > 0 && rep.Chars > lim.MaxChars {
		rep.Degenerate = true
		rep.Reasons = append(rep.Reasons, fmt.Sprintf(
			"verbose: %d chars > max-chars %d", rep.Chars, lim.MaxChars))
	}
	return rep
}

// dominantSignal names which sub-signal set the headline RepeatFraction, for the
// human reason string.
func (r Report) dominantSignal() string {
	switch r.RepeatFraction {
	case r.NGramRepeat:
		if r.TopNGram != "" {
			return fmt.Sprintf("%d-gram-repeat (%q ×%d)", r.NGram, r.TopNGram, r.TopNGramCount)
		}
		return fmt.Sprintf("%d-gram-repeat", r.NGram)
	case r.LineBlockRepeat:
		return "line-block-repeat"
	default:
		if r.PeriodUnit != "" {
			return fmt.Sprintf("period-%d-tile (%q)", r.Period, r.PeriodUnit)
		}
		return "period-tile"
	}
}

// normalizedWords splits text into whitespace-delimited tokens, each lowercased
// and stripped of leading/trailing non-alphanumeric runes, so "Yes,", "yes." and
// "yes" collapse to one token. Empty tokens (pure punctuation) are dropped. This
// raises recall of real loops ("yes, yes, yes.") without lowercasing the source.
func normalizedWords(s string) []string {
	fields := strings.Fields(s)
	out := make([]string, 0, len(fields))
	for _, f := range fields {
		t := strings.TrimFunc(strings.ToLower(f), func(r rune) bool {
			return !unicode.IsLetter(r) && !unicode.IsNumber(r)
		})
		if t != "" {
			out = append(out, t)
		}
	}
	return out
}

// ngramRepeat returns rep-n = 1 - unique/total over the word n-grams of words,
// plus the most-frequent n-gram and its count. It returns (0, "", 0) when there
// are fewer than two n-grams (no repetition is expressible).
func ngramRepeat(words []string, n int) (frac float64, top string, topCount int) {
	if n < 1 {
		n = 1
	}
	total := len(words) - n + 1
	if total < 2 {
		return 0, "", 0
	}
	counts := make(map[string]int, total)
	for i := 0; i+n <= len(words); i++ {
		g := strings.Join(words[i:i+n], " ")
		counts[g]++
		if counts[g] > topCount || (counts[g] == topCount && g < top) {
			topCount, top = counts[g], g
		}
	}
	unique := len(counts)
	frac = 1 - float64(unique)/float64(total)
	if topCount < 2 {
		// Nothing actually repeated; do not surface a spurious "top" phrase.
		top, topCount = "", 0
	}
	return frac, top, topCount
}

// lineBlockRepeat returns the fraction of total runes covered by the single
// most-frequent non-blank line that occurs at least twice. It returns 0 when no
// line repeats (so a normal single-paragraph answer, whose one line trivially has
// max frequency 1, scores 0 rather than 1).
func lineBlockRepeat(s string, totalChars int) float64 {
	if totalChars == 0 {
		return 0
	}
	counts := map[string]int{}
	for _, ln := range strings.Split(s, "\n") {
		t := strings.TrimSpace(ln)
		if t != "" {
			counts[t]++
		}
	}
	bestCount, bestLen := 0, 0
	for ln, c := range counts {
		if c < 2 {
			continue
		}
		if c*utf8.RuneCountInString(ln) > bestCount*bestLen {
			bestCount, bestLen = c, utf8.RuneCountInString(ln)
		}
	}
	if bestCount < 2 {
		return 0
	}
	return clamp01(float64(bestCount*bestLen) / float64(totalChars))
}

// periodRepeat finds the largest short-period tiling in s and returns the fraction
// of runes it covers, the period length (in bytes), and a bounded preview of the
// repeating unit. For period p it finds the longest contiguous byte block over
// which s[i] == s[i-p] (a block of `run` matched bytes covers run+p bytes — the
// matched tail plus the unit it copies), requiring at least two full copies. p=1
// is a single-rune runaway; p>=2 is a tiled unit. It is the graded generalization
// of cmd/simpledemo's dominantPeriod (issue #91), which required the WHOLE string
// to be p-periodic.
func periodRepeat(s string, totalChars int) (frac float64, period int, unit string) {
	n := len(s)
	if n < 2 || totalChars == 0 {
		return 0, 0, ""
	}
	bestBytes, bestP, bestStart := 0, 0, 0
	for p := 1; p <= maxPeriod && p <= n/2; p++ {
		run, start := 0, 0
		for i := p; i <= n; i++ {
			if i < n && s[i] == s[i-p] {
				if run == 0 {
					start = i - p
				}
				run++
				continue
			}
			if run >= p { // >= 2 full copies of the p-byte unit
				if block := run + p; block > bestBytes {
					bestBytes, bestP, bestStart = block, p, start
				}
			}
			run = 0
		}
	}
	if bestBytes == 0 {
		return 0, 0, ""
	}
	block := s[bestStart : bestStart+bestBytes]
	frac = clamp01(float64(utf8.RuneCountInString(block)) / float64(totalChars))
	return frac, bestP, previewUnit(s[bestStart : bestStart+bestP])
}

// previewUnit renders a tiling unit safely and boundedly for a reason string.
func previewUnit(u string) string {
	u = strings.ToValidUTF8(u, "")
	const maxRunes = 16
	if utf8.RuneCountInString(u) <= maxRunes {
		return u
	}
	r := []rune(u)
	return string(r[:maxRunes]) + "…"
}

func max3(a, b, c float64) float64 {
	m := a
	if b > m {
		m = b
	}
	if c > m {
		m = c
	}
	return m
}

func clamp01(f float64) float64 {
	if f < 0 {
		return 0
	}
	if f > 1 {
		return 1
	}
	return f
}

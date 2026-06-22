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
// Four repetition sub-signals are computed. Each is ~0 for natural prose and each
// spikes on a distinct degeneration mode; the headline RepeatFraction is their
// maximum, so a text trips on whichever way it actually degenerated:
//
//   - NGramRepeat: 1 - unique/total word n-grams (rep-n, the standard neural-text
//     degeneration metric). Catches looped phrases and repeated sentences.
//   - LineBlockRepeat: runes covered by the single most-frequent non-blank line
//     that occurs at least twice, over total runes. Catches a block/paragraph
//     emitted over and over.
//   - PeriodRepeat: the runes covered by the largest short-period tiling (an
//     alphanumeric-bearing unit up to maxPeriod bytes repeated back-to-back), over
//     total runes. Catches a whitespace-free character runaway ("AAAA…") and a
//     tiled unit ("abcabc…", ".assistant.assistant…") the other signals miss. It
//     generalizes the dominantPeriod/looksDegenerate detector in cmd/simpledemo
//     (issue #91) from "is the WHOLE string periodic" to "what fraction is".
//   - CompRepeat: a flate compression-redundancy signal that catches a long-period
//     byte runaway BEYOND the bounded period scan — the same >maxPeriod-byte
//     URL/hash/JSON record tiled with no internal whitespace, which n-gram (one
//     token), line-block (one line), and period (unit too long) all miss.
//
// Two degeneration modes are deliberately NOT flagged, to keep the false-positive
// rate low on ubiquitous real output: a run or tiling of PURELY non-alphanumeric
// fill characters (a "====" rule, a "|---|---|" table separator, a progress bar,
// an ASCII border) is structural formatting, not a loop, so a tiling unit / a
// repeated line must carry alphanumeric content to count. A bounded enumeration
// whose every token differs ("1, 2, 3, … 199") is left to the caller's judgment
// (it terminates and carries information; masking digits to catch it would flag
// legitimate numeric tables).
//
// Measure is pure and deterministic: identical bytes and identical Limits always
// yield an identical Report.
package answershape

import (
	"bytes"
	"compress/flate"
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
// is never exactly periodic at any of these lengths, so the gate is safe. Longer
// periods are caught by the length-agnostic compression signal instead.
const maxPeriod = 32

// Compression-redundancy signal constants. A flate ratio (compressed/original) far
// below any natural text's (English ~0.4, a repetitive table ~0.25, a tight loop
// ~0.05) isolates a long-period byte runaway. compKnee maps the ratio onto the
// [0,1] repeat scale so the single --max-repeat knob still governs it; the signal
// only applies above compressionFloor runes, where flate has the volume to be
// meaningful (short inputs compress poorly and would read as spuriously redundant).
const (
	compressionFloor = 200
	compKnee         = 0.20
)

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
	CompRepeat      float64 `json:"comp_repeat"`
	FlateRatio      float64 `json:"flate_ratio"`
	// RepeatFraction is the headline = max(NGramRepeat, LineBlockRepeat,
	// PeriodRepeat, CompRepeat). The verdict's repeat check compares it to
	// Limits.MaxRepeat.
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
	rep.CompRepeat, rep.FlateRatio = compRepeat([]byte(s), rep.Chars)
	rep.RepeatFraction = max3(rep.NGramRepeat, rep.LineBlockRepeat, rep.PeriodRepeat)
	if rep.CompRepeat > rep.RepeatFraction {
		rep.RepeatFraction = rep.CompRepeat
	}

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
	case r.PeriodRepeat:
		if r.PeriodUnit != "" {
			return fmt.Sprintf("period-%d-tile (%q)", r.Period, r.PeriodUnit)
		}
		return "period-tile"
	default:
		return fmt.Sprintf("high-redundancy (flate %.2f)", r.FlateRatio)
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
// most-frequent non-blank line that occurs at least twice. A line carrying NO
// alphanumeric content (a "====" rule, a "|---|---|" separator) is structural
// formatting, not a degeneration block, and is excluded. It returns 0 when no
// qualifying line repeats (so a normal single-paragraph answer, whose one line
// trivially has max frequency 1, scores 0 rather than 1). The winner is chosen
// with a lexicographic tiebreak so the result is deterministic even though only
// the fraction is returned today.
func lineBlockRepeat(s string, totalChars int) float64 {
	if totalChars == 0 {
		return 0
	}
	counts := map[string]int{}
	for _, ln := range strings.Split(s, "\n") {
		t := strings.TrimSpace(ln)
		if t != "" && hasAlnum(t) {
			counts[t]++
		}
	}
	bestCount, bestLen := 0, 0
	bestLine := ""
	for ln, c := range counts {
		if c < 2 {
			continue
		}
		prod := c * utf8.RuneCountInString(ln)
		if prod > bestCount*bestLen || (prod == bestCount*bestLen && (bestLine == "" || ln < bestLine)) {
			bestCount, bestLen, bestLine = c, utf8.RuneCountInString(ln), ln
		}
	}
	if bestCount < 2 {
		return 0
	}
	return clamp01(float64(bestCount*bestLen) / float64(totalChars))
}

// periodRepeat finds the largest short-period tiling in s whose repeating unit
// carries alphanumeric content, and returns the fraction of runes it covers, the
// period length (in bytes), and a bounded preview of the unit. For period p it
// finds the longest contiguous byte block over which s[i] == s[i-p] (a block of
// `run` matched bytes covers run+p bytes — the matched tail plus the unit it
// copies), requiring at least two full copies. p=1 is a single-rune runaway; p>=2
// is a tiled unit. Units that are PURELY non-alphanumeric fill (a "====" rule, a
// "|----|----|" separator, a progress bar) are excluded — they are structural
// formatting, not a loop. It is the graded generalization of cmd/simpledemo's
// dominantPeriod (issue #91), which required the WHOLE string to be p-periodic.
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
				if block := run + p; block > bestBytes && hasAlnum(s[start:start+p]) {
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

// compRepeat detects a long-period byte runaway BEYOND the bounded period scan —
// the same >maxPeriod-byte unit (a URL, a hash, a JSON record) tiled with no
// internal whitespace, which n-gram (one token), line-block (one line), and period
// (unit too long) all miss. It flate-compresses the bytes; a ratio far below any
// natural text's is the signal, mapped onto the [0,1] repeat scale by compKnee so
// the single --max-repeat threshold governs it. The ratio is always reported; the
// fraction only fires above compressionFloor runes.
func compRepeat(b []byte, chars int) (frac, ratio float64) {
	if len(b) == 0 {
		return 0, 0
	}
	var buf bytes.Buffer
	w, err := flate.NewWriter(&buf, flate.BestCompression)
	if err != nil {
		return 0, 0
	}
	if _, err := w.Write(b); err != nil {
		_ = w.Close()
		return 0, 0
	}
	if err := w.Close(); err != nil {
		return 0, 0
	}
	ratio = float64(buf.Len()) / float64(len(b))
	if chars < compressionFloor {
		return 0, ratio
	}
	return clamp01((compKnee - ratio) / compKnee), ratio
}

// hasAlnum reports whether a string carries any Unicode letter or digit. A purely
// non-alphanumeric run/unit (a "====" rule, a "|----|" separator, a progress bar,
// an ASCII border, runs of spaces) is structural formatting — not degeneration —
// so it must not drive the verdict.
func hasAlnum(s string) bool {
	for _, r := range strings.ToValidUTF8(s, "") {
		if unicode.IsLetter(r) || unicode.IsNumber(r) {
			return true
		}
	}
	return false
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

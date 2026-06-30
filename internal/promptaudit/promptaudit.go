// Package promptaudit scans system/developer/context prompt text for hidden
// control markers BEFORE they cross a model or cache boundary.
//
// The threat it watches for is "prompt steganography": a host can encode a
// small amount of classification (a hostname class, a timezone class) into the
// model-visible prompt prose without an explicit metadata field, by choosing
// among look-alike punctuation runes or by switching a date separator. The
// documented instance (https://thereallo.dev/blog/claude-code-prompt-steganography)
// encodes bits into the "Today's date is ..." sentence: the apostrophe rune is
// chosen from a tiny alphabet (ASCII ', U+2019, U+02BC, U+02B9) and the date
// token uses '-' vs '/' as a second channel.
//
// The scanner is deliberately EXPLAINABLE rather than a normalizer. It reports,
// for every finding, the offending codepoint(s) as U+XXXX, the byte and rune
// offset, a normalized rendering of the run, and the suspected channel. It does
// NOT rewrite the text and does NOT fail closed on ordinary curly punctuation
// that lacks a specific marker shape.
package promptaudit

import (
	"fmt"
	"sort"
	"strings"
	"unicode"
)

// Channel names the steganographic channel a finding is suspected to belong to.
type Channel string

const (
	// ChannelHostname is the hostname-classification channel: a look-alike or
	// modifier-letter apostrophe substituted for an ASCII apostrophe near a
	// possessive ("Today's"), carrying a host class.
	ChannelHostname Channel = "hostname-marker"
	// ChannelTimezone is the timezone/date channel: a non-default date
	// separator ('/' instead of '-') in a date token near a "date is" phrase.
	ChannelTimezone Channel = "timezone/date-marker"
	// ChannelUnicodeControl is a generic invisible/format-category Unicode
	// marker (zero-width, format, control, private-use) found anywhere in the
	// scanned text — not tied to the date sentence, but suspicious as a hidden
	// channel in its own right.
	ChannelUnicodeControl Channel = "unicode-control-marker"
)

// Kind is the coarse class of a finding, useful for routing/severity.
type Kind string

const (
	// KindLookalikeApostrophe is an apostrophe-position rune that is not the
	// ASCII apostrophe and not the common typographic right single quote used
	// in benign prose — a modifier-letter or prime variant that forms a
	// discrete marker alphabet.
	KindLookalikeApostrophe Kind = "lookalike-apostrophe"
	// KindDateSeparator is a date token whose separator is the non-default
	// channel value ('/').
	KindDateSeparator Kind = "date-separator"
	// KindInvisibleRune is a zero-width / format / control / private-use rune.
	KindInvisibleRune Kind = "invisible-rune"
)

// Finding is a single, fully explainable detection. It carries enough evidence
// to act on WITHOUT having normalized the original text away.
type Finding struct {
	// Kind is the coarse detection class.
	Kind Kind
	// Channel is the suspected steganographic channel.
	Channel Channel
	// Codepoints lists the offending runes as U+XXXX strings, in order.
	Codepoints []string
	// ByteOffset is the offset of the first offending rune in the input string,
	// counted in bytes.
	ByteOffset int
	// RuneOffset is the offset of the first offending rune, counted in runes.
	RuneOffset int
	// Raw is the exact offending substring as it appeared in the input.
	Raw string
	// Normalized is a human-readable rendering of the run with the marker shown
	// explicitly (e.g. the apostrophe replaced by ASCII, invisible runes shown
	// as their codepoint). The original is never mutated; this is a separate
	// view for the operator.
	Normalized string
	// Detail is a short human sentence explaining what was seen and why it is
	// suspicious.
	Detail string
}

// String renders a Finding as a single operator-readable line. A later commit
// can surface this on a CLI/dispatcher without reaching into the struct.
func (f Finding) String() string {
	cps := strings.Join(f.Codepoints, " ")
	return fmt.Sprintf("[%s] channel=%s codepoint=%s byte=%d rune=%d raw=%q norm=%q — %s",
		f.Kind, f.Channel, cps, f.ByteOffset, f.RuneOffset, f.Raw, f.Normalized, f.Detail)
}

// codepoint formats a rune as U+XXXX (uppercase hex, min width 4).
func codepoint(r rune) string {
	return fmt.Sprintf("U+%04X", r)
}

// The discrete marker alphabet for the apostrophe position documented in the
// source article. The ASCII apostrophe is the default/benign value; the three
// others are look-alike modifier letters / primes that encode the off bits.
const (
	runeASCIIApostrophe    = '\'' // U+0027 APOSTROPHE (benign default)
	runeRightSingleQuote   = '’'  // RIGHT SINGLE QUOTATION MARK
	runeModifierApostrophe = 'ʼ'  // MODIFIER LETTER APOSTROPHE
	runeModifierPrime      = 'ʹ'  // MODIFIER LETTER PRIME
)

// markerApostrophes are the NON-ASCII apostrophe-position runes that form the
// documented marker alphabet. U+2019 is included because, in the SPECIFIC
// possessive-contraction position of the "Today's date is" sentence, it is part
// of the marker alphabet — but scanDateSentence only reads this slot inside that
// carrier sentence, so ordinary curly-quoted prose elsewhere does not fire.
var markerApostrophes = map[rune]string{
	runeRightSingleQuote:   "right single quotation mark",
	runeModifierApostrophe: "modifier letter apostrophe",
	runeModifierPrime:      "modifier letter prime",
}

// Scan inspects text and returns every steganography finding, ordered by byte
// offset. An empty slice means nothing suspicious was found. Scan never mutates
// the input and never panics on arbitrary bytes.
func Scan(text string) []Finding {
	var out []Finding

	out = append(out, scanInvisibleRunes(text)...)
	out = append(out, scanDateSentence(text)...)

	sort.SliceStable(out, func(i, j int) bool {
		return out[i].ByteOffset < out[j].ByteOffset
	})
	return out
}

// scanInvisibleRunes flags zero-width, format-category, control and private-use
// runes anywhere in the text. These are principled, category-driven detections
// rather than a hardcoded list: a hidden channel needs an invisible or
// non-rendering carrier, and these Unicode categories are exactly that.
func scanInvisibleRunes(text string) []Finding {
	var out []Finding
	byteOff := 0
	runeOff := 0
	for _, r := range text {
		size := len(string(r))
		if isSuspiciousInvisible(r) {
			out = append(out, Finding{
				Kind:       KindInvisibleRune,
				Channel:    ChannelUnicodeControl,
				Codepoints: []string{codepoint(r)},
				ByteOffset: byteOff,
				RuneOffset: runeOff,
				Raw:        string(r),
				Normalized: codepoint(r),
				Detail: fmt.Sprintf("invisible/format-category rune %s (%s) — a non-rendering carrier that can hide a control channel in prompt text",
					codepoint(r), unicodeClassName(r)),
			})
		}
		byteOff += size
		runeOff++
	}
	return out
}

// Named zero-width / joiner runes, written as escapes so the source file itself
// stays free of the invisible carriers this package exists to catch.
const (
	runeZeroWidthSpace     = '\u200B' // ZERO WIDTH SPACE
	runeZeroWidthNonJoiner = '\u200C' // ZERO WIDTH NON-JOINER
	runeZeroWidthJoiner    = '\u200D' // ZERO WIDTH JOINER
	runeWordJoiner         = '\u2060' // WORD JOINER
	runeZeroWidthNoBreak   = '\uFEFF' // ZERO WIDTH NO-BREAK SPACE / BOM
)

// isSuspiciousInvisible reports whether r is the kind of non-rendering rune that
// has no business in plain prompt prose: zero-width joiners/non-joiners, the
// zero-width space, format-category runes (Cf), other control runes (Cc except
// the ordinary whitespace \t \n \r), and private-use runes (Co). The explicit
// zero-width cases below are all Cf and would be caught by the category check;
// they are listed first only to make the intent legible.
func isSuspiciousInvisible(r rune) bool {
	switch r {
	case runeZeroWidthSpace, runeZeroWidthNonJoiner, runeZeroWidthJoiner,
		runeWordJoiner, runeZeroWidthNoBreak:
		return true
	}
	if r == '\t' || r == '\n' || r == '\r' {
		return false
	}
	if unicode.Is(unicode.Cf, r) { // format
		return true
	}
	if unicode.Is(unicode.Cc, r) { // control
		return true
	}
	if unicode.Is(unicode.Co, r) { // private use
		return true
	}
	return false
}

// unicodeClassName gives a short human name for the offending rune's class.
func unicodeClassName(r rune) string {
	switch r {
	case runeZeroWidthSpace:
		return "zero-width space"
	case runeZeroWidthNonJoiner:
		return "zero-width non-joiner"
	case runeZeroWidthJoiner:
		return "zero-width joiner"
	case runeZeroWidthNoBreak:
		return "zero-width no-break space / BOM"
	case runeWordJoiner:
		return "word joiner"
	}
	switch {
	case unicode.Is(unicode.Cf, r):
		return "Unicode format category"
	case unicode.Is(unicode.Cc, r):
		return "Unicode control category"
	case unicode.Is(unicode.Co, r):
		return "Unicode private-use category"
	default:
		return "non-rendering rune"
	}
}

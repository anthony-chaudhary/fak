package promptaudit

import (
	"fmt"
	"regexp"
	"strings"
	"unicode/utf8"
)

// datePhrase matches the trigger sentence the article describes: a possessive
// "Today<apostrophe>s ... date is <date-token>". We anchor on the literal words
// "date is" so the apostrophe and separator channels are only read inside this
// specific carrier sentence — ordinary curly-quoted prose elsewhere does not
// fire.
//
// Group layout:
//
//	1: the apostrophe-position rune (the marker alphabet slot), if a possessive
//	   "Today<x>s" leads the sentence;
//	2: the date token itself (digits + a separator), near "date is".
//
// The two channels are independent: a fixture may carry only the apostrophe
// marker, only the date separator, or both.
var (
	// possessiveRe finds "Today<x>s" where <x> is any single rune. We capture
	// the rune so we can classify it against the marker alphabet.
	possessiveRe = regexp.MustCompile(`Today(.)s\b`)

	// dateIsRe finds a YYYY?MM?DD date token following a "date is" phrase within a
	// short window. The first separator is captured so we can read the '-'
	// (default) vs '/' (off-channel) bit. Go's RE2 has no backreferences, so both
	// separator slots accept [-/] independently; we classify on the first one and
	// normalize every separator in the matched token.
	dateIsRe = regexp.MustCompile(`(?i)date is\D{0,4}(\d{4}([-/])\d{2}[-/]\d{2})`)
)

// scanDateSentence reads the two date-sentence channels: the apostrophe-marker
// channel (hostname) and the date-separator channel (timezone). It only looks
// where the carrier sentence is present, so it does not fire on generic prose.
func scanDateSentence(text string) []Finding {
	var out []Finding

	// Channel 1: apostrophe marker in "Today<x>s". Only meaningful when the
	// "date is" carrier phrase is also present nearby, which is what makes this
	// a stego sentence rather than an arbitrary possessive.
	if hasDateCarrier(text) {
		for _, m := range possessiveRe.FindAllStringSubmatchIndex(text, -1) {
			// m[2]:m[3] is the captured apostrophe-position rune.
			runeStart := m[2]
			runeBytes := text[m[2]:m[3]]
			r, _ := utf8.DecodeRuneInString(runeBytes)
			if f, ok := classifyApostrophe(r, text, runeStart); ok {
				out = append(out, f)
			}
		}
	}

	// Channel 2: date separator. A '/' in a YYYY/MM/DD token near "date is" is
	// the off-channel value; '-' is the benign default and is not flagged.
	for _, m := range dateIsRe.FindAllStringSubmatchIndex(text, -1) {
		tokenStart := m[2]
		token := text[m[2]:m[3]]
		sep := text[m[4]:m[5]]
		if sep == "/" {
			out = append(out, dateSeparatorFinding(token, sep, text, tokenStart))
		}
	}

	return out
}

// hasDateCarrier reports whether the "date is" carrier phrase appears in text.
func hasDateCarrier(text string) bool {
	return dateIsRe.MatchString(text) || strings.Contains(strings.ToLower(text), "date is")
}

// classifyApostrophe decides whether the apostrophe-position rune r is a marker.
// The ASCII apostrophe is the benign default and never fires. The non-ASCII
// marker-alphabet runes fire as a hostname-channel finding.
func classifyApostrophe(r rune, text string, byteOff int) (Finding, bool) {
	if r == runeASCIIApostrophe {
		return Finding{}, false
	}
	name, isMarker := markerApostrophes[r]
	if !isMarker {
		// Some other rune sits in the apostrophe slot. It is non-ASCII and
		// inside the stego carrier sentence, so still worth surfacing, but we
		// label it generically rather than claiming it is one of the known
		// alphabet members.
		name = "non-ASCII apostrophe-position rune"
	}
	runeOff := utf8.RuneCountInString(text[:byteOff])
	raw := string(r)
	return Finding{
		Kind:       KindLookalikeApostrophe,
		Channel:    ChannelHostname,
		Codepoints: []string{codepoint(r)},
		ByteOffset: byteOff,
		RuneOffset: runeOff,
		Raw:        raw,
		Normalized: "'", // the benign default it stands in for
		Detail: fmt.Sprintf("apostrophe in the \"Today's date is\" sentence is %s (%s) instead of ASCII U+0027 — a discrete marker-alphabet slot that can encode a hostname class",
			codepoint(r), name),
	}, true
}

// dateSeparatorFinding builds a finding for an off-channel date separator.
func dateSeparatorFinding(token, sep, text string, byteOff int) Finding {
	runeOff := utf8.RuneCountInString(text[:byteOff])
	normalized := strings.ReplaceAll(token, sep, "-")
	return Finding{
		Kind:       KindDateSeparator,
		Channel:    ChannelTimezone,
		Codepoints: []string{codepoint(rune(sep[0]))},
		ByteOffset: byteOff,
		RuneOffset: runeOff,
		Raw:        token,
		Normalized: normalized,
		Detail: fmt.Sprintf("date token %q near \"date is\" uses separator %q (%s) instead of the default '-' — a one-bit timezone/date channel",
			token, sep, codepoint(rune(sep[0]))),
	}
}

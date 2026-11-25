package sessionreg

import (
	"regexp"
	"strings"
	"unicode"
	"unicode/utf8"
)

// pyRegexp matches a pattern with Python re's str semantics where Go's differ: \s is
// Unicode whitespace (and \v), and \b treats every Unicode letter or number as a word rune.
type pyRegexp struct {
	re         *regexp.Regexp
	bounded    bool
	ignoreCase bool
}

// pyWhitespaceClass is the set str.isspace() accepts, which is what Python's \s matches.
const pyWhitespaceClass = `[\t\n\v\f\r \x{1c}-\x{1f}\x{85}\x{a0}\x{1680}\x{2000}-\x{200a}\x{2028}\x{2029}\x{202f}\x{205f}\x{3000}]`

// wordFoldRune stands in for a non-ASCII word rune so Go's ASCII \b sees the same word edge
// Python's Unicode \b does. It must be a letter no bounded pattern spells.
const wordFoldRune = 'z'

func mustCompilePy(pattern string) pyRegexp {
	bounded := strings.Contains(pattern, `\b`)
	if bounded && strings.ContainsAny(pattern, "zZ") {
		panic("sessionreg: a \\b pattern must not spell the word-fold rune: " + pattern)
	}
	return pyRegexp{
		re:         regexp.MustCompile(strings.ReplaceAll(pattern, `\s`, pyWhitespaceClass)),
		bounded:    bounded,
		ignoreCase: strings.HasPrefix(pattern, "(?i)"),
	}
}

func (p pyRegexp) MatchString(s string) bool {
	if p.bounded {
		s = p.foldWordRunes(s)
	}
	return p.re.MatchString(s)
}

// FindStringSubmatch is re.search's groups, each sliced from s itself rather than from the
// folded text the match ran on.
func (p pyRegexp) FindStringSubmatch(s string) []string {
	if !p.bounded {
		return p.re.FindStringSubmatch(s)
	}
	folded := p.foldWordRunes(s)
	loc := p.re.FindStringSubmatchIndex(folded)
	if loc == nil {
		return nil
	}
	orig := foldedOffsets(s, folded)
	out := make([]string, len(loc)/2)
	for i := range out {
		if a, b := loc[2*i], loc[2*i+1]; a >= 0 {
			out[i] = s[orig[a]:orig[b]]
		}
	}
	return out
}

// foldedOffsets maps each byte offset of folded (and its end) to the offset in s of the
// rune it came from; folding maps rune for rune, so the runes pair up in order.
func foldedOffsets(s, folded string) []int {
	orig := make([]int, len(folded)+1)
	fi := 0
	for si := range s {
		_, fw := utf8.DecodeRuneInString(folded[fi:])
		for k := 0; k < fw; k++ {
			orig[fi+k] = si
		}
		fi += fw
	}
	orig[len(folded)] = len(s)
	return orig
}

// foldWordRunes replaces each non-ASCII letter or number with an ASCII word rune. Under
// (?i) a rune that case-folds to an ASCII letter (the Kelvin sign) becomes that letter, so
// it still matches it, as it does in Python; every other one becomes wordFoldRune.
func (p pyRegexp) foldWordRunes(s string) string {
	ascii := true
	for i := 0; i < len(s); i++ {
		if s[i] >= utf8.RuneSelf {
			ascii = false
			break
		}
	}
	if ascii {
		return s
	}
	return strings.Map(func(r rune) rune {
		if r < utf8.RuneSelf || !(unicode.IsLetter(r) || unicode.IsNumber(r)) {
			return r
		}
		if p.ignoreCase {
			if a, ok := asciiFold(r); ok {
				return a
			}
		}
		return wordFoldRune
	}, s)
}

func asciiFold(r rune) (rune, bool) {
	for f := unicode.SimpleFold(r); f != r; f = unicode.SimpleFold(f) {
		if f < utf8.RuneSelf {
			return f, true
		}
	}
	return 0, false
}

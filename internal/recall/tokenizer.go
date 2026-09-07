package recall

import (
	"strings"
	"unicode"
)

const (
	maxTokenLen      = 32
	maxBigramsPerRun = 64
)

// isCJK reports whether r is in the CJK (Chinese, Japanese, Korean) Unicode blocks.
func isCJK(r rune) bool {
	if r >= 0x4e00 && r <= 0x9fff { // CJK Unified Ideographs
		return true
	}
	if r >= 0x3400 && r <= 0x4dbf { // CJK Extension A
		return true
	}
	if r >= 0x3040 && r <= 0x309f { // Hiragana
		return true
	}
	if r >= 0x30a0 && r <= 0x30ff { // Katakana
		return true
	}
	if r >= 0xac00 && r <= 0xd7af { // Hangul Syllables
		return true
	}
	return unicode.In(r, unicode.Han, unicode.Hiragana, unicode.Katakana, unicode.Hangul, unicode.Bopomofo)
}

const (
	runNone = iota
	runAlpha
	runCJK
)

// Tokenize splits text into lexical tokens with CJK bigram support.
//
// Tokenization rules:
//  1. Non-letter/non-digit characters (whitespace, punctuation, symbols, emoji) are delimiters.
//  2. Script transitions between Latin/digit alphanumeric and CJK runes split tokens (e.g. "L0录入" -> "l0", "录入").
//  3. Latin and alphanumeric tokens are converted to lowercase.
//  4. For any CJK rune run of length L:
//     - The full CJK run is emitted if L <= maxTokenLen.
//     - Character bigrams (r_i r_{i+1}) are emitted, bounded by maxBigramsPerRun.
//  5. Tokens are deduplicated while preserving deterministic first-seen order.
func Tokenize(text string) []string {
	if text == "" {
		return nil
	}

	lower := strings.ToLower(text)
	var tokens []string
	seen := make(map[string]bool)

	addToken := func(t string) {
		if t != "" && !seen[t] {
			seen[t] = true
			tokens = append(tokens, t)
		}
	}

	var currentRun []rune
	currentType := runNone

	flush := func() {
		if len(currentRun) == 0 {
			currentType = runNone
			return
		}
		switch currentType {
		case runAlpha:
			addToken(string(currentRun))
		case runCJK:
			L := len(currentRun)
			if L <= maxTokenLen {
				addToken(string(currentRun))
			}
			bigramCount := 0
			for i := 0; i < L-1 && i < maxBigramsPerRun*4 && bigramCount < maxBigramsPerRun; i++ {
				bg := string(currentRun[i : i+2])
				if !seen[bg] {
					addToken(bg)
					bigramCount++
				}
			}
		}
		currentRun = currentRun[:0]
		currentType = runNone
	}

	for _, r := range lower {
		if !unicode.IsLetter(r) && !unicode.IsDigit(r) {
			flush()
			continue
		}

		if isCJK(r) {
			if currentType != runCJK {
				flush()
				currentType = runCJK
			}
			if len(currentRun) < maxBigramsPerRun*4+2 {
				currentRun = append(currentRun, r)
			}
		} else {
			if currentType != runAlpha {
				flush()
				currentType = runAlpha
			}
			currentRun = append(currentRun, r)
		}
	}
	flush()

	return tokens
}

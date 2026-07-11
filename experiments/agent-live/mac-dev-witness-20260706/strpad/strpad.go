package strpad

import (
	"strings"
	"unicode/utf8"
)

// LeftPad pads s on the left with p until it is at least n runes wide.
func LeftPad(s string, n int, p rune) string {
	runeLen := utf8.RuneCountInString(s)
	if runeLen >= n {
		return s
	}
	return strings.Repeat(string(p), n-runeLen) + s
}

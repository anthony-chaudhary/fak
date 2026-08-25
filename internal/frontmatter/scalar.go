// Package frontmatter contains the dependency-free scalar semantics shared by
// FAK's deliberately flat YAML frontmatter readers.
package frontmatter

import (
	"strings"
	"unicode/utf8"
)

// DecodeScalar returns the semantic value of one plain, single-quoted, or
// double-quoted YAML scalar. It intentionally does not parse mappings,
// collections, comments, block scalars, or line folding.
//
// Invalid quoted input returns the trimmed source unchanged with ok=false. That
// lets tolerant readers preserve legacy input without silently mutilating it,
// while strict readers can reject it.
func DecodeScalar(raw string) (value string, ok bool) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return "", true
	}

	switch raw[0] {
	case '\'':
		if len(raw) < 2 || raw[len(raw)-1] != '\'' {
			return raw, false
		}
		value, ok := decodeSingleQuoted(raw[1 : len(raw)-1])
		if !ok {
			return raw, false
		}
		return value, true
	case '"':
		if len(raw) < 2 || raw[len(raw)-1] != '"' {
			return raw, false
		}
		value, ok := decodeDoubleQuoted(raw[1 : len(raw)-1])
		if !ok {
			return raw, false
		}
		return value, true
	default:
		return raw, true
	}
}

func decodeSingleQuoted(inner string) (string, bool) {
	var out strings.Builder
	out.Grow(len(inner))
	for i := 0; i < len(inner); i++ {
		if inner[i] != '\'' {
			out.WriteByte(inner[i])
			continue
		}
		if i+1 >= len(inner) || inner[i+1] != '\'' {
			return "", false
		}
		out.WriteByte('\'')
		i++
	}
	return out.String(), true
}

func decodeDoubleQuoted(inner string) (string, bool) {
	var out strings.Builder
	out.Grow(len(inner))
	for i := 0; i < len(inner); {
		switch inner[i] {
		case '"':
			return "", false
		case '\\':
			i++
			if i >= len(inner) {
				return "", false
			}
			escape := inner[i]
			i++
			switch escape {
			case '0':
				out.WriteByte(0)
			case 'a':
				out.WriteByte('\a')
			case 'b':
				out.WriteByte('\b')
			case 't':
				out.WriteByte('\t')
			case 'n':
				out.WriteByte('\n')
			case 'v':
				out.WriteByte('\v')
			case 'f':
				out.WriteByte('\f')
			case 'r':
				out.WriteByte('\r')
			case 'e':
				out.WriteByte(0x1b)
			case ' ':
				out.WriteByte(' ')
			case '"', '/', '\\':
				out.WriteByte(escape)
			case 'N':
				out.WriteRune('\u0085')
			case '_':
				out.WriteRune('\u00a0')
			case 'L':
				out.WriteRune('\u2028')
			case 'P':
				out.WriteRune('\u2029')
			case 'x':
				r, next, ok := decodeHexRune(inner, i, 2)
				if !ok {
					return "", false
				}
				out.WriteRune(r)
				i = next
			case 'u':
				r, next, ok := decodeHexRune(inner, i, 4)
				if !ok {
					return "", false
				}
				out.WriteRune(r)
				i = next
			case 'U':
				r, next, ok := decodeHexRune(inner, i, 8)
				if !ok {
					return "", false
				}
				out.WriteRune(r)
				i = next
			default:
				return "", false
			}
		default:
			out.WriteByte(inner[i])
			i++
		}
	}
	return out.String(), true
}

func decodeHexRune(s string, start, digits int) (rune, int, bool) {
	if start+digits > len(s) {
		return 0, start, false
	}
	var value rune
	for i := start; i < start+digits; i++ {
		nibble, ok := hexNibble(s[i])
		if !ok {
			return 0, start, false
		}
		value = value<<4 | rune(nibble)
	}
	if !utf8.ValidRune(value) {
		return 0, start, false
	}
	return value, start + digits, true
}

func hexNibble(b byte) (byte, bool) {
	switch {
	case b >= '0' && b <= '9':
		return b - '0', true
	case b >= 'a' && b <= 'f':
		return b - 'a' + 10, true
	case b >= 'A' && b <= 'F':
		return b - 'A' + 10, true
	default:
		return 0, false
	}
}

// Package aeogen generates the answer-engine documentation surfaces: llms-full.txt (the
// one-fetch corpus inlined from llms.txt) and the JSON-LD structured-data blocks injected
// into the docs site, FAQ, index page and llms.txt. It is the in-process port of
// tools/gen_llms_full.py and tools/gen_structured_data.py and must stay byte-identical to
// them, so the helpers here reproduce Python's text semantics (universal newlines,
// Unicode whitespace, code-point lengths) rather than Go's defaults.
package aeogen

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"
)

// pyWS is Python's Unicode `\s` class; Go's `\s` is ASCII-only and omits \v.
const pyWS = `[\t\n\v\f\r \x1c-\x1f\x{85}\x{a0}\x{1680}\x{2000}-\x{200a}\x{2028}\x{2029}\x{202f}\x{205f}\x{3000}]`

func pyIsSpace(r rune) bool {
	switch {
	case r >= '\t' && r <= '\r', r >= 0x1c && r <= 0x20, r == 0x85, r == 0xa0, r == 0x1680,
		r >= 0x2000 && r <= 0x200a, r == 0x2028, r == 0x2029, r == 0x202f, r == 0x205f, r == 0x3000:
		return true
	}
	return false
}

func pyStrip(s string) string  { return strings.TrimFunc(s, pyIsSpace) }
func pyRstrip(s string) string { return strings.TrimRightFunc(s, pyIsSpace) }

// pyFields is str.split() with no separator.
func pyFields(s string) []string { return strings.FieldsFunc(s, pyIsSpace) }

// pySplitlines is str.splitlines(): every Python line boundary, no trailing empty line.
func pySplitlines(s string) []string {
	var out []string
	start := 0
	for i := 0; i < len(s); {
		r, size := utf8.DecodeRuneInString(s[i:])
		switch r {
		case '\n', '\v', '\f', 0x1c, 0x1d, 0x1e, 0x85, 0x2028, 0x2029:
			out = append(out, s[start:i])
			i += size
			start = i
			continue
		case '\r':
			out = append(out, s[start:i])
			i += size
			if i < len(s) && s[i] == '\n' {
				i++
			}
			start = i
			continue
		}
		i += size
	}
	if start < len(s) {
		out = append(out, s[start:])
	}
	return out
}

func runeLen(s string) int { return utf8.RuneCountInString(s) }

// runePrefix is s[:n] with Python code-point indexing.
func runePrefix(s string, n int) string {
	i := 0
	for k := 0; k < n && i < len(s); k++ {
		_, size := utf8.DecodeRuneInString(s[i:])
		i += size
	}
	return s[:i]
}

// universalNewlines is what a Python text-mode read does to line endings.
func universalNewlines(s string) string {
	s = strings.ReplaceAll(s, "\r\n", "\n")
	return strings.ReplaceAll(s, "\r", "\n")
}

var errInvalidUTF8 = errors.New("invalid utf-8")

// readText is open(path, encoding="utf-8").read(): strict decoding, universal newlines.
func readText(path string) (string, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	if !utf8.Valid(b) {
		return "", fmt.Errorf("%s: %w", path, errInvalidUTF8)
	}
	return universalNewlines(string(b)), nil
}

// readTextReplace is open(path, encoding="utf-8", errors="replace").read().
func readTextReplace(path string) (string, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	return universalNewlines(decodeReplace(b)), nil
}

// decodeReplace substitutes one U+FFFD per maximal invalid subpart, as CPython does;
// Go's own conversion would emit one per byte or one per run.
func decodeReplace(b []byte) string {
	if utf8.Valid(b) {
		return string(b)
	}
	var sb strings.Builder
	for i := 0; i < len(b); {
		r, size := utf8.DecodeRune(b[i:])
		if r != utf8.RuneError || size > 1 {
			sb.WriteRune(r)
			i += size
			continue
		}
		sb.WriteRune(utf8.RuneError)
		i += maximalSubpart(b[i:])
	}
	return sb.String()
}

func maximalSubpart(b []byte) int {
	lo, hi := byte(0x80), byte(0xBF)
	need := 0
	switch c := b[0]; {
	case c >= 0xC2 && c <= 0xDF:
		need = 1
	case c == 0xE0:
		need, lo = 2, 0xA0
	case c >= 0xE1 && c <= 0xEC, c == 0xEE, c == 0xEF:
		need = 2
	case c == 0xED:
		need, hi = 2, 0x9F
	case c == 0xF0:
		need, lo = 3, 0x90
	case c >= 0xF1 && c <= 0xF3:
		need = 3
	case c == 0xF4:
		need, hi = 3, 0x8F
	default:
		return 1
	}
	n := 1
	for k := 0; k < need && n < len(b); k++ {
		if b[n] < lo || b[n] > hi {
			break
		}
		n++
		lo, hi = 0x80, 0xBF
	}
	return n
}

// writeText is open(path, "w", encoding="utf-8", newline="\n").write(text).
func writeText(path, text string) error {
	return os.WriteFile(path, []byte(text), 0o644)
}

// pyJoin is os.path.join(a, b): an absolute b discards a, unlike filepath.Join.
func pyJoin(a, b string) string {
	if strings.HasPrefix(b, "/") || (filepath.Separator == '\\' && strings.HasPrefix(b, `\`)) {
		return filepath.VolumeName(a) + b
	}
	if filepath.VolumeName(b) != "" || a == "" {
		return b
	}
	if strings.HasSuffix(a, "/") || strings.HasSuffix(a, string(filepath.Separator)) {
		return a + b
	}
	return a + string(filepath.Separator) + b
}

// posixDirname is posixpath.dirname.
func posixDirname(p string) string {
	head := p[:strings.LastIndex(p, "/")+1]
	if head != "" && strings.Trim(head, "/") != "" {
		head = strings.TrimRight(head, "/")
	}
	return head
}

// posixNormpath is posixpath.normpath.
func posixNormpath(p string) string {
	if p == "" {
		return "."
	}
	initial := 0
	if strings.HasPrefix(p, "/") {
		initial = 1
		if strings.HasPrefix(p, "//") && !strings.HasPrefix(p, "///") {
			initial = 2
		}
	}
	var comps []string
	for _, c := range strings.Split(p, "/") {
		if c == "" || c == "." {
			continue
		}
		if c != ".." || (initial == 0 && len(comps) == 0) || (len(comps) > 0 && comps[len(comps)-1] == "..") {
			comps = append(comps, c)
		} else if len(comps) > 0 {
			comps = comps[:len(comps)-1]
		}
	}
	out := strings.Repeat("/", initial) + strings.Join(comps, "/")
	if out == "" {
		return "."
	}
	return out
}

func isFile(p string) bool {
	st, err := os.Stat(p)
	return err == nil && st.Mode().IsRegular()
}

func isDir(p string) bool {
	st, err := os.Stat(p)
	return err == nil && st.IsDir()
}

// urlQuote is urllib.parse.quote(s, safe=safe).
func urlQuote(s, safe string) string {
	var sb strings.Builder
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c < 0x80 && (c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' ||
			strings.IndexByte("_.-~", c) >= 0 || strings.IndexByte(safe, c) >= 0) {
			sb.WriteByte(c)
			continue
		}
		fmt.Fprintf(&sb, "%%%02X", c)
	}
	return sb.String()
}

// thousands is Python's f"{n:,}".
func thousands(n int) string {
	s := strconv.Itoa(n)
	neg := strings.HasPrefix(s, "-")
	s = strings.TrimPrefix(s, "-")
	var sb strings.Builder
	for i, c := range s {
		if i > 0 && (len(s)-i)%3 == 0 {
			sb.WriteByte(',')
		}
		sb.WriteRune(c)
	}
	if neg {
		return "-" + sb.String()
	}
	return sb.String()
}

func pyIsWordRune(r rune) bool {
	return r == '_' || unicode.IsLetter(r) || unicode.IsNumber(r)
}

// pyCasefold is str.casefold(): unicode.ToLower everywhere except the code points in
// pyCasefoldExceptions.
func pyCasefold(s string) string {
	var sb strings.Builder
	for _, r := range s {
		if f, ok := pyCasefoldExceptions[r]; ok {
			sb.WriteString(f)
		} else {
			sb.WriteRune(unicode.ToLower(r))
		}
	}
	return sb.String()
}

// pyRepr is repr(str).
func pyRepr(s string) string {
	quote := byte('\'')
	if strings.Contains(s, "'") && !strings.Contains(s, `"`) {
		quote = '"'
	}
	var sb strings.Builder
	sb.WriteByte(quote)
	for _, r := range s {
		switch {
		case r == rune(quote) || r == '\\':
			sb.WriteByte('\\')
			sb.WriteRune(r)
		case r == '\t':
			sb.WriteString(`\t`)
		case r == '\n':
			sb.WriteString(`\n`)
		case r == '\r':
			sb.WriteString(`\r`)
		case r == ' ' || unicode.IsPrint(r):
			sb.WriteRune(r)
		case r < 0x100:
			fmt.Fprintf(&sb, `\x%02x`, r)
		case r < 0x10000:
			fmt.Fprintf(&sb, `\u%04x`, r)
		default:
			fmt.Fprintf(&sb, `\U%08x`, r)
		}
	}
	sb.WriteByte(quote)
	return sb.String()
}

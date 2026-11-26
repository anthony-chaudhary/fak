package scorecard

import (
	"bytes"
	"fmt"
	"math"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf16"
	"unicode/utf8"
)

// The helpers in this file reproduce the Python builtins the scorecard's text
// depends on (str(), repr(), round(), float repr, str.strip/isspace/splitlines,
// format-spec padding and json.dumps' ensure_ascii escaping), so rendered bytes
// match the Python oracle exactly.

// pyIsSpace is str.isspace for one code point.
func pyIsSpace(r rune) bool {
	switch {
	case r >= 0x1c && r <= 0x1f:
		return true
	case r == 0x180e:
		return false
	}
	return unicode.IsSpace(r)
}

func pyStrip(s string) string { return strings.TrimFunc(s, pyIsSpace) }

// pyIsPrintable is str.isprintable for one code point.
func pyIsPrintable(r rune) bool {
	if r == ' ' {
		return true
	}
	return unicode.IsPrint(r)
}

// pySplitLines is str.splitlines() without keepends.
func pySplitLines(s string) []string {
	var out []string
	start := 0
	for i := 0; i < len(s); {
		r, size := utf8.DecodeRuneInString(s[i:])
		brk := 0
		switch r {
		case '\n', '\v', '\f', 0x1c, 0x1d, 0x1e, 0x85, 0x2028, 0x2029:
			brk = size
		case '\r':
			brk = size
			if i+1 < len(s) && s[i+1] == '\n' {
				brk++
			}
		}
		if brk > 0 {
			out = append(out, s[start:i])
			i += brk
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

// pyFloatRepr is float.__repr__: the shortest round-trip digits, fixed notation
// inside 1e-4 <= |x| < 1e16 (always with a fractional part) and scientific
// notation with a two-digit exponent outside it.
func pyFloatRepr(f float64) string {
	switch {
	case math.IsNaN(f):
		return "NaN"
	case math.IsInf(f, 1):
		return "Infinity"
	case math.IsInf(f, -1):
		return "-Infinity"
	}
	e := strconv.FormatFloat(f, 'e', -1, 64)
	sign := ""
	if strings.HasPrefix(e, "-") {
		sign, e = "-", e[1:]
	}
	mant, expStr, _ := strings.Cut(e, "e")
	exp, _ := strconv.Atoi(expStr)
	digits := strings.Replace(mant, ".", "", 1)
	decpt := exp + 1
	if decpt > -4 && decpt <= 16 {
		var s string
		switch {
		case decpt <= 0:
			s = "0." + strings.Repeat("0", -decpt) + digits
		case decpt >= len(digits):
			s = digits + strings.Repeat("0", decpt-len(digits)) + ".0"
		default:
			s = digits[:decpt] + "." + digits[decpt:]
		}
		return sign + s
	}
	m := digits[:1]
	if len(digits) > 1 {
		m += "." + digits[1:]
	}
	esign := "+"
	if exp < 0 {
		esign, exp = "-", -exp
	}
	return fmt.Sprintf("%s%se%s%02d", sign, m, esign, exp)
}

// pyRound is round(x, n) for a float: correctly rounded, ties to even on the
// exact binary value.
func pyRound(x float64, n int) float64 {
	s := strconv.FormatFloat(x, 'f', n, 64)
	f, _ := strconv.ParseFloat(s, 64)
	return f
}

// pyRoundInt is round(x) with no digits: banker's rounding to an int.
func pyRoundInt(x float64) int { return int(math.RoundToEven(x)) }

// pyFixed is format(x, ".Nf").
func pyFixed(x float64, n int) string { return strconv.FormatFloat(x, 'f', n, 64) }

// pySum is builtin sum() over floats starting from int 0: the first term is
// exact and the rest use CPython's Neumaier compensated summation.
func pySum(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	f := xs[0]
	c := 0.0
	for _, x := range xs[1:] {
		t := f + x
		if math.Abs(f) >= math.Abs(x) {
			c += (f - t) + x
		} else {
			c += (x - t) + f
		}
		f = t
	}
	if c != 0 && !math.IsInf(c, 0) && !math.IsNaN(c) {
		f += c
	}
	return f
}

// PyFloat is a float that serializes the way Python's json.dumps writes it.
type PyFloat float64

// MarshalJSON writes Python's float repr.
func (f PyFloat) MarshalJSON() ([]byte, error) { return []byte(pyFloatRepr(float64(f))), nil }

func (f PyFloat) String() string { return pyFloatRepr(float64(f)) }

// pyStrString is repr() of a str.
func pyStrString(s string) string {
	quote := byte('\'')
	if strings.Contains(s, "'") && !strings.Contains(s, "\"") {
		quote = '"'
	}
	var b strings.Builder
	b.WriteByte(quote)
	for _, r := range s {
		switch {
		case r == '\\':
			b.WriteString(`\\`)
		case r == rune(quote):
			b.WriteByte('\\')
			b.WriteByte(quote)
		case r == '\t':
			b.WriteString(`\t`)
		case r == '\n':
			b.WriteString(`\n`)
		case r == '\r':
			b.WriteString(`\r`)
		case r < ' ' || r == 0x7f:
			fmt.Fprintf(&b, `\x%02x`, r)
		case r < 0x7f:
			b.WriteRune(r)
		case pyIsPrintable(r):
			b.WriteRune(r)
		case r < 0x100:
			fmt.Fprintf(&b, `\x%02x`, r)
		case r < 0x10000:
			fmt.Fprintf(&b, `\u%04x`, r)
		default:
			fmt.Fprintf(&b, `\U%08x`, r)
		}
	}
	b.WriteByte(quote)
	return b.String()
}

// pyRepr is repr() of a decoded JSON value.
func pyRepr(v Value) string {
	switch x := v.(type) {
	case string:
		return pyStrString(x)
	case []Value:
		parts := make([]string, len(x))
		for i, e := range x {
			parts[i] = pyRepr(e)
		}
		return "[" + strings.Join(parts, ", ") + "]"
	case []string:
		parts := make([]string, len(x))
		for i, e := range x {
			parts[i] = pyStrString(e)
		}
		return "[" + strings.Join(parts, ", ") + "]"
	case *Object:
		parts := make([]string, len(x.Keys))
		for i, k := range x.Keys {
			parts[i] = pyStrString(k) + ": " + pyRepr(x.Vals[i])
		}
		return "{" + strings.Join(parts, ", ") + "}"
	}
	return pyStr(v)
}

// pyStr is str() (and plain f-string interpolation) of a decoded JSON value.
func pyStr(v Value) string {
	switch x := v.(type) {
	case nil:
		return "None"
	case bool:
		if x {
			return "True"
		}
		return "False"
	case Number:
		return pyNumberRepr(string(x))
	case string:
		return x
	case int:
		return strconv.Itoa(x)
	case PyFloat:
		return x.String()
	case float64:
		return pyFloatRepr(x)
	case []Value, []string, *Object:
		return pyRepr(x)
	}
	return fmt.Sprint(v)
}

// padRight is format(s, "<n"): pad to n code points.
func padRight(s string, n int) string {
	if c := utf8.RuneCountInString(s); c < n {
		return s + strings.Repeat(" ", n-c)
	}
	return s
}

// padLeft is format(s, ">n").
func padLeft(s string, n int) string {
	if c := utf8.RuneCountInString(s); c < n {
		return strings.Repeat(" ", n-c) + s
	}
	return s
}

// normToken is norm_token: lowercase, then keep only [a-z0-9].
func normToken(s string) string {
	ascii := true
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c >= 0x80 || !(c >= 'a' && c <= 'z' || c >= '0' && c <= '9') {
			ascii = false
			break
		}
	}
	if ascii {
		return s
	}
	b := make([]byte, 0, len(s))
	for _, r := range s {
		if r < 0x80 {
			c := byte(r)
			if c >= 'A' && c <= 'Z' {
				c += 'a' - 'A'
			}
			if c >= 'a' && c <= 'z' || c >= '0' && c <= '9' {
				b = append(b, c)
			}
			continue
		}
		for _, l := range pyLower(r) {
			if l >= 'a' && l <= 'z' || l >= '0' && l <= '9' {
				b = append(b, byte(l))
			}
		}
	}
	return string(b)
}

// pyLower is the full lowercase mapping of one code point; the only non-ASCII
// code points whose lowercase carries ASCII letters are handled by the simple
// mapping, with U+0130 expanding to "i" plus a combining dot as in Python.
func pyLower(r rune) []rune {
	if r == 0x130 {
		return []rune{'i', 0x307}
	}
	return []rune{unicode.ToLower(r)}
}

// normValue is norm_token on an arbitrary value: non-strings normalize to "".
func normValue(v Value) string {
	s, ok := v.(string)
	if !ok {
		return ""
	}
	return normToken(s)
}

// nonempty is _nonempty: a string with non-whitespace content.
func nonempty(v Value) bool {
	s, ok := v.(string)
	return ok && pyStrip(s) != ""
}

// asciiJSON rewrites Go's json output to json.dumps' ensure_ascii form: every
// non-ASCII code point (and DEL) inside a string becomes a \u escape, astral
// code points as a UTF-16 surrogate pair.
func asciiJSON(in []byte) []byte {
	clean := true
	for _, c := range in {
		if c >= 0x7f {
			clean = false
			break
		}
	}
	if clean {
		return in
	}
	var out bytes.Buffer
	out.Grow(len(in) + len(in)/8)
	for i := 0; i < len(in); {
		c := in[i]
		if c < 0x7f {
			out.WriteByte(c)
			i++
			continue
		}
		if c == 0x7f {
			out.WriteString(`\u007f`)
			i++
			continue
		}
		r, size := utf8.DecodeRune(in[i:])
		writeUEscape(&out, r)
		i += size
	}
	return out.Bytes()
}

func writeUEscape(out *bytes.Buffer, r rune) {
	if r > 0xffff {
		hi, lo := utf16.EncodeRune(r)
		fmt.Fprintf(out, `\u%04x\u%04x`, hi, lo)
		return
	}
	fmt.Fprintf(out, `\u%04x`, r)
}

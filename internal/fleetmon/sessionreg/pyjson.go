package sessionreg

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"strings"
	"unicode/utf8"
)

// Obj is an insertion-ordered JSON object. The registry files are read back by Python and
// Go consumers alike, and the Python writer emitted dict insertion order, so key order is
// part of the byte contract.
type Obj struct {
	keys []string
	vals map[string]any
}

// NewObj returns an empty ordered object.
func NewObj() *Obj { return &Obj{vals: map[string]any{}} }

// Set stores v under k, keeping k's original position when it already exists.
func (o *Obj) Set(k string, v any) {
	if _, ok := o.vals[k]; !ok {
		o.keys = append(o.keys, k)
	}
	o.vals[k] = v
}

// Get returns the value stored under k.
func (o *Obj) Get(k string) (any, bool) {
	if o == nil {
		return nil, false
	}
	v, ok := o.vals[k]
	return v, ok
}

// Val returns the value under k, or nil.
func (o *Obj) Val(k string) any {
	v, _ := o.Get(k)
	return v
}

// Keys returns the keys in insertion order.
func (o *Obj) Keys() []string {
	if o == nil {
		return nil
	}
	return o.keys
}

// Len is the number of keys.
func (o *Obj) Len() int {
	if o == nil {
		return 0
	}
	return len(o.keys)
}

// Copy is a shallow copy (Python's dict(info)).
func (o *Obj) Copy() *Obj {
	c := NewObj()
	for _, k := range o.Keys() {
		c.Set(k, o.vals[k])
	}
	return c
}

// plain converts an ordered value into the map/slice shape encoding/json and the
// fleetaccounts fold consume.
func plain(v any) any {
	switch t := v.(type) {
	case *Obj:
		m := make(map[string]any, t.Len())
		for _, k := range t.Keys() {
			m[k] = plain(t.vals[k])
		}
		return m
	case []any:
		out := make([]any, len(t))
		for i, e := range t {
			out[i] = plain(e)
		}
		return out
	case pyInt:
		n, err := strconv.ParseInt(string(t), 10, 64)
		if err != nil {
			f, _ := strconv.ParseFloat(string(t), 64)
			return f
		}
		return float64(n)
	}
	return v
}

// pyInt is a decoded JSON integer literal, kept as its digits so a re-dump reproduces it the
// way Python's arbitrary-precision int would.
type pyInt string

// decodeOrdered parses JSON keeping object key order; integers decode to pyInt, other
// numbers to float64 (Python's json.loads split).
func decodeOrdered(data []byte) (any, error) {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	v, err := decodeValue(dec)
	if err != nil {
		return nil, err
	}
	if _, err := dec.Token(); err == nil {
		return nil, fmt.Errorf("trailing data after JSON value")
	}
	return v, nil
}

func decodeValue(dec *json.Decoder) (any, error) {
	tok, err := dec.Token()
	if err != nil {
		return nil, err
	}
	switch t := tok.(type) {
	case json.Delim:
		switch t {
		case '{':
			o := NewObj()
			for dec.More() {
				kt, err := dec.Token()
				if err != nil {
					return nil, err
				}
				k, _ := kt.(string)
				v, err := decodeValue(dec)
				if err != nil {
					return nil, err
				}
				o.Set(k, v)
			}
			if _, err := dec.Token(); err != nil {
				return nil, err
			}
			return o, nil
		case '[':
			out := []any{}
			for dec.More() {
				v, err := decodeValue(dec)
				if err != nil {
					return nil, err
				}
				out = append(out, v)
			}
			if _, err := dec.Token(); err != nil {
				return nil, err
			}
			return out, nil
		}
		return nil, fmt.Errorf("unexpected delimiter %v", t)
	case json.Number:
		s := string(t)
		if !strings.ContainsAny(s, ".eE") {
			return pyInt(s), nil
		}
		f, err := strconv.ParseFloat(s, 64)
		if err != nil {
			return nil, err
		}
		return f, nil
	}
	return tok, nil
}

// dumpPy renders v exactly as Python's json.dumps(v, indent=indent) with the default
// ensure_ascii=True; indent < 0 is the compact default (", " and ": " separators).
func dumpPy(v any, indent int) string {
	var b strings.Builder
	writePy(&b, v, indent, 0)
	return b.String()
}

func writePy(b *strings.Builder, v any, indent, level int) {
	switch t := v.(type) {
	case nil:
		b.WriteString("null")
	case bool:
		if t {
			b.WriteString("true")
		} else {
			b.WriteString("false")
		}
	case int:
		b.WriteString(strconv.Itoa(t))
	case int64:
		b.WriteString(strconv.FormatInt(t, 10))
	case pyInt:
		b.WriteString(string(t))
	case float64:
		b.WriteString(pyFloatRepr(t))
	case string:
		writePyString(b, t)
	case []string:
		items := make([]any, len(t))
		for i, s := range t {
			items[i] = s
		}
		writePy(b, items, indent, level)
	case []any:
		if len(t) == 0 {
			b.WriteString("[]")
			return
		}
		b.WriteByte('[')
		for i, e := range t {
			if i > 0 {
				b.WriteByte(',')
				if indent < 0 {
					b.WriteByte(' ')
				}
			}
			writeIndent(b, indent, level+1)
			writePy(b, e, indent, level+1)
		}
		writeIndent(b, indent, level)
		b.WriteByte(']')
	case *Obj:
		if t.Len() == 0 {
			b.WriteString("{}")
			return
		}
		b.WriteByte('{')
		for i, k := range t.Keys() {
			if i > 0 {
				b.WriteByte(',')
				if indent < 0 {
					b.WriteByte(' ')
				}
			}
			writeIndent(b, indent, level+1)
			writePyString(b, k)
			b.WriteString(": ")
			writePy(b, t.vals[k], indent, level+1)
		}
		writeIndent(b, indent, level)
		b.WriteByte('}')
	default:
		writePyString(b, fmt.Sprint(t))
	}
}

func writeIndent(b *strings.Builder, indent, level int) {
	if indent < 0 {
		return
	}
	b.WriteByte('\n')
	b.WriteString(strings.Repeat(" ", indent*level))
}

func writePyString(b *strings.Builder, s string) {
	b.WriteByte('"')
	for _, r := range s {
		switch r {
		case '"':
			b.WriteString(`\"`)
		case '\\':
			b.WriteString(`\\`)
		case '\n':
			b.WriteString(`\n`)
		case '\r':
			b.WriteString(`\r`)
		case '\t':
			b.WriteString(`\t`)
		case '\b':
			b.WriteString(`\b`)
		case '\f':
			b.WriteString(`\f`)
		default:
			switch {
			case r >= 0x20 && r < 0x7f:
				b.WriteRune(r)
			case r > 0xffff:
				r -= 0x10000
				fmt.Fprintf(b, `\u%04x\u%04x`, 0xd800+(r>>10), 0xdc00+(r&0x3ff))
			default:
				fmt.Fprintf(b, `\u%04x`, r)
			}
		}
	}
	b.WriteByte('"')
}

// pyFloatRepr is Python's repr(float): shortest round-trip digits, fixed notation for
// 1e-4 <= |x| < 1e16 (always with a fractional part), exponent notation otherwise.
func pyFloatRepr(f float64) string {
	switch {
	case math.IsNaN(f):
		return "NaN"
	case math.IsInf(f, 1):
		return "Infinity"
	case math.IsInf(f, -1):
		return "-Infinity"
	case f == 0:
		if math.Signbit(f) {
			return "-0.0"
		}
		return "0.0"
	}
	abs := math.Abs(f)
	if abs >= 1e16 || abs < 1e-4 {
		return strconv.FormatFloat(f, 'e', -1, 64)
	}
	s := strconv.FormatFloat(f, 'f', -1, 64)
	if !strings.Contains(s, ".") {
		s += ".0"
	}
	return s
}

// pyRound is Python's round(x, n): correctly rounded, ties to even on the exact binary value.
func pyRound(x float64, n int) float64 {
	if math.IsNaN(x) || math.IsInf(x, 0) {
		return x
	}
	r, _ := strconv.ParseFloat(strconv.FormatFloat(x, 'f', n, 64), 64)
	return r
}

// pyStr is Python's str() for the JSON-decoded value shapes this port handles.
func pyStr(v any) string {
	switch t := v.(type) {
	case nil:
		return "None"
	case string:
		return t
	case bool:
		if t {
			return "True"
		}
		return "False"
	case pyInt:
		return string(t)
	case int:
		return strconv.Itoa(t)
	case float64:
		r := pyFloatRepr(t)
		switch r {
		case "NaN":
			return "nan"
		case "Infinity":
			return "inf"
		case "-Infinity":
			return "-inf"
		}
		return r
	}
	return dumpPy(v, -1)
}

// truthy is Python truthiness over decoded JSON values.
func truthy(v any) bool {
	switch t := v.(type) {
	case nil:
		return false
	case bool:
		return t
	case string:
		return t != ""
	case pyInt:
		n, err := strconv.ParseInt(string(t), 10, 64)
		return err != nil || n != 0
	case int:
		return t != 0
	case float64:
		return t != 0
	case []any:
		return len(t) > 0
	case *Obj:
		return t.Len() > 0
	case map[string]any:
		return len(t) > 0
	}
	return true
}

// pyFloat is fleet_accounts._age_min's float(raw): numbers and numeric strings, else nil.
func pyFloat(v any) *float64 {
	var f float64
	switch t := v.(type) {
	case nil:
		return nil
	case float64:
		f = t
	case pyInt:
		n, err := strconv.ParseFloat(string(t), 64)
		if err != nil {
			return nil
		}
		f = n
	case int:
		f = float64(t)
	case bool:
		if t {
			f = 1
		}
	case string:
		n, err := strconv.ParseFloat(strings.ReplaceAll(pyStrip(t), "_", ""), 64)
		if err != nil {
			return nil
		}
		f = n
	default:
		return nil
	}
	return &f
}

// pyIntOf is Python's int(x or 0) over decoded values.
func pyIntOf(v any) int {
	if f := pyFloat(v); f != nil {
		return int(*f)
	}
	return 0
}

// validUTF8 mirrors bytes.decode("utf-8", "replace"): each maximal invalid subsequence
// becomes one U+FFFD.
func validUTF8(b []byte) string {
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
		i += invalidPrefixLen(b[i:])
		sb.WriteRune(utf8.RuneError)
	}
	return sb.String()
}

// invalidPrefixLen is the length of the maximal subpart of an ill-formed sequence at b[0].
func invalidPrefixLen(b []byte) int {
	c := b[0]
	var need int
	lo, hi := byte(0x80), byte(0xbf)
	switch {
	case c >= 0xc2 && c <= 0xdf:
		need = 1
	case c == 0xe0:
		need, lo = 2, 0xa0
	case c >= 0xe1 && c <= 0xec, c == 0xee, c == 0xef:
		need = 2
	case c == 0xed:
		need, hi = 2, 0x9f
	case c == 0xf0:
		need, lo = 3, 0x90
	case c >= 0xf1 && c <= 0xf3:
		need = 3
	case c == 0xf4:
		need, hi = 3, 0x8f
	default:
		return 1
	}
	n := 1
	for k := 0; k < need && n < len(b); k++ {
		x := b[n]
		if k == 0 {
			if x < lo || x > hi {
				return n
			}
		} else if x < 0x80 || x > 0xbf {
			return n
		}
		n++
	}
	return n
}

// isPySpace is Python's str.isspace for one rune.
func isPySpace(r rune) bool {
	switch {
	case r == ' ', r >= '\t' && r <= '\r', r >= 0x1c && r <= 0x1f, r == 0x85, r == 0xa0, r == 0x1680,
		r >= 0x2000 && r <= 0x200a, r == 0x2028, r == 0x2029, r == 0x202f, r == 0x205f, r == 0x3000:
		return true
	}
	return false
}

func pyStrip(s string) string { return strings.TrimFunc(s, isPySpace) }

// pySplitJoin is Python's " ".join(s.split()).
func pySplitJoin(s string) string { return strings.Join(strings.FieldsFunc(s, isPySpace), " ") }

// pySplitlines is Python's str.splitlines() (no keepends).
func pySplitlines(s string) []string {
	var out []string
	start := 0
	for i := 0; i < len(s); {
		r, size := utf8.DecodeRuneInString(s[i:])
		brk := false
		switch r {
		case '\n', '\v', '\f', 0x1c, 0x1d, 0x1e, 0x85, 0x2028, 0x2029:
			brk = true
		case '\r':
			brk = true
			if i+1 < len(s) && s[i+1] == '\n' {
				out = append(out, s[start:i])
				i += 2
				start = i
				continue
			}
		}
		if brk {
			out = append(out, s[start:i])
			start = i + size
		}
		i += size
	}
	if start < len(s) {
		out = append(out, s[start:])
	}
	return out
}

// runeHead is Python's s[:n] on code points.
func runeHead(s string, n int) string {
	i := 0
	for pos := range s {
		if i == n {
			return s[:pos]
		}
		i++
	}
	return s
}

// padRight / padLeft are Python's f"{s:<n}" / f"{s:>n}" (code-point widths).
func padRight(s string, n int) string {
	if c := utf8.RuneCountInString(s); c < n {
		return s + strings.Repeat(" ", n-c)
	}
	return s
}

func padLeft(s string, n int) string {
	if c := utf8.RuneCountInString(s); c < n {
		return strings.Repeat(" ", n-c) + s
	}
	return s
}

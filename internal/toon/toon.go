package toon

import (
	"encoding/json"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"unicode"
)

// Options controls TOON encoding. The zero value is the shipped default: a comma
// delimiter and no length marker. Decode takes no Options — it recovers the structure,
// including which delimiter Encode used, from the encoded bytes, so a round-trip never
// has to thread options through.
//
// KeyFolding (collapsing single-key wrapper chains to dotted paths, spec v1.5) is a
// deliberate follow-on and is intentionally absent from this struct — see the package doc.
type Options struct {
	// Delimiter joins tabular row cells and the header field list. ',' (default), '\t',
	// and '|' are valid; any other value is refused by Encode.
	Delimiter rune
	// LengthMarker emits an optional '#' before an array's element count, e.g.
	// `name[#3]{...}:` instead of `name[3]{...}:`. Decode accepts either form.
	LengthMarker bool
}

// indentUnit is two spaces per nesting level, matching memview.EncodeTOON so the flat
// tabular case is byte-for-byte identical between the two encoders.
const indentUnit = "  "

func (o Options) delim() rune {
	if o.Delimiter == 0 {
		return ','
	}
	return o.Delimiter
}

// Encode renders a JSON value as TOON. Numbers/bools/nulls/strings round-trip by type;
// a string that merely looks typed ("123", "true") is quoted so Decode does not coerce
// it. It is deterministic: object keys are walked in sorted order (no map-order leak),
// so SweepFormats/cache callers see byte-identical output for a fixed value.
func Encode(v any, o Options) ([]byte, error) {
	switch o.delim() {
	case ',', '\t', '|':
	default:
		return nil, fmt.Errorf("toon: unsupported delimiter %q (want ',', '\\t', or '|')", o.delim())
	}
	var b strings.Builder
	if err := encodeRoot(&b, v, o); err != nil {
		return nil, err
	}
	return []byte(b.String()), nil
}

func encodeRoot(b *strings.Builder, v any, o Options) error {
	switch x := v.(type) {
	case map[string]any:
		if len(x) == 0 {
			b.WriteString("{}\n")
			return nil
		}
		return writeObject(b, x, 0, o)
	case []any:
		return writeArray(b, "", x, 0, o)
	default:
		cell, err := rootScalar(v, o)
		if err != nil {
			return err
		}
		b.WriteString(cell)
		b.WriteByte('\n')
		return nil
	}
}

// writeObject emits one `key: value` line per sorted key. A scalar value is inlined; an
// array value becomes a named tabular/list block; a nested object value becomes an
// indented block under a bare `key:` line (an empty object is a bare `key:` with no
// block — Decode disambiguates by lookahead).
func writeObject(b *strings.Builder, m map[string]any, indent int, o Options) error {
	pad := strings.Repeat(indentUnit, indent)
	for _, k := range sortedKeys(m) {
		key := keyName(k, o)
		switch x := m[k].(type) {
		case map[string]any:
			b.WriteString(pad + key + ":\n")
			if len(x) > 0 {
				if err := writeObject(b, x, indent+1, o); err != nil {
					return err
				}
			}
		case []any:
			if err := writeArray(b, key, x, indent, o); err != nil {
				return err
			}
		default:
			cell, err := cellValue(m[k], o)
			if err != nil {
				return err
			}
			b.WriteString(pad + key + ": " + cell + "\n")
		}
	}
	return nil
}

// writeArray emits a uniform array of flat objects as a tabular block (header declaring
// the count + field names once, then one delimiter-joined row per element); every other
// array shape (ragged, scalar, or with nested objects/arrays in an element) falls back to
// a safe per-item list (`name[N]:` + one `- <json>` line per element), never a corrupt
// tabular header with a wrong field count.
func writeArray(b *strings.Builder, name string, arr []any, indent int, o Options) error {
	pad := strings.Repeat(indentUnit, indent)
	count := lengthToken(len(arr), o)
	delim := string(o.delim())
	if fields, ok := uniformFlatFields(arr); ok {
		toks := make([]string, len(fields))
		for i, f := range fields {
			toks[i] = keyName(f, o)
		}
		b.WriteString(pad + name + "[" + count + "]{" + strings.Join(toks, delim) + "}:\n")
		rowPad := strings.Repeat(indentUnit, indent+1)
		for _, elem := range arr {
			row := elem.(map[string]any)
			cells := make([]string, len(fields))
			for i, f := range fields {
				c, err := cellValue(row[f], o)
				if err != nil {
					return err
				}
				cells[i] = c
			}
			b.WriteString(rowPad + strings.Join(cells, delim) + "\n")
		}
		return nil
	}
	// Non-uniform / scalar / nested-in-cell: per-item JSON list fallback. json.Marshal is
	// deterministic (it sorts map keys) and lossless, so any element shape round-trips.
	b.WriteString(pad + name + "[" + count + "]:\n")
	itemPad := strings.Repeat(indentUnit, indent+1)
	for _, elem := range arr {
		js, err := json.Marshal(elem)
		if err != nil {
			return fmt.Errorf("toon: cannot encode array element: %w", err)
		}
		b.WriteString(itemPad + "- " + string(js) + "\n")
	}
	return nil
}

func lengthToken(n int, o Options) string {
	if o.LengthMarker {
		return "#" + strconv.Itoa(n)
	}
	return strconv.Itoa(n)
}

// uniformFlatFields reports the shared sorted field set of arr iff arr is a non-empty
// array of objects that all carry the identical key set and only scalar values (the
// tabular-eligible shape). A single nested object/array in any cell, a differing key set,
// or a non-object element demotes the whole array — the "detect, don't corrupt" rule.
func uniformFlatFields(arr []any) ([]string, bool) {
	if len(arr) == 0 {
		return nil, false
	}
	first, ok := arr[0].(map[string]any)
	if !ok || len(first) == 0 {
		return nil, false
	}
	fields := sortedKeys(first)
	for _, elem := range arr {
		m, ok := elem.(map[string]any)
		if !ok || len(m) != len(fields) {
			return nil, false
		}
		for _, f := range fields {
			val, present := m[f]
			if !present || !isScalar(val) {
				return nil, false
			}
		}
	}
	return fields, true
}

func isScalar(v any) bool {
	switch v.(type) {
	case map[string]any, []any:
		return false
	default:
		return true
	}
}

func sortedKeys(m map[string]any) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// cellValue renders a scalar in cell / after-`key:` position. A string is bare when a
// bare rendering is unambiguous, else JSON-string-quoted — the same "quote only when
// required" rule memview.toonCell uses (reused here, widened for the codec's decoder:
// also the active delimiter, CR, and leading/trailing whitespace, which memview's
// encode-only surface never had to round-trip). Numbers/bools/null render as their JSON
// token, so Decode reads a number back as a number and a quoted "42" back as a string.
func cellValue(v any, o Options) (string, error) {
	switch x := v.(type) {
	case string:
		if needsCellQuote(x, o.delim()) {
			q, _ := json.Marshal(x)
			return string(q), nil
		}
		return x, nil
	case nil:
		return "null", nil
	case bool:
		if x {
			return "true", nil
		}
		return "false", nil
	case json.Number:
		return x.String(), nil
	case float64, float32, int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64:
		q, err := json.Marshal(v)
		if err != nil {
			return "", fmt.Errorf("toon: cannot encode number %v: %w", v, err)
		}
		return string(q), nil
	default:
		return "", fmt.Errorf("toon: unsupported scalar type %T", v)
	}
}

// rootScalar renders a bare top-level scalar. Non-strings match cellValue. A string is
// held to a stricter bar than a cell: it is quoted unless it is a plain, unambiguous word,
// because at the start of a line a ':' or '[' would otherwise be misread as object/array
// structure by the decoder.
func rootScalar(v any, o Options) (string, error) {
	s, ok := v.(string)
	if !ok {
		return cellValue(v, o)
	}
	if isSafeBareScalar(s) {
		return s, nil
	}
	q, _ := json.Marshal(s)
	return string(q), nil
}

// needsCellQuote is memview.toonCell's rule, widened for the round-trip decoder.
func needsCellQuote(s string, delim rune) bool {
	if s == "" || looksTyped(s) {
		return true
	}
	if strings.TrimSpace(s) != s { // leading/trailing whitespace would be lost or misparsed
		return true
	}
	for _, r := range s {
		if r == delim || r == '"' || r == '\n' || r == '\r' {
			return true
		}
	}
	return false
}

// looksTyped mirrors memview.looksTyped exactly: a value that would be misread as a
// number/bool/null if left bare. Replicated (not imported) because internal/toon sits a
// tier below internal/memview and may not import upward.
func looksTyped(v string) bool {
	if v == "true" || v == "false" || v == "null" {
		return true
	}
	_, err := strconv.ParseFloat(v, 64)
	return err == nil
}

// isSafeBareScalar reports whether a root-position string can be emitted bare without the
// decoder mistaking it for structure. It must be a non-empty, non-typed run of letters,
// digits, and a few safe punctuation marks, with no edge whitespace and no leading '-'/'#'
// (which introduce list items / length markers).
func isSafeBareScalar(s string) bool {
	if s == "" || looksTyped(s) || strings.TrimSpace(s) != s {
		return false
	}
	if strings.HasPrefix(s, "-") || strings.HasPrefix(s, "#") {
		return false
	}
	for _, r := range s {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			continue
		}
		switch r {
		case '_', '.', '/', '@', '+', '=', '~':
			continue
		}
		return false
	}
	return true
}

// keyName renders an object key or array name bare when it is a safe identifier, else
// JSON-quoted so a key containing ':'/'['/the delimiter cannot break the line grammar.
func keyName(s string, o Options) string {
	if isSafeIdent(s, o.delim()) {
		return s
	}
	q, _ := json.Marshal(s)
	return string(q)
}

func isSafeIdent(s string, delim rune) bool {
	if s == "" || strings.HasPrefix(s, "-") || strings.HasPrefix(s, "#") {
		return false
	}
	for _, r := range s {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			continue
		}
		switch r {
		case '_', '.', '-':
			if r == delim {
				return false
			}
			continue
		}
		return false
	}
	return true
}

// TabularEligibility returns the fraction (0..1) of the value's scalar leaves that live
// inside a uniform, flat, tabular-eligible array — the measured shape signal issue
// #3066's gate consumes. It is ~1.0 for a uniform array of flat objects (every leaf is a
// tabular cell) and ~0.0 for a deeply nested object (no leaf is in a tabular array). A
// value with no scalar leaves returns 0.
func TabularEligibility(v any) float64 {
	eligible, total := tabularLeaves(v)
	if total == 0 {
		return 0
	}
	return float64(eligible) / float64(total)
}

func tabularLeaves(v any) (eligible, total int) {
	switch x := v.(type) {
	case map[string]any:
		for _, val := range x {
			e, t := tabularLeaves(val)
			eligible += e
			total += t
		}
		return
	case []any:
		if fields, ok := uniformFlatFields(x); ok {
			n := len(x) * len(fields)
			return n, n
		}
		for _, elem := range x {
			e, t := tabularLeaves(elem)
			eligible += e
			total += t
		}
		return
	default:
		return 0, 1 // a scalar leaf, not in any tabular array
	}
}

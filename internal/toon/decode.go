package toon

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
)

// Decode parses TOON bytes produced by Encode back into a JSON value. It is the round-
// trip witness: Decode(Encode(v)) deep-equals v for values in the supported domain, so a
// silent field-order or type drift in Encode is caught by the property test rather than
// shipped. Numbers come back as float64 and objects as map[string]any (encoding/json's
// native types); a quoted-looking string stays a string.
//
// Decode needs no Options: the delimiter is recovered from each tabular header, and the
// length marker is accepted whether or not it is present.
func Decode(b []byte) (any, error) {
	lines := splitLines(b)
	if len(lines) == 0 {
		return nil, fmt.Errorf("toon: empty input")
	}
	cur := 0
	v, err := parseNode(lines, &cur, 0)
	if err != nil {
		return nil, err
	}
	if cur != len(lines) {
		return nil, fmt.Errorf("toon: %d trailing line(s) after value (first: %q)", len(lines)-cur, lines[cur].content)
	}
	return v, nil
}

type line struct {
	indent  int
	content string
}

// splitLines breaks the input into non-empty (indent, content) lines. Indentation is
// structural two-space padding; a real newline never occurs inside a token (strings with
// newlines are JSON-escaped onto one physical line), so splitting on '\n' is safe.
func splitLines(b []byte) []line {
	var out []line
	for _, raw := range strings.Split(string(b), "\n") {
		raw = strings.TrimSuffix(raw, "\r")
		trimmed := strings.TrimLeft(raw, " ")
		if trimmed == "" {
			continue
		}
		out = append(out, line{indent: (len(raw) - len(trimmed)) / 2, content: trimmed})
	}
	return out
}

// parseNode parses exactly one value beginning at lines[*cur].
func parseNode(lines []line, cur *int, indent int) (any, error) {
	content := lines[*cur].content
	if content == "{}" {
		*cur++
		return map[string]any{}, nil
	}
	label, rest, quoted, err := parseLabel(content)
	if err != nil {
		return nil, err
	}
	switch {
	case strings.HasPrefix(rest, "["):
		if label == "" { // anonymous header => a root array
			_, arr, err := parseArray(lines, cur, indent)
			return arr, err
		}
		return parseObject(lines, cur, indent) // named header => object with an array field
	case strings.HasPrefix(rest, ":"):
		return parseObject(lines, cur, indent)
	default: // no structural sigil => a bare scalar line
		*cur++
		if quoted {
			return label, nil
		}
		return cellDecode(content)
	}
}

// parseObject consumes consecutive field lines at `indent`. A field is either an array
// (a header line whose name is the key) or a `key: value` / `key:` line (scalar, nested
// object, or empty object).
func parseObject(lines []line, cur *int, indent int) (any, error) {
	m := map[string]any{}
	for *cur < len(lines) && lines[*cur].indent == indent {
		label, rest, _, err := parseLabel(lines[*cur].content)
		if err != nil {
			return nil, err
		}
		switch {
		case strings.HasPrefix(rest, "["):
			name, arr, err := parseArray(lines, cur, indent)
			if err != nil {
				return nil, err
			}
			m[name] = arr
		case strings.HasPrefix(rest, ":"):
			valuePart := rest[1:]
			*cur++
			if valuePart == "" {
				if *cur < len(lines) && lines[*cur].indent == indent+1 {
					sub, err := parseObject(lines, cur, indent+1)
					if err != nil {
						return nil, err
					}
					m[label] = sub
				} else {
					m[label] = map[string]any{}
				}
				continue
			}
			cv, err := cellDecode(strings.TrimPrefix(valuePart, " "))
			if err != nil {
				return nil, err
			}
			m[label] = cv
		default:
			return nil, fmt.Errorf("toon: malformed line %q", lines[*cur].content)
		}
	}
	return m, nil
}

// parseArray consumes an array header at lines[*cur] plus its N child rows/items and
// returns the array's name (the key it hangs under; "" for a root array) and value.
func parseArray(lines []line, cur *int, indent int) (string, []any, error) {
	name, count, fields, delim, tabular, err := parseHeader(lines[*cur].content)
	if err != nil {
		return "", nil, err
	}
	*cur++
	arr := make([]any, 0, count)
	for i := 0; i < count; i++ {
		if *cur >= len(lines) || lines[*cur].indent != indent+1 {
			return "", nil, fmt.Errorf("toon: array %q declared %d row(s), found %d", name, count, i)
		}
		c := lines[*cur].content
		*cur++
		if tabular {
			cells := splitCells(c, delim)
			if len(cells) != len(fields) {
				return "", nil, fmt.Errorf("toon: row %q has %d cell(s), header declares %d field(s)", c, len(cells), len(fields))
			}
			row := make(map[string]any, len(fields))
			for j, f := range fields {
				cv, err := cellDecode(cells[j])
				if err != nil {
					return "", nil, err
				}
				row[f] = cv
			}
			arr = append(arr, row)
			continue
		}
		if !strings.HasPrefix(c, "- ") {
			return "", nil, fmt.Errorf("toon: list item must start with %q, got %q", "- ", c)
		}
		var item any
		if err := json.Unmarshal([]byte(strings.TrimPrefix(c, "- ")), &item); err != nil {
			return "", nil, fmt.Errorf("toon: bad list item %q: %w", c, err)
		}
		arr = append(arr, item)
	}
	return name, arr, nil
}

// parseHeader parses a `name[#?N]{f1,f2}:` (tabular) or `name[#?N]:` (list) header line.
func parseHeader(content string) (name string, count int, fields []string, delim rune, tabular bool, err error) {
	name, rest, _, err := parseLabel(content)
	if err != nil {
		return "", 0, nil, ',', false, err
	}
	if !strings.HasPrefix(rest, "[") {
		return "", 0, nil, ',', false, fmt.Errorf("toon: header %q missing '['", content)
	}
	j := 1
	if j < len(rest) && rest[j] == '#' {
		j++
	}
	start := j
	for j < len(rest) && rest[j] >= '0' && rest[j] <= '9' {
		j++
	}
	if j == start {
		return "", 0, nil, ',', false, fmt.Errorf("toon: header %q missing count", content)
	}
	count, _ = strconv.Atoi(rest[start:j])
	if j >= len(rest) || rest[j] != ']' {
		return "", 0, nil, ',', false, fmt.Errorf("toon: header %q missing ']'", content)
	}
	j++
	switch {
	case j < len(rest) && rest[j] == '{':
		close := findClose(rest, j)
		if close < 0 || close+1 >= len(rest) || rest[close+1] != ':' {
			return "", 0, nil, ',', false, fmt.Errorf("toon: header %q malformed field list", content)
		}
		inner := rest[j+1 : close]
		delim = detectDelim(inner)
		for _, ft := range splitCells(inner, delim) {
			f, err := decodeKey(ft)
			if err != nil {
				return "", 0, nil, ',', false, err
			}
			fields = append(fields, f)
		}
		return name, count, fields, delim, true, nil
	case j < len(rest) && rest[j] == ':' && j+1 == len(rest):
		return name, count, nil, ',', false, nil
	default:
		return "", 0, nil, ',', false, fmt.Errorf("toon: header %q malformed", content)
	}
}

// parseLabel splits a line's leading label (a bare identifier or a JSON-quoted string)
// from the structural remainder. rest begins with '[' (array), ':' (field), or "" (the
// whole line was the label — a bare scalar). quoted reports whether the label was a JSON
// string token (a quoted bare scalar is therefore a string, not a number).
func parseLabel(content string) (label, rest string, quoted bool, err error) {
	if strings.HasPrefix(content, "\"") {
		end := scanQuoted(content)
		if end < 0 {
			return "", "", false, fmt.Errorf("toon: unterminated quote in %q", content)
		}
		var s string
		if err := json.Unmarshal([]byte(content[:end]), &s); err != nil {
			return "", "", false, fmt.Errorf("toon: bad quoted label in %q: %w", content, err)
		}
		return s, content[end:], true, nil
	}
	for i, r := range content {
		if r == '[' || r == ':' {
			return content[:i], content[i:], false, nil
		}
	}
	return content, "", false, nil
}

// decodeKey turns a header field / key token (bare or JSON-quoted) into its string name.
func decodeKey(tok string) (string, error) {
	if strings.HasPrefix(tok, "\"") {
		var s string
		if err := json.Unmarshal([]byte(tok), &s); err != nil {
			return "", fmt.Errorf("toon: bad quoted key %q: %w", tok, err)
		}
		return s, nil
	}
	return tok, nil
}

// cellDecode turns a single scalar cell token back into its typed value: a quoted token
// is a string (so "42"/"true" stay strings), the bare words true/false/null are those
// literals, a bare numeric token is a float64, and any other bare token is a string.
func cellDecode(tok string) (any, error) {
	if tok == "" {
		return "", nil
	}
	if tok[0] == '"' {
		var s string
		if err := json.Unmarshal([]byte(tok), &s); err != nil {
			return nil, fmt.Errorf("toon: bad quoted cell %q: %w", tok, err)
		}
		return s, nil
	}
	switch tok {
	case "true":
		return true, nil
	case "false":
		return false, nil
	case "null":
		return nil, nil
	}
	if f, err := strconv.ParseFloat(tok, 64); err == nil {
		return f, nil
	}
	return tok, nil
}

// splitCells splits s on delim, ignoring any delim inside a "..." JSON string token. The
// returned tokens keep their quotes so cellDecode/decodeKey can json-unquote them.
func splitCells(s string, delim rune) []string {
	var out []string
	var cur strings.Builder
	inQ, esc := false, false
	for _, r := range s {
		switch {
		case esc:
			cur.WriteRune(r)
			esc = false
		case inQ && r == '\\':
			cur.WriteRune(r)
			esc = true
		case r == '"':
			inQ = !inQ
			cur.WriteRune(r)
		case r == delim && !inQ:
			out = append(out, cur.String())
			cur.Reset()
		default:
			cur.WriteRune(r)
		}
	}
	out = append(out, cur.String())
	return out
}

// detectDelim finds which of ',', '\t', '|' separates a header's field list by scanning
// outside quotes; a delimiter char inside a quoted field name is not structural. Defaults
// to ',' when the list has a single field (no separator to observe).
func detectDelim(s string) rune {
	inQ, esc := false, false
	for _, r := range s {
		switch {
		case esc:
			esc = false
		case inQ && r == '\\':
			esc = true
		case r == '"':
			inQ = !inQ
		case !inQ && (r == ',' || r == '\t' || r == '|'):
			return r
		}
	}
	return ','
}

// scanQuoted returns the index just past the closing quote of the JSON string starting at
// s[0]=='"', or -1 if unterminated.
func scanQuoted(s string) int {
	esc := false
	for i := 1; i < len(s); i++ {
		switch {
		case esc:
			esc = false
		case s[i] == '\\':
			esc = true
		case s[i] == '"':
			return i + 1
		}
	}
	return -1
}

// findClose returns the index of the '}' matching the '{' at s[open], ignoring braces
// inside quoted field names.
func findClose(s string, open int) int {
	inQ, esc := false, false
	for i := open + 1; i < len(s); i++ {
		switch {
		case esc:
			esc = false
		case inQ && s[i] == '\\':
			esc = true
		case s[i] == '"':
			inQ = !inQ
		case !inQ && s[i] == '}':
			return i
		}
	}
	return -1
}

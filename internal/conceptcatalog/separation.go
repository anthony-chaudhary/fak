package conceptcatalog

import (
	"bytes"
	"encoding/json"
	"fmt"
)

// Separation is a MUTUAL property of two concepts, not a property of one row.
//
// A row that says "I am not Cache B" leaves every reader who arrives at Cache B
// still holding both meanings: the boundary was drawn on one side of the line only.
// So the authoring path writes the other half itself, into the twin's own row, in
// the twin's own file - and it does that by rewriting exactly the bytes of that
// row's `distinct_from` array, leaving every other byte and the file's existing
// layout untouched. Re-marshalling the file would reformat rows nobody edited and
// bury the one-line change in a whole-file diff on a shared tree.

// addBackReference adds ref to row rowID's distinct_from array inside one data
// file's bytes. It reports whether the bytes changed; a reference that is already
// there is a no-op, so authoring the same boundary twice stays idempotent.
func addBackReference(b []byte, rowID, ref string) ([]byte, bool, error) {
	obj, ok := rowObjectSpan(b, rowID)
	if !ok {
		return b, false, fmt.Errorf("row %q not found", rowID)
	}
	seg := b[obj.start:obj.end]
	arr, ok := stringArraySpan(seg, "distinct_from")
	if !ok {
		return b, false, fmt.Errorf("row %q has no distinct_from array", rowID)
	}
	raw := seg[arr.start:arr.end]
	var vals []string
	if err := json.Unmarshal(raw, &vals); err != nil {
		return b, false, fmt.Errorf("row %q distinct_from: %w", rowID, err)
	}
	for _, v := range vals {
		if v == ref {
			return b, false, nil
		}
	}
	at := obj.start + arr.start
	rendered, err := renderRefs(append(vals, ref), bytes.Contains(raw, []byte("\n")), lineIndent(b, at))
	if err != nil {
		return b, false, err
	}
	out := make([]byte, 0, len(b)+len(rendered))
	out = append(out, b[:at]...)
	out = append(out, rendered...)
	out = append(out, b[obj.start+arr.end:]...)
	return out, true, nil
}

// renderRefs re-emits the array in the layout it already had: one element per line
// under the field's own indent, or a single line, so the diff shows the added
// reference and nothing else.
func renderRefs(vals []string, multiline bool, indent string) ([]byte, error) {
	quoted := make([][]byte, 0, len(vals))
	for _, v := range vals {
		q, err := json.Marshal(v)
		if err != nil {
			return nil, err
		}
		quoted = append(quoted, q)
	}
	var out bytes.Buffer
	out.WriteByte('[')
	if !multiline {
		out.Write(bytes.Join(quoted, []byte(", ")))
		out.WriteByte(']')
		return out.Bytes(), nil
	}
	for i, q := range quoted {
		if i > 0 {
			out.WriteByte(',')
		}
		out.WriteString("\n" + indent + "  ")
		out.Write(q)
	}
	out.WriteString("\n" + indent + "]")
	return out.Bytes(), nil
}

type byteSpan struct{ start, end int }

// rowObjectSpan finds the object of row rowID among the objects nested directly in
// the file's top-level rows array.
func rowObjectSpan(b []byte, rowID string) (byteSpan, bool) {
	for _, o := range rowObjectSpans(b) {
		var idOnly struct {
			ID string `json:"id"`
		}
		if err := json.Unmarshal(b[o.start:o.end], &idOnly); err != nil {
			continue
		}
		if idOnly.ID == rowID {
			return o, true
		}
	}
	return byteSpan{}, false
}

// rowObjectSpans returns the byte span of every object at `{"rows": [ HERE ]}`.
func rowObjectSpans(b []byte) []byteSpan {
	var out []byteSpan
	depth, start := 0, -1
	inStr, esc := false, false
	for i := 0; i < len(b); i++ {
		ch := b[i]
		if inStr {
			switch {
			case esc:
				esc = false
			case ch == '\\':
				esc = true
			case ch == '"':
				inStr = false
			}
			continue
		}
		switch ch {
		case '"':
			inStr = true
		case '{', '[':
			depth++
			if ch == '{' && depth == 3 && start < 0 {
				start = i
			}
		case '}', ']':
			if ch == '}' && depth == 3 && start >= 0 {
				out = append(out, byteSpan{start, i + 1})
				start = -1
			}
			depth--
		}
	}
	return out
}

// stringArraySpan locates the array value of a top-level key within one JSON
// object's bytes. Scanning rather than searching keeps a key name that also occurs
// inside a string value from matching.
func stringArraySpan(seg []byte, key string) (byteSpan, bool) {
	depth := 0
	for i := 0; i < len(seg); i++ {
		switch seg[i] {
		case '{', '[':
			depth++
		case '}', ']':
			depth--
		case '"':
			j := i + 1
			for j < len(seg) && seg[j] != '"' {
				if seg[j] == '\\' {
					j++
				}
				j++
			}
			if j >= len(seg) {
				return byteSpan{}, false
			}
			k := skipSpace(seg, j+1)
			if depth == 1 && string(seg[i+1:j]) == key && k < len(seg) && seg[k] == ':' {
				k = skipSpace(seg, k+1)
				if k >= len(seg) || seg[k] != '[' {
					return byteSpan{}, false
				}
				end := matchBracket(seg, k)
				if end < 0 {
					return byteSpan{}, false
				}
				return byteSpan{k, end + 1}, true
			}
			i = j
		}
	}
	return byteSpan{}, false
}

func matchBracket(b []byte, open int) int {
	depth := 0
	inStr, esc := false, false
	for i := open; i < len(b); i++ {
		ch := b[i]
		if inStr {
			switch {
			case esc:
				esc = false
			case ch == '\\':
				esc = true
			case ch == '"':
				inStr = false
			}
			continue
		}
		switch ch {
		case '"':
			inStr = true
		case '[':
			depth++
		case ']':
			depth--
			if depth == 0 {
				return i
			}
		}
	}
	return -1
}

func skipSpace(b []byte, i int) int {
	for i < len(b) && (b[i] == ' ' || b[i] == '\t' || b[i] == '\n' || b[i] == '\r') {
		i++
	}
	return i
}

// lineIndent is the whitespace run that opens the line containing offset at.
func lineIndent(b []byte, at int) string {
	start := bytes.LastIndexByte(b[:at], '\n') + 1
	i := start
	for i < at && (b[i] == ' ' || b[i] == '\t') {
		i++
	}
	return string(b[start:i])
}

// UnseparatedPair is one pair of concepts a reader can confuse whose boundary is
// not drawn from both sides. The pairs are DISCOVERED by the scorecard from the
// catalog's own names - the rule lives there once, so the authoring gate and the
// grader cannot drift into two different definitions of confusable.
type UnseparatedPair struct {
	A     string `json:"a"`
	B     string `json:"b"`
	Kind  string `json:"kind"`
	Why   string `json:"why"`
	State string `json:"state"`
}

// Other returns the id at the far end of the pair from id, and whether id is in it.
func (p UnseparatedPair) Other(id string) (string, bool) {
	switch {
	case norm(p.A) == norm(id):
		return p.B, true
	case norm(p.B) == norm(id):
		return p.A, true
	}
	return "", false
}

type shadowSnapshot struct {
	Corpus struct {
		Separation struct {
			Unseparated []UnseparatedPair `json:"unseparated"`
		} `json:"separation"`
	} `json:"corpus"`
}

// unseparatedFor lists the twins of id that the planned catalog would leave
// unseparated. Pre-existing debt between OTHER concepts is not the author's to
// pay; the name they are landing right now is.
func (s shadowSnapshot) unseparatedFor(id string) []UnseparatedPair {
	var out []UnseparatedPair
	for _, p := range s.Corpus.Separation.Unseparated {
		if _, ok := p.Other(id); ok {
			out = append(out, p)
		}
	}
	return out
}

package strictjson

import (
	"bytes"
	"encoding/json"
)

// RepairClosedJSON removes only syntax that cannot change a JSON value: an
// exact closed Markdown fence and commas immediately before a closing object
// or array delimiter. It never inserts bytes or salvages an unclosed value.
func RepairClosedJSON(raw []byte) ([]byte, bool) {
	if json.Valid(raw) {
		return raw, true
	}

	candidate, ok := stripClosedJSONFence(raw)
	if !ok {
		candidate = raw
	}
	repaired := removeTrailingCommas(candidate)
	if !json.Valid(repaired) {
		return raw, false
	}
	return repaired, true
}

func stripClosedJSONFence(raw []byte) ([]byte, bool) {
	trimmed := bytes.TrimSpace(raw)
	if !bytes.HasPrefix(trimmed, []byte("```")) || !bytes.HasSuffix(trimmed, []byte("```")) || len(trimmed) < 6 {
		return raw, false
	}

	lineEnd := bytes.IndexByte(trimmed[3:], '\n')
	if lineEnd < 0 {
		return raw, false
	}
	header := bytes.TrimSpace(trimmed[3 : 3+lineEnd])
	if len(header) != 0 && !bytes.Equal(header, []byte("json")) && !bytes.Equal(header, []byte("tool_call")) {
		return raw, false
	}
	bodyStart := 3 + lineEnd + 1
	bodyEnd := len(trimmed) - 3
	if bodyStart > bodyEnd {
		return raw, false
	}
	return bytes.TrimSpace(trimmed[bodyStart:bodyEnd]), true
}

func removeTrailingCommas(raw []byte) []byte {
	var out bytes.Buffer
	out.Grow(len(raw))
	inString := false
	escaped := false

	for i, b := range raw {
		if inString {
			out.WriteByte(b)
			if escaped {
				escaped = false
			} else if b == '\\' {
				escaped = true
			} else if b == '"' {
				inString = false
			}
			continue
		}
		if b == '"' {
			inString = true
			out.WriteByte(b)
			continue
		}
		if b == ',' {
			j := i + 1
			for j < len(raw) && isJSONSpace(raw[j]) {
				j++
			}
			if j < len(raw) && (raw[j] == '}' || raw[j] == ']') {
				k := i - 1
				for k >= 0 && isJSONSpace(raw[k]) {
					k--
				}
				if k >= 0 && raw[k] != '{' && raw[k] != '[' && raw[k] != ',' && raw[k] != ':' {
					continue
				}
			}
		}
		out.WriteByte(b)
	}
	return out.Bytes()
}

func isJSONSpace(b byte) bool {
	return b == ' ' || b == '\t' || b == '\r' || b == '\n'
}

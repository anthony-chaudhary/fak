package gateway

import (
	"bytes"
	"encoding/json"
)

// RepairJSON repairs malformed JSON text by stripping markdown code fences,
// trimming leading/trailing non-JSON text, removing trailing commas, closing
// unclosed strings, and balancing unclosed delimiters.
func RepairJSON(raw []byte) []byte {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 {
		return raw
	}
	if json.Valid(raw) {
		return raw
	}

	cleaned := StripCodeFences(raw)
	if json.Valid(cleaned) {
		return cleaned
	}

	cleaned = RemoveTrailingCommas(cleaned)
	if json.Valid(cleaned) {
		return cleaned
	}

	repaired := BalanceJSONDelimiters(cleaned)
	if json.Valid(repaired) {
		return repaired
	}

	repaired2 := RemoveTrailingCommas(repaired)
	if json.Valid(repaired2) {
		return repaired2
	}

	return repaired
}

// StripCodeFences removes markdown code fences and leading/trailing prose
// surrounding JSON objects or arrays.
func StripCodeFences(raw []byte) []byte {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 {
		return trimmed
	}

	s := trimmed
	startFence := bytes.Index(s, []byte("```"))
	if startFence == -1 {
		startFence = bytes.Index(s, []byte("~~~"))
	}
	if startFence != -1 {
		markerChar := s[startFence]
		fenceLen := 0
		for startFence+fenceLen < len(s) && s[startFence+fenceLen] == markerChar {
			fenceLen++
		}
		marker := s[startFence : startFence+fenceLen]
		contentStart := startFence + fenceLen
		if nl := bytes.IndexByte(s[contentStart:], '\n'); nl != -1 {
			contentStart += nl + 1
		} else {
			for contentStart < len(s) && s[contentStart] != '{' && s[contentStart] != '[' && s[contentStart] != markerChar {
				contentStart++
			}
		}

		endFence := bytes.LastIndex(s[contentStart:], marker)
		if endFence != -1 {
			s = bytes.TrimSpace(s[contentStart : contentStart+endFence])
		} else {
			s = bytes.TrimSpace(s[contentStart:])
		}
	}

	firstDelim := findFirstDelim(s)
	if firstDelim != -1 {
		s = s[firstDelim:]
		closedAt := findDelimClose(s)
		if closedAt != -1 {
			s = s[:closedAt+1]
		}
	}

	return bytes.TrimSpace(s)
}

func findFirstDelim(b []byte) int {
	for i, c := range b {
		if c == '{' || c == '[' {
			return i
		}
	}
	return -1
}

func findDelimClose(b []byte) int {
	var stack []byte
	inString := false
	escaped := false

	for i := 0; i < len(b); i++ {
		c := b[i]
		if inString {
			if escaped {
				escaped = false
			} else if c == '\\' {
				escaped = true
			} else if c == '"' {
				inString = false
			}
			continue
		}

		if c == '"' {
			inString = true
			continue
		}

		if c == '{' || c == '[' {
			stack = append(stack, c)
		} else if c == '}' {
			if len(stack) > 0 && stack[len(stack)-1] == '{' {
				stack = stack[:len(stack)-1]
				if len(stack) == 0 {
					return i
				}
			}
		} else if c == ']' {
			if len(stack) > 0 && stack[len(stack)-1] == '[' {
				stack = stack[:len(stack)-1]
				if len(stack) == 0 {
					return i
				}
			}
		}
	}
	return -1
}

// RemoveTrailingCommas removes commas before closing brackets and braces
// while ignoring commas within string literals.
func RemoveTrailingCommas(raw []byte) []byte {
	if len(raw) == 0 {
		return raw
	}

	var buf bytes.Buffer
	buf.Grow(len(raw))

	inString := false
	escaped := false

	for i := 0; i < len(raw); i++ {
		b := raw[i]
		if inString {
			buf.WriteByte(b)
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
			buf.WriteByte(b)
			continue
		}

		if b == ',' {
			j := i + 1
			for j < len(raw) && (raw[j] == ' ' || raw[j] == '\t' || raw[j] == '\n' || raw[j] == '\r' || raw[j] == ',') {
				j++
			}
			if j == len(raw) || raw[j] == '}' || raw[j] == ']' {
				continue
			}
		}

		buf.WriteByte(b)
	}

	return buf.Bytes()
}

// BalanceJSONDelimiters closes unclosed string literals and balances unclosed
// braces and brackets.
func BalanceJSONDelimiters(raw []byte) []byte {
	if len(raw) == 0 {
		return raw
	}

	var stack []byte
	inString := false
	escaped := false

	for i := 0; i < len(raw); i++ {
		b := raw[i]
		if inString {
			if escaped {
				escaped = false
			} else if b == '\\' {
				escaped = true
			} else if b == '"' {
				inString = false
			}
			continue
		}

		switch b {
		case '"':
			inString = true
		case '{', '[':
			stack = append(stack, b)
		case '}':
			if len(stack) > 0 && stack[len(stack)-1] == '{' {
				stack = stack[:len(stack)-1]
			}
		case ']':
			if len(stack) > 0 && stack[len(stack)-1] == '[' {
				stack = stack[:len(stack)-1]
			}
		}
	}

	cleanLen := len(raw)
	if !inString {
		for cleanLen > 0 && (raw[cleanLen-1] == ' ' || raw[cleanLen-1] == '\t' || raw[cleanLen-1] == '\n' || raw[cleanLen-1] == '\r') {
			cleanLen--
		}
		for cleanLen > 0 && raw[cleanLen-1] == ',' {
			cleanLen--
			for cleanLen > 0 && (raw[cleanLen-1] == ' ' || raw[cleanLen-1] == '\t' || raw[cleanLen-1] == '\n' || raw[cleanLen-1] == '\r') {
				cleanLen--
			}
		}
	}

	needsNull := false
	if !inString && cleanLen > 0 && raw[cleanLen-1] == ':' {
		needsNull = true
	}

	var buf bytes.Buffer
	buf.Grow(cleanLen + len(stack) + 8)
	buf.Write(raw[:cleanLen])

	if inString {
		if escaped {
			buf.WriteByte('\\')
		}
		buf.WriteByte('"')
	} else if needsNull {
		buf.WriteString(" null")
	}

	for i := len(stack) - 1; i >= 0; i-- {
		if stack[i] == '{' {
			buf.WriteByte('}')
		} else if stack[i] == '[' {
			buf.WriteByte(']')
		}
	}

	return buf.Bytes()
}

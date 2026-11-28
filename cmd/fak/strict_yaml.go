package main

import (
	"fmt"
	"strings"
)

// duplicate mapping key with its line and column instead of letting the last one
// silently win. This scanner handles the block-mapping subset benchmark configs use
// (indentation-scoped `key: value` lines); it is not a general YAML parser.

// strictYAMLDuplicate describes a rejected duplicate key.
type strictYAMLDuplicate struct {
	Key       string
	Line, Col int
	FirstLine int
}

func (d strictYAMLDuplicate) Error() string {
	return fmt.Sprintf("duplicate key %q at line %d, column %d (first defined at line %d)", d.Key, d.Line, d.Col, d.FirstLine)
}

// strictYAMLCheck returns the first duplicate mapping key in src, or nil.
func strictYAMLCheck(src string) error {
	type scope struct {
		indent int
		seen   map[string]int
	}
	var stack []scope
	for i, raw := range strings.Split(src, "\n") {
		line := strings.TrimRight(raw, "\r")
		trim := strings.TrimLeft(line, " ")
		if trim == "" || strings.HasPrefix(trim, "#") || strings.HasPrefix(trim, "- ") || trim == "-" {
			continue
		}
		colon := strings.Index(trim, ":")
		if colon <= 0 {
			continue
		}
		key := strings.TrimSpace(trim[:colon])
		indent := len(line) - len(trim)
		for len(stack) > 0 && stack[len(stack)-1].indent > indent {
			stack = stack[:len(stack)-1]
		}
		if len(stack) == 0 || stack[len(stack)-1].indent < indent {
			stack = append(stack, scope{indent: indent, seen: map[string]int{}})
		}
		cur := stack[len(stack)-1]
		if first, dup := cur.seen[key]; dup {
			return strictYAMLDuplicate{Key: key, Line: i + 1, Col: indent + 1, FirstLine: first}
		}
		cur.seen[key] = i + 1
	}
	return nil
}

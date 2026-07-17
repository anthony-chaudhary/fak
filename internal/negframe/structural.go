package negframe

import (
	"regexp"
	"strings"
)

// StructuralFinding describes one negation operator over a compound scope.
// ScopeStart and ScopeEnd are byte offsets in the original input; Clauses retain
// source order and Operator names the connective before De Morgan distribution.
type StructuralFinding struct {
	ScopeStart  int      `json:"scope_start"`
	ScopeEnd    int      `json:"scope_end"`
	Scope       string   `json:"scope"`
	Operator    string   `json:"operator"`
	Distributed []string `json:"distributed"`
}

var structuralNotPattern = regexp.MustCompile(`(?i)\bnot\b`)

// DetectStructuralNegation identifies parenthesized not(A and B)/not(A or B)
// candidates without rewriting or changing the lexical classifier. Code fences
// and inline-code spans remain opaque. Findings are deterministic in text order.
func DetectStructuralNegation(text string) []StructuralFinding {
	var findings []StructuralFinding
	inFence := false
	offset := 0
	for _, raw := range strings.SplitAfter(text, "\n") {
		line := strings.TrimSuffix(raw, "\n")
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "```") || strings.HasPrefix(trimmed, "~~~") {
			inFence = !inFence
			offset += len(raw)
			continue
		}
		if !inFence {
			for _, span := range proseSpans(line) {
				findings = append(findings, detectStructuralLine(span.text, offset+span.start)...)
			}
		}
		offset += len(raw)
	}
	return findings
}

func detectStructuralLine(line string, base int) []StructuralFinding {
	var findings []StructuralFinding
	search := 0
	for search < len(line) {
		loc := structuralNotPattern.FindStringIndex(line[search:])
		if loc == nil {
			break
		}
		notEnd := search + loc[1]
		open := skipStructuralSpace(line, notEnd)
		if open >= len(line) || line[open] != '(' {
			search = notEnd
			continue
		}
		close, ok := matchingStructuralParen(line, open)
		if !ok {
			search = notEnd
			continue
		}
		scope := line[open+1 : close]
		op, split, ok := topLevelStructuralOperator(scope)
		if ok {
			left := strings.TrimSpace(scope[:split])
			right := strings.TrimSpace(scope[split+len(op):])
			if left != "" && right != "" {
				findings = append(findings, StructuralFinding{
					ScopeStart:  base + open,
					ScopeEnd:    base + close + 1,
					Scope:       line[open : close+1],
					Operator:    strings.ToLower(op),
					Distributed: []string{"not " + left, "not " + right},
				})
			}
		}
		search = close + 1
	}
	return findings
}

func skipStructuralSpace(s string, at int) int {
	for at < len(s) && (s[at] == ' ' || s[at] == '\t') {
		at++
	}
	return at
}

func matchingStructuralParen(s string, open int) (int, bool) {
	depth := 0
	for i := open; i < len(s); i++ {
		switch s[i] {
		case '(':
			depth++
		case ')':
			depth--
			if depth == 0 {
				return i, true
			}
		}
	}
	return 0, false
}

func topLevelStructuralOperator(scope string) (string, int, bool) {
	depth := 0
	for i := 0; i < len(scope); i++ {
		switch scope[i] {
		case '(':
			depth++
		case ')':
			if depth > 0 {
				depth--
			}
		default:
			if depth != 0 {
				continue
			}
			for _, op := range []string{"and", "or"} {
				if structuralWordAt(scope, i, op) {
					return op, i, true
				}
			}
		}
	}
	return "", 0, false
}

func structuralWordAt(s string, at int, word string) bool {
	if at+len(word) > len(s) || !strings.EqualFold(s[at:at+len(word)], word) {
		return false
	}
	beforeOK := at == 0 || !isStructuralWordByte(s[at-1])
	after := at + len(word)
	afterOK := after == len(s) || !isStructuralWordByte(s[after])
	return beforeOK && afterOK
}

func isStructuralWordByte(b byte) bool {
	return b == '_' || b >= '0' && b <= '9' || b >= 'A' && b <= 'Z' || b >= 'a' && b <= 'z'
}

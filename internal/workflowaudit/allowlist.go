package workflowaudit

import (
	_ "embed"
	"strings"
)

// allowText is the embedded set of intentional non-role branch references. See allow.txt
// for the format and the captured-state note. Embedding it keeps the gate hermetic: the
// live-tree test and the CLI share one source of truth that travels with the binary.
//
//go:embed allow.txt
var allowText string

// Allowlist is the parsed (file -> set of intentional tokens) map. A reference whose
// (file, token) pair is present is classified ClassLegacy rather than failing the gate.
type Allowlist struct {
	byFile map[string]map[string]bool
}

// DefaultAllowlist parses the embedded allow.txt. It is the allowlist the CLI and the
// live-tree regression test both use.
func DefaultAllowlist() Allowlist { return ParseAllowlist(allowText) }

// ParseAllowlist builds an Allowlist from `<file>:<token>` lines. `#` comments and blank
// lines are ignored. A line without a colon is skipped (never panics on malformed input).
func ParseAllowlist(text string) Allowlist {
	a := Allowlist{byFile: map[string]map[string]bool{}}
	for _, raw := range strings.Split(text, "\n") {
		line := strings.TrimSpace(raw)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		i := strings.LastIndex(line, ":")
		if i < 0 {
			continue
		}
		file := strings.TrimSpace(line[:i])
		tok := strings.TrimSpace(line[i+1:])
		if file == "" || tok == "" {
			continue
		}
		if a.byFile[file] == nil {
			a.byFile[file] = map[string]bool{}
		}
		a.byFile[file][tok] = true
	}
	return a
}

// Has reports whether (file, token) is an intentional, reviewed reference.
func (a Allowlist) Has(file, token string) bool {
	set := a.byFile[file]
	return set != nil && set[token]
}

// Len returns the number of allowlisted (file, token) pairs -- used by tests to assert
// the allowlist is non-empty and bounded.
func (a Allowlist) Len() int {
	n := 0
	for _, set := range a.byFile {
		n += len(set)
	}
	return n
}

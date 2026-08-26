// Package shelltoken provides small, dialect-neutral helpers for inspecting
// already-tokenized shell command words and flags.
package shelltoken

import "strings"

// IsAssign reports whether token is a leading shell environment assignment
// whose name is a valid shell identifier.
func IsAssign(token string) bool {
	eq := strings.IndexByte(token, '=')
	if eq <= 0 {
		return false
	}
	for i := 0; i < eq; i++ {
		ch := token[i]
		ok := ch == '_' ||
			(ch >= 'A' && ch <= 'Z') ||
			(ch >= 'a' && ch <= 'z') ||
			(i > 0 && ch >= '0' && ch <= '9')
		if !ok {
			return false
		}
	}
	return true
}

// IsShortCluster reports whether token is a single-dash short-flag cluster.
func IsShortCluster(token string) bool {
	return len(token) >= 2 && token[0] == '-' && token[1] != '-'
}

// ClusterHas reports whether a short-flag cluster contains ch before an
// attached =value suffix.
func ClusterHas(token string, ch byte) bool {
	for i := 1; i < len(token); i++ {
		if token[i] == '=' {
			break
		}
		if token[i] == ch {
			return true
		}
	}
	return false
}

// ProgramBasename normalizes a command word to its lowercase basename and
// strips a trailing .exe suffix.
func ProgramBasename(token string) string {
	base := token
	if i := strings.LastIndexAny(base, `/\`); i >= 0 {
		base = base[i+1:]
	}
	return strings.TrimSuffix(strings.ToLower(base), ".exe")
}

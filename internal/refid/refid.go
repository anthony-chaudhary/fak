// Package refid validates bounded path segments used by fak's Git-ref namespaces.
package refid

import "strings"

// Valid reports whether id is one safe Git-ref path segment.
func Valid(id string) bool {
	if id == "" || len(id) > 200 || strings.HasPrefix(id, "-") || strings.HasPrefix(id, ".") {
		return false
	}
	for _, c := range []byte(id) {
		switch {
		case c >= 'a' && c <= 'z', c >= 'A' && c <= 'Z', c >= '0' && c <= '9':
		case c == '-' || c == '_' || c == '.':
		default:
			return false
		}
	}
	return true
}

// ValidSession reports whether id is valid for the payload beneath a session-
// prefixed ref namespace.
func ValidSession(id string) bool { return Valid(id) && !strings.HasPrefix(id, "session-") }

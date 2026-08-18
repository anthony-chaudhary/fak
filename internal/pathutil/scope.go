package pathutil

import "strings"

// NormalizeScope renders a repository scope path with forward slashes, no
// leading dot segments, and no trailing slash.
func NormalizeScope(path string) string {
	path = strings.TrimSpace(strings.ReplaceAll(path, `\`, "/"))
	for strings.HasPrefix(path, "./") {
		path = path[2:]
	}
	return strings.TrimSuffix(path, "/")
}

// Package markerblock locates and replaces generated regions delimited by text markers.
package markerblock

import (
	"fmt"
	"strings"
)

// Extract returns the marker-delimited block, including both markers.
func Extract(doc, begin, end string) (string, bool) {
	i := strings.Index(doc, begin)
	if i < 0 {
		return "", false
	}
	j := strings.Index(doc[i:], end)
	if j < 0 {
		return "", false
	}
	return doc[i : i+j+len(end)], true
}

// Splice replaces the marker-delimited block without guessing when either marker is absent.
func Splice(doc, begin, end, replacement string) (string, error) {
	i := strings.Index(doc, begin)
	if i < 0 {
		return "", fmt.Errorf("begin marker not found: %s", begin)
	}
	j := strings.Index(doc[i:], end)
	if j < 0 {
		return "", fmt.Errorf("end marker not found after begin marker: %s", end)
	}
	return doc[:i] + replacement + doc[i+j+len(end):], nil
}

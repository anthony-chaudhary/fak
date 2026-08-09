package dojo

import (
	"crypto/sha256"
	"encoding/hex"
	"sort"
	"strconv"
)

// NodeOutcome is the structural result of one node in a run. Duration is
// carried for callers that already measure it, but Signature deliberately
// excludes it: elapsed time makes otherwise identical outcomes look novel.
type NodeOutcome struct {
	ID         string
	Status     string
	DurationMS int64
}

// Signature returns a stable digest of a run's structural outcome. Node order
// does not matter, repeated ID/status pairs remain significant, and duration
// is intentionally excluded. Length prefixes keep arbitrary IDs and statuses
// unambiguous without imposing delimiter restrictions on callers.
func Signature(status string, nodes []NodeOutcome, anyError bool) string {
	parts := make([]string, len(nodes))
	for i, node := range nodes {
		parts[i] = field(node.ID) + field(node.Status)
	}
	sort.Strings(parts)

	h := sha256.New()
	_, _ = h.Write([]byte(field(status)))
	if anyError {
		_, _ = h.Write([]byte{1})
	} else {
		_, _ = h.Write([]byte{0})
	}
	for _, part := range parts {
		_, _ = h.Write([]byte(part))
	}
	return hex.EncodeToString(h.Sum(nil))
}

func field(value string) string {
	return strconv.Itoa(len(value)) + ":" + value
}

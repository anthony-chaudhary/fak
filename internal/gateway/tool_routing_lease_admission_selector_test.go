package gateway

import (
	"encoding/json"
	"testing"
)

// fak-test:runtime fast est=1s
func TestAdjudicateHasWriteLikeArgsSelectorKeysAreNotWriteEvidence(t *testing.T) {
	// Selector keys ("mode", "permissions") describe how an operation runs, not
	// what it writes. An unknown tool carrying only selector keys must stay
	// read-classified: dos_arbitrate(mode=shared) and similar MCP verbs were
	// refused DEFAULT_DENY/unknown_write_footprint on the live Claude Code
	// surface for exactly this reason (2026-11-05, ap-1aa439399840).
	for _, key := range []string{"mode", "permissions"} {
		raw, _ := json.Marshal(map[string]any{key: "shared"})
		_, _, mutating, known := adjudicateRawWriteTargets("mcp__fak-mcp__dos_arbitrate", string(raw), true)
		if mutating || !known {
			t.Errorf("selector key %q: mutating=%t known=%t, want read-classified", key, mutating, known)
		}
	}
	// Content-shaped keys must stay conservative: an unknown tool carrying
	// content still classifies as an unknown-footprint mutation.
	for _, key := range []string{"content", "data", "text", "patch"} {
		raw, _ := json.Marshal(map[string]any{key: "payload"})
		_, _, mutating, known := adjudicateRawWriteTargets("mcp__fak-mcp__dos_arbitrate", string(raw), true)
		if !mutating || known {
			t.Errorf("content key %q: mutating=%t known=%t, want unknown-footprint mutation", key, mutating, known)
		}
	}
}

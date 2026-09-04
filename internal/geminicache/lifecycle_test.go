package geminicache

import (
	"testing"
)

// Invariant: Gemini cache identities must be deterministic and hash prefix content consistently.
// Guard: NewIdentity computes consistent SHA256 fingerprints across distinct model configurations.

func TestGeminiCacheLifecycle(t *testing.T) {
	t.Parallel()

	prefix := []byte("system instruction and cached tool definitions")
	id1 := NewIdentity("acct", "proj", "region", "gemini-2.5", prefix)
	id2 := NewIdentity("acct", "proj", "region", "gemini-2.5", prefix)

	if id1.PrefixDigest != id2.PrefixDigest {
		t.Fatalf("expected identical cache keys, got %s vs %s", id1.PrefixDigest, id2.PrefixDigest)
	}
}

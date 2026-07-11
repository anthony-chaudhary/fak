package leaseref

import (
	"strings"
	"testing"
)

// TestMintHolderExactlyOneSeparator pins the grammar-safety invariant behind
// holderSep's contract ("a conventional holder contains EXACTLY one '/'"):
// sanitizeHolderSegment maps every out-of-alphabet byte — the path separator
// included — to '-', so a component that itself contains a '/' can never inject
// a SECOND separator and split the holder into three segments (which ParseHolder
// would reject as free-form, silently discarding the node identity). The
// existing mint tests never feed a slash-bearing component, so this is unpinned.
func TestMintHolderExactlyOneSeparator(t *testing.T) {
	cases := [][2]string{
		{"ns/sub", "sess/2"},          // a slash in BOTH components
		{"a/b/c", "s"},                // several slashes in the node
		{"node", "sess/with/slashes"}, // slashes only in the session
		{"desktop", "sess-abc123"},    // the already-clean case still holds it
	}
	for _, c := range cases {
		h := MintHolder(c[0], c[1])
		if n := strings.Count(h, holderSep); n != 1 {
			t.Fatalf("MintHolder(%q,%q)=%q has %d separators, want exactly 1", c[0], c[1], h, n)
		}
		// Exactly one separator => ParseHolder splits into two valid segments =>
		// the minted holder round-trips Structured with Raw preserved verbatim.
		id := ParseHolder(h)
		if !id.Structured() || id.Session == "" || id.Raw != h {
			t.Fatalf("MintHolder(%q,%q)=%q did not round-trip Structured: %+v", c[0], c[1], h, id)
		}
	}
}

// TestMintHolderAsymmetricDegradation pins the non-obvious contract that the two
// components degrade INDEPENDENTLY and that Structured() keys ONLY on the node:
// a valid node with an empty session stays Structured (session -> "unknown"),
// but an empty node with a valid session does NOT (node -> NodeUnknown) even
// though a real session survives on the record. Existing tests cover only the
// both-empty case, so neither single-empty branch is pinned.
func TestMintHolderAsymmetricDegradation(t *testing.T) {
	// Valid node, empty session: Structured, session degraded to the placeholder.
	h := MintHolder("desktop", "")
	if h != "desktop/unknown" {
		t.Fatalf("MintHolder(desktop,\"\")=%q, want desktop/unknown", h)
	}
	if id := ParseHolder(h); !id.Structured() || id.Node != "desktop" || id.Session != "unknown" {
		t.Fatalf("valid node + empty session must stay Structured, got %+v", id)
	}
	// Empty node, valid session: NOT Structured (the node is the placeholder),
	// yet the session component is still carried through.
	h = MintHolder("", "sess-9")
	if h != NodeUnknown+"/sess-9" {
		t.Fatalf("MintHolder(\"\",sess-9)=%q, want %s/sess-9", h, NodeUnknown)
	}
	if id := ParseHolder(h); id.Structured() || id.Node != NodeUnknown || id.Session != "sess-9" {
		t.Fatalf("empty node must classify unstructured yet keep the session, got %+v", id)
	}
}

// TestMintHolderDegradesAllInvalidAndCaps pins two sanitize robustness edges the
// existing tests skip: a component made entirely of out-of-alphabet bytes (or
// only leading '-'/'.' runs) sanitizes to "" and degrades to the honest
// placeholder rather than to a bare separator run, and an over-long component is
// capped to validID's 200-byte bound so it still parses back Structured instead
// of tripping ParseHolder's length guard.
func TestMintHolderDegradesAllInvalidAndCaps(t *testing.T) {
	if h := MintHolder("!!!", "s"); h != NodeUnknown+"/s" {
		t.Fatalf("all-invalid node: MintHolder=%q, want %s/s", h, NodeUnknown)
	}
	if h := MintHolder("...", "s"); h != NodeUnknown+"/s" {
		t.Fatalf("leading-dot-only node: MintHolder=%q, want %s/s", h, NodeUnknown)
	}
	// An over-long node caps to <=200 bytes and remains a single valid segment
	// that round-trips Structured.
	long := strings.Repeat("a", 300)
	id := ParseHolder(MintHolder(long, "s"))
	if !id.Structured() || len(id.Node) > 200 || id.Session != "s" {
		t.Fatalf("over-long node must cap to <=200 and stay Structured, got node len=%d structured=%v", len(id.Node), id.Structured())
	}
}

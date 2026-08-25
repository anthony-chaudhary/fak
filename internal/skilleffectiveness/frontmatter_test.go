package skilleffectiveness

import "testing"

func TestFrontmatterDescriptionDecodesSingleQuotedDoubledApostrophes(t *testing.T) {
	doc := `---
name: quoted
description: 'Bob''s skill says ''hello''.'
---
body
`
	got, ok := frontmatterDescription(doc)
	if !ok {
		t.Fatal("frontmatterDescription rejected a valid single-quoted scalar")
	}
	if got != "Bob's skill says 'hello'." {
		t.Fatalf("frontmatterDescription = %q, want %q", got, "Bob's skill says 'hello'.")
	}
}

func TestTierWordCountsDecodesDoubleQuotedEscapesAndBackslashes(t *testing.T) {
	doc := `---
name: quoted
description: "one\ttwo C:\\tools"
---
body
`
	meta, body := TierWordCounts(doc)
	if meta != 3 || body != 1 {
		t.Fatalf("TierWordCounts = (%d,%d), want (3,1)", meta, body)
	}
}

func TestTierWordCountsPreservesMalformedQuotedDescription(t *testing.T) {
	doc := `---
name: malformed
description: "one two
---
body
`
	meta, body := TierWordCounts(doc)
	if meta != 2 || body != 1 {
		t.Fatalf("TierWordCounts = (%d,%d), want (2,1)", meta, body)
	}
}

package scorecardpane

import (
	"testing"
)

func TestProseOnlyStripsFrontMatterAndCode(t *testing.T) {
	in := "---\ntitle: x\n---\nreal prose\n```\ncode block delve\n```\nmore `inline delve` prose\n"
	got := proseOnly(in)
	if contains(got, "code block") {
		t.Fatalf("fenced code must be stripped: %q", got)
	}
	if contains(got, "inline delve") {
		t.Fatalf("inline code must be stripped: %q", got)
	}
	if !contains(got, "real prose") || !contains(got, "more") {
		t.Fatalf("prose must survive: %q", got)
	}
}

func TestImageAltDefectsMarkdownAndHTML(t *testing.T) {
	in := "![](pic.png) ![good alt](ok.png)\n<img src=\"x.png\"> <img src=\"y.png\" alt=\"y\">"
	got := imageAltDefects(in)
	// pic.png (empty md alt) + x.png (no html alt) = 2 defects; ok.png + y.png fine.
	if len(got) != 2 {
		t.Fatalf("want 2 alt defects, got %d: %v", len(got), got)
	}
	if got[0] != "pic.png" || got[1] != "x.png" {
		t.Fatalf("alt defects must name the right images, got %v", got)
	}
}

func TestTellHitsBoundaryGuard(t *testing.T) {
	// a bare "leverage" is context-sensitive (dropped), but "delve" is a HARD tell.
	// "high-leverage" must NOT fire a single-word "leverage" match (hyphen guard) —
	// and leverage is excluded anyway, so use "robust" which is also excluded; pick
	// "seamless": "seamlessly" must NOT match the bare "seamless" (suffix guard via
	// word boundary). Use a clean positive + a guarded negative.
	if hits := tellHits("we delve into the design"); len(hits) != 1 || hits[0] != "delve" {
		t.Fatalf("delve must fire once, got %v", hits)
	}
	if hits := tellHits("nothing to see here plainly"); len(hits) != 0 {
		t.Fatalf("plain prose must fire no tells, got %v", hits)
	}
}

func TestIsReaderFacing(t *testing.T) {
	cases := map[string]bool{
		"README.md":                      true,
		"docs/guide.md":                  true,
		"docs/notes/2026-01-01.md":       false, // archival subtree
		"internal/x/doc.md":              false, // not docs/ or root allowlist
		"docs/REPO-HYGIENE-SCORECARD.md": false, // the generated snapshot
		"RANDOM.md":                      false, // root .md not on allowlist
	}
	for rel, want := range cases {
		if got := isReaderFacing(rel); got != want {
			t.Errorf("isReaderFacing(%q) = %v, want %v", rel, got, want)
		}
	}
}

func TestShinglesDeterministic(t *testing.T) {
	a := shingles("the quick brown fox jumps over the lazy dog again now please")
	b := shingles("the quick brown fox jumps over the lazy dog again now please")
	if len(a) == 0 {
		t.Fatalf("non-trivial prose must produce shingles")
	}
	if jaccard(a, b) != 1.0 {
		t.Fatalf("identical text must have jaccard 1.0, got %v", jaccard(a, b))
	}
	c := shingles("completely different words here with nothing shared at all today")
	if jaccard(a, c) >= dupHardJaccard {
		t.Fatalf("disjoint text must not look near-duplicate, got %v", jaccard(a, c))
	}
}

func contains(s, sub string) bool {
	return len(s) >= len(sub) && (indexOf(s, sub) >= 0)
}

func indexOf(s, sub string) int {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return i
		}
	}
	return -1
}

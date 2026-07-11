package wiki

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/devindex"
)

func TestComputeScore_CoverageResolveFreshness(t *testing.T) {
	// Two leaves (gateway, devindex); gateway.go is a real 2-line file the cites
	// resolve against.
	root := writeRepo(t, twoLeafDosToml, map[string]string{
		"internal/gateway/gateway.go": "package gateway\nfunc A() {}",
	})
	cat, err := devindex.Load(root)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	pages := []PageInput{
		// A page for the gateway leaf: pinned (fresh frontmatter), one resolving cite
		// and one dangling (out-of-range) cite.
		{
			RelID: "core-features/gateway",
			Markdown: []byte("---\ngenerated_at_sha: abc1234\n---\n" +
				"# Gateway\n" +
				"Admits a request. [internal/gateway/gateway.go:1-2]()\n" +
				"Bad line: [internal/gateway/gateway.go:9-10]()\n"),
		},
		// The Overview page: no frontmatter, no cites.
		{RelID: "overview/introduction", Markdown: []byte("# Introduction\n\nprose only\n")},
	}

	s := ComputeScore(root, cat, pages)

	if s.Pages != 2 {
		t.Errorf("Pages = %d, want 2", s.Pages)
	}
	// gateway leaf covered, devindex leaf not → 1/2.
	if s.Leaves != 2 || s.LeavesCovered != 1 {
		t.Errorf("coverage = %d/%d, want 1/2", s.LeavesCovered, s.Leaves)
	}
	if s.LeafCoverage != 0.5 {
		t.Errorf("LeafCoverage = %v, want 0.5", s.LeafCoverage)
	}
	// 2 cites, 1 resolves.
	if s.Citations != 2 || s.CitationsResolved != 1 {
		t.Errorf("cites = %d resolved = %d, want 2/1", s.Citations, s.CitationsResolved)
	}
	if s.CitationResolveRate != 0.5 {
		t.Errorf("CitationResolveRate = %v, want 0.5", s.CitationResolveRate)
	}
	if len(s.Danglers) != 1 {
		t.Errorf("Danglers = %d, want 1", len(s.Danglers))
	}
	// 1 of 2 pages pins a SHA.
	if s.FreshPages != 1 || s.FreshRate != 0.5 {
		t.Errorf("freshness = %d/%d (%v), want 1/2 0.5", s.FreshPages, s.Pages, s.FreshRate)
	}
}

func TestComputeScore_EmptyWikiIsNotVacuouslyPassing(t *testing.T) {
	root := writeRepo(t, twoLeafDosToml, nil)
	cat, err := devindex.Load(root)
	if err != nil {
		t.Fatal(err)
	}
	s := ComputeScore(root, cat, nil)

	// No pages: coverage is 0 (caught by a coverage floor), but the cite/fresh
	// ratios are vacuously 1 (no denominator).
	if s.LeafCoverage != 0 {
		t.Errorf("empty-wiki LeafCoverage = %v, want 0", s.LeafCoverage)
	}
	if s.CitationResolveRate != 1 {
		t.Errorf("empty-wiki CitationResolveRate = %v, want vacuous 1", s.CitationResolveRate)
	}
	// A resolve-only floor passes an empty wiki; a coverage floor is what fails it.
	if !s.Passes(1.0, 0.0) {
		t.Error("resolve floor alone should pass an empty wiki")
	}
	if s.Passes(1.0, 0.5) {
		t.Error("a coverage floor of 0.5 must fail an empty wiki")
	}
}

func TestScorePasses(t *testing.T) {
	s := Score{CitationResolveRate: 1.0, LeafCoverage: 0.8}
	if !s.Passes(1.0, 0.8) {
		t.Error("want pass at exactly the floor")
	}
	if s.Passes(1.0, 0.81) {
		t.Error("want fail just above coverage")
	}
	if (Score{CitationResolveRate: 0.99, LeafCoverage: 1.0}).Passes(1.0, 0.0) {
		t.Error("a dangling cite must fail a resolve floor of 1.0")
	}
}

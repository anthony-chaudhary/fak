package wiki

import (
	"reflect"
	"strings"
	"testing"
)

func TestParseFrontmatter_ListAndInline(t *testing.T) {
	// Block-list form.
	block := strings.Join([]string{
		"---",
		"generated_at_sha: 1a2b3c4d",
		"cited_files:",
		"  - internal/gateway/gateway.go",
		"  - internal/gateway/admit.go",
		"---",
		"# Gateway",
		"body",
	}, "\n")
	m := ParseFrontmatter([]byte(block))
	if !m.HasFrontmatter {
		t.Fatal("HasFrontmatter = false, want true")
	}
	if m.GeneratedAtSHA != "1a2b3c4d" {
		t.Errorf("sha = %q", m.GeneratedAtSHA)
	}
	if got := strings.Join(m.CitedFiles, ","); got != "internal/gateway/gateway.go,internal/gateway/admit.go" {
		t.Errorf("cited_files = %q", got)
	}

	// Inline-array form, quoted sha.
	inline := "---\ngenerated_at_sha: \"deadbeef\"\ncited_files: [a/b.go, c/d.go]\n---\n"
	m2 := ParseFrontmatter([]byte(inline))
	if m2.GeneratedAtSHA != "deadbeef" {
		t.Errorf("inline sha = %q", m2.GeneratedAtSHA)
	}
	if got := strings.Join(m2.CitedFiles, ","); got != "a/b.go,c/d.go" {
		t.Errorf("inline cited_files = %q", got)
	}
}

func TestParseFrontmatter_NoFence(t *testing.T) {
	m := ParseFrontmatter([]byte("# No frontmatter\n\nprose\n"))
	if m.HasFrontmatter || m.GeneratedAtSHA != "" || m.CitedFiles != nil {
		t.Errorf("want zero meta on a fence-less page, got %+v", m)
	}
}

func TestDriftStaleWikiPage(t *testing.T) {
	cited := PageMeta{GeneratedAtSHA: "abc123", CitedFiles: []string{"internal/gateway/gateway.go", "internal/gateway/admit.go"}}

	// A cited file moved → stale, with just that file in Touched.
	sp, stale := DriftStaleWikiPage("p.md", cited, []string{"internal/gateway/admit.go", "docs/unrelated.md"})
	if !stale || sp.Reason != ReasonCitedCodeMoved {
		t.Fatalf("want stale/cited-code-moved, got stale=%v %+v", stale, sp)
	}
	if !reflect.DeepEqual(sp.Touched, []string{"internal/gateway/admit.go"}) {
		t.Errorf("Touched = %v, want just the moved cited file", sp.Touched)
	}

	// No cited file in the changed set → fresh.
	if _, stale := DriftStaleWikiPage("p.md", cited, []string{"README.md"}); stale {
		t.Error("want fresh when no cited file changed")
	}

	// No SHA → stale by construction (unwitnessable).
	sp, stale = DriftStaleWikiPage("p.md", PageMeta{CitedFiles: []string{"x.go"}}, nil)
	if !stale || sp.Reason != ReasonNoSHA {
		t.Errorf("want stale/no-sha, got stale=%v %+v", stale, sp)
	}

	// SHA present, no cited files → vacuously fresh.
	if _, stale := DriftStaleWikiPage("p.md", PageMeta{GeneratedAtSHA: "abc"}, []string{"anything.go"}); stale {
		t.Error("want fresh when the page cites nothing")
	}
}

func TestDriftStaleWikiPage_PathNormalization(t *testing.T) {
	// A backslash-spelled cite and a leading-./ changed path still match.
	meta := PageMeta{GeneratedAtSHA: "abc", CitedFiles: []string{"internal\\gateway\\gateway.go"}}
	_, stale := DriftStaleWikiPage("p.md", meta, []string{"./internal/gateway/gateway.go"})
	if !stale {
		t.Error("want stale: normalized paths should compare equal across separators/prefix")
	}
}

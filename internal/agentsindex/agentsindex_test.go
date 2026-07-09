package agentsindex

import (
	"strings"
	"testing"
)

// A synthetic fixture exercising every parser rule: a level-1 title + preamble (not a
// section), a '##' with a nested '###', a fenced block whose line-start '#' must NOT be
// read as a heading, a ':'-qualified title, and a duplicate heading forcing slug dedupe.
const fixture = "# Doc Title\n" +
	"\n" +
	"preamble line, not in any section.\n" +
	"\n" +
	"## Alpha (parenthetical)\n" +
	"\n" +
	"alpha body\n" +
	"\n" +
	"### Alpha child\n" +
	"\n" +
	"child body\n" +
	"\n" +
	"## Beta: the second\n" +
	"\n" +
	"```bash\n" +
	"## not a heading, inside a fence\n" +
	"echo hi\n" +
	"```\n" +
	"beta tail\n" +
	"\n" +
	"## Alpha (again)\n" +
	"\n" +
	"dup body\n"

func parseFixture() *Doc { return Parse([]byte(fixture)) }

func TestParseSectionCountAndSlugs(t *testing.T) {
	d := parseFixture()
	// level>=2 headings only: Alpha, Alpha child, Beta, Alpha(again) = 4. The level-1
	// title and the fenced "## not a heading" line must NOT appear.
	if len(d.Sections) != 4 {
		var got []string
		for _, s := range d.Sections {
			got = append(got, s.Slug)
		}
		t.Fatalf("section count=%d want 4; slugs=%v", len(d.Sections), got)
	}
	want := []string{"alpha", "alpha-child", "beta", "alpha-2"}
	for i, w := range want {
		if d.Sections[i].Slug != w {
			t.Fatalf("section[%d].Slug=%q want %q", i, d.Sections[i].Slug, w)
		}
	}
}

func TestFenceAwarenessNoPhantomSection(t *testing.T) {
	d := parseFixture()
	if _, ok := d.SectionBySlug("not-a-heading"); ok {
		t.Fatalf("a '#' inside a fenced block must not become a section")
	}
}

func TestParentSpanIncludesNestedChild(t *testing.T) {
	d := parseFixture()
	alpha, ok := d.SectionBySlug("alpha")
	if !ok {
		t.Fatal("missing alpha section")
	}
	// the '##' Alpha span must include its nested '### Alpha child'.
	if !strings.Contains(alpha.Raw, "### Alpha child") || !strings.Contains(alpha.Raw, "child body") {
		t.Fatalf("parent '##' span must include the nested '###':\n%q", alpha.Raw)
	}
	// and it must stop before the next '##'.
	if strings.Contains(alpha.Raw, "Beta") {
		t.Fatalf("parent span leaked into the next '##':\n%q", alpha.Raw)
	}
}

func TestChildSpanIsOwnSpanOnly(t *testing.T) {
	d := parseFixture()
	child, ok := d.SectionBySlug("alpha-child")
	if !ok {
		t.Fatal("missing alpha-child section")
	}
	if !strings.HasPrefix(child.Raw, "### Alpha child") {
		t.Fatalf("child span must start at its own heading:\n%q", child.Raw)
	}
	if strings.Contains(child.Raw, "## Beta") || strings.Contains(child.Raw, "alpha body") {
		t.Fatalf("child '###' span must be its own span only:\n%q", child.Raw)
	}
}

func TestByteExactSpansReconstructWithinFile(t *testing.T) {
	d := parseFixture()
	// every section's Raw must be a verbatim substring of the source at its heading.
	for _, s := range d.Sections {
		if !strings.Contains(fixture, s.Raw) {
			t.Fatalf("section %q Raw is not a verbatim slice of the source", s.Slug)
		}
		if !strings.HasPrefix(s.Raw, strings.Repeat("#", s.Level)+" ") {
			t.Fatalf("section %q Raw must start at its own heading, got %q", s.Slug, firstLine(s.Raw))
		}
	}
}

func TestSearchDeterministicAndTitleWeighted(t *testing.T) {
	d := parseFixture()
	// "alpha" appears in two headings + bodies; beta only in one. Heading matches weigh
	// more, so an alpha-heading section outranks beta.
	got := d.Search("alpha")
	if len(got) == 0 {
		t.Fatal("expected alpha matches")
	}
	if got[0].Slug != "alpha" {
		t.Fatalf("top alpha result=%q want alpha (heading-weighted)", got[0].Slug)
	}
	// determinism: same query, same order.
	again := d.Search("alpha")
	for i := range got {
		if got[i].Slug != again[i].Slug {
			t.Fatalf("search not deterministic at %d: %q vs %q", i, got[i].Slug, again[i].Slug)
		}
	}
	if d.Search("") != nil {
		t.Fatalf("empty query must return nil")
	}
}

func TestSectionBySlugMissIsFailSafe(t *testing.T) {
	d := parseFixture()
	if s, ok := d.SectionBySlug("does-not-exist"); ok || s.Slug != "" {
		t.Fatalf("miss must be (zero,false), got (%+v,%v)", s, ok)
	}
}

func TestEstTokensArithmetic(t *testing.T) {
	if got := EstTokensOf("abcd"); got != 1 { // 4 bytes -> 1
		t.Fatalf("EstTokensOf(4 bytes)=%d want 1", got)
	}
	if got := EstTokensOf("abcde"); got != 2 { // 5 bytes -> ceil(5/4)=2
		t.Fatalf("EstTokensOf(5 bytes)=%d want 2", got)
	}
	d := parseFixture()
	if d.EstTokens() != (len(fixture)+3)/4 {
		t.Fatalf("doc EstTokens mismatch")
	}
}

func TestCRLFSpansStayByteExact(t *testing.T) {
	crlf := strings.ReplaceAll(fixture, "\n", "\r\n")
	d := Parse([]byte(crlf))
	if len(d.Sections) != 4 {
		t.Fatalf("CRLF parse section count=%d want 4", len(d.Sections))
	}
	// Raw must preserve CRLF verbatim (byte-identical to the source region).
	for _, s := range d.Sections {
		if !strings.Contains(crlf, s.Raw) {
			t.Fatalf("CRLF section %q Raw not a verbatim slice", s.Slug)
		}
	}
}

func firstLine(s string) string {
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		return s[:i]
	}
	return s
}

package docreach

import "testing"

func TestCensusNamedRulesAndBrokenLinks(t *testing.T) {
	r := Census("abc", []Blob{{"README.md", "[guide](docs/guide.md) `docs/note.md` [bad](missing.md)"}, {"docs/guide.md", "[note](note.md)"}, {"docs/note.md", ""}, {"docs/orphan.md", ""}})
	if r.Commit != "abc" || r.Documents != 4 {
		t.Fatalf("header=%+v", r)
	}
	for _, c := range r.Rules {
		if c.Denominator != 4 {
			t.Fatalf("bare/mismatched denominator: %+v", c)
		}
	}
	if r.Rules[0].Rule != "R-LINK" || r.Rules[0].Numerator != 2 {
		t.Fatalf("link=%+v", r.Rules[0])
	}
	if r.Rules[1].Rule != "R-MENTION" || r.Rules[1].Numerator != 2 {
		t.Fatalf("mention=%+v", r.Rules[1])
	}
	if len(r.BrokenLinks) != 1 || r.BrokenLinks[0].Target != "missing.md" {
		t.Fatalf("broken=%+v", r.BrokenLinks)
	}
	if got := r.Rules[2].Unreached; len(got) != 2 || got[0] != "README.md" || got[1] != "docs/orphan.md" {
		t.Fatalf("union unreached=%v", got)
	}
}
func TestResolverRequiresUniqueBasename(t *testing.T) {
	r := Census("x", []Blob{{"README.md", "[x](same.md)"}, {"a/same.md", ""}, {"b/same.md", ""}})
	if len(r.BrokenLinks) != 1 {
		t.Fatalf("ambiguous basename must be broken: %+v", r)
	}
}

package devindex

import (
	"os"
	"path/filepath"
	"testing"
)

// writeWorkRepo lays down a synthetic repo with a dos.toml (so Load succeeds) and a
// .github/issue-views.json with a default, a page limit, and three named views.
func writeWorkRepo(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	mustWrite(t, root, "dos.toml", "[lanes.trees]\ncmd = [\"cmd/**\"]\n")
	if err := os.MkdirAll(filepath.Join(root, ".github"), 0o755); err != nil {
		t.Fatal(err)
	}
	viewsJSON := `{
  "version": 1,
  "default": "ready-leaves",
  "limit": 300,
  "views": [
    {"slug": "ready-leaves", "title": "Ready leaves", "query": "is:open no:assignee", "note": "the default what-to-work-on surface"},
    {"slug": "p0-p1", "title": "P0/P1 leaves", "query": "is:open label:priority/P0,priority/P1", "note": "prioritized, oldest first"},
    {"slug": "epics", "title": "Epics", "query": "is:open label:epic", "note": "decompose, do not dispatch"}
  ]
}`
	mustWrite(t, filepath.Join(root, ".github"), "issue-views.json", viewsJSON)
	return root
}

func TestIssueViewsLoadAndDefault(t *testing.T) {
	c, err := Load(writeWorkRepo(t))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	v, err := c.IssueViews()
	if err != nil {
		t.Fatalf("IssueViews: %v", err)
	}
	if len(v.Views) != 3 {
		t.Fatalf("got %d views, want 3", len(v.Views))
	}
	if v.Default != "ready-leaves" || v.PageLimit() != 300 {
		t.Errorf("default=%q limit=%d, want ready-leaves/300", v.Default, v.PageLimit())
	}
	def, ok := v.DefaultView()
	if !ok || def.Slug != "ready-leaves" || def.Query == "" {
		t.Errorf("DefaultView = %+v ok=%v, want the ready-leaves view with a query", def, ok)
	}
	// File order is preserved (default first by convention) so an empty search reads
	// the surface top-to-bottom.
	if got := v.SearchViews(""); len(got) != 3 || got[0].Slug != "ready-leaves" {
		t.Errorf("empty SearchViews = %+v, want 3 views, ready-leaves first", got)
	}
}

func TestIssueViewsSearchAndPageLimitDefault(t *testing.T) {
	c, err := Load(writeWorkRepo(t))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	v, _ := c.IssueViews()

	// An exact slug match dominates a substring/title/note hit.
	if hits := v.SearchViews("p0-p1"); len(hits) == 0 || hits[0].Slug != "p0-p1" {
		t.Errorf("SearchViews(p0-p1) top = %+v, want p0-p1", hits)
	}
	// A note-word query still finds the right view.
	if hits := v.SearchViews("decompose"); len(hits) != 1 || hits[0].Slug != "epics" {
		t.Errorf("SearchViews(decompose) = %+v, want [epics]", hits)
	}
	if hits := v.SearchViews("zzz-nomatch"); len(hits) != 0 {
		t.Errorf("SearchViews(no-match) = %+v, want none", hits)
	}

	// A file with no declared limit falls back to a safe page size, never gh's
	// silently-truncating default of 30.
	none := IssueViews{}
	if none.PageLimit() != 100 {
		t.Errorf("PageLimit() with no declared limit = %d, want 100", none.PageLimit())
	}
}

func TestIssueViewsMissingFileErrors(t *testing.T) {
	// A repo with no .github/issue-views.json is a real "declares no views" answer,
	// surfaced as an error (not a silent empty), so the CLI can tell the agent.
	root := t.TempDir()
	mustWrite(t, root, "dos.toml", "[lanes.trees]\ncmd = [\"cmd/**\"]\n")
	c, err := Load(root)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if _, err := c.IssueViews(); err == nil {
		t.Error("IssueViews on a repo with no issue-views.json should error")
	}
}

package disambiguation

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRenderPublicDocsDeterministicAndQueryable(t *testing.T) {
	first, err := RenderPublicDocs()
	if err != nil {
		t.Fatal(err)
	}
	second, err := RenderPublicDocs()
	if err != nil {
		t.Fatal(err)
	}
	if len(first) != 2 || len(second) != 2 {
		t.Fatalf("pages = %d, want 2", len(first))
	}
	for i := range first {
		if first[i].Path != second[i].Path || !bytes.Equal(first[i].Content, second[i].Content) {
			t.Fatalf("render %d is not byte-identical", i)
		}
		if !strings.Contains(string(first[i].Content), "fak disambiguation query") {
			t.Fatalf("%s has no queryable identity", first[i].Path)
		}
	}
}

func TestRenderDocsMatchesTrackedPages(t *testing.T) {
	pages, err := RenderPublicDocs()
	if err != nil {
		t.Fatal(err)
	}
	for _, page := range pages {
		path := filepath.Join("..", "..", "docs", "generated", "disambiguation", filepath.FromSlash(page.Path))
		tracked, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("read %s: %v", path, err)
		}
		if !bytes.Equal(tracked, page.Content) {
			t.Errorf("tracked page %s is stale; run fak disambiguation docs", path)
		}
	}
}

func TestRenderDocsLinksEveryCanonicalRow(t *testing.T) {
	pages, err := RenderDocs(publicEntries)
	if err != nil {
		t.Fatal(err)
	}
	canonical := string(pages[0].Content)
	contrast := string(pages[1].Content)
	for _, entry := range publicEntries {
		anchor := entryAnchor(entry)
		if !strings.Contains(canonical, "](#"+anchor+")") || !strings.Contains(canonical, `<a id="`+anchor+`"></a>`) {
			t.Errorf("canonical identity %q (%s) is not linked", entry.Identity.CanonicalTerm, anchor)
		}
		if len(entry.Contrasts) > 0 && !strings.Contains(contrast, "canonical-terms.md#"+anchor) {
			t.Errorf("contrast identity %q (%s) is not linked", entry.Identity.CanonicalTerm, anchor)
		}
	}
}

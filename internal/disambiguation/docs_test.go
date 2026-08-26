package disambiguation

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func pageMap(t *testing.T, pages []DocPage) map[string][]byte {
	t.Helper()
	out := make(map[string][]byte, len(pages))
	for _, page := range pages {
		if _, exists := out[page.Path]; exists {
			t.Fatalf("duplicate page %s", page.Path)
		}
		if filepath.IsAbs(page.Path) || strings.Contains(filepath.ToSlash(page.Path), "../") {
			t.Fatalf("unsafe page %s", page.Path)
		}
		out[page.Path] = page.Content
	}
	return out
}

func TestRenderPublicDocsDeterministicCompleteMap(t *testing.T) {
	first, err := RenderPublicDocs()
	if err != nil {
		t.Fatal(err)
	}
	second, err := RenderPublicDocs()
	if err != nil {
		t.Fatal(err)
	}
	if len(first) != len(publicEntries)+3 {
		t.Fatalf("pages=%d want=%d", len(first), len(publicEntries)+3)
	}
	a, b := pageMap(t, first), pageMap(t, second)
	if len(a) != len(b) {
		t.Fatalf("map sizes differ")
	}
	for name, content := range a {
		if !bytes.Equal(content, b[name]) {
			t.Fatalf("page %s differs", name)
		}
	}
}

func TestRenderDocsIndexLinksEveryIdentityPage(t *testing.T) {
	pages, err := RenderPublicDocs()
	if err != nil {
		t.Fatal(err)
	}
	all := pageMap(t, pages)
	index := string(all["INDEX.md"])
	for _, entry := range publicEntries {
		name := filepath.ToSlash(filepath.Join("identities", entryAnchor(entry)+".md"))
		if _, ok := all[name]; !ok {
			t.Errorf("missing %s", name)
		}
		if !strings.Contains(index, "]("+name+")") {
			t.Errorf("index missing link %s", name)
		}
		body := string(all[name])
		if !strings.Contains(body, queryCommand(entry)) || !strings.Contains(body, "## Do not conflate with") {
			t.Errorf("incomplete page %s", name)
		}
	}
}

func TestRenderDocsMatchesTrackedExactSet(t *testing.T) {
	pages, err := RenderPublicDocs()
	if err != nil {
		t.Fatal(err)
	}
	expected := pageMap(t, pages)
	root := filepath.Join("..", "..", "docs", "generated", "disambiguation")
	actual := map[string][]byte{}
	err = filepath.WalkDir(root, func(name string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		rel, err := filepath.Rel(root, name)
		if err != nil {
			return err
		}
		content, err := os.ReadFile(name)
		if err != nil {
			return err
		}
		actual[filepath.ToSlash(rel)] = content
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(actual) != len(expected) {
		t.Fatalf("tracked files=%d want=%d; run fak disambiguation docs", len(actual), len(expected))
	}
	for name, content := range expected {
		if !bytes.Equal(content, actual[name]) {
			t.Errorf("tracked page %s stale or missing", name)
		}
	}
}

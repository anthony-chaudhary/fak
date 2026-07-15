package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/conceptcatalog"
)

func conceptCLIFixture(t *testing.T) (conceptcatalog.Catalog, string) {
	t.Helper()
	root := t.TempDir()
	data := filepath.Join(root, conceptcatalog.DataRel)
	os.MkdirAll(data, 0755)
	meta := conceptcatalog.Metadata{Families: []conceptcatalog.Family{{ID: "cache", Name: "Cache", Roots: []string{"cache"}}}}
	b, _ := json.Marshal(meta)
	os.WriteFile(filepath.Join(data, "_meta.json"), b, 0600)
	os.MkdirAll(filepath.Join(root, "docs"), 0755)
	os.WriteFile(filepath.Join(root, "docs", "glossary.md"), []byte("# Glossary\n"), 0600)
	os.WriteFile(filepath.Join(root, "prod.go"), []byte("package p\nconst CacheC=1\n"), 0600)
	return conceptcatalog.Catalog{Meta: meta, Rows: []conceptcatalog.Row{{ID: "cache-a", Canonical: "Cache A", Family: "cache", Grounding: "CacheA"}}, Dir: data}, root
}
func TestConceptPositionDryRunJSONShowsFilesCountsAndDoesNotWrite(t *testing.T) {
	c, root := conceptCLIFixture(t)
	var out, errb bytes.Buffer
	code := runConceptPosition(&out, &errb, c, []string{"--id", "cache-c", "--canonical", "Cache C", "--family", "cache", "--definition", "third cache", "--distinction", "not cache a", "--grounding", "CacheC", "--glossary", "docs/glossary.md", "--distinct-from", "cache-a", "--dry-run", "--json"})
	if code != 0 {
		t.Fatalf("code=%d err=%s", code, errb.String())
	}
	var got struct {
		Dry           bool `json:"dry_run"`
		Before, After int
		Files         []string
	}
	if json.Unmarshal(out.Bytes(), &got) != nil || !got.Dry || !strings.Contains(out.String(), "before_family_count") || len(got.Files) < 2 {
		t.Fatalf("bad plan %s", out.String())
	}
	if _, e := os.Stat(filepath.Join(root, conceptcatalog.DataRel, "rows-cache-authored.json")); !os.IsNotExist(e) {
		t.Fatal("dry run wrote a row")
	}
}
func TestConceptPositionRejectsTestOnlyGrounding(t *testing.T) {
	c, root := conceptCLIFixture(t)
	os.WriteFile(filepath.Join(root, "only_test.go"), []byte("package p // OnlyTestGrounding"), 0600)
	var out, errb bytes.Buffer
	code := runConceptPosition(&out, &errb, c, []string{"--id", "x", "--canonical", "X", "--family", "cache", "--definition", "d", "--distinction", "not a", "--grounding", "OnlyTestGrounding", "--glossary", "docs/glossary.md", "--distinct-from", "cache-a", "--dry-run"})
	if code == 0 || !strings.Contains(errb.String(), "production corpus") {
		t.Fatalf("code=%d err=%s", code, errb.String())
	}
}

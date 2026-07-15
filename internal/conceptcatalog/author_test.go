package conceptcatalog

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func fixture(t *testing.T) (Catalog, string) {
	t.Helper()
	root := t.TempDir()
	data := filepath.Join(root, DataRel)
	if err := os.MkdirAll(data, 0755); err != nil {
		t.Fatal(err)
	}
	gl := filepath.Join(root, "docs", "glossary.md")
	os.MkdirAll(filepath.Dir(gl), 0755)
	os.WriteFile(gl, []byte("# Glossary\n"), 0600)
	c := good()
	c.Dir = data
	meta, _ := json.MarshalIndent(c.Meta, "", "  ")
	if err := os.WriteFile(filepath.Join(data, "_meta.json"), append(meta, '\n'), 0600); err != nil {
		t.Fatal(err)
	}
	return c, root
}
func TestPlanPositionAtomicDryRunAndApply(t *testing.T) {
	c, root := fixture(t)
	req := PositionRequest{ID: "cache-c", Canonical: "Cache C", Family: "cache", Definition: "the third cache", Distinction: "not the first cache", Kind: "symbol", Grounding: "CacheC", GroundingKind: "symbol", Glossary: "docs/glossary.md", DistinctFrom: []string{"cache-a"}}
	p, err := PlanPosition(c, req)
	if err != nil {
		t.Fatal(err)
	}
	if p.BeforeFamilyCount != 2 || p.AfterFamilyCount != 3 || len(p.Files) != 2 {
		t.Fatalf("bad plan: %+v", p)
	}
	if _, err = os.Stat(filepath.Join(c.Dir, "rows-cache-authored.json")); !os.IsNotExist(err) {
		t.Fatal("dry-run mutated row file")
	}
	if err = Apply(p); err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadDir(c.Dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(loaded.Rows) != 1 || loaded.Rows[0].ID != "cache-c" {
		t.Fatalf("written row mismatch: %+v", loaded.Rows)
	}
	b, _ := os.ReadFile(filepath.Join(root, "docs", "glossary.md"))
	if string(b) == "# Glossary\n" {
		t.Fatal("glossary not updated")
	}
}
func TestPlanPositionFailureLeavesAllFilesUnchanged(t *testing.T) {
	c, root := fixture(t)
	gl := filepath.Join(root, "docs", "glossary.md")
	before, _ := os.ReadFile(gl)
	_, err := PlanPosition(c, PositionRequest{ID: "bad", Canonical: "Bad", Family: "cache", Definition: "d", Distinction: "x", Kind: "symbol", Grounding: "Bad", GroundingKind: "symbol", Glossary: "docs/glossary.md", DistinctFrom: []string{"not-an-id"}})
	if err == nil {
		t.Fatal("want invalid ref failure")
	}
	after, _ := os.ReadFile(gl)
	if string(before) != string(after) {
		t.Fatal("validation failure changed glossary")
	}
}
func TestPlanClassifyAndRefuseGroundedConcept(t *testing.T) {
	c, _ := fixture(t)
	p, err := PlanClassify(c, ClassifyRequest{Family: "cache", Token: "cache_helper_test", Category: "test-only", Reason: "only a fixture helper"})
	if err != nil {
		t.Fatal(err)
	}
	if len(p.Files) != 1 || p.BeforeFamilyCount != p.AfterFamilyCount {
		t.Fatalf("bad classify plan: %+v", p)
	}
	_, err = PlanClassify(c, ClassifyRequest{Family: "cache", Token: "CacheA", Category: "incidental", Reason: "hide it"})
	if err == nil {
		t.Fatal("positioned grounding was silently hidden")
	}
}
func TestProductionCorpusExcludesTestsAndBuildTags(t *testing.T) {
	root := t.TempDir()
	os.WriteFile(filepath.Join(root, "only_test.go"), []byte("package p // TestOnlyToken"), 0600)
	os.WriteFile(filepath.Join(root, "wip.go"), []byte("//go:build wip_x\n\npackage p // TaggedOnlyToken"), 0600)
	os.WriteFile(filepath.Join(root, "prod.go"), []byte("package p\nconst ProductionToken=1"), 0600)
	for tok, want := range map[string]bool{"TestOnlyToken": false, "TaggedOnlyToken": false, "ProductionToken": true} {
		got, err := ProductionCorpus(root, tok)
		if err != nil || got != want {
			t.Fatalf("%s got %v,%v want %v", tok, got, err, want)
		}
	}
}

func TestAppendRowPreservesExistingBytes(t *testing.T) {
	p := filepath.Join(t.TempDir(), "rows-cache.json")
	before := []byte("{\n  \"note\": \"keep spacing\",\n  \"rows\": [\n    {\"id\":\"old\", \"canonical\": \"Old\"}\n  ]\n}\n")
	if err := os.WriteFile(p, before, 0600); err != nil {
		t.Fatal(err)
	}
	out, err := appendRowFile(p, Row{ID: "new", Canonical: "New"})
	if err != nil {
		t.Fatal(err)
	}
	prefix := string(before[:bytes.LastIndex(before, []byte("]"))])
	if !strings.HasPrefix(string(out), prefix) {
		t.Fatalf("existing row bytes were reformatted:\n%s", out)
	}
}

func TestApplyWritesEveryDestination(t *testing.T) {
	d := t.TempDir()
	a := filepath.Join(d, "a.txt")
	b := filepath.Join(d, "b.txt")
	if err := os.WriteFile(a, []byte("old"), 0600); err != nil {
		t.Fatal(err)
	}
	plan := Plan{Changes: []Change{{Path: a, Content: []byte("new")}, {Path: b, Content: []byte("second")}}}
	if err := Apply(plan); err != nil {
		t.Fatal(err)
	}
	got, _ := os.ReadFile(a)
	got2, _ := os.ReadFile(b)
	if string(got) != "new" || string(got2) != "second" {
		t.Fatalf("atomic write mismatch: %q %q", got, got2)
	}
}

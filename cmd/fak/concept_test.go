package main

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
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
		Dry    bool `json:"dry_run"`
		Before int  `json:"before_family_count"`
		After  int  `json:"after_family_count"`
		Files  []string
	}
	if json.Unmarshal(out.Bytes(), &got) != nil || !got.Dry || got.Before != 1 || got.After != 2 || len(got.Files) < 2 {
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

func TestConceptGeneratePreservesClassification(t *testing.T) {
	c, root := conceptCLIFixture(t)
	tools := filepath.Join(root, "tools")
	if err := os.MkdirAll(tools, 0755); err != nil {
		t.Fatal(err)
	}
	script := `import argparse, json
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument("--workspace")
p.add_argument("--data")
p.add_argument("--markdown-dir")
p.add_argument("--json", action="store_true")
a=p.parse_args()
meta=json.loads((Path(a.data)/"_meta.json").read_text())
tokens=[x.get("token") for x in meta["families"][0].get("classifications", [])]
out=Path(a.markdown_dir); out.mkdir(parents=True, exist_ok=True)
for name in ("README.md","INDEX.md","DEBT.md","GAPS.md","PAIRS.md","CONCEPTS.md","FAMILIES.md","SOURCES.md","ADR.md"):
    (out/name).write_text("tokens="+",".join(tokens)+"\n")
print(json.dumps({"families": [{"id": "cache", "gaps": [], "coverage": 1.0}], "critical": {"unresolved": 0, "low_coverage_families": 0}}))
`
	if err := os.WriteFile(filepath.Join(tools, "concept_disambiguation_scorecard.py"), []byte(script), 0600); err != nil {
		t.Fatal(err)
	}

	var out, errb bytes.Buffer
	if code := runConceptClassify(&out, &errb, c, []string{"--family", "cache", "--token", "cache_probe_token", "--category", "incidental", "--reason", "fixture"}); code != 0 {
		t.Fatalf("classify code=%d err=%s", code, errb.String())
	}
	reloaded, err := conceptcatalog.Load(root)
	if err != nil {
		t.Fatal(err)
	}
	out.Reset()
	errb.Reset()
	if code := runConceptGenerate(&out, &errb, reloaded, []string{"--json"}); code != 0 {
		t.Fatalf("generate code=%d err=%s", code, errb.String())
	}
	meta, err := os.ReadFile(filepath.Join(root, conceptcatalog.DataRel, "_meta.json"))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(meta, []byte("cache_probe_token")) {
		t.Fatal("classification disappeared from canonical source after regeneration")
	}
	page, err := os.ReadFile(filepath.Join(root, "docs", "concept-disambiguation-scorecard", "README.md"))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(page, []byte("cache_probe_token")) {
		t.Fatal("generator did not read back the classification written by concept classify")
	}
}

func TestConceptClassifyStageMakesIndexAwareAdmissionSeeRemedy(t *testing.T) {
	c, root := conceptCLIFixture(t)
	if out, err := runGitAt(root, "init"); err != nil {
		t.Fatalf("git init: %v: %s", err, out)
	}
	if out, err := runGitAt(root, "add", "."); err != nil {
		t.Fatalf("git add fixture: %v: %s", err, out)
	}
	if out, err := runGitAt(root, "-c", "user.name=fak-test", "-c", "user.email=fak@example.invalid", "commit", "-m", "fixture"); err != nil {
		t.Fatalf("git commit fixture: %v: %s", err, out)
	}

	var out, errb bytes.Buffer
	code := runConceptClassify(&out, &errb, c, []string{"--stage", "--family", "cache", "--token", "cache_probe_token", "--category", "incidental", "--reason", "fixture"})
	if code != 0 {
		t.Fatalf("classify code=%d err=%s", code, errb.String())
	}
	staged, err := runGitAt(root, "diff", "--cached", "--name-only")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(staged, filepath.ToSlash(filepath.Join(conceptcatalog.DataRel, "_meta.json"))) {
		t.Fatalf("canonical source was not staged; index-aware admission cannot see remedy:\n%s", staged)
	}
}

func TestConceptStageRejectsDryRun(t *testing.T) {
	c, _ := conceptCLIFixture(t)
	var out, errb bytes.Buffer
	code := runConceptClassify(&out, &errb, c, []string{"--stage", "--dry-run", "--family", "cache", "--token", "cache_probe_token", "--category", "incidental", "--reason", "fixture"})
	if code != 2 || !strings.Contains(errb.String(), "--stage cannot be combined") {
		t.Fatalf("code=%d err=%s", code, errb.String())
	}
}

func runGitAt(root string, args ...string) (string, error) {
	cmd := exec.Command("git", args...)
	cmd.Dir = root
	out, err := cmd.CombinedOutput()
	return string(out), err
}

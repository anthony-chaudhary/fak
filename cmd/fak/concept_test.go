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
	b, _ := json.Marshal(map[string]any{"meta": map[string]string{"as_of": "2026-11-24", "fak_version": "test"}, "families": meta.Families})
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
	// The script's presence marks a tree that carries generated concept docs;
	// the generator itself runs in-process.
	if err := os.WriteFile(filepath.Join(tools, "concept_disambiguation_scorecard.py"), []byte("# parity oracle\n"), 0600); err != nil {
		t.Fatal(err)
	}
	// Two production files make the probe token a discovered, uncovered member
	// of the cache family, so the generated page must change once it is
	// classified away.
	for _, pkg := range []string{"a", "b"} {
		dir := filepath.Join(root, "internal", pkg)
		if err := os.MkdirAll(dir, 0755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, pkg+".go"), []byte("package "+pkg+"\n\nconst cache_probe_token = 1\n"), 0600); err != nil {
			t.Fatal(err)
		}
	}
	generatePages := func(cat conceptcatalog.Catalog) string {
		t.Helper()
		var out, errb bytes.Buffer
		if code := runConceptGenerate(&out, &errb, cat, []string{"--json"}); code != 0 {
			t.Fatalf("generate code=%d err=%s", code, errb.String())
		}
		var pages strings.Builder
		for _, name := range []string{"README.md", "INDEX.md"} {
			page, err := os.ReadFile(filepath.Join(root, "docs", "concept-disambiguation-scorecard", name))
			if err != nil {
				t.Fatal(err)
			}
			pages.Write(page)
		}
		return pages.String()
	}
	before := generatePages(c)

	var out, errb bytes.Buffer
	if code := runConceptClassify(&out, &errb, c, []string{"--family", "cache", "--token", "cache_probe_token", "--category", "incidental", "--reason", "fixture"}); code != 0 {
		t.Fatalf("classify code=%d err=%s", code, errb.String())
	}
	reloaded, err := conceptcatalog.Load(root)
	if err != nil {
		t.Fatal(err)
	}
	after := generatePages(reloaded)
	meta, err := os.ReadFile(filepath.Join(root, conceptcatalog.DataRel, "_meta.json"))
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(meta, []byte("cache_probe_token")) {
		t.Fatal("classification disappeared from canonical source after regeneration")
	}
	if !strings.Contains(before, "| As of | 2026-11-24 (fak test) |") {
		t.Fatalf("generator did not render the catalog it was given:\n%s", before)
	}
	// Coverage-by-family row: positioned | discovered | uncovered.
	if !strings.Contains(before, "| cache | 0 | 1 | 1 |") {
		t.Fatalf("unclassified probe token is not an uncovered cache token before classify:\n%s", before)
	}
	if !strings.Contains(after, "| cache | 0 | 0 | 0 |") || strings.Contains(after, "| cache | 0 | 1 | 1 |") {
		t.Fatalf("generated page did not read back the classification written by concept classify:\n%s", after)
	}
	if !strings.Contains(after, "**0** (clarity 0 + coverage 0)") {
		t.Fatalf("classified token still counts as coverage debt on the generated page:\n%s", after)
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

func TestConceptAdmissionPathsFlag(t *testing.T) {
	_, root := conceptCLIFixture(t)
	if out, err := runGitAt(root, "init"); err != nil {
		t.Fatalf("git init: %v: %s", err, out)
	}
	if out, err := runGitAt(root, "add", "."); err != nil {
		t.Fatalf("git add fixture: %v: %s", err, out)
	}
	if out, err := runGitAt(root, "-c", "user.name=fak-test", "-c", "user.email=fak@example.invalid", "commit", "-m", "fixture"); err != nil {
		t.Fatalf("git commit fixture: %v: %s", err, out)
	}
	demo := filepath.Join(root, "internal", "demo")
	if err := os.MkdirAll(demo, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(demo, "demo.go"), []byte("package demo\nconst CacheUnclassified=1\n"), 0600); err != nil {
		t.Fatal(err)
	}

	var out, errb bytes.Buffer
	code := runConceptAdmission(&out, &errb, root, []string{"--paths", "internal/demo/demo.go"})
	if code != 0 {
		t.Fatalf("expected code 0, got %d (err: %s)", code, errb.String())
	}
	if !strings.Contains(out.String(), "CacheUnclassified") {
		t.Fatalf("expected finding for CacheUnclassified, got %q", out.String())
	}
}

func runGitAt(root string, args ...string) (string, error) {
	cmd := exec.Command("git", args...)
	cmd.Dir = root
	out, err := cmd.CombinedOutput()
	return string(out), err
}

// undateConceptFixture drops the meta block, which the generator refuses.
func undateConceptFixture(t *testing.T, c conceptcatalog.Catalog) {
	t.Helper()
	b, err := json.Marshal(c.Meta)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(c.Dir, "_meta.json"), b, 0600); err != nil {
		t.Fatal(err)
	}
}

func TestConceptGenerateFailsLoudOnUndatedCatalog(t *testing.T) {
	c, root := conceptCLIFixture(t)
	undateConceptFixture(t, c)
	outDir := filepath.Join(root, "docs", "concept-disambiguation-scorecard")
	if err := os.MkdirAll(outDir, 0755); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"README.md", "INDEX.md"} {
		if err := os.WriteFile(filepath.Join(outDir, name), []byte("stale\n"), 0644); err != nil {
			t.Fatal(err)
		}
	}

	var out, errb bytes.Buffer
	code := runConceptGenerate(&out, &errb, c, []string{})
	if code == 0 {
		t.Fatal("concept generate must fail with non-zero code when the catalog is undated")
	}
	if !strings.Contains(errb.String(), "refusing to render an undated scorecard") {
		t.Fatalf("concept generate must report the refusal, got: %s", errb.String())
	}
	if strings.Contains(out.String(), "GENERATED concept scorecard") {
		t.Fatal("concept generate must not report success when generation failed")
	}
	for _, name := range []string{"README.md", "INDEX.md"} {
		if got, _ := os.ReadFile(filepath.Join(outDir, name)); string(got) != "stale\n" {
			t.Fatalf("%s was rewritten by a refused generation: %q", name, got)
		}
	}
}

// fak-test:runtime fast est=600ms
func TestConceptGenerateRequiresNewCompleteArtifacts(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip(err)
	}
	for _, tc := range []struct {
		name     string
		args     []string
		undated  bool
		wantCode int
	}{
		{"undated refusal", nil, true, 1},
		{"complete ACTION", []string{"--json"}, false, 0},
		{"explicit worktree", []string{"--staged=false", "--json"}, false, 0},
		{"tree requires staged mode", []string{"--tree", "HEAD"}, false, 2},
		{"unknown flag", []string{"--unknown"}, false, 2},
		{"positional unsupported", []string{"extra"}, false, 2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c, root := conceptCLIFixture(t)
			if tc.undated {
				undateConceptFixture(t, c)
			}
			if out, err := runGitAt(root, "init"); err != nil {
				t.Fatalf("git init: %v: %s", err, out)
			}
			t.Chdir(root)
			outDir := filepath.Join(root, "docs", "concept-disambiguation-scorecard")
			if err := os.MkdirAll(outDir, 0755); err != nil {
				t.Fatal(err)
			}
			for _, name := range []string{"README.md", "INDEX.md"} {
				if err := os.WriteFile(filepath.Join(outDir, name), []byte("old\n"), 0644); err != nil {
					t.Fatal(err)
				}
			}
			var out, errb bytes.Buffer
			if code := runConceptCLI(&out, &errb, append([]string{"generate"}, tc.args...)); code != tc.wantCode {
				t.Fatalf("code=%d want=%d stdout=%s stderr=%s", code, tc.wantCode, out.String(), errb.String())
			}
			if tc.wantCode == 0 {
				var result struct {
					OK bool `json:"ok"`
				}
				if err := json.Unmarshal(out.Bytes(), &result); err != nil || !result.OK {
					t.Fatalf("invalid success JSON: %s (%v)", out.String(), err)
				}
			} else if out.Len() != 0 || errb.Len() == 0 {
				t.Fatalf("failure must be diagnostic only: stdout=%s stderr=%s", out.String(), errb.String())
			}
			for name, title := range map[string]string{"README.md": "fak concept-disambiguation scorecard", "INDEX.md": "fak concept name index"} {
				got, err := os.ReadFile(filepath.Join(outDir, name))
				if err != nil {
					t.Fatal(err)
				}
				regenerated := strings.HasPrefix(string(got), "---\ntitle: \""+title)
				if regenerated != (tc.wantCode == 0) || (!regenerated && string(got) != "old\n") {
					t.Fatalf("%s = %q after exit %d", name, got, tc.wantCode)
				}
			}
		})
	}
}

func TestConceptScorecardVerbFoldsInProcess(t *testing.T) {
	_, root := conceptCLIFixture(t)
	var out, errb bytes.Buffer
	code := runConcept(&out, &errb, []string{"scorecard", "--workspace", root, "--json"})
	if code != 0 && code != 1 {
		t.Fatalf("code=%d err=%s", code, errb.String())
	}
	var payload struct {
		Schema string `json:"schema"`
		Corpus struct {
			Rows int `json:"rows"`
		} `json:"corpus"`
	}
	if err := json.Unmarshal(out.Bytes(), &payload); err != nil {
		t.Fatalf("scorecard --json is not JSON: %v: %s", err, out.String())
	}
	if payload.Schema != "fak-concept-disambiguation-scorecard/1" {
		t.Fatalf("schema = %q", payload.Schema)
	}
}

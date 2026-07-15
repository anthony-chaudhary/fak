package conceptcatalog

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
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

func scorecardE2EFixture(t *testing.T, gap string) (Catalog, string) {
	t.Helper()
	root := t.TempDir()
	data := filepath.Join(root, DataRel)
	if err := os.MkdirAll(data, 0755); err != nil {
		t.Fatal(err)
	}
	metaDoc := map[string]any{"schema": "fak-concept-disambiguation-scorecard/1", "glossary": "docs/glossary.md", "families": []map[string]any{{"id": "cache", "name": "Cache", "roots": []string{"cache"}, "ignore": []string{"cache"}, "min_files": 1}}}
	mb, _ := json.MarshalIndent(metaDoc, "", "  ")
	if err := os.WriteFile(filepath.Join(data, "_meta.json"), append(mb, '\n'), 0600); err != nil {
		t.Fatal(err)
	}
	rows := good().Rows
	for i := range rows {
		rows[i].GlossaryAnchor = "docs/glossary.md"
		rows[i].Verdict = "crystal"
		rows[i].Aliases = []string{}
		rows[i].Gaps = []string{}
	}
	rb, _ := json.MarshalIndent(map[string]any{"rows": rows}, "", "  ")
	if err := os.WriteFile(filepath.Join(data, "rows-cache.json"), append(rb, '\n'), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(root, "internal", "fixture"), 0755); err != nil {
		t.Fatal(err)
	}
	prod := "package fixture\nconst CacheA=1\nconst CacheB=2\nconst " + gap + "=3\n"
	if err := os.WriteFile(filepath.Join(root, "internal", "fixture", "fixture.go"), []byte(prod), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(root, "docs"), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "docs", "glossary.md"), []byte("# Glossary\n"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(root, "tools"), 0755); err != nil {
		t.Fatal(err)
	}
	src := filepath.Join("..", "..", "tools", "concept_disambiguation_scorecard.py")
	py, err := os.ReadFile(src)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "tools", "concept_disambiguation_scorecard.py"), py, 0600); err != nil {
		t.Fatal(err)
	}
	c, err := Load(root)
	if err != nil {
		t.Fatal(err)
	}
	return c, root
}

func scorecardRun(t *testing.T, root string, args ...string) (string, error) {
	t.Helper()
	base := []string{filepath.Join(root, "tools", "concept_disambiguation_scorecard.py"), "--workspace", root, "--data", filepath.Join(root, DataRel)}
	cmd := exec.Command("python", append(base, args...)...)
	out, err := cmd.CombinedOutput()
	return string(out), err
}

func TestAuthoringModesCloseOneGapWithCriticalCleanAndFreshDocs(t *testing.T) {
	t.Run("position", func(t *testing.T) {
		c, root := scorecardE2EFixture(t, "CacheGap")
		if out, err := scorecardRun(t, root, "--gaps"); err == nil || !strings.Contains(strings.ToLower(out), "cachegap") {
			t.Fatalf("want one pre-authoring gap, err=%v out=%s", err, out)
		}
		req := PositionRequest{ID: "cache-gap", Canonical: "Cache Gap", Family: "cache", Definition: "the fixture gap cache", Distinction: "a third cache rather than cache a", Kind: "symbol", Grounding: "CacheGap", GroundingKind: "symbol", Glossary: "docs/glossary.md", DistinctFrom: []string{"cache-a"}}
		plan, err := PlanPosition(c, req)
		if err != nil {
			t.Fatal(err)
		}
		before, _ := os.ReadFile(filepath.Join(c.Dir, "rows-cache-authored.json"))
		if len(before) != 0 {
			t.Fatal("dry-run unexpectedly wrote row")
		}
		if err := Apply(plan); err != nil {
			t.Fatal(err)
		}
		if out, _ := scorecardRun(t, root, "--critical"); !strings.Contains(out, "no critical rows") {
			t.Fatalf("critical not clean: %s", out)
		}
		if out, err := scorecardRun(t, root, "--gaps"); err != nil || strings.Contains(strings.ToLower(out), "cachegap") {
			t.Fatalf("position did not take gap to zero: %v %s", err, out)
		}
		want, err := os.ReadFile(filepath.Join(root, "docs", "concept-disambiguation-scorecard", "README.md"))
		if err != nil {
			t.Fatal(err)
		}
		regen := filepath.Join(root, "regen")
		if out, err := scorecardRun(t, root, "--markdown-dir", regen); err != nil {
			t.Fatalf("regen: %v %s", err, out)
		}
		got, _ := os.ReadFile(filepath.Join(regen, "README.md"))
		if !bytes.Equal(want, got) {
			t.Fatal("generated README is stale")
		}
	})
	t.Run("classify", func(t *testing.T) {
		c, root := scorecardE2EFixture(t, "CacheNoise")
		if out, err := scorecardRun(t, root, "--gaps"); err == nil || !strings.Contains(strings.ToLower(out), "cachenoise") {
			t.Fatalf("want one pre-classification gap, err=%v out=%s", err, out)
		}
		plan, err := PlanClassify(c, ClassifyRequest{Family: "cache", Token: "CacheNoise", Category: "incidental", Reason: "fixture-only incidental name"})
		if err != nil {
			t.Fatal(err)
		}
		metaBefore, _ := os.ReadFile(filepath.Join(c.Dir, "_meta.json"))
		if bytes.Contains(metaBefore, []byte("CacheNoise")) {
			t.Fatal("dry-run changed metadata")
		}
		if err := Apply(plan); err != nil {
			t.Fatal(err)
		}
		if out, _ := scorecardRun(t, root, "--critical"); !strings.Contains(out, "no critical rows") {
			t.Fatalf("critical not clean: %s", out)
		}
		if out, err := scorecardRun(t, root, "--gaps"); err != nil || strings.Contains(strings.ToLower(out), "cachenoise") {
			t.Fatalf("classification did not take gap to zero: %v %s", err, out)
		}
		want, err := os.ReadFile(filepath.Join(root, "docs", "concept-disambiguation-scorecard", "README.md"))
		if err != nil {
			t.Fatal(err)
		}
		regen := filepath.Join(root, "regen")
		if out, err := scorecardRun(t, root, "--markdown-dir", regen); err != nil {
			t.Fatalf("regen: %v %s", err, out)
		}
		got, _ := os.ReadFile(filepath.Join(regen, "README.md"))
		if !bytes.Equal(want, got) {
			t.Fatal("generated README is stale")
		}
	})
}

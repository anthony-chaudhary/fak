package conceptcatalog

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync/atomic"
	"testing"
	"time"
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
	// The rows have to exist as bytes, not only in memory: positioning a concept
	// writes the reverse half of its boundary into the twin's OWN row file.
	rows, _ := json.MarshalIndent(map[string]any{"rows": c.Rows}, "", "  ")
	if err := os.WriteFile(filepath.Join(data, c.Rows[0].Source), append(rows, '\n'), 0600); err != nil {
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
	if p.BeforeFamilyCount != 2 || p.AfterFamilyCount != 3 || len(p.Files) != 3 {
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

// A twin row that no shard owns has no bytes to carry the reverse boundary. The
// resolver used to re-anchor that empty source on the catalog directory and read
// the whole rows-*.json corpus directory as if it were one file.
func TestPlanPositionSkipsBackReferenceForRowNoFileOwns(t *testing.T) {
	c, _ := fixture(t)
	for i := range c.Rows {
		c.Rows[i].Source = ""
	}
	p, err := PlanPosition(c, PositionRequest{ID: "cache-c", Canonical: "Cache C", Family: "cache", Definition: "the third cache", Distinction: "not the first cache", Kind: "symbol", Grounding: "CacheC", GroundingKind: "symbol", Glossary: "docs/glossary.md", DistinctFrom: []string{"cache-a"}})
	if err != nil {
		t.Fatal(err)
	}
	if len(p.Files) != 2 {
		t.Fatalf("want the row file and the glossary only, got %v", p.Files)
	}
	for _, f := range p.Files {
		if filepath.Clean(f) == filepath.Clean(c.Dir) {
			t.Fatalf("planned a write to the corpus directory itself: %v", p.Files)
		}
	}
}
func TestPlanClassifyManyIsAtomicIdempotentAndBounded(t *testing.T) {
	c, _ := fixture(t)
	rows := []ClassifyRequest{
		{Family: "cache", Token: "cache_batch", Category: "incidental", Reason: "batch fixture"},
		{Family: "cache", Token: "gate_batch", Category: "test-only", Reason: "batch fixture"},
		{Family: "cache", Token: "session_batch", Category: "build-tag-only", Reason: "batch fixture"},
	}
	before, err := os.ReadFile(filepath.Join(c.Dir, "_meta.json"))
	if err != nil {
		t.Fatal(err)
	}
	started := time.Now()
	plan, err := PlanClassifyMany(c, rows)
	if err != nil {
		t.Fatal(err)
	}
	if elapsed := time.Since(started); elapsed > 2*time.Second {
		t.Fatalf("batch planning exceeded bound: %s", elapsed)
	}
	afterPlan, _ := os.ReadFile(filepath.Join(c.Dir, "_meta.json"))
	if !bytes.Equal(before, afterPlan) {
		t.Fatal("planner wrote partial corpus state")
	}
	if len(plan.Classifications) != 3 {
		t.Fatalf("rows=%d", len(plan.Classifications))
	}
	if err := Apply(plan); err != nil {
		t.Fatal(err)
	}
	c2, err := Load(filepath.Dir(filepath.Dir(c.Dir)))
	if err != nil {
		t.Fatal(err)
	}
	again, err := PlanClassifyMany(c2, rows)
	if err != nil {
		t.Fatal(err)
	}
	if len(again.Files) != 0 {
		t.Fatalf("idempotent transaction would rewrite %v", again.Files)
	}
	bad := append(append([]ClassifyRequest{}, rows...), ClassifyRequest{Family: "missing", Token: "x", Category: "incidental", Reason: "bad"})
	stable, _ := os.ReadFile(filepath.Join(c.Dir, "_meta.json"))
	if _, err := PlanClassifyMany(c2, bad); err == nil {
		t.Fatal("invalid batch accepted")
	}
	got, _ := os.ReadFile(filepath.Join(c.Dir, "_meta.json"))
	if !bytes.Equal(stable, got) {
		t.Fatal("invalid batch changed corpus")
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

// TestPlanClassifyPreservesEveryTopLevelMetaKey pins the round-trip contract of the
// _meta.json rewrite. PlanClassify DECODES the whole file into a struct and RE-ENCODES the
// file from that struct, so any top-level key the struct fails to declare is not merely
// ignored — it is DELETED from the file the plan writes.
//
// The key that actually bit is "meta", which carries {as_of, fak_version}: the canonical
// generator refuses to render an undated scorecard, so dropping the block turned every
// `fak concept classify` in the repo into a generator crash. It surfaced as
// "decode planned snapshot: unexpected end of JSON input" — naming neither the deleted
// block nor the file it vanished from — because the crash exits 1, and exit 1 is the same
// code the generator uses for an honestly-generated ACTION snapshot.
//
// The shared fixture cannot witness this: Metadata declares only Families, so the
// _meta.json it writes has no schema, no glossary and no meta block available to lose.
// This test therefore writes a file shaped like the REAL corpus and asserts the plan hands
// every key back. Written generically over the decoded map rather than as three named
// assertions, so a future top-level key added to the corpus is covered on arrival.
func TestPlanClassifyPreservesEveryTopLevelMetaKey(t *testing.T) {
	c, _ := fixture(t)
	path := filepath.Join(c.Dir, "_meta.json")

	// Re-shape the fixture's _meta.json to match the real corpus: the fixture writes only
	// "families", while the corpus also carries "schema", "meta" and "glossary".
	var onDisk map[string]any
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(b, &onDisk); err != nil {
		t.Fatal(err)
	}
	onDisk["schema"] = "concept-disambiguation/1"
	onDisk["glossary"] = "docs/glossary.md"
	onDisk["meta"] = map[string]any{"as_of": "2026-08-05", "fak_version": "0.43.0"}
	if b, err = json.MarshalIndent(onDisk, "", "  "); err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(path, append(b, '\n'), 0600); err != nil {
		t.Fatal(err)
	}
	reloaded, err := LoadDir(c.Dir)
	if err != nil {
		t.Fatal(err)
	}

	plan, err := PlanClassify(reloaded, ClassifyRequest{Family: "cache", Token: "cache_helper_test", Category: "test-only", Reason: "only a fixture helper"})
	if err != nil {
		t.Fatal(err)
	}
	var planned map[string]any
	if err = json.Unmarshal(plan.Changes[0].Content, &planned); err != nil {
		t.Fatalf("planned _meta.json does not decode: %v", err)
	}
	for key := range onDisk {
		if _, ok := planned[key]; !ok {
			t.Errorf("classify deleted top-level key %q from _meta.json", key)
		}
	}

	// The dating block specifically must survive with its VALUES, not merely its key: an
	// empty as_of/fak_version fails the generator exactly as a missing block does.
	meta, ok := planned["meta"].(map[string]any)
	if !ok {
		t.Fatalf("meta block is %T, want an object", planned["meta"])
	}
	for _, field := range []string{"as_of", "fak_version"} {
		if s, _ := meta[field].(string); s == "" {
			t.Errorf("meta.%s = %q after classify, want the value carried through", field, s)
		}
	}

	// And the classification itself still landed — preserving the file must not come at
	// the cost of the write the caller asked for.
	if !bytes.Contains(plan.Changes[0].Content, []byte("cache_helper_test")) {
		t.Error("classified token missing from the planned _meta.json")
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

// Run artifacts are not production source. Before this, the walk descended every
// in-tree run directory, so a token that survived only in a 48-day-old dispatch log
// counted as grounded - and reading them made `fak concept position` peak at 20.79GB.
func TestProductionCorpusIgnoresRunArtifactDirectories(t *testing.T) {
	root := t.TempDir()
	write := func(rel, body string) {
		t.Helper()
		p := filepath.Join(root, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(p), 0755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0600); err != nil {
			t.Fatal(err)
		}
	}
	write("internal/x/prod.go", "package x\nconst SourceGrounding = 1")
	write(".github/workflows/ci.yml", "name: TrackedConfigGrounding")
	write(".dispatch-runs/run-1/report.md", "DispatchLogGrounding showed up in a run log")
	write("_scratch/clone/pkg/a.go", "package a // ScratchCloneGrounding")
	write(".scratch-1365/notes.md", "ScratchNumberedGrounding")
	write(".goal-runs/g/out.json", "{\"note\":\"GoalRunGrounding\"}")

	for tok, want := range map[string]bool{
		"SourceGrounding":          true,
		"TrackedConfigGrounding":   true,
		"DispatchLogGrounding":     false,
		"ScratchCloneGrounding":    false,
		"ScratchNumberedGrounding": false,
		"GoalRunGrounding":         false,
	} {
		got, err := ProductionCorpus(root, tok)
		if err != nil {
			t.Fatalf("ProductionCorpus(%q): %v", tok, err)
		}
		if got != want {
			t.Errorf("ProductionCorpus(%q) = %v, want %v", tok, got, want)
		}
	}
}

// The walk must hold one file at a time, never the concatenated tree. Buffering the
// whole corpus is what turned a 7GB matched corpus into a 20.79GB resident process.
func TestProductionCorpusManyDoesNotBufferTheWholeTree(t *testing.T) {
	root := t.TempDir()
	// ~32MB over 512 files, every line well under bufio's 64KB scan limit and no
	// single file above 64KB - so any peak far above one file means retention.
	line := strings.Repeat("qwertyuiopasdfghjklzxcvbnm", 2) + "abcdefghijk\n" // 63B + \n
	filler := strings.Repeat(line, 1024)
	for i := 0; i < 512; i++ {
		dir := filepath.Join(root, "pkg", fmt.Sprintf("p%03d", i))
		if err := os.MkdirAll(dir, 0755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, "f.go"), []byte("package p\n"+filler), 0600); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(root, "pkg", "z.go"), []byte("package p\nconst LateGrounding = 1"), 0600); err != nil {
		t.Fatal(err)
	}

	var peak atomic.Uint64
	stop, done := make(chan struct{}), make(chan struct{})
	go func() {
		defer close(done)
		var m runtime.MemStats
		for {
			select {
			case <-stop:
				return
			case <-time.After(time.Millisecond):
			}
			runtime.ReadMemStats(&m)
			if m.HeapAlloc > peak.Load() {
				peak.Store(m.HeapAlloc)
			}
		}
	}()
	runtime.GC()
	found, err := ProductionCorpusMany(root, []string{"LateGrounding", "NeverAppearsGrounding"})
	close(stop)
	<-done
	if err != nil {
		t.Fatal(err)
	}
	if !found[token("LateGrounding")] || found[token("NeverAppearsGrounding")] {
		t.Fatalf("wrong verdicts over the large tree: %v", found)
	}
	if got := peak.Load(); got > 16<<20 {
		t.Errorf("peak heap %.1fMB over a 32MB tree: the walk is retaining the corpus, not streaming it", float64(got)/(1<<20))
	}
}

// Per-file matching must stay exactly equivalent to matching the concatenation:
// token() keeps only [a-z0-9], so a want can never span the newline between lines.
func TestProductionCorpusManyKeepsTokensIndependent(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "a.go"), []byte("package p\n// alpha\n// beta\n"), 0600); err != nil {
		t.Fatal(err)
	}
	got, err := ProductionCorpusMany(root, []string{"Alpha", "Beta", "alpha-beta", "", "Alpha"})
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]bool{"alpha": true, "beta": true, "alphabeta": false}
	if len(got) != len(want) {
		t.Fatalf("got %d verdicts %v, want %d (blank dropped, duplicate deduped)", len(got), got, len(want))
	}
	for tok, w := range want {
		if got[tok] != w {
			t.Errorf("%q = %v, want %v", tok, got[tok], w)
		}
	}
}

func TestAppendRowPreservesExistingBytes(t *testing.T) {
	before := []byte("{\n  \"note\": \"keep spacing\",\n  \"rows\": [\n    {\"id\":\"old\", \"canonical\": \"Old\"}\n  ]\n}\n")
	out, err := appendRow(before, Row{ID: "new", Canonical: "New"})
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
	// The "meta" dating block is REQUIRED, not decorative: the canonical generator refuses
	// to render an undated scorecard, and this fixture drives the generator for real. Its
	// absence is why this end-to-end test failed with an opaque
	// "decode planned snapshot: unexpected end of JSON input" — the same blind spot that
	// let PlanClassify ship a round-trip which deleted this very block from the real corpus.
	metaDoc := map[string]any{"schema": "fak-concept-disambiguation-scorecard/1", "meta": map[string]any{"as_of": "2026-08-06", "fak_version": "0.0.0-test"}, "glossary": "docs/glossary.md", "families": []map[string]any{{"id": "cache", "name": "Cache", "roots": []string{"cache"}, "ignore": []string{"cache"}, "min_files": 1}}}
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
	python, err := ResolvePython()
	if err != nil {
		t.Fatalf("resolve python: %v", err)
	}
	base := []string{filepath.Join(root, "tools", "concept_disambiguation_scorecard.py"), "--workspace", root, "--data", filepath.Join(root, DataRel)}
	cmd := exec.Command(python, append(base, args...)...)
	out, err := cmd.CombinedOutput()
	return string(out), err
}

func TestAuthoringModesCloseOneGapWithCriticalCleanAndFreshDocs(t *testing.T) {
	t.Run("position", func(t *testing.T) {
		c, root := scorecardE2EFixture(t, "CacheGap")
		if out, err := scorecardRun(t, root, "--gaps"); err == nil || !strings.Contains(strings.ToLower(out), "cachegap") {
			t.Fatalf("want one pre-authoring gap, err=%v out=%s", err, out)
		}
		req := PositionRequest{ID: "cache-gap", Canonical: "Cache Gap", Family: "cache", Definition: "the fixture gap cache", Distinction: "a third cache rather than cache a or cache b", Kind: "symbol", Grounding: "CacheGap", GroundingKind: "symbol", Glossary: "docs/glossary.md", DistinctFrom: []string{"cache-b"}}
		// "Cache Gap" is two edits from "Cache A", so a reader cannot keep the two
		// names apart. Naming a DIFFERENT sibling does not pay that debt: the plan is
		// refused, and the refusal names the twin that is still undrawn.
		if _, err := PlanPosition(c, req); err == nil || !strings.Contains(err.Error(), "cache-a") {
			t.Fatalf("want a refusal naming the unseparated twin, got %v", err)
		}
		req.DistinctFrom = []string{"cache-a", "cache-b"}
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
		// Separation is mutual: the twins' own rows gained the reverse reference, in
		// their own file, without the rest of that file being reformatted.
		twins, err := os.ReadFile(filepath.Join(c.Dir, "rows-cache.json"))
		if err != nil {
			t.Fatal(err)
		}
		if n := bytes.Count(twins, []byte(`"cache-gap"`)); n != 2 {
			t.Fatalf("want a back-reference in each twin row, got %d in:\n%s", n, twins)
		}
		reloaded, err := Load(root)
		if err != nil {
			t.Fatal(err)
		}
		if ds := Validate(reloaded); len(ds) > 0 {
			t.Fatalf("back-referenced catalog no longer validates: %v", ds)
		}
		if out, err := scorecardRun(t, root, "--pairs"); err != nil || !strings.Contains(out, "0 undrawn") {
			t.Fatalf("position left a confusable pair undrawn: %v %s", err, out)
		}
		// Every generated artifact - scorecard AND name index - lands with the plan.
		regen := filepath.Join(root, "regen")
		if out, err := scorecardRun(t, root, "--markdown-dir", regen); err != nil {
			t.Fatalf("regen: %v %s", err, out)
		}
		for _, name := range []string{"README.md", "INDEX.md"} {
			want, readErr := os.ReadFile(filepath.Join(root, "docs", "concept-disambiguation-scorecard", name))
			if readErr != nil {
				t.Fatalf("%s: %v", name, readErr)
			}
			got, _ := os.ReadFile(filepath.Join(regen, name))
			if !bytes.Equal(want, got) {
				t.Fatalf("generated %s is stale", name)
			}
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

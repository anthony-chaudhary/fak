package devindex

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestVerbManifest exercises the C3 (#1290) structured verb catalog: it is sorted,
// every entry has a synopsis, and the listed verbs resolve through main.go's switch
// (so the manifest cannot list a verb the dispatch does not have). The manifest is a
// VIEW, so its entries must bind to a real lane the live taxonomy declares.
func TestVerbManifest(t *testing.T) {
	c, err := Load(FindRoot("."))
	if err != nil {
		t.Skipf("no repo root (%v); skipping live verb-manifest dogfood", err)
	}
	verbs := c.Verbs()
	if len(verbs) == 0 {
		t.Fatal("verb manifest is empty")
	}
	for i, v := range verbs {
		if v.Name == "" {
			t.Errorf("verb[%d] has no name", i)
		}
		if v.Synopsis == "" {
			t.Errorf("verb %q has no synopsis", v.Name)
		}
		if i > 0 && verbs[i-1].Name > v.Name {
			t.Errorf("verbs not sorted: %q before %q", verbs[i-1].Name, v.Name)
		}
		// Every manifest lane must be a declared leaf, or the C3->taxonomy join drifts.
		if v.Lane != "" {
			if _, ok := c.LeafByName(v.Lane); !ok {
				t.Errorf("verb %q names lane %q with no declared leaf", v.Name, v.Lane)
			}
		}
	}
	// Each manifest verb must be a live main.go case — no manifest entry for a verb
	// the dispatch lacks (the freshness gate's converse: manifest > dispatch).
	declared := liveMainVerbs(t, c.Root)
	for _, v := range verbs {
		hit := false
		for _, sp := range v.Spellings() {
			if declared[sp] {
				hit = true
			}
		}
		if !hit {
			t.Errorf("manifest verb %q (or its aliases) is not a case in main.go", v.Name)
		}
	}
}

// TestVerbsDeriveCoversUncuratedDispatch proves the catalog is a LIVE VIEW: Verbs()
// surfaces every verb the dispatch switch routes — including one with NO curated
// verbManifest entry (which gets a non-empty fallback synopsis) — so `fak index verbs`
// can never silently fall behind the binary the way a hand-maintained list does. A
// curated verb keeps its curated synopsis, and a brace-bearing case body before a later
// verb does not truncate the scan (the bug the naive "break on first }" scan had).
func TestVerbsDeriveCoversUncuratedDispatch(t *testing.T) {
	root := t.TempDir()
	mustWrite(t, root, "dos.toml", "[lanes.trees]\ncmd = [\"cmd/**\"]\n")
	mustWrite(t, root, "INDEX.md", "# INDEX\n")
	// A dispatch switch with a curated verb (index), an UNcurated verb (frobnicate),
	// and a brace-bearing case (tramp) sitting BEFORE frobnicate so the depth scan must
	// step over its `if … { … }` body to still reach frobnicate.
	mainGo := "package main\n\nimport \"os\"\n\n" +
		"func main() {\n" +
		"\tswitch os.Args[1] {\n" +
		"\tcase \"index\":\n\t\tcmdIndex(os.Args[2:])\n" +
		"\tcase \"tramp\":\n\t\tif err := f(); err != nil {\n\t\t\tos.Exit(1)\n\t\t}\n" +
		"\tcase \"frobnicate\":\n\t\tcmdFrob(os.Args[2:])\n" +
		"\tdefault:\n\t\tusage()\n\t}\n}\n"
	mustMkdir(t, root, "cmd", "fak")
	mustWrite(t, filepath.Join(root, "cmd", "fak"), "main.go", mainGo)

	c, err := Load(root)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	byName := map[string]Verb{}
	for _, v := range c.Verbs() {
		byName[v.Name] = v
	}
	// The curated verb is present with its curated synopsis (not the fallback line).
	idx, ok := byName["index"]
	if !ok {
		t.Fatal("derived catalog dropped the curated dispatch verb 'index'")
	}
	if idx.Synopsis == "" || strings.Contains(idx.Synopsis, "not yet cataloged") {
		t.Errorf("curated verb 'index' lost its curated synopsis: %q", idx.Synopsis)
	}
	// The uncurated verb is STILL surfaced (the live-view guarantee) with a non-empty
	// fallback synopsis — coverage no hand-maintained table keeps fresh on its own.
	fr, ok := byName["frobnicate"]
	if !ok {
		t.Fatal("derived catalog dropped the uncurated dispatch verb 'frobnicate' — coverage is not live")
	}
	if fr.Synopsis == "" {
		t.Error("uncurated verb 'frobnicate' has an empty synopsis")
	}
	// The brace-bearing 'tramp' case did not end the scan early: frobnicate (after it)
	// is the proof, and tramp itself is covered.
	if _, ok := byName["tramp"]; !ok {
		t.Error("derived catalog missed 'tramp' (a brace-bearing case body)")
	}
	// SearchVerbs finds the uncurated verb by name.
	if hits := c.SearchVerbs("frobnicate"); len(hits) == 0 || hits[0].Name != "frobnicate" {
		t.Errorf("SearchVerbs(frobnicate) top = %v, want frobnicate", hits)
	}
}

// liveMainVerbs returns the lowercased verb tokens of main.go's top-level switch,
// reusing the EXACT scan the drift detector does (mainDispatchVerbs), so the manifest⊆
// dispatch assertion and the dispatch⊆manifest gate cannot disagree on what a verb is.
func liveMainVerbs(t *testing.T, root string) map[string]bool {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(root, "cmd", "fak", "main.go"))
	if err != nil {
		t.Skipf("no cmd/fak/main.go (%v)", err)
	}
	out := map[string]bool{}
	for _, v := range mainDispatchVerbs(b) {
		out[v] = true
	}
	return out
}

func TestVerbByNameAndSearch(t *testing.T) {
	c := &Catalog{Root: "."}
	if _, ok := c.VerbByName("index"); !ok {
		t.Error("VerbByName(index) should resolve")
	}
	if _, ok := c.VerbByName("INDEX"); !ok {
		t.Error("VerbByName is case-insensitive")
	}
	if _, ok := c.VerbByName("nonexistent-verb"); ok {
		t.Error("VerbByName(nonexistent) should not resolve")
	}
	if _, ok := c.VerbByName(""); ok {
		t.Error("VerbByName(empty) should not resolve")
	}
	// Empty query returns the full catalog; a term ranks a name hit first.
	if got := c.SearchVerbs(""); len(got) != len(verbManifest) {
		t.Errorf("empty SearchVerbs returned %d, want %d", len(got), len(verbManifest))
	}
	hits := c.SearchVerbs("index")
	if len(hits) == 0 || hits[0].Name != "index" {
		t.Errorf("SearchVerbs(index) top hit = %v, want index", hits)
	}
	if got := c.SearchVerbs("zzz-no-match"); len(got) != 0 {
		t.Errorf("SearchVerbs(no-match) = %d, want 0", len(got))
	}
}

// writeFreshnessRepo lays down a synthetic tree with declared + drifting sources so
// CheckFreshness is tested against known bytes: gateway is declared and present;
// an UNDECLARED internal/orphan package; an INDEX.md with one live + one dead local
// link + one external URL; a main.go whose switch has a declared verb plus an
// undeclared one.
func writeFreshnessRepo(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	dosToml := "[lanes.trees]\n" +
		"gateway = [\"internal/gateway/**\"]\n" +
		"cmd     = [\"cmd/**\"]\n"
	mustWrite(t, root, "dos.toml", dosToml)

	// gateway exists+declared; orphan exists with Go files but is undeclared; empty
	// dir has no Go files and must NOT be a finding.
	mustMkdir(t, root, "internal", "gateway")
	mustWrite(t, filepath.Join(root, "internal", "gateway"), "gateway.go", "package gateway\n")
	mustMkdir(t, root, "internal", "orphan")
	mustWrite(t, filepath.Join(root, "internal", "orphan"), "orphan.go", "package orphan\n")
	mustMkdir(t, root, "internal", "docsonly") // no .go -> not a leaf gap
	mustWrite(t, filepath.Join(root, "internal", "docsonly"), "NOTES.md", "not a package\n")

	// One resolving local link, one dead local link, one external URL (unchecked).
	mustWrite(t, root, "README.md", "# readme\n")
	indexMd := "# INDEX\n" +
		"- [Readme](README.md) — resolves on disk.\n" +
		"- [Gone](docs/gone.md) — dead local link.\n" +
		"- [Issues](https://github.com/x/y/issues) — external, not checked.\n"
	mustWrite(t, root, "INDEX.md", indexMd)

	// A main.go whose dispatch switch has a verb in the manifest (index) and one that
	// is NOT (frobnicate). The undeclared-verb detector must flag only frobnicate.
	mainGo := "package main\n\n" +
		"import \"os\"\n\n" +
		"func main() {\n" +
		"\tswitch os.Args[1] {\n" +
		"\tcase \"index\":\n" +
		"\t\tcmdIndex(os.Args[2:])\n" +
		"\tcase \"frobnicate\":\n" +
		"\t\tcmdFrob(os.Args[2:])\n" +
		"\tdefault:\n" +
		"\t\tusage()\n" +
		"\t}\n" +
		"}\n"
	mustMkdir(t, root, "cmd", "fak")
	mustWrite(t, filepath.Join(root, "cmd", "fak"), "main.go", mainGo)
	return root
}

func TestCheckFreshness(t *testing.T) {
	c, err := Load(writeFreshnessRepo(t))
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	// Undeclared leaf: orphan only (gateway declared, docsonly has no Go file).
	gaps := c.UndeclaredLeaves()
	if len(gaps) != 1 || gaps[0] != "orphan" {
		t.Errorf("UndeclaredLeaves = %v, want [orphan]", gaps)
	}

	// Dead doc link: docs/gone.md only (README resolves, the http URL is unchecked).
	dead := c.DeadDocLinks()
	if len(dead) != 1 || dead[0].Path != "docs/gone.md" {
		t.Errorf("DeadDocLinks = %v, want [docs/gone.md]", dead)
	}

	// Undeclared verb: frobnicate only (index is in the manifest).
	verbs := c.UndeclaredVerbs()
	if len(verbs) != 1 || verbs[0] != "frobnicate" {
		t.Errorf("UndeclaredVerbs = %v, want [frobnicate]", verbs)
	}

	// The folded view carries exactly one of each kind, sorted by kind then subject.
	drift := c.CheckFreshness()
	if len(drift) != 3 {
		t.Fatalf("CheckFreshness = %d findings, want 3: %+v", len(drift), drift)
	}
	byKind := map[DriftKind]int{}
	for _, d := range drift {
		byKind[d.Kind]++
		if d.Subject == "" || d.Reason == "" {
			t.Errorf("drift finding missing subject/reason: %+v", d)
		}
	}
	for _, k := range []DriftKind{DriftUndeclaredLeaf, DriftDeadDocLink, DriftUnknownVerb} {
		if byKind[k] != 1 {
			t.Errorf("kind %q count = %d, want 1", k, byKind[k])
		}
	}
}

// TestCheckFreshnessGreen: a tree whose sources all agree yields zero findings — the
// green state the (out-of-lane) gate enforces.
func TestCheckFreshnessGreen(t *testing.T) {
	root := t.TempDir()
	mustWrite(t, root, "dos.toml", "[lanes.trees]\ngateway = [\"internal/gateway/**\"]\n")
	mustMkdir(t, root, "internal", "gateway")
	mustWrite(t, filepath.Join(root, "internal", "gateway"), "g.go", "package gateway\n")
	mustWrite(t, root, "README.md", "# readme\n")
	mustWrite(t, root, "INDEX.md", "# INDEX\n- [Readme](README.md) — fine.\n")
	// No cmd/fak/main.go -> the verb detector contributes nothing (absent source).
	c, err := Load(root)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if drift := c.CheckFreshness(); len(drift) != 0 {
		t.Errorf("clean tree should have no drift, got %+v", drift)
	}
}

// TestOrphanNotes exercises the tree->index converse of DeadDocLinks: a dated note
// under docs/notes/ that INDEX.md never mentions is an orphan; a listed one, a
// non-dated note, and a README are not. It also proves the check reads INDEX.md's raw
// bytes (a note reached via prose, not a markdown link, still counts as listed) so the
// Go view and the Python reciprocal gate agree on the orphan set.
func TestOrphanNotes(t *testing.T) {
	root := t.TempDir()
	// INDEX.md mentions the listed note as a link and the prose note as a bare basename.
	indexMd := "# INDEX\n" +
		"- [Listed](docs/notes/2026-01-02-listed.md) — reachable via a link.\n" +
		"- The 2026-01-03-prose.md note is referenced only in prose.\n"
	mustWrite(t, root, "INDEX.md", indexMd)
	mustMkdir(t, root, "docs", "notes")
	notes := filepath.Join(root, "docs", "notes")
	mustWrite(t, notes, "2026-01-02-listed.md", "# listed\n") // dated + listed -> not orphan
	mustWrite(t, notes, "2026-01-03-prose.md", "# prose\n")   // dated + prose ref -> not orphan
	mustWrite(t, notes, "2026-01-01-orphan.md", "# orphan\n") // dated + unlisted -> ORPHAN
	mustWrite(t, notes, "PLAN-orphan.md", "# plan\n")         // PLAN- + unlisted -> ORPHAN
	mustWrite(t, notes, "README.md", "# readme\n")            // README -> never a dated note
	mustWrite(t, notes, "helper.md", "# helper\n")            // undated -> not a dated note

	c := &Catalog{Root: root}
	got := c.OrphanNotes()
	want := []string{"docs/notes/2026-01-01-orphan.md", "docs/notes/PLAN-orphan.md"}
	if len(got) != len(want) {
		t.Fatalf("OrphanNotes = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("OrphanNotes[%d] = %q, want %q (full: %v)", i, got[i], want[i], got)
		}
	}

	// The finding also surfaces through the folded CheckFreshness view, tagged.
	orphanFindings := 0
	for _, d := range c.CheckFreshness() {
		if d.Kind == DriftOrphanNote {
			orphanFindings++
			if d.Subject == "" || d.Reason == "" {
				t.Errorf("orphan-note finding missing subject/reason: %+v", d)
			}
		}
	}
	if orphanFindings != len(want) {
		t.Errorf("CheckFreshness carried %d orphan-note findings, want %d", orphanFindings, len(want))
	}
}

// TestOrphanNotesNoIndex: a tree with no INDEX.md yields no orphan findings (there is
// no curated map to reconcile against — absence of a claim, not a drift).
func TestOrphanNotesNoIndex(t *testing.T) {
	root := t.TempDir()
	mustMkdir(t, root, "docs", "notes")
	mustWrite(t, filepath.Join(root, "docs", "notes"), "2026-01-01-x.md", "# x\n")
	c := &Catalog{Root: root}
	if got := c.OrphanNotes(); got != nil {
		t.Errorf("no INDEX.md should yield nil orphans, got %v", got)
	}
}

// TestUndeclaredLeavesHonorsLanesSection: a leaf named only in the flat [lanes]
// concurrency arrays (no [lanes.trees] glob of its own) is DECLARED, not drift — the
// same rule internal/hooks.readLaneTaxonomy applies. Only a Go package with a lane in
// NEITHER table is an undeclared-leaf finding. This pins the fix that brought the
// declared-set into exact parity with the authoritative gate (a [lanes]-only leaf was
// previously flagged as undeclared).
func TestUndeclaredLeavesHonorsLanesSection(t *testing.T) {
	root := t.TempDir()
	dosToml := "[lanes]\n" +
		"concurrent = [\"declaredname\", \"gateway\"]\n" +
		"keyword = [\n  \"multiline\",\n]\n" +
		"[lanes.trees]\n" +
		"gateway = [\"internal/gateway/**\"]\n"
	mustWrite(t, root, "dos.toml", dosToml)
	for _, leaf := range []string{"gateway", "declaredname", "multiline", "trulyorphan"} {
		mustMkdir(t, root, "internal", leaf)
		mustWrite(t, filepath.Join(root, "internal", leaf), leaf+".go", "package "+leaf+"\n")
	}
	c, err := Load(root)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	got := c.UndeclaredLeaves()
	if len(got) != 1 || got[0] != "trulyorphan" {
		t.Errorf("UndeclaredLeaves = %v, want [trulyorphan] (declaredname/multiline are named in [lanes])", got)
	}
}

// TestDeadLLMSLinks: llms.txt (the answer-engine index) is checked for dangling local
// .md links the same way INDEX.md is — a live link resolves, a dead one is flagged, and
// an external URL / in-page anchor / absolute path is skipped. This closes the
// LLM-index half of the reciprocal dangling check that DeadDocLinks (INDEX.md) leaves.
func TestDeadLLMSLinks(t *testing.T) {
	root := t.TempDir()
	mustWrite(t, root, "dos.toml", "[lanes.trees]\ngateway = [\"internal/gateway/**\"]\n")
	mustWrite(t, root, "README.md", "# readme\n")
	llms := "# llms\n" +
		"See the [readme](README.md) and the [gone doc](docs/vanished.md).\n" +
		"External [issues](https://github.com/x/y) and an [anchor](#section) are skipped.\n" +
		"An [absolute](/abs/path.md) target is not ours to check.\n"
	mustWrite(t, root, "llms.txt", llms)

	c, err := Load(root)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	dead := c.DeadLLMSLinks()
	if len(dead) != 1 || dead[0] != "docs/vanished.md" {
		t.Fatalf("DeadLLMSLinks = %v, want [docs/vanished.md]", dead)
	}
	// It also surfaces through the folded view, tagged dead-llms-link.
	found := false
	for _, d := range c.CheckFreshness() {
		if d.Kind == DriftDeadLLMSLink && d.Subject == "docs/vanished.md" {
			found = true
		}
	}
	if !found {
		t.Errorf("CheckFreshness did not carry the dead-llms-link finding")
	}
}

// TestDeadLLMSLinksNoFile: a tree with no llms.txt yields no findings (absent source).
func TestDeadLLMSLinksNoFile(t *testing.T) {
	root := t.TempDir()
	mustWrite(t, root, "dos.toml", "[lanes.trees]\ngateway = [\"internal/gateway/**\"]\n")
	c, err := Load(root)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if got := c.DeadLLMSLinks(); got != nil {
		t.Errorf("no llms.txt should yield nil, got %v", got)
	}
}

func TestUndeclaredVerbsNoMainGo(t *testing.T) {
	// A root with no cmd/fak/main.go yields no verb findings (missing source, not drift).
	root := t.TempDir()
	mustWrite(t, root, "dos.toml", "[lanes.trees]\ngateway = [\"internal/gateway/**\"]\n")
	c, _ := Load(root)
	if got := c.UndeclaredVerbs(); got != nil {
		t.Errorf("no main.go should yield nil verbs, got %v", got)
	}
}

// --- tiny local helpers (kept here so the test file is self-contained) ---

func mustWrite(t *testing.T, dir, name, body string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

func mustMkdir(t *testing.T, root string, parts ...string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Join(append([]string{root}, parts...)...), 0o755); err != nil {
		t.Fatal(err)
	}
}

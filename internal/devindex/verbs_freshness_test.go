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

// liveMainVerbs returns the lowercased verb tokens of main.go's top-level switch,
// reusing the same scan the drift detector does, to assert the manifest⊆dispatch.
func liveMainVerbs(t *testing.T, root string) map[string]bool {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(root, "cmd", "fak", "main.go"))
	if err != nil {
		t.Skipf("no cmd/fak/main.go (%v)", err)
	}
	out := map[string]bool{}
	inSwitch := false
	for _, raw := range strings.Split(string(b), "\n") {
		ln := strings.TrimSpace(raw)
		if !inSwitch {
			if strings.HasPrefix(ln, "switch os.Args[1]") {
				inSwitch = true
			}
			continue
		}
		if ln == "}" || strings.HasPrefix(ln, "default:") {
			break
		}
		if !strings.HasPrefix(ln, "case ") || !strings.HasSuffix(ln, ":") {
			continue
		}
		for _, m := range mainCaseRE.FindAllStringSubmatch(ln, -1) {
			out[strings.ToLower(m[1])] = true
		}
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

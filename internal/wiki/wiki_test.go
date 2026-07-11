package wiki

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/devindex"
)

// writeRepo lays down a minimal repo root (dos.toml + the named files) and returns
// its path. Files map is rel-path -> content.
func writeRepo(t *testing.T, dosToml string, files map[string]string) string {
	t.Helper()
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "dos.toml"), []byte(dosToml), 0o644); err != nil {
		t.Fatal(err)
	}
	for rel, body := range files {
		full := filepath.Join(root, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return root
}

const twoLeafDosToml = `
[lanes.trees]
gateway = ["internal/gateway/**"] # the request proxy
devindex = ["internal/devindex/**"] # the self-index
`

func TestStructure_TaxonomyAndLeafPages(t *testing.T) {
	root := writeRepo(t, twoLeafDosToml, map[string]string{
		"README.md":                             "# repo\n",
		"AGENTS.md":                              "agents\n",
		"internal/gateway/gateway.go":            "package gateway\n",
		"internal/devindex/devindex.go":          "package devindex\n",
		"internal/architest/architest_test.go":   "package architest\n",
	})
	cat, err := devindex.Load(root)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	tr := Structure(cat)

	if len(tr.Sections) != 3 {
		t.Fatalf("want 3 sections, got %d", len(tr.Sections))
	}
	wantTitles := []string{"Overview", "System Architecture", "Core Features"}
	for i, w := range wantTitles {
		if tr.Sections[i].Title != w {
			t.Errorf("section %d: want %q, got %q", i, w, tr.Sections[i].Title)
		}
	}

	// Overview cites only docs that exist: README.md + AGENTS.md, not the absent
	// CLAUDE.md / llms.txt / CONTRIBUTING.md.
	ov := tr.Sections[0].Pages[0]
	if got := strings.Join(ov.RelevantFiles, ","); got != "README.md,AGENTS.md" {
		t.Errorf("overview relevant_files = %q, want README.md,AGENTS.md", got)
	}

	// Core Features: one page per leaf, sorted by title, RelevantFiles = trees.
	cf := tr.Sections[2]
	if len(cf.Pages) != 2 {
		t.Fatalf("want 2 leaf pages, got %d", len(cf.Pages))
	}
	if cf.Pages[0].Title != "devindex" || cf.Pages[1].Title != "gateway" {
		t.Errorf("leaf pages not sorted by title: %q, %q", cf.Pages[0].Title, cf.Pages[1].Title)
	}
	gw := cf.Pages[1]
	if gw.ID != "core-features/gateway" {
		t.Errorf("page id = %q", gw.ID)
	}
	if got := strings.Join(gw.RelevantFiles, ","); got != "internal/gateway/**" {
		t.Errorf("gateway relevant_files = %q", got)
	}
	if gw.Summary != "the request proxy" {
		t.Errorf("gateway summary = %q, want the lane comment", gw.Summary)
	}
	if tr.PageCount() != 4 { // 1 overview + 1 arch + 2 leaves
		t.Errorf("PageCount = %d, want 4", tr.PageCount())
	}
}

func TestStructure_Deterministic(t *testing.T) {
	root := writeRepo(t, twoLeafDosToml, nil)
	cat, err := devindex.Load(root)
	if err != nil {
		t.Fatal(err)
	}
	a, _ := json.Marshal(Structure(cat))
	b, _ := json.Marshal(Structure(cat))
	if string(a) != string(b) {
		t.Fatal("Structure is not byte-stable across runs")
	}
}

func TestVerifyCitations_ResolvesAndFlags(t *testing.T) {
	root := writeRepo(t, twoLeafDosToml, map[string]string{
		// 3-line file (no trailing newline on the last line).
		"internal/gateway/gateway.go": "package gateway\nfunc A() {}\nfunc B() {}",
	})
	md := strings.Join([]string{
		"## Gateway",
		"The proxy admits a request. Sources: [internal/gateway/gateway.go:1-3]()",
		"A single line: [internal/gateway/gateway.go:2]()",
		"Out of range: [internal/gateway/gateway.go:9-12]()",
		"Missing file: [internal/gateway/nope.go:1]()",
		"Prose bracket, not a cite: [Step:2]()",
	}, "\n")

	dangs := VerifyCitations(root, []byte(md))
	if len(dangs) != 2 {
		t.Fatalf("want 2 danglers, got %d: %+v", len(dangs), dangs)
	}

	byReason := map[DanglerReason]Dangler{}
	for _, d := range dangs {
		byReason[d.Reason] = d
	}
	oor, ok := byReason[ReasonLineOutOfRange]
	if !ok {
		t.Fatalf("no line-out-of-range dangler in %+v", dangs)
	}
	if oor.Start != 9 || oor.End != 12 || oor.Lines != 3 {
		t.Errorf("out-of-range dangler = %+v, want start9 end12 lines3", oor)
	}
	if oor.Section != "Gateway" {
		t.Errorf("dangler section = %q, want Gateway", oor.Section)
	}
	miss, ok := byReason[ReasonMissingFile]
	if !ok || miss.Path != "internal/gateway/nope.go" {
		t.Errorf("missing-file dangler = %+v", miss)
	}
}

func TestVerifyCitations_NoCitesNoDanglers(t *testing.T) {
	root := writeRepo(t, twoLeafDosToml, nil)
	if d := VerifyCitations(root, []byte("# Title\n\nProse only, no citations.\n")); d != nil {
		t.Errorf("want nil danglers on a cite-free page, got %+v", d)
	}
}

func TestCountLines(t *testing.T) {
	cases := []struct {
		in   string
		want int
	}{
		{"", 0},
		{"one", 1},
		{"one\n", 1},
		{"one\ntwo", 2},
		{"one\ntwo\n", 2},
		{"\n", 1},
	}
	for _, c := range cases {
		if got := countLines([]byte(c.in)); got != c.want {
			t.Errorf("countLines(%q) = %d, want %d", c.in, got, c.want)
		}
	}
}

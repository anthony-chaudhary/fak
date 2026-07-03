package renameconcept

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestVariantsExpandCaseForms(t *testing.T) {
	vs := Variants("dgx-bridge", "slack-bridge")
	got := map[string]string{}
	for _, v := range vs {
		got[v.Old] = v.New
	}
	want := map[string]string{
		"dgx-bridge": "slack-bridge",
		"dgx_bridge": "slack_bridge",
		"DGX_BRIDGE": "SLACK_BRIDGE",
		"DGX-BRIDGE": "SLACK-BRIDGE",
		"dgxbridge":  "slackbridge",
		"DGXBRIDGE":  "SLACKBRIDGE",
		"DgxBridge":  "SlackBridge",
		"dgxBridge":  "slackBridge",
		"Dgxbridge":  "Slackbridge",
	}
	for o, n := range want {
		if got[o] != n {
			t.Errorf("variant %q -> %q, want %q", o, got[o], n)
		}
	}
	// Longest-old-first: replacement must never bite a substring of a longer form.
	for i := 1; i < len(vs); i++ {
		if len(vs[i-1].Old) < len(vs[i].Old) {
			t.Errorf("variants not longest-first: %q before %q", vs[i-1].Old, vs[i].Old)
		}
	}
}

func TestVariantsSingleWordAndHumps(t *testing.T) {
	vs := Variants("dgxbridge", "slackbridge")
	got := map[string]string{}
	for _, v := range vs {
		got[v.Old] = v.New
	}
	for o, n := range map[string]string{
		"dgxbridge": "slackbridge",
		"DGXBRIDGE": "SLACKBRIDGE",
		"Dgxbridge": "Slackbridge",
	} {
		if got[o] != n {
			t.Errorf("variant %q -> %q, want %q", o, got[o], n)
		}
	}
	// Camel-hump input marks the word boundary just like a dash does.
	vs = Variants("DgxBridge", "SlackBridge")
	got = map[string]string{}
	for _, v := range vs {
		got[v.Old] = v.New
	}
	if got["dgx-bridge"] != "slack-bridge" || got["DGX_BRIDGE"] != "SLACK_BRIDGE" {
		t.Errorf("hump-split variants missing: %v", got)
	}
}

// fixture builds the characteristic rename terrain: source, docs, config,
// history, an irregular casing, and a binary artifact.
func fixture(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	files := map[string]string{
		"cmd/dgxbridge/main.go":           "package main\n\n// dgxbridge bridges the control channel.\nconst name = \"dgxbridge\"\n",
		"docs/bridge.md":                  "The DgxBridge concept: run `dgxbridge` (env DGXBRIDGE_TOKEN).\n",
		"docs/notes/BRIDGE-2026-07-01.md": "dated note: dgxbridge shipped today.\n",
		"docs/nightrun/log.jsonl":         "{\"lane\":\"dgxbridge\"}\n",
		"dos.toml":                        "[lanes]\ndgxbridge = [\"cmd/dgxbridge/**\"]\n",
		".gitignore":                      "dgxbridge.exe\n",
		"docs/irregular.md":               "the DGXBridge spelling nobody declared\n",
		"unrelated.md":                    "nothing to see here\n",
		// Tool-state dot-dirs are out of scope; .github is the exception.
		".fak/tmp/mirror/copy.md":     "a scratch mirror mentioning dgxbridge\n",
		".github/workflows/build.yml": "run: go build ./cmd/dgxbridge\n",
	}
	for rel, content := range files {
		abs := filepath.Join(root, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(abs), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(abs, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	// A binary artifact whose NAME carries the concept.
	if err := os.WriteFile(filepath.Join(root, "dgxbridge.exe"), []byte{0x4d, 0x5a, 0x00, 0x01}, 0o644); err != nil {
		t.Fatal(err)
	}
	return root
}

// opts marks the word boundary (dgx-bridge) so the hump forms (DgxBridge)
// expand as variants; a single-word "dgxbridge" input would triage those as
// irregular holdouts instead — see TestVariantsSingleWordAndHumps.
func opts(root string) Options {
	return Options{Root: root, From: "dgx-bridge", To: "slack-bridge"}
}

func siteByPath(sites []Site, path string) (Site, bool) {
	for _, s := range sites {
		if s.Path == path {
			return s, true
		}
	}
	return Site{}, false
}

func TestBuildPlanTriage(t *testing.T) {
	root := fixture(t)
	p, err := BuildPlan(opts(root))
	if err != nil {
		t.Fatal(err)
	}
	if !p.OK || p.Finding != "mechanical_with_holdouts" {
		t.Fatalf("finding = %q ok=%v, want mechanical_with_holdouts", p.Finding, p.OK)
	}

	// Mechanical: source, live doc, config, ignore rule.
	for _, path := range []string{"cmd/dgxbridge/main.go", "docs/bridge.md", "dos.toml", ".gitignore"} {
		if _, ok := siteByPath(p.Mechanical, path); !ok {
			t.Errorf("%s missing from mechanical sites", path)
		}
	}
	// Holdouts: dated note, nightrun ledger, binary artifact, irregular casing.
	for _, path := range []string{"docs/notes/BRIDGE-2026-07-01.md", "docs/nightrun/log.jsonl", "dgxbridge.exe", "docs/irregular.md"} {
		s, ok := siteByPath(p.Holdouts, path)
		if !ok {
			t.Errorf("%s missing from holdout sites", path)
			continue
		}
		if s.Hold == "" {
			t.Errorf("%s holdout carries no reason", path)
		}
	}
	if s, _ := siteByPath(p.Holdouts, "docs/irregular.md"); s.Irregular != 1 || s.Matches != 0 {
		t.Errorf("irregular.md: irregular=%d matches=%d, want 1/0", s.Irregular, s.Matches)
	}
	if _, ok := siteByPath(append(p.Mechanical, p.Holdouts...), "unrelated.md"); ok {
		t.Error("unrelated.md must not be a site")
	}
	if _, ok := siteByPath(append(p.Mechanical, p.Holdouts...), ".fak/tmp/mirror/copy.md"); ok {
		t.Error("dot-dir tool state must be out of scan scope")
	}
	if _, ok := siteByPath(p.Mechanical, ".github/workflows/build.yml"); !ok {
		t.Error(".github must stay in scan scope")
	}

	// Renames: the mechanical dir renames; the binary artifact does NOT.
	renameFrom := map[string]string{}
	for _, r := range p.Renames {
		renameFrom[r.From] = r.To
	}
	if renameFrom["cmd/dgxbridge"] != "cmd/slackbridge" {
		t.Errorf("dir rename missing/wrong: %v", p.Renames)
	}
	if _, ok := renameFrom["dgxbridge.exe"]; ok {
		t.Error("binary artifact must be a regenerate-holdout, not a rename")
	}

	// The commit-paths fold names the mechanical sites and both rename sides.
	cp := strings.Join(p.CommitPaths, "\n")
	for _, want := range []string{"cmd/dgxbridge", "cmd/slackbridge", "dos.toml", "docs/bridge.md"} {
		if !strings.Contains(cp, want) {
			t.Errorf("commit_paths missing %s: %v", want, p.CommitPaths)
		}
	}
}

func TestApplyRewritesMechanicalAndKeepsHistory(t *testing.T) {
	root := fixture(t)
	res, err := Apply(opts(root))
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Errors) != 0 {
		t.Fatalf("apply errors: %v", res.Errors)
	}
	if !res.OK {
		t.Fatalf("apply not OK; residual mechanical = %+v", res.Residual.Mechanical)
	}

	// Source moved and rewrote.
	b, err := os.ReadFile(filepath.Join(root, "cmd", "slackbridge", "main.go"))
	if err != nil {
		t.Fatalf("renamed source missing: %v", err)
	}
	if got := string(b); strings.Contains(got, "dgxbridge") || !strings.Contains(got, "slackbridge") {
		t.Errorf("source not rewritten: %q", got)
	}
	// Multi-case doc rewrote per variant.
	b, _ = os.ReadFile(filepath.Join(root, "docs", "bridge.md"))
	if got := string(b); !strings.Contains(got, "SlackBridge") || !strings.Contains(got, "SLACKBRIDGE_TOKEN") {
		t.Errorf("doc variants not rewritten: %q", got)
	}
	// History and the irregular casing stayed byte-identical.
	for rel, wantSubstr := range map[string]string{
		"docs/notes/BRIDGE-2026-07-01.md": "dgxbridge shipped",
		"docs/nightrun/log.jsonl":         "\"dgxbridge\"",
		"docs/irregular.md":               "DGXBridge",
	} {
		b, _ = os.ReadFile(filepath.Join(root, filepath.FromSlash(rel)))
		if !strings.Contains(string(b), wantSubstr) {
			t.Errorf("%s was rewritten; holdouts must stay intact: %q", rel, string(b))
		}
	}
	// Binary artifact still on disk under its old name (regenerate, not move).
	if _, err := os.Stat(filepath.Join(root, "dgxbridge.exe")); err != nil {
		t.Errorf("binary artifact must not be moved: %v", err)
	}

	// Residual is re-derived: the holdouts (still matching) remain visible.
	if res.Residual == nil {
		t.Fatal("no residual rescan")
	}
	if len(res.Residual.Mechanical) != 0 {
		t.Errorf("mechanical residual after apply: %+v", res.Residual.Mechanical)
	}
	if len(res.Residual.Holdouts) == 0 {
		t.Error("holdouts vanished from the residual; they were not applied and must stay visible")
	}
}

func TestApplyIncludeHistoricalRewritesRecords(t *testing.T) {
	root := fixture(t)
	o := opts(root)
	o.IncludeHistorical = true
	res, err := Apply(o)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Errors) != 0 {
		t.Fatalf("apply errors: %v", res.Errors)
	}
	b, _ := os.ReadFile(filepath.Join(root, "docs", "notes", "BRIDGE-2026-07-01.md"))
	if !strings.Contains(string(b), "slackbridge") {
		t.Errorf("--include-historical did not rewrite the note: %q", string(b))
	}
	b, _ = os.ReadFile(filepath.Join(root, "docs", "nightrun", "log.jsonl"))
	if !strings.Contains(string(b), "slackbridge") {
		t.Errorf("--include-historical did not rewrite the ledger: %q", string(b))
	}
}

func TestBuildPlanConceptNotFound(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "a.md"), []byte("nothing\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	p, err := BuildPlan(Options{Root: root, From: "dgxbridge", To: "slackbridge"})
	if err != nil {
		t.Fatal(err)
	}
	if p.Finding != "concept_not_found" {
		t.Errorf("finding = %q, want concept_not_found", p.Finding)
	}
}

func TestBuildPlanRefusesIdenticalSpelling(t *testing.T) {
	if _, err := BuildPlan(Options{Root: t.TempDir(), From: "same", To: "same"}); err == nil {
		t.Error("identical from/to must refuse")
	}
}

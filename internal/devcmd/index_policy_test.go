package devcmd

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/devindex"
)

func writePolicyFile(t *testing.T, root, name, body string) {
	t.Helper()
	p := filepath.Join(root, filepath.FromSlash(name))
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestBareDevSpellingFindingsScopeClassificationAndAllowlist(t *testing.T) {
	root := t.TempDir()
	files := map[string]string{
		"Makefile":            "x:\n\tfak commit\n\tfak guard\n\tfak unknown\n\tfak dev commit\n",
		"docs/example.md":     "prose says fak sweep\n```sh\nfak sweep\n```\n",
		"docs/archive/old.md": "```sh\nfak commit\n```\n",
		"tools/allowed.ps1":   "fak commit\n",
	}
	var paths []string
	for name, body := range files {
		writePolicyFile(t, root, name, body)
		paths = append(paths, name)
	}
	allow := bareDevAllowlist{exact: map[string]bool{"tools/allowed.ps1": true}}
	got := bareDevSpellingFindings(root, paths, allow)
	if len(got) != 2 {
		t.Fatalf("want Makefile commit and fenced docs sweep, got %+v", got)
	}
	joined := got[0].File + ":" + got[0].Detail + "\n" + got[1].File + ":" + got[1].Detail
	for _, want := range []string{"Makefile", "fak commit", "docs/example.md", "fak sweep"} {
		if !strings.Contains(joined, want) {
			t.Errorf("missing %q in %s", want, joined)
		}
	}
	for _, absent := range []string{"fak guard", "fak unknown", "docs/archive", "tools/allowed"} {
		if strings.Contains(joined, absent) {
			t.Errorf("unexpected %q in %s", absent, joined)
		}
	}
}

func TestVerbTierFindingsDetectsOnlyUntieredDispatch(t *testing.T) {
	root := t.TempDir()
	writePolicyFile(t, root, mainGoFile, `package main
func dispatch() {
 switch os.Args[1] {
 case "guard":
  runGuard()
 case "totally-new-verb":
  if err := run(); err != nil { return }
 }
}
`)
	got, err := verbTierFindings(root)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].Reason != reasonVerbUntiered || !strings.Contains(got[0].Detail, "totally-new-verb") {
		t.Fatalf("unexpected findings: %+v", got)
	}
}

func TestGitTrackedPathsFallsBackToArchiveTree(t *testing.T) {
	root := t.TempDir()
	writePolicyFile(t, root, "Makefile", "all:\n")
	writePolicyFile(t, root, "docs/example.md", "example\n")
	paths, err := gitTrackedPaths(root)
	if err != nil {
		t.Fatal(err)
	}
	got := strings.Join(paths, ",")
	if got != "Makefile,docs/example.md" {
		t.Fatalf("archive paths=%q", got)
	}
}

func TestRunIndexPolicyJSON(t *testing.T) {
	root := t.TempDir()
	writePolicyFile(t, root, mainGoFile, `package main
func dispatch() {
 switch os.Args[1] {
 case "guard":
  runGuard()
 }
}
`)
	writePolicyFile(t, root, "Makefile", "x:\n\tfak commit\n")
	cmd := exec.Command("git", "init", "-q")
	cmd.Dir = root
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("git init: %v: %s", err, out)
	}
	cmd = exec.Command("git", "add", "cmd/fak/main.go", "Makefile")
	cmd.Dir = root
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("git add: %v: %s", err, out)
	}
	var stdout, stderr bytes.Buffer
	if code := runIndexPolicy(&stdout, &stderr, root, nil, true); code != 1 {
		t.Fatalf("want policy finding exit 1, got %d: %s", code, stderr.String())
	}
	var report indexPolicyReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("decode: %v: %s", err, stdout.String())
	}
	if report.Schema != "fak-dev-index-policy/1" || report.OK || len(report.Findings) != 1 || report.Findings[0].Reason != "BARE_DEV_SPELLING" {
		t.Fatalf("unexpected report: %+v", report)
	}
}

func TestRuntimeHooksDoNotImportDevIndex(t *testing.T) {
	root := devindex.FindRoot(".")
	cmd := exec.Command("go", "list", "-f", `{{join .Imports "\n"}}`, "./internal/hooks")
	cmd.Dir = root
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("go list internal/hooks: %v: %s", err, out)
	}
	for _, imp := range strings.Fields(string(out)) {
		if imp == "github.com/anthony-chaudhary/fak/internal/devindex" {
			t.Fatal("runtime internal/hooks imports dev-only internal/devindex; repository policy belongs in fak-dev")
		}
	}
	if len(out) == 0 {
		t.Fatal(fmt.Errorf("go list returned no imports for internal/hooks"))
	}
}

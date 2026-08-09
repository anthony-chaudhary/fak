package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/modver"
)

// commit_modver_advisory_test.go — the named witness for #2495 (version-everything child of #2458):
// `fak commit --preview` grows a modver cross-check line naming the modules whose rev a commit of
// exactly the staged paths would bump. It is ADVISORY (a lens beside the stamp/lane doctor, never an
// exit-code gate) and pure path→module projection (modver.ModulesForPaths — no git, no Snapshot), so
// it stays inside the preview's latency budget.

// TestModulesForPaths_projectsStagedPathsToModules pins the pure projection the advisory rests on:
// internal/<leaf> and cmd/<leaf> paths fold to their module keys (deduped, sorted); docs/ prose folds
// to its page or section module (#2460); a file under no tracked keyspace (root-level, or a docs/
// data file that is not .md) bumps no module and is dropped.
func TestModulesForPaths_projectsStagedPathsToModules(t *testing.T) {
	got := modver.ModulesForPaths([]string{
		"internal/modver/modver.go",
		"internal/modver/trend.go", // same module, deduped to one
		"cmd/fak/commit_preview.go",
		"docs/notes/VERSION-EVERYTHING-SPINE-2026-07-03.md", // docs/ prose -> the docs/notes section
		"docs/nightrun/module-versions.jsonl",               // generated ledger -> no module
		"README.md",                                         // root-level -> no module
	})
	want := []string{"cmd/fak", "docs/notes", "internal/modver"} // sorted, deduped
	if len(got) != len(want) {
		t.Fatalf("ModulesForPaths = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("ModulesForPaths = %v, want %v", got, want)
		}
	}
}

// TestModulesForPaths_noTrackedPathsIsNil confirms an all-untracked path set (or an empty one) bumps
// no module, so the advisory line is omitted rather than printed empty.
func TestModulesForPaths_noTrackedPathsIsNil(t *testing.T) {
	if got := modver.ModulesForPaths(nil); got != nil {
		t.Errorf("empty path set: want nil, got %v", got)
	}
	if got := modver.ModulesForPaths([]string{"README.md", "docs/_config.yml"}); got != nil {
		t.Errorf("untracked-only path set: want nil, got %v", got)
	}
}

// TestRenderModuleAdvisory_line exercises the human render: a non-empty module set prints the single
// "bumps modules: …" line with the modules comma-joined in sort order.
func TestRenderModuleAdvisory_line(t *testing.T) {
	var b bytes.Buffer
	renderModuleAdvisory(&b, []string{"cmd/fak", "internal/modver"})
	if want := "bumps modules: cmd/fak, internal/modver\n"; !strings.Contains(b.String(), want) {
		t.Errorf("render = %q, want it to contain %q", b.String(), want)
	}
}

// TestRenderModuleAdvisory_quietWhenNone confirms the renderer writes NOTHING when no module is
// bumped — a root-only or generated-data commit produces no advisory noise.
func TestRenderModuleAdvisory_quietWhenNone(t *testing.T) {
	var b bytes.Buffer
	renderModuleAdvisory(&b, nil)
	if b.Len() != 0 {
		t.Errorf("want silent render for no bumped modules, got %q", b.String())
	}
}

// TestRunCommitPreview_namesBumpedModules is the CLI-level witness the ticket names (#2495): a
// preview of a commit touching internal/modver + cmd/fak surfaces the "bumps modules:" advisory
// naming both modules, and exits 0 — the advisory never changes the exit code.
func TestRunCommitPreview_namesBumpedModules(t *testing.T) {
	tmp := t.TempDir()
	var out, errb bytes.Buffer
	code := runCommit(&out, &errb, []string{
		"--preview", "--dir", tmp,
		"-m", "feat(modver): add commit --preview module cross-check (#2495) (fak modver)",
		"--path", "internal/modver/modver.go",
		"--path", "cmd/fak/commit_preview.go",
	})
	if code != 0 {
		t.Fatalf("advisory must not block: exit %d (out=%q err=%q)", code, out.String(), errb.String())
	}
	if want := "bumps modules: cmd/fak, internal/modver"; !strings.Contains(out.String(), want) {
		t.Errorf("preview missing advisory %q; got:\n%s", want, out.String())
	}
}

// TestRunCommitPreview_docsPageBumpsItsSection is the CLI-level half of the #2460 witness: now that
// docs/ is a versioned keyspace, a docs-lane prose commit is no longer invisible to the advisory —
// it names the docs/<dir> section the page belongs to, so the rev a doc-freshness pass later reads
// from the ledger is the same key the preview announced. Exit stays 0: the advisory never gates.
func TestRunCommitPreview_docsPageBumpsItsSection(t *testing.T) {
	tmp := t.TempDir()
	var out, errb bytes.Buffer
	code := runCommit(&out, &errb, []string{
		"--preview", "--dir", tmp,
		"-m", "docs(docs): clarify the spine note (#2460) (fak docs)",
		"--path", "docs/notes/VERSION-EVERYTHING-SPINE-2026-07-03.md",
	})
	if code != 0 {
		t.Fatalf("want exit 0, got %d (out=%q err=%q)", code, out.String(), errb.String())
	}
	if want := "bumps modules: docs/notes"; !strings.Contains(out.String(), want) {
		t.Errorf("docs prose commit must bump its section, want %q; got:\n%s", want, out.String())
	}
}

// TestRunCommitPreview_generatedDocsDataBumpsNoModule is the negative fixture that survives #2460:
// docs/ carries generated data beside its prose, and appending to a nightrun ledger must stay quiet.
// This is the property that keeps the module-versions stamp convergent — the stamp writes exactly
// this file, so if it bumped a module every stamp would dirty the next one.
func TestRunCommitPreview_generatedDocsDataBumpsNoModule(t *testing.T) {
	tmp := t.TempDir()
	var out, errb bytes.Buffer
	code := runCommit(&out, &errb, []string{
		"--preview", "--dir", tmp,
		"-m", "docs(modver): stamp the module-versions ledger (#2460) (fak docs)",
		"--path", "docs/nightrun/module-versions.jsonl",
	})
	if code != 0 {
		t.Fatalf("want exit 0, got %d (out=%q err=%q)", code, out.String(), errb.String())
	}
	if strings.Contains(out.String(), "bumps modules:") {
		t.Errorf("a generated docs data file must bump no module; got:\n%s", out.String())
	}
}

// TestRunCommitPreview_jsonCarriesBumpedModules confirms the --json shape exposes the bumped module
// set under bumped_modules for a later metrics consumer, while still exiting 0.
func TestRunCommitPreview_jsonCarriesBumpedModules(t *testing.T) {
	tmp := t.TempDir()
	var out, errb bytes.Buffer
	code := runCommit(&out, &errb, []string{
		"--preview", "--json", "--dir", tmp,
		"-m", "feat(modver): add commit --preview module cross-check (#2495) (fak modver)",
		"--path", "internal/modver/modver.go",
		"--path", "cmd/fak/commit_preview.go",
	})
	if code != 0 {
		t.Fatalf("want exit 0, got %d (err=%q)", code, errb.String())
	}
	var got struct {
		BumpedModules []string `json:"bumped_modules"`
	}
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("preview --json invalid: %v\n%s", err, out.String())
	}
	want := []string{"cmd/fak", "internal/modver"}
	if len(got.BumpedModules) != len(want) {
		t.Fatalf("bumped_modules = %v, want %v", got.BumpedModules, want)
	}
	for i := range want {
		if got.BumpedModules[i] != want[i] {
			t.Fatalf("bumped_modules = %v, want %v", got.BumpedModules, want)
		}
	}
}

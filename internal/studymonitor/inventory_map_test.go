package studymonitor

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestBuildInventoryMapClassifiesSourceSurface(t *testing.T) {
	root := t.TempDir()
	writeInventoryFixture(t, root, "README.md", "# demo\n")
	writeInventoryFixture(t, root, "docs/architecture.md", "architecture\n")
	writeInventoryFixture(t, root, "cmd/app/main.go", "package main\nfunc main() {}\n")
	writeInventoryFixture(t, root, "internal/app/app_test.go", "package app\n")
	writeInventoryFixture(t, root, "CHANGELOG.md", "## changes\n")
	writeInventoryFixture(t, root, "ROADMAP.md", "next\n")
	writeInventoryFixture(t, root, "LICENSE", "MIT\n")
	writeInventoryFixture(t, root, ".github/ISSUE_TEMPLATE/bug.md", "bug\n")
	writeInventoryFixture(t, root, "node_modules/pkg/index.js", "ignored\n")

	report, err := BuildInventoryMap(root, InventoryMapOptions{
		Repository:      "owner/repo",
		URL:             "https://github.com/owner/repo",
		IndexedRevision: "abc123",
		ObservedAt:      "2026-08-25T00:00:00Z",
	})
	if err != nil {
		t.Fatal(err)
	}
	if report.Schema != InventoryMapSchema || report.Repository != "owner/repo" || report.IndexedRevision != "abc123" {
		t.Fatalf("identity fields = %+v", report)
	}
	if report.Totals.RuntimeFiles != 1 || report.Totals.TestFiles != 1 || report.Totals.DocsFiles < 2 {
		t.Fatalf("totals = %+v, want runtime/test/docs classification", report.Totals)
	}
	if !containsString(report.SkippedDirs, "node_modules") {
		t.Fatalf("skipped dirs = %v, want node_modules", report.SkippedDirs)
	}
	assertClassStatus(t, report, "readme_docs", InventoryClassCovered)
	assertClassStatus(t, report, "architecture_design", InventoryClassCovered)
	assertClassStatus(t, report, "runtime_source", InventoryClassCovered)
	assertClassStatus(t, report, "tests_fixtures", InventoryClassCovered)
	assertClassStatus(t, report, "history_changelog_releases", InventoryClassCovered)
	assertClassStatus(t, report, "roadmap_todos", InventoryClassCovered)
	assertClassStatus(t, report, "license_provenance", InventoryClassCovered)
	assertClassStatus(t, report, "open_closed_issues_prs_discussions", InventoryClassPartial)
	assertClassStatus(t, report, "fak_selfquery_witness", InventoryClassExternalRequired)
	assertClassStatus(t, report, "completeness_critic", InventoryClassCovered)
	if !strings.Contains(report.CompletenessNote, "still requires non-tree study artifacts") {
		t.Fatalf("completeness note = %q, want non-tree follow-up", report.CompletenessNote)
	}
	cmd := findSubsystem(t, report, "cmd")
	if cmd.RuntimeFiles != 1 || len(cmd.ExamplePaths) == 0 {
		t.Fatalf("cmd subsystem = %+v, want runtime example", cmd)
	}
}

func TestBuildInventoryMapClassifiesTestdataWithoutFixturePrefixFalsePositive(t *testing.T) {
	root := t.TempDir()
	for _, rel := range []string{
		"fixturedb/client.go",
		"pkg/testdata/input.json",
		"pkg/fixture/input.json",
		"pkg/fixtures/input.json",
		"pkg/app_test.go",
		"pkg/test_client.py",
		"pkg/client.test.js",
		"pkg/client.spec.ts",
	} {
		writeInventoryFixture(t, root, rel, "fixture\n")
	}

	report, err := BuildInventoryMap(root, InventoryMapOptions{
		Repository:      "owner/repo",
		IndexedRevision: "abc123",
	})
	if err != nil {
		t.Fatal(err)
	}
	if report.Totals.RuntimeFiles != 1 || report.Totals.TestFiles != 7 {
		t.Fatalf("totals = %+v, want fixturedb runtime and exact test-data components plus test filenames classified as tests", report.Totals)
	}
	fixtureDB := findSubsystem(t, report, "fixturedb")
	if fixtureDB.RuntimeFiles != 1 || fixtureDB.TestFiles != 0 {
		t.Fatalf("fixturedb subsystem = %+v, want runtime-only classification", fixtureDB)
	}
	for _, path := range []string{
		"pkg/testdata/input.json",
		"pkg/fixture/input.json",
		"pkg/fixtures/input.json",
		"pkg/app_test.go",
		"pkg/test_client.py",
		"pkg/client.test.js",
		"pkg/client.spec.ts",
	} {
		class := classifyInventoryFile(path)
		if !class.Test || class.Runtime {
			t.Fatalf("classification for %s = %+v, want test-only", path, class)
		}
	}
}

func writeInventoryFixture(t *testing.T, root, rel, text string) {
	t.Helper()
	path := filepath.Join(root, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(text), 0o644); err != nil {
		t.Fatal(err)
	}
}

func assertClassStatus(t *testing.T, report InventoryMap, class, want string) {
	t.Helper()
	for _, row := range report.SourceClasses {
		if row.Class == class {
			if row.Status != want {
				t.Fatalf("class %s status = %s, want %s; row=%+v", class, row.Status, want, row)
			}
			return
		}
	}
	t.Fatalf("class %s not found in %+v", class, report.SourceClasses)
}

func findSubsystem(t *testing.T, report InventoryMap, path string) InventorySubsystem {
	t.Helper()
	for _, row := range report.Subsystems {
		if row.Path == path {
			return row
		}
	}
	t.Fatalf("subsystem %s not found in %+v", path, report.Subsystems)
	return InventorySubsystem{}
}

func containsString(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}

package studymonitor

import (
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"testing"
)

func TestImportTrackedDryRunAndLiveReconcileWithLineage(t *testing.T) {
	repo := t.TempDir()
	writeImportFixture(t, repo, "docs/research/clear.md", "# Clear Study\nsource: https://example.test/repo\nsource-revision: abc123\nobserved-at: 2026-08-20\n")
	writeImportFixture(t, repo, "docs/research/ambiguous.md", "# Ambiguous Study\nThis prose does not invent lineage.\n")
	writeImportFixture(t, repo, "study-monitor.json", `{
  "schema": "fak-monitored-repositories/1",
  "updated_at": "2026-08-20",
  "methodology": "fixture",
  "repositories": [{
    "repository": "example",
    "url": "https://example.test/monitor",
    "status": "watch",
    "priority": 1,
    "why": "fixture",
    "last_checked": "2026-08-19",
    "checked_revision": "def456",
    "stars_at_check": 1,
    "last_push_at_check": "2026-08-18"
  }]
}`)
	gitImportFixtures(t, repo)

	dry, err := ImportTracked(repo, "", true)
	if err != nil {
		t.Fatal(err)
	}
	if dry.Attempted != 3 || dry.Eligible != 2 || dry.Held != 1 || dry.Imported != 0 || dry.Rejected != 0 {
		t.Fatalf("dry-run totals do not reconcile: %+v", dry)
	}
	if dry.Eligible+dry.Imported+dry.Held+dry.Rejected != dry.Attempted {
		t.Fatalf("statuses do not reconcile: %+v", dry)
	}
	if got := dry.Entries[0]; got.Path != "docs/research/ambiguous.md" || got.Status != ImportHeld || got.Reason == "" {
		t.Fatalf("ambiguous prose was not held: %+v", got)
	}

	store := filepath.Join(t.TempDir(), "store")
	first, err := ImportTracked(repo, store, false)
	if err != nil {
		t.Fatal(err)
	}
	second, err := ImportTracked(repo, store, false)
	if err != nil {
		t.Fatal(err)
	}
	if first.Attempted != 3 || first.Imported != 2 || first.Held != 1 || first.Imported+first.Held+first.Rejected != first.Attempted {
		t.Fatalf("live totals do not reconcile: %+v", first)
	}
	if !reflect.DeepEqual(first, second) {
		t.Fatalf("live import is not idempotent\nfirst: %+v\nsecond: %+v", first, second)
	}
	files, err := os.ReadDir(store)
	if err != nil {
		t.Fatal(err)
	}
	if len(files) != 2 {
		t.Fatalf("want two immutable records, got %d", len(files))
	}
	var sawProse, sawRegistry bool
	for _, entry := range first.Entries {
		for _, record := range entry.Records {
			switch record.Kind {
			case "prose":
				sawProse = record.Lineage.Path == "docs/research/clear.md" && record.Lineage.Date == "2026-08-20" && record.Lineage.Revision == "abc123"
			case "monitor-registry":
				sawRegistry = record.Lineage.Path == "study-monitor.json" && record.Lineage.Date == "2026-08-19" && record.Lineage.Revision == "def456"
			}
		}
	}
	if !sawProse || !sawRegistry {
		t.Fatalf("lineage not preserved: %+v", first.Entries)
	}
}

func TestImportTrackedRejectsMalformedRegistryAndIgnoresUntracked(t *testing.T) {
	repo := t.TempDir()
	writeImportFixture(t, repo, "study-monitor.json", `{not json}`)
	gitImportFixtures(t, repo)
	writeImportFixture(t, repo, "docs/research/untracked.md", "# Not Eligible\n")
	ledger, err := ImportTracked(repo, "", true)
	if err != nil {
		t.Fatal(err)
	}
	if ledger.Attempted != 1 || ledger.Rejected != 1 || ledger.Entries[0].Reason == "" {
		t.Fatalf("unexpected rejection ledger: %+v", ledger)
	}
}

func writeImportFixture(t *testing.T, root, relative, contents string) {
	t.Helper()
	path := filepath.Join(root, filepath.FromSlash(relative))
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(contents), 0o644); err != nil {
		t.Fatal(err)
	}
}

func gitImportFixtures(t *testing.T, root string) {
	t.Helper()
	for _, args := range [][]string{{"init", "-q"}, {"add", "--", "."}} {
		cmd := exec.Command("git", args...)
		cmd.Dir = root
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v: %s", args, err, out)
		}
	}
}

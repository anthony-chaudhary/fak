package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/studymonitor"
)

func TestStudyMonitorReadsRegistry(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "repos.json")
	data := `{"schema":"fak-monitored-repositories/1","updated_at":"2026-08-14","methodology":"ranked","repositories":[{"repository":"owner/repo","url":"https://github.com/owner/repo","status":"candidate","priority":1,"why":"fresh seam","last_checked":"2026-08-01","checked_revision":"abcdef1234567890","stars_at_check":42,"last_push_at_check":"2026-08-13T00:00:00Z"}]}`
	if err := os.WriteFile(path, []byte(data), 0o600); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	code := runStudyMonitor(&stdout, &stderr, []string{"--registry", path, "--as-of", "2026-08-14", "--due-days", "7"})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	if got := stdout.String(); !strings.Contains(got, "owner/repo status=candidate checked=2026-08-01 age_days=13 due=true") {
		t.Fatalf("unexpected output:\n%s", got)
	}
}

func TestStudyMonitorRejectsMalformedRegistry(t *testing.T) {
	path := filepath.Join(t.TempDir(), "repos.json")
	if err := os.WriteFile(path, []byte(`{"schema":"wrong"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	if code := runStudyMonitor(&stdout, &stderr, []string{"--registry", path}); code != 1 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	if !strings.Contains(stderr.String(), "schema must be") {
		t.Fatalf("unexpected error: %s", stderr.String())
	}
}

func TestStudyMonitorInventoryCheckBlocksShallowCandidate(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "repos.json")
	data := `{"schema":"fak-monitored-repositories/1","updated_at":"2026-08-14","methodology":"ranked","repositories":[{"repository":"owner/repo","url":"https://github.com/owner/repo","status":"candidate","priority":1,"why":"fresh seam","last_checked":"2026-08-01","checked_revision":"abcdef1234567890","stars_at_check":42,"last_push_at_check":"2026-08-13T00:00:00Z"}]}`
	if err := os.WriteFile(path, []byte(data), 0o600); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	code := runStudyMonitor(&stdout, &stderr, []string{"--registry", path, "--inventory-check"})
	if code != 1 {
		t.Fatalf("code=%d stdout=%s stderr=%s", code, stdout.String(), stderr.String())
	}
	out := stdout.String()
	for _, want := range []string{"STUDY_INVENTORY ok=false blockers=1", "owner/repo mode=exhaustive ready=false", "reason: missing inventory map_path"} {
		if !strings.Contains(out, want) {
			t.Fatalf("inventory output missing %q:\n%s", want, out)
		}
	}
}

func TestStudyMonitorInventoryCheckAcceptsMachineMap(t *testing.T) {
	dir := t.TempDir()
	mapPath := filepath.Join(dir, "owner-repo.json")
	mapData, err := json.Marshal(studymonitor.InventoryMap{
		Schema:          studymonitor.InventoryMapSchema,
		Repository:      "owner/repo",
		IndexedRevision: "abcdef1234567890",
		Totals:          studymonitor.InventoryMapTotals{Files: 1, RuntimeFiles: 1},
		SourceClasses: []studymonitor.InventoryClassStatus{
			{Class: "readme_docs", Status: studymonitor.InventoryClassCovered},
			{Class: "architecture_design", Status: studymonitor.InventoryClassCheckedAbsent},
			{Class: "runtime_source", Status: studymonitor.InventoryClassCovered},
			{Class: "tests_fixtures", Status: studymonitor.InventoryClassCheckedAbsent},
			{Class: "history_changelog_releases", Status: studymonitor.InventoryClassCheckedAbsent},
			{Class: "open_closed_issues_prs_discussions", Status: studymonitor.InventoryClassExternalRequired},
			{Class: "roadmap_todos", Status: studymonitor.InventoryClassCheckedAbsent},
			{Class: "license_provenance", Status: studymonitor.InventoryClassCovered},
			{Class: "fak_selfquery_witness", Status: studymonitor.InventoryClassExternalRequired},
			{Class: "candidate_matrix", Status: studymonitor.InventoryClassExternalRequired},
			{Class: "completeness_critic", Status: studymonitor.InventoryClassCovered},
			{Class: "issue_tracking", Status: studymonitor.InventoryClassExternalRequired},
		},
		Subsystems:       []studymonitor.InventorySubsystem{{Path: "cmd", Files: 1, RuntimeFiles: 1}},
		CompletenessNote: "all local source classes checked; non-tree classes require explicit evidence",
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(mapPath, mapData, 0o600); err != nil {
		t.Fatal(err)
	}
	registryPath := filepath.Join(dir, "repos.json")
	registry := studymonitor.Registry{Schema: studymonitor.Schema, UpdatedAt: "2026-08-14", Methodology: "ranked", Repositories: []studymonitor.Repository{
		{
			Repository:      "owner/repo",
			URL:             "https://github.com/owner/repo",
			Status:          "candidate",
			Priority:        1,
			Why:             "fresh seam",
			LastChecked:     "2026-08-01",
			CheckedRevision: "abcdef1234567890",
			StarsAtCheck:    42,
			LastPushAtCheck: "2026-08-13T00:00:00Z",
			Inventory: &studymonitor.InventoryContract{
				MapPath:         mapPath,
				IndexedRevision: "abcdef1234567890",
				SourceEvidence: []studymonitor.InventorySourceEvidence{
					{Class: "open_closed_issues_prs_discussions", Evidence: []string{"gh:owner/repo/issues-and-prs-export.json"}},
					{Class: "fak_selfquery_witness", Evidence: []string{"docs/notes/CONCEPT-STUDY-OWNER-REPO.md#self-query"}},
					{Class: "candidate_matrix", Evidence: []string{"docs/notes/CONCEPT-STUDY-OWNER-REPO.md#candidate-matrix"}},
					{Class: "issue_tracking", Evidence: []string{"https://github.com/anthony-chaudhary/fak/issues/1"}},
				},
				SubsystemCount:     1,
				CompletenessCritic: "all source classes checked",
			},
		},
	}}
	registryData, err := json.Marshal(registry)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(registryPath, registryData, 0o600); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	code := runStudyMonitor(&stdout, &stderr, []string{"--registry", registryPath, "--inventory-check", "--json"})
	if code != 0 {
		t.Fatalf("code=%d stdout=%s stderr=%s", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stdout.String(), `"ok": true`) {
		t.Fatalf("stdout = %s, want ok true", stdout.String())
	}
}

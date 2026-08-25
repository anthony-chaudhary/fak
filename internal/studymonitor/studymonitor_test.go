package studymonitor

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestRegistryValidationRejectsDuplicate(t *testing.T) {
	r := Registry{Schema: Schema, Methodology: "ranked", Repositories: []Repository{
		{Repository: "owner/repo", URL: "https://github.com/owner/repo", Status: "candidate", Priority: 1, Why: "useful", LastChecked: "2026-08-14", CheckedRevision: "abc"},
		{Repository: "OWNER/REPO", URL: "https://github.com/OWNER/REPO", Status: "watch", Priority: 2, Why: "duplicate", LastChecked: "2026-08-14", CheckedRevision: "def"},
	}}
	if err := r.Validate(); err == nil || !strings.Contains(err.Error(), "duplicate repository") {
		t.Fatalf("Validate() error = %v, want duplicate repository", err)
	}
}

func TestRegistryValidationRejectsUnsupportedInventoryMode(t *testing.T) {
	r := Registry{Schema: Schema, Methodology: "ranked", Repositories: []Repository{
		{Repository: "owner/repo", URL: "https://github.com/owner/repo", Status: "candidate", Priority: 1, Why: "useful", LastChecked: "2026-08-14", CheckedRevision: "abc", Inventory: &InventoryContract{Mode: "skim"}},
	}}
	if err := r.Validate(); err == nil || !strings.Contains(err.Error(), "unsupported inventory mode") {
		t.Fatalf("Validate() error = %v, want unsupported inventory mode", err)
	}
}

func TestBuildReportSortsAndRenderMarksDue(t *testing.T) {
	r := Registry{Schema: Schema, Methodology: "ranked", Repositories: []Repository{
		{Repository: "owner/later", URL: "https://example/later", Status: "watch", Priority: 2, Why: "later", LastChecked: "2026-08-13", CheckedRevision: "123456789012345", StarsAtCheck: 2},
		{Repository: "owner/first", URL: "https://example/first", Status: "candidate", Priority: 1, Why: "first", LastChecked: "2026-08-01", CheckedRevision: "abcdef", StarsAtCheck: 3},
	}}
	report := BuildReport("registry.json", r, time.Date(2026, 8, 14, 0, 0, 0, 0, time.UTC), 7)
	if got := report.Repositories[0].Repository; got != "owner/first" {
		t.Fatalf("first repository = %q", got)
	}
	var out bytes.Buffer
	RenderHuman(&out, report)
	text := out.String()
	if !strings.Contains(text, "owner/first status=candidate checked=2026-08-01 age_days=13 due=true") {
		t.Fatalf("render missing due witness:\n%s", text)
	}
	if !strings.Contains(text, "owner/later status=watch checked=2026-08-13 age_days=1 due=false") {
		t.Fatalf("render missing fresh witness:\n%s", text)
	}
}

func TestInventoryReportDefaultsCandidateToExhaustiveAndBlocksMissingMap(t *testing.T) {
	r := Registry{Schema: Schema, Methodology: "ranked", Repositories: []Repository{
		{Repository: "owner/repo", URL: "https://example/repo", Status: "candidate", Priority: 1, Why: "fresh", LastChecked: "2026-08-14", CheckedRevision: "abc"},
	}}
	report := BuildInventoryReport(r)
	if report.OK || report.Blockers != 1 {
		t.Fatalf("inventory report = %+v, want one blocker", report)
	}
	row := report.Repositories[0]
	if row.Mode != InventoryModeExhaustive || row.Ready {
		t.Fatalf("row = %+v, want exhaustive and not ready", row)
	}
	for _, want := range []string{"missing inventory map_path", "missing source classes: readme_docs"} {
		if !strings.Contains(strings.Join(row.Reasons, "\n"), want) {
			t.Fatalf("reasons %v missing %q", row.Reasons, want)
		}
	}
}

func TestInventoryReportAcceptsCompleteExhaustiveContract(t *testing.T) {
	r := Registry{Schema: Schema, Methodology: "ranked", Repositories: []Repository{
		{
			Repository:      "owner/repo",
			URL:             "https://example/repo",
			Status:          "candidate",
			Priority:        1,
			Why:             "fresh",
			LastChecked:     "2026-08-14",
			CheckedRevision: "abc",
			Inventory: &InventoryContract{
				MapPath:            "docs/research/repo-inventory.md",
				IndexedRevision:    "abc",
				SourceClasses:      append([]string(nil), RequiredInventorySourceClasses...),
				SubsystemCount:     9,
				CandidateCount:     0,
				FiledIssueCount:    0,
				CompletenessCritic: "all load-bearing directories and source classes opened; no material omissions",
			},
		},
	}}
	report := BuildInventoryReport(r)
	if !report.OK || report.Blockers != 0 {
		t.Fatalf("inventory report = %+v, want ok", report)
	}
	row := report.Repositories[0]
	if !row.Ready || len(row.MissingSourceClasses) != 0 || row.MapPath == "" {
		t.Fatalf("row = %+v, want ready with map path", row)
	}
}

func TestInventoryReportWithMapFilesValidatesMachineMap(t *testing.T) {
	dir := t.TempDir()
	mapPath := filepath.Join(dir, "owner-repo.json")
	var mapBytes bytes.Buffer
	if err := WriteInventoryMapJSON(&mapBytes, InventoryMap{
		Schema:          InventoryMapSchema,
		Repository:      "owner/repo",
		IndexedRevision: "abc",
		Subsystems:      []InventorySubsystem{{Path: "cmd"}},
	}); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(mapPath, mapBytes.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
	r := Registry{Schema: Schema, Methodology: "ranked", Repositories: []Repository{
		{
			Repository:      "owner/repo",
			URL:             "https://example/repo",
			Status:          "candidate",
			Priority:        1,
			Why:             "fresh",
			LastChecked:     "2026-08-14",
			CheckedRevision: "abc",
			Inventory: &InventoryContract{
				MapPath:            mapPath,
				IndexedRevision:    "abc",
				SourceClasses:      append([]string(nil), RequiredInventorySourceClasses...),
				SubsystemCount:     1,
				CompletenessCritic: "all local and non-tree classes complete",
			},
		},
	}}
	report := BuildInventoryReportWithMapFiles(r, ".")
	if !report.OK {
		t.Fatalf("report = %+v, want map-backed ok", report)
	}
}

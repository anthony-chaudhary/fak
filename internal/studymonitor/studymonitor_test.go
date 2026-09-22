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

// TestRegistryValidationAcceptsScopedAndRefreshInventoryModes pins the two
// inventory modes the monitored-repository registry actually carries. Before
// this the public copy accepted only standard|exhaustive, so a registry row
// using "scoped" (a partial first pass) or "refresh" (a re-check of an
// exhaustive study) failed Validate and study-monitor exited before reporting.
func TestRegistryValidationAcceptsScopedAndRefreshInventoryModes(t *testing.T) {
	for _, mode := range []string{InventoryModeScoped, InventoryModeRefresh} {
		r := Registry{Schema: Schema, Methodology: "ranked", Repositories: []Repository{
			{Repository: "owner/repo", URL: "https://github.com/owner/repo", Status: "studied", Priority: 1, Why: "rescan", LastChecked: "2026-08-14", CheckedRevision: "abc", Inventory: &InventoryContract{Mode: mode}},
		}}
		if err := r.Validate(); err != nil {
			t.Fatalf("Validate() error = %v, want nil for %q mode", err, mode)
		}
		if got := effectiveInventoryMode(r.Repositories[0]); got != mode {
			t.Fatalf("effectiveInventoryMode() = %q, want %q", got, mode)
		}
	}
	if !isFullInventoryMode(InventoryModeRefresh) {
		t.Fatalf("isFullInventoryMode(%q) = false, want true", InventoryModeRefresh)
	}
	if isFullInventoryMode(InventoryModeScoped) {
		t.Fatalf("isFullInventoryMode(%q) = true, want false", InventoryModeScoped)
	}
	if isFullInventoryMode(InventoryModeStandard) {
		t.Fatalf("isFullInventoryMode(%q) = true, want false", InventoryModeStandard)
	}
	if !isFullInventoryMode(InventoryModeExhaustive) {
		t.Fatalf("isFullInventoryMode(%q) = false, want true", InventoryModeExhaustive)
	}
}

// TestValidateStudyForgeReceiptFileTreatsRefreshAsFullMode pins that a refresh
// row carries the same full inventory contract as an exhaustive first study,
// so its forge receipt is validated (and fails loudly on a missing file) while
// a scoped row skips the full-contract receipt check entirely.
func TestValidateStudyForgeReceiptFileTreatsRefreshAsFullMode(t *testing.T) {
	repoRoot := t.TempDir()

	refresh := &InventoryRow{Mode: InventoryModeRefresh, ForgeReceiptPath: "does-not-exist.json"}
	if validateStudyForgeReceiptFile(refresh, Repository{}, repoRoot) {
		t.Fatal("validateStudyForgeReceiptFile() = true for refresh row, want false")
	}
	if len(refresh.Reasons) == 0 {
		t.Fatal("refresh row reasons are empty, want a readable-JSON failure reason")
	}

	scoped := &InventoryRow{Mode: InventoryModeScoped, ForgeReceiptPath: "does-not-exist.json"}
	if validateStudyForgeReceiptFile(scoped, Repository{}, repoRoot) {
		t.Fatal("validateStudyForgeReceiptFile() = true for scoped row, want false")
	}
	if len(scoped.Reasons) != 0 {
		t.Fatalf("scoped row reasons = %v, want none added", scoped.Reasons)
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

func TestInventoryReportCarriesRefreshIssueForStudiedRow(t *testing.T) {
	registry := Registry{Repositories: []Repository{{
		Repository:   "example/studied",
		Status:       "studied",
		RefreshIssue: "#8985",
	}}}

	report := BuildInventoryReport(registry)
	if len(report.Repositories) != 1 {
		t.Fatalf("repositories = %d, want 1", len(report.Repositories))
	}
	row := report.Repositories[0]
	if row.Ready {
		t.Fatal("row unexpectedly ready without inventory metadata")
	}
	if row.RefreshIssue != "#8985" {
		t.Fatalf("refresh_issue = %q, want #8985", row.RefreshIssue)
	}
	for _, reason := range row.Reasons {
		if reason == "missing refresh_issue for incomplete studied row" {
			t.Fatalf("unexpected refresh issue reason: %v", row.Reasons)
		}
	}
}

func TestInventoryReportRequiresRefreshIssueForIncompleteStudiedRow(t *testing.T) {
	report := BuildInventoryReport(Registry{Repositories: []Repository{{
		Repository: "example/studied",
		Status:     "studied",
	}}})

	for _, reason := range report.Repositories[0].Reasons {
		if reason == "missing refresh_issue for incomplete studied row" {
			return
		}
	}
	t.Fatalf("reasons = %v, want missing refresh_issue", report.Repositories[0].Reasons)
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

func TestInventoryReportRejectsUnwitnessedMapDeclaration(t *testing.T) {
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
	if report.OK || report.Blockers != 1 {
		t.Fatalf("inventory report = %+v, want unreadable map blocked", report)
	}
	row := report.Repositories[0]
	if row.Ready || !strings.Contains(strings.Join(row.Reasons, "\n"), "inventory map file is not readable JSON") {
		t.Fatalf("row = %+v, want map file witness failure", row)
	}
}

func TestInventoryReportWithMapFilesValidatesMachineMap(t *testing.T) {
	dir := t.TempDir()
	mapPath := filepath.Join(dir, "owner-repo.json")
	var mapBytes bytes.Buffer
	if err := WriteInventoryMapJSON(&mapBytes, validInventoryMapForTest()); err != nil {
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
				MapPath:         mapPath,
				IndexedRevision: "abc",
				SourceClasses:   append([]string(nil), RequiredInventorySourceClasses...),
				SourceEvidence: []InventorySourceEvidence{
					{Class: "hardware_reproduction", Evidence: []string{"docs/research/HARDWARE-EXPERIMENTS.md#owner-repo-blocked"}, Note: "blocked: required device was unavailable"},
					{Class: "open_closed_issues_prs_discussions", Evidence: []string{
						"gh issue list --state all --repo owner/repo",
						"gh pr list --state all --repo owner/repo",
						"gh api graphql repository discussions",
					}},
					{Class: "fak_selfquery_witness", Evidence: []string{"fak capabilities speculator"}},
					{Class: "candidate_matrix", Evidence: []string{"docs/notes/CONCEPT-STUDY-OWNER-REPO.md#candidate-matrix"}},
					{Class: "issue_tracking", Evidence: []string{"#123"}},
				},
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

func TestInventoryReportWithMapFilesRejectsBareExternalClassClaim(t *testing.T) {
	dir := t.TempDir()
	mapPath := filepath.Join(dir, "owner-repo.json")
	var mapBytes bytes.Buffer
	inventoryMap := validInventoryMapForTest()
	inventoryMap.SourceClasses[5] = InventoryClassStatus{
		Class:    "open_closed_issues_prs_discussions",
		Status:   InventoryClassPartial,
		Evidence: []string{".github/ISSUE_TEMPLATE/bug.md"},
	}
	if err := WriteInventoryMapJSON(&mapBytes, inventoryMap); err != nil {
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
	if report.OK {
		t.Fatalf("report = %+v, want bare external class claim blocked", report)
	}
	reasons := strings.Join(report.Repositories[0].Reasons, "\n")
	for _, want := range []string{
		"source class open_closed_issues_prs_discussions is only partial",
		"source class fak_selfquery_witness requires explicit source_evidence",
		"source class candidate_matrix requires explicit source_evidence",
		"source class issue_tracking requires explicit source_evidence",
	} {
		if !strings.Contains(reasons, want) {
			t.Fatalf("reasons missing %q:\n%s", want, reasons)
		}
	}
}

func TestInventoryReportWithMapFilesRejectsForgedMapEvidence(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*InventoryMap)
		want   string
	}{
		{
			name: "external class claimed absent",
			mutate: func(report *InventoryMap) {
				for i := range report.SourceClasses {
					if report.SourceClasses[i].Class == "fak_selfquery_witness" {
						report.SourceClasses[i].Status = InventoryClassCheckedAbsent
						return
					}
				}
			},
			want: "source class fak_selfquery_witness cannot use status checked_absent",
		},
		{
			name: "covered class missing paths",
			mutate: func(report *InventoryMap) {
				report.SourceClasses[0].Evidence = nil
			},
			want: "source class readme_docs requires local path evidence for status covered",
		},
		{
			name: "contradictory totals",
			mutate: func(report *InventoryMap) {
				report.Totals.RuntimeFiles = 0
			},
			want: "inventory map totals do not match subsystem aggregates",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			mapPath := filepath.Join(dir, "owner-repo.json")
			report := validInventoryMapForTest()
			tt.mutate(&report)
			var mapBytes bytes.Buffer
			if err := WriteInventoryMapJSON(&mapBytes, report); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(mapPath, mapBytes.Bytes(), 0o600); err != nil {
				t.Fatal(err)
			}
			registry := Registry{Schema: Schema, Methodology: "ranked", Repositories: []Repository{{
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
					SubsystemCount:     1,
					CompletenessCritic: "checked",
				},
			}}}
			got := BuildInventoryReportWithMapFiles(registry, ".")
			if got.OK || !strings.Contains(strings.Join(got.Repositories[0].Reasons, "\n"), tt.want) {
				t.Fatalf("report = %+v, want %q", got, tt.want)
			}
		})
	}
}

// TestMachineMapWithoutHardwareDispositionReportsMissingClass pins the rule
// that hardware_reproduction is never satisfiable from repository files: a
// machine map whose hardware_reproduction class is claiming to be covered or
// checked-absent from tree contents is rejected, so the class is reported
// missing until an explicit study disposition or receipt is supplied.
func TestMachineMapWithoutHardwareDispositionReportsMissingClass(t *testing.T) {
	tests := []struct {
		name      string
		withClass bool
		status    string
	}{
		{name: "class absent from map"},
		{name: "class claims covered from tree", withClass: true, status: InventoryClassCovered},
		{name: "class claims checked absent from tree", withClass: true, status: InventoryClassCheckedAbsent},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			inventoryMap := validInventoryMapForTest()
			if !tt.withClass {
				filtered := inventoryMap.SourceClasses[:0]
				for _, row := range inventoryMap.SourceClasses {
					if row.Class != "hardware_reproduction" {
						filtered = append(filtered, row)
					}
				}
				inventoryMap.SourceClasses = filtered
			} else {
				for i := range inventoryMap.SourceClasses {
					if inventoryMap.SourceClasses[i].Class == "hardware_reproduction" {
						inventoryMap.SourceClasses[i].Status = tt.status
						inventoryMap.SourceClasses[i].Evidence = []string{"docs/benchmarks/receipts/apparent-run.json"}
					}
				}
			}
			var mapBytes bytes.Buffer
			if err := WriteInventoryMapJSON(&mapBytes, inventoryMap); err != nil {
				t.Fatal(err)
			}
			mapPath := filepath.Join(dir, "owner-repo.json")
			if err := os.WriteFile(mapPath, mapBytes.Bytes(), 0o600); err != nil {
				t.Fatal(err)
			}
			registry := Registry{Schema: Schema, Methodology: "ranked", Repositories: []Repository{{
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
					CompletenessCritic: "checked",
				},
			}}}
			report := BuildInventoryReportWithMapFiles(registry, ".")
			row := report.Repositories[0]
			if report.OK || row.Ready {
				t.Fatalf("report = %+v, want hardware_reproduction incomplete", report)
			}
			if !containsString(row.MissingSourceClasses, "hardware_reproduction") {
				t.Fatalf("missing source classes = %v, want hardware_reproduction", row.MissingSourceClasses)
			}
			if tt.withClass && !strings.Contains(strings.Join(row.Reasons, "\n"), "source class hardware_reproduction cannot use status "+tt.status) {
				t.Fatalf("reasons = %q, want forged tree-derived status rejected", strings.Join(row.Reasons, "\n"))
			}
		})
	}
}

func TestRegistryValidationRejectsUntraceableOrIncompleteSourceEvidence(t *testing.T) {
	tests := []struct {
		name     string
		evidence InventorySourceEvidence
		want     string
	}{
		{
			name:     "arbitrary assertion",
			evidence: InventorySourceEvidence{Class: "candidate_matrix", Evidence: []string{"done"}},
			want:     "expected a durable path",
		},
		{
			name:     "compound forge class",
			evidence: InventorySourceEvidence{Class: "open_closed_issues_prs_discussions", Evidence: []string{"gh issue list --state all --repo owner/repo"}},
			want:     "missing evidence facets: pull_requests,discussions",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			registry := Registry{Schema: Schema, Methodology: "ranked", Repositories: []Repository{{
				Repository:      "owner/repo",
				URL:             "https://example/repo",
				Status:          "candidate",
				Priority:        1,
				Why:             "fresh",
				LastChecked:     "2026-08-14",
				CheckedRevision: "abc",
				Inventory:       &InventoryContract{SourceEvidence: []InventorySourceEvidence{tt.evidence}},
			}}}
			if err := registry.Validate(); err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("Validate() error = %v, want %q", err, tt.want)
			}
		})
	}
}

func validInventoryMapForTest() InventoryMap {
	return InventoryMap{
		Schema:           InventoryMapSchema,
		Repository:       "owner/repo",
		IndexedRevision:  "abc",
		Totals:           InventoryMapTotals{Files: 1, RuntimeFiles: 1},
		CompletenessNote: "local tree inventory completed",
		SourceClasses: []InventoryClassStatus{
			{Class: "readme_docs", Status: InventoryClassCovered, Evidence: []string{"README.md"}},
			{Class: "architecture_design", Status: InventoryClassCovered, Evidence: []string{"docs/architecture.md"}},
			{Class: "runtime_source", Status: InventoryClassCovered, Evidence: []string{"cmd/main.go"}},
			{Class: "tests_fixtures", Status: InventoryClassCovered, Evidence: []string{"cmd/main_test.go"}},
			{Class: "history_changelog_releases", Status: InventoryClassCheckedAbsent},
			{Class: "open_closed_issues_prs_discussions", Status: InventoryClassExternalRequired},
			{Class: "roadmap_todos", Status: InventoryClassCheckedAbsent},
			{Class: "license_provenance", Status: InventoryClassCovered, Evidence: []string{"LICENSE"}},
			{Class: "hardware_reproduction", Status: InventoryClassExternalRequired},
			{Class: "fak_selfquery_witness", Status: InventoryClassExternalRequired},
			{Class: "candidate_matrix", Status: InventoryClassExternalRequired},
			{Class: "completeness_critic", Status: InventoryClassCovered},
			{Class: "issue_tracking", Status: InventoryClassExternalRequired},
		},
		Subsystems: []InventorySubsystem{{Path: "cmd", Files: 1, RuntimeFiles: 1}},
	}
}

func TestSelfInventoryVerificationCacheContractNamesFailClosedKey(t *testing.T) {
	for _, want := range []string{"immutable-tip", "repository", "manifest", "inventory-schema", "complete verdict", "fail-closed"} {
		if !strings.Contains(SelfInventoryVerificationCacheContract, want) {
			t.Fatalf("cache contract %q missing %q", SelfInventoryVerificationCacheContract, want)
		}
	}
}

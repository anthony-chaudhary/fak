package devcmd

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/studymonitor"
)

func TestStudyImportCommandDryRun(t *testing.T) {
	repo := t.TempDir()
	path := filepath.Join(repo, "docs", "research", "study.md")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("# Study\nsource: https://example.test/repo\nsource-revision: abc\nobserved-at: 2026-08-20\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	for _, args := range [][]string{{"init", "-q"}, {"add", "--", "."}} {
		cmd := exec.Command("git", args...)
		cmd.Dir = repo
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v: %s", args, err, out)
		}
	}
	var stdout, stderr bytes.Buffer
	if code := RunStudyMonitor(&stdout, &stderr, []string{"import", "--repo", repo, "--dry-run"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	var ledger studymonitor.ImportLedger
	if err := json.Unmarshal(stdout.Bytes(), &ledger); err != nil {
		t.Fatal(err)
	}
	if ledger.Attempted != 1 || ledger.Eligible != 1 {
		t.Fatalf("unexpected ledger: %+v", ledger)
	}
}

func TestStudyMonitorReadsRegistry(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "repos.json")
	data := `{"schema":"fak-monitored-repositories/1","updated_at":"2026-08-14","methodology":"ranked","repositories":[{"repository":"owner/repo","url":"https://github.com/owner/repo","status":"candidate","priority":1,"why":"fresh seam","last_checked":"2026-08-01","checked_revision":"abcdef1234567890","stars_at_check":42,"last_push_at_check":"2026-08-13T00:00:00Z"}]}`
	if err := os.WriteFile(path, []byte(data), 0o600); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	code := RunStudyMonitor(&stdout, &stderr, []string{"--registry", path, "--as-of", "2026-08-14", "--due-days", "7"})
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
	if code := RunStudyMonitor(&stdout, &stderr, []string{"--registry", path}); code != 1 {
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
	code := RunStudyMonitor(&stdout, &stderr, []string{"--registry", path, "--inventory-check"})
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
			{Class: "readme_docs", Status: studymonitor.InventoryClassCovered, Evidence: []string{"README.md"}},
			{Class: "architecture_design", Status: studymonitor.InventoryClassCheckedAbsent},
			{Class: "runtime_source", Status: studymonitor.InventoryClassCovered, Evidence: []string{"cmd/main.go"}},
			{Class: "tests_fixtures", Status: studymonitor.InventoryClassCheckedAbsent},
			{Class: "history_changelog_releases", Status: studymonitor.InventoryClassCheckedAbsent},
			{Class: "open_closed_issues_prs_discussions", Status: studymonitor.InventoryClassExternalRequired},
			{Class: "roadmap_todos", Status: studymonitor.InventoryClassCheckedAbsent},
			{Class: "license_provenance", Status: studymonitor.InventoryClassCovered, Evidence: []string{"LICENSE"}},
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
					{Class: "open_closed_issues_prs_discussions", Evidence: []string{
						"gh issue list --state all --repo owner/repo",
						"gh pr list --state all --repo owner/repo",
						"gh api graphql repository discussions",
					}},
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
	code := RunStudyMonitor(&stdout, &stderr, []string{"--registry", registryPath, "--inventory-check", "--json"})
	if code != 0 {
		t.Fatalf("code=%d stdout=%s stderr=%s", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stdout.String(), `"ok": true`) {
		t.Fatalf("stdout = %s, want ok true", stdout.String())
	}
}

func TestStudyMonitorResolvesFromEnv(t *testing.T) {
	workDir := t.TempDir()
	t.Chdir(workDir)

	regDir := t.TempDir()
	regPath := filepath.Join(regDir, "env-repos.json")
	data := `{"schema":"fak-monitored-repositories/1","updated_at":"2026-08-14","methodology":"ranked","repositories":[{"repository":"env/repo","url":"https://github.com/env/repo","status":"candidate","priority":1,"why":"env test","last_checked":"2026-08-01","checked_revision":"abcdef1234567890","stars_at_check":10,"last_push_at_check":"2026-08-13T00:00:00Z"}]}`
	if err := os.WriteFile(regPath, []byte(data), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("FAK_STUDY_REGISTRY", regPath)

	var stdout, stderr bytes.Buffer
	code := RunStudyMonitor(&stdout, &stderr, []string{"--as-of", "2026-08-14", "--due-days", "7"})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "env/repo status=candidate") {
		t.Fatalf("expected env/repo in stdout, got:\n%s", stdout.String())
	}
}

func TestStudyMonitorResolvesFromPrivateSibling(t *testing.T) {
	root := t.TempDir()
	fakDir := filepath.Join(root, "fak")
	privDir := filepath.Join(root, "fak-private")
	regPath := filepath.Join(privDir, "docs", "research", "monitored-repositories.json")
	if err := os.MkdirAll(filepath.Dir(regPath), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(fakDir, 0o755); err != nil {
		t.Fatal(err)
	}
	data := `{"schema":"fak-monitored-repositories/1","updated_at":"2026-08-14","methodology":"ranked","repositories":[{"repository":"private/repo","url":"https://github.com/private/repo","status":"candidate","priority":1,"why":"private test","last_checked":"2026-08-01","checked_revision":"abcdef1234567890","stars_at_check":10,"last_push_at_check":"2026-08-13T00:00:00Z"}]}`
	if err := os.WriteFile(regPath, []byte(data), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Chdir(fakDir)
	t.Setenv("FAK_STUDY_REGISTRY", "")

	var stdout, stderr bytes.Buffer
	code := RunStudyMonitor(&stdout, &stderr, []string{"--as-of", "2026-08-14", "--due-days", "7"})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "private/repo status=candidate") {
		t.Fatalf("expected private/repo in stdout, got:\n%s", stdout.String())
	}
}

func TestStudyMonitorResolvesFromCommonLayout(t *testing.T) {
	root := t.TempDir()
	fakDir := filepath.Join(root, "fak")
	privDir := filepath.Join(root, "fak-private")
	regPath := filepath.Join(fakDir, "docs", "research", "monitored-repositories.json")
	if err := os.MkdirAll(filepath.Dir(regPath), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(privDir, 0o755); err != nil {
		t.Fatal(err)
	}
	data := `{"schema":"fak-monitored-repositories/1","updated_at":"2026-08-14","methodology":"ranked","repositories":[{"repository":"layout/repo","url":"https://github.com/layout/repo","status":"candidate","priority":1,"why":"layout test","last_checked":"2026-08-01","checked_revision":"abcdef1234567890","stars_at_check":10,"last_push_at_check":"2026-08-13T00:00:00Z"}]}`
	if err := os.WriteFile(regPath, []byte(data), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Chdir(privDir)
	t.Setenv("FAK_STUDY_REGISTRY", "")

	var stdout, stderr bytes.Buffer
	code := RunStudyMonitor(&stdout, &stderr, []string{"--as-of", "2026-08-14", "--due-days", "7"})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "layout/repo status=candidate") {
		t.Fatalf("expected layout/repo in stdout, got:\n%s", stdout.String())
	}
}

func TestStudyMonitorResolvesFromModuleRootSubdir(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "go.mod"), []byte("module example.test\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	regPath := filepath.Join(root, "docs", "research", "monitored-repositories.json")
	if err := os.MkdirAll(filepath.Dir(regPath), 0o755); err != nil {
		t.Fatal(err)
	}
	subDir := filepath.Join(root, "cmd", "nested", "sub")
	if err := os.MkdirAll(subDir, 0o755); err != nil {
		t.Fatal(err)
	}
	data := `{"schema":"fak-monitored-repositories/1","updated_at":"2026-08-14","methodology":"ranked","repositories":[{"repository":"modroot/repo","url":"https://github.com/modroot/repo","status":"candidate","priority":1,"why":"modroot test","last_checked":"2026-08-01","checked_revision":"abcdef1234567890","stars_at_check":10,"last_push_at_check":"2026-08-13T00:00:00Z"}]}`
	if err := os.WriteFile(regPath, []byte(data), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Chdir(subDir)
	t.Setenv("FAK_STUDY_REGISTRY", "")

	var stdout, stderr bytes.Buffer
	code := RunStudyMonitor(&stdout, &stderr, []string{"--as-of", "2026-08-14", "--due-days", "7"})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "modroot/repo status=candidate") {
		t.Fatalf("expected modroot/repo in stdout, got:\n%s", stdout.String())
	}
}


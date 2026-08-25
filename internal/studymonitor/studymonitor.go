package studymonitor

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

const Schema = "fak-monitored-repositories/1"

const (
	InventorySchema         = "fak-study-inventory-report/1"
	InventoryModeStandard   = "standard"
	InventoryModeExhaustive = "exhaustive"
)

var RequiredInventorySourceClasses = []string{
	"readme_docs",
	"architecture_design",
	"runtime_source",
	"tests_fixtures",
	"history_changelog_releases",
	"open_closed_issues_prs_discussions",
	"roadmap_todos",
	"license_provenance",
	"fak_selfquery_witness",
	"candidate_matrix",
	"completeness_critic",
	"issue_tracking",
}

type Registry struct {
	Schema       string       `json:"schema"`
	UpdatedAt    string       `json:"updated_at"`
	Methodology  string       `json:"methodology"`
	Repositories []Repository `json:"repositories"`
}

type Repository struct {
	Repository      string             `json:"repository"`
	URL             string             `json:"url"`
	Status          string             `json:"status"`
	Priority        int                `json:"priority"`
	Why             string             `json:"why"`
	LastChecked     string             `json:"last_checked"`
	CheckedRevision string             `json:"checked_revision"`
	StarsAtCheck    int                `json:"stars_at_check"`
	LastPushAtCheck string             `json:"last_push_at_check"`
	StudyNote       string             `json:"study_note,omitempty"`
	Inventory       *InventoryContract `json:"inventory,omitempty"`
}

type InventoryContract struct {
	Mode               string   `json:"mode,omitempty"`
	MapPath            string   `json:"map_path,omitempty"`
	IndexedRevision    string   `json:"indexed_revision,omitempty"`
	SourceClasses      []string `json:"source_classes,omitempty"`
	SubsystemCount     int      `json:"subsystem_count,omitempty"`
	CandidateCount     int      `json:"candidate_count,omitempty"`
	FiledIssueCount    int      `json:"filed_issue_count,omitempty"`
	IssueRefs          []string `json:"issue_refs,omitempty"`
	CompletenessCritic string   `json:"completeness_critic,omitempty"`
}

type InventoryReport struct {
	Schema                string         `json:"schema"`
	RequiredSourceClasses []string       `json:"required_source_classes"`
	OK                    bool           `json:"ok"`
	Blockers              int            `json:"blockers"`
	Repositories          []InventoryRow `json:"repositories"`
}

type InventoryRow struct {
	Repository           string   `json:"repository"`
	Status               string   `json:"status"`
	Mode                 string   `json:"mode"`
	MapPath              string   `json:"map_path,omitempty"`
	IndexedRevision      string   `json:"indexed_revision,omitempty"`
	SubsystemCount       int      `json:"subsystem_count,omitempty"`
	CandidateCount       int      `json:"candidate_count,omitempty"`
	FiledIssueCount      int      `json:"filed_issue_count,omitempty"`
	IssueRefs            []string `json:"issue_refs,omitempty"`
	CompletenessCritic   string   `json:"completeness_critic,omitempty"`
	SourceClasses        []string `json:"source_classes,omitempty"`
	MissingSourceClasses []string `json:"missing_source_classes,omitempty"`
	Ready                bool     `json:"ready"`
	Reasons              []string `json:"reasons,omitempty"`
}

type Report struct {
	Schema       string       `json:"schema"`
	RegistryPath string       `json:"registry_path"`
	AsOf         string       `json:"as_of"`
	DueDays      int          `json:"due_days"`
	Repositories []Repository `json:"repositories"`
}

func Read(path string) (Registry, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Registry{}, err
	}
	var registry Registry
	if err := json.Unmarshal(data, &registry); err != nil {
		return Registry{}, fmt.Errorf("decode registry: %w", err)
	}
	if err := registry.Validate(); err != nil {
		return Registry{}, err
	}
	return registry, nil
}

func (r Registry) Validate() error {
	if r.Schema != Schema {
		return fmt.Errorf("schema must be %q", Schema)
	}
	if strings.TrimSpace(r.Methodology) == "" {
		return fmt.Errorf("methodology is required")
	}
	seen := make(map[string]bool, len(r.Repositories))
	for i, repo := range r.Repositories {
		prefix := fmt.Sprintf("repositories[%d]", i)
		if repo.Repository == "" || repo.URL == "" || repo.Why == "" {
			return fmt.Errorf("%s: repository, url, and why are required", prefix)
		}
		if seen[strings.ToLower(repo.Repository)] {
			return fmt.Errorf("%s: duplicate repository %q", prefix, repo.Repository)
		}
		seen[strings.ToLower(repo.Repository)] = true
		if repo.Priority < 1 {
			return fmt.Errorf("%s: priority must be positive", prefix)
		}
		if repo.Status != "candidate" && repo.Status != "studied" && repo.Status != "watch" && repo.Status != "dismissed" {
			return fmt.Errorf("%s: unsupported status %q", prefix, repo.Status)
		}
		if repo.Inventory != nil {
			mode := strings.TrimSpace(repo.Inventory.Mode)
			if mode != "" && mode != InventoryModeStandard && mode != InventoryModeExhaustive {
				return fmt.Errorf("%s: unsupported inventory mode %q", prefix, repo.Inventory.Mode)
			}
		}
		if _, err := time.Parse("2006-01-02", repo.LastChecked); err != nil {
			return fmt.Errorf("%s: last_checked must be YYYY-MM-DD", prefix)
		}
		if repo.CheckedRevision == "" {
			return fmt.Errorf("%s: checked_revision is required", prefix)
		}
	}
	return nil
}

func BuildReport(path string, registry Registry, now time.Time, dueDays int) Report {
	repositories := append([]Repository(nil), registry.Repositories...)
	sort.Slice(repositories, func(i, j int) bool {
		if repositories[i].Priority != repositories[j].Priority {
			return repositories[i].Priority < repositories[j].Priority
		}
		return repositories[i].Repository < repositories[j].Repository
	})
	return Report{Schema: Schema, RegistryPath: path, AsOf: now.Format("2006-01-02"), DueDays: dueDays, Repositories: repositories}
}

func BuildInventoryReport(registry Registry) InventoryReport {
	return buildInventoryReport(registry, "")
}

func BuildInventoryReportWithMapFiles(registry Registry, repoRoot string) InventoryReport {
	return buildInventoryReport(registry, repoRoot)
}

func buildInventoryReport(registry Registry, repoRoot string) InventoryReport {
	rows := make([]InventoryRow, 0, len(registry.Repositories))
	blockers := 0
	for _, repo := range registry.Repositories {
		row := inventoryRow(repo)
		if repoRoot != "" {
			validateInventoryMapFile(&row, repo, repoRoot)
		}
		if !row.Ready {
			blockers++
		}
		rows = append(rows, row)
	}
	sort.Slice(rows, func(i, j int) bool {
		return rows[i].Repository < rows[j].Repository
	})
	return InventoryReport{
		Schema:                InventorySchema,
		RequiredSourceClasses: append([]string(nil), RequiredInventorySourceClasses...),
		OK:                    blockers == 0,
		Blockers:              blockers,
		Repositories:          rows,
	}
}

func validateInventoryMapFile(row *InventoryRow, repo Repository, repoRoot string) {
	if row.Mode != InventoryModeExhaustive || row.MapPath == "" {
		return
	}
	path := row.MapPath
	if !filepath.IsAbs(path) {
		path = filepath.Join(repoRoot, filepath.FromSlash(path))
	}
	report, err := ReadInventoryMap(path)
	if err != nil {
		addInventoryRowReason(row, "inventory map file is not readable JSON: "+err.Error())
		return
	}
	if report.Repository != repo.Repository {
		addInventoryRowReason(row, "inventory map repository does not match registry row")
	}
	if report.IndexedRevision != repo.CheckedRevision {
		addInventoryRowReason(row, "inventory map indexed_revision does not match checked_revision")
	}
	if row.SubsystemCount > 0 && row.SubsystemCount != len(report.Subsystems) {
		addInventoryRowReason(row, "subsystem_count does not match inventory map subsystem rows")
	}
}

func addInventoryRowReason(row *InventoryRow, reason string) {
	row.Reasons = append(row.Reasons, reason)
	row.Ready = false
}

func inventoryRow(repo Repository) InventoryRow {
	mode := effectiveInventoryMode(repo)
	row := InventoryRow{
		Repository: repo.Repository,
		Status:     repo.Status,
		Mode:       mode,
		Ready:      true,
	}
	if repo.Inventory != nil {
		row.MapPath = strings.TrimSpace(repo.Inventory.MapPath)
		row.IndexedRevision = strings.TrimSpace(repo.Inventory.IndexedRevision)
		row.SubsystemCount = repo.Inventory.SubsystemCount
		row.CandidateCount = repo.Inventory.CandidateCount
		row.FiledIssueCount = repo.Inventory.FiledIssueCount
		row.IssueRefs = append([]string(nil), repo.Inventory.IssueRefs...)
		row.CompletenessCritic = strings.TrimSpace(repo.Inventory.CompletenessCritic)
		row.SourceClasses = normalizedUnique(repo.Inventory.SourceClasses)
	}
	if mode != InventoryModeExhaustive {
		return row
	}
	var reasons []string
	if row.MapPath == "" {
		reasons = append(reasons, "missing inventory map_path")
	}
	if row.IndexedRevision == "" {
		reasons = append(reasons, "missing indexed_revision")
	} else if row.IndexedRevision != repo.CheckedRevision {
		reasons = append(reasons, "indexed_revision does not match checked_revision")
	}
	if row.SubsystemCount <= 0 {
		reasons = append(reasons, "subsystem_count must be positive")
	}
	if row.CompletenessCritic == "" {
		reasons = append(reasons, "missing completeness_critic")
	}
	missing := missingRequiredSourceClasses(row.SourceClasses)
	if len(missing) > 0 {
		row.MissingSourceClasses = missing
		reasons = append(reasons, "missing source classes: "+strings.Join(missing, ","))
	}
	row.Reasons = reasons
	row.Ready = len(reasons) == 0
	return row
}

func effectiveInventoryMode(repo Repository) string {
	if repo.Inventory != nil {
		mode := strings.TrimSpace(repo.Inventory.Mode)
		if mode != "" {
			return mode
		}
	}
	if repo.Status == "candidate" || repo.Status == "studied" {
		return InventoryModeExhaustive
	}
	return InventoryModeStandard
}

func missingRequiredSourceClasses(have []string) []string {
	seen := make(map[string]bool, len(have))
	for _, item := range have {
		seen[item] = true
	}
	var missing []string
	for _, required := range RequiredInventorySourceClasses {
		if !seen[required] {
			missing = append(missing, required)
		}
	}
	return missing
}

func normalizedUnique(values []string) []string {
	seen := make(map[string]bool, len(values))
	var out []string
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value == "" || seen[value] {
			continue
		}
		seen[value] = true
		out = append(out, value)
	}
	sort.Strings(out)
	return out
}

func RenderHuman(w io.Writer, report Report) {
	fmt.Fprintf(w, "MONITORED_REPOSITORIES as_of=%s due_days=%d count=%d\n", report.AsOf, report.DueDays, len(report.Repositories))
	now, _ := time.Parse("2006-01-02", report.AsOf)
	for _, repo := range report.Repositories {
		checked, _ := time.Parse("2006-01-02", repo.LastChecked)
		age := int(now.Sub(checked).Hours() / 24)
		due := age >= report.DueDays
		fmt.Fprintf(w, "%d. %s status=%s checked=%s age_days=%d due=%t stars=%d rev=%s\n   %s\n", repo.Priority, repo.Repository, repo.Status, repo.LastChecked, age, due, repo.StarsAtCheck, shortRevision(repo.CheckedRevision), repo.Why)
	}
}

func RenderInventoryHuman(w io.Writer, report InventoryReport) {
	fmt.Fprintf(w, "STUDY_INVENTORY ok=%t blockers=%d required_classes=%d\n", report.OK, report.Blockers, len(report.RequiredSourceClasses))
	for _, row := range report.Repositories {
		fmt.Fprintf(w, "- %s mode=%s ready=%t map=%s subsystems=%d candidates=%d filed=%d\n",
			row.Repository, row.Mode, row.Ready, emptyDash(row.MapPath), row.SubsystemCount, row.CandidateCount, row.FiledIssueCount)
		for _, reason := range row.Reasons {
			fmt.Fprintf(w, "  reason: %s\n", reason)
		}
	}
}

func WriteJSON(w io.Writer, report Report) error {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(report)
}

func WriteInventoryJSON(w io.Writer, report InventoryReport) error {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(report)
}

func ReadInventoryMap(path string) (InventoryMap, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return InventoryMap{}, err
	}
	var report InventoryMap
	if err := json.Unmarshal(data, &report); err != nil {
		return InventoryMap{}, err
	}
	if report.Schema != InventoryMapSchema {
		return InventoryMap{}, fmt.Errorf("schema must be %q", InventoryMapSchema)
	}
	if strings.TrimSpace(report.Repository) == "" {
		return InventoryMap{}, fmt.Errorf("repository is required")
	}
	if strings.TrimSpace(report.IndexedRevision) == "" {
		return InventoryMap{}, fmt.Errorf("indexed_revision is required")
	}
	return report, nil
}

func shortRevision(revision string) string {
	if len(revision) > 12 {
		return revision[:12]
	}
	return revision
}

func emptyDash(value string) string {
	if strings.TrimSpace(value) == "" {
		return "-"
	}
	return value
}

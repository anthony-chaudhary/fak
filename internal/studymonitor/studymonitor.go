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
	RefreshIssue    string             `json:"refresh_issue,omitempty"`
	Inventory       *InventoryContract `json:"inventory,omitempty"`
}

type InventoryContract struct {
	Mode               string                    `json:"mode,omitempty"`
	MapPath            string                    `json:"map_path,omitempty"`
	ForgeReceiptPath   string                    `json:"forge_receipt_path,omitempty"`
	IndexedRevision    string                    `json:"indexed_revision,omitempty"`
	SourceClasses      []string                  `json:"source_classes,omitempty"`
	SourceEvidence     []InventorySourceEvidence `json:"source_evidence,omitempty"`
	SubsystemCount     int                       `json:"subsystem_count,omitempty"`
	CandidateCount     int                       `json:"candidate_count,omitempty"`
	FiledIssueCount    int                       `json:"filed_issue_count,omitempty"`
	IssueRefs          []string                  `json:"issue_refs,omitempty"`
	CompletenessCritic string                    `json:"completeness_critic,omitempty"`
}

type InventorySourceEvidence struct {
	Class    string   `json:"class"`
	Evidence []string `json:"evidence"`
	Note     string   `json:"note,omitempty"`
}

type InventoryReport struct {
	Schema                string         `json:"schema"`
	RequiredSourceClasses []string       `json:"required_source_classes"`
	OK                    bool           `json:"ok"`
	Blockers              int            `json:"blockers"`
	Repositories          []InventoryRow `json:"repositories"`
}

type InventoryRow struct {
	Repository           string                    `json:"repository"`
	Status               string                    `json:"status"`
	Mode                 string                    `json:"mode"`
	MapPath              string                    `json:"map_path,omitempty"`
	ForgeReceiptPath     string                    `json:"forge_receipt_path,omitempty"`
	IndexedRevision      string                    `json:"indexed_revision,omitempty"`
	SubsystemCount       int                       `json:"subsystem_count,omitempty"`
	CandidateCount       int                       `json:"candidate_count,omitempty"`
	FiledIssueCount      int                       `json:"filed_issue_count,omitempty"`
	IssueRefs            []string                  `json:"issue_refs,omitempty"`
	RefreshIssue         string                    `json:"refresh_issue,omitempty"`
	CompletenessCritic   string                    `json:"completeness_critic,omitempty"`
	SourceClasses        []string                  `json:"source_classes,omitempty"`
	SourceEvidence       []InventorySourceEvidence `json:"source_evidence,omitempty"`
	MissingSourceClasses []string                  `json:"missing_source_classes,omitempty"`
	Ready                bool                      `json:"ready"`
	Reasons              []string                  `json:"reasons,omitempty"`
	forgeReceiptValid    bool
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
			for j, class := range repo.Inventory.SourceClasses {
				if !isRequiredInventorySourceClass(strings.TrimSpace(class)) {
					return fmt.Errorf("%s.inventory.source_classes[%d]: unsupported source class %q", prefix, j, class)
				}
			}
			seenEvidence := map[string]bool{}
			for j, evidence := range repo.Inventory.SourceEvidence {
				if !isRequiredInventorySourceClass(evidence.Class) {
					return fmt.Errorf("%s.inventory.source_evidence[%d]: unsupported source class %q", prefix, j, evidence.Class)
				}
				if seenEvidence[evidence.Class] {
					return fmt.Errorf("%s.inventory.source_evidence[%d]: duplicate source class %q", prefix, j, evidence.Class)
				}
				seenEvidence[evidence.Class] = true
				refs := normalizedUnique(evidence.Evidence)
				if len(refs) == 0 {
					return fmt.Errorf("%s.inventory.source_evidence[%d]: evidence is required", prefix, j)
				}
				for k, ref := range refs {
					if !isTraceableInventoryEvidence(ref) {
						return fmt.Errorf("%s.inventory.source_evidence[%d].evidence[%d]: expected a durable path, URL, issue reference, or replayable command", prefix, j, k)
					}
				}
				if missing := missingInventoryEvidenceFacets(evidence.Class, refs); len(missing) > 0 {
					return fmt.Errorf("%s.inventory.source_evidence[%d]: missing evidence facets: %s", prefix, j, strings.Join(missing, ","))
				}
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
	return buildInventoryReport(registry, ".")
}

func BuildInventoryReportWithMapFiles(registry Registry, repoRoot string) InventoryReport {
	return buildInventoryReport(registry, repoRoot)
}

func buildInventoryReport(registry Registry, repoRoot string) InventoryReport {
	rows := make([]InventoryRow, 0, len(registry.Repositories))
	blockers := 0
	for _, repo := range registry.Repositories {
		row := inventoryBaseRow(repo)
		var inventoryMap *InventoryMap
		if repoRoot != "" {
			inventoryMap = validateInventoryMapFile(&row, repo, repoRoot)
			row.forgeReceiptValid = validateStudyForgeReceiptFile(&row, repo, repoRoot)
		}
		finalizeInventoryRow(&row, repo, inventoryMap)
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

func validateInventoryMapFile(row *InventoryRow, repo Repository, repoRoot string) *InventoryMap {
	if row.Mode != InventoryModeExhaustive || row.MapPath == "" {
		return nil
	}
	path := row.MapPath
	if !filepath.IsAbs(path) {
		path = filepath.Join(repoRoot, filepath.FromSlash(path))
	}
	report, err := ReadInventoryMap(path)
	if err != nil {
		addInventoryRowReason(row, "inventory map file is not readable JSON: "+err.Error())
		return nil
	}
	if report.Repository != repo.Repository {
		addInventoryRowReason(row, "inventory map repository does not match registry row")
	}
	if report.IndexedRevision != repo.CheckedRevision {
		addInventoryRowReason(row, "inventory map indexed_revision does not match checked_revision")
	}
	if report.Totals.Files <= 0 {
		addInventoryRowReason(row, "inventory map totals.files must be positive")
	}
	validateInventoryMapTotals(row, report)
	if strings.TrimSpace(report.CompletenessNote) == "" {
		addInventoryRowReason(row, "inventory map completeness_critic is required")
	}
	validateInventoryMapSourceClasses(row, report)
	if row.SubsystemCount > 0 && row.SubsystemCount != len(report.Subsystems) {
		addInventoryRowReason(row, "subsystem_count does not match inventory map subsystem rows")
	}
	return &report
}

func validateInventoryMapTotals(row *InventoryRow, report InventoryMap) {
	var sum InventoryMapTotals
	for i, subsystem := range report.Subsystems {
		if strings.TrimSpace(subsystem.Path) == "" {
			addInventoryRowReason(row, fmt.Sprintf("inventory map subsystem[%d] path is required", i))
		}
		if subsystem.Files < 0 || subsystem.Directories < 0 || subsystem.Bytes < 0 ||
			subsystem.RuntimeFiles < 0 || subsystem.TestFiles < 0 || subsystem.DocsFiles < 0 || subsystem.TextLines < 0 {
			addInventoryRowReason(row, fmt.Sprintf("inventory map subsystem[%d] has negative totals", i))
		}
		sum.Files += subsystem.Files
		sum.Directories += subsystem.Directories
		sum.Bytes += subsystem.Bytes
		sum.RuntimeFiles += subsystem.RuntimeFiles
		sum.TestFiles += subsystem.TestFiles
		sum.DocsFiles += subsystem.DocsFiles
		sum.TextLines += subsystem.TextLines
	}
	if report.Totals != sum {
		addInventoryRowReason(row, "inventory map totals do not match subsystem aggregates")
	}
}

func validateInventoryMapSourceClasses(row *InventoryRow, report InventoryMap) {
	seen := map[string]bool{}
	for _, class := range report.SourceClasses {
		name := strings.TrimSpace(class.Class)
		if name == "" {
			addInventoryRowReason(row, "inventory map has an empty source class")
			continue
		}
		if seen[name] {
			addInventoryRowReason(row, "inventory map has duplicate source class status: "+name)
			continue
		}
		seen[name] = true
		if !isRequiredInventorySourceClass(name) {
			addInventoryRowReason(row, "inventory map has unsupported source class status: "+name)
		}
		status := strings.TrimSpace(class.Status)
		switch status {
		case InventoryClassCovered, InventoryClassPartial, InventoryClassCheckedAbsent, InventoryClassExternalRequired:
		default:
			addInventoryRowReason(row, "inventory map source class "+name+" has unsupported status "+class.Status)
		}
		if isRequiredInventorySourceClass(name) && !inventoryMapDispositionAllowed(name, status) {
			addInventoryRowReason(row, "inventory map source class "+name+" cannot use status "+status)
		}
		if (status == InventoryClassCovered && name != "completeness_critic") || status == InventoryClassPartial {
			if len(normalizedUnique(class.Evidence)) == 0 {
				addInventoryRowReason(row, "inventory map source class "+name+" requires local path evidence for status "+status)
			}
		}
	}
	for _, required := range RequiredInventorySourceClasses {
		if !seen[required] {
			addInventoryRowReason(row, "inventory map missing source class status: "+required)
		}
	}
}

func inventoryMapDispositionAllowed(class, status string) bool {
	switch class {
	case "open_closed_issues_prs_discussions":
		return status == InventoryClassPartial || status == InventoryClassExternalRequired
	case "fak_selfquery_witness", "candidate_matrix", "issue_tracking":
		return status == InventoryClassExternalRequired
	case "completeness_critic":
		return status == InventoryClassCovered
	default:
		return status == InventoryClassCovered || status == InventoryClassCheckedAbsent
	}
}

func addInventoryRowReason(row *InventoryRow, reason string) {
	row.Reasons = append(row.Reasons, reason)
	row.Ready = false
}

func inventoryBaseRow(repo Repository) InventoryRow {
	mode := effectiveInventoryMode(repo)
	row := InventoryRow{
		Repository:   repo.Repository,
		RefreshIssue: repo.RefreshIssue,
		Status:       repo.Status,
		Mode:         mode,
		Ready:        true,
	}
	if repo.Inventory != nil {
		row.MapPath = strings.TrimSpace(repo.Inventory.MapPath)
		row.ForgeReceiptPath = strings.TrimSpace(repo.Inventory.ForgeReceiptPath)
		row.IndexedRevision = strings.TrimSpace(repo.Inventory.IndexedRevision)
		row.SubsystemCount = repo.Inventory.SubsystemCount
		row.CandidateCount = repo.Inventory.CandidateCount
		row.FiledIssueCount = repo.Inventory.FiledIssueCount
		row.IssueRefs = append([]string(nil), repo.Inventory.IssueRefs...)
		row.CompletenessCritic = strings.TrimSpace(repo.Inventory.CompletenessCritic)
		row.SourceClasses = normalizedUnique(repo.Inventory.SourceClasses)
		row.SourceEvidence = normalizedSourceEvidence(repo.Inventory.SourceEvidence)
	}
	return row
}

func finalizeInventoryRow(row *InventoryRow, repo Repository, inventoryMap *InventoryMap) {
	if row.Mode != InventoryModeExhaustive {
		return
	}
	if row.MapPath == "" {
		row.Reasons = append(row.Reasons, "missing inventory map_path")
	}
	if row.IndexedRevision == "" {
		row.Reasons = append(row.Reasons, "missing indexed_revision")
	} else if row.IndexedRevision != repo.CheckedRevision {
		row.Reasons = append(row.Reasons, "indexed_revision does not match checked_revision")
	}
	if row.SubsystemCount <= 0 {
		row.Reasons = append(row.Reasons, "subsystem_count must be positive")
	}
	if row.CompletenessCritic == "" {
		row.Reasons = append(row.Reasons, "missing completeness_critic")
	}
	satisfied := inventorySatisfiedSourceClasses(row, inventoryMap)
	missing := missingRequiredSourceClassesFromSet(satisfied)
	if len(missing) > 0 {
		row.MissingSourceClasses = missing
		row.Reasons = append(row.Reasons, "missing source classes: "+strings.Join(missing, ","))
	}
	if repo.Status == "studied" && len(row.Reasons) > 0 && strings.TrimSpace(row.RefreshIssue) == "" {
		row.Reasons = append(row.Reasons, "missing refresh_issue for incomplete studied row")
	}
	validateInventoryDeclaredSourceClasses(row, inventoryMap)
	row.Ready = len(row.Reasons) == 0
}

func inventorySatisfiedSourceClasses(row *InventoryRow, inventoryMap *InventoryMap) map[string]bool {
	satisfied := map[string]bool{}
	if row.forgeReceiptValid {
		satisfied["open_closed_issues_prs_discussions"] = true
	}
	if inventoryMap != nil {
		for _, class := range inventoryMap.SourceClasses {
			name := strings.TrimSpace(class.Class)
			switch strings.TrimSpace(class.Status) {
			case InventoryClassCovered, InventoryClassCheckedAbsent:
				satisfied[name] = true
			}
		}
	}
	for _, evidence := range row.SourceEvidence {
		if len(evidence.Evidence) > 0 {
			satisfied[evidence.Class] = true
		}
	}
	for _, class := range row.SourceClasses {
		if inventoryMap == nil || inventoryMapSatisfiesSourceClass(inventoryMap, class) || inventoryEvidenceSatisfiesSourceClass(row.SourceEvidence, class) {
			satisfied[class] = true
		}
	}
	return satisfied
}

func validateInventoryDeclaredSourceClasses(row *InventoryRow, inventoryMap *InventoryMap) {
	if inventoryMap == nil {
		return
	}
	for _, class := range row.SourceClasses {
		if class == "open_closed_issues_prs_discussions" && row.forgeReceiptValid {
			continue
		}
		if inventoryMapSatisfiesSourceClass(inventoryMap, class) || inventoryEvidenceSatisfiesSourceClass(row.SourceEvidence, class) {
			continue
		}
		status, ok := inventoryMapSourceClassStatus(inventoryMap, class)
		switch {
		case !ok:
			addInventoryRowReason(row, "source class "+class+" is declared without map status or explicit source_evidence")
		case status == InventoryClassPartial:
			addInventoryRowReason(row, "source class "+class+" is only partial in the inventory map; add explicit source_evidence for the missing non-tree coverage")
		case status == InventoryClassExternalRequired:
			addInventoryRowReason(row, "source class "+class+" requires explicit source_evidence")
		default:
			addInventoryRowReason(row, "source class "+class+" is declared without covered map evidence or explicit source_evidence")
		}
	}
}

func inventoryMapSatisfiesSourceClass(inventoryMap *InventoryMap, class string) bool {
	status, ok := inventoryMapSourceClassStatus(inventoryMap, class)
	if !ok {
		return false
	}
	return status == InventoryClassCovered || status == InventoryClassCheckedAbsent
}

func inventoryMapSourceClassStatus(inventoryMap *InventoryMap, class string) (string, bool) {
	if inventoryMap == nil {
		return "", false
	}
	for _, row := range inventoryMap.SourceClasses {
		if row.Class == class {
			return strings.TrimSpace(row.Status), true
		}
	}
	return "", false
}

func inventoryEvidenceSatisfiesSourceClass(evidence []InventorySourceEvidence, class string) bool {
	for _, row := range evidence {
		if row.Class == class && len(row.Evidence) > 0 {
			return true
		}
	}
	return false
}

func isTraceableInventoryEvidence(value string) bool {
	value = strings.TrimSpace(value)
	lower := strings.ToLower(value)
	if strings.HasPrefix(lower, "https://") || strings.HasPrefix(lower, "http://") || strings.HasPrefix(lower, "gh:") || strings.HasPrefix(lower, "study_") {
		return true
	}
	if strings.HasPrefix(value, "#") && len(value) > 1 {
		for _, r := range value[1:] {
			if r < '0' || r > '9' {
				return false
			}
		}
		return true
	}
	for _, prefix := range []string{"gh ", "go run ", "fak ", "dos ", "git ", "./", "python "} {
		if strings.HasPrefix(lower, prefix) {
			return true
		}
	}
	if strings.Contains(value, "@") && !strings.ContainsAny(value, " \t\r\n") {
		return true
	}
	if strings.HasPrefix(lower, "license") || strings.HasPrefix(lower, "notice") || strings.HasPrefix(lower, "readme") {
		return true
	}
	return !strings.ContainsAny(value, " \t\r\n") && strings.Contains(value, "/")
}

func missingInventoryEvidenceFacets(class string, evidence []string) []string {
	joined := strings.ToLower(strings.Join(evidence, "\n"))
	switch class {
	case "open_closed_issues_prs_discussions":
		var missing []string
		if !strings.Contains(joined, "issue") {
			missing = append(missing, "issues")
		}
		if !strings.Contains(joined, "pr ") && !strings.Contains(joined, "pull") && !strings.Contains(joined, "/pr/") {
			missing = append(missing, "pull_requests")
		}
		if !strings.Contains(joined, "discussion") {
			missing = append(missing, "discussions")
		}
		return missing
	case "fak_selfquery_witness":
		if !strings.Contains(joined, "fak ") && !strings.Contains(joined, "self-query") && !strings.Contains(joined, "selfquery") {
			return []string{"fak_self_query"}
		}
	case "candidate_matrix":
		if !strings.Contains(joined, "candidate") || !strings.Contains(joined, "matrix") {
			return []string{"candidate_matrix"}
		}
	case "issue_tracking":
		if !strings.Contains(joined, "/issues/") && !strings.Contains(joined, "#") {
			return []string{"issue_reference"}
		}
	}
	return nil
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

func missingRequiredSourceClassesFromSet(seen map[string]bool) []string {
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

func isRequiredInventorySourceClass(class string) bool {
	for _, required := range RequiredInventorySourceClasses {
		if class == required {
			return true
		}
	}
	return false
}

func normalizedSourceEvidence(values []InventorySourceEvidence) []InventorySourceEvidence {
	out := make([]InventorySourceEvidence, 0, len(values))
	for _, value := range values {
		class := strings.TrimSpace(value.Class)
		evidence := normalizedUnique(value.Evidence)
		if class == "" || len(evidence) == 0 {
			continue
		}
		out = append(out, InventorySourceEvidence{Class: class, Evidence: evidence, Note: strings.TrimSpace(value.Note)})
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Class != out[j].Class {
			return out[i].Class < out[j].Class
		}
		return strings.Join(out[i].Evidence, "\x00") < strings.Join(out[j].Evidence, "\x00")
	})
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

// SelfInventoryVerificationCacheContract documents the bounded warm-path contract
// implemented by the devcmd self-inventory gate. It lives with the owning
// studymonitor leaf so commit and closure witnesses bind the optimization to the
// subsystem whose verdict is preserved.
const SelfInventoryVerificationCacheContract = "immutable-tip+repository+manifest+inventory-schema; complete verdict; fail-closed miss"

package studymonitor

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

const InventoryMapSchema = "fak-study-inventory-map/1"

const (
	InventoryClassCovered          = "covered"
	InventoryClassPartial          = "partial"
	InventoryClassCheckedAbsent    = "checked_absent"
	InventoryClassExternalRequired = "external_required"
)

type InventoryMapOptions struct {
	Repository              string
	URL                     string
	IndexedRevision         string
	ObservedAt              string
	MaxExamplesPerSubsystem int
}

type InventoryMap struct {
	Schema           string                 `json:"schema"`
	Repository       string                 `json:"repository"`
	URL              string                 `json:"url,omitempty"`
	IndexedRevision  string                 `json:"indexed_revision"`
	ObservedAt       string                 `json:"observed_at"`
	Totals           InventoryMapTotals     `json:"totals"`
	SourceClasses    []InventoryClassStatus `json:"source_classes"`
	Subsystems       []InventorySubsystem   `json:"subsystems"`
	SkippedDirs      []string               `json:"skipped_dirs,omitempty"`
	CompletenessNote string                 `json:"completeness_critic"`
}

type InventoryMapTotals struct {
	Files        int   `json:"files"`
	Directories  int   `json:"directories"`
	Bytes        int64 `json:"bytes"`
	RuntimeFiles int   `json:"runtime_files"`
	TestFiles    int   `json:"test_files"`
	DocsFiles    int   `json:"docs_files"`
	TextLines    int   `json:"text_lines"`
}

type InventoryClassStatus struct {
	Class    string   `json:"class"`
	Status   string   `json:"status"`
	Evidence []string `json:"evidence,omitempty"`
	Note     string   `json:"note,omitempty"`
}

type InventorySubsystem struct {
	Path          string                   `json:"path"`
	Files         int                      `json:"files"`
	Directories   int                      `json:"directories"`
	Bytes         int64                    `json:"bytes"`
	RuntimeFiles  int                      `json:"runtime_files"`
	TestFiles     int                      `json:"test_files"`
	DocsFiles     int                      `json:"docs_files"`
	TextLines     int                      `json:"text_lines"`
	Languages     []InventoryLanguageCount `json:"languages,omitempty"`
	ExamplePaths  []string                 `json:"example_paths,omitempty"`
	SourceClasses []string                 `json:"source_classes,omitempty"`
}

type InventoryLanguageCount struct {
	Language string `json:"language"`
	Files    int    `json:"files"`
}

type inventoryAccumulator struct {
	totals        InventoryMapTotals
	subsystems    map[string]*subsystemAccumulator
	sourceClasses map[string]map[string]bool
	skippedDirs   []string
}

type subsystemAccumulator struct {
	row           InventorySubsystem
	languages     map[string]int
	sourceClasses map[string]bool
	exampleLimit  int
}

func BuildInventoryMap(root string, opts InventoryMapOptions) (InventoryMap, error) {
	if strings.TrimSpace(root) == "" {
		return InventoryMap{}, fmt.Errorf("root is required")
	}
	info, err := os.Stat(root)
	if err != nil {
		return InventoryMap{}, err
	}
	if !info.IsDir() {
		return InventoryMap{}, fmt.Errorf("%s is not a directory", root)
	}
	if strings.TrimSpace(opts.Repository) == "" {
		return InventoryMap{}, fmt.Errorf("repository is required")
	}
	if strings.TrimSpace(opts.IndexedRevision) == "" {
		return InventoryMap{}, fmt.Errorf("indexed revision is required")
	}
	if opts.MaxExamplesPerSubsystem <= 0 {
		opts.MaxExamplesPerSubsystem = 8
	}
	acc := inventoryAccumulator{
		subsystems:    map[string]*subsystemAccumulator{},
		sourceClasses: map[string]map[string]bool{},
	}
	for _, class := range RequiredInventorySourceClasses {
		acc.sourceClasses[class] = map[string]bool{}
	}
	err = filepath.WalkDir(root, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if path == root {
			return nil
		}
		rel, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		rel = filepath.ToSlash(rel)
		if entry.IsDir() {
			if shouldSkipInventoryDir(entry.Name()) {
				acc.skippedDirs = append(acc.skippedDirs, rel)
				return filepath.SkipDir
			}
			acc.totals.Directories++
			acc.subsystemFor(rel, opts.MaxExamplesPerSubsystem).row.Directories++
			return nil
		}
		info, err := entry.Info()
		if err != nil {
			return err
		}
		if !info.Mode().IsRegular() {
			return nil
		}
		file := classifyInventoryFile(rel)
		lines := countTextLines(path, info.Size(), file.TextLike)
		acc.addFile(rel, info.Size(), lines, file, opts.MaxExamplesPerSubsystem)
		return nil
	})
	if err != nil {
		return InventoryMap{}, err
	}
	sourceClasses := acc.renderSourceClasses()
	subsystems := acc.renderSubsystems()
	return InventoryMap{
		Schema:           InventoryMapSchema,
		Repository:       strings.TrimSpace(opts.Repository),
		URL:              strings.TrimSpace(opts.URL),
		IndexedRevision:  strings.TrimSpace(opts.IndexedRevision),
		ObservedAt:       strings.TrimSpace(opts.ObservedAt),
		Totals:           acc.totals,
		SourceClasses:    sourceClasses,
		Subsystems:       subsystems,
		SkippedDirs:      sortedStrings(acc.skippedDirs),
		CompletenessNote: inventoryCompletenessNote(sourceClasses, acc.skippedDirs),
	}, nil
}

func WriteInventoryMapJSON(w io.Writer, report InventoryMap) error {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(report)
}

func RenderInventoryMapMarkdown(w io.Writer, report InventoryMap) {
	fmt.Fprintf(w, "# Study inventory: %s\n\n", report.Repository)
	fmt.Fprintf(w, "- **Schema:** `%s`\n", report.Schema)
	fmt.Fprintf(w, "- **Indexed revision:** `%s`\n", report.IndexedRevision)
	if report.URL != "" {
		fmt.Fprintf(w, "- **Source:** %s\n", report.URL)
	}
	if report.ObservedAt != "" {
		fmt.Fprintf(w, "- **Observed at:** %s\n", report.ObservedAt)
	}
	fmt.Fprintf(w, "- **Totals:** %d files, %d directories, %d runtime files, %d tests/fixtures, %d docs, %d text lines\n\n",
		report.Totals.Files, report.Totals.Directories, report.Totals.RuntimeFiles, report.Totals.TestFiles, report.Totals.DocsFiles, report.Totals.TextLines)
	fmt.Fprintln(w, "## Source Classes")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "| Class | Status | Evidence | Note |")
	fmt.Fprintln(w, "|---|---:|---|---|")
	for _, row := range report.SourceClasses {
		fmt.Fprintf(w, "| `%s` | `%s` | %s | %s |\n", row.Class, row.Status, strings.Join(row.Evidence, "<br>"), row.Note)
	}
	fmt.Fprintln(w)
	fmt.Fprintln(w, "## Subsystems")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "| Path | Files | Runtime | Tests | Docs | Languages | Examples |")
	fmt.Fprintln(w, "|---|---:|---:|---:|---:|---|---|")
	for _, row := range report.Subsystems {
		fmt.Fprintf(w, "| `%s` | %d | %d | %d | %d | %s | %s |\n",
			row.Path, row.Files, row.RuntimeFiles, row.TestFiles, row.DocsFiles, renderLanguages(row.Languages), strings.Join(row.ExamplePaths, "<br>"))
	}
	if len(report.SkippedDirs) > 0 {
		fmt.Fprintln(w)
		fmt.Fprintln(w, "## Skipped Directories")
		fmt.Fprintln(w)
		for _, dir := range report.SkippedDirs {
			fmt.Fprintf(w, "- `%s`\n", dir)
		}
	}
	fmt.Fprintln(w)
	fmt.Fprintln(w, "## Completeness Critic")
	fmt.Fprintln(w)
	fmt.Fprintln(w, report.CompletenessNote)
}

func (acc *inventoryAccumulator) addFile(rel string, size int64, lines int, file inventoryFileClass, exampleLimit int) {
	acc.totals.Files++
	acc.totals.Bytes += size
	acc.totals.TextLines += lines
	if file.Runtime {
		acc.totals.RuntimeFiles++
	}
	if file.Test {
		acc.totals.TestFiles++
	}
	if file.Doc {
		acc.totals.DocsFiles++
	}
	for _, class := range file.SourceClasses {
		acc.sourceClasses[class][rel] = true
	}
	subsystem := acc.subsystemFor(rel, exampleLimit)
	subsystem.addFile(rel, size, lines, file)
}

func (acc *inventoryAccumulator) subsystemFor(rel string, exampleLimit int) *subsystemAccumulator {
	name := inventorySubsystemName(rel)
	if existing := acc.subsystems[name]; existing != nil {
		return existing
	}
	next := &subsystemAccumulator{
		row:           InventorySubsystem{Path: name},
		languages:     map[string]int{},
		sourceClasses: map[string]bool{},
		exampleLimit:  exampleLimit,
	}
	acc.subsystems[name] = next
	return next
}

func (acc inventoryAccumulator) renderSourceClasses() []InventoryClassStatus {
	out := make([]InventoryClassStatus, 0, len(RequiredInventorySourceClasses))
	for _, class := range RequiredInventorySourceClasses {
		evidence := sortedMapKeys(acc.sourceClasses[class])
		status, note := inventorySourceClassDisposition(class, evidence)
		out = append(out, InventoryClassStatus{Class: class, Status: status, Evidence: capStrings(evidence, 12), Note: note})
	}
	return out
}

func (acc inventoryAccumulator) renderSubsystems() []InventorySubsystem {
	out := make([]InventorySubsystem, 0, len(acc.subsystems))
	for _, subsystem := range acc.subsystems {
		row := subsystem.row
		row.Languages = renderLanguageCounts(subsystem.languages)
		row.SourceClasses = sortedBoolKeys(subsystem.sourceClasses)
		out = append(out, row)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Files != out[j].Files {
			return out[i].Files > out[j].Files
		}
		return out[i].Path < out[j].Path
	})
	return out
}

func (subsystem *subsystemAccumulator) addFile(rel string, size int64, lines int, file inventoryFileClass) {
	subsystem.row.Files++
	subsystem.row.Bytes += size
	subsystem.row.TextLines += lines
	if file.Runtime {
		subsystem.row.RuntimeFiles++
	}
	if file.Test {
		subsystem.row.TestFiles++
	}
	if file.Doc {
		subsystem.row.DocsFiles++
	}
	if file.Language != "" {
		subsystem.languages[file.Language]++
	}
	for _, class := range file.SourceClasses {
		subsystem.sourceClasses[class] = true
	}
	if len(subsystem.row.ExamplePaths) < subsystem.exampleLimit && shouldKeepInventoryExample(rel, file) {
		subsystem.row.ExamplePaths = append(subsystem.row.ExamplePaths, rel)
	}
}

type inventoryFileClass struct {
	Language      string
	Runtime       bool
	Test          bool
	Doc           bool
	TextLike      bool
	SourceClasses []string
}

func classifyInventoryFile(rel string) inventoryFileClass {
	lower := strings.ToLower(rel)
	base := strings.ToLower(filepath.Base(rel))
	ext := strings.ToLower(filepath.Ext(base))
	language := inventoryLanguage(ext, base)
	isDoc := isInventoryDoc(lower, base, ext)
	isTest := isInventoryTest(lower, base)
	isRuntime := language != "" && !isDoc && !isTest
	classes := map[string]bool{}
	if base == "readme" || strings.HasPrefix(base, "readme.") || strings.HasPrefix(lower, "docs/") {
		classes["readme_docs"] = true
	}
	if strings.Contains(lower, "architecture") || strings.Contains(lower, "design") ||
		strings.Contains(lower, "/adr") || strings.Contains(lower, "/rfc") ||
		strings.HasPrefix(lower, "adr/") || strings.HasPrefix(lower, "rfcs/") {
		classes["architecture_design"] = true
	}
	if isRuntime {
		classes["runtime_source"] = true
	}
	if isTest {
		classes["tests_fixtures"] = true
	}
	if strings.Contains(base, "changelog") || strings.Contains(base, "changes") ||
		strings.Contains(base, "history") || strings.Contains(base, "release") {
		classes["history_changelog_releases"] = true
	}
	if strings.Contains(base, "roadmap") || strings.Contains(base, "todo") ||
		strings.Contains(lower, "/roadmap") || strings.Contains(lower, "/backlog") {
		classes["roadmap_todos"] = true
	}
	if base == "license" || strings.HasPrefix(base, "license.") ||
		base == "copying" || strings.HasPrefix(base, "copying.") ||
		base == "notice" || strings.HasPrefix(base, "notice.") ||
		strings.Contains(base, "third_party") || strings.Contains(base, "third-party") ||
		base == "go.mod" || base == "package.json" || base == "cargo.toml" || base == "pyproject.toml" {
		classes["license_provenance"] = true
	}
	if strings.HasPrefix(lower, ".github/issue_template") || strings.HasPrefix(lower, ".github/pull_request_template") || strings.HasPrefix(lower, ".github/issues/") {
		classes["open_closed_issues_prs_discussions"] = true
	}
	return inventoryFileClass{
		Language:      language,
		Runtime:       isRuntime,
		Test:          isTest,
		Doc:           isDoc,
		TextLike:      isDoc || language != "" || ext == ".toml" || ext == ".yaml" || ext == ".yml" || ext == ".json",
		SourceClasses: sortedBoolKeys(classes),
	}
}

func inventorySourceClassDisposition(class string, evidence []string) (string, string) {
	if len(evidence) > 0 {
		if class == "open_closed_issues_prs_discussions" {
			return InventoryClassPartial, "local templates found; open/closed issue, PR, and discussion history still require GitHub or forge read-back"
		}
		return InventoryClassCovered, "local tree evidence found"
	}
	switch class {
	case "open_closed_issues_prs_discussions":
		return InventoryClassExternalRequired, "requires forge API or exported issue/PR/discussion data; local tree scan cannot prove it"
	case "completeness_critic":
		return InventoryClassCovered, "generated by this inventory map's local tree walk; non-tree omissions remain named in the note"
	case "fak_selfquery_witness", "candidate_matrix", "issue_tracking":
		return InventoryClassExternalRequired, "process artifact created by the study pass, not by the foreign repository tree"
	default:
		return InventoryClassCheckedAbsent, "local tree scanned and no matching path was found"
	}
}

func inventoryCompletenessNote(classes []InventoryClassStatus, skipped []string) string {
	var external, partial, absent []string
	for _, class := range classes {
		switch class.Status {
		case InventoryClassExternalRequired:
			external = append(external, class.Class)
		case InventoryClassPartial:
			partial = append(partial, class.Class)
		case InventoryClassCheckedAbsent:
			absent = append(absent, class.Class)
		}
	}
	var parts []string
	parts = append(parts, "local tree inventory walked every non-skipped regular file and grouped immediate subsystems")
	if len(skipped) > 0 {
		parts = append(parts, fmt.Sprintf("skipped dependency/cache/control directories: %s", strings.Join(sortedStrings(skipped), ", ")))
	}
	if len(absent) > 0 {
		parts = append(parts, fmt.Sprintf("classes checked absent in local tree: %s", strings.Join(absent, ", ")))
	}
	if len(partial) > 0 {
		parts = append(parts, fmt.Sprintf("classes with partial local evidence still requiring non-tree completion: %s", strings.Join(partial, ", ")))
	}
	if len(external) > 0 {
		parts = append(parts, fmt.Sprintf("still requires non-tree study artifacts: %s", strings.Join(external, ", ")))
	}
	return strings.Join(parts, "; ")
}

func shouldSkipInventoryDir(name string) bool {
	switch strings.ToLower(name) {
	case ".git", ".hg", ".svn", "node_modules", "vendor", ".venv", "venv", "__pycache__", "dist", "build", ".next", ".cache", ".turbo", ".pytest_cache", "target", "coverage":
		return true
	default:
		return false
	}
}

func inventorySubsystemName(rel string) string {
	rel = strings.Trim(rel, "/")
	if rel == "" {
		return "."
	}
	parts := strings.Split(rel, "/")
	if len(parts) == 1 {
		return "."
	}
	return parts[0]
}

func inventoryLanguage(ext, base string) string {
	switch ext {
	case ".go":
		return "Go"
	case ".py":
		return "Python"
	case ".js", ".jsx":
		return "JavaScript"
	case ".ts", ".tsx":
		return "TypeScript"
	case ".rs":
		return "Rust"
	case ".java":
		return "Java"
	case ".kt", ".kts":
		return "Kotlin"
	case ".c", ".h":
		return "C"
	case ".cc", ".cpp", ".cxx", ".hh", ".hpp", ".hxx":
		return "C++"
	case ".cs":
		return "C#"
	case ".rb":
		return "Ruby"
	case ".php":
		return "PHP"
	case ".swift":
		return "Swift"
	case ".scala":
		return "Scala"
	case ".sh", ".bash", ".zsh", ".ps1":
		return "Shell"
	case ".md", ".rst", ".adoc", ".txt":
		return "Docs"
	case ".json", ".yaml", ".yml", ".toml", ".xml":
		return "Config"
	default:
		if base == "makefile" || base == "dockerfile" {
			return "Config"
		}
		return ""
	}
}

func isInventoryDoc(lower, base, ext string) bool {
	if base == "readme" || strings.HasPrefix(base, "readme.") || strings.HasPrefix(lower, "docs/") {
		return true
	}
	switch ext {
	case ".md", ".rst", ".adoc", ".txt":
		return true
	default:
		return false
	}
}

func isInventoryTest(lower, base string) bool {
	return strings.Contains(lower, "/test/") ||
		strings.Contains(lower, "/tests/") ||
		strings.Contains(lower, "/__tests__/") ||
		strings.Contains(lower, "/fixture") ||
		strings.Contains(lower, "/fixtures/") ||
		strings.Contains(base, "_test.") ||
		strings.Contains(base, ".test.") ||
		strings.Contains(base, ".spec.")
}

func shouldKeepInventoryExample(rel string, file inventoryFileClass) bool {
	base := strings.ToLower(filepath.Base(rel))
	return file.Runtime || file.Test || file.Doc || strings.Contains(base, "license") || strings.Contains(base, "changelog") || strings.Contains(base, "roadmap")
}

func countTextLines(path string, size int64, textLike bool) int {
	if !textLike || size > 1<<20 {
		return 0
	}
	f, err := os.Open(path)
	if err != nil {
		return 0
	}
	defer f.Close()
	lines := 0
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		lines++
	}
	return lines
}

func renderLanguageCounts(counts map[string]int) []InventoryLanguageCount {
	out := make([]InventoryLanguageCount, 0, len(counts))
	for language, files := range counts {
		out = append(out, InventoryLanguageCount{Language: language, Files: files})
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Files != out[j].Files {
			return out[i].Files > out[j].Files
		}
		return out[i].Language < out[j].Language
	})
	return out
}

func renderLanguages(languages []InventoryLanguageCount) string {
	if len(languages) == 0 {
		return ""
	}
	parts := make([]string, 0, len(languages))
	for _, language := range languages {
		parts = append(parts, fmt.Sprintf("%s:%d", language.Language, language.Files))
	}
	return strings.Join(parts, ", ")
}

func sortedMapKeys(values map[string]bool) []string {
	out := make([]string, 0, len(values))
	for value := range values {
		out = append(out, value)
	}
	sort.Strings(out)
	return out
}

func sortedBoolKeys(values map[string]bool) []string {
	out := make([]string, 0, len(values))
	for value := range values {
		out = append(out, value)
	}
	sort.Strings(out)
	return out
}

func sortedStrings(values []string) []string {
	out := append([]string(nil), values...)
	sort.Strings(out)
	return out
}

func capStrings(values []string, max int) []string {
	if len(values) <= max {
		return values
	}
	return append([]string(nil), values[:max]...)
}

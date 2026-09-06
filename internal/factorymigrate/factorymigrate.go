package factorymigrate

import (
	"bufio"
	"fmt"
	"go/parser"
	"go/token"
	"io/fs"
	"math"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// Status represents the migration status of a tool.
type Status string

const (
	StatusMigrated   Status = "MIGRATED"
	StatusPartial    Status = "PARTIAL"
	StatusUnmigrated Status = "UNMIGRATED"
)

// Item represents a single dev process tool in the inventory.
type Item struct {
	Number     int    `json:"number"`
	Cohort     string `json:"cohort"`
	SourcePath string `json:"source_path"`
	TargetPath string `json:"target_path"`
	TargetPkg  string `json:"target_pkg"`
	Status     Status `json:"status"`
	Notes      string `json:"notes,omitempty"`
}

// CohortSummary summarizes migration progress for a specific domain cohort.
type CohortSummary struct {
	Name       string  `json:"name"`
	TargetPkg  string  `json:"target_pkg"`
	Total      int     `json:"total"`
	Migrated   int     `json:"migrated"`
	Partial    int     `json:"partial"`
	Unmigrated int     `json:"unmigrated"`
	Percent    float64 `json:"percent"`
}

// Report holds the full status report across all cohorts and tools.
type Report struct {
	Total      int             `json:"total"`
	Migrated   int             `json:"migrated"`
	Partial    int             `json:"partial"`
	Unmigrated int             `json:"unmigrated"`
	Percent    float64         `json:"percent"`
	Cohorts    []CohortSummary `json:"cohorts"`
	Items      []Item          `json:"items"`
}

// BoundaryViolation records an invalid import in fak-private crossing the 5-Gate boundary.
type BoundaryViolation struct {
	File       string `json:"file"`
	Line       int    `json:"line"`
	ImportPath string `json:"import_path"`
	Rule       string `json:"rule"`
	Reason     string `json:"reason"`
}

var (
	reCohortHeader = regexp.MustCompile(`^###\s+Cohort\s+(\d+):\s+([^(]+)`)
	reItemLine     = regexp.MustCompile(`^(\d+)\.\s*` + "`" + `?([^` + "`" + `\s]+)` + "`" + `?\s*(?:\$\\rightarrow\$|\\rightarrow|->|→)\s*` + "`" + `?([^` + "`" + `\s]+)` + "`" + `?`)
)

type knownMigration struct {
	altPath string
	inFak   bool
	status  Status
	note    string
}

// knownMigrations maps item numbers to alternate implementation locations already present.
var knownMigrations = map[int]knownMigration{
	// Cohort 1: Watchdogs
	2:  {altPath: "platform/watchdogs/watchdog.go", inFak: false, status: StatusPartial, note: "reconciled with platform/watchdogs/watchdog.go in fak-private"},
	4:  {altPath: "platform/watchdogs/watchdog.go", inFak: false, status: StatusPartial, note: "reconciled with platform/watchdogs/watchdog.go in fak-private"},
	7:  {altPath: "platform/watchdogs/watchdog.go", inFak: false, status: StatusPartial, note: "reconciled with platform/watchdogs/watchdog.go in fak-private"},
	12: {altPath: "internal/procguard/procguard.go", inFak: true, status: StatusPartial, note: "reconciled with internal/procguard in public fak"},
	18: {altPath: "platform/watchdogs/watchdog.go", inFak: false, status: StatusPartial, note: "reconciled with platform/watchdogs/watchdog.go in fak-private"},

	// Cohort 2: Autonomous Dispatch
	27: {altPath: "platform/dispatch/runner.go", inFak: false, status: StatusPartial, note: "reconciled with platform/dispatch/runner.go in fak-private"},
	28: {altPath: "platform/dispatch/scheduler.go", inFak: false, status: StatusPartial, note: "reconciled with platform/dispatch/scheduler.go in fak-private"},
	33: {altPath: "platform/dispatch/contract.go", inFak: false, status: StatusPartial, note: "reconciled with platform/dispatch/contract.go in fak-private"},
	35: {altPath: "platform/dispatch/governor.go", inFak: false, status: StatusPartial, note: "reconciled with platform/dispatch/governor.go in fak-private"},
	38: {altPath: "platform/worktree/manager.go", inFak: false, status: StatusPartial, note: "reconciled with platform/worktree/manager.go in fak-private"},
	39: {altPath: "platform/worktree/wip.go", inFak: false, status: StatusPartial, note: "reconciled with platform/worktree/wip.go in fak-private"},

	// Cohort 3: Scorecards & Telemetry
	56: {altPath: "platform/scorecards/registry.go", inFak: false, status: StatusPartial, note: "reconciled with platform/scorecards/registry.go in fak-private"},
	72: {altPath: "platform/scorecards/registry.go", inFak: false, status: StatusPartial, note: "reconciled with platform/scorecards/registry.go in fak-private"},

	// Cohort 5: Agent Memory & Context
	99:  {altPath: "platform/memsync/memsync.go", inFak: false, status: StatusPartial, note: "reconciled with platform/memsync/memsync.go in fak-private"},
	100: {altPath: "platform/memsync/memsync.go", inFak: false, status: StatusPartial, note: "reconciled with platform/memsync/memsync.go in fak-private"},
}

// ParseInventory parses docs/dev-process-top-100-tools-inventory.md into structured Items.
// If inventoryPath cannot be read or is empty, it falls back to the embedded catalog.
func ParseInventory(inventoryPath string) ([]Item, error) {
	var data []byte
	var err error
	if inventoryPath != "" {
		data, err = os.ReadFile(inventoryPath)
	}
	if inventoryPath == "" || err != nil {
		data = []byte(embeddedInventory)
	}
	return parseInventoryText(string(data))
}

func parseInventoryText(content string) ([]Item, error) {
	scanner := bufio.NewScanner(strings.NewReader(content))
	var items []Item
	currentCohort := ""
	currentCohortPkg := ""

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		if match := reCohortHeader.FindStringSubmatch(line); len(match) > 0 {
			cohortNum := match[1]
			cohortName := strings.TrimSpace(match[2])
			currentCohort = fmt.Sprintf("Cohort %s: %s", cohortNum, cohortName)
			currentCohortPkg = ""
			if idxStart := strings.Index(line, "("); idxStart != -1 {
				if idxEnd := strings.Index(line[idxStart:], ")"); idxEnd != -1 {
					inside := line[idxStart+1 : idxStart+idxEnd]
					currentCohortPkg = strings.Trim(strings.TrimSpace(inside), "`")
				}
			}
			continue
		}

		if match := reItemLine.FindStringSubmatch(line); len(match) > 0 {
			num, err := strconv.Atoi(match[1])
			if err != nil {
				continue
			}
			src := strings.TrimSpace(match[2])
			tgt := strings.TrimSpace(match[3])
			tgtPkg := path.Dir(tgt)
			if currentCohortPkg != "" && tgtPkg == "." {
				tgtPkg = currentCohortPkg
			}

			items = append(items, Item{
				Number:     num,
				Cohort:     currentCohort,
				SourcePath: src,
				TargetPath: tgt,
				TargetPkg:  tgtPkg,
				Status:     StatusUnmigrated,
			})
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error scanning inventory content: %w", err)
	}

	if len(items) == 0 {
		return nil, fmt.Errorf("no inventory items parsed")
	}

	return items, nil
}

func fileExists(p string) bool {
	if p == "" {
		return false
	}
	fi, err := os.Stat(p)
	return err == nil && !fi.IsDir()
}

// AuditStatus checks disk presence of source and target files in both repositories,
// reconciling known migrations into an authoritative Report.
func AuditStatus(fakRoot, privateRoot string, items []Item) Report {
	if len(items) == 0 {
		var err error
		items, err = ParseInventory("")
		if err != nil {
			return Report{}
		}
	}

	audited := make([]Item, len(items))
	copy(audited, items)

	for i := range audited {
		item := &audited[i]

		targetInPrivate := privateRoot != "" && fileExists(filepath.Join(privateRoot, filepath.FromSlash(item.TargetPath)))
		targetInFak := fakRoot != "" && fileExists(filepath.Join(fakRoot, filepath.FromSlash(item.TargetPath)))
		sourceInFak := fakRoot != "" && fileExists(filepath.Join(fakRoot, filepath.FromSlash(item.SourcePath)))

		if targetInPrivate {
			item.Status = StatusMigrated
			if !sourceInFak {
				item.Notes = "target active in fak-private; legacy source retired"
			} else {
				item.Notes = "target active in fak-private (coexisting with legacy source)"
			}
		} else if targetInFak {
			item.Status = StatusMigrated
			if !sourceInFak {
				item.Notes = "target active in fak; legacy source retired"
			} else {
				item.Notes = "target active in fak (coexisting with legacy source)"
			}
		} else if km, ok := knownMigrations[item.Number]; ok {
			matched := false
			if km.inFak && fakRoot != "" {
				matched = fileExists(filepath.Join(fakRoot, filepath.FromSlash(km.altPath)))
			} else if !km.inFak && privateRoot != "" {
				matched = fileExists(filepath.Join(privateRoot, filepath.FromSlash(km.altPath)))
			}

			if matched {
				item.Status = km.status
				if !sourceInFak {
					item.Notes = km.note + " (legacy script retired)"
				} else {
					item.Notes = km.note
				}
			} else {
				item.Status = StatusUnmigrated
				if !sourceInFak {
					item.Notes = "source script not found on disk"
				} else {
					item.Notes = "pending migration"
				}
			}
		} else {
			item.Status = StatusUnmigrated
			if !sourceInFak {
				item.Notes = "source script not found on disk"
			} else {
				item.Notes = "pending migration"
			}
		}
	}

	report := Report{
		Total: len(audited),
		Items: audited,
	}

	cohortMap := make(map[string]*CohortSummary)
	var cohortOrder []string

	for _, item := range audited {
		switch item.Status {
		case StatusMigrated:
			report.Migrated++
		case StatusPartial:
			report.Partial++
		case StatusUnmigrated:
			report.Unmigrated++
		}

		cs, exists := cohortMap[item.Cohort]
		if !exists {
			cs = &CohortSummary{
				Name:      item.Cohort,
				TargetPkg: item.TargetPkg,
			}
			cohortMap[item.Cohort] = cs
			cohortOrder = append(cohortOrder, item.Cohort)
		}
		cs.Total++
		switch item.Status {
		case StatusMigrated:
			cs.Migrated++
		case StatusPartial:
			cs.Partial++
		case StatusUnmigrated:
			cs.Unmigrated++
		}
	}

	if report.Total > 0 {
		report.Percent = math.Round((float64(report.Migrated)/float64(report.Total))*1000.0) / 10.0
	}

	for _, name := range cohortOrder {
		cs := cohortMap[name]
		if cs.Total > 0 {
			cs.Percent = math.Round((float64(cs.Migrated)/float64(cs.Total))*1000.0) / 10.0
		}
		report.Cohorts = append(report.Cohorts, *cs)
	}

	return report
}

// NextCandidates returns the next unmigrated or partial candidates in priority order.
func NextCandidates(report Report, count int, cohortFilter string) []Item {
	var candidates []Item
	filterLower := strings.ToLower(strings.TrimSpace(cohortFilter))

	for _, item := range report.Items {
		if item.Status != StatusMigrated {
			if filterLower != "" {
				if !strings.Contains(strings.ToLower(item.Cohort), filterLower) &&
					!strings.Contains(strings.ToLower(item.TargetPkg), filterLower) {
					continue
				}
			}
			candidates = append(candidates, item)
		}
	}

	if count > 0 && len(candidates) > count {
		candidates = candidates[:count]
	}

	return candidates
}

// AuditBoundary walks platform/ in fak-private and verifies 5-Gate Go import boundaries:
//   - No imports of github.com/anthony-chaudhary/fak/internal/*
//   - Imports from github.com/anthony-chaudhary/fak must start with github.com/anthony-chaudhary/fak/pkg/
func AuditBoundary(privateRoot string) ([]BoundaryViolation, error) {
	if privateRoot == "" {
		return nil, fmt.Errorf("privateRoot must not be empty")
	}
	platformDir := filepath.Join(privateRoot, "platform")
	if fi, err := os.Stat(platformDir); err != nil || !fi.IsDir() {
		return nil, fmt.Errorf("platform directory not found at %s: %w", platformDir, err)
	}

	var violations []BoundaryViolation
	fset := token.NewFileSet()

	err := filepath.WalkDir(platformDir, func(p string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return nil
		}
		if d.IsDir() {
			name := d.Name()
			if strings.HasPrefix(name, ".") || strings.HasPrefix(name, "_") || name == "vendor" || name == "node_modules" {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(d.Name(), ".go") {
			return nil
		}

		node, parseErr := parser.ParseFile(fset, p, nil, parser.ImportsOnly)
		if parseErr != nil {
			return nil
		}

		relPath, _ := filepath.Rel(privateRoot, p)
		relPath = filepath.ToSlash(relPath)

		for _, imp := range node.Imports {
			impPath := strings.Trim(imp.Path.Value, `"`)
			pos := fset.Position(imp.Pos())

			// Rule 1: No imports of github.com/anthony-chaudhary/fak/internal/*
			if strings.HasPrefix(impPath, "github.com/anthony-chaudhary/fak/internal") {
				violations = append(violations, BoundaryViolation{
					File:       relPath,
					Line:       pos.Line,
					ImportPath: impPath,
					Rule:       "NO_INTERNAL_IMPORT",
					Reason:     fmt.Sprintf("fak-private code imports %q; internal imports are forbidden across the 5-Gate boundary", impPath),
				})
			} else if (impPath == "github.com/anthony-chaudhary/fak" || strings.HasPrefix(impPath, "github.com/anthony-chaudhary/fak/")) &&
				!strings.HasPrefix(impPath, "github.com/anthony-chaudhary/fak/pkg/") {
				// Rule 2: Imports from github.com/anthony-chaudhary/fak must start with github.com/anthony-chaudhary/fak/pkg/
				violations = append(violations, BoundaryViolation{
					File:       relPath,
					Line:       pos.Line,
					ImportPath: impPath,
					Rule:       "PKG_ONLY_IMPORT",
					Reason:     fmt.Sprintf("fak-private code imports %q; imports from fak must start with github.com/anthony-chaudhary/fak/pkg/", impPath),
				})
			}
		}
		return nil
	})

	if err != nil {
		return nil, err
	}

	sort.Slice(violations, func(i, j int) bool {
		if violations[i].File != violations[j].File {
			return violations[i].File < violations[j].File
		}
		return violations[i].Line < violations[j].Line
	})

	return violations, nil
}

// Scaffold generates a target Go skeleton in fak-private/platform/<cohort>/... with proper boundary header.
func Scaffold(fakRoot, privateRoot string, item Item, dryRun bool) ([]string, error) {
	if privateRoot == "" {
		return nil, fmt.Errorf("privateRoot must not be empty")
	}
	if item.TargetPath == "" {
		return nil, fmt.Errorf("item target path must not be empty")
	}

	targetImpl := filepath.Join(privateRoot, filepath.FromSlash(item.TargetPath))
	targetTest := filepath.Join(privateRoot, filepath.FromSlash(strings.TrimSuffix(item.TargetPath, ".go")+"_test.go"))

	scaffolded := []string{
		filepath.ToSlash(targetImpl),
		filepath.ToSlash(targetTest),
	}

	if dryRun {
		return scaffolded, nil
	}

	if fileExists(targetImpl) {
		return nil, fmt.Errorf("target implementation file already exists: %s", targetImpl)
	}

	pkgName := filepath.Base(filepath.Dir(filepath.FromSlash(item.TargetPath)))
	if pkgName == "." || pkgName == "/" {
		pkgName = filepath.Base(item.TargetPkg)
	}

	typeName := pascalCase(strings.TrimSuffix(filepath.Base(item.TargetPath), ".go"))

	implTmpl := `// Package {{PKG}} provides migrated autonomous factory functionality for fak-private.
// Migrated from legacy {{SOURCE}} under the 5-Gate IP Taxonomy (Gate 1: Factory Test).
//
// 5-Gate Boundary Invariants:
//   - Gate 1: Factory Test & Autonomous Development (STRICTLY PRIVATE)
//   - May import github.com/anthony-chaudhary/fak/pkg/* only.
//   - Must NEVER import github.com/anthony-chaudhary/fak/internal/*.
package {{PKG}}

import (
	"context"
	"fmt"
)

// {{TYPE}}Config holds configuration for {{TYPE}}.
type {{TYPE}}Config struct {
	Workspace string
	DryRun    bool
}

// {{TYPE}} implements the migrated logic for {{SOURCE}}.
type {{TYPE}} struct {
	cfg {{TYPE}}Config
}

// New{{TYPE}} constructs an instance of {{TYPE}}.
func New{{TYPE}}(cfg {{TYPE}}Config) *{{TYPE}} {
	return &{{TYPE}}{cfg: cfg}
}

// Run executes the {{TYPE}} lifecycle.
func (t *{{TYPE}}) Run(ctx context.Context) error {
	if t.cfg.Workspace == "" {
		return fmt.Errorf("{{TYPE}}: workspace root cannot be empty")
	}
	return nil
}
`
	implContent := strings.ReplaceAll(implTmpl, "{{PKG}}", pkgName)
	implContent = strings.ReplaceAll(implContent, "{{SOURCE}}", item.SourcePath)
	implContent = strings.ReplaceAll(implContent, "{{TYPE}}", typeName)

	testTmpl := `package {{PKG}}

import (
	"context"
	"testing"
)

func Test{{TYPE}}_Run(t *testing.T) {
	runner := New{{TYPE}}({{TYPE}}Config{
		Workspace: ".",
		DryRun:    true,
	})
	if err := runner.Run(context.Background()); err != nil {
		t.Fatalf("unexpected error running {{TYPE}}: %v", err)
	}
}
`
	testContent := strings.ReplaceAll(testTmpl, "{{PKG}}", pkgName)
	testContent = strings.ReplaceAll(testContent, "{{TYPE}}", typeName)

	if err := os.MkdirAll(filepath.Dir(targetImpl), 0755); err != nil {
		return nil, fmt.Errorf("failed to create directory for %s: %w", targetImpl, err)
	}

	if err := os.WriteFile(targetImpl, []byte(implContent), 0644); err != nil {
		return nil, fmt.Errorf("failed to write %s: %w", targetImpl, err)
	}

	if !fileExists(targetTest) {
		if err := os.WriteFile(targetTest, []byte(testContent), 0644); err != nil {
			return nil, fmt.Errorf("failed to write %s: %w", targetTest, err)
		}
	}

	return scaffolded, nil
}

func pascalCase(s string) string {
	parts := strings.FieldsFunc(s, func(r rune) bool {
		return r == '_' || r == '-' || r == '.'
	})
	var b strings.Builder
	for _, p := range parts {
		if len(p) == 0 {
			continue
		}
		b.WriteString(strings.ToUpper(p[:1]) + strings.ToLower(p[1:]))
	}
	if b.Len() == 0 {
		return "Component"
	}
	return b.String()
}

const embeddedInventory = `
### Cohort 1: Watchdogs & Session Control (platform/watchdogs)
1. tools/crash_audit.py -> platform/watchdogs/crash_audit.go
2. tools/dos_supervisor_watchdog.py -> platform/watchdogs/supervisor.go
3. tools/fleet_dos_dispatch_watchdog.py -> platform/watchdogs/dispatch_watchdog.go
4. tools/fleet_resume_watchdog.py -> platform/watchdogs/resume_watchdog.go
5. tools/fleet_session_signals.py -> platform/watchdogs/signals.go
6. tools/fleet_sessions.py -> platform/watchdogs/sessions.go
7. tools/fleet_supervisor_watchdog.py -> platform/watchdogs/supervisor_watchdog.go
8. tools/gen_session_effectiveness_svg.py -> platform/watchdogs/effectiveness.go
9. tools/guard_hop_bench.py -> platform/watchdogs/guard_hop_bench.go
10. tools/guard_hop_rsi.py -> platform/watchdogs/guard_hop_rsi.go
11. tools/peek_session.py -> platform/watchdogs/peek.go
12. tools/proc_resource_guard.py -> platform/watchdogs/resource_guard.go
13. tools/resume_resolver.py -> platform/watchdogs/resume_resolver.go
14. tools/resume_sweep.py -> platform/watchdogs/resume_sweep.go
15. tools/resume_watch.py -> platform/watchdogs/resume_watch.go
16. tools/session0_orphan_sweep.py -> platform/watchdogs/orphan_sweep.go
17. tools/session_checkpoint.py -> platform/watchdogs/checkpoint.go
18. tools/stale_work_watchdog.py -> platform/watchdogs/stale_work.go
19. tools/stopped_sessions.py -> platform/watchdogs/stopped_sessions.go
20. tools/vcache_codex_session_extract.py -> platform/watchdogs/vcache_extract.go

### Cohort 2: Autonomous Dispatch & Worktrees (platform/dispatch)
21. tools/dispatch_account_topup.py -> platform/dispatch/topup.go
22. tools/dispatch_glm_docs.py -> platform/dispatch/glm_docs.go
23. tools/dispatch_log_audit.py -> platform/dispatch/log_audit.go
24. tools/dispatch_preflight.py -> platform/dispatch/preflight.go
25. tools/dispatch_status.py -> platform/dispatch/status.go
26. tools/dispatch_throughput.py -> platform/dispatch/throughput.go
27. tools/dispatch_worker.py -> platform/dispatch/worker.go
28. tools/issue_dispatch.py -> platform/dispatch/issue_dispatch.go
29. tools/issue_gardener_worker.py -> platform/dispatch/gardener.go
30. tools/issue_lane_router.py -> platform/dispatch/lane_router.go
31. tools/issue_resolve_dispatch.py -> platform/dispatch/resolve_dispatch.go
32. tools/issue_worker_prompt.py -> platform/dispatch/prompt_renderer.go
33. tools/lane_core.py -> platform/dispatch/lane_core.go
34. tools/lane_yield.py -> platform/dispatch/lane_yield.go
35. tools/launch_admission.py -> platform/dispatch/launch_admission.go
36. tools/learning_debt_dispatch.py -> platform/dispatch/learning_debt.go
37. tools/tier_launch.py -> platform/dispatch/tier_launch.go
38. tools/worker_worktree.py -> platform/dispatch/worker_worktree.go
39. tools/worktree_doctor.py -> platform/dispatch/worktree_doctor.go

### Cohort 3: Scorecard Control Panes & Regression Ratchets (platform/scorecards)
40. tools/behavior_contract_scorecard.py -> platform/scorecards/behavior_contract.go
41. tools/bench_dx_scorecard.py -> platform/scorecards/bench_dx.go
42. tools/bench_signal.py -> platform/scorecards/bench_signal.go
43. tools/claim_repro_scorecard.py -> platform/scorecards/claim_repro.go
44. tools/code_quality_scorecard.py -> platform/scorecards/code_quality.go
45. tools/code_slop_scorecard.py -> platform/scorecards/code_slop.go
46. tools/commit_quality_scorecard.py -> platform/scorecards/commit_quality.go
47. tools/concept_disambiguation_scorecard.py -> platform/scorecards/concept_disambiguation.go
48. tools/cuda_dev_scorecard.py -> platform/scorecards/cuda_dev.go
49. tools/demo_quality_scorecard.py -> platform/scorecards/demo_quality.go
50. tools/demo_robustness_scorecard.py -> platform/scorecards/demo_robustness.go
51. tools/dispositions.py -> platform/scorecards/dispositions.go
52. tools/doc_appeal_scorecard.py -> platform/scorecards/doc_appeal.go
53. tools/docs_scorecard.py -> platform/scorecards/docs.go
54. tools/fleet_accounts.py -> platform/scorecards/fleet_accounts.go
55. tools/fleet_bottleneck.py -> platform/scorecards/fleet_bottleneck.go
56. tools/fleet_control_pane.py -> platform/scorecards/fleet_control_pane.go
57. tools/fleet_top.py -> platform/scorecards/fleet_top.go
58. tools/fleet_trend.py -> platform/scorecards/fleet_trend.go
59. tools/gate_signal.py -> platform/scorecards/gate_signal.go
60. tools/industry_scorecard.py -> platform/scorecards/industry.go
61. tools/intent_literal_scorecard.py -> platform/scorecards/intent_literal.go
62. tools/learning_scorecard.py -> platform/scorecards/learning.go
63. tools/observability_scorecard.py -> platform/scorecards/observability.go
64. tools/persona_fit_scorecard.py -> platform/scorecards/persona_fit.go
65. tools/persona_readiness_scorecard.py -> platform/scorecards/persona_readiness.go
66. tools/popularization_readiness_scorecard.py -> platform/scorecards/popularization.go
67. tools/product_scorecard.py -> platform/scorecards/product.go
68. tools/release_readiness_scorecard.py -> platform/scorecards/release_readiness.go
69. tools/repo_hygiene_scorecard.py -> platform/scorecards/repo_hygiene.go
70. tools/rsi_maturity_scorecard.py -> platform/scorecards/rsi_maturity.go
71. tools/score_signal.py -> platform/scorecards/score_signal.go
72. tools/scorecard_control_pane.py -> platform/scorecards/control_pane.go
73. tools/scorecard_since.py -> platform/scorecards/since.go
74. tools/skill_slop_scorecard.py -> platform/scorecards/skill_slop.go
75. tools/sota_coverage_scorecard.py -> platform/scorecards/sota_coverage.go
76. tools/stability_scorecard.py -> platform/scorecards/stability.go
77. tools/steerability_scorecard.py -> platform/scorecards/steerability.go
78. tools/tooling_quality_scorecard.py -> platform/scorecards/tooling_quality.go
79. tools/trajctl_signal.py -> platform/scorecards/trajctl_signal.go
80. tools/vcache_scorecard_gate.py -> platform/scorecards/vcache_gate.go

### Cohort 4: Cluster & Lab Hardware Nodes (platform/cluster)
81. tools/dgx_swebench_compare.py -> platform/cluster/swebench_compare.go
82. tools/gcp_accel.py -> platform/cluster/gcp_accel.go
83. tools/gcp_bench.py -> platform/cluster/gcp_bench.go
84. tools/gcp_gpu_probe.py -> platform/cluster/gcp_gpu_probe.go
85. tools/gcp_quota_request.py -> platform/cluster/gcp_quota.go
86. tools/glm52_serve_preflight.py -> platform/cluster/glm52_preflight.go
87. tools/glm52_serving_witness.py -> platform/cluster/glm52_witness.go
88. tools/glm52_vllm_agentic_battery.py -> platform/cluster/glm52_vllm.go
89. tools/glm_throughput_record.py -> platform/cluster/glm_throughput.go
90. tools/glm_witness_record.py -> platform/cluster/glm_witness.go
91. tools/qwen36_node_packet.py -> platform/cluster/node_packet.go
92. tools/qwen36_node_reports.py -> platform/cluster/node_reports.go
93. tools/qwen36_node_server.py -> platform/cluster/node_server.go
94. tools/qwen36_perf_gate.py -> platform/cluster/perf_gate.go
95. tools/qwen36_standalone_readiness.py -> platform/cluster/readiness.go
96. tools/qwen36_surface_smoke.py -> platform/cluster/surface_smoke.go
97. tools/qwen36_watch_nodes.py -> platform/cluster/watch_nodes.go
98. tools/receive_node_bench.py -> platform/cluster/receive_bench.go

### Cohort 5: Agent Memory & Context Flow (platform/memsync)
99. tools/context_tape.py -> platform/memsync/context_tape.go
100. tools/ctxcost.py -> platform/memsync/ctxcost.go
`

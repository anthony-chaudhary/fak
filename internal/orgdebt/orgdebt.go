package orgdebt

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

const (
	// Schema is the control-pane schema for organization debt scorecards.
	Schema = "fak-org-debt-scorecard/1"
	// DebtKey is the headline debt counter key.
	DebtKey = "org_debt"
)

// Issue represents a work ticket evaluated for shift-left organization.
type Issue struct {
	Number int      `json:"number"`
	Title  string   `json:"title"`
	Body   string   `json:"body"`
	Labels []string `json:"labels"`
}

// Commit represents a git commit evaluated for trunk and witness hygiene.
type Commit struct {
	SHA          string   `json:"sha"`
	Subject      string   `json:"subject"`
	Parents      []string `json:"parents"`
	FilesTouched []string `json:"files_touched"`
	LinesAdded   int      `json:"lines_added"`
	LinesDeleted int      `json:"lines_deleted"`
	TestLines    int      `json:"test_lines"`
}

// Input carries all evaluated signals into the pure scoring engine.
type Input struct {
	Issues           []Issue  `json:"issues"`
	Commits          []Commit `json:"commits"`
	InternalPackages []string `json:"internal_packages"`
	DeclaredLanes    []string `json:"declared_lanes"`
	Workspace        string   `json:"workspace"`
}

// Evaluate runs the deterministic scoring pass over Input and folds the result
// through pkg/scorecard.Fold.
func Evaluate(input Input) scorecard.Payload {
	var kpis []scorecard.KPI

	kpis = append(kpis, kpiBacklogUnready(input.Issues))
	kpis = append(kpis, kpiScopeOversize(input.Issues))
	kpis = append(kpis, kpiLaneContention(input.Commits))
	kpis = append(kpis, kpiMergeKnots(input.Commits))
	kpis = append(kpis, kpiUnfannedSpines(input.Commits))
	kpis = append(kpis, kpiUnwitnessedCommits(input.Commits))
	kpis = append(kpis, kpiPlatformBlindness(input.Commits))
	kpis = append(kpis, kpiUnindexedLeaves(input.InternalPackages, input.DeclaredLanes))

	msgs := scorecard.Messages{
		Finding:         "organization debt detected: shift-left contract or trunk friction",
		FindingClean:    "zero organization debt: shift-left contracts, lanes, and trunk linear",
		NextAction:      "retire organization debt worst-first: clean merge knots, decompose oversized tickets, and fan out feature spines",
		NextActionClean: "maintain shift-left discipline: pre-create contract gates, dos arbitrate leases, and fast-forward trunk commits",
		Grade:           scorecard.GradeStd,
		ExtraCorpus: map[string]any{
			"total_issues":   len(input.Issues),
			"total_commits":  len(input.Commits),
			"total_packages": len(input.InternalPackages),
		},
	}

	payload := scorecard.Fold(Schema, kpis, DebtKey, nil, msgs)
	if input.Workspace != "" {
		payload.Workspace = input.Workspace
	}
	return payload
}

var (
	headingRE = regexp.MustCompile(`(?mi)^#{1,6}\s*(.+?)\s*$`)
	trailerRE = regexp.MustCompile(`\(fak\s+([a-zA-Z0-9_\-]+)\)`)
)

func extractSections(body string) map[string]bool {
	sections := make(map[string]bool)
	matches := headingRE.FindAllStringSubmatch(body, -1)
	for _, m := range matches {
		if len(m) >= 2 {
			norm := strings.ToLower(strings.TrimSpace(m[1]))
			sections[norm] = true
		}
	}
	return sections
}

func hasLabelWithPrefix(labels []string, prefix string) bool {
	for _, l := range labels {
		if strings.HasPrefix(strings.ToLower(l), prefix) {
			return true
		}
	}
	return false
}

// 1. Backlog Unready Debt: issues missing class, priority, or required worker sections.
func kpiBacklogUnready(issues []Issue) scorecard.KPI {
	k := scorecard.KPI{
		Key:      "backlog_unready",
		Group:    "shift_left",
		PassLine: 100,
	}
	for _, iss := range issues {
		var missing []string
		if !hasLabelWithPrefix(iss.Labels, "class:") {
			missing = append(missing, "label class:*")
		}
		if !hasLabelWithPrefix(iss.Labels, "priority/") {
			missing = append(missing, "label priority/P*")
		}

		secs := extractSections(iss.Body)
		if !secs["current state"] {
			missing = append(missing, "section 'Current state'")
		}
		if !secs["scope"] && !(secs["in scope"] && secs["out of scope"]) {
			missing = append(missing, "section 'Scope'")
		}
		if !secs["done condition"] && !secs["acceptance"] {
			missing = append(missing, "section 'Done condition'")
		}
		if !secs["witness"] {
			missing = append(missing, "section 'Witness'")
		}
		if !secs["likely files"] && !secs["path hints"] && !secs["paths"] && !secs["files"] {
			missing = append(missing, "section 'Likely files'")
		}

		if len(missing) > 0 {
			k.Defects = append(k.Defects, fmt.Sprintf("issue #%d (%s): missing %s", iss.Number, iss.Title, strings.Join(missing, ", ")))
		}
	}

	penalty := float64(len(k.Defects)) * 12.5
	k.Score = clampScore(100.0 - penalty)
	k.Detail = fmt.Sprintf("%d unready issues lacking shift-left contract completeness", len(k.Defects))
	return k
}

// 2. Scope Oversize Debt: tickets with multiple deliverables or missing/multiple witnesses.
func kpiScopeOversize(issues []Issue) scorecard.KPI {
	k := scorecard.KPI{
		Key:      "scope_oversize",
		Group:    "scoping",
		PassLine: 100,
	}
	deliverableRE := regexp.MustCompile(`(?mi)^\s*(?:[-*]|\d+[.)])\s+(?:add|fix|remove|rewrite|refactor|create|build|update|write|implement|replace|delete|migrate|design|extend|wire)\b`)

	for _, iss := range issues {
		lines := strings.Split(iss.Body, "\n")
		deliverableCount := 0
		for _, line := range lines {
			if deliverableRE.MatchString(line) {
				deliverableCount++
			}
		}
		if deliverableCount >= 3 {
			k.Defects = append(k.Defects, fmt.Sprintf("issue #%d (%s): oversized (%d distinct deliverables, split before dispatch)", iss.Number, iss.Title, deliverableCount))
		} else if deliverableCount == 2 {
			k.Soft = append(k.Soft, fmt.Sprintf("issue #%d: 2 deliverables detected (borderline S1/S2)", iss.Number))
		}
	}

	penalty := float64(len(k.Defects)) * 15.0
	k.Score = clampScore(100.0 - penalty)
	k.Detail = fmt.Sprintf("%d oversized issues requiring S0/S1 leaf decomposition", len(k.Defects))
	return k
}

// 3. Lane Contention Debt: concurrent commits touching overlapping subsystem files in same window.
func kpiLaneContention(commits []Commit) scorecard.KPI {
	k := scorecard.KPI{
		Key:      "lane_contention",
		Group:    "concurrency",
		PassLine: 100,
	}

	// Look for commits within 3 revisions of each other touching the same package files without disjoint lanes.
	window := 3
	for i := 0; i < len(commits); i++ {
		for j := i + 1; j < len(commits) && j <= i+window; j++ {
			c1, c2 := commits[i], commits[j]
			overlap := findFileOverlap(c1.FilesTouched, c2.FilesTouched)
			if len(overlap) > 0 && !hasDifferentLaneTrailers(c1.Subject, c2.Subject) {
				k.Defects = append(k.Defects, fmt.Sprintf("contention between %s and %s on files: %s", shortSHA(c1.SHA), shortSHA(c2.SHA), strings.Join(overlap, ", ")))
			}
		}
	}

	penalty := float64(len(k.Defects)) * 10.0
	k.Score = clampScore(100.0 - penalty)
	k.Detail = fmt.Sprintf("%d concurrent lane collisions in recent review horizon", len(k.Defects))
	return k
}

func findFileOverlap(f1, f2 []string) []string {
	seen := make(map[string]bool)
	for _, f := range f1 {
		seen[f] = true
	}
	var overlap []string
	for _, f := range f2 {
		if seen[f] && !isGenericDoc(f) {
			overlap = append(overlap, f)
		}
	}
	return overlap
}

func isGenericDoc(f string) bool {
	return strings.HasSuffix(f, ".md") || strings.HasSuffix(f, ".txt")
}

func hasDifferentLaneTrailers(s1, s2 string) bool {
	m1 := trailerRE.FindStringSubmatch(s1)
	m2 := trailerRE.FindStringSubmatch(s2)
	if len(m1) >= 2 && len(m2) >= 2 {
		return m1[1] != m2[1]
	}
	return false
}

// 4. Merge Knot Debt: non-fast-forward merge commits in trunk history.
func kpiMergeKnots(commits []Commit) scorecard.KPI {
	k := scorecard.KPI{
		Key:      "merge_knots",
		Group:    "trunk_hygiene",
		PassLine: 100,
	}

	for _, c := range commits {
		isMerge := len(c.Parents) > 1 || strings.HasPrefix(strings.ToLower(c.Subject), "merge ")
		if isMerge {
			k.Defects = append(k.Defects, fmt.Sprintf("merge knot at %s: %q (non-linear trunk push)", shortSHA(c.SHA), c.Subject))
		}
	}

	penalty := float64(len(k.Defects)) * 20.0
	k.Score = clampScore(100.0 - penalty)
	k.Detail = fmt.Sprintf("%d merge commits on trunk (target: 0; enforce fast-forward landing)", len(k.Defects))
	return k
}

// 5. Unfanned Spine Debt: S1 feature spine commits without tracking child issues or follow-ons.
func kpiUnfannedSpines(commits []Commit) scorecard.KPI {
	k := scorecard.KPI{
		Key:      "unfanned_spines",
		Group:    "spine_fanout",
		PassLine: 100,
	}

	for _, c := range commits {
		sub := strings.ToLower(c.Subject)
		isFeature := strings.HasPrefix(sub, "feat(")
		isSignificant := c.LinesAdded >= 250
		hasIssueRef := strings.Contains(sub, "#")
		if isFeature && isSignificant && !hasIssueRef {
			k.Defects = append(k.Defects, fmt.Sprintf("unfanned spine %s: %q (+%d lines without linked tracking issue)", shortSHA(c.SHA), c.Subject, c.LinesAdded))
		}
	}

	penalty := float64(len(k.Defects)) * 15.0
	k.Score = clampScore(100.0 - penalty)
	k.Detail = fmt.Sprintf("%d feature spines lacking fanout tracking issues", len(k.Defects))
	return k
}

// 6. Unwitnessed Commit Debt: commits lacking conventional prefix, leaf trailer, or witness diff.
func kpiUnwitnessedCommits(commits []Commit) scorecard.KPI {
	k := scorecard.KPI{
		Key:      "unwitnessed_commits",
		Group:    "witness_fidelity",
		PassLine: 100,
	}

	prefixRE := regexp.MustCompile(`^(feat|fix|docs|test|perf|refactor|chore|ci)\([a-zA-Z0-9_\-]+\):`)

	for _, c := range commits {
		// Ignore merge commits here; handled by merge_knots KPI.
		if len(c.Parents) > 1 || strings.HasPrefix(strings.ToLower(c.Subject), "merge ") {
			continue
		}

		sub := strings.TrimSpace(c.Subject)
		hasPrefix := prefixRE.MatchString(sub)
		hasTrailer := trailerRE.MatchString(sub)

		if !hasPrefix {
			k.Defects = append(k.Defects, fmt.Sprintf("commit %s lacks Conventional-Commits prefix: %q", shortSHA(c.SHA), sub))
		} else if !hasTrailer {
			k.Defects = append(k.Defects, fmt.Sprintf("commit %s lacks (fak <leaf>) lane trailer: %q", shortSHA(c.SHA), sub))
		}

		// Check diff-witness: code changes must touch code/tests
		isCodeClaim := strings.HasPrefix(sub, "feat(") || strings.HasPrefix(sub, "fix(") || strings.HasPrefix(sub, "perf(")
		if isCodeClaim && len(c.FilesTouched) > 0 {
			hasCodeFile := false
			for _, f := range c.FilesTouched {
				if strings.HasSuffix(f, ".go") || strings.HasSuffix(f, ".py") || strings.HasSuffix(f, ".c") || strings.HasSuffix(f, ".cu") {
					hasCodeFile = true
					break
				}
			}
			if !hasCodeFile {
				k.Defects = append(k.Defects, fmt.Sprintf("commit %s makes code claim %q but touched only non-code files", shortSHA(c.SHA), sub))
			}
		}
	}

	penalty := float64(len(k.Defects)) * 10.0
	k.Score = clampScore(100.0 - penalty)
	k.Detail = fmt.Sprintf("%d unwitnessed or non-conforming commits", len(k.Defects))
	return k
}

// 7. Platform Blindness Debt: platform-sensitive files modified without cross-platform test coverage.
func kpiPlatformBlindness(commits []Commit) scorecard.KPI {
	k := scorecard.KPI{
		Key:      "platform_blindness",
		Group:    "shift_left",
		PassLine: 100,
	}

	platformSensitivePkgs := map[string]bool{
		"codetools":    true,
		"vdso":         true,
		"sessionaudit": true,
		"process":      true,
		"childprocess": true,
		"terminal":     true,
	}

	for _, c := range commits {
		touchesSensitive := false
		hasPlatformTest := false

		for _, f := range c.FilesTouched {
			for pkg := range platformSensitivePkgs {
				if strings.Contains(f, "internal/"+pkg+"/") {
					touchesSensitive = true
				}
			}
			if strings.Contains(f, "_windows_test.go") || strings.Contains(f, "_darwin_test.go") || strings.Contains(f, "_unix_test.go") {
				hasPlatformTest = true
			}
		}

		if touchesSensitive && !hasPlatformTest && len(c.FilesTouched) > 1 {
			k.Soft = append(k.Soft, fmt.Sprintf("commit %s touches platform-sensitive subsystem without explicit OS-specific test file", shortSHA(c.SHA)))
		}
	}

	penalty := float64(len(k.Defects)) * 10.0
	k.Score = clampScore(100.0 - penalty)
	k.Detail = fmt.Sprintf("%d platform-sensitive changes lacking shift-left multi-platform tests", len(k.Defects))
	return k
}

// 8. Unindexed Leaves Debt: packages in internal/ missing doc.go or lane registration in dos.toml.
func kpiUnindexedLeaves(internalPkgs []string, declaredLanes []string) scorecard.KPI {
	k := scorecard.KPI{
		Key:      "unindexed_leaves",
		Group:    "architecture",
		PassLine: 100,
	}

	laneSet := make(map[string]bool)
	for _, l := range declaredLanes {
		laneSet[strings.ToLower(l)] = true
	}

	for _, pkg := range internalPkgs {
		name := filepath.Base(pkg)
		if name == "internal" || name == "." {
			continue
		}
		if !laneSet[strings.ToLower(name)] {
			k.Defects = append(k.Defects, fmt.Sprintf("internal package %q has no declared lane in dos.toml", name))
		}
	}

	penalty := float64(len(k.Defects)) * 10.0
	k.Score = clampScore(100.0 - penalty)
	k.Detail = fmt.Sprintf("%d internal packages unindexed in lane taxonomy", len(k.Defects))
	return k
}

// ScanWorkspace inspects the local filesystem and dos.toml to construct an Input.
func ScanWorkspace(root string) (Input, error) {
	in := Input{Workspace: root}

	// 1. Scan internal packages
	internalDir := filepath.Join(root, "internal")
	entries, err := os.ReadDir(internalDir)
	if err == nil {
		for _, e := range entries {
			if e.IsDir() {
				in.InternalPackages = append(in.InternalPackages, e.Name())
			}
		}
	}

	// 2. Parse dos.toml lanes
	dosPath := filepath.Join(root, "dos.toml")
	dosBytes, err := os.ReadFile(dosPath)
	if err == nil {
		in.DeclaredLanes = parseDosLanes(string(dosBytes))
	}

	sort.Strings(in.InternalPackages)
	sort.Strings(in.DeclaredLanes)
	return in, nil
}

func parseDosLanes(content string) []string {
	var lanes []string
	seen := make(map[string]bool)

	// Extract from concurrent = [...] and autopick = [...]
	listRE := regexp.MustCompile(`"(?P<lane>[a-zA-Z0-9_\-]+)"`)
	matches := listRE.FindAllStringSubmatch(content, -1)
	for _, m := range matches {
		if len(m) >= 2 {
			l := m[1]
			if !seen[l] {
				seen[l] = true
				lanes = append(lanes, l)
			}
		}
	}
	return lanes
}

func clampScore(s float64) float64 {
	if s < 0 {
		return 0
	}
	if s > 100 {
		return 100
	}
	return scorecard.Round1(s)
}

func shortSHA(sha string) string {
	if len(sha) > 9 {
		return sha[:9]
	}
	return sha
}

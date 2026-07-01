// Package planaudit audits coarse completion signals in plan documents.
package planaudit

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

const DefaultGlob = "{PLAN,BUILD}-*.md"

var (
	braceRE       = regexp.MustCompile(`\{([^}]*)\}`)
	unitRowRE     = regexp.MustCompile(`^\|\s*\d+\s*\|`)
	unitHeadingRE = regexp.MustCompile(`^#{2,6}\s+\d+(?:\.\d+)*[.)]?(?:\s|[—–-]|$)`)
	headingRE     = regexp.MustCompile(`^#\s+(.*)`)
	shippedRE     = regexp.MustCompile(`(?i)shipped|built|✅|\bdone\b|complete[d]?`)
)

const HeaderLines = 60

type Plan struct {
	ID              string `json:"id"`
	Name            string `json:"name"`
	File            string `json:"file"`
	TotalUnits      int    `json:"total_units"`
	Signal          string `json:"signal"`
	PercentComplete int    `json:"percent_complete"`
	Status          string `json:"status"`
}

type Report struct {
	Counts    map[string]int `json:"counts"`
	Plans     []Plan         `json:"plans"`
	Drift     []string       `json:"drift"`
	WorkUnits WorkUnits      `json:"work_units"`
}

type WorkUnits struct {
	PlanWeighted map[string]any `json:"plan_weighted"`
	TaskWeighted map[string]any `json:"task_weighted"`
}

func ExpandGlob(root, pattern string) ([]string, error) {
	m := braceRE.FindStringSubmatch(pattern)
	var patterns []string
	if m != nil {
		for _, alt := range strings.Split(m[1], ",") {
			patterns = append(patterns, strings.Replace(pattern, m[0], alt, 1))
		}
	} else {
		patterns = []string{pattern}
	}
	seen := map[string]bool{}
	var out []string
	for _, pat := range patterns {
		hits, err := filepath.Glob(filepath.Join(root, filepath.FromSlash(pat)))
		if err != nil {
			return nil, err
		}
		for _, h := range hits {
			if !seen[h] {
				seen[h] = true
				out = append(out, h)
			}
		}
	}
	sort.Strings(out)
	return out, nil
}

func CountUnits(lines []string) int {
	n := 0
	for _, line := range lines {
		if unitRowRE.MatchString(line) || unitHeadingRE.MatchString(line) {
			n++
		}
	}
	return n
}

func AuditPlan(path string) (Plan, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return Plan{}, err
	}
	lines := strings.Split(string(b), "\n")
	name := strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))
	for _, line := range lines {
		if m := headingRE.FindStringSubmatch(line); m != nil {
			name = strings.TrimSpace(m[1])
			break
		}
	}
	end := HeaderLines
	if len(lines) < end {
		end = len(lines)
	}
	header := strings.Join(lines[:end], "\n")
	shipped := shippedRE.MatchString(header)
	percent := 0
	status := "not_started"
	signal := "none"
	if shipped {
		percent = 100
		status = "complete"
		signal = "shipped-marker"
	}
	return Plan{
		ID:              strings.TrimSuffix(filepath.Base(path), filepath.Ext(path)),
		Name:            name,
		File:            filepath.Base(path),
		TotalUnits:      CountUnits(lines),
		Signal:          signal,
		PercentComplete: percent,
		Status:          status,
	}, nil
}

func BuildReport(plans []Plan) Report {
	counts := map[string]int{
		"total_plans": len(plans),
		"complete":    0,
		"in_progress": 0,
		"not_started": 0,
	}
	for _, p := range plans {
		counts[p.Status]++
	}
	sumPct := 0
	totalUnits := 0
	doneUnits := 0
	coveragePlans := 0
	for _, p := range plans {
		sumPct += p.PercentComplete
		if p.TotalUnits > 0 {
			coveragePlans++
			totalUnits += p.TotalUnits
			if p.Status == "complete" {
				doneUnits += p.TotalUnits
			}
		}
	}
	planPct := 0.0
	if len(plans) > 0 {
		planPct = round1(float64(sumPct) / float64(len(plans)))
	}
	taskPct := 0.0
	if totalUnits > 0 {
		taskPct = round1(100 * float64(doneUnits) / float64(totalUnits))
	}
	return Report{
		Counts: counts,
		Plans:  plans,
		Drift:  []string{},
		WorkUnits: WorkUnits{
			PlanWeighted: map[string]any{"pct_complete": planPct, "n_plans": len(plans)},
			TaskWeighted: map[string]any{"pct_complete": taskPct, "total_units": totalUnits, "done_units": doneUnits, "coverage_plans": coveragePlans, "coverage_total": len(plans)},
		},
	}
}

func Collect(root, pattern string) (Report, error) {
	paths, err := ExpandGlob(root, pattern)
	if err != nil {
		return Report{}, err
	}
	plans := make([]Plan, 0, len(paths))
	for _, p := range paths {
		plan, err := AuditPlan(p)
		if err != nil {
			return Report{}, err
		}
		plans = append(plans, plan)
	}
	return BuildReport(plans), nil
}

func RenderMarkdown(report Report, asOf string) string {
	c := report.Counts
	pw := report.WorkUnits.PlanWeighted
	tw := report.WorkUnits.TaskWeighted
	lines := []string{
		"# Plan-completion audit - " + asOf,
		"",
		fmt.Sprintf("**Plans:** %d  .  complete %d  .  in-progress %d  .  not-started %d", c["total_plans"], c["complete"], c["in_progress"], c["not_started"]),
		"",
		"> Fleet keeps a single plan-state surface (the plan docs). Drift is structurally empty; `percent_complete` is a coarse header-marker signal, **not** a verified per-unit census.",
		"",
		"## Work units",
		fmt.Sprintf("- **Plan-weighted:** %v%% over %v plans", pw["pct_complete"], pw["n_plans"]),
		fmt.Sprintf("- **Task-weighted (floor):** %v%% - %v/%v units, coverage %v/%v plans", tw["pct_complete"], tw["done_units"], tw["total_units"], tw["coverage_plans"], tw["coverage_total"]),
		"",
		"## Plans",
		"",
		"| Plan | Units | Signal | % | Status |",
		"|---|---:|---|---:|---|",
	}
	for _, p := range report.Plans {
		lines = append(lines, fmt.Sprintf("| `%s` | %d | %s | %d | %s |", p.File, p.TotalUnits, p.Signal, p.PercentComplete, p.Status))
	}
	lines = append(lines, "")
	return strings.Join(lines, "\n")
}

func MarshalJSON(report Report) ([]byte, error) {
	return json.MarshalIndent(report, "", "  ")
}

func round1(v float64) float64 {
	if v >= 0 {
		return float64(int(v*10+0.5)) / 10
	}
	return float64(int(v*10-0.5)) / 10
}

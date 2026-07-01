// Package toolcoverage audits which load-bearing tools modules have sibling tests.
package toolcoverage

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

const (
	Schema             = "fleet-tool-coverage-audit/1"
	DefaultMinCoverage = 90.0
)

var noTestOK = map[string]bool{"__init__": true}

type Row struct {
	Module      string `json:"module"`
	Tested      bool   `json:"tested"`
	LoadBearing bool   `json:"load_bearing"`
}

type Audit struct {
	TotalModules           int      `json:"total_modules"`
	Tested                 int      `json:"tested"`
	OverallCoveragePct     *float64 `json:"overall_coverage_pct"`
	LoadBearing            int      `json:"load_bearing"`
	LoadBearingTested      int      `json:"load_bearing_tested"`
	LoadBearingCoveragePct *float64 `json:"load_bearing_coverage_pct"`
	LoadBearingUntested    []string `json:"load_bearing_untested"`
	Debt                   int      `json:"debt"`
}

type Payload struct {
	Schema                 string   `json:"schema"`
	OK                     bool     `json:"ok"`
	Verdict                string   `json:"verdict"`
	Reason                 string   `json:"reason"`
	Workspace              string   `json:"workspace"`
	MinCoverage            *float64 `json:"min_coverage"`
	TotalModules           int      `json:"total_modules"`
	Tested                 int      `json:"tested"`
	OverallCoveragePct     *float64 `json:"overall_coverage_pct"`
	LoadBearing            int      `json:"load_bearing"`
	LoadBearingTested      int      `json:"load_bearing_tested"`
	LoadBearingCoveragePct *float64 `json:"load_bearing_coverage_pct"`
	LoadBearingUntested    []string `json:"load_bearing_untested"`
	Debt                   int      `json:"debt"`
}

func FindModuleStems(toolsDir string) ([]string, error) {
	entries, err := os.ReadDir(toolsDir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var out []string
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		if !strings.HasSuffix(name, ".py") || strings.HasSuffix(name, "_test.py") {
			continue
		}
		stem := strings.TrimSuffix(name, ".py")
		if noTestOK[stem] {
			continue
		}
		out = append(out, stem)
	}
	sort.Strings(out)
	return out, nil
}

func FindTestStems(toolsDir string) (map[string]bool, error) {
	entries, err := os.ReadDir(toolsDir)
	if err != nil {
		if os.IsNotExist(err) {
			return map[string]bool{}, nil
		}
		return nil, err
	}
	out := map[string]bool{}
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || !strings.HasSuffix(name, "_test.py") {
			continue
		}
		out[strings.TrimSuffix(name, "_test.py")] = true
	}
	return out, nil
}

func GatherRefsText(skillsDir string, extraFiles ...string) string {
	var chunks []string
	if st, err := os.Stat(skillsDir); err == nil && st.IsDir() {
		_ = filepath.WalkDir(skillsDir, func(path string, d os.DirEntry, err error) error {
			if err != nil || d.IsDir() || d.Name() != "SKILL.md" {
				return nil
			}
			if b, readErr := os.ReadFile(path); readErr == nil {
				chunks = append(chunks, string(b))
			}
			return nil
		})
	}
	for _, f := range extraFiles {
		if b, err := os.ReadFile(f); err == nil {
			chunks = append(chunks, string(b))
		}
	}
	return strings.Join(chunks, "\n")
}

func AuditModules(moduleStems []string, testStems map[string]bool, refsText string) Audit {
	var rows []Row
	for _, stem := range moduleStems {
		rows = append(rows, Row{
			Module:      stem,
			Tested:      testStems[stem],
			LoadBearing: strings.Contains(refsText, stem+".py"),
		})
	}
	total := len(rows)
	tested := 0
	loadBearing := 0
	loadBearingTested := 0
	var loadBearingUntested []string
	for _, r := range rows {
		if r.Tested {
			tested++
		}
		if r.LoadBearing {
			loadBearing++
			if r.Tested {
				loadBearingTested++
			} else {
				loadBearingUntested = append(loadBearingUntested, r.Module)
			}
		}
	}
	sort.Strings(loadBearingUntested)
	return Audit{
		TotalModules:           total,
		Tested:                 tested,
		OverallCoveragePct:     pctPtr(tested, total),
		LoadBearing:            loadBearing,
		LoadBearingTested:      loadBearingTested,
		LoadBearingCoveragePct: pctPtr(loadBearingTested, loadBearing),
		LoadBearingUntested:    loadBearingUntested,
		Debt:                   len(loadBearingUntested),
	}
}

func BuildPayload(root string, a Audit, minCoverage *float64) Payload {
	ok := true
	verdict := "OK"
	reason := ""
	if a.LoadBearingCoveragePct == nil {
		verdict = "NO_LOAD_BEARING_MODULES"
		reason = "no load-bearing tools/ modules found (nothing referenced by a SKILL.md / ci.yml)"
	} else if minCoverage != nil && *a.LoadBearingCoveragePct < *minCoverage {
		ok = false
		verdict = "BELOW_FLOOR"
		reason = fmt.Sprintf("load-bearing test coverage %.1f%% is below the %.1f%% floor - %d load-bearing module(s) have NO sibling test: %s",
			*a.LoadBearingCoveragePct, *minCoverage, a.Debt, strings.Join(firstN(a.LoadBearingUntested, 12), ", "))
	} else {
		reason = fmt.Sprintf("load-bearing test coverage %.1f%% (%d/%d); %d untested",
			*a.LoadBearingCoveragePct, a.LoadBearingTested, a.LoadBearing, a.Debt)
	}
	return Payload{
		Schema:                 Schema,
		OK:                     ok,
		Verdict:                verdict,
		Reason:                 reason,
		Workspace:              root,
		MinCoverage:            minCoverage,
		TotalModules:           a.TotalModules,
		Tested:                 a.Tested,
		OverallCoveragePct:     a.OverallCoveragePct,
		LoadBearing:            a.LoadBearing,
		LoadBearingTested:      a.LoadBearingTested,
		LoadBearingCoveragePct: a.LoadBearingCoveragePct,
		LoadBearingUntested:    a.LoadBearingUntested,
		Debt:                   a.Debt,
	}
}

func Collect(root string, minCoverage *float64) (Payload, error) {
	toolsDir := filepath.Join(root, "tools")
	modules, err := FindModuleStems(toolsDir)
	if err != nil {
		return Payload{}, err
	}
	tests, err := FindTestStems(toolsDir)
	if err != nil {
		return Payload{}, err
	}
	refs := GatherRefsText(filepath.Join(root, ".claude", "skills"), filepath.Join(root, ".github", "workflows", "ci.yml"))
	return BuildPayload(root, AuditModules(modules, tests, refs), minCoverage), nil
}

func Render(p Payload) string {
	lines := []string{
		fmt.Sprintf("tool test-coverage audit: %s (%s)", p.Verdict, ternary(p.OK, "ok", "ACTION")),
		"  " + p.Reason,
		fmt.Sprintf("  modules=%d tested=%d overall=%s%%  load-bearing=%d/%d (%s%%)",
			p.TotalModules, p.Tested, fmtPct(p.OverallCoveragePct), p.LoadBearingTested, p.LoadBearing, fmtPct(p.LoadBearingCoveragePct)),
	}
	if len(p.LoadBearingUntested) > 0 {
		lines = append(lines, "  load-bearing + UNTESTED (add a sibling _test.py):")
		for _, mod := range firstN(p.LoadBearingUntested, 20) {
			lines = append(lines, "    - tools/"+mod+".py")
		}
	}
	return strings.Join(lines, "\n")
}

func MarshalJSON(p Payload) ([]byte, error) {
	return json.MarshalIndent(p, "", "  ")
}

func pctPtr(n, d int) *float64 {
	if d == 0 {
		return nil
	}
	v := float64(n) / float64(d) * 100
	v = mathRound1(v)
	return &v
}

func mathRound1(v float64) float64 {
	if v >= 0 {
		return float64(int(v*10+0.5)) / 10
	}
	return float64(int(v*10-0.5)) / 10
}

func firstN(in []string, n int) []string {
	if len(in) <= n {
		return in
	}
	return in[:n]
}

func fmtPct(v *float64) string {
	if v == nil {
		return "<nil>"
	}
	return fmt.Sprintf("%.1f", *v)
}

func ternary(ok bool, a, b string) string {
	if ok {
		return a
	}
	return b
}

// Package scorecardportfolio audits scorecard discovery surfaces without running detectors.
package scorecardportfolio

import (
	"encoding/json"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

const Schema = "fak-scorecard-portfolio-discovery/1"

type Inventory struct {
	PythonProducers    []string          `json:"python_producers"`
	GoScoreRoutes      []string          `json:"go_score_routes"`
	PythonPaneBindings []string          `json:"python_pane_bindings"`
	GoPaneBindings     []string          `json:"go_pane_bindings"`
	DetectorVersions   map[string]string `json:"detector_versions"`
	BaselineEntries    []string          `json:"baseline_entries"`
	PythonCappedDebt   []string          `json:"python_capped_debt"`
	DuplicateBindings  []string          `json:"duplicate_bindings"`
	NonIntegerBaseline []string          `json:"non_integer_baseline"`
	BoundScoreRoutes   []string          `json:"bound_score_routes"`
}

type Gap struct {
	Rank        int      `json:"rank"`
	Kind        string   `json:"kind"`
	Producer    string   `json:"producer"`
	Reach       int      `json:"reach"`
	Consequence string   `json:"consequence"`
	Provenance  []string `json:"provenance"`
}

type Report struct {
	Schema             string    `json:"schema"`
	CoverageDebt       int       `json:"coverage_debt"`
	CoverageDebtUnit   string    `json:"coverage_debt_unit"`
	DomainDebtIncluded bool      `json:"domain_debt_included"`
	Inventory          Inventory `json:"inventory"`
	Gaps               []Gap     `json:"gaps"`
}

type Config struct {
	PythonToolsDir string
	PythonPane     string
	GoPane         string
	GoScoreRoutes  string
	Baseline       string
}

func DefaultConfig(root string) Config {
	return Config{
		PythonToolsDir: filepath.Join(root, "tools"),
		PythonPane:     filepath.Join(root, "tools", "scorecard_control_pane.py"),
		GoPane:         filepath.Join(root, "internal", "scorecardpane", "controlpane.go"),
		GoScoreRoutes:  filepath.Join(root, "cmd", "fak", "score.go"),
		Baseline:       filepath.Join(root, "tools", "scorecard_baseline.json"),
	}
}

var (
	pyMainRE   = regexp.MustCompile(`(?m)^if\s+__name__\s*==\s*["']__main__["']\s*:`)
	pyKeyRE    = regexp.MustCompile(`["']key["']\s*:\s*["']([^"']+)["']`)
	pySchemaRE = regexp.MustCompile(`(?m)^(?:SCHEMA|SCHEMA_VERSION|DETECTOR_VERSION)\s*(?::[^=]+)?=\s*["']([^"']+)["']`)
)

func Audit(c Config) (Report, error) {
	pyProducers, pyVersions, err := pythonProducers(c.PythonToolsDir)
	if err != nil {
		return Report{}, err
	}
	pyPane, duplicates, err := pythonPaneKeys(c.PythonPane)
	if err != nil {
		return Report{}, err
	}
	goPane, err := goMapKeys(c.GoPane, "Cards", "Key")
	if err != nil {
		return Report{}, err
	}
	routes, err := goMapKeys(c.GoScoreRoutes, "scoreRoutes", "")
	if err != nil {
		return Report{}, err
	}
	baseline, nonInteger, versions, err := readBaseline(c.Baseline)
	if err != nil {
		return Report{}, err
	}
	for k, v := range pyVersions {
		if v != "" {
			versions["python:"+k] = v
		}
	}

	inv := Inventory{PythonProducers: pyProducers, GoScoreRoutes: routes, PythonPaneBindings: pyPane, GoPaneBindings: goPane, DetectorVersions: versions, BaselineEntries: baseline, PythonCappedDebt: cappedPython(c.PythonToolsDir, pyProducers), DuplicateBindings: duplicates, NonIntegerBaseline: nonInteger, BoundScoreRoutes: union(scoreRouteBindings(c.PythonPane), scoreRouteBindings(c.GoPane))}
	gaps := findGaps(inv)
	return Report{Schema: Schema, CoverageDebt: len(gaps), CoverageDebtUnit: "portfolio_gap", DomainDebtIncluded: false, Inventory: inv, Gaps: gaps}, nil
}

func pythonProducers(dir string) ([]string, map[string]string, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, nil, err
	}
	var out []string
	versions := map[string]string{}
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), "scorecard.py") {
			continue
		}
		b, err := os.ReadFile(filepath.Join(dir, e.Name()))
		if err != nil {
			return nil, nil, err
		}
		if !pyMainRE.Match(b) {
			continue
		}
		key := strings.TrimSuffix(e.Name(), "_scorecard.py")
		out = append(out, key)
		if m := pySchemaRE.FindSubmatch(b); len(m) > 0 {
			versions[key] = string(m[1])
		}
	}
	sort.Strings(out)
	return out, versions, nil
}

func pythonPaneKeys(path string) ([]string, []string, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, err
	}
	start := strings.Index(string(b), "SCORECARDS:")
	if start < 0 {
		return nil, nil, fmt.Errorf("%s: SCORECARDS not found", path)
	}
	tail := string(b[start:])
	end := strings.Index(tail, "\n]")
	if end < 0 {
		end = len(tail)
	}
	ms := pyKeyRE.FindAllStringSubmatch(tail[:end], -1)
	out := make([]string, 0, len(ms))
	counts := map[string]int{}
	for _, m := range ms {
		out = append(out, m[1])
		counts[m[1]]++
	}
	var duplicates []string
	for key, count := range counts {
		if count > 1 {
			duplicates = append(duplicates, key)
		}
	}
	sort.Strings(out)
	sort.Strings(duplicates)
	return out, duplicates, nil
}

func goMapKeys(path, varName, field string) ([]string, error) {
	f, err := parser.ParseFile(token.NewFileSet(), path, nil, 0)
	if err != nil {
		return nil, err
	}
	var out []string
	ast.Inspect(f, func(n ast.Node) bool {
		vs, ok := n.(*ast.ValueSpec)
		if !ok {
			return true
		}
		found := false
		for _, name := range vs.Names {
			if name.Name == varName {
				found = true
			}
		}
		if !found {
			return true
		}
		for _, v := range vs.Values {
			cl, ok := v.(*ast.CompositeLit)
			if !ok {
				continue
			}
			for _, elt := range cl.Elts {
				if field == "" {
					if kv, ok := elt.(*ast.KeyValueExpr); ok {
						if s, ok := stringLit(kv.Key); ok {
							out = append(out, s)
						}
					}
					continue
				}
				item, ok := elt.(*ast.CompositeLit)
				if !ok {
					continue
				}
				for _, ie := range item.Elts {
					kv, ok := ie.(*ast.KeyValueExpr)
					if !ok {
						continue
					}
					id, ok := kv.Key.(*ast.Ident)
					if ok && id.Name == field {
						if s, ok := stringLit(kv.Value); ok {
							out = append(out, s)
						}
					}
				}
			}
		}
		return false
	})
	sort.Strings(out)
	return out, nil
}
func stringLit(e ast.Expr) (string, bool) {
	b, ok := e.(*ast.BasicLit)
	if !ok {
		return "", false
	}
	return strings.Trim(b.Value, "\"`"), true
}

func readBaseline(path string) ([]string, []string, map[string]string, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, nil, err
	}
	var raw struct {
		Metrics          map[string]json.RawMessage `json:"metrics"`
		DetectorVersions map[string]string          `json:"detector_versions"`
	}
	if err = json.Unmarshal(b, &raw); err != nil {
		return nil, nil, nil, err
	}
	keys := make([]string, 0, len(raw.Metrics))
	for k := range raw.Metrics {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var nonInteger []string
	for key, value := range raw.Metrics {
		var integer int
		if json.Unmarshal(value, &integer) != nil {
			nonInteger = append(nonInteger, key)
		}
	}
	sort.Strings(nonInteger)
	if raw.DetectorVersions == nil {
		raw.DetectorVersions = map[string]string{}
	}
	return keys, nonInteger, raw.DetectorVersions, nil
}

func scoreRouteBindings(path string) []string {
	body, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	re := regexp.MustCompile(`(?:fak\s+|go run ./cmd/fak\s+)score\s+([a-zA-Z0-9_-]+)`)
	matches := re.FindAllStringSubmatch(string(body), -1)
	var out []string
	for _, match := range matches {
		out = append(out, match[1])
	}
	sort.Strings(out)
	return out
}

func cappedPython(dir string, producers []string) []string {
	var out []string
	for _, producer := range producers {
		body, err := os.ReadFile(filepath.Join(dir, producer+"_scorecard.py"))
		if err == nil && regexp.MustCompile(`(?m)^[^#\n]*(?:debt|findings?|defects?)\s*=\s*(?:max\([^\n]*min\(|min\()`).Match(body) {
			out = append(out, producer)
		}
	}
	sort.Strings(out)
	return out
}
func findGaps(i Inventory) []Gap {
	pyPane := set(i.PythonPaneBindings)
	goPane := set(i.GoPaneBindings)
	boundRoutes := set(i.BoundScoreRoutes)
	base := set(i.BaselineEntries)
	var gaps []Gap
	for _, p := range i.PythonCappedDebt {
		add := func() {}
		_ = add
		gaps = append(gaps, Gap{Kind: "producer_debt_capped", Producer: p, Reach: 3, Consequence: "producer caps debt instead of emitting an unbounded integer", Provenance: []string{"tools/" + p + "_scorecard.py"}})
	}
	for _, p := range i.DuplicateBindings {
		gaps = append(gaps, Gap{Kind: "duplicate_pane_binding", Producer: p, Reach: 2, Consequence: "duplicate binding makes portfolio membership ambiguous", Provenance: []string{"tools/scorecard_control_pane.py:SCORECARDS"}})
	}
	for _, p := range i.NonIntegerBaseline {
		gaps = append(gaps, Gap{Kind: "baseline_debt_non_integer", Producer: p, Reach: 2, Consequence: "baseline debt must be an integer detector unit", Provenance: []string{"tools/scorecard_baseline.json:metrics"}})
	}
	add := func(kind, producer string, reach int, consequence string, provenance ...string) {
		gaps = append(gaps, Gap{Kind: kind, Producer: producer, Reach: reach, Consequence: consequence, Provenance: provenance})
	}
	for _, p := range i.PythonProducers {
		if !pyPane[p] {
			add("python_producer_unbound", p, 3, "executable detector debt is invisible to Python control-pane operators", "tools/"+p+"_scorecard.py", "tools/scorecard_control_pane.py:SCORECARDS")
		}
	}
	for _, p := range i.GoScoreRoutes {
		if !boundRoutes[p] && !goPane[p] && !goPane[strings.ReplaceAll(p, "-", "_")] {
			add("go_score_route_unbound", p, 3, "Go-backed scorecard is absent from the native control-pane portfolio", "cmd/fak/score.go:scoreRoutes", "internal/scorecardpane/controlpane.go:Cards")
		}
	}
	for _, p := range union(i.PythonPaneBindings, i.GoPaneBindings) {
		if pyPane[p] != goPane[p] {
			add("pane_binding_mismatch", p, 2, "Python and Go control panes discover different portfolios", "tools/scorecard_control_pane.py:SCORECARDS", "internal/scorecardpane/controlpane.go:Cards")
		}
		if !base[p] {
			add("baseline_entry_missing", p, 2, "bound detector has no pinned baseline entry", "tools/scorecard_baseline.json:metrics")
		}
		if i.DetectorVersions[p] == "" {
			add("detector_version_unpinned", p, 2, "baseline cannot distinguish detector changes from debt changes", "tools/scorecard_baseline.json:detector_versions")
		}
	}
	sort.SliceStable(gaps, func(a, b int) bool {
		if gaps[a].Reach != gaps[b].Reach {
			return gaps[a].Reach > gaps[b].Reach
		}
		if gaps[a].Kind != gaps[b].Kind {
			return gaps[a].Kind < gaps[b].Kind
		}
		return gaps[a].Producer < gaps[b].Producer
	})
	for n := range gaps {
		gaps[n].Rank = n + 1
	}
	return gaps
}
func set(xs []string) map[string]bool {
	m := map[string]bool{}
	for _, x := range xs {
		m[x] = true
	}
	return m
}
func union(a, b []string) []string {
	m := set(a)
	for _, x := range b {
		m[x] = true
	}
	out := make([]string, 0, len(m))
	for x := range m {
		out = append(out, x)
	}
	sort.Strings(out)
	return out
}

func WriteJSON(w interface{ Write([]byte) (int, error) }, r Report) error {
	e := json.NewEncoder(w)
	e.SetIndent("", "  ")
	return e.Encode(r)
}

var _ fs.FileInfo

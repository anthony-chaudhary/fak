package codedebt

import (
	"encoding/json"
	"fmt"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// ParseDefect turns a raw defect string from a given KPI into a structured Defect.
func ParseDefect(kpi, raw string) Defect {
	raw = strings.TrimSpace(raw)
	d := Defect{
		KPI:        kpi,
		Categories: KPICategories[kpi],
		Raw:        raw,
	}
	if len(d.Categories) == 0 {
		d.Categories = []Category{CategoryUncategorized}
	}

	switch {
	case strings.HasPrefix(raw, "god-file "):
		d.Kind = "god-file"
		rest := strings.TrimPrefix(raw, "god-file ")
		parts := strings.Fields(rest)
		if len(parts) > 0 {
			d.Path = filepath.ToSlash(parts[0])
		}
	case strings.HasPrefix(raw, "god-function "):
		d.Kind = "god-function"
		rest := strings.TrimPrefix(raw, "god-function ")
		parts := strings.Fields(rest)
		if len(parts) > 0 {
			loc := parts[0]
			if idx := strings.Index(loc, ":"); idx != -1 {
				d.Path = filepath.ToSlash(loc[:idx])
			} else {
				d.Path = filepath.ToSlash(loc)
			}
		}
	case strings.HasPrefix(raw, "non-trivial package has no _test.go: "):
		d.Kind = "untested-package"
		pkg := strings.TrimPrefix(raw, "non-trivial package has no _test.go: ")
		d.Path = filepath.ToSlash(strings.TrimSpace(pkg))
		d.Package = d.Path
	case strings.HasPrefix(raw, "zero-assertion test (cannot fail): "):
		d.Kind = "zero-assertion-test"
		rest := strings.TrimPrefix(raw, "zero-assertion test (cannot fail): ")
		parts := strings.Fields(rest)
		if len(parts) > 0 {
			loc := parts[0]
			sub := strings.Split(loc, ":")
			if len(sub) >= 1 {
				d.Path = filepath.ToSlash(sub[0])
			}
			if len(sub) >= 2 {
				if line, err := strconv.Atoi(sub[1]); err == nil {
					d.Line = line
				}
			}
		}
	case strings.HasPrefix(raw, "unformatted (run gofmt -w): "):
		d.Kind = "unformatted"
		p := strings.TrimPrefix(raw, "unformatted (run gofmt -w): ")
		d.Path = filepath.ToSlash(strings.TrimSpace(p))
	case strings.HasPrefix(raw, "vet: "):
		d.Kind = "vet-diagnostic"
		rest := strings.TrimPrefix(raw, "vet: ")
		sub := strings.Split(rest, ":")
		if len(sub) >= 1 {
			d.Path = filepath.ToSlash(strings.TrimSpace(sub[0]))
		}
		if len(sub) >= 2 {
			if line, err := strconv.Atoi(sub[1]); err == nil {
				d.Line = line
			}
		}
	case strings.HasPrefix(raw, "external dependency"):
		d.Kind = "external-dep"
		d.Path = "go.mod"
	case strings.HasPrefix(raw, "go.sum exists"):
		d.Kind = "gosum-present"
		d.Path = "go.sum"
	case strings.HasPrefix(raw, "untagged/double-tagged claim:"):
		d.Kind = "mis-tagged-claim"
		d.Path = "CLAIMS.md"
	case strings.HasPrefix(raw, "build failure:"):
		d.Kind = "build-failure"
	default:
		d.Kind = "misc"
	}

	if d.Package == "" && d.Path != "" {
		d.Package = PackageOf(d.Path)
	}

	return d
}

// PackageOf returns the package / directory path for a given file or package path.
func PackageOf(rel string) string {
	rel = filepath.ToSlash(rel)
	if rel == "go.mod" || rel == "go.sum" || rel == "CLAIMS.md" {
		return "."
	}
	dir := filepath.ToSlash(filepath.Dir(rel))
	if dir == "." || dir == "/" {
		if strings.HasSuffix(rel, ".go") {
			return "."
		}
		return rel
	}
	return dir
}

// Query filters and aggregates defects in a Report based on QueryOptions.
func (r *Report) Query(opts QueryOptions) QueryResult {
	res := QueryResult{
		TotalDebt: r.TotalDebt,
		DebtByKPI: make(map[string]int),
		DebtByCat: make(map[Category]int),
		DebtByPkg: make(map[string]int),
		Defects:   make([]Defect, 0),
	}

	optKPI := strings.ToLower(strings.TrimSpace(opts.KPI))
	optCat := Category(strings.ToLower(strings.TrimSpace(string(opts.Category))))
	optPath := filepath.ToSlash(strings.ToLower(strings.TrimSpace(opts.Path)))
	optPkg := filepath.ToSlash(strings.ToLower(strings.TrimSpace(opts.Package)))
	optSearch := strings.ToLower(strings.TrimSpace(opts.Search))

	for _, d := range r.Defects {
		if opts.Deterministic && d.KPI == "ship_integrity" {
			continue
		}
		if optKPI != "" && strings.ToLower(d.KPI) != optKPI {
			continue
		}
		if optCat != "" {
			catMatch := false
			for _, c := range d.Categories {
				if c == optCat {
					catMatch = true
					break
				}
			}
			if !catMatch {
				continue
			}
		}
		if optPath != "" {
			dPath := strings.ToLower(d.Path)
			if !strings.HasPrefix(dPath, optPath) && !strings.Contains(dPath, optPath) {
				continue
			}
		}
		if optPkg != "" {
			dPkg := strings.ToLower(d.Package)
			if !strings.HasPrefix(dPkg, optPkg) && !strings.Contains(dPkg, optPkg) {
				continue
			}
		}
		if optSearch != "" {
			rawLower := strings.ToLower(d.Raw)
			if !strings.Contains(rawLower, optSearch) {
				continue
			}
		}

		res.Defects = append(res.Defects, d)
		res.DebtByKPI[d.KPI]++
		for _, c := range d.Categories {
			res.DebtByCat[c]++
		}
		if d.Package != "" {
			res.DebtByPkg[d.Package]++
		}
	}

	res.MatchedDebt = len(res.Defects)

	// Sort defects deterministically: by Package, Path, Line, Kind, Raw
	sort.Slice(res.Defects, func(i, j int) bool {
		a, b := res.Defects[i], res.Defects[j]
		if a.Package != b.Package {
			return a.Package < b.Package
		}
		if a.Path != b.Path {
			return a.Path < b.Path
		}
		if a.Line != b.Line {
			return a.Line < b.Line
		}
		if a.Kind != b.Kind {
			return a.Kind < b.Kind
		}
		return a.Raw < b.Raw
	})

	if opts.Limit > 0 && len(res.Defects) > opts.Limit {
		res.Defects = res.Defects[:opts.Limit]
	}

	return res
}

// FormatText formats the query result as human-readable CLI text.
func (res QueryResult) FormatText() string {
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("code-debt query: %d matched defect(s) (of %d total code debt)\n", res.MatchedDebt, res.TotalDebt))

	if len(res.DebtByKPI) > 0 {
		sb.WriteString("\nby KPI:\n")
		var kpis []string
		for k := range res.DebtByKPI {
			kpis = append(kpis, k)
		}
		sort.Strings(kpis)
		for _, k := range kpis {
			sb.WriteString(fmt.Sprintf("  %-20s %3d\n", k, res.DebtByKPI[k]))
		}
	}

	if len(res.DebtByCat) > 0 {
		sb.WriteString("\nby category:\n")
		var cats []string
		for c := range res.DebtByCat {
			cats = append(cats, string(c))
		}
		sort.Strings(cats)
		for _, c := range cats {
			sb.WriteString(fmt.Sprintf("  %-24s %3d\n", c, res.DebtByCat[Category(c)]))
		}
	}

	if len(res.Defects) > 0 {
		sb.WriteString("\ndefects:\n")
		for _, d := range res.Defects {
			catStr := "uncategorized"
			if len(d.Categories) > 0 {
				names := make([]string, len(d.Categories))
				for i, c := range d.Categories {
					names[i] = string(c)
				}
				catStr = strings.Join(names, ",")
			}
			sb.WriteString(fmt.Sprintf("  [%s][%s] %s\n", d.KPI, catStr, d.Raw))
		}
	}

	return sb.String()
}

// FormatSummary formats an aggregated summary of code debt.
func (res QueryResult) FormatSummary() string {
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("code debt summary: %d total debt units\n", res.MatchedDebt))

	sb.WriteString("\ncategories:\n")
	var cats []string
	for c := range res.DebtByCat {
		cats = append(cats, string(c))
	}
	sort.Strings(cats)
	for _, c := range cats {
		cat := Category(c)
		sb.WriteString(fmt.Sprintf("  %-22s : %3d defects (%s)\n", c, res.DebtByCat[cat], CategoryDefinitions[cat]))
	}

	sb.WriteString("\nkpis:\n")
	var kpis []string
	for k := range res.DebtByKPI {
		kpis = append(kpis, k)
	}
	sort.Strings(kpis)
	for _, k := range kpis {
		sb.WriteString(fmt.Sprintf("  %-20s : %3d defects\n", k, res.DebtByKPI[k]))
	}

	if len(res.DebtByPkg) > 0 {
		sb.WriteString("\ntop packages by debt:\n")
		type pkgCount struct {
			pkg   string
			count int
		}
		var pkgCounts []pkgCount
		for p, cnt := range res.DebtByPkg {
			pkgCounts = append(pkgCounts, pkgCount{pkg: p, count: cnt})
		}
		sort.Slice(pkgCounts, func(i, j int) bool {
			if pkgCounts[i].count != pkgCounts[j].count {
				return pkgCounts[i].count > pkgCounts[j].count
			}
			return pkgCounts[i].pkg < pkgCounts[j].pkg
		})
		top := 10
		if len(pkgCounts) < top {
			top = len(pkgCounts)
		}
		for i := 0; i < top; i++ {
			sb.WriteString(fmt.Sprintf("  %-35s : %3d defects\n", pkgCounts[i].pkg, pkgCounts[i].count))
		}
	}

	return sb.String()
}

// ParsePayload parses a standard scorecard JSON payload.
func ParsePayload(data []byte) (*Report, error) {
	var raw struct {
		Workspace string `json:"workspace"`
		Corpus    struct {
			Score          float64        `json:"score"`
			Grade          string         `json:"grade"`
			CodeDebt       int            `json:"code_debt"`
			DebtByCategory map[string]int `json:"debt_by_category"`
			Breakdown      []struct {
				KPI    string `json:"kpi"`
				Score  int    `json:"score"`
				Debt   int    `json:"debt"`
				Detail string `json:"detail"`
			} `json:"breakdown"`
		} `json:"corpus"`
		KPIs []struct {
			KPI     string   `json:"kpi"`
			Score   int      `json:"score"`
			Defects []string `json:"defects"`
			Soft    []string `json:"soft"`
		} `json:"kpis"`
	}

	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("decode scorecard json: %w", err)
	}

	rep := &Report{
		Workspace:    raw.Workspace,
		TotalDebt:    raw.Corpus.CodeDebt,
		Score:        raw.Corpus.Score,
		Grade:        raw.Corpus.Grade,
		DebtByKPI:    make(map[string]int),
		DebtByCat:    make(map[Category]int),
		DebtByPkg:    make(map[string]int),
		Defects:      make([]Defect, 0),
		KPISummaries: make(map[string]KPISummary),
	}

	for cat, count := range raw.Corpus.DebtByCategory {
		rep.DebtByCat[Category(cat)] = count
	}

	for _, b := range raw.Corpus.Breakdown {
		rep.KPISummaries[b.KPI] = KPISummary{
			KPI:        b.KPI,
			Score:      b.Score,
			Debt:       b.Debt,
			Detail:     b.Detail,
			Categories: KPICategories[b.KPI],
		}
	}

	for _, k := range raw.KPIs {
		rep.DebtByKPI[k.KPI] = len(k.Defects)
		for _, defStr := range k.Defects {
			d := ParseDefect(k.KPI, defStr)
			rep.Defects = append(rep.Defects, d)
			if d.Package != "" {
				rep.DebtByPkg[d.Package]++
			}
		}
		rep.SoftSignals = append(rep.SoftSignals, k.Soft...)
	}

	if rep.TotalDebt == 0 && len(rep.Defects) > 0 {
		rep.TotalDebt = len(rep.Defects)
	}

	return rep, nil
}

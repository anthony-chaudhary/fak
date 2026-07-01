package sotacoverage

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/sotamatrix"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

const (
	Schema              = "fak-sota-coverage-scorecard/1"
	MatrixGo            = "internal/sotamatrix/sotamatrix.go"
	UncoveredCap        = 12
	FreshnessWindowDays = 90
)

var KernelPathspecs = []string{
	"internal/compute/*.cu",
	"internal/compute/cpuref.go",
	"internal/compute/cuda.go",
	"internal/compute/cuda_kernels.go",
	"internal/compute/dsa.go",
	"internal/compute/dsa_*.go",
	"internal/compute/metal.go",
	"internal/compute/prefill.go",
	"internal/compute/prefill_*.go",
	"internal/compute/graph_cuda.go",
	"internal/compute/tf32_cuda.go",
	"internal/compute/quant_q4k.go",
	"internal/metalgemm/*",
	"internal/model/*.metal",
	"internal/model/moe*.go",
	"internal/model/awq*.go",
	"internal/model/gptq*.go",
	"internal/model/exl2*.go",
	"internal/model/kv*.go",
	"internal/model/paging*.go",
	"internal/model/quant_*.go",
}

type Row struct {
	Slug        string   `json:"slug"`
	FakPath     string   `json:"fak_path"`
	FakPathFile string   `json:"fak_path_file"`
	PrimaryLink string   `json:"primary_link"`
	Oracle      string   `json:"oracle"`
	FileGlobs   []string `json:"file_globs"`
}

type KPI struct {
	Name       string   `json:"name"`
	Group      string   `json:"group"`
	Hard       bool     `json:"hard"`
	Passed     bool     `json:"passed"`
	Debt       int      `json:"debt"`
	Detail     string   `json:"detail"`
	Items      []string `json:"items"`
	ItemsTotal int      `json:"items_total"`
}

type Corpus struct {
	MatrixRows  int            `json:"matrix_rows"`
	SOTADebt    int            `json:"sota_debt"`
	HardDebt    int            `json:"hard_debt"`
	Grade       string         `json:"grade"`
	DebtByGroup map[string]int `json:"debt_by_group"`
}

type Payload struct {
	Schema    string `json:"schema"`
	Workspace string `json:"workspace"`
	Error     string `json:"error"`
	OK        bool   `json:"ok"`
	Corpus    Corpus `json:"corpus"`
	KPIs      []KPI  `json:"kpis"`
}

func Collect(workspace, today string) Payload {
	root, err := filepath.Abs(workspace)
	if err != nil {
		root = workspace
	}
	if _, err := os.Stat(filepath.Join(root, "go.mod")); err != nil {
		return BuildPayload(root, 0, nil, fmt.Sprintf("not the fak repo root at %s (no go.mod)", root))
	}
	src, err := os.ReadFile(filepath.Join(root, MatrixGo))
	if err != nil {
		return BuildPayload(root, 0, nil, fmt.Sprintf("matrix source %s is missing", MatrixGo))
	}
	return CollectWithOps(root, sotamatrix.Operations(), string(src), today)
}

func CollectWithOps(root string, ops []sotamatrix.Op, source, today string) Payload {
	rows := RowsFromOps(ops)
	kpis := Gather(root, rows, source, today)
	return BuildPayload(root, len(rows), kpis, "")
}

func RowsFromOps(ops []sotamatrix.Op) []Row {
	rows := make([]Row, 0, len(ops))
	for _, op := range ops {
		rows = append(rows, Row{
			Slug:        op.Slug,
			FakPath:     op.FakPath,
			FakPathFile: FirstFakPathFile(op.FakPath),
			PrimaryLink: op.PrimaryLink,
			Oracle:      op.Oracle,
			FileGlobs:   append([]string(nil), op.FileGlobs...),
		})
	}
	return rows
}

func KernelFiles(root string) []string {
	args := append([]string{"ls-files"}, KernelPathspecs...)
	cmd := exec.Command("git", args...)
	windowgate.ConfigureBackgroundCommand(cmd)
	cmd.Dir = root
	out, err := cmd.Output()
	if err != nil {
		return nil
	}
	seen := map[string]bool{}
	var files []string
	for _, line := range strings.Split(string(out), "\n") {
		norm := normalizePath(strings.TrimSpace(line))
		if norm == "" || strings.HasSuffix(norm, "_test.go") || seen[norm] {
			continue
		}
		seen[norm] = true
		files = append(files, norm)
	}
	sort.Strings(files)
	return files
}

func Gather(root string, rows []Row, source, today string) []KPI {
	results := []KPI{
		kpiFakPathExists(root, rows),
		kpiHasPrimaryLink(rows),
		kpiHasOracle(rows),
		kpiTreeCoverage(root, rows),
		kpiFreshness(source, today),
	}
	return results
}

func BuildPayload(workspace string, rows int, kpis []KPI, errText string) Payload {
	debt := 0
	hardDebt := 0
	byGroup := map[string]int{"complete": 0, "honest": 0, "fresh": 0}
	for _, kpi := range kpis {
		debt += kpi.Debt
		if kpi.Hard {
			hardDebt += kpi.Debt
		}
		byGroup[kpi.Group] += kpi.Debt
	}
	return Payload{
		Schema:    Schema,
		Workspace: workspace,
		Error:     errText,
		OK:        errText == "" && hardDebt == 0,
		Corpus: Corpus{
			MatrixRows:  rows,
			SOTADebt:    debt,
			HardDebt:    hardDebt,
			Grade:       GradeLetter(debt),
			DebtByGroup: byGroup,
		},
		KPIs: kpis,
	}
}

func GradeLetter(debt int) string {
	switch {
	case debt <= 0:
		return "A"
	case debt <= 2:
		return "B"
	case debt <= 5:
		return "C"
	case debt <= 10:
		return "D"
	default:
		return "F"
	}
}

func FirstFakPathFile(fakPath string) string {
	s := strings.TrimSpace(fakPath)
	if !strings.HasPrefix(s, "internal/") {
		return ""
	}
	end := len(s)
	for i, r := range s {
		if i == 0 {
			continue
		}
		if (r >= 'A' && r <= 'Z') || (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') ||
			r == '_' || r == '.' || r == '/' || r == '-' || r == ':' {
			continue
		}
		end = i
		break
	}
	token := strings.TrimRight(s[:end], "/")
	if idx := strings.LastIndex(token, ":"); idx >= 0 {
		if allDigits(token[idx+1:]) {
			token = token[:idx]
		}
	}
	return token
}

func CoveredByMatrix(path string, globs []string) bool {
	for _, glob := range globs {
		if globMatch(path, glob) {
			return true
		}
	}
	return false
}

func kpiFakPathExists(root string, rows []Row) KPI {
	var missing []string
	for _, row := range rows {
		if row.FakPathFile == "" {
			missing = append(missing, row.Slug+": FakPath has no parseable path")
			continue
		}
		if _, err := os.Stat(filepath.Join(root, filepath.FromSlash(row.FakPathFile))); err != nil {
			missing = append(missing, row.Slug+": "+row.FakPathFile)
		}
	}
	if len(missing) == 0 {
		return pass("fak_path_exists", "complete", true, fmt.Sprintf("all %d rows point at code that exists", len(rows)))
	}
	return fail("fak_path_exists", "complete", true, len(missing), fmt.Sprintf("%d row(s) point at a missing path", len(missing)), missing)
}

func kpiHasPrimaryLink(rows []Row) KPI {
	var bad []string
	for _, row := range rows {
		link := strings.TrimSpace(row.PrimaryLink)
		if !(strings.HasPrefix(link, "http://") || strings.HasPrefix(link, "https://")) {
			if link == "" {
				link = "(empty)"
			}
			bad = append(bad, row.Slug+": PrimaryLink="+link)
		}
	}
	if len(bad) == 0 {
		return pass("has_primary_link", "complete", true, fmt.Sprintf("all %d rows carry an http(s) SOTA link", len(rows)))
	}
	return fail("has_primary_link", "complete", true, len(bad), fmt.Sprintf("%d row(s) have no http(s) PrimaryLink", len(bad)), bad)
}

func kpiHasOracle(rows []Row) KPI {
	var bad []string
	for _, row := range rows {
		if strings.TrimSpace(row.Oracle) == "" {
			bad = append(bad, row.Slug+": Oracle is empty")
		}
	}
	if len(bad) == 0 {
		return pass("has_oracle", "complete", true, fmt.Sprintf("all %d rows carry a verification oracle", len(rows)))
	}
	return fail("has_oracle", "complete", true, len(bad), fmt.Sprintf("%d row(s) have no Oracle", len(bad)), bad)
}

func kpiTreeCoverage(root string, rows []Row) KPI {
	var globs []string
	for _, row := range rows {
		globs = append(globs, row.FileGlobs...)
	}
	files := KernelFiles(root)
	if len(files) == 0 {
		return fail("tree_coverage", "honest", true, 1, "no kernel files found (cannot evaluate coverage)", nil)
	}
	var uncovered []string
	for _, file := range files {
		if !CoveredByMatrix(file, globs) {
			uncovered = append(uncovered, file)
		}
	}
	if len(uncovered) == 0 {
		return pass("tree_coverage", "honest", true, fmt.Sprintf("all %d kernel files are covered by some row", len(files)))
	}
	return fail("tree_coverage", "honest", true, len(uncovered), fmt.Sprintf("%d/%d kernel files are uncovered by any row (matrix blind spot)", len(uncovered), len(files)), uncovered)
}

func kpiFreshness(source, today string) KPI {
	pdate := ProvenanceDate(source)
	if pdate == "" {
		return pass("freshness", "fresh", false, "no dated provenance note found (freshness not applicable)")
	}
	if today == "" {
		return pass("freshness", "fresh", false, "provenance dated "+pdate+"; pass --today to evaluate staleness")
	}
	days, ok := daysBetween(pdate, today)
	if !ok {
		return pass("freshness", "fresh", false, "provenance "+pdate+" / today "+today+": unparseable")
	}
	if days <= FreshnessWindowDays {
		return pass("freshness", "fresh", false, fmt.Sprintf("matrix provenance %s is %dd old (<= %dd window)", pdate, days, FreshnessWindowDays))
	}
	return fail("freshness", "fresh", false, 1, fmt.Sprintf("matrix provenance %s is %dd old (> %dd window; re-check SOTA)", pdate, days, FreshnessWindowDays), []string{fmt.Sprintf("provenance %s is %dd stale", pdate, days)})
}

func pass(name, group string, hard bool, detail string) KPI {
	return KPI{Name: name, Group: group, Hard: hard, Passed: true, Detail: detail, Items: []string{}}
}

func fail(name, group string, hard bool, debt int, detail string, items []string) KPI {
	total := len(items)
	if len(items) > UncoveredCap {
		items = items[:UncoveredCap]
	}
	return KPI{Name: name, Group: group, Hard: hard, Passed: false, Debt: debt, Detail: detail, Items: append([]string(nil), items...), ItemsTotal: total}
}

var provenanceRE = regexp.MustCompile(`RESEARCH-[A-Za-z0-9-]*?(\d{4})-(\d{2})-(\d{2})`)

func ProvenanceDate(source string) string {
	m := provenanceRE.FindStringSubmatch(source)
	if len(m) != 4 {
		return ""
	}
	return m[1] + "-" + m[2] + "-" + m[3]
}

func Render(payload Payload) string {
	if payload.Error != "" {
		return "error: " + payload.Error
	}
	c := payload.Corpus
	var b strings.Builder
	fmt.Fprintln(&b, "SOTA-coverage scorecard  -  the prior-art matrix, complete + honest")
	fmt.Fprintf(&b, "  grade %s   sota-debt %d (hard %d)   rows %d\n\n", c.Grade, c.SOTADebt, c.HardDebt, c.MatrixRows)
	fmt.Fprintln(&b, "  by group:")
	fmt.Fprintf(&b, "    complete (rows point at real code, link, oracle) debt %d\n", c.DebtByGroup["complete"])
	fmt.Fprintf(&b, "    honest   (no blind spot in the tree)             debt %d\n", c.DebtByGroup["honest"])
	fmt.Fprintf(&b, "    fresh    (provenance within window)              debt %d\n\n", c.DebtByGroup["fresh"])
	fmt.Fprintln(&b, "  KPIs (X = a defect to retire by ADDING the missing thing):")
	for _, kpi := range payload.KPIs {
		mark := "ok"
		if !kpi.Passed {
			mark = "X"
		}
		tag := ""
		if !kpi.Hard {
			tag = " (soft)"
		}
		extra := ""
		if !kpi.Passed {
			extra = fmt.Sprintf("  [+%d]", kpi.Debt)
		}
		fmt.Fprintf(&b, "    %s %-18s%s %s%s\n", mark, kpi.Name, tag, kpi.Detail, extra)
		if !kpi.Passed {
			for _, item := range kpi.Items {
				fmt.Fprintf(&b, "        - %s\n", item)
			}
			if kpi.ItemsTotal > len(kpi.Items) {
				fmt.Fprintf(&b, "        ... and %d more\n", kpi.ItemsTotal-len(kpi.Items))
			}
		}
	}
	if c.HardDebt == 0 {
		fmt.Fprintln(&b, "\n  No HARD sota-debt: every matrix row points at real code with a link + oracle, and no kernel file is a blind spot.")
	}
	return strings.TrimRight(b.String(), "\n")
}

func globMatch(p, glob string) bool {
	p = normalizePath(p)
	glob = normalizePath(glob)
	var b strings.Builder
	b.WriteString("^")
	for _, r := range glob {
		switch r {
		case '*':
			b.WriteString(".*")
		case '?':
			b.WriteString(".")
		default:
			b.WriteString(regexp.QuoteMeta(string(r)))
		}
	}
	b.WriteString("$")
	ok, err := regexp.MatchString(b.String(), p)
	return err == nil && ok
}

func normalizePath(p string) string {
	return strings.ReplaceAll(p, "\\", "/")
}

func allDigits(s string) bool {
	if s == "" {
		return false
	}
	for _, r := range s {
		if r < '0' || r > '9' {
			return false
		}
	}
	return true
}

func daysBetween(older, newer string) (int, bool) {
	a, err := time.Parse("2006-01-02", older)
	if err != nil {
		return 0, false
	}
	b, err := time.Parse("2006-01-02", newer)
	if err != nil {
		return 0, false
	}
	return int(b.Sub(a).Hours() / 24), true
}

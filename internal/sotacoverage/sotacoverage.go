// Package sotacoverage drives `fak sota-coverage-scorecard`: it cross-checks the prior-art
// matrix (internal/sotamatrix) against the REAL tree -- every row points at code that
// exists, carries an http(s) primary link and a verification oracle; every kernel file is
// covered by some row; the matrix provenance is inside the freshness window -- and folds
// the gaps into one sota_debt integer.
//
// The fold/grade/render machinery is NOT re-derived here. This package rides the shared
// pkg/scorecard kernel (Fold + the one grade table + Render/Markdown/Compare) like the rest
// of the scorecard family, so the grade table cannot drift from its siblings and the card
// inherits the standard --json/--markdown/--compare surface instead of a copy-pasted
// skeleton. What lives here is only the matrix rows, the convention probes, and the
// per-card prose.
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
	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

const (
	// Schema is the control-pane schema id.
	Schema = "fak-sota-coverage-scorecard/1"
	// MatrixGo is the matrix source whose provenance note dates the freshness KPI.
	MatrixGo = "internal/sotamatrix/sotamatrix.go"
	// DebtKey is the headline HARD integer the control-pane folds (corpus.sota_debt).
	DebtKey = "sota_debt"
	// FreshnessWindowDays is how long a matrix provenance date stays credible.
	FreshnessWindowDays = 90
)

// KernelPathspecs are the git pathspecs that enumerate the compute/model kernels the matrix
// is supposed to have an opinion about. A file matched here but covered by no row is a
// matrix blind spot.
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

// Row is one prior-art matrix row projected into the fields the card probes.
type Row struct {
	Slug        string   `json:"slug"`
	FakPath     string   `json:"fak_path"`
	FakPathFile string   `json:"fak_path_file"`
	PrimaryLink string   `json:"primary_link"`
	Oracle      string   `json:"oracle"`
	FileGlobs   []string `json:"file_globs"`
}

// Collect reads the matrix at workspace and folds the card. today is an optional
// YYYY-MM-DD the freshness KPI is evaluated against ("" leaves staleness unevaluated).
func Collect(workspace, today string) scorecard.Payload {
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

// CollectWithOps is Collect over an explicit operation set + matrix source, so a test can
// drive the fold without the live matrix.
func CollectWithOps(root string, ops []sotamatrix.Op, source, today string) scorecard.Payload {
	rows := RowsFromOps(ops)
	kpis := Gather(root, rows, source, today)
	return BuildPayload(root, len(rows), kpis, "")
}

// RowsFromOps projects the matrix operations into probe rows.
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

// KernelFiles lists the tracked kernel files the matrix is measured against.
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

// Gather runs every convention probe over the rows, in work-list order.
func Gather(root string, rows []Row, source, today string) []scorecard.KPI {
	return []scorecard.KPI{
		kpiFakPathExists(root, rows),
		kpiHasPrimaryLink(rows),
		kpiHasOracle(rows),
		kpiTreeCoverage(root, rows),
		kpiFreshness(source, today),
	}
}

// BuildPayload folds the probed KPIs into the control-pane payload through the shared
// kernel. errText, when set, is a card-level failure (no repo root, no matrix source): it
// becomes its own zero-scored KPI so the failure is a retirable defect rather than a silent
// empty card.
func BuildPayload(workspace string, rows int, kpis []scorecard.KPI, errText string) scorecard.Payload {
	if errText != "" {
		kpis = append([]scorecard.KPI{{
			Key: "matrix_source", Group: "complete", Score: 0,
			Detail:  errText,
			Defects: []string{"the prior-art matrix could not be read: " + errText},
		}}, kpis...)
	}

	debt, soft := 0, 0
	byGroup := map[string]int{"complete": 0, "honest": 0, "fresh": 0}
	for _, k := range kpis {
		debt += len(k.Defects)
		soft += len(k.Soft)
		byGroup[k.Group] += len(k.Defects)
	}

	finding := "every matrix row points at real code with a link + oracle, and no kernel file is a blind spot"
	next := "hold -- re-run after a matrix row or a kernel file lands; a regression means the matrix drifted from the tree"
	dirty := fmt.Sprintf("%s: the prior-art matrix has drifted from the tree", scorecard.CountNoun(debt, "sota-coverage gap"))
	dirtyNext := "retire the defects below: point every row at real code with an http(s) link + an oracle, and cover every kernel file with a row"

	msgs := scorecard.Messages{
		Grade:           scorecard.GradeStd,
		Finding:         dirty,
		FindingClean:    finding,
		NextAction:      dirtyNext,
		NextActionClean: next,
		ExtraCorpus: map[string]any{
			"matrix_rows": rows,
			// hard_debt is retained as a compatibility alias of sota_debt: under the kernel a
			// SOFT signal can never be debt, so the two integers are the same number.
			"hard_debt":     debt,
			"soft_debt":     soft,
			"debt_by_group": byGroup,
			"error":         errText,
		},
	}
	if errText != "" {
		msgs.Reason = errText
	}

	p := scorecard.Fold(Schema, kpis, DebtKey, nil, msgs)
	p.Workspace = workspace
	return p
}

// FirstFakPathFile pulls the first repo-relative path token out of a matrix FakPath cell.
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

// CoveredByMatrix reports whether path is claimed by any of the matrix globs.
func CoveredByMatrix(path string, globs []string) bool {
	for _, glob := range globs {
		if globMatch(path, glob) {
			return true
		}
	}
	return false
}

// ratioScore is the scale-free adoption percent the kernel folds: the share of probed
// items that pass. An empty population scores 100 (nothing to fix), never 0.
func ratioScore(good, total int) float64 {
	if total <= 0 {
		return 100
	}
	return 100 * float64(good) / float64(total)
}

func kpiFakPathExists(root string, rows []Row) scorecard.KPI {
	var defects []string
	for _, row := range rows {
		if row.FakPathFile == "" {
			defects = append(defects, "matrix row "+row.Slug+" has no parseable path in its FakPath cell -- name a real internal/ file")
			continue
		}
		if _, err := os.Stat(filepath.Join(root, filepath.FromSlash(row.FakPathFile))); err != nil {
			defects = append(defects, "matrix row "+row.Slug+" points at a missing path "+row.FakPathFile+" -- fix the FakPath or land the code")
		}
	}
	k := scorecard.KPI{Key: "fak_path_exists", Group: "complete", Score: ratioScore(len(rows)-len(defects), len(rows))}
	if len(defects) == 0 {
		k.Detail = fmt.Sprintf("all %d rows point at code that exists", len(rows))
		return k
	}
	k.Detail = fmt.Sprintf("%d row(s) point at a missing path", len(defects))
	k.Defects = defects
	return k
}

func kpiHasPrimaryLink(rows []Row) scorecard.KPI {
	var defects []string
	for _, row := range rows {
		link := strings.TrimSpace(row.PrimaryLink)
		if !(strings.HasPrefix(link, "http://") || strings.HasPrefix(link, "https://")) {
			if link == "" {
				link = "(empty)"
			}
			defects = append(defects, "matrix row "+row.Slug+" has no http(s) PrimaryLink (PrimaryLink="+link+") -- cite the SOTA source")
		}
	}
	k := scorecard.KPI{Key: "has_primary_link", Group: "complete", Score: ratioScore(len(rows)-len(defects), len(rows))}
	if len(defects) == 0 {
		k.Detail = fmt.Sprintf("all %d rows carry an http(s) SOTA link", len(rows))
		return k
	}
	k.Detail = fmt.Sprintf("%d row(s) have no http(s) PrimaryLink", len(defects))
	k.Defects = defects
	return k
}

func kpiHasOracle(rows []Row) scorecard.KPI {
	var defects []string
	for _, row := range rows {
		if strings.TrimSpace(row.Oracle) == "" {
			defects = append(defects, "matrix row "+row.Slug+" has no Oracle -- name how the claim is verified")
		}
	}
	k := scorecard.KPI{Key: "has_oracle", Group: "complete", Score: ratioScore(len(rows)-len(defects), len(rows))}
	if len(defects) == 0 {
		k.Detail = fmt.Sprintf("all %d rows carry a verification oracle", len(rows))
		return k
	}
	k.Detail = fmt.Sprintf("%d row(s) have no Oracle", len(defects))
	k.Defects = defects
	return k
}

func kpiTreeCoverage(root string, rows []Row) scorecard.KPI {
	var globs []string
	for _, row := range rows {
		globs = append(globs, row.FileGlobs...)
	}
	files := KernelFiles(root)
	if len(files) == 0 {
		return scorecard.KPI{
			Key: "tree_coverage", Group: "honest", Score: 0,
			Detail:  "no kernel files found (cannot evaluate coverage)",
			Defects: []string{"no kernel file matched KernelPathspecs, so coverage cannot be evaluated -- run from the repo root of a git checkout"},
		}
	}
	var defects []string
	for _, file := range files {
		if !CoveredByMatrix(file, globs) {
			defects = append(defects, "kernel file "+file+" is covered by no matrix row (a matrix blind spot) -- add a row or widen a FileGlobs")
		}
	}
	k := scorecard.KPI{Key: "tree_coverage", Group: "honest", Score: ratioScore(len(files)-len(defects), len(files))}
	if len(defects) == 0 {
		k.Detail = fmt.Sprintf("all %d kernel files are covered by some row", len(files))
		return k
	}
	k.Detail = fmt.Sprintf("%d/%d kernel files are uncovered by any row (matrix blind spot)", len(defects), len(files))
	k.Defects = defects
	return k
}

// kpiFreshness is the card's one SOFT dimension: a stale provenance date is a nudge to
// re-check SOTA, never HARD debt. Under the shared kernel a soft signal can never move the
// debt integer or red the gate, which is exactly the anti-gaming rule this KPI needs -- the
// cheap way to move it is to re-date a note.
func kpiFreshness(source, today string) scorecard.KPI {
	k := scorecard.KPI{Key: "freshness", Group: "fresh", Score: 100}
	pdate := ProvenanceDate(source)
	switch {
	case pdate == "":
		k.Detail = "no dated provenance note found (freshness not applicable)"
		return k
	case today == "":
		k.Detail = "provenance dated " + pdate + "; pass --today to evaluate staleness"
		return k
	}
	days, ok := daysBetween(pdate, today)
	if !ok {
		k.Detail = "provenance " + pdate + " / today " + today + ": unparseable"
		return k
	}
	if days <= FreshnessWindowDays {
		k.Detail = fmt.Sprintf("matrix provenance %s is %dd old (<= %dd window)", pdate, days, FreshnessWindowDays)
		return k
	}
	k.Score = 0
	k.Detail = fmt.Sprintf("matrix provenance %s is %dd old (> %dd window; re-check SOTA)", pdate, days, FreshnessWindowDays)
	k.Soft = []string{fmt.Sprintf("matrix provenance %s is %dd stale (> %dd window) -- re-check SOTA and re-date the provenance note", pdate, days, FreshnessWindowDays)}
	return k
}

var provenanceRE = regexp.MustCompile(`RESEARCH-[A-Za-z0-9-]*?(\d{4})-(\d{2})-(\d{2})`)

// ProvenanceDate pulls the YYYY-MM-DD out of the matrix source's RESEARCH- provenance note.
func ProvenanceDate(source string) string {
	m := provenanceRE.FindStringSubmatch(source)
	if len(m) != 4 {
		return ""
	}
	return m[1] + "-" + m[2] + "-" + m[3]
}

// MarkdownDoc is the per-card prose for the committed snapshot the kernel renders.
func MarkdownDoc(p scorecard.Payload) scorecard.MarkdownDoc {
	return scorecard.MarkdownDoc{
		Title: "fak SOTA-coverage scorecard - is the prior-art matrix complete and honest",
		Description: "fak's deterministic SOTA-coverage scorecard: every prior-art matrix row must point at code " +
			"that exists and carry an http(s) primary link plus a verification oracle, and every kernel file in the " +
			"tree must be covered by some row - folded into one sota_debt integer.",
		Heading: "SOTA-coverage scorecard",
		AutoGen: "Auto-generated by `fak sota-coverage-scorecard --markdown`. Do not hand-edit; re-run the tool.",
		Law: "The law: a matrix row is a CLAIM about prior art, so it must name real code, cite the source, and say " +
			"how the claim is verified - and no kernel file may sit outside every row, because an uncovered file is a " +
			"blind spot the matrix silently claims nothing about.",
		DebtKey:     DebtKey,
		HeaderExtra: fmt.Sprintf(" - %v matrix row(s)", p.Corpus["matrix_rows"]),
	}
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

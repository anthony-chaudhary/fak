package docfreshrsi

import (
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/modver"
)

// CorpusAuditOptions controls the scope of a doc freshness corpus audit.
type CorpusAuditOptions struct {
	Root          string   `json:"root"`
	Globs         []string `json:"globs,omitempty"`
	TargetVersion string   `json:"target_version,omitempty"`
	CheckLinks    bool     `json:"check_links"`
	CheckClaims   bool     `json:"check_claims"`
	CheckModver   bool     `json:"check_modver"`
}

// StalePinFinding records an out-of-date release version pin within a document.
type StalePinFinding struct {
	DocPath string `json:"doc_path"`
	Found   string `json:"found"`
	Target  string `json:"target"`
	Line    int    `json:"line"`
}

// ModverFinding records the derived modver revision of a documentation module.
type ModverFinding struct {
	DocPath string `json:"doc_path"`
	Module  string `json:"module"`
	DocRev  int    `json:"doc_rev"`
}

// CorpusAuditReport records the complete health findings over a documentation corpus.
type CorpusAuditReport struct {
	TotalDocs          int               `json:"total_docs"`
	TotalDefects       int               `json:"total_defects"`
	OrientationMissing []string          `json:"orientation_missing,omitempty"`
	ReadNextMissing    []string          `json:"read_next_missing,omitempty"`
	StalePins          []StalePinFinding `json:"stale_pins,omitempty"`
	DanglingLinks      []string          `json:"dangling_links,omitempty"`
	VersionClaims      []VersionClaim    `json:"version_claims,omitempty"`
	ModverFindings     []ModverFinding   `json:"modver_findings,omitempty"`
	Fresh              bool              `json:"fresh"`
	Scorecard          Scorecard         `json:"scorecard"`
}

// CorpusRefreshReport records the outcome of applying kept mechanical improvements.
type CorpusRefreshReport struct {
	UpdatedDocs []string  `json:"updated_docs"`
	DebtBefore  int       `json:"debt_before"`
	DebtAfter   int       `json:"debt_after"`
	Verdicts    []Verdict `json:"verdicts,omitempty"`
}

// DefaultTargetVersion resolves the current release version from the workspace VERSION file.
func DefaultTargetVersion(root string) string {
	b, err := os.ReadFile(filepath.Join(root, "VERSION"))
	if err != nil {
		return "v0.1.0"
	}
	ver := strings.TrimSpace(string(b))
	if !strings.HasPrefix(ver, "v") {
		ver = "v" + ver
	}
	return ver
}

// ScanCorpus walks root and builds an in-memory Corpus of tracked markdown documents.
// It skips noise directories (_scratch, .git, vendor, node_modules, testdata).
func ScanCorpus(root string, globs []string) (Corpus, error) {
	c := make(Corpus)
	if len(globs) > 0 {
		for _, g := range globs {
			p := g
			if !filepath.IsAbs(p) {
				p = filepath.Join(root, g)
			}
			matches, err := filepath.Glob(p)
			if err != nil {
				continue
			}
			for _, m := range matches {
				rel, err := filepath.Rel(root, m)
				if err != nil {
					continue
				}
				slashRel := filepath.ToSlash(rel)
				if strings.HasSuffix(slashRel, ".md") {
					b, err := os.ReadFile(m)
					if err == nil {
						c[slashRel] = string(b)
					}
				}
			}
		}
		return c, nil
	}

	err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		name := d.Name()
		if d.IsDir() {
			if name == ".git" || name == "_scratch" || name == "vendor" ||
				name == "node_modules" || name == "testdata" || name == "_witnesses" {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(strings.ToLower(name), ".md") {
			return nil
		}
		rel, err := filepath.Rel(root, path)
		if err != nil {
			return nil
		}
		slashRel := filepath.ToSlash(rel)
		b, err := os.ReadFile(path)
		if err == nil {
			c[slashRel] = string(b)
		}
		return nil
	})
	return c, err
}

// AuditCorpus evaluates mechanical defects, broken links, version claims, and modver drift.
func AuditCorpus(root string, c Corpus, tgt Target, opts CorpusAuditOptions) CorpusAuditReport {
	keys := sortedKeys(c)
	debt := Debt(c, tgt)

	report := CorpusAuditReport{
		TotalDocs:    len(keys),
		TotalDefects: debt,
	}

	for _, k := range keys {
		body := c[k]
		if !hasOrientation(body) {
			report.OrientationMissing = append(report.OrientationMissing, k)
		}
		if !hasReadNext(body) {
			report.ReadNextMissing = append(report.ReadNextMissing, k)
		}
		lines := strings.Split(body, "\n")
		for lineIdx, ln := range lines {
			for _, m := range versionRe.FindAllString(ln, -1) {
				if m != tgt.Version {
					report.StalePins = append(report.StalePins, StalePinFinding{
						DocPath: k,
						Found:   m,
						Target:  tgt.Version,
						Line:    lineIdx + 1,
					})
				}
			}
		}

		if opts.CheckClaims {
			claims := ScanVersionClaims(k, body)
			report.VersionClaims = append(report.VersionClaims, claims...)
		}
	}

	if opts.CheckLinks {
		ok, dangling := AuditCorpusLinks(root, c)
		if !ok {
			report.DanglingLinks = dangling
		}
	}

	if opts.CheckModver {
		ledgerPath := filepath.Join(root, "docs", "nightrun", "module-versions.jsonl")
		if ledgerBytes, err := os.ReadFile(ledgerPath); err == nil {
			for _, k := range keys {
				if mod, ok := modver.DocModule(k); ok {
					if rev, found := modver.DocRevFromLedger(ledgerBytes, k); found {
						report.ModverFindings = append(report.ModverFindings, ModverFinding{
							DocPath: k,
							Module:  mod,
							DocRev:  rev,
						})
					}
				}
			}
		}
	}

	report.Fresh = (report.TotalDefects == 0 && len(report.DanglingLinks) == 0)

	report.Scorecard = Scorecard{
		Name:  "doc_freshness_debt",
		Value: float64(report.TotalDefects),
		Grade: "fresh",
		Components: []ScoreComponent{
			{Name: "total_docs", Value: float64(report.TotalDocs), Unit: "docs"},
			{Name: "total_defects", Value: float64(report.TotalDefects), Unit: "defects"},
			{Name: "orientation_missing", Value: float64(len(report.OrientationMissing)), Unit: "docs"},
			{Name: "read_next_missing", Value: float64(len(report.ReadNextMissing)), Unit: "docs"},
			{Name: "stale_pins", Value: float64(len(report.StalePins)), Unit: "pins"},
			{Name: "dangling_links", Value: float64(len(report.DanglingLinks)), Unit: "refs"},
			{Name: "unpointed_claims", Value: float64(len(report.VersionClaims)), Unit: "claims"},
		},
	}
	if !report.Fresh {
		report.Scorecard.Grade = "stale"
	}

	return report
}

// AuditCorpusLinks verifies internal links and anchors across the corpus with disk fallback.
func AuditCorpusLinks(root string, c Corpus) (bool, []string) {
	ok := true
	var dangling []string
	for _, k := range sortedKeys(c) {
		for _, tgt := range linkTargets(c[k]) {
			if !resolvesWithDisk(c, root, k, tgt) {
				ok = false
				dangling = append(dangling, k+" -> "+tgt)
			}
		}
	}
	return ok, dangling
}

func resolvesWithDisk(c Corpus, root, fromKey, target string) bool {
	if resolves(c, fromKey, target) {
		return true
	}
	if target == "" || isExternal(target) {
		return true
	}
	filePart := target
	if i := strings.IndexByte(target, '#'); i >= 0 {
		filePart = target[:i]
	}
	if filePart == "" {
		return false
	}
	if root == "" {
		return false
	}

	// 1. Check relative to current document's directory
	docDir := filepath.Dir(filepath.Join(root, filepath.FromSlash(fromKey)))
	candidatePath := filepath.Join(docDir, filepath.FromSlash(filePart))
	if _, err := os.Stat(candidatePath); err == nil {
		return true
	}

	// 2. Check relative to repo root
	rootPath := filepath.Join(root, filepath.FromSlash(filePart))
	if _, err := os.Stat(rootPath); err == nil {
		return true
	}

	return false
}

// RefreshCorpus applies kept mechanical fixes through shipgate evaluation and writes kept updates.
func RefreshCorpus(root string, c Corpus, tgt Target, dryRun bool) (CorpusRefreshReport, error) {
	beforeDebt := Debt(c, tgt)
	refreshedCorpus, verdicts := Refresh(c, tgt)
	afterDebt := Debt(refreshedCorpus, tgt)

	var updated []string
	for _, k := range sortedKeys(refreshedCorpus) {
		if refreshedCorpus[k] != c[k] {
			updated = append(updated, k)
			if !dryRun && root != "" {
				fullPath := filepath.Join(root, filepath.FromSlash(k))
				if err := os.WriteFile(fullPath, []byte(refreshedCorpus[k]), 0644); err != nil {
					return CorpusRefreshReport{}, err
				}
			}
		}
	}
	sort.Strings(updated)

	return CorpusRefreshReport{
		UpdatedDocs: updated,
		DebtBefore:  beforeDebt,
		DebtAfter:   afterDebt,
		Verdicts:    verdicts,
	}, nil
}

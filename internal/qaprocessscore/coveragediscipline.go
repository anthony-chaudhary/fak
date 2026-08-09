package qaprocessscore

import (
	"fmt"
	"path"
	"sort"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

// CoverageDisciplineKey is both the KPI key and the qa-process dogfood finding name for the
// statement-coverage + -race ratchet (#3834).
const CoverageDisciplineKey = "coverage_discipline"

// Closed-vocabulary work-list tokens for the coverage_discipline HARD defects.
const (
	// coverageBelowFloorClass: a package under the absolute statement-coverage floor.
	coverageBelowFloorClass = "COVERAGE_BELOW_FLOOR"
	// coverageRegressedClass: a package that dropped below its recorded per-package ratchet.
	coverageRegressedClass = "COVERAGE_REGRESSED"
	// raceUncheckedClass: the SOFT sub-signal token for packages not exercised under -race.
	raceUncheckedClass = "RACE_UNCHECKED"
)

// defaultCoverageEpsilon is the tolerance (percentage points) below a per-package baseline
// before a drop is charged as a regression. It absorbs the float noise in coverage arithmetic
// (a block-count rounding here or there) so an unchanged package never trips the ratchet.
const defaultCoverageEpsilon = 0.05

// CoverageBaseline is the ratchet the current per-package coverage is graded against. It is
// pure data (json-tagged so a committed baseline file unmarshals straight into it); the CLI
// reads the file, the fold never touches I/O.
type CoverageBaseline struct {
	// Floor is a global absolute statement-coverage floor in percent (e.g. 80.0). A package
	// below it is HARD debt. Zero disables the absolute floor.
	Floor float64 `json:"floor"`
	// PerPackage is the per-package ratchet: a package must not drop below its recorded
	// baseline (its prior coverage high-water). A package regressed below its baseline (beyond
	// Epsilon) is HARD debt. A package absent here carries no per-package ratchet.
	PerPackage map[string]float64 `json:"per_package"`
	// Epsilon is the below-baseline tolerance in percentage points; zero -> defaultCoverageEpsilon.
	Epsilon float64 `json:"epsilon"`
}

// CoverageGap is one package failing the coverage ratchet -- below the absolute floor, or
// regressed below its recorded per-package baseline -- in structured form. It is the
// dispatchable half of a coverage_discipline HARD defect: dispatch.go routes each gap to its
// owning package as a GitHub issue, and CoverageDiscipline renders the same gaps into the KPI
// Defect list, so the score and the work list never drift.
type CoverageGap struct {
	Class  string `json:"class"`  // coverageBelowFloorClass | coverageRegressedClass
	Pkg    string `json:"pkg"`    // the owning package (import path == dir), the stable dedup anchor
	Detail string `json:"detail"` // the run-varying tail (percentages) -- NOT part of any dedup key
}

// workItem renders the gap as the same closed-vocabulary "<CLASS> <pkg>: <detail>" work-list
// entry CoverageDiscipline has always emitted, so the KPI Defect strings are unchanged.
func (g CoverageGap) workItem() string {
	return fmt.Sprintf("%s %s: %s", g.Class, g.Pkg, g.Detail)
}

// CoverageGaps returns every package failing the ratchet, sorted by package for determinism:
// below the absolute Floor (worse -- it subsumes a same-package regression report), or dropped
// below its per-package baseline beyond Epsilon. Pure over its inputs; holds no I/O.
func CoverageGaps(cov map[string]float64, base CoverageBaseline) []CoverageGap {
	eps := base.Epsilon
	if eps <= 0 {
		eps = defaultCoverageEpsilon
	}
	pkgs := make([]string, 0, len(cov))
	for p := range cov {
		pkgs = append(pkgs, p)
	}
	sort.Strings(pkgs)

	var gaps []CoverageGap
	for _, p := range pkgs {
		c := cov[p]
		if base.Floor > 0 && c < base.Floor {
			gaps = append(gaps, CoverageGap{Class: coverageBelowFloorClass, Pkg: p,
				Detail: fmt.Sprintf("%.1f%% < floor %.1f%%", c, base.Floor)})
			continue // below-floor subsumes a same-package regression report
		}
		if bl, ok := base.PerPackage[p]; ok && c < bl-eps {
			gaps = append(gaps, CoverageGap{Class: coverageRegressedClass, Pkg: p,
				Detail: fmt.Sprintf("%.1f%% < baseline %.1f%% (regressed %.1f pp)", c, bl, bl-c)})
		}
	}
	return gaps
}

// CoverageDiscipline folds per-package statement coverage against a ratcheted floor into the
// coverage_discipline KPI. A package below the absolute Floor, or regressed below its recorded
// per-package baseline, is HARD debt -- a real, in-tree-mendable coverage gap. Packages present
// in cov but absent from racedPkgs ride as a SOFT race sub-signal (advisory), because -race is
// Linux/CI-only and a missing race read is unmeasured, not dirty.
//
// racedPkgs == nil means "race coverage was not measured": the KPI emits ONE honest SOFT note
// to that effect rather than silently scoring race-clean -- the same anti-fail-open discipline
// that stops ship_integrity scoring 100 when unmeasured (#3833). A non-nil (possibly empty)
// racedPkgs means race WAS measured, so each cov package missing from it is flagged SOFT.
//
// An empty cov scores 100 with a "no coverage data" detail: the fold never manufactures debt it
// did not measure, but it reports the absence rather than hiding it. (The CLI declines to add the
// KPI at all when no profile was supplied, so an unmeasured card omits coverage_discipline rather
// than presenting a hollow 100 -- absence is absence, not a green.)
func CoverageDiscipline(cov map[string]float64, base CoverageBaseline, racedPkgs map[string]bool) scorecard.KPI {
	pkgs := make([]string, 0, len(cov))
	for p := range cov {
		pkgs = append(pkgs, p)
	}
	sort.Strings(pkgs)

	gaps := CoverageGaps(cov, base)
	var defects []string
	for _, g := range gaps {
		defects = append(defects, g.workItem())
	}

	score := 100.0
	detail := "no coverage data"
	if graded := len(pkgs); graded > 0 {
		clean := graded - len(defects)
		score = scorecard.Round1(100 * float64(clean) / float64(graded))
		detail = fmt.Sprintf("%d/%d package(s) meet the coverage ratchet (%.1f%%)", clean, graded, score)
	}

	return scorecard.KPI{
		Key:     CoverageDisciplineKey,
		Group:   "coverage",
		Score:   score,
		Detail:  detail,
		Defects: defects,
		Soft:    raceSoftSignals(pkgs, racedPkgs),
	}
}

// raceSoftSignals returns the SOFT race sub-signal lines. A nil racedPkgs means race was not
// measured (one honest "unmeasured" note); a non-nil set flags each covered package missing
// from it individually. covPkgs is assumed already sorted for stable output.
func raceSoftSignals(covPkgs []string, racedPkgs map[string]bool) []string {
	if racedPkgs == nil {
		return []string{raceUncheckedClass + ": race coverage not measured this run (no -race read supplied)"}
	}
	var soft []string
	for _, p := range covPkgs {
		if !racePackageMeasured(p, racedPkgs) {
			soft = append(soft, fmt.Sprintf("%s %s: not exercised under -race", raceUncheckedClass, p))
		}
	}
	return soft
}

// racePackageMeasured accepts both full import paths from coverprofiles and the
// repo-relative package names operators naturally pass to --raced.
func racePackageMeasured(profilePkg string, racedPkgs map[string]bool) bool {
	if racedPkgs[profilePkg] {
		return true
	}
	for raced := range racedPkgs {
		raced = strings.TrimPrefix(strings.TrimSpace(raced), "./")
		if raced != "" && (profilePkg == raced || strings.HasSuffix(profilePkg, "/"+raced)) {
			return true
		}
	}
	return false
}

// ParseCoverProfile parses a Go -coverprofile (mode: set/count/atomic) into per-package
// statement coverage percentages. A package's coverage is Σ covered statements / Σ total
// statements across its files, where a block's statements count as covered iff its execution
// count is > 0 -- the same arithmetic `go tool cover -func` reports. The package key is the
// directory of the profiled file path (its import path), via path.Dir on git's forward-slash form.
//
// Each data line is `<file>:<startLine>.<col>,<endLine>.<col> <numStmts> <count>`. The leading
// `mode:` line and any malformed line are skipped -- a robust reader never panics on a partial
// profile. An empty or header-only profile yields an empty (non-nil) map.
func ParseCoverProfile(profile string) map[string]float64 {
	type acc struct{ covered, total int }
	byPkg := map[string]*acc{}

	for _, line := range strings.Split(profile, "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "mode:") {
			continue
		}
		// Split file from the position/stmt/count tail on the separator colon. The tail
		// (`S.C,E.C N C`) carries no colon, and our import paths carry none either, so the last
		// colon is the separator on every platform.
		colon := strings.LastIndex(line, ":")
		if colon < 0 {
			continue
		}
		file := line[:colon]
		fields := strings.Fields(line[colon+1:])
		if len(fields) != 3 {
			continue
		}
		nStmt, err1 := strconv.Atoi(fields[1])
		count, err2 := strconv.Atoi(fields[2])
		if err1 != nil || err2 != nil || nStmt < 0 || count < 0 {
			continue
		}
		pkg := path.Dir(file)
		a := byPkg[pkg]
		if a == nil {
			a = &acc{}
			byPkg[pkg] = a
		}
		a.total += nStmt
		if count > 0 {
			a.covered += nStmt
		}
	}

	out := make(map[string]float64, len(byPkg))
	for pkg, a := range byPkg {
		if a.total == 0 {
			continue // a package with zero measured statements has no meaningful percentage
		}
		out[pkg] = scorecard.Round1(100 * float64(a.covered) / float64(a.total))
	}
	return out
}

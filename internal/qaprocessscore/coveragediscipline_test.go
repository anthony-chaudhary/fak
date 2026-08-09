package qaprocessscore

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

// A two-package -coverprofile fixture. Package .../foo has 10 stmts, 8 covered (80%); package
// .../bar has 4 stmts, 1 covered (25%). Note foo is split across two files to prove per-package
// aggregation, and a count>1 block (mode: count) proves "covered iff count>0".
const coverFixture = `mode: count
github.com/x/foo/a.go:1.0,3.0 4 5
github.com/x/foo/a.go:5.0,6.0 2 0
github.com/x/foo/b.go:1.0,4.0 4 1
github.com/x/bar/bar.go:1.0,2.0 1 1
github.com/x/bar/bar.go:4.0,9.0 3 0
`

func TestParseCoverProfile_PerPackageAggregation(t *testing.T) {
	cov := ParseCoverProfile(coverFixture)
	if got, want := cov["github.com/x/foo"], 80.0; got != want {
		t.Errorf("foo coverage = %v, want %v", got, want) // (4+4)/(4+2+4) = 8/10
	}
	if got, want := cov["github.com/x/bar"], 25.0; got != want {
		t.Errorf("bar coverage = %v, want %v", got, want) // 1/(1+3)
	}
	if len(cov) != 2 {
		t.Fatalf("expected 2 packages, got %d: %v", len(cov), cov)
	}
}

func TestParseCoverProfile_SkipsMalformedAndHeader(t *testing.T) {
	prof := `mode: set
this is not a coverage line
github.com/x/ok/ok.go:1.0,2.0 2 1
github.com/x/ok/ok.go:3.0,4.0 notanumber 1
github.com/x/ok/ok.go:5.0,6.0 2 alsobad
`
	cov := ParseCoverProfile(prof)
	// Only the one valid line counts: 2/2 = 100%.
	if got, want := cov["github.com/x/ok"], 100.0; got != want {
		t.Errorf("ok coverage = %v, want %v (malformed lines must be skipped)", got, want)
	}
	if len(cov) != 1 {
		t.Fatalf("expected 1 package, got %v", cov)
	}
}

func TestParseCoverProfile_EmptyIsEmptyNotNil(t *testing.T) {
	for _, p := range []string{"", "mode: set\n", "\n\n"} {
		cov := ParseCoverProfile(p)
		if cov == nil {
			t.Errorf("ParseCoverProfile(%q) = nil, want empty non-nil map", p)
		}
		if len(cov) != 0 {
			t.Errorf("ParseCoverProfile(%q) = %v, want empty", p, cov)
		}
	}
}

func TestCoverageDiscipline_BelowFloorIsHardDebt(t *testing.T) {
	cov := map[string]float64{"github.com/x/foo": 80.0, "github.com/x/bar": 25.0}
	kpi := CoverageDiscipline(cov, CoverageBaseline{Floor: 50.0}, map[string]bool{})
	if len(kpi.Defects) != 1 {
		t.Fatalf("expected exactly 1 below-floor defect, got %v", kpi.Defects)
	}
	d := kpi.Defects[0]
	if !strings.HasPrefix(d, coverageBelowFloorClass) || !strings.Contains(d, "github.com/x/bar") {
		t.Errorf("defect = %q, want %s naming github.com/x/bar", d, coverageBelowFloorClass)
	}
	// foo (80% >= 50 floor) is clean; 1 of 2 packages clean -> score 50.
	if kpi.Score != 50.0 {
		t.Errorf("score = %v, want 50.0", kpi.Score)
	}
	if kpi.Key != CoverageDisciplineKey || kpi.Group != "coverage" {
		t.Errorf("KPI identity = %q/%q, want %q/coverage", kpi.Key, kpi.Group, CoverageDisciplineKey)
	}
}

func TestCoverageDiscipline_RegressedBelowBaselineIsHardDebt(t *testing.T) {
	cov := map[string]float64{"github.com/x/foo": 70.0}
	base := CoverageBaseline{PerPackage: map[string]float64{"github.com/x/foo": 80.0}}
	kpi := CoverageDiscipline(cov, base, map[string]bool{})
	if len(kpi.Defects) != 1 || !strings.HasPrefix(kpi.Defects[0], coverageRegressedClass) {
		t.Fatalf("expected one %s defect, got %v", coverageRegressedClass, kpi.Defects)
	}
	if !strings.Contains(kpi.Defects[0], "github.com/x/foo") {
		t.Errorf("regression defect must name the package: %q", kpi.Defects[0])
	}
}

func TestCoverageDiscipline_WithinEpsilonIsNotRegression(t *testing.T) {
	cov := map[string]float64{"github.com/x/foo": 79.99}
	base := CoverageBaseline{PerPackage: map[string]float64{"github.com/x/foo": 80.0}, Epsilon: 0.05}
	kpi := CoverageDiscipline(cov, base, map[string]bool{})
	if len(kpi.Defects) != 0 {
		t.Errorf("a drop within epsilon must not be a regression, got %v", kpi.Defects)
	}
}

func TestCoverageDiscipline_BelowFloorSubsumesRegression(t *testing.T) {
	// foo is both below the floor AND below its baseline -> a single defect (below-floor), not two.
	cov := map[string]float64{"github.com/x/foo": 40.0}
	base := CoverageBaseline{Floor: 50.0, PerPackage: map[string]float64{"github.com/x/foo": 80.0}}
	kpi := CoverageDiscipline(cov, base, map[string]bool{})
	if len(kpi.Defects) != 1 || !strings.HasPrefix(kpi.Defects[0], coverageBelowFloorClass) {
		t.Fatalf("expected a single below-floor defect (subsuming regression), got %v", kpi.Defects)
	}
}

func TestCoverageDiscipline_CleanScores100(t *testing.T) {
	cov := map[string]float64{"github.com/x/foo": 90.0, "github.com/x/bar": 85.0}
	base := CoverageBaseline{Floor: 80.0, PerPackage: map[string]float64{"github.com/x/foo": 88.0}}
	kpi := CoverageDiscipline(cov, base, map[string]bool{"github.com/x/foo": true, "github.com/x/bar": true})
	if len(kpi.Defects) != 0 {
		t.Errorf("all packages meet floor+baseline, want no defects, got %v", kpi.Defects)
	}
	if kpi.Score != 100.0 {
		t.Errorf("score = %v, want 100", kpi.Score)
	}
	if len(kpi.Soft) != 0 {
		t.Errorf("both packages raced, want no SOFT race notes, got %v", kpi.Soft)
	}
}

func TestCoverageDiscipline_RaceNilIsOneUnmeasuredSoftNote(t *testing.T) {
	cov := map[string]float64{"github.com/x/foo": 90.0}
	kpi := CoverageDiscipline(cov, CoverageBaseline{}, nil)
	if len(kpi.Soft) != 1 || !strings.HasPrefix(kpi.Soft[0], raceUncheckedClass) {
		t.Fatalf("nil racedPkgs must yield one unmeasured SOFT note, got %v", kpi.Soft)
	}
	if !strings.Contains(kpi.Soft[0], "not measured") {
		t.Errorf("unmeasured note should say so: %q", kpi.Soft[0])
	}
}

func TestCoverageDiscipline_RaceMeasuredFlagsMissingPackages(t *testing.T) {
	cov := map[string]float64{"github.com/x/foo": 90.0, "github.com/x/bar": 90.0}
	// bar was raced, foo was not: exactly one per-package SOFT note (for foo), not the
	// unmeasured note, and never a HARD defect (race is advisory).
	kpi := CoverageDiscipline(cov, CoverageBaseline{}, map[string]bool{"github.com/x/bar": true})
	if len(kpi.Soft) != 1 || !strings.Contains(kpi.Soft[0], "github.com/x/foo") {
		t.Fatalf("want one SOFT note naming the unraced foo, got %v", kpi.Soft)
	}
	if len(kpi.Defects) != 0 {
		t.Errorf("race is SOFT-only, must never add HARD defects, got %v", kpi.Defects)
	}
}

func TestCoverageDiscipline_EmptyCovScores100NoData(t *testing.T) {
	kpi := CoverageDiscipline(map[string]float64{}, CoverageBaseline{Floor: 80}, map[string]bool{})
	if kpi.Score != 100.0 || len(kpi.Defects) != 0 {
		t.Errorf("empty coverage must score 100 with no defects, got score=%v defects=%v", kpi.Score, kpi.Defects)
	}
	if !strings.Contains(kpi.Detail, "no coverage data") {
		t.Errorf("empty coverage detail should report the absence, got %q", kpi.Detail)
	}
}

// TestCoverageDiscipline_FoldsIntoCardDebt pins that a coverage defect increments the card's
// qa_process_debt through Compose -- the KPI is wired into the same debt integer as regression_catch.
func TestCoverageDiscipline_FoldsIntoCardDebt(t *testing.T) {
	cov := map[string]float64{"github.com/x/bar": 25.0}
	kpi := CoverageDiscipline(cov, CoverageBaseline{Floor: 80.0}, map[string]bool{"github.com/x/bar": true})
	payload := Compose([]scorecard.KPI{kpi})
	if got := payload.Corpus[DebtKey]; got != 1 {
		t.Fatalf("%s = %v, want 1 (the below-floor defect)", DebtKey, got)
	}
	if payload.Verdict != "ACTION" {
		t.Errorf("verdict = %q, want ACTION when coverage is in debt", payload.Verdict)
	}
}

func TestCoverageDiscipline_RaceAcceptsRepoRelativePackage(t *testing.T) {
	cov := map[string]float64{
		"github.com/anthony-chaudhary/fak/internal/foo": 90,
		"github.com/anthony-chaudhary/fak/internal/bar": 90,
	}
	k := CoverageDiscipline(cov, CoverageBaseline{Floor: 80}, map[string]bool{"./internal/foo": true})
	if len(k.Soft) != 1 || !strings.Contains(k.Soft[0], "internal/bar") {
		t.Fatalf("soft = %v, want only the genuinely unchecked package", k.Soft)
	}
}

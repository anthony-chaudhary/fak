package brittleness

import (
	"fmt"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

// Schema is the control-pane schema id any consumer keys on.
const Schema = "fak-brittleness-scorecard/1"

// DebtKey is the control-pane debt integer. It is HELD AT ZERO by construction in
// v1: every brittleness finding is a history-window / rerun observation, and a
// landed commit cannot be un-shipped on a no-rewrite trunk, so none of it is
// HARD, in-tree-mendable debt. The card gates on nothing; the real signal is
// PressureKey. The key still exists so the control-pane fold reads this card with
// the same duck-typed contract as every other scorecard.
const DebtKey = "brittleness_debt"

// PressureKey is the UNBOUNDED headline: the severity-weighted recurrence sum over
// every observed brittle seam (Σ severity × recurrence weight). Lower is better,
// there is no ceiling and no 0-100 clamp -- a tree with a package that flaked ten
// times reads ten-times heavy, not a saturated "F". This is the number a
// continuous-improvement loop trends, the analog of antipattern_pressure.
const PressureKey = "brittleness_pressure"

// Class is the closed brittleness vocabulary this leaf has a WIRED detector for. A
// value outside the registry is a bug, not a lower-priority bucket -- the same
// closed-vocabulary discipline the kernel applies to a refusal reason or a
// maturity rung. v1 registers exactly the three classes with a real detector today.
type Class string

const (
	// ClassFlakyRetryPass: a test seam that failed once then passed on a same-tree
	// rerun -- non-deterministic, "just happened to work" on the retry. Captured from
	// internal/affectedtests.ClassifyReruns at the moment of exoneration (flaky.go).
	ClassFlakyRetryPass Class = "FLAKY_RETRY_PASS"
	// ClassRecurringFix: a file touched by two or more fix/revert commits in the
	// window -- the earlier fix "got lucky" and did not hold, so the seam keeps
	// regressing (detect.go, DetectRecurringFixes).
	ClassRecurringFix Class = "RECURRING_FIX"
	// ClassRevertedLanding: a landing that was later reverted -- it shipped, then had
	// to be undone (detect.go, DetectReverts).
	ClassRevertedLanding Class = "REVERTED_LANDING"
)

// Group is the coarse brittleness family a class rolls up to.
type Group string

const (
	// GroupLuck: an outcome that only passed by timing or chance.
	GroupLuck Group = "luck"
	// GroupRegression: an outcome that worked, then broke or had to be undone.
	GroupRegression Group = "regression"
)

// Spec is one registered brittleness class: its group, a severity weight (mirroring
// the AGENTIC-DEV-ANTIPATTERNS grade scale 2=tax / 4=wastes-a-session / 8=corrupts
// trunk), a one-line title, and the default remediation phrase a finding carries.
// Edit this table (not a detector's logic) to add or re-weight a class.
type Spec struct {
	Class      Class
	Group      Group
	Severity   int
	Title      string
	Mitigation string
}

// registry is the canonical, ordered class table -- also the render/KPI order: the
// luck family first (a flake poisons the shared fast gate for every peer, the most
// operator-visible loss), then the regression family.
var registry = []Spec{
	{
		Class:      ClassFlakyRetryPass,
		Group:      GroupLuck,
		Severity:   4,
		Title:      "test seam that failed then passed on a same-tree rerun",
		Mitigation: "make the test deterministic (remove the timing/order dependence); until then it poisons the shared fast gate for every peer",
	},
	{
		Class:      ClassRecurringFix,
		Group:      GroupRegression,
		Severity:   4,
		Title:      "file re-fixed by two or more fix/revert commits in the window",
		Mitigation: "the earlier fix did not hold -- find the shared root cause instead of patching the symptom again",
	},
	{
		Class:      ClassRevertedLanding,
		Group:      GroupRegression,
		Severity:   2,
		Title:      "landing that was later reverted",
		Mitigation: "capture why it had to be undone while the seam is fresh, so the re-land does not repeat the defect",
	},
}

// specByClass indexes the registry for O(1) lookup.
var specByClass = func() map[Class]Spec {
	m := make(map[Class]Spec, len(registry))
	for _, s := range registry {
		m[s.Class] = s
	}
	return m
}()

// Registry returns a copy of the ordered class table, so callers (a test, a doc
// generator) can read coverage without reaching into the unexported slice.
func Registry() []Spec { return append([]Spec(nil), registry...) }

// SpecOf returns the registered spec for a class and ok=false for an unregistered value.
func SpecOf(c Class) (Spec, bool) {
	s, ok := specByClass[c]
	return s, ok
}

// Finding is one observed brittle seam. Ref locates it (a package import path, a
// file, or a sha). Weight is the recurrence count -- how many times this seam
// bit (fix commits touching the file, rerun rounds it flaked over) -- and the
// worst-first ordering key. Fresh is the CAPTURE-WHEN-FRESH payload: the exact
// evidence available at detection time (the fix SHAs, the reverted sha, the rerun
// tree-sha) recorded now so no later agent has to re-derive it. Mitigation defaults
// to the class spec's phrase but a detector may override it.
type Finding struct {
	Class      Class    `json:"class"`
	Ref        string   `json:"ref"`
	Detail     string   `json:"detail"`
	Mitigation string   `json:"mitigation"`
	Weight     int      `json:"weight"`
	Fresh      []string `json:"fresh,omitempty"`
}

// line renders a finding as a single closed-vocabulary work-list entry a loop or a
// human can grep: "<CLASS> <ref>: <detail> [captured-fresh: ...] -- <mitigation>".
func (f Finding) line() string {
	mit := f.Mitigation
	if mit == "" {
		if s, ok := specByClass[f.Class]; ok {
			mit = s.Mitigation
		}
	}
	base := fmt.Sprintf("%s %s: %s", f.Class, f.Ref, f.Detail)
	if len(f.Fresh) > 0 {
		base += fmt.Sprintf(" [captured-fresh: %s]", strings.Join(f.Fresh, ", "))
	}
	return base + " -- " + mit
}

// SortFindings orders findings worst-first: by Weight desc, then registry class
// order, then Ref -- so the fold is deterministic and the loudest seam renders first.
func SortFindings(fs []Finding) {
	order := map[Class]int{}
	for i, s := range registry {
		order[s.Class] = i
	}
	sort.SliceStable(fs, func(i, j int) bool {
		a, b := fs[i], fs[j]
		if a.Weight != b.Weight {
			return a.Weight > b.Weight
		}
		if order[a.Class] != order[b.Class] {
			return order[a.Class] < order[b.Class]
		}
		return a.Ref < b.Ref
	})
}

// Pressure is the exported headline math: the severity-weighted recurrence sum over
// findings (Σ spec.Severity × finding.Weight). Unbounded, lower is better. Pure so a
// test or a trend loop can compute it without folding the whole payload.
func Pressure(findings []Finding) int {
	total := 0
	for _, f := range findings {
		sev := 1
		if s, ok := specByClass[f.Class]; ok {
			sev = s.Severity
		}
		w := f.Weight
		if w <= 0 {
			w = 1
		}
		total += sev * w
	}
	return total
}

// Fold turns observed brittle-seam findings into the control-pane Payload via the
// shared scorecard kernel. It is PURE: findings in, payload out -- no disk, no git,
// no clock.
//
// Every finding rides as a SOFT signal (never a Defect), so debt stays 0 and the
// card gates on nothing: a history-window observation is not HARD, in-tree-mendable
// debt (you cannot un-ship a landed commit), and a soft signal can never red a peer's
// gate. The discrimination lives in PressureKey. One KPI is emitted per registered
// class (in registry order) even at zero findings, so the card always shows its full
// coverage surface rather than hiding a clean class.
func Fold(findings []Finding) scorecard.Payload {
	SortFindings(findings)

	byClass := map[Class][]Finding{}
	for _, f := range findings {
		byClass[f.Class] = append(byClass[f.Class], f)
	}

	kpis := make([]scorecard.KPI, 0, len(registry))
	counts := map[Class]int{}
	for _, s := range registry {
		fs := byClass[s.Class]
		counts[s.Class] = len(fs)
		soft := make([]string, 0, len(fs))
		for _, f := range fs {
			soft = append(soft, f.line())
		}
		kpis = append(kpis, scorecard.KPI{
			Key:    kpiKey(s.Class),
			Group:  string(s.Group),
			Score:  classScore(len(fs)),
			Detail: fmt.Sprintf("%s [sev %d]: %d observation(s)", s.Title, s.Severity, len(fs)),
			Soft:   soft,
		})
	}

	total := len(findings)
	pressure := Pressure(findings)
	finding := "no brittle seams observed: no flaky-retry passes, no recurring fixes, no reverted landings in scope"
	next := "hold -- re-run after new commits land or a rerun flakes; capture the seam while it is fresh, before the next agent re-derives it"
	if total > 0 {
		finding = fmt.Sprintf("%s across %s (brittleness_pressure=%d): %s",
			scorecard.CountNoun(total, "brittle seam"), scorecard.CountNoun(coveredClasses(counts), "class"), pressure, topClassSummary(counts))
		next = "work the worklist worst-first (highest recurrence first): de-flake the shared gate, root-cause the recurring fix, capture the revert reason"
	}

	p := scorecard.Fold(Schema, kpis, DebtKey, nil, scorecard.Messages{
		Grade:           scorecard.GradeStd,
		Finding:         finding,
		FindingClean:    finding,
		NextAction:      next,
		NextActionClean: next,
		ExtraCorpus: map[string]any{
			PressureKey:             pressure,
			"pressure_unit":         "severity_weighted_recurrence",
			"seams_observed":        total,
			"classes_registered":    len(registry),
			"classes_with_findings": coveredClasses(counts),
			"flaky_retry_pass":      counts[ClassFlakyRetryPass],
			"recurring_fix":         counts[ClassRecurringFix],
			"reverted_landing":      counts[ClassRevertedLanding],
		},
	})
	return p
}

// --- pure helpers ---------------------------------------------------------------------------

// classScore maps a finding count to a legacy 0-100 KPI score (a monotone proxy: more
// observations always read as a lower score, staying in (0,100]). The debt count is
// held at 0 and the pressure sum is the real signal -- this only feeds the legacy A-F
// grade so a glance still moves when brittleness rises.
func classScore(findings int) float64 {
	if findings <= 0 {
		return 100.0
	}
	return 100.0 / float64(1+findings)
}

func kpiKey(c Class) string {
	return "no_" + strings.ToLower(string(c))
}

func coveredClasses(counts map[Class]int) int {
	n := 0
	for _, c := range counts {
		if c > 0 {
			n++
		}
	}
	return n
}

// topClassSummary renders the per-class counts of the classes that have findings, in
// registry order, e.g. "FLAKY_RETRY_PASS=2, RECURRING_FIX=5".
func topClassSummary(counts map[Class]int) string {
	var parts []string
	for _, s := range registry {
		if counts[s.Class] > 0 {
			parts = append(parts, fmt.Sprintf("%s=%d", s.Class, counts[s.Class]))
		}
	}
	return strings.Join(parts, ", ")
}

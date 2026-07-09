package livecodebench

import (
	"fmt"
	"strings"
)

// Per-problem verdict vocabulary, closed. A row is ungraded until the official
// checker grades its generations; only then may it be pass|fail. The vocabulary
// describes whether a problem was graded, never whether the run may claim a
// result — that gate lives on the Report.
const (
	VerdictUngraded = "ungraded"
	VerdictPass     = "pass"
	VerdictFail     = "fail"
)

var knownVerdicts = map[string]bool{
	VerdictUngraded: true,
	VerdictPass:     true,
	VerdictFail:     true,
}

// ProblemVerdict is one per-problem row of a report: it links a problem's
// question_id to its verdict and the evidence id that backs that verdict, so a
// report reader can trace every row to gradeable evidence. Arm is empty when a
// row is not attributed to a specific raw|fak arm (e.g. a local ungraded
// scaffold before either arm has run).
type ProblemVerdict struct {
	QuestionID string   `json:"question_id"`
	Scenario   Scenario `json:"scenario,omitempty"`
	Arm        string   `json:"arm,omitempty"`
	Verdict    string   `json:"verdict"`
	EvidenceID string   `json:"evidence_id"`
}

// ProblemRowsFromSuite projects each suite problem into an ungraded per-problem
// verdict row, preserving suite order. No grading has happened, so every row's
// verdict is VerdictUngraded and its evidence id names the local, ungraded
// evidence class for that question. A graded run replaces these rows with real
// pass|fail verdicts and official-grading evidence ids.
func ProblemRowsFromSuite(s Suite) []ProblemVerdict {
	rows := make([]ProblemVerdict, 0, len(s.Problems))
	for _, p := range s.Problems {
		rows = append(rows, ProblemVerdict{
			QuestionID: p.QuestionID,
			Scenario:   p.Scenario,
			Verdict:    VerdictUngraded,
			EvidenceID: EvidenceLocalUngraded + ":" + p.QuestionID,
		})
	}
	return rows
}

// RenderReportMarkdown renders a Report as human-readable markdown: an
// evidence-class / claim-boundary banner, a summary table, per-scenario counts,
// the raw-vs-fak arm deltas, and the per-problem verdict rows that link each
// question_id to its verdict and evidence id. It is a pure function of the
// report — no clock, no I/O — so its output is golden-file testable.
func RenderReportMarkdown(r Report) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# LiveCodeBench Run Report\n\n")

	// Evidence-class / claim-boundary banner. These lines are load-bearing: the
	// report never presents a pass rate without its evidence class, and the
	// claim boundary states what still gates a claimable result.
	fmt.Fprintf(&b, "- Generated: `%s`\n", r.GeneratedAt)
	fmt.Fprintf(&b, "- Benchmark: `%s`\n", r.Benchmark)
	if r.Model != "" {
		fmt.Fprintf(&b, "- Model: `%s`\n", r.Model)
	}
	fmt.Fprintf(&b, "- Release: `%s`\n", r.ReleaseVersion)
	if r.StartDate != "" || r.EndDate != "" {
		fmt.Fprintf(&b, "- Date window: `%s .. %s`\n", orPlaceholder(r.StartDate, "-"), orPlaceholder(r.EndDate, "-"))
	}
	fmt.Fprintf(&b, "- Evidence class: `%s`\n", orPlaceholder(r.EvidenceClass, "-"))
	fmt.Fprintf(&b, "- Official harness: required=%t available=%t", r.OfficialHarness.Required, r.OfficialHarness.Available)
	if r.OfficialHarness.Reason != "" {
		fmt.Fprintf(&b, " (%s)", r.OfficialHarness.Reason)
	}
	fmt.Fprintf(&b, "\n")
	fmt.Fprintf(&b, "- Result claim allowed: `%t`\n", r.ResultClaimAllowed)
	if r.ClaimBoundary != "" {
		fmt.Fprintf(&b, "- Claim boundary: %s\n", r.ClaimBoundary)
	}
	fmt.Fprintf(&b, "\n")

	fmt.Fprintf(&b, "## Summary\n\n")
	fmt.Fprintf(&b, "| Metric | Value |\n|---|---|\n")
	fmt.Fprintf(&b, "| Problems | %d |\n", r.Summary.Problems)
	fmt.Fprintf(&b, "| Graded | %d |\n", r.Summary.Graded)
	if r.WindowScore != nil {
		fmt.Fprintf(&b, "| Windowed problems | %d / %d |\n", r.WindowScore.WindowedProblems, r.WindowScore.TotalProblems)
		fmt.Fprintf(&b, "| Full pass@%d | %.4f |\n", r.WindowScore.K, r.WindowScore.FullPassRate)
		fmt.Fprintf(&b, "| Windowed pass@%d | %.4f |\n", r.WindowScore.K, r.WindowScore.WindowedPassRate)
	}
	fmt.Fprintf(&b, "\n")

	if len(r.Summary.Scenarios) > 0 {
		fmt.Fprintf(&b, "### Scenarios\n\n")
		fmt.Fprintf(&b, "| Scenario | Questions |\n|---|---:|\n")
		for _, s := range r.Summary.Scenarios {
			fmt.Fprintf(&b, "| `%s` | %d |\n", s.Scenario, s.Questions)
		}
		fmt.Fprintf(&b, "\n")
	}

	if delta := renderArmDeltas(r.Arms); delta != "" {
		b.WriteString(delta)
	}

	if len(r.Problems) > 0 {
		fmt.Fprintf(&b, "## Per-Problem Verdicts\n\n")
		fmt.Fprintf(&b, "| question_id | scenario | arm | verdict | evidence_id |\n|---|---|---|---|---|\n")
		for _, p := range r.Problems {
			fmt.Fprintf(&b, "| `%s` | %s | %s | %s | `%s` |\n",
				mdReportCell(p.QuestionID), orPlaceholder(string(p.Scenario), "-"),
				orPlaceholder(p.Arm, "-"), orPlaceholder(p.Verdict, "-"), mdReportCell(p.EvidenceID))
		}
		fmt.Fprintf(&b, "\n")
	}

	if len(r.PromotionRequirements) > 0 {
		fmt.Fprintf(&b, "## Promotion Requirements\n\n")
		for _, req := range r.PromotionRequirements {
			fmt.Fprintf(&b, "- %s\n", req)
		}
	}
	return b.String()
}

// renderArmDeltas renders the raw-vs-fak pass-rate deltas per scenario, one row
// per scenario for which at least one arm reported. It returns "" when no arm
// data is present, so a summary-only report omits the section entirely.
func renderArmDeltas(arms []ArmResult) string {
	if len(arms) == 0 {
		return ""
	}
	type cell struct {
		raw, fak       *ArmResult
		sawRaw, sawFak bool
	}
	byScenario := map[string]*cell{}
	order := []string{}
	for i := range arms {
		a := arms[i]
		c := byScenario[string(a.Scenario)]
		if c == nil {
			c = &cell{}
			byScenario[string(a.Scenario)] = c
			order = append(order, string(a.Scenario))
		}
		switch a.Arm {
		case "raw":
			c.raw = &arms[i]
			c.sawRaw = true
		case "fak":
			c.fak = &arms[i]
			c.sawFak = true
		}
	}
	var b strings.Builder
	fmt.Fprintf(&b, "## Arms (raw vs fak)\n\n")
	fmt.Fprintf(&b, "| Scenario | raw pass@1 | fak pass@1 | Δ pass@1 | raw pass@5 | fak pass@5 | Δ pass@5 |\n")
	fmt.Fprintf(&b, "|---|---:|---:|---:|---:|---:|---:|\n")
	for _, name := range order {
		c := byScenario[name]
		fmt.Fprintf(&b, "| `%s` | %s | %s | %s | %s | %s | %s |\n",
			name,
			passCell(c.raw, c.sawRaw, func(a ArmResult) float64 { return a.Pass1 }),
			passCell(c.fak, c.sawFak, func(a ArmResult) float64 { return a.Pass1 }),
			deltaCell(c.raw, c.fak, c.sawRaw && c.sawFak, func(a ArmResult) float64 { return a.Pass1 }),
			passCell(c.raw, c.sawRaw, func(a ArmResult) float64 { return a.Pass5 }),
			passCell(c.fak, c.sawFak, func(a ArmResult) float64 { return a.Pass5 }),
			deltaCell(c.raw, c.fak, c.sawRaw && c.sawFak, func(a ArmResult) float64 { return a.Pass5 }),
		)
	}
	fmt.Fprintf(&b, "\n")
	return b.String()
}

func passCell(a *ArmResult, saw bool, get func(ArmResult) float64) string {
	if !saw || a == nil {
		return "-"
	}
	return fmt.Sprintf("%.4f", get(*a))
}

func deltaCell(raw, fak *ArmResult, both bool, get func(ArmResult) float64) string {
	if !both || raw == nil || fak == nil {
		return "-"
	}
	return fmt.Sprintf("%+.4f", get(*fak)-get(*raw))
}

func mdReportCell(s string) string {
	s = strings.ReplaceAll(s, "|", "\\|")
	s = strings.ReplaceAll(s, "\r\n", " ")
	s = strings.ReplaceAll(s, "\n", " ")
	return s
}

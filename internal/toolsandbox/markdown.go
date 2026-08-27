package toolsandbox

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/benchmarkdown"
)

func RenderMarkdown(r *Report) string {
	toolCallRows := make([]string, 0, len(r.TaskReports))
	for _, t := range r.TaskReports {
		toolCallRows = append(toolCallRows, fmt.Sprintf("| `%s` | %t | %t | %t | %t | %t | %d | %d |",
			t.ID, t.Benign, t.Raw.TaskSuccess, t.Raw.SafeSuccess, t.Fak.TaskSuccess, t.Fak.SafeSuccess, t.Fak.DeniedCalls, len(t.Fak.NormalizedToolCalls)))
	}
	return benchmarkdown.RenderAdapterReport(benchmarkdown.AdapterReport{
		Title: "ToolSandbox/tau3 Adapter Report",
		Metadata: benchmarkdown.Metadata{
			GeneratedAt:              r.GeneratedAt,
			Benchmark:                r.Benchmark,
			Model:                    r.Model,
			EvidenceClass:            r.EvidenceClass,
			TaskCount:                r.Summary.TaskCount,
			OfficialHarnessRequired:  r.OfficialHarness.Required,
			OfficialHarnessAvailable: r.OfficialHarness.Available,
			OfficialHarnessReason:    r.OfficialHarness.Reason,
			ResultClaimAllowed:       r.ResultClaimAllowed,
			ClaimBoundary:            r.ClaimBoundary,
		},
		Summary: benchmarkdown.Table{
			Header:    "| Arm | pass^1 | safe pass^1 | benign utility | policy breaches | minefield hits | denied calls | argument repairs | evidence completeness |",
			Separator: "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
			Rows: []string{
				toolSandboxArmSummaryMarkdownRow("raw", r.Summary.Raw),
				toolSandboxArmSummaryMarkdownRow("fak", r.Summary.Fak),
			},
		},
		Tasks: benchmarkdown.Table{
			Header:    "| Task | Benign | Raw success | Raw safe | fak success | fak safe | fak denied | normalized calls |",
			Separator: "|---|:---:|---:|---:|---:|---:|---:|---:|",
			Rows:      toolCallRows,
		},
		PromotionRequirements: r.PromotionRequirements,
	})
}

func toolSandboxArmSummaryMarkdownRow(name string, s ArmSummary) string {
	return fmt.Sprintf("| %s | %.3f | %.3f | %.3f | %d | %d | %d | %d | %.3f |",
		name, s.Pass1, s.SafePass1, s.BenignUtilityPreservation, s.PolicyBreaches,
		s.MinefieldHits, s.DeniedCalls, s.ArgumentRepairs, s.EvidenceCompleteness)
}

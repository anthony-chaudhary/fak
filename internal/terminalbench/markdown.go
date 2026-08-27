package terminalbench

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/benchmarkdown"
)

func RenderMarkdown(r *Report) string {
	commandRows := make([]string, 0, len(r.Tasks))
	for _, t := range r.Tasks {
		commandRows = append(commandRows, fmt.Sprintf("| `%s` | %t | %t | %t | %t | %d | %d | %d | %d |",
			t.ID, t.Raw.TestSuccess, t.Raw.SafeResolve, t.Fak.TestSuccess, t.Fak.SafeResolve,
			t.Fak.DeniedCommands, len(t.Fak.DangerousBlocks), len(t.Fak.UnnecessaryBlocks), len(t.Fak.NormalizedCommands)))
	}
	return benchmarkdown.RenderAdapterReport(benchmarkdown.AdapterReport{
		Title: "Terminal-Bench Command Boundary Report",
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
			Header:    "| Arm | pass^1 | safe resolve | policy breaches | minefield hits | blocked dangerous | unnecessary blocks | denied commands | evidence completeness |",
			Separator: "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
			Rows: []string{
				terminalArmSummaryMarkdownRow("raw", r.Summary.Raw),
				terminalArmSummaryMarkdownRow("fak", r.Summary.Fak),
			},
		},
		Tasks: benchmarkdown.Table{
			Header:    "| Task | Raw tests | Raw safe | fak tests | fak safe | fak denied | dangerous blocks | unnecessary blocks | normalized commands |",
			Separator: "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
			Rows:      commandRows,
		},
		PromotionRequirements: r.PromotionRequirements,
	})
}

func terminalArmSummaryMarkdownRow(name string, s ArmSummary) string {
	return fmt.Sprintf("| %s | %.3f | %.3f | %d | %d | %d | %d | %d | %.3f |",
		name, s.Pass1, s.SafeResolveRate, s.PolicyBreaches, s.MinefieldHits,
		s.DangerousBlocks, s.UnnecessaryBlocks, s.DeniedCommands, s.EvidenceCompleteness)
}

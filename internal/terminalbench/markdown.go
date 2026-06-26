package terminalbench

import (
	"fmt"
	"strings"
)

func RenderMarkdown(r *Report) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# Terminal-Bench Command Boundary Report\n\n")
	fmt.Fprintf(&b, "- Generated: `%s`\n", r.GeneratedAt)
	fmt.Fprintf(&b, "- Benchmark: `%s`\n", r.Benchmark)
	if r.Model != "" {
		fmt.Fprintf(&b, "- Model: `%s`\n", r.Model)
	}
	if r.EvidenceClass != "" {
		fmt.Fprintf(&b, "- Evidence class: `%s`\n", r.EvidenceClass)
	}
	fmt.Fprintf(&b, "- Tasks: `%d`\n", r.Summary.TaskCount)
	fmt.Fprintf(&b, "- Official harness: required=%t available=%t", r.OfficialHarness.Required, r.OfficialHarness.Available)
	if r.OfficialHarness.Reason != "" {
		fmt.Fprintf(&b, " (%s)", r.OfficialHarness.Reason)
	}
	fmt.Fprintf(&b, "\n")
	fmt.Fprintf(&b, "- Result claim allowed: `%t`\n", r.ResultClaimAllowed)
	fmt.Fprintf(&b, "- Boundary: %s\n\n", r.ClaimBoundary)

	fmt.Fprintf(&b, "| Arm | pass^1 | safe resolve | policy breaches | minefield hits | blocked dangerous | unnecessary blocks | denied commands | evidence completeness |\n")
	fmt.Fprintf(&b, "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
	writeArmSummary(&b, "raw", r.Summary.Raw)
	writeArmSummary(&b, "fak", r.Summary.Fak)
	fmt.Fprintf(&b, "\n")

	fmt.Fprintf(&b, "## Tasks\n\n")
	fmt.Fprintf(&b, "| Task | Raw tests | Raw safe | fak tests | fak safe | fak denied | dangerous blocks | unnecessary blocks | normalized commands |\n")
	fmt.Fprintf(&b, "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
	for _, t := range r.Tasks {
		fmt.Fprintf(&b, "| `%s` | %t | %t | %t | %t | %d | %d | %d | %d |\n",
			t.ID, t.Raw.TestSuccess, t.Raw.SafeResolve, t.Fak.TestSuccess, t.Fak.SafeResolve,
			t.Fak.DeniedCommands, len(t.Fak.DangerousBlocks), len(t.Fak.UnnecessaryBlocks), len(t.Fak.NormalizedCommands))
	}
	if len(r.PromotionRequirements) > 0 {
		fmt.Fprintf(&b, "\n## Promotion Requirements\n\n")
		for _, req := range r.PromotionRequirements {
			fmt.Fprintf(&b, "- %s\n", req)
		}
	}
	return b.String()
}

func writeArmSummary(b *strings.Builder, name string, s ArmSummary) {
	fmt.Fprintf(b, "| %s | %.3f | %.3f | %d | %d | %d | %d | %d | %.3f |\n",
		name, s.Pass1, s.SafeResolveRate, s.PolicyBreaches, s.MinefieldHits,
		s.DangerousBlocks, s.UnnecessaryBlocks, s.DeniedCommands, s.EvidenceCompleteness)
}

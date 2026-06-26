package toolsandbox

import (
	"fmt"
	"strings"
)

func RenderMarkdown(r *Report) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# ToolSandbox/tau3 Adapter Report\n\n")
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

	fmt.Fprintf(&b, "| Arm | pass^1 | safe pass^1 | benign utility | policy breaches | minefield hits | denied calls | argument repairs | evidence completeness |\n")
	fmt.Fprintf(&b, "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
	writeArmSummary(&b, "raw", r.Summary.Raw)
	writeArmSummary(&b, "fak", r.Summary.Fak)
	fmt.Fprintf(&b, "\n")

	fmt.Fprintf(&b, "## Tasks\n\n")
	fmt.Fprintf(&b, "| Task | Benign | Raw success | Raw safe | fak success | fak safe | fak denied | normalized calls |\n")
	fmt.Fprintf(&b, "|---|:---:|---:|---:|---:|---:|---:|---:|\n")
	for _, t := range r.TaskReports {
		fmt.Fprintf(&b, "| `%s` | %t | %t | %t | %t | %t | %d | %d |\n",
			t.ID, t.Benign, t.Raw.TaskSuccess, t.Raw.SafeSuccess, t.Fak.TaskSuccess, t.Fak.SafeSuccess, t.Fak.DeniedCalls, len(t.Fak.NormalizedToolCalls))
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
	fmt.Fprintf(b, "| %s | %.3f | %.3f | %.3f | %d | %d | %d | %d | %.3f |\n",
		name, s.Pass1, s.SafePass1, s.BenignUtilityPreservation, s.PolicyBreaches,
		s.MinefieldHits, s.DeniedCalls, s.ArgumentRepairs, s.EvidenceCompleteness)
}

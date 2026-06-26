package browseraction

import (
	"fmt"
	"strings"
)

func RenderActionMediationMarkdown(r *ActionMediationReport) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# Browser Action Mediation Report\n\n")
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

	fmt.Fprintf(&b, "| Arm | pass^1 | safe pass^1 | policy breaches | minefield hits | denied actions | invalid actions | evidence completeness |\n")
	fmt.Fprintf(&b, "|---|---:|---:|---:|---:|---:|---:|---:|\n")
	writeActionArmSummary(&b, "raw", r.Summary.Raw)
	writeActionArmSummary(&b, "fak", r.Summary.Fak)
	fmt.Fprintf(&b, "\n")

	fmt.Fprintf(&b, "## Failure Split\n\n")
	fmt.Fprintf(&b, "| Arm | model perception/grounding | harness/tool-boundary | boundary interventions |\n")
	fmt.Fprintf(&b, "|---|---:|---:|---:|\n")
	writeFailureSplit(&b, "raw", r.Summary.Raw)
	writeFailureSplit(&b, "fak", r.Summary.Fak)
	fmt.Fprintf(&b, "\n")

	fmt.Fprintf(&b, "## Tasks\n\n")
	fmt.Fprintf(&b, "| Task | Raw success | Raw safe | fak success | fak safe | fak denied | fak evidence | normalized calls |\n")
	fmt.Fprintf(&b, "|---|---:|---:|---:|---:|---:|---:|---:|\n")
	for _, t := range r.Tasks {
		fmt.Fprintf(&b, "| `%s` | %t | %t | %t | %t | %d | %.3f | %d |\n",
			t.ID, t.Raw.TaskSuccess, t.Raw.SafeSuccess, t.Fak.TaskSuccess, t.Fak.SafeSuccess,
			t.Fak.DeniedActions, t.Fak.EvidenceCompleteness, len(t.Fak.NormalizedToolCalls))
	}
	if len(r.PromotionRequirements) > 0 {
		fmt.Fprintf(&b, "\n## Promotion Requirements\n\n")
		for _, req := range r.PromotionRequirements {
			fmt.Fprintf(&b, "- %s\n", req)
		}
	}
	return b.String()
}

func writeActionArmSummary(b *strings.Builder, name string, s ActionArmSummary) {
	fmt.Fprintf(b, "| %s | %.3f | %.3f | %d | %d | %d | %d | %.3f |\n",
		name, s.Pass1, s.SafePass1, s.PolicyBreaches, s.MinefieldHits,
		s.DeniedActions, s.InvalidActions, s.EvidenceCompleteness)
}

func writeFailureSplit(b *strings.Builder, name string, s ActionArmSummary) {
	fmt.Fprintf(b, "| %s | %d | %d | %d |\n",
		name, s.ModelPerceptionFailures, s.HarnessToolBoundaryFailures, s.BoundaryInterventions)
}

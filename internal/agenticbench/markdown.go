package agenticbench

import (
	"fmt"
	"strings"
)

func RenderMarkdown(r *Report) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# Agentic Benchmark Epic #868 Rollup\n\n")
	fmt.Fprintf(&b, "- Generated: `%s`\n", r.GeneratedAt)
	fmt.Fprintf(&b, "- Status: `%s`\n", r.Status)
	fmt.Fprintf(&b, "- Result claim allowed: `%t`\n", r.ResultClaimAllowed)
	fmt.Fprintf(&b, "- Children parsed: `%d/%d`\n", r.Summary.ChildrenParsed, r.Summary.ChildrenTotal)
	fmt.Fprintf(&b, "- Result-claim artifacts: `%d`\n", r.Summary.ResultClaimArtifacts)
	fmt.Fprintf(&b, "- Boundary: %s\n\n", r.ClaimBoundary)

	fmt.Fprintf(&b, "## Children\n\n")
	fmt.Fprintf(&b, "| Issue | Packet | Artifact | Gate | Status | Detail |\n")
	fmt.Fprintf(&b, "|---:|---|---|---|---|---|\n")
	for _, child := range r.Children {
		fmt.Fprintf(&b, "| #%d | `%s` | `%s` | `%s` | `%s` | %s |\n",
			child.Issue, child.Packet, child.Artifact, child.Gate, child.Status, mdCell(child.Detail))
	}

	fmt.Fprintf(&b, "\n## Acceptance Gates\n\n")
	fmt.Fprintf(&b, "| Gate | OK | Detail |\n")
	fmt.Fprintf(&b, "|---|---:|---|\n")
	for _, gate := range r.Acceptance {
		fmt.Fprintf(&b, "| `%s` | %t | %s |\n", gate.Name, gate.OK, mdCell(gate.Detail))
	}
	return b.String()
}

func mdCell(s string) string {
	s = strings.ReplaceAll(s, "\r\n", " ")
	s = strings.ReplaceAll(s, "\n", " ")
	s = strings.ReplaceAll(s, "|", `\|`)
	return s
}

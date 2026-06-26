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
	fmt.Fprintf(&b, "- Result packets: `%d` passed / `%d` total\n", r.Summary.ResultPacketsPassed, r.Summary.ResultPacketsTotal)
	fmt.Fprintf(&b, "- Boundary: %s\n\n", r.ClaimBoundary)

	fmt.Fprintf(&b, "## Children\n\n")
	fmt.Fprintf(&b, "| Issue | Packet | Artifact | Gate | Status | Detail |\n")
	fmt.Fprintf(&b, "|---:|---|---|---|---|---|\n")
	for _, child := range r.Children {
		fmt.Fprintf(&b, "| #%d | `%s` | `%s` | `%s` | `%s` | %s |\n",
			child.Issue, child.Packet, child.Artifact, child.Gate, child.Status, mdCell(child.Detail))
	}

	fmt.Fprintf(&b, "\n## Result Packet Intake\n\n")
	fmt.Fprintf(&b, "- Directory: `%s/*.json`\n", DefaultResultPacketDir)
	fmt.Fprintf(&b, "- Schema: `%s`\n", ResultPacketSchema)
	fmt.Fprintf(&b, "- Required gates: `benchmark_native`, `same_task_ids`, `same_model`, `same_budget`, `official_grader.available`, raw/fak arms, checked-in artifacts, and metric categories `%s`.\n",
		strings.Join(requiredMetricCategories, "`, `"))

	if len(r.ResultPackets) > 0 {
		fmt.Fprintf(&b, "\n### Result Packets\n\n")
		fmt.Fprintf(&b, "| Path | Issue | Gate | Status | Detail |\n")
		fmt.Fprintf(&b, "|---|---:|---|---|---|\n")
		for _, packet := range r.ResultPackets {
			fmt.Fprintf(&b, "| `%s` | #%d | `%s` | `%s` | %s |\n",
				packet.Path, packet.Issue, packet.Gate, packet.Status, mdCell(packet.Detail))
		}
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
